"""Opt-in stabilization of stored OFIQ face-crop corners.

Face crops shake and breathe even though the landmarks are smoothed upstream:
the crop square is re-fit per frame, so the fit's scale and roll wobble
frame-to-frame. That residual is an alignment artifact, not camera motion.

Remedy: decompose each stored corner-square into center / scale / roll and
treat the degrees of freedom separately. LOCATION is Savitzky-Golay smoothed
(a gentle pan that still follows the subject); SCALE and ROLL follow a
configurable mode, with the default recipe locking both to per-TRACK constants
(p90 of |R|, median roll) and widening by a zoom factor — a steady, upright,
slightly wider crop with no per-frame zoom or roll wobble. See the
``corner_smoothing_*`` keys on ``FaceCropConfig``; everything is off by
default (``corner_smoothing_enabled: false``).

Caveat: this stabilizes the rendered OFIQ image only. The sidecar's
``keypoints``/``bbox`` are produced from the raw keypoints, so when smoothing
is ON they describe the raw per-frame alignment, not the rendered pixels.
Harmless for quality scoring (OFIQ/MagFace read the image via the constant
ArcFace region, never the sidecar keypoints), but a debug overlay of those
keypoints on the crop would be slightly offset.
"""

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import savgol_filter

from dardcollect.face_geometry import _get_or_compute_corners

if TYPE_CHECKING:
    from dardcollect.config import FaceCropConfig


def _resolve_savgol_window(window: int, polyorder: int, n: int) -> int | None:
    """Largest valid odd Savitzky-Golay window ≤ *n*, or None if the run is too short.

    Mirrors the window logic of tracker.smooth_segment_keypoints: force odd, floor
    at polyorder+2, clamp to the run length. Returns None when the run cannot be
    meaningfully smoothed (caller should then leave it raw).
    """
    win = max(int(window) | 1, polyorder + 2)  # |1 forces odd; floor above polyorder
    if win % 2 == 0:
        win += 1
    win = min(win, n)
    if win % 2 == 0:
        win -= 1
    if win < polyorder + 2 or win > n:
        return None
    return win


def _recompose_square(cx: float, cy: float, scale: float, theta: float) -> np.ndarray:
    """Build a [TL,TR,BR,BL] similarity-square from center/scale/roll."""
    rx, ry = scale * np.cos(theta), scale * np.sin(theta)
    hdx, hdy = -ry / 2.0, rx / 2.0
    return np.array(
        [
            [cx - rx / 2 - hdx, cy - ry / 2 - hdy],
            [cx + rx / 2 - hdx, cy + ry / 2 - hdy],
            [cx + rx / 2 + hdx, cy + ry / 2 + hdy],
            [cx - rx / 2 + hdx, cy - ry / 2 + hdy],
        ],
        dtype=np.float64,
    )


def _decompose_run(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split (N,4,2) corner-squares into per-frame (cx, cy, scale, theta) tracks.

    The top-edge vector R = TR - TL carries scale (|R|) and roll (angle); theta is
    unwrapped so smoothing/median never straddle the ±π seam.
    """
    c = corners.mean(axis=1)  # (N, 2) center
    r = corners[:, 1, :] - corners[:, 0, :]  # (N, 2) top edge TR - TL
    theta = np.unwrap(np.arctan2(r[:, 1], r[:, 0]))
    return c[:, 0], c[:, 1], np.hypot(r[:, 0], r[:, 1]), theta


def _recompose_run(
    cx: np.ndarray, cy: np.ndarray, scale: np.ndarray, theta: np.ndarray
) -> np.ndarray:
    """Rebuild (N,4,2) float32 corner-squares from per-frame scalar tracks."""
    n = cx.shape[0]
    rx = scale * np.cos(theta)
    ry = scale * np.sin(theta)
    half_dx, half_dy = -ry / 2.0, rx / 2.0  # half_d = (-Ry/2, Rx/2)
    out = np.empty((n, 4, 2), dtype=np.float32)
    out[:, 0, 0], out[:, 0, 1] = cx - rx / 2.0 - half_dx, cy - ry / 2.0 - half_dy  # TL
    out[:, 1, 0], out[:, 1, 1] = cx + rx / 2.0 - half_dx, cy + ry / 2.0 - half_dy  # TR
    out[:, 2, 0], out[:, 2, 1] = cx + rx / 2.0 + half_dx, cy + ry / 2.0 + half_dy  # BR
    out[:, 3, 0], out[:, 3, 1] = cx - rx / 2.0 + half_dx, cy - ry / 2.0 + half_dy  # BL
    return out


def smooth_corner_sequence(
    corners: np.ndarray,
    window: int,
    polyorder: int = 2,
    scale_mode: str = "constant",
    rotation_mode: str = "constant",
    zoom: float = 1.0,
    constant_scale_pct: float = 90.0,
    const_scale: float | None = None,
    const_theta: float | None = None,
) -> np.ndarray:
    """Stabilize a CONTIGUOUS run of OFIQ face-crop corner-squares.

    Each frame's corners are a similarity-square [TL, TR, BR, BL] in source-frame
    pixels (rotation + uniform scale + translation, no shear). Every frame is
    decomposed into center / scale / roll, each degree of freedom is transformed
    across the run, and the squares are recomposed.

    LOCATION (Cx, Cy) is always Savitzky-Golay-smoothed (a gentle pan). SCALE and
    ROTATION each follow a mode:
      "raw"      — keep the per-frame value (round-trips exactly).
      "smooth"   — Savitzky-Golay-smooth the scalar across the run.
      "constant" — lock across the run: scale -> the ``constant_scale_pct`` percentile
                   of |R| (a steady, wide framing); rotation -> the median roll.
    ``zoom`` then multiplies the final scale (>1 widens / zooms out).

    Args:
        corners: (N, 4, 2) array for ONE contiguous run, ordered [TL, TR, BR, BL].
        window: target Savitzky-Golay window in *frames* (clamped to the run length).
        polyorder: Savitzky-Golay polynomial order.
        scale_mode / rotation_mode: one of "raw" | "smooth" | "constant".
        zoom: final scale multiplier (>1 = wider / zoom out).
        constant_scale_pct: percentile of |R| used when scale_mode == "constant".
        const_scale / const_theta: TRACK-LEVEL constants to lock to (pre-zoom scale,
            roll in radians). CRITICAL for tracks split into multiple contiguous
            runs: without these, each run locks to its OWN percentile/median and
            the crop jumps discontinuously at every run boundary (measured up to
            ~45-50 deg of roll jump on gappy tracks). Callers processing a
            multi-run track must compute these once per track and pass them to
            every run's call. When None, falls back to per-run values (only safe
            for single-run tracks).

    Returns:
        (N, 4, 2) float32 stabilized corners. Location is left raw when the run is
        too short for the savgol window; constant/raw scale & rotation still apply.
    """
    corners = np.asarray(corners, dtype=np.float64)
    cx, cy, scale, theta = _decompose_run(corners)

    win = _resolve_savgol_window(window, polyorder, corners.shape[0])
    if win is not None:  # LOCATION: always smoothed (gentle pan)
        cx = savgol_filter(cx, win, polyorder)
        cy = savgol_filter(cy, win, polyorder)

    if scale_mode == "smooth" and win is not None:
        scale = savgol_filter(scale, win, polyorder)
    elif scale_mode == "constant":
        # Prefer the track-level constant; per-run percentile only as fallback.
        lock_s = (
            const_scale
            if const_scale is not None
            else float(np.percentile(scale, constant_scale_pct))
        )
        scale = np.full_like(scale, lock_s)

    if rotation_mode == "smooth" and win is not None:
        theta = savgol_filter(theta, win, polyorder)
    elif rotation_mode == "constant":
        lock_t = const_theta if const_theta is not None else float(np.median(theta))
        theta = np.full_like(theta, lock_t)

    return _recompose_run(cx, cy, scale * float(zoom), theta)


def _collect_detections_by_track(frame_data: dict) -> dict[int, list]:
    """Group detections by track_id as (abs_frame, det, corners-or-None) triples.

    Detections WITHOUT stored corners are collected too — they would otherwise
    fall through to the raw per-frame re-fit at warp time; the caller synthesizes
    locked corners for them instead.
    """
    by_track: dict[int, list[tuple[int, dict, np.ndarray | None]]] = defaultdict(list)
    for key, dets in frame_data.items():
        try:
            abs_frame = int(key)
        except (TypeError, ValueError):
            continue
        if not isinstance(dets, list):
            continue
        for det in dets:
            tid = det.get("track_id")
            if tid is None:
                continue
            arr = None
            stored = det.get("face_crop_corners_ofiq")
            if stored is not None:
                a = np.asarray(stored, dtype=np.float64)
                if a.shape == (4, 2):
                    arr = a
            by_track[tid].append((abs_frame, det, arr))
    return by_track


def _lock_track_geometry(anchors: list) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Compute the TRACK-LEVEL locked geometry shared by every run of the track.

    p90 |R| (steady wide framing) and median roll over the WHOLE track — never
    per run, so a detection gap cannot re-lock the crop to a new angle/zoom.
    Also returns the anchor frame numbers and centers for gap interpolation.
    """
    anchor_c = np.stack([a for _, a in anchors])  # (M, 4, 2)
    centers_k = anchor_c.mean(axis=1)
    rvec = anchor_c[:, 1, :] - anchor_c[:, 0, :]
    scales_k = np.hypot(rvec[:, 0], rvec[:, 1])
    thetas_k = np.unwrap(np.arctan2(rvec[:, 1], rvec[:, 0]))
    const_scale = float(np.percentile(scales_k, 90.0))
    const_theta = float(np.median(thetas_k))
    anchor_frames = np.array([f for f, _ in anchors], dtype=np.float64)
    return const_scale, const_theta, anchor_frames, centers_k


def _fill_missing_corners(
    items: list,
    anchor_frames: np.ndarray,
    centers_k: np.ndarray,
    const_scale: float,
    const_theta: float,
    face_config: "FaceCropConfig | None",
) -> None:
    """Synthesize locked corners for detections that lack stored ones (in place).

    Center comes from the fallback keypoint fit when possible (a real
    measurement), else neighbor interpolation; scale/roll are the track
    constants either way (the fallback's scale/roll are its noisiest parts).
    """
    for j, (f, det, a) in enumerate(items):
        if a is not None:
            continue
        center = None
        if face_config is not None:
            fb = _get_or_compute_corners(det, face_config)
            if fb is not None:
                fc = np.asarray(fb, dtype=np.float64).mean(axis=0)
                center = (float(fc[0]), float(fc[1]))
        if center is None:
            center = (
                float(np.interp(f, anchor_frames, centers_k[:, 0])),
                float(np.interp(f, anchor_frames, centers_k[:, 1])),
            )
        items[j] = (f, det, _recompose_square(center[0], center[1], const_scale, const_theta))


def _smooth_track_runs(
    items: list,
    target_win: int,
    polyorder: int,
    scale_mode: str,
    rotation_mode: str,
    zoom: float,
    const_scale: float,
    const_theta: float,
) -> int:
    """Smooth location per contiguous run and write back rounded corners.

    Savitzky-Golay needs uniform sampling, hence per-run; the scale/roll locks
    are the track-level constants, shared by every run. Every item has corners
    by now (``_fill_missing_corners`` discharged the Nones).
    """
    n_updated = 0
    frames = [f for f, _, _ in items]
    run_start = 0
    for i in range(1, len(frames) + 1):
        if i == len(frames) or frames[i] - frames[i - 1] != 1:
            run = items[run_start:i]
            run_start = i
            stack = np.stack([np.asarray(a, dtype=np.float64) for _, _, a in run])
            smoothed = smooth_corner_sequence(
                stack,
                target_win,
                polyorder,
                scale_mode=scale_mode,
                rotation_mode=rotation_mode,
                zoom=zoom,
                const_scale=const_scale,
                const_theta=const_theta,
            )
            for (_, det, _a), corner in zip(run, smoothed):
                det["face_crop_corners_ofiq"] = [
                    [round(float(x), 2), round(float(y), 2)] for x, y in corner
                ]
                n_updated += 1
    return n_updated


def _smooth_frame_data_corners(
    frame_data: dict,
    fps: float,
    window_seconds: float,
    polyorder: int,
    scale_mode: str = "constant",
    rotation_mode: str = "constant",
    zoom: float = 1.15,
    face_config: "FaceCropConfig | None" = None,
) -> int:
    """Stabilize the stored OFIQ crop corners in *frame_data*, in place.

    Scale and rotation are locked PER TRACK (p90 |R| and median roll over the
    whole track), not per contiguous run — per-run locking made the crop jump to
    a different angle/zoom at every detection gap (measured up to ~50 deg of roll
    jump on fragmented tracks). Location is Savitzky-Golay smoothed per contiguous
    run (savgol needs uniform sampling; the track-level constants do not).

    Detections that LACK stored corners are also given stabilized corners, so the
    per-frame raw re-fit fallback in ``_get_or_compute_corners`` can never inject
    unsmoothed geometry into the render. Downstream warping then uses the smoothed
    corners automatically. See the module docstring for the sidecar-keypoints
    caveat.

    Returns the number of detections whose corners were updated.
    """
    target_win = max(int(window_seconds * fps) | 1, polyorder + 2) if fps > 0 else polyorder + 2

    n_updated = 0
    for items in _collect_detections_by_track(frame_data).values():
        items.sort(key=lambda t: t[0])
        anchors = [(f, a) for f, _, a in items if a is not None]
        if not anchors:
            continue  # no stored geometry anywhere on this track — nothing to anchor on
        const_scale, const_theta, anchor_frames, centers_k = _lock_track_geometry(anchors)
        _fill_missing_corners(
            items, anchor_frames, centers_k, const_scale, const_theta, face_config
        )
        n_updated += _smooth_track_runs(
            items, target_win, polyorder, scale_mode, rotation_mode, zoom, const_scale, const_theta
        )
    return n_updated
