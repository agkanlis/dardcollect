"""Unit tests for dardcollect.crop_stabilization (CPU-only: no models, no video I/O)."""

import numpy as np

from dardcollect.crop_stabilization import (
    _resolve_savgol_window,
    _smooth_frame_data_corners,
    smooth_corner_sequence,
)


def _square(cx: float, cy: float, scale: float, theta: float) -> np.ndarray:
    """Build one [TL,TR,BR,BL] similarity-square from (center, scale, roll)."""
    rx, ry = scale * np.cos(theta), scale * np.sin(theta)
    hdx, hdy = -ry / 2.0, rx / 2.0  # half_d = (-Ry/2, Rx/2)
    return np.array(
        [
            [cx - rx / 2 - hdx, cy - ry / 2 - hdy],
            [cx + rx / 2 - hdx, cy + ry / 2 - hdy],
            [cx + rx / 2 + hdx, cy + ry / 2 + hdy],
            [cx - rx / 2 + hdx, cy - ry / 2 + hdy],
        ],
        dtype=np.float64,
    )


def _seq(cx, cy, scale, theta) -> np.ndarray:
    return np.stack([_square(*p) for p in zip(cx, cy, scale, theta)])


def _decompose(corners: np.ndarray):
    c = corners.mean(axis=1)
    r = corners[:, 1, :] - corners[:, 0, :]
    return c[:, 0], c[:, 1], np.hypot(r[:, 0], r[:, 1]), np.arctan2(r[:, 1], r[:, 0])


def _rms_diff(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.diff(a))))) if a.size > 1 else 0.0


def test_window_resolver():
    # 0.4s @ ~30fps -> 11 (odd, > polyorder)
    assert _resolve_savgol_window(11, 2, 750) == 11
    # clamps to run length, stays odd
    assert _resolve_savgol_window(11, 2, 8) == 7
    # too short -> None (leave raw)
    assert _resolve_savgol_window(11, 2, 3) is None
    assert _resolve_savgol_window(11, 2, 1) is None


def test_constant_square_roundtrip():
    n = 50
    base = _square(100.0, 80.0, 120.0, 0.3)
    corners = np.repeat(base[None], n, axis=0)
    out = smooth_corner_sequence(corners, window=11, polyorder=2)
    # A perfectly steady crop must be returned unchanged.
    assert np.max(np.abs(out - corners)) < 1e-4, np.max(np.abs(out - corners))


def test_short_run_is_noop_copy():
    base = _square(10.0, 10.0, 50.0, 0.0)[None]  # n=1
    out = smooth_corner_sequence(base, window=11, polyorder=2)
    assert out.shape == (1, 4, 2)
    assert np.max(np.abs(out - base)) < 1e-4


def test_scale_dewobbled_mean_preserved_no_zoom_in():
    rng = np.random.default_rng(0)
    n = 300
    cx = np.full(n, 100.0)
    cy = np.full(n, 100.0)
    scale = 120.0 + rng.normal(0, 6.0, n)  # zoom in/out wobble, mean 120
    theta = np.zeros(n)
    out = smooth_corner_sequence(
        _seq(cx, cy, scale, theta), window=11, polyorder=2, scale_mode="smooth", rotation_mode="raw"
    )
    _, _, s_out, _ = _decompose(out)
    # zoom wobble reduced
    assert _rms_diff(s_out) < 0.6 * _rms_diff(scale)
    # mean zoom preserved -> NOT zoomed in (within 1%)
    assert abs(s_out.mean() - scale.mean()) / scale.mean() < 0.01
    assert s_out.mean() >= scale.mean() * 0.99  # no systematic shrink (zoom-in)


def test_rotation_left_raw_by_default():
    rng = np.random.default_rng(1)
    n = 200
    cx = np.full(n, 50.0)
    cy = np.full(n, 50.0)
    scale = 100.0 + rng.normal(0, 4.0, n)
    theta = rng.normal(0, 0.15, n)  # roll jitter
    out = smooth_corner_sequence(
        _seq(cx, cy, scale, theta),
        window=11,
        polyorder=2,
        scale_mode="smooth",
        rotation_mode="raw",
    )
    _, _, _, th_out = _decompose(out)
    # rotation untouched -> output roll equals raw roll
    assert np.max(np.abs(np.unwrap(th_out) - np.unwrap(theta))) < 1e-5


def test_constant_scale_locks_zoom_no_wobble_no_zoom_in():
    rng = np.random.default_rng(4)
    n = 200
    scale = 120.0 + rng.normal(0, 8.0, n)
    out = smooth_corner_sequence(
        _seq(np.full(n, 50.0), np.full(n, 50.0), scale, np.zeros(n)),
        window=11,
        polyorder=2,
        scale_mode="constant",
        rotation_mode="raw",
    )
    _, _, s_out, _ = _decompose(out)
    assert np.allclose(s_out, s_out[0])  # constant -> zero zoom wobble
    assert s_out[0] >= np.median(scale)  # p90 anchor -> wide, never zoomed in


def test_constant_rotation_locks_roll():
    rng = np.random.default_rng(5)
    n = 200
    theta = rng.normal(0, 0.2, n)  # roll wobble
    out = smooth_corner_sequence(
        _seq(np.full(n, 50.0), np.full(n, 50.0), np.full(n, 100.0), theta),
        window=11,
        polyorder=2,
        scale_mode="raw",
        rotation_mode="constant",
    )
    _, _, _, th_out = _decompose(out)
    assert np.allclose(th_out, th_out[0])  # locked roll -> no wobble
    assert abs(th_out[0] - np.median(theta)) < 1e-6  # locked to the median roll


def test_zoom_widens_scale():
    n = 50
    base = _seq(np.full(n, 50.0), np.full(n, 50.0), np.full(n, 100.0), np.zeros(n))
    out = smooth_corner_sequence(
        base,
        window=11,
        polyorder=2,
        scale_mode="constant",
        rotation_mode="constant",
        zoom=1.2,
    )
    _, _, s_out, _ = _decompose(out)
    assert np.allclose(s_out, 120.0)  # 100 (constant) * 1.2 zoom


def test_location_jitter_reduced():
    rng = np.random.default_rng(2)
    n = 300
    cx = 100.0 + np.cumsum(rng.normal(0, 0.5, n))  # drifting + jittery
    cy = 80.0 + rng.normal(0, 2.0, n)
    scale = np.full(n, 110.0)
    theta = np.zeros(n)
    out = smooth_corner_sequence(_seq(cx, cy, scale, theta), window=11, polyorder=2)
    cx_o, cy_o, _, _ = _decompose(out)
    assert _rms_diff(cx_o) < _rms_diff(cx)
    assert _rms_diff(cy_o) < _rms_diff(cy)


def test_output_is_similarity_square_each_frame():
    rng = np.random.default_rng(3)
    n = 120
    out = smooth_corner_sequence(
        _seq(
            50 + rng.normal(0, 1, n),
            50 + rng.normal(0, 1, n),
            100 + rng.normal(0, 5, n),
            rng.normal(0, 0.1, n),
        ),
        window=11,
        polyorder=2,
    )
    for q in out:
        sides = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
        # all four sides equal (square) within tolerance
        assert max(sides) - min(sides) < 1e-3 * max(sides)


# ── integration tests for the pipeline-facing _smooth_frame_data_corners ───────
def _fd(frames, scales, thetas, tid=1, cx=100.0, cy=100.0):
    """Build a frame_data dict {str(abs_frame): [det]} for one track."""
    fd = {}
    for f, s, th in zip(frames, scales, thetas):
        q = _square(cx, cy, s, th)
        fd[str(f)] = [
            {"track_id": tid, "face_crop_corners_ofiq": [[float(x), float(y)] for x, y in q]}
        ]
    return fd


def _scale_of(det) -> float:
    c = np.asarray(det["face_crop_corners_ofiq"], float)
    r = c[1] - c[0]
    return float(np.hypot(r[0], r[1]))


def test_fd_fps_zero_no_crash():
    n = 20
    rng = np.random.default_rng(7)
    fd = _fd(range(n), 120 + rng.normal(0, 5, n), np.zeros(n))
    updated = _smooth_frame_data_corners(fd, fps=0.0, window_seconds=0.4, polyorder=2)
    assert updated == n
    for dets in fd.values():
        assert np.asarray(dets[0]["face_crop_corners_ofiq"]).shape == (4, 2)


def test_fd_single_frame_locked_to_self():
    # A single-frame track is now stabilized too (locked to its own geometry):
    # with zoom=1.0 that is an exact no-op on the corners.
    fd = _fd([5], [100.0], [0.0])
    before = np.asarray(fd["5"][0]["face_crop_corners_ofiq"], float)
    updated = _smooth_frame_data_corners(fd, fps=30.0, window_seconds=0.4, polyorder=2, zoom=1.0)
    assert updated == 1
    after = np.asarray(fd["5"][0]["face_crop_corners_ofiq"], float)
    assert np.max(np.abs(after - before)) < 1e-2


def test_fd_isolated_frames_all_locked():
    # Every frame isolated (gaps > 1). Previously these were left completely RAW
    # (len<2 runs skipped) — the raw-wobble leak. Now every det is locked to the
    # track constants; identical inputs + zoom=1.0 -> corners unchanged.
    frames = [0, 3, 6, 9, 12]
    fd = _fd(frames, [100.0] * 5, [0.0] * 5)
    before = {f: np.asarray(fd[str(f)][0]["face_crop_corners_ofiq"], float) for f in frames}
    updated = _smooth_frame_data_corners(fd, fps=30.0, window_seconds=0.4, polyorder=2, zoom=1.0)
    assert updated == 5
    for f in frames:
        after = np.asarray(fd[str(f)][0]["face_crop_corners_ofiq"], float)
        assert np.max(np.abs(after - before[f])) < 1e-2


def test_fd_two_runs_share_one_locked_scale():
    # run A frames 0..4 @ scale 100 ; run B frames 8..12 @ scale 200 (gap at 4->8).
    # TRACK-level locking: both runs must lock to the SAME p90 scale — the crop
    # must NOT jump to a different zoom at the run boundary (the regression that
    # made gappy tracks wobble violently under per-run locking).
    frames = list(range(5)) + list(range(8, 13))
    scales = [100.0] * 5 + [200.0] * 5
    fd = _fd(frames, scales, [0.0] * 10)
    updated = _smooth_frame_data_corners(
        fd,
        fps=30.0,
        window_seconds=0.4,
        polyorder=2,
        scale_mode="constant",
        rotation_mode="constant",
        zoom=1.0,
    )
    assert updated == 10
    locked = [_scale_of(fd[str(f)][0]) for f in frames]
    assert max(locked) - min(locked) < 1e-2, locked  # single zoom across the whole track
    assert locked[0] >= 100.0  # p90 of the track sits at the wide end


def test_fd_gappy_track_one_rotation_and_gap_fill():
    # Two runs with very different roll (+20deg vs -5deg) and a cornerless det in
    # the gap. After stabilization: ONE locked rotation across the entire track
    # (median), and the cornerless det receives synthesized locked corners with an
    # interpolated center (the raw-fallback leak is closed).
    fr_a = [0, 1, 2, 3]
    fr_b = [7, 8, 9, 10]
    th_a, th_b = np.radians(20.0), np.radians(-5.0)
    fd = _fd(fr_a, [100.0] * 4, [th_a] * 4, cx=50.0, cy=50.0)
    fd.update(_fd(fr_b, [100.0] * 4, [th_b] * 4, cx=90.0, cy=50.0))
    fd["5"] = [{"track_id": 1}]  # detection with NO corners, mid-gap
    updated = _smooth_frame_data_corners(
        fd,
        fps=30.0,
        window_seconds=0.4,
        polyorder=2,
        scale_mode="constant",
        rotation_mode="constant",
        zoom=1.0,
    )
    assert updated == 9  # 8 with corners + 1 synthesized
    thetas = []
    for f in [*fr_a, 5, *fr_b]:
        c = np.asarray(fd[str(f)][0]["face_crop_corners_ofiq"], float)
        v = c[1] - c[0]
        thetas.append(np.degrees(np.arctan2(v[1], v[0])))
    assert max(thetas) - min(thetas) < 0.05, thetas  # zero roll jump anywhere
    # gap det: interpolated center sits between the two runs' centers
    gap_c = np.asarray(fd["5"][0]["face_crop_corners_ofiq"], float).mean(axis=0)
    assert 50.0 < gap_c[0] < 90.0


def test_fd_garbage_keys_and_values_skipped():
    fd = _fd(range(15), [110.0] * 15, [0.0] * 15)
    fd["meta"] = {"not": "a list"}  # non-int key + non-list value
    fd["999"] = "garbage"  # int key, non-list value
    updated = _smooth_frame_data_corners(fd, fps=30.0, window_seconds=0.4, polyorder=2)
    assert updated == 15  # only the 15 valid dets


# ── the shipped default must stay OFF (stabilization is opt-in) ───────────────
def test_shipped_configs_keep_stabilization_off():
    """Stabilization is additive: a config that does not ask for it must not get it.

    Pinned from the shipped configs rather than the dataclass alone, so flipping
    a YAML default fails here instead of silently changing everyone's crops.
    """
    from pathlib import Path

    from dardcollect.config import FaceCropConfig

    configs = Path(__file__).resolve().parent.parent / "configs"
    for name in ("config.archive_all.yaml", "config.custom_videos.yaml"):
        cfg = FaceCropConfig.from_yaml(str(configs / name))
        assert cfg.corner_smoothing_enabled is False, name
    assert FaceCropConfig.__dataclass_fields__["corner_smoothing_enabled"].default is False
