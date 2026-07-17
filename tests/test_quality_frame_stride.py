"""Regression tests for frame sampling in the quality-scoring loops.

Both `dardcollect.quality.score_video` and
`pipeline.annotate_face_quality._generate_ofiq_attr_json` sample every
`frame_stride`-th frame, up to `max_frames`. These tests pin that contract, so a
regression that stops advancing the frame counter (and therefore scores only
frame 0) fails here instead of silently shipping single-frame aggregates.

CPU-only: the per-frame scoring call and the video reader are stubbed, so no
models, GPU, or real video files are required.
"""

import json
import logging
import uuid
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

import dardcollect.pipeline_utils as pipeline_utils
import dardcollect.quality as quality
from dardcollect.quality import QualityModels
from pipeline.annotate_face_quality import _generate_ofiq_attr_json


def _fake_frame_scores(*_args, **_kwargs) -> dict:
    """Minimal stand-in for `score_frame_all()` output."""
    return {
        "sharpness": 50.0,
        "compression_artifacts": 50.0,
        "expression_neutrality": 50.0,
        "no_head_coverings": 50.0,
        "face_occlusion_prevention": 50.0,
        "head_pose": {
            "yaw_deg": 0.0,
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
            "yaw_quality": 100.0,
            "pitch_quality": 100.0,
            "roll_quality": 100.0,
        },
    }


def _models() -> QualityModels:
    """Stub models object: `score_frame_all` is patched out, so it is never dereferenced."""
    return cast(QualityModels, SimpleNamespace(magface=SimpleNamespace(get_providers=lambda: [])))


def _make_crop(tmp_path, n_frames: int, monkeypatch):
    """Create a crop file plus the sidecar the scorers read for provenance."""
    frames = [np.zeros((616, 616, 3), dtype=np.uint8)] * n_frames
    monkeypatch.setattr(pipeline_utils, "_get_frames_from_crop", lambda _path: frames)

    crop_path = tmp_path / "crop.mp4"
    crop_path.write_bytes(b"")
    crop_path.with_suffix(".json").write_text(
        json.dumps({"uuid": str(uuid.uuid4()), "source_video": "source.mp4"}),
        encoding="utf-8",
    )
    return crop_path


def _expected_indices(n_frames: int, frame_stride: int, max_frames: int) -> list[int]:
    sampled = list(range(0, n_frames, frame_stride))
    return sampled[:max_frames] if max_frames > 0 else sampled


@pytest.fixture(autouse=True)
def _stub_scoring(monkeypatch):
    monkeypatch.setattr(quality, "score_frame_all", _fake_frame_scores)
    # Suppress the one-shot provider log so the stub models object is never touched.
    monkeypatch.setattr(quality, "_provider_logged", True)


# (n_frames, frame_stride, max_frames)
SAMPLING_CASES = [
    (100, 5, 30),  # shipped video defaults; crop shorter than the max_frames cap
    (200, 5, 30),  # long crop: max_frames caps the sample
    (10, 1, 0),  # stride 1, uncapped: every frame
    (7, 3, 0),  # stride does not divide the frame count evenly
    (100, 5, 1),  # max_frames=1
]


@pytest.mark.parametrize(("n_frames", "frame_stride", "max_frames"), SAMPLING_CASES)
def test_score_video_samples_every_stride_th_frame(
    tmp_path, monkeypatch, n_frames, frame_stride, max_frames
):
    crop_path = _make_crop(tmp_path, n_frames, monkeypatch)

    result = quality.score_video(
        crop_path=crop_path,
        models=_models(),
        frame_stride=frame_stride,
        max_frames=max_frames,
        overwrite=True,
    )

    expected = _expected_indices(n_frames, frame_stride, max_frames)
    assert result is not None
    assert [f["frame_index"] for f in result["frame_data"]] == expected
    assert result["frames_scored"] == len(expected)


@pytest.mark.parametrize(("n_frames", "frame_stride", "max_frames"), SAMPLING_CASES)
def test_generate_ofiq_attr_json_samples_every_stride_th_frame(
    tmp_path, monkeypatch, n_frames, frame_stride, max_frames
):
    crop_path = _make_crop(tmp_path, n_frames, monkeypatch)
    cfg = SimpleNamespace(frame_stride=frame_stride, max_frames=max_frames, overwrite=True)

    assert _generate_ofiq_attr_json(crop_path, _models(), cfg) is True

    written = json.loads(crop_path.with_suffix(".ofiq_attr.json").read_text(encoding="utf-8"))
    expected = _expected_indices(n_frames, frame_stride, max_frames)
    assert [f["frame_index"] for f in written["frame_data"]] == expected
    assert written["frames_scored"] == len(expected)


def _fail_every_other_frame(monkeypatch):
    """Make `score_frame_all` raise on every second call."""
    calls = {"n": 0}

    def flaky(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            raise RuntimeError("simulated scoring failure")
        return _fake_frame_scores()

    monkeypatch.setattr(quality, "score_frame_all", flaky)


def test_failed_frames_are_skipped_not_counted_as_scored(tmp_path, monkeypatch, caplog):
    """A frame whose scoring raises must be absent from frame_data, and reported."""
    crop_path = _make_crop(tmp_path, n_frames=20, monkeypatch=monkeypatch)
    _fail_every_other_frame(monkeypatch)

    with caplog.at_level(logging.WARNING, logger=quality.__name__):
        result = quality.score_video(
            crop_path=crop_path,
            models=_models(),
            frame_stride=5,
            max_frames=0,
            overwrite=True,
        )

    # Sampled frames are 0, 5, 10, 15; calls 2 and 4 (frames 5 and 15) raise.
    assert result is not None
    assert [f["frame_index"] for f in result["frame_data"]] == [0, 10]
    assert result["frames_scored"] == 2
    assert "2 of 4 sampled frames failed to score" in caplog.text


def test_no_warning_when_every_frame_scores(tmp_path, monkeypatch, caplog):
    crop_path = _make_crop(tmp_path, n_frames=20, monkeypatch=monkeypatch)

    with caplog.at_level(logging.WARNING, logger=quality.__name__):
        quality.score_video(
            crop_path=crop_path,
            models=_models(),
            frame_stride=5,
            max_frames=0,
            overwrite=True,
        )

    assert "failed to score" not in caplog.text
