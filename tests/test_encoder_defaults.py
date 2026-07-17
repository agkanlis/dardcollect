"""Pin the encoder defaults, so the configurable codec stays additive.

Making the encoder configurable must not change what anyone gets by default: a
config that never mentions a codec has to keep producing libx264/aac, and the
default has to reach moviepy rather than merely sit in a dataclass.

CPU-only: moviepy is stubbed, so no encoder is invoked and no file is written.
"""

from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

import dardcollect.pipeline_utils as pipeline_utils
from dardcollect.config import ClipExtractionConfig, FaceCropConfig

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
SHIPPED_VIDEO_CONFIGS = ["config.archive_all.yaml", "config.custom_videos.yaml"]


@pytest.mark.parametrize("config_name", SHIPPED_VIDEO_CONFIGS)
def test_shipped_configs_keep_software_encoder(config_name):
    """Shipped configs must not require a hardware encoder to run."""
    face = FaceCropConfig.from_yaml(str(CONFIGS / config_name))
    clips = ClipExtractionConfig.from_yaml(str(CONFIGS / config_name))
    assert (face.video_codec, face.audio_codec) == ("libx264", "aac")
    assert (clips.video_codec, clips.audio_codec) == ("libx264", "aac")


def test_dataclass_defaults_are_the_software_encoder():
    """A config omitting the codec keys entirely still gets libx264/aac."""
    for cls in (FaceCropConfig, ClipExtractionConfig):
        assert cls.__dataclass_fields__["video_codec"].default == "libx264"
        assert cls.__dataclass_fields__["audio_codec"].default == "aac"


class _StubClip:
    """Records the kwargs write_videofile() was called with."""

    last_kwargs: ClassVar[dict] = {}

    def __init__(self, *_args, **_kwargs):
        pass

    def write_videofile(self, _path, **kwargs):
        _StubClip.last_kwargs = kwargs


@pytest.fixture
def stub_moviepy(monkeypatch):
    monkeypatch.setattr(
        "moviepy.video.io.ImageSequenceClip.ImageSequenceClip", _StubClip, raising=False
    )
    _StubClip.last_kwargs = {}
    return _StubClip


def test_default_codec_reaches_moviepy(stub_moviepy, tmp_path):
    """The default encoder must arrive at moviepy, not just live in a dataclass."""
    frames = [np.zeros((8, 8, 3), dtype=np.uint8)] * 2
    pipeline_utils._write_video_with_moviepy(frames, tmp_path / "out.mp4", fps=25.0)

    assert stub_moviepy.last_kwargs.get("codec") == "libx264"
    assert stub_moviepy.last_kwargs.get("audio_codec") == "aac"


def test_explicit_codec_is_honoured(stub_moviepy, tmp_path):
    """A caller opting into a hardware encoder must actually get it."""
    frames = [np.zeros((8, 8, 3), dtype=np.uint8)] * 2
    pipeline_utils._write_video_with_moviepy(
        frames, tmp_path / "out.mp4", fps=25.0, video_codec="h264_nvenc"
    )

    assert stub_moviepy.last_kwargs.get("codec") == "h264_nvenc"
