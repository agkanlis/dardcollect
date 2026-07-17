"""Regression tests for ``scripts/make_test_config.py``.

The bug these guard against: the redirect rules were literal string replacements
(``"DARD/archive_org_public_domain/videos"`` → fixture path). When the shipped
config switched to ``{root}/…`` templating, every rule matched nothing, so the
generated ``config.test.yaml`` was a byte-for-byte copy of the production
config — pointing the "fixture" gate at the real dataset and the real output
tree, while the golden compare ran against an empty ``DARD_test/``. A no-op
substitution is invisible, so the fix must fail loudly when nothing matches.
"""

from __future__ import annotations

import re

import pytest

from dardcollect.config import _resolve_path_templates
from scripts.make_test_config import (
    FIXTURE_MEDIA,
    TEST_ROOT,
    TemplateMismatch,
    render_test_config,
)

# A minimal stand-in for configs/config.archive_all.yaml: one top-level root, a
# templated input base per modality, and templated derived outputs.
TEMPLATED_CONFIG = """\
media_types: ["video", "image", "audio", "text"]
root: "DARD"

person_extraction:
  input_dir: "{root}/archive_org_public_domain/videos"
  output_clips_dir: "{root}/extracted_person_clips"

image_extraction:
  input_dir: "{root}/archive_org_public_domain/images"

audio_transcription:
  audio_files_dir: "{root}/archive_org_public_domain/audio"

face_crop_extraction:
  output_dir: "{root}/video_face_crops"
"""


def _resolved(config_text: str) -> dict:
    """Parse + apply {root} templating exactly as the stage scripts do."""
    import yaml

    return _resolve_path_templates(yaml.safe_load(config_text))


def _all_strings(obj):
    """Yield every string value in a nested config dict/list (comments already gone)."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _all_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _all_strings(v)
    elif isinstance(obj, str):
        yield obj


def test_inputs_redirect_to_fixture_media():
    cfg = _resolved(render_test_config(TEMPLATED_CONFIG))
    assert cfg["person_extraction"]["input_dir"] == f"{FIXTURE_MEDIA}/videos"
    assert cfg["image_extraction"]["input_dir"] == f"{FIXTURE_MEDIA}/images"
    assert cfg["audio_transcription"]["audio_files_dir"] == f"{FIXTURE_MEDIA}/audio"


def test_outputs_redirect_to_test_root():
    cfg = _resolved(render_test_config(TEMPLATED_CONFIG))
    # Repointing `root` alone must carry every derived {root}/… output to DARD_test/.
    assert cfg["person_extraction"]["output_clips_dir"] == f"{TEST_ROOT}/extracted_person_clips"
    assert cfg["face_crop_extraction"]["output_dir"] == f"{TEST_ROOT}/video_face_crops"


def test_no_production_paths_survive():
    """The whole point: the real dataset dir must not appear in the test config."""
    out = render_test_config(TEMPLATED_CONFIG)
    assert "archive_org_public_domain" not in out
    assert 'root: "DARD"' not in out


def test_output_differs_from_source():
    """A no-op copy (the original bug) is itself a failure, even without a raise."""
    assert render_test_config(TEMPLATED_CONFIG) != TEMPLATED_CONFIG


def test_raises_when_no_input_base_matches():
    """The exact regression: templating drifts so no input path matches -> raise,
    never emit a production config wearing a test name."""
    drifted = TEMPLATED_CONFIG.replace("archive_org_public_domain", "raw_downloads")
    with pytest.raises(TemplateMismatch, match="archive_org_public_domain"):
        render_test_config(drifted)


def test_raises_when_root_key_missing():
    no_root = re.sub(r"^root:.*$", "", TEMPLATED_CONFIG, flags=re.MULTILINE)
    with pytest.raises(TemplateMismatch, match="root"):
        render_test_config(no_root)


def test_raises_on_duplicate_root_key():
    """Two top-level roots is ambiguous — fail rather than guess which to redirect."""
    two_roots = TEMPLATED_CONFIG.replace('root: "DARD"', 'root: "DARD"\nroot: "DARD2"')
    with pytest.raises(TemplateMismatch, match="found 2"):
        render_test_config(two_roots)


def test_real_shipped_config_redirects_cleanly():
    """Guard the actual configs/config.archive_all.yaml, not just the stand-in:
    this is what would have caught the {root} migration that broke the gate."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    shipped = (repo_root / "configs" / "config.archive_all.yaml").read_text(encoding="utf-8")
    cfg = _resolved(render_test_config(shipped))
    # Every configured input_dir / *_dir that fed off the dataset now points at fixtures,
    # and nothing writes into the real DARD/ root.
    assert cfg["person_extraction"]["input_dir"].startswith(FIXTURE_MEDIA)
    assert cfg["person_extraction"]["output_clips_dir"].startswith(TEST_ROOT + "/")
    # No *live* config value may still name the production dataset dir. Check parsed values,
    # not raw text — comments (e.g. base_output_dir's note, or an inline output_subdir comment)
    # legitimately mention the path and are dropped on parse.
    bad = [s for s in _all_strings(cfg) if "archive_org_public_domain" in s]
    assert not bad, f"a live config value still points at the production dataset dir: {bad}"
