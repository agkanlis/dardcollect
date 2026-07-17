#!/usr/bin/env python3
"""Generate ``configs/config.test.yaml`` from ``configs/config.archive_all.yaml``.

Produces the fast fixture-gate config.

The test config is the production config with input/output paths redirected to
the committed fixture media (``tests/fixtures/media/``) and a throwaway output
tree (``DARD_test/``). Regenerate whenever ``configs/config.archive_all.yaml``
changes so the test config never goes stale — do NOT hand-edit
``configs/config.test.yaml``.

Usage::

    python scripts/make_test_config.py            # writes configs/config.test.yaml

Idempotent: overwrites the output. Run once per machine (the output is
gitignored — it is a derived artifact, not source).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Every downloaded modality lives under ``{root}/archive_org_public_domain/<subdir>``,
# so rewriting that one prefix redirects all four inputs at once.
INPUT_BASE = re.compile(r"\{root\}/archive_org_public_domain")
FIXTURE_MEDIA = "tests/fixtures/media"

# Every derived output is ``{root}/<subdir>``, so repointing ``root`` itself sends the
# whole output tree to DARD_test/ — including any stage added later, which a
# per-directory list would silently leave writing into the real DARD/.
ROOT_KEY = re.compile(r"^root:[ \t]*\S.*$", re.MULTILINE)
TEST_ROOT = "DARD_test"

# The download stage writes to ``base_output_dir`` (newer configs) rather than
# ``{root}/archive_org_public_domain``. The fixture gate skips download, so this is
# never written — but leaving the real dataset path here would let the "test" config
# still name the production tree. Redirect it to the fixture too. Optional: absent in
# older configs, so no raise if it's missing.
BASE_OUTPUT_DIR = re.compile(r'^(base_output_dir:[ \t]*)"[^"]*"', re.MULTILINE)


class TemplateMismatch(ValueError):
    """The source config lacks the templated paths this redirect depends on.

    Raised instead of writing a config, because a substitution that silently
    matches nothing produces a "test" config that is really the production one —
    reading the real dataset and writing the real output tree while the gate
    compares an empty ``DARD_test/``. That is the failure this script exists to
    prevent, so it must be loud.
    """


def render_test_config(src: str) -> str:
    """Redirect production config text to the fixture inputs + throwaway outputs.

    Rewrites the ``{root}/archive_org_public_domain`` input base to the fixture
    media dir and the top-level ``root`` to ``DARD_test`` (which carries every
    ``{root}/…`` output there at load time). Raises ``TemplateMismatch`` if
    either target is absent, so a config whose templating drifted fails loudly
    instead of yielding a production config wearing a test name.
    """
    out, n_inputs = INPUT_BASE.subn(FIXTURE_MEDIA, src)
    out, n_root = ROOT_KEY.subn(f'root: "{TEST_ROOT}"', out)
    out, _ = BASE_OUTPUT_DIR.subn(rf'\g<1>"{FIXTURE_MEDIA}"', out)
    if not n_inputs:
        raise TemplateMismatch("no '{root}/archive_org_public_domain' input paths found")
    if n_root != 1:
        raise TemplateMismatch(f"expected exactly one top-level 'root:' key, found {n_root}")
    return out


def main(argv: list[str] | None = None) -> int:
    src_path = REPO_ROOT / "configs" / "config.archive_all.yaml"
    out_path = REPO_ROOT / "configs" / "config.test.yaml"
    if not src_path.exists():
        print(f"error: {src_path} not found", file=sys.stderr)
        return 2
    try:
        out = render_test_config(src_path.read_text(encoding="utf-8"))
    except TemplateMismatch as exc:
        print(f"error: {exc} in {src_path.name}", file=sys.stderr)
        return 2
    out_path.write_text(out, encoding="utf-8")
    print(
        f"[make_test_config] wrote {out_path.relative_to(REPO_ROOT)} "
        f"(inputs -> {FIXTURE_MEDIA}/, outputs -> {TEST_ROOT}/)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
