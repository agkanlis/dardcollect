#!/usr/bin/env python3
"""Classify downloaded videos as colour vs black-and-white from their pixels.

This is a *content-based* check: it samples frames from each video and measures
mean chroma (see ``dardcollect.pipeline_utils.video_color_score``), so it works
regardless of whether archive.org tagged the item's ``color`` metadata field.

Outputs ``color_classification.csv`` next to the videos, one row per file:
``filename, color_score, is_color, metadata_color, path`` — where
``metadata_color`` is cross-referenced from ``downloads.csv`` when available so
you can compare the content verdict against archive.org's tag.

With ``--move``, black-and-white videos are moved to a sibling ``<input>_bw/``
folder (relative sub-paths preserved) so only colour videos remain in place for
the rest of the video pipeline.

CPU-only — no GPU or model required.

Examples:
    python pipeline/filter_videos_by_color.py
    python pipeline/filter_videos_by_color.py --input-dir DARD/archive_org_public_domain/videos_held
    python pipeline/filter_videos_by_color.py --threshold 12 --move

Config (optional, config → color_filtering):
    input_dir, threshold, sample_frames, move_non_color
CLI flags override config; if neither is set, input defaults to
``person_extraction.input_dir``.
"""

import argparse
import csv
import logging
import os
import shutil
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

from dardcollect.config import get_log_level
from dardcollect.pipeline_utils import _TqdmHandler, video_color_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
for h in logging.getLogger().handlers:
    logging.getLogger().removeHandler(h)
logging.getLogger().addHandler(_TqdmHandler())
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(
    os.environ.get(
        "DARDCOLLECT_CONFIG",
        Path(__file__).resolve().parent.parent / "configs" / "config.archive_all.yaml",
    )
)
VIDEO_GLOBS = ("*.mp4", "*.avi", "*.mkv", "*.mov")


def _load_metadata_color_map(videos_dir: Path) -> dict[str, str]:
    """Map filename stem → archive.org ``color`` tag from downloads.csv (best effort)."""
    downloads_csv = videos_dir.parent / "downloads.csv"
    mapping: dict[str, str] = {}
    if not downloads_csv.exists():
        return mapping
    with open(downloads_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("filename_downloaded")
            if name:
                mapping[Path(name).stem] = (row.get("color") or "").strip()
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify videos as colour vs black-and-white.")
    parser.add_argument("config_path", nargs="?", default=str(CONFIG_PATH))
    parser.add_argument("--input-dir", help="Folder of videos to classify (recursive).")
    parser.add_argument("--threshold", type=float, help="Min mean chroma to count as colour.")
    parser.add_argument("--sample-frames", type=int, help="Frames sampled per video.")
    parser.add_argument(
        "--move", action="store_true", help="Move black-and-white videos to <input>_bw/."
    )
    args = parser.parse_args()

    with open(args.config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logging.getLogger().setLevel(get_log_level(args.config_path))

    cfg = config.get("color_filtering", {}) or {}
    input_dir = Path(
        args.input_dir or cfg.get("input_dir") or config["person_extraction"]["input_dir"]
    )
    threshold = args.threshold if args.threshold is not None else float(cfg.get("threshold", 10.0))
    sample_frames = args.sample_frames or int(cfg.get("sample_frames", 30))
    move_non_color = args.move or bool(cfg.get("move_non_color", False))

    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    videos = sorted({p for g in VIDEO_GLOBS for p in input_dir.rglob(g)})
    if not videos:
        logger.info("No videos found in %s", input_dir)
        return

    logger.info(
        "Classifying %d video(s) in %s — threshold %.1f, %d frames/video%s",
        len(videos),
        input_dir,
        threshold,
        sample_frames,
        " (moving B&W aside)" if move_non_color else "",
    )

    metadata_color = _load_metadata_color_map(input_dir)
    bw_dir = input_dir.parent / f"{input_dir.name}_bw"
    out_csv = input_dir / "color_classification.csv"

    n_color = n_bw = n_unreadable = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "color_score", "classification", "metadata_color", "path"])
        for v in tqdm(videos, desc="Colour check", unit="video"):
            score = video_color_score(v, sample_frames=sample_frames)
            if score < 0:
                classification = "unreadable"  # no frame decoded (e.g. AV1 unsupported)
                n_unreadable += 1
            elif score >= threshold:
                classification = "color"
                n_color += 1
            else:
                classification = "bw"
                n_bw += 1
            writer.writerow(
                [v.name, f"{score:.2f}", classification, metadata_color.get(v.stem, ""), str(v)]
            )

            # Only move confirmed black-and-white aside; never move unreadable files.
            if move_non_color and classification == "bw":
                dest = bw_dir / v.relative_to(input_dir)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(v), str(dest))
                logger.info("B&W → moved %s (score=%.2f)", v.name, score)

    msg = "Done — %d colour, %d black-and-white, %d unreadable. Report: %s"
    args_ = [n_color, n_bw, n_unreadable, out_csv]
    if move_non_color:
        msg += "  (B&W moved to %s)"
        args_.append(bw_dir)
    logger.info(msg, *args_)
    if n_unreadable:
        logger.warning(
            "%d video(s) could not be decoded (often AV1 on a platform without AV1 "
            "support) — left in place and NOT classified as B&W.",
            n_unreadable,
        )


if __name__ == "__main__":
    main()
