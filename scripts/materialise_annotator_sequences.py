"""
CLI script turning an import plan into ingest-ready sequence folders.

A pure function of the export and the plan: no ledger, no randomness, no
decisions. That is what lets it be a DVC stage — `plan_annotator_import.py`
holds the accumulated state and makes every choice, this script only copies.

Writes one folder per planned alert, in the layout `add_data.py` expects, plus
the flat `splits.json` it reads with `--splits-from`.

See docs/specs/2026-08-13-annotator-export-sequential-import-design.md.

Usage:
    python scripts/materialise_annotator_sequences.py

Arguments:
    --export-dir   Export directory holding manifest.jsonl and images/
                   (default: data/raw/pyro-annotator/export, DVC-tracked).
    --plan         Import plan written by plan_annotator_import.py
                   (default: data/raw/pyro-annotator/import_plan.json).
    --output-dir   Where to write sequence folders
                   (default: data/interim/pyro-annotator/sequences).
    -log, --loglevel   Logging level (default: info).
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

from pyro_dataset.annotator.convert import (
    build_meta,
    folder_name,
    frame_stem,
    label_lines,
)


def make_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Turn an annotator import plan into ingest-ready sequence folders."
    )
    parser.add_argument(
        "--export-dir", type=Path, default=Path("data/raw/pyro-annotator/export")
    )
    parser.add_argument(
        "--plan", type=Path, default=Path("data/raw/pyro-annotator/import_plan.json")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/interim/pyro-annotator/sequences")
    )
    parser.add_argument(
        "-log",
        "--loglevel",
        default="info",
        choices=["debug", "info", "warning", "error"],
    )
    return parser


def index_alerts(export_dir: Path) -> dict[str, dict[str, Any]]:
    """Manifest alerts keyed by the folder name they materialise into.

    The key is second-resolution, so two alerts on one camera view inside the
    same second collapse into one folder. Nothing here can un-collapse them —
    the plan is keyed the same way — but the loser must not disappear silently.
    """
    with (export_dir / "manifest.jsonl").open() as fh:
        alerts = [json.loads(line) for line in fh if line.strip()]
    index: dict[str, dict[str, Any]] = {}
    for alert in alerts:
        name = folder_name(alert)
        if name in index:
            logging.warning(
                f"{name}: alerts {index[name]['platform_alert_id']} and "
                f"{alert['platform_alert_id']} share a folder name, keeping the latter"
            )
        index[name] = alert
    return index


def write_sequence(
    export_dir: Path, alert: dict[str, Any], dest: Path, kind: str, ro_id: str | None
) -> bool:
    """Materialise one alert as a sequence folder: images, labels, meta.json.

    `kind` is the one the plan pinned, and it decides the labels and the
    sidecar as well as the directory: re-annotation can flip an alert whose
    folder is already registered, and a folder filed as a false positive
    holding class-0 labels would train the temporal model on inverted evidence.

    Captures are deduplicated by timestamp — sibling lanes hold their own copies
    of the same capture, and the folder holds one image per capture.

    Returns whether any frame was written. An alert whose images have gone
    missing since it was planned leaves no folder behind: an empty one would be
    rejected structurally by `add_data.py`.
    """
    (dest / "images").mkdir(parents=True, exist_ok=True)
    (dest / "labels").mkdir(parents=True, exist_ok=True)

    written: set[str] = set()
    for obj in alert["objects"]:
        for frame in obj["frames"]:
            stem = frame_stem(alert, frame)
            if stem in written:
                continue
            if not frame.get("image_path"):
                logging.warning(
                    f"alert {alert['platform_alert_id']}: frame "
                    f"{frame['detection_id']} has no image, skipped"
                )
                continue
            source = export_dir / frame["image_path"]
            if not source.is_file():
                logging.warning(f"missing image on disk, skipped: {source}")
                continue
            written.add(stem)
            shutil.copy2(source, dest / "images" / f"{stem}.jpg")
            lines = label_lines(alert, frame, kind)
            (dest / "labels" / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else "")
            )

    if not written:
        logging.warning(
            f"alert {alert['platform_alert_id']}: no usable image, folder dropped"
        )
        shutil.rmtree(dest, ignore_errors=True)
        return False

    (dest / "meta.json").write_text(
        json.dumps(build_meta(alert, ro_id, kind), indent=2) + "\n"
    )
    return True


def main() -> None:
    args = vars(make_cli_parser().parse_args())
    logging.basicConfig(level=args["loglevel"].upper())

    export_dir: Path = args["export_dir"]
    output_dir: Path = args["output_dir"]

    plan: dict[str, dict[str, Any]] = json.loads(Path(args["plan"]).read_text())
    alerts = index_alerts(export_dir)

    splits: dict[str, str] = {}
    written = 0
    missing = 0
    for name in sorted(plan):
        entry = plan[name]
        alert = alerts.get(name)
        if alert is None:
            # Staging is transient and the export is a full re-pull, so a
            # folder already consumed into data/raw can drop out of a later
            # export. Failing here would wedge the stage for good.
            logging.warning(f"{name}: planned but absent from the export, skipped")
            missing += 1
            continue
        dest = output_dir / entry["kind"] / name
        if dest.exists():
            splits[name] = entry["split"]
            continue
        if not write_sequence(
            export_dir, alert, dest, entry["kind"], entry["recurring_object"]
        ):
            missing += 1
            continue
        splits[name] = entry["split"]
        written += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "splits.json").write_text(
        json.dumps(splits, indent=2, sort_keys=True) + "\n"
    )

    print("\nAnnotator materialisation")
    print(f"  planned  : {len(plan)}")
    print(
        f"  written  : {written} new folder(s), {len(splits) - written} already there"
    )
    if missing:
        print(f"  skipped  : {missing} planned folder(s) not in the export")
    print(
        f"  splits   : train {sum(1 for s in splits.values() if s == 'train')}, "
        f"val {sum(1 for s in splits.values() if s == 'val')}"
    )
    print(f"\n  staging  : {output_dir}")


if __name__ == "__main__":
    sys.exit(main())
