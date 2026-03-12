"""
CLI Script to add new sequence folders to a raw dataset and update its registry.

Copies sequence subfolders from a source directory into data/raw/<type>/data/,
validates naming (strict: source_cam_azimuth_YYYY-MM-DDTHH-MM-SS, azimuth 0-360 or 999)
and structure (images/, labels/, ≥2 non-empty labels).
Rejected folders are reported and skipped. Existing entries are never modified.

Usage:
    python add_data.py --src ./path/to/new/sequences --type wildfire
    python add_data.py --src ./path/to/new/sequences --type fp
    python add_data.py --src ./path/to/new/sequences --type wildfire --dry-run

Arguments:
    --src <path>      Source directory containing sequence subfolders (required).
    --type <str>      Dataset type: "wildfire" or "fp" (required).
    --random-seed     Random seed (default: 0).
    --dry-run         Preview without copying or writing.
    -log, --loglevel  Logging level (default: info).
"""

import argparse
import logging
import shutil
from pathlib import Path

from pyro_dataset.ingest import (
    compute_new_assignments,
    load_registry,
    next_id,
    print_summary,
    save_registry,
    validate_source_folders,
)

DATASET_TYPES = {
    "wildfire": ("data/raw/wildfire", "wf"),
    "fp": ("data/raw/fp", "fp"),
}


def make_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add new sequence folders to a raw dataset and update its registry."
    )
    parser.add_argument(
        "--src",
        help="Source directory containing sequence subfolders to add.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--type",
        help='Dataset type: "wildfire" or "fp".',
        choices=list(DATASET_TYPES.keys()),
        required=True,
    )
    parser.add_argument(
        "--random-seed",
        help="Random seed for shuffling sequences within each camera group (default: 0).",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--dry-run",
        help="Preview without copying or writing.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-log",
        "--loglevel",
        default="info",
        help="Provide logging level. Example --loglevel debug, default=info",
    )
    return parser


def validate_parsed_args(args: dict) -> bool:
    if not args["src"].exists():
        logging.error(f"--src {args['src']} does not exist")
        return False
    if not args["src"].is_dir():
        logging.error(f"--src {args['src']} is not a directory")
        return False
    return True


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logging.basicConfig(level=args["loglevel"].upper())

    if not validate_parsed_args(args):
        exit(1)

    src = args["src"]
    dry_run = args["dry_run"]
    default_dir, prefix = DATASET_TYPES[args["type"]]
    dir_raw = Path(default_dir)
    dir_data = dir_raw / "data"
    registry_path = dir_raw / "registry.json"

    dir_data.mkdir(parents=True, exist_ok=True)

    existing = load_registry(registry_path)
    registered_folders = {s["folder"] for s in existing}
    already_on_disk = {d.name for d in dir_data.iterdir() if d.is_dir()}

    incoming = sorted(
        d.name for d in src.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    if not incoming:
        print("No sequence folders found in source.")
        exit(0)

    to_copy = [f for f in incoming if f not in already_on_disk]
    skipped_unregistered = [
        f for f in incoming if f in already_on_disk and f not in registered_folders
    ]
    already_registered = [f for f in incoming if f in registered_folders]

    if already_registered:
        logging.info(f"Skipping {len(already_registered)} already registered folder(s).")
    if skipped_unregistered:
        logging.warning(
            f"{len(skipped_unregistered)} folder(s) exist on disk but are not in "
            "registry — run add_data.py to register them."
        )
    if not to_copy:
        print("Nothing new to add.")
        exit(0)

    logging.info(f"Validating {len(to_copy)} folder(s) ...")
    summary = validate_source_folders(src, to_copy)

    if summary.rejected:
        structural = [r for r in summary.rejected if r.has_structural_issues]
        naming_only = [r for r in summary.rejected if not r.has_structural_issues]

        if structural:
            print(f"\n{'='*60}")
            print(f"STRUCTURAL ISSUES — {len(structural)} folder(s) skipped")
            print("These folders are missing required structure and cannot be ingested.\n")
            for r in structural:
                print(f"  {r.folder}")
                for issue in r.structural_issues:
                    print(f"    ✗ {issue}")
                for issue in r.naming_issues:
                    print(f"    ~ {issue}")
            print(f"\nFix the issues above and re-run to include them.")

        if naming_only:
            print(f"\n{'='*60}")
            print(f"NAMING ISSUES — {len(naming_only)} folder(s) skipped")
            print(
                "Expected format: <source>_<camera>_<azimuth 0-360|999>_<YYYY-MM-DDTHH-MM-SS>\n"
                "  azimuth 999 = unknown/unspecified\n"
            )
            for r in naming_only:
                print(f"  {r.folder}")
                for issue in r.naming_issues:
                    print(f"    ~ {issue}")

    to_copy = list(summary.valid)

    if not to_copy:
        print("Nothing to copy.")
        exit(0 if not summary.rejected else 1)

    logging.info(f"Copying {len(to_copy)} folder(s) to {dir_data} ...")
    if not dry_run:
        for folder in to_copy:
            shutil.copytree(src=src / folder, dst=dir_data / folder)
            logging.debug(f"  copied {folder}")

    start_id = next_id(existing, prefix)
    new_assignments = compute_new_assignments(
        to_copy, existing, start_id, prefix, random_seed=args["random_seed"]
    )
    all_sequences = existing + new_assignments

    print_summary(new_assignments, all_sequences)

    if dry_run:
        print("Dry run — nothing written.")
    else:
        save_registry(registry_path, all_sequences)
        logging.info(f"Registry updated: {registry_path}")

    if summary.rejected:
        skipped = [r.folder for r in summary.rejected]
        print(f"\nSkipped ({len(skipped)}):")
        for name in skipped:
            print(f"  {name}")
