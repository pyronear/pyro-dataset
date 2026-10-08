"""
CLI freezing the FP half of `sequential_test` into an append-only lockfile.

Manual, never a DVC stage, for the same reason as plan_annotator_import.py:
it reads and rewrites accumulated state, and a stage would regenerate that
state from deps, destroying the pinning it exists to provide. The build
(`build_sequential_dataset.py`) copies the lockfile verbatim for test and
performs no selection — so every release's test set is a byte-identical
superset of the previous one, and models stay comparable across releases.

Each run: the lockfile is loaded (or, once, bootstrapped from the built
dataset — never recomputed), then every registered test FP not yet frozen is
appended in registry order, unless it has fewer than MIN_SEQUENCE_IMAGES
images. Test takes all its negatives: the 1:1 balance only matters for
training, and FPR is prevalence-free, so more negatives just give
a tighter estimate. Entries are never removed or reordered.

See docs/specs/2026-08-14-annotator-test-growth-design.md.

Usage:
    python scripts/freeze_test_selection.py
    python scripts/freeze_test_selection.py --dry-run

Arguments:
    --fp-registry      FP registry (default: data/raw/fp/registry.json).
    --fp-data-dir      FP sequence folders (default: data/raw/fp/data).
    --lockfile         The lockfile (default: data/raw/sequential_test_lock.json).
    --bootstrap-from   Built sequential test dir to seed a missing lockfile from
                       (default: data/processed/sequential_test).
    --dry-run          Print the outcome without writing.
    -log, --loglevel   Logging level (default: info).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from pyro_dataset.constants import MIN_SEQUENCE_IMAGES
from pyro_dataset.fp.lockfile import (
    extend_with_pins,
    load_lockfile,
    save_lockfile,
    validate_lockfile,
)


def make_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze the FP half of sequential_test into the lockfile."
    )
    parser.add_argument(
        "--fp-registry", type=Path, default=Path("data/raw/fp/registry.json")
    )
    parser.add_argument("--fp-data-dir", type=Path, default=Path("data/raw/fp/data"))
    parser.add_argument(
        "--lockfile", type=Path, default=Path("data/raw/sequential_test_lock.json")
    )
    parser.add_argument(
        "--bootstrap-from",
        type=Path,
        default=Path("data/processed/sequential_test"),
        help="Built test dataset to seed a missing lockfile from, exactly.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "-log",
        "--loglevel",
        default="info",
        choices=["debug", "info", "warning", "error"],
    )
    return parser


def load_registry(path: Path) -> list[dict]:
    return json.loads(path.read_text())["sequences"]


def bootstrap_folders(built_test: Path) -> list[str]:
    """Seed a first lockfile from the built dataset, never by recomputing.

    Recomputing would trust the two-stage selection to reproduce what shipped
    bit-identically — the exact assumption the lockfile exists to retire. The
    built dataset is what current models were evaluated on, so it is the truth.
    """
    fp_dir = built_test / "test" / "fp"
    if not fp_dir.is_dir():
        raise SystemExit(
            f"no lockfile and no built test set at {fp_dir} — "
            "run `dvc pull` or `dvc repro build_sequential_dataset` first"
        )
    return sorted(d.name for d in fp_dir.iterdir() if d.is_dir())


def main() -> None:
    args = vars(make_cli_parser().parse_args())
    logging.basicConfig(level=args["loglevel"].upper())
    lockfile_path: Path = args["lockfile"]
    data_dir: Path = args["fp_data_dir"]
    dry_run: bool = args["dry_run"]

    fp_sequences = load_registry(args["fp_registry"])
    # Too short for the sequential build, which leaves them out of test too.
    test_fp = [
        s["folder"]
        for s in fp_sequences
        if s["split"] == "test"
        and len(list((data_dir / s["folder"] / "images").glob("*.jpg")))
        >= MIN_SEQUENCE_IMAGES
    ]
    quota = len(test_fp)

    bootstrapped = not lockfile_path.is_file()
    frozen = (
        bootstrap_folders(args["bootstrap_from"])
        if bootstrapped
        else load_lockfile(lockfile_path)
    )

    try:
        folders = extend_with_pins(frozen, test_fp, quota)
    except ValueError as error:
        raise SystemExit(str(error))

    problems = validate_lockfile(folders, set(test_fp), data_dir)
    if problems:
        for problem in problems:
            logging.error(problem)
        raise SystemExit(f"{len(problems)} problem(s) — lockfile not written")

    print(f"\n{'DRY RUN — ' if dry_run else ''}Test selection freeze")
    print(f"  test FPs : {quota}")
    print(f"  frozen   : {len(frozen)}{'  (bootstrapped)' if bootstrapped else ''}")
    print(f"  appended : +{len(folders) - len(frozen)}")
    if dry_run:
        print("Dry run — nothing written.")
        return
    save_lockfile(lockfile_path, folders)
    print(f"  lockfile : {lockfile_path}")
    print("\nCommit the lockfile together with the registries it froze against.")


if __name__ == "__main__":
    sys.exit(main())
