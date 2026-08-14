"""
CLI script deciding which alerts of a pyro-annotator export to import, and into
which split.

This is the half of the import that carries accumulated state, which is why it
stays a manual command rather than a DVC stage: it reads the recurring-object
ledger and writes it back, and its output depends on every import that came
before. It copies no images — `materialise_annotator_sequences.py` turns the
plan it writes into sequence folders, and that half *is* a stage.

Every smoke alert is imported; that count sets the false-positive quota, filled
one alert per recurring object walking a hard-negatives-first ranking. Splits
come from the ledger, so an artefact always lands in the split it first went to
— test included, at 80/10/10.

The plan is accumulated, not regenerated: it carries forward the folders earlier
runs decided on, so it is committed to git alongside the ledger.

See docs/specs/2026-08-13-annotator-export-sequential-import-design.md.

Usage:
    python scripts/plan_annotator_import.py
    python scripts/plan_annotator_import.py --dry-run

Arguments:
    --export-dir       Export directory holding manifest.jsonl and images/
                       (default: data/raw/pyro-annotator/export, DVC-tracked).
    --plan             Import plan to read and update
                       (default: data/raw/pyro-annotator/import_plan.json).
    --ledger           Recurring-object ledger (default: data/raw/pyro-annotator/recurring_objects.json).
    --max-per-object   Lifetime cap of sequences per recurring object (default: 1).
    --hard-negative-threshold  Score at or above which an object counts as fooling (default: 0.5).
    --match-iou        IoU for matching an alert to an existing recurring object (default: 0.3).
    --same-fire-window Hours within which same-view smoke alerts count as one fire (default: 12).
    --force-train      Recurring object id to force into train (repeatable).
    --random-seed      Seed for split assignment (default: 0).
    --dry-run          Print the plan without writing anything.
    -log, --loglevel   Logging level (default: info).
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

from pyro_dataset.annotator.convert import alert_kind, camera_key, folder_name
from pyro_dataset.annotator.recurring import Ledger, assign_new_split, main_bbox
from pyro_dataset.annotator.same_fire import group_new_smoke
from pyro_dataset.annotator.select import rank_objects, select_fp


def make_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decide which pyro-annotator alerts to import, and into which split."
    )
    parser.add_argument(
        "--export-dir", type=Path, default=Path("data/raw/pyro-annotator/export")
    )
    parser.add_argument(
        "--plan", type=Path, default=Path("data/raw/pyro-annotator/import_plan.json")
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("data/raw/pyro-annotator/recurring_objects.json"),
    )
    parser.add_argument("--max-per-object", type=int, default=1)
    parser.add_argument("--hard-negative-threshold", type=float, default=0.5)
    parser.add_argument("--match-iou", type=float, default=0.3)
    parser.add_argument(
        "--same-fire-window",
        type=float,
        default=12.0,
        help="Hours within which same-view smoke alerts count as one fire.",
    )
    parser.add_argument("--force-train", action="append", default=[])
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "-log",
        "--loglevel",
        default="info",
        choices=["debug", "info", "warning", "error"],
    )
    return parser


def load_manifest(export_dir: Path) -> list[dict[str, Any]]:
    with (export_dir / "manifest.jsonl").open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_plan(path: Path) -> dict[str, dict[str, Any]]:
    """Folders decided by earlier runs: name -> kind, split, recurring object."""
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def has_usable_image(export_dir: Path, alert: dict[str, Any]) -> bool:
    """Whether at least one of the alert's frames has an image on disk.

    Checked here rather than while copying so that materialisation is a total
    function of the plan. It also stops an unusable alert from consuming its
    recurring object's one slot: a folder with no image is rejected
    structurally by `add_data.py`, so planning it wastes the slot forever.
    """
    return any(
        frame.get("image_path") and (export_dir / frame["image_path"]).is_file()
        for obj in alert["objects"]
        for frame in obj["frames"]
    )


def apply_force_train(ledger: Ledger, ro_ids: list[str]) -> None:
    """Move named recurring objects to train, refusing any already used elsewhere.

    Forcing to train can only remove evaluation contamination, never create it —
    but only while the object has no sequences in another split.
    """
    for ro_id in ro_ids:
        entry = ledger.entries.get(ro_id)
        if entry is None:
            raise SystemExit(f"--force-train {ro_id}: unknown recurring object")
        staged = entry["ingested_folders"]
        if entry["split"] != "train" and staged:
            raise SystemExit(
                f"--force-train {ro_id}: refused, it already has "
                f"{len(staged)} sequence(s) in {entry['split']}: {', '.join(staged)}"
            )
        entry["split"] = "train"


def resolve_recurring_objects(
    fp_alerts: list[dict[str, Any]],
    ledger: Ledger,
    rng: random.Random,
    split_counts: dict[str, int],
    match_iou: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Attach every FP alert to a recurring object, minting on first sight.

    Returns the per-object ranking inputs and, for each object, its candidate
    alerts ordered best-first: highest score, then most frames.
    """
    object_meta: dict[str, dict[str, Any]] = {}
    candidates: dict[str, list[dict[str, Any]]] = {}

    for alert in fp_alerts:
        bbox = main_bbox(alert)
        if bbox is None:
            logging.warning(
                f"alert {alert['platform_alert_id']}: no usable box, not a candidate"
            )
            continue
        alert_key = f"{alert['source_api']}:{alert['platform_alert_id']}"
        camera, azimuth = camera_key(alert).rsplit("_", 1)
        ro_id = ledger.match(camera, int(azimuth), bbox, match_iou)
        if ro_id is None:
            split = assign_new_split(rng, split_counts)
            ro_id = ledger.mint(camera, int(azimuth), bbox, split, alert_key)
            split_counts[split] = split_counts.get(split, 0) + 1
        else:
            ledger.record_sighting(ro_id, bbox, alert_key)

        score = alert.get("temporal_model_score")
        meta = object_meta.setdefault(ro_id, {"score": None, "seen": 0})
        meta["seen"] = ledger.seen(ro_id)
        if score is not None and (meta["score"] is None or score > meta["score"]):
            meta["score"] = score
        candidates.setdefault(ro_id, []).append(
            {
                "alert": alert,
                "folder": folder_name(alert),
                "score": score if score is not None else -1.0,
                "frames": sum(len(obj["frames"]) for obj in alert["objects"]),
            }
        )

    for alerts in candidates.values():
        alerts.sort(
            key=lambda c: (-c["score"], -c["frames"], c["alert"]["platform_alert_id"])
        )
    return object_meta, candidates


def main() -> None:
    args = vars(make_cli_parser().parse_args())
    logging.basicConfig(level=args["loglevel"].upper())

    export_dir: Path = args["export_dir"]
    plan_path: Path = args["plan"]
    ledger_path: Path = args["ledger"]
    dry_run: bool = args["dry_run"]
    rng = random.Random(args["random_seed"])

    alerts = load_manifest(export_dir)
    usable = []
    for alert in alerts:
        if has_usable_image(export_dir, alert):
            usable.append(alert)
        else:
            logging.warning(
                f"alert {alert['platform_alert_id']}: no image on disk, not planned"
            )
    smoke = [a for a in usable if alert_kind(a) == "wildfire"]
    fp_alerts = [a for a in usable if alert_kind(a) == "fp"]

    ledger = Ledger.load(ledger_path)
    apply_force_train(ledger, args["force_train"])

    # The export is a full re-pull, so `smoke` is the cumulative total while
    # already-planned folders are kept. Discount the false positives previous
    # runs contributed, or the quota pays twice for the same smoke.
    already_ingested = sum(len(e["ingested_folders"]) for e in ledger.entries.values())
    quota = max(len(smoke) - already_ingested, 0)
    logging.info(
        f"{len(alerts)} alerts: {len(smoke)} smoke, {len(fp_alerts)} false positive; "
        f"FP quota {quota} ({already_ingested} already ingested)"
    )

    # Seed from the ledger *and* from smoke folders earlier runs planned: smoke
    # has no recurring object, so the ledger alone under-counts and the 90/10
    # balance drifts as imports accumulate. Only folders the ledger does not
    # already account for are added, or every planned false positive would be
    # counted twice — once as its object, once as its folder.
    split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for entry in ledger.entries.values():
        split_counts[entry["split"]] = split_counts.get(entry["split"], 0) + 1
    ledger_folders = {
        folder for e in ledger.entries.values() for folder in e["ingested_folders"]
    }
    plan = load_plan(plan_path)
    for name, entry in plan.items():
        if name not in ledger_folders and entry["split"] in split_counts:
            split_counts[entry["split"]] += 1

    # One fire, one split: same-view smoke alerts chaining within the window
    # share a draw, inheriting from already planned smoke when the chain
    # touches any (docs/specs/2026-08-14-annotator-test-growth-design.md §6).
    planned_smoke = {
        name: entry["split"]
        for name, entry in plan.items()
        if entry["kind"] == "wildfire"
    }
    new_smoke = [
        folder_name(alert) for alert in smoke if folder_name(alert) not in plan
    ]
    smoke_splits: dict[str, str] = {}
    for group, inherited in group_new_smoke(
        new_smoke, planned_smoke, args["same_fire_window"]
    ):
        split = inherited or assign_new_split(rng, split_counts)
        for name in group:
            smoke_splits[name] = split
            split_counts[split] = split_counts.get(split, 0) + 1

    object_meta, candidates = resolve_recurring_objects(
        fp_alerts, ledger, rng, split_counts, args["match_iou"]
    )
    threshold = args["hard_negative_threshold"]
    ranked = rank_objects(object_meta, threshold=threshold)
    fooling = sum(
        1
        for ro_id in ranked
        if (object_meta[ro_id]["score"] is not None)
        and object_meta[ro_id]["score"] >= threshold
    )
    logging.info(
        f"{len(ranked)} recurring objects, {fooling} in the hard-negative bucket "
        f"(score >= {threshold})"
    )

    picked = select_fp(
        ranked=ranked,
        per_object_alerts=candidates,
        quota=quota,
        max_per_object=args["max_per_object"],
        ingested={ro_id: ledger.ingested(ro_id) for ro_id in ledger.entries},
    )
    logging.info(f"selected {len(picked)}/{quota} false-positive sequences")

    chosen: list[tuple[dict[str, Any], str, str | None]] = [
        (alert, "wildfire", None) for alert in smoke
    ]
    chosen += [(entry["alert"], "fp", entry["recurring_object"]) for entry in picked]

    # Re-annotation can flip an alert between fp and wildfire after its folder
    # was planned. The plan keeps the kind the folder was registered under — the
    # registry is append-only, so moving it would leave two entries for one
    # sequence — but a sequence quietly meaning the opposite of what it did is
    # not something to swallow. Built from every alert, not just the usable
    # ones: a flip is worth reporting even when the images have since gone.
    export_kinds = {folder_name(alert): alert_kind(alert) for alert in alerts}
    flipped = {
        name
        for name, entry in plan.items()
        if export_kinds.get(name) not in (None, entry["kind"])
    }
    for name in sorted(flipped):
        logging.warning(
            f"{name}: changed kind in the export, {plan[name]['kind']} -> "
            f"{export_kinds[name]}; kept as {plan[name]['kind']}"
        )

    added = 0
    for alert, kind, ro_id in chosen:
        name = folder_name(alert)
        if name in flipped:
            # This pick is void. Recording it would claim a wildfire folder as
            # a recurring object's false positive, burning the object's
            # lifetime slot and shorting the quota, permanently and silently.
            continue
        if name not in plan:
            added += 1
            split = (
                ledger.entries[ro_id]["split"]
                if ro_id is not None
                else smoke_splits[name]
            )
            plan[name] = {"kind": kind, "split": split, "recurring_object": ro_id}
        if ro_id is not None:
            # Record even when the folder was already planned: an interrupted
            # run, or a ledger rebuilt beside a surviving plan, would otherwise
            # leave the object with nothing recorded and no way to ever record
            # it — the quota would under-count and the --force-train guard
            # would be bypassed for that object, permanently.
            ledger.record_ingested(ro_id, name)

    # The ledger is authoritative over splits, and the plan now outlives the run
    # that wrote it. Re-reading the ledger onto every entry is what makes
    # --force-train reach a folder an earlier run already planned, and what
    # stops a stale entry from putting one artefact in two splits.
    for entry in plan.values():
        ro_id = entry["recurring_object"]
        if ro_id in ledger.entries:
            entry["split"] = ledger.entries[ro_id]["split"]

    print(f"\n{'DRY RUN — ' if dry_run else ''}Annotator import plan")
    print(f"  wildfire : {len(smoke)}")
    print(f"  fp       : {len(picked)} of a {quota} quota, {fooling} objects fooling")
    print(
        f"  splits   : train {sum(1 for e in plan.values() if e['split'] == 'train')}, "
        f"val {sum(1 for e in plan.values() if e['split'] == 'val')}, "
        f"test {sum(1 for e in plan.values() if e['split'] == 'test')}"
    )
    print(f"  planned  : {added} new folder(s), {len(plan) - added} already planned")

    if dry_run:
        print("Dry run — nothing written.")
        return

    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    ledger.save(ledger_path)
    print(f"\n  plan     : {plan_path}")
    print(f"  ledger   : {ledger_path}")
    print("\nNext:")
    print("  dvc repro materialise_annotator_sequences")
    for kind in ("wildfire", "fp"):
        print(
            f"  uv run python scripts/add_data.py "
            f"--src data/interim/pyro-annotator/sequences/{kind} --type {kind} "
            f"--splits-from data/interim/pyro-annotator/sequences/splits.json"
        )
    print("  uv run python scripts/freeze_test_selection.py")
    print("  dvc repro")


if __name__ == "__main__":
    sys.exit(main())
