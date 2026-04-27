"""Apply the azimuth remap (azimuth_remap.csv) to data/raw/fp.

For each (site, discard_azimuth) → keep_azimuth in the remap CSV:
  - Rename every sequence folder `<site>_<discard_az>_<ts>` to
    `<site>_<keep_az>_<ts>`.
  - Rename every .jpg in images/ and every .txt in labels/ accordingly.
  - Update the matching entries in registry.json: `camera` and `folder` fields.

Idempotent: if a folder has already been renamed (destination exists, source
missing) it is skipped. Aborts before any change if a destination collision
is detected (different sequences would land on the same target name).

Usage (dry-run, default — nothing is written):
  uv run python scripts/apply_azimuth_remap.py

Apply for real:
  uv run python scripts/apply_azimuth_remap.py --apply

Then refresh the DVC tracking:
  dvc add data/raw/fp
"""

import argparse
import csv
import json
import logging
import re
from collections import defaultdict
from pathlib import Path


CAM_RE = re.compile(r"^(.+)_(\d+)$")


def make_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=Path("data/raw/fp"))
    p.add_argument(
        "--remap",
        type=Path,
        default=Path("data/interim/camera_kp_matches/all/azimuth_remap.csv"),
    )
    p.add_argument("--apply", action="store_true", help="Actually perform the rename + registry update.")
    p.add_argument("-log", "--loglevel", default="info")
    return p


def split_camera(cam: str) -> tuple[str, int | None]:
    m = CAM_RE.match(cam)
    if not m:
        return cam, None
    return m.group(1), int(m.group(2))


def main() -> None:
    args = make_cli_parser().parse_args()
    logging.basicConfig(level=args.loglevel.upper(), format="%(message)s")

    src: Path = args.source
    data_dir = src / "data"
    registry_path = src / "registry.json"
    if not data_dir.is_dir() or not registry_path.exists():
        raise SystemExit(f"missing data/ or registry.json in {src}")

    # Build remap dict: (site, discard) -> keep
    remap: dict[tuple[str, int], int] = {}
    with args.remap.open() as f:
        for r in csv.DictReader(f):
            remap[(r["site"], int(r["discard_azimuth"]))] = int(r["keep_azimuth"])
    logging.info(f"loaded {len(remap)} (site, azimuth) remappings from {args.remap}")

    registry = json.loads(registry_path.read_text())
    sequences = registry["sequences"]

    # Plan renames + registry updates. Detect destination collisions early.
    plan: list[dict] = []
    dest_targets: dict[Path, str] = {}
    for seq in sequences:
        site, az = split_camera(seq["camera"])
        if az is None or (site, az) not in remap:
            continue
        keep_az = remap[(site, az)]
        old_camera = seq["camera"]
        new_camera = f"{site}_{keep_az}"
        old_folder = seq["folder"]
        # Replace the first occurrence of "_<az>_" with "_<keep_az>_" to be safe
        prefix = f"{site}_{az}_"
        if not old_folder.startswith(prefix):
            logging.warning(f"unexpected folder prefix, skipping: {old_folder}")
            continue
        new_folder = f"{site}_{keep_az}_" + old_folder[len(prefix):]
        old_path = data_dir / old_folder
        new_path = data_dir / new_folder

        if not old_path.exists() and new_path.exists():
            # Already renamed in a previous run; just update registry.
            seq_action = "registry-only"
        elif not old_path.exists():
            logging.warning(f"folder missing, skipping: {old_path}")
            continue
        else:
            seq_action = "rename"

        if new_path in dest_targets and dest_targets[new_path] != old_folder:
            raise SystemExit(
                f"COLLISION: two folders would target {new_path}: "
                f"{dest_targets[new_path]} and {old_folder}"
            )
        dest_targets[new_path] = old_folder

        plan.append(
            {
                "id": seq["id"],
                "old_camera": old_camera,
                "new_camera": new_camera,
                "old_folder": old_folder,
                "new_folder": new_folder,
                "old_path": old_path,
                "new_path": new_path,
                "action": seq_action,
            }
        )

    n_rename = sum(1 for p in plan if p["action"] == "rename")
    n_registry_only = sum(1 for p in plan if p["action"] == "registry-only")
    by_site = defaultdict(int)
    for p in plan:
        by_site[split_camera(p["old_camera"])[0]] += 1
    logging.info(
        f"plan: {len(plan)} sequences  "
        f"(rename folders: {n_rename}, registry-only: {n_registry_only})"
    )
    for site in sorted(by_site):
        logging.info(f"  {site}: {by_site[site]} sequences")

    if not args.apply:
        print("\n[DRY RUN] sample of first 8 actions:")
        for p in plan[:8]:
            print(f"  {p['action']:<14} {p['old_folder']}  →  {p['new_folder']}")
        if len(plan) > 8:
            print(f"  ... and {len(plan) - 8} more")
        print(f"\n→ run with --apply to perform {len(plan)} updates.")
        return

    # Apply renames
    n_files_renamed = 0
    for p in plan:
        if p["action"] == "rename":
            old_path: Path = p["old_path"]
            new_path: Path = p["new_path"]
            new_path.parent.mkdir(parents=True, exist_ok=True)
            # Step 1: rename files inside images/ and labels/ (filenames embed camera+azimuth)
            old_prefix = f"{split_camera(p['old_camera'])[0]}_{split_camera(p['old_camera'])[1]}_"
            new_prefix = f"{split_camera(p['new_camera'])[0]}_{split_camera(p['new_camera'])[1]}_"
            for sub in ("images", "labels"):
                d = old_path / sub
                if not d.is_dir():
                    continue
                for f in d.iterdir():
                    if not f.is_file():
                        continue
                    if not f.name.startswith(old_prefix):
                        continue
                    target = d / (new_prefix + f.name[len(old_prefix):])
                    f.rename(target)
                    n_files_renamed += 1
            # Step 2: rename the folder itself
            old_path.rename(new_path)

    # Apply registry updates
    new_camera_by_id = {p["id"]: p["new_camera"] for p in plan}
    new_folder_by_id = {p["id"]: p["new_folder"] for p in plan}
    for seq in sequences:
        if seq["id"] in new_camera_by_id:
            seq["camera"] = new_camera_by_id[seq["id"]]
            seq["folder"] = new_folder_by_id[seq["id"]]
    registry_path.write_text(json.dumps(registry, indent=2) + "\n")

    logging.info(
        f"\nDone. renamed folders={n_rename}  "
        f"renamed inner files={n_files_renamed}  "
        f"registry entries updated={len(plan)}\n"
        f"Now refresh DVC: `dvc add {src}`"
    )


if __name__ == "__main__":
    main()
