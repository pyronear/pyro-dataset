"""Move sequences from fp to wildfire, updating both registry files.

Usage:
    # Move specific sequences
    uv run python scripts/move_fp_to_wildfire.py pyronear-sdis-07_brison_110_2024-03-27T15-58-18

    # Move multiple sequences
    uv run python scripts/move_fp_to_wildfire.py seq1 seq2 seq3

    # Move all sequences matching a pattern
    uv run python scripts/move_fp_to_wildfire.py --pattern "pyronear-sdis-07_brison_110"

    # Dry run (no changes applied)
    uv run python scripts/move_fp_to_wildfire.py --dry-run pyronear-sdis-07_brison_110_2024-03-27T15-58-18
"""

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
FP_DIR = ROOT / "data/raw/fp"
WF_DIR = ROOT / "data/raw/wildfire"
FP_REGISTRY = FP_DIR / "registry.json"
WF_REGISTRY = WF_DIR / "registry.json"


def load_registry(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_registry(path: Path, registry: dict) -> None:
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)
        f.write("\n")


def next_wf_id(sequences: list[dict]) -> str:
    if not sequences:
        return "wf_00000001"
    max_num = max(int(s["id"].split("_")[1]) for s in sequences)
    return f"wf_{max_num + 1:08d}"


def move_sequences(folders: list[str], dry_run: bool = False) -> None:
    fp_reg = load_registry(FP_REGISTRY)
    wf_reg = load_registry(WF_REGISTRY)

    fp_by_folder = {s["folder"]: s for s in fp_reg["sequences"]}
    wf_folders = {s["folder"] for s in wf_reg["sequences"]}

    to_move = []
    for folder in folders:
        if folder not in fp_by_folder:
            print(f"  SKIP  {folder} — not found in fp registry")
            continue
        if folder in wf_folders:
            print(f"  SKIP  {folder} — already in wildfire registry")
            continue
        src = FP_DIR / "data" / folder
        if not src.exists():
            print(f"  SKIP  {folder} — folder not found at {src}")
            continue
        to_move.append(fp_by_folder[folder])

    if not to_move:
        print("Nothing to move.")
        return

    print(f"\n{'DRY RUN — ' if dry_run else ''}Moving {len(to_move)} sequence(s):\n")

    new_fp_sequences = [s for s in fp_reg["sequences"] if s["folder"] not in {s["folder"] for s in to_move}]

    for entry in to_move:
        folder = entry["folder"]
        src = FP_DIR / "data" / folder
        dst = WF_DIR / "data" / folder
        new_id = next_wf_id(wf_reg["sequences"])
        new_entry = {"id": new_id, "folder": folder, "camera": entry["camera"], "split": entry["split"]}

        print(f"  {entry['id']} → {new_id}  {folder}  (split={entry['split']})")
        if not dry_run:
            shutil.move(str(src), str(dst))
            wf_reg["sequences"].append(new_entry)

    if not dry_run:
        fp_reg["sequences"] = new_fp_sequences
        save_registry(FP_REGISTRY, fp_reg)
        save_registry(WF_REGISTRY, wf_reg)
        print(f"\nRegistries updated. wildfire: {len(wf_reg['sequences'])} seqs, fp: {len(fp_reg['sequences'])} seqs.")
    else:
        print("\n(Dry run — no changes applied)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Move sequences from fp to wildfire.")
    parser.add_argument("folders", nargs="*", help="Sequence folder name(s) to move")
    parser.add_argument("--pattern", help="Move all fp sequences whose folder name contains this substring")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying them")
    args = parser.parse_args()

    if args.pattern:
        fp_reg = load_registry(FP_REGISTRY)
        folders = [s["folder"] for s in fp_reg["sequences"] if args.pattern in s["folder"]]
        print(f"Pattern '{args.pattern}' matched {len(folders)} sequences in fp registry.")
    else:
        folders = args.folders

    if not folders:
        parser.print_help()
        return

    move_sequences(folders, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
