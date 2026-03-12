"""Core logic for sequence registry management and train/val/test split assignment."""

import dataclasses
import json
import random
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Sequence validation
# ---------------------------------------------------------------------------

_FOLDER_RE = re.compile(
    r"^[a-zA-Z0-9-]+_[a-zA-Z0-9-]+_(\d{1,3})_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}$"
)
_FILE_RE = re.compile(
    r"^[a-zA-Z0-9-]+_[a-zA-Z0-9-]+_\d{1,3}_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.[a-zA-Z]+$"
)
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclasses.dataclass
class ValidationResult:
    folder: str
    naming_issues: list[str]   # warnings: folder/image name format
    structural_issues: list[str]  # errors: missing dirs, too few labels

    @property
    def has_naming_issues(self) -> bool:
        return len(self.naming_issues) > 0

    @property
    def has_structural_issues(self) -> bool:
        return len(self.structural_issues) > 0

    @property
    def is_valid(self) -> bool:
        return not self.has_naming_issues and not self.has_structural_issues


def validate_sequence_folder(folder_path: Path) -> ValidationResult:
    """
    Validate a sequence folder before ingestion.

    Naming issues (warnings — can be overridden by user):
      - Folder name does not match source_camera[_azimuth]_timestamp format.
      - Image filenames do not match the naming convention.

    Structural issues (hard errors — folder must be fixed or skipped):
      - Missing images/ or labels/ subdirectory.
      - Fewer than 2 non-empty label files.
    """
    naming: list[str] = []
    structural: list[str] = []
    name = folder_path.name

    m = _FOLDER_RE.match(name)
    if not m:
        naming.append(
            f"folder name does not match <source>_<camera>_<azimuth(0-360|999)>_<timestamp>: '{name}'"
        )
    else:
        az = int(m.group(1))
        if not (az <= 360 or az == 999):
            naming.append(f"azimuth {az} out of valid range (must be 0-360 or 999)")

    dir_images = folder_path / "images"
    dir_labels = folder_path / "labels"

    if not dir_images.is_dir():
        structural.append("missing 'images/' subdirectory")
    if not dir_labels.is_dir():
        structural.append("missing 'labels/' subdirectory")

    if not dir_images.is_dir() or not dir_labels.is_dir():
        return ValidationResult(folder=name, naming_issues=naming, structural_issues=structural)

    # Image filename format (naming warning)
    bad_images = [
        f.name
        for f in dir_images.iterdir()
        if f.suffix.lower() in _IMAGE_EXTENSIONS and not _FILE_RE.match(f.name)
    ]
    if bad_images:
        sample = bad_images[:3]
        extra = f" (+{len(bad_images) - 3} more)" if len(bad_images) > 3 else ""
        naming.append(
            f"{len(bad_images)} image(s) with invalid filename: {sample}{extra}"
        )

    total_images = sum(1 for f in dir_images.iterdir() if f.suffix.lower() in _IMAGE_EXTENSIONS)
    non_empty_labels = [f for f in dir_labels.glob("*.txt") if f.stat().st_size > 0]
    if len(non_empty_labels) == 0 or (len(non_empty_labels) == 1 and total_images > 2):
        structural.append(
            f"{len(non_empty_labels)} non-empty label file(s) for {total_images} image(s) "
            f"(need at least 1, or 2+ if images > 2)"
        )

    return ValidationResult(folder=name, naming_issues=naming, structural_issues=structural)


@dataclasses.dataclass
class ValidationSummary:
    valid: list[str]
    rejected: list[ValidationResult]


def validate_source_folders(src: Path, folders: list[str]) -> ValidationSummary:
    valid, rejected = [], []
    for folder in folders:
        result = validate_sequence_folder(src / folder)
        if result.is_valid:
            valid.append(folder)
        else:
            rejected.append(result)
    return ValidationSummary(valid=valid, rejected=rejected)


# ---------------------------------------------------------------------------
# Split targets and registry logic
# ---------------------------------------------------------------------------

SPLIT_TARGETS = {"train": 0.8, "val": 0.1, "test": 0.1}
SPLITS = list(SPLIT_TARGETS.keys())

# Matches the start of a timestamp: _YYYY (4-digit year)
_TIMESTAMP_RE = re.compile(r"_\d{4}[^a-zA-Z]")


def extract_camera(folder_name: str) -> str:
    """
    Extract the camera ref from a folder name.

    Camera ref is everything before the first timestamp-like pattern (_YYYY...).

    Examples:
        "sdis83_2024-01-15T10-30-00"   -> "sdis83"
        "pyro_cam01_2023-06-01T08-00"  -> "pyro_cam01"
        "awf_station3_20240315"         -> "awf_station3"
    """
    match = _TIMESTAMP_RE.search(folder_name)
    if match:
        return folder_name[: match.start()]
    return folder_name


def load_registry(registry_path: Path) -> list[dict]:
    """Load registry from JSON, return empty list if file does not exist."""
    if not registry_path.exists():
        return []
    with registry_path.open() as f:
        data = json.load(f)
    return data.get("sequences", [])


def save_registry(registry_path: Path, sequences: list[dict]) -> None:
    with registry_path.open("w") as f:
        json.dump({"sequences": sequences}, f, indent=2)
        f.write("\n")


def next_id(sequences: list[dict], prefix: str) -> int:
    """Return the next integer ID for the given prefix."""
    existing = [
        int(s["id"].split("_")[1])
        for s in sequences
        if s["id"].startswith(f"{prefix}_")
    ]
    return max(existing, default=0) + 1


def assign_split(counts: dict[str, int]) -> str:
    """
    Return the split with the largest deficit vs the 80/10/10 target.

    counts: current number of sequences already assigned to each split
            (for this camera, including any already assigned in this batch).
    """
    total = sum(counts.values()) + 1  # +1 for the sequence being assigned
    deficits = {
        split: SPLIT_TARGETS[split] - counts[split] / total for split in SPLITS
    }
    return max(deficits, key=lambda s: deficits[s])


def rebalance_minority_splits(
    new_assignments: list[dict],
    existing: list[dict],
) -> list[dict]:
    """
    Swap val↔test in new_assignments to improve global balance.

    After per-camera greedy assignment, val and test may be unequal because
    cameras with few sequences can only fill train before reaching minority
    splits. This pass swaps sequences between val and test (never touching
    train or existing entries) to minimise |val_count - test_count| globally.

    Swaps are taken from cameras with the largest surplus in the source split,
    subject to keeping at least 1 sequence per split per camera (where possible).
    """
    all_seqs = existing + new_assignments

    current_val = sum(1 for s in all_seqs if s["split"] == "val")
    current_test = sum(1 for s in all_seqs if s["split"] == "test")

    # Build mutable per-camera counts across ALL sequences
    camera_counts: dict[str, dict[str, int]] = {}
    for s in all_seqs:
        cam = s["camera"]
        camera_counts.setdefault(cam, {sp: 0 for sp in SPLITS})
        camera_counts[cam][s["split"]] += 1

    result = [dict(s) for s in new_assignments]

    def _swap(src: str, dst: str, n: int) -> None:
        """Swap up to n sequences from src→dst in result (in-place)."""
        swapped = 0
        idxs = [i for i, s in enumerate(result) if s["split"] == src]
        idxs.sort(
            key=lambda i: (-camera_counts[result[i]["camera"]][src], result[i]["id"])
        )
        for i in idxs:
            if swapped >= n:
                break
            cam = result[i]["camera"]
            if camera_counts[cam][src] <= 1:
                continue
            result[i]["split"] = dst
            camera_counts[cam][src] -= 1
            camera_counts[cam][dst] += 1
            swapped += 1

    if current_val > current_test:
        _swap("val", "test", (current_val - current_test) // 2)
    elif current_test > current_val:
        _swap("test", "val", (current_test - current_val) // 2)

    return result


def compute_new_assignments(
    new_folders: list[str],
    existing: list[dict],
    start_id: int,
    prefix: str,
    random_seed: int = 0,
) -> list[dict]:
    """
    Assign IDs and splits to new_folders, then rebalance val/test globally.

    Groups by camera. Within each camera, sequences are shuffled (using
    random_seed) before assignment to avoid temporal bias. For each camera,
    loads existing split counts then greedily assigns each new sequence to
    the most under-represented split. A rebalance pass follows to fix any
    global val/test imbalance introduced by cameras with few sequences.
    """
    camera_counts: dict[str, dict[str, int]] = {}
    for entry in existing:
        cam = entry["camera"]
        camera_counts.setdefault(cam, {s: 0 for s in SPLITS})
        camera_counts[cam][entry["split"]] += 1

    camera_folders: dict[str, list[str]] = {}
    for folder in sorted(new_folders):
        cam = extract_camera(folder)
        camera_folders.setdefault(cam, []).append(folder)

    rng = random.Random(random_seed)

    assignments: list[dict] = []
    current_id = start_id

    for cam in sorted(camera_folders):
        folders = list(camera_folders[cam])
        rng.shuffle(folders)
        counts = camera_counts.get(cam, {s: 0 for s in SPLITS})

        for folder in folders:
            split = assign_split(counts)
            assignments.append(
                {
                    "id": f"{prefix}_{current_id:08d}",
                    "folder": folder,
                    "camera": cam,
                    "split": split,
                }
            )
            counts[split] += 1
            current_id += 1

    return rebalance_minority_splits(assignments, existing)


def print_summary(new_assignments: list[dict], all_sequences: list[dict]) -> None:
    """Print a summary of new assignments and overall split distribution."""
    if not new_assignments:
        print("No new sequences found.")
        return

    print(f"\n{'='*50}")
    print(f"New sequences: {len(new_assignments)}")

    cameras: dict[str, dict] = {}
    for a in new_assignments:
        cam = a["camera"]
        cameras.setdefault(cam, {s: 0 for s in SPLITS})
        cameras[cam][a["split"]] += 1

    print(f"\n{'Camera':<30} {'Train':>6} {'Val':>6} {'Test':>6}")
    print("-" * 50)
    for cam, counts in sorted(cameras.items()):
        print(f"{cam:<30} {counts['train']:>6} {counts['val']:>6} {counts['test']:>6}")

    total_counts = {s: 0 for s in SPLITS}
    for seq in all_sequences:
        total_counts[seq["split"]] += 1
    total = sum(total_counts.values())

    print(f"\n{'='*50}")
    print(f"Overall split distribution ({total} total sequences):")
    for split in SPLITS:
        n = total_counts[split]
        pct = n / total * 100 if total else 0
        target_pct = SPLIT_TARGETS[split] * 100
        print(f"  {split:<6}: {n:>4}  ({pct:.1f}%  target {target_pct:.0f}%)")
    print(f"{'='*50}\n")
