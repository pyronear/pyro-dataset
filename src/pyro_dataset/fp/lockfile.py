"""Append-only lockfile of the FP folders frozen into `sequential_test`.

The built test set is a selection, not a copy: without recorded state, any
growth of the test pool re-rolls the KMeans pick of negatives and two models
end up scored on different data. This lockfile *is* the FP half of the built
test split — ordered, git-committed, appended to by
`scripts/freeze_test_selection.py` and read verbatim by
`scripts/build_sequential_dataset.py`, which performs no selection for test.

Design: docs/specs/2026-08-14-annotator-test-growth-design.md.
"""

import json
from pathlib import Path


def load_lockfile(path: Path) -> list[str]:
    """The frozen folder list, [] when no lockfile exists yet."""
    if not path.is_file():
        return []
    return json.loads(path.read_text())["folders"]


def save_lockfile(path: Path, folders: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"folders": folders}, indent=2) + "\n")


def extend_with_pins(frozen: list[str], pins: list[str], quota: int) -> list[str]:
    """Append pins in order until the quota is met. Frozen entries are never
    removed or reordered — the list is history. Pins beyond the quota are
    deferred, not dropped: they stay in the pool for a later freeze (spec §5).
    """
    if len(frozen) > quota:
        raise ValueError(
            f"lockfile holds {len(frozen)} folders, which exceeds the quota "
            f"of {quota} — a lockfile is never truncated; fix it by hand only "
            "if you know why it grew"
        )
    out = list(frozen)
    seen = set(frozen)
    for folder in pins:
        if len(out) >= quota:
            break
        if folder in seen:
            continue
        out.append(folder)
        seen.add(folder)
    return out


def validate_lockfile(
    folders: list[str], registered: set[str], data_dir: Path
) -> list[str]:
    """Problems that make the lockfile impossible to materialise exactly.

    An empty return is the only pass: a frozen test set that cannot be built
    byte-for-byte is a comparability bug, so callers must error, not warn.
    """
    problems: list[str] = []
    seen: set[str] = set()
    for folder in folders:
        if folder in seen:
            problems.append(f"{folder}: duplicate lockfile entry")
            continue
        seen.add(folder)
        if folder not in registered:
            problems.append(f"{folder}: not a registered test FP sequence")
        elif not (data_dir / folder).is_dir():
            problems.append(f"{folder}: missing on disk under {data_dir}")
    return problems
