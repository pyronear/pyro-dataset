"""add_data.py must not touch the pool when the split assignment is refused.

`assignments_from_splits` rejects a folder missing from splits.json and one
pre-assigned to test. Copying before that check left folders in
data/raw/<type>/data with no registry entry, and a re-run skipped them as
"already on disk" — stranding them until someone deleted them by hand.
"""

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def make_sequence_folder(root: Path, name: str) -> None:
    """A folder ingest accepts: frame stems carry the camera key and a
    timestamp, per _FILE_RE."""
    camera_key = name.rsplit("_", 1)[0]
    (root / name / "images").mkdir(parents=True)
    (root / name / "labels").mkdir(parents=True)
    for i in range(3):
        stem = f"{camera_key}_2026-08-05T13-46-{8 + i:02d}"
        (root / name / "images" / f"{stem}.jpg").write_bytes(b"x")
        (root / name / "labels" / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1\n")


def run_add_data(cwd: Path, src: Path, splits: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "add_data.py"),
            "--src",
            str(src),
            "--type",
            "wildfire",
            "--splits-from",
            str(splits),
        ],
        capture_output=True,
        text=True,
        cwd=cwd,
        env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"},
    )


def prepare(tmp_path: Path, split: str | None) -> tuple[Path, Path, Path]:
    src = tmp_path / "staging"
    folder = "sdis-91_cam-a_285_2026-08-05T13-46-08"
    make_sequence_folder(src, folder)
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({folder: split} if split else {}))
    pool = tmp_path / "data" / "raw" / "wildfire" / "data"
    pool.mkdir(parents=True)
    return src, splits, pool


def test_a_test_split_is_refused_without_copying(tmp_path):
    src, splits, pool = prepare(tmp_path, "test")
    result = run_add_data(tmp_path, src, splits)
    assert result.returncode != 0
    assert list(pool.iterdir()) == []


def test_a_missing_split_entry_is_refused_without_copying(tmp_path):
    src, splits, pool = prepare(tmp_path, None)
    result = run_add_data(tmp_path, src, splits)
    assert result.returncode != 0
    assert list(pool.iterdir()) == []


def test_a_valid_split_copies_and_registers(tmp_path):
    src, splits, pool = prepare(tmp_path, "train")
    result = run_add_data(tmp_path, src, splits)
    assert result.returncode == 0, result.stderr
    assert [p.name for p in pool.iterdir()] == ["sdis-91_cam-a_285_2026-08-05T13-46-08"]
    registry = json.loads(
        (tmp_path / "data" / "raw" / "wildfire" / "registry.json").read_text()
    )["sequences"]
    assert registry[0]["split"] == "train"
    assert registry[0]["source"] == "pyro-annotator"
