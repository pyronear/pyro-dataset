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


def make_sequence_folder(root: Path, name: str, content: bytes = b"x") -> None:
    """A folder ingest accepts: frame stems carry the camera key and a
    timestamp, per _FILE_RE."""
    camera_key = name.rsplit("_", 1)[0]
    (root / name / "images").mkdir(parents=True)
    (root / name / "labels").mkdir(parents=True)
    for i in range(3):
        stem = f"{camera_key}_2026-08-05T13-46-{8 + i:02d}"
        (root / name / "images" / f"{stem}.jpg").write_bytes(content)
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


def test_a_test_split_is_copied_and_registered(tmp_path):
    src, splits, pool = prepare(tmp_path, "test")
    result = run_add_data(tmp_path, src, splits)
    assert result.returncode == 0, result.stderr
    assert [p.name for p in pool.iterdir()] == ["sdis-91_cam-a_285_2026-08-05T13-46-08"]
    registry = json.loads(
        (tmp_path / "data" / "raw" / "wildfire" / "registry.json").read_text()
    )["sequences"]
    assert registry[0]["split"] == "test"


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


def test_a_folder_registered_in_the_other_pool_is_refused(tmp_path):
    src, splits, pool = prepare(tmp_path, "train")
    folder = "sdis-91_cam-a_285_2026-08-05T13-46-08"
    fp_raw = tmp_path / "data" / "raw" / "fp"
    fp_raw.mkdir(parents=True)
    (fp_raw / "registry.json").write_text(
        json.dumps({"sequences": [{"folder": folder, "split": "train"}]})
    )
    result = run_add_data(tmp_path, src, splits)
    assert result.returncode != 0
    assert "already registered in" in result.stdout
    assert list(pool.iterdir()) == []


def test_a_folder_sharing_an_image_with_a_pool_is_refused(tmp_path):
    src, splits, pool = prepare(tmp_path, "train")
    make_sequence_folder(
        tmp_path / "data" / "raw" / "fp" / "data",
        "sdis-91_cam-a_300_2026-08-05T13-46-00",
    )
    result = run_add_data(tmp_path, src, splits)
    assert result.returncode != 0
    assert "already in a pool by content" in result.stdout
    assert list(pool.iterdir()) == []


def test_two_incoming_folders_sharing_an_image_keep_only_the_first(tmp_path):
    src, splits, pool = prepare(tmp_path, "train")
    second = "sdis-91_cam-a_285_2026-08-05T13-50-00"
    make_sequence_folder(src, second)
    splits.write_text(
        json.dumps({"sdis-91_cam-a_285_2026-08-05T13-46-08": "train", second: "train"})
    )
    result = run_add_data(tmp_path, src, splits)
    assert [p.name for p in pool.iterdir()] == ["sdis-91_cam-a_285_2026-08-05T13-46-08"]
    assert second in result.stdout


def test_distinct_images_are_accepted(tmp_path):
    src, splits, pool = prepare(tmp_path, "train")
    make_sequence_folder(
        tmp_path / "data" / "raw" / "fp" / "data",
        "sdis-91_cam-a_300_2026-08-05T13-46-00",
        content=b"other",
    )
    result = run_add_data(tmp_path, src, splits)
    assert result.returncode == 0, result.stderr
    assert len(list(pool.iterdir())) == 1
