import subprocess
import sys
from pathlib import Path

SCRIPT = Path("scripts/merge_yolo_dataset.py")


def make_yolo(root: Path, stem: str, label: str) -> Path:
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
    (root / "images" / "train" / f"{stem}.jpg").write_bytes(b"jpg")
    (root / "labels" / "train" / f"{stem}.txt").write_text(label)
    return root


def run_merge(tmp_path: Path, wf: Path, fp: Path):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--wf-dataset",
            str(wf),
            "--fp-dataset",
            str(fp),
            "--output-train-val",
            str(tmp_path / "out_tv"),
            "--output-test",
            str(tmp_path / "out_test"),
        ],
        capture_output=True,
        text=True,
    )


def test_a_name_in_both_datasets_fails_instead_of_overwriting(tmp_path):
    wf = make_yolo(tmp_path / "wf", "cam_1_2024-01-01T00-00-00", "0 0.5 0.5 0.1 0.1\n")
    fp = make_yolo(tmp_path / "fp", "cam_1_2024-01-01T00-00-00", "")
    previous = tmp_path / "out_tv" / "keep.txt"
    previous.parent.mkdir()
    previous.write_text("previous output")

    result = run_merge(tmp_path, wf, fp)

    assert result.returncode != 0
    assert "exist in both" in result.stderr
    assert previous.exists(), "nothing is deleted before the collision check"


def test_distinct_names_merge(tmp_path):
    wf = make_yolo(tmp_path / "wf", "cam_1_2024-01-01T00-00-00", "0 0.5 0.5 0.1 0.1\n")
    fp = make_yolo(tmp_path / "fp", "cam_1_2024-01-01T00-00-30", "")

    result = run_merge(tmp_path, wf, fp)

    assert result.returncode == 0, result.stderr
    assert len(list((tmp_path / "out_tv" / "images" / "train").iterdir())) == 2
