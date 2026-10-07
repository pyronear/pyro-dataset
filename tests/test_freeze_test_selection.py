"""The freeze command owns the append-only test-negatives lockfile.

Covers bootstrap (seed from the built dataset, never recompute), appending
every registered test FP, and the hard errors.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path("scripts/freeze_test_selection.py")

spec = importlib.util.spec_from_file_location("freeze_test_selection", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def write_registry(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"sequences": entries}))


def make_seq_dir(root: Path, name: str) -> None:
    (root / name / "images").mkdir(parents=True)
    (root / name / "labels").mkdir(parents=True)


def setup_world(tmp_path: Path, fp_test_folders: list[str]) -> dict:
    """FP registry + data dirs for a freeze run. Returns paths and CLI argv."""
    fp_registry = tmp_path / "fp" / "registry.json"
    fp_data = tmp_path / "fp" / "data"
    entries = []
    for i, folder in enumerate(fp_test_folders):
        entries.append(
            {"id": f"fp_{i:08d}", "folder": folder, "camera": "c", "split": "test"}
        )
        make_seq_dir(fp_data, folder)
    write_registry(fp_registry, entries)
    return {
        "fp_registry": fp_registry,
        "fp_data": fp_data,
        "lockfile": tmp_path / "lock.json",
        "argv": [
            "freeze_test_selection.py",
            "--fp-registry",
            str(fp_registry),
            "--fp-data-dir",
            str(fp_data),
            "--lockfile",
            str(tmp_path / "lock.json"),
            "--bootstrap-from",
            str(tmp_path / "built"),
        ],
    }


def read_lock(world: dict) -> list[str]:
    return json.loads(world["lockfile"].read_text())["folders"]


def test_bootstrap_seeds_from_the_built_dataset(tmp_path, monkeypatch):
    world = setup_world(tmp_path, fp_test_folders=["f1", "f2"])
    for name in ("f2", "f1"):
        (tmp_path / "built" / "test" / "fp" / name).mkdir(parents=True)
    monkeypatch.setattr(sys, "argv", world["argv"])
    module.main()
    assert read_lock(world) == ["f1", "f2"], "sorted at bootstrap, exact set"


def test_bootstrap_without_a_built_dataset_is_an_error(tmp_path, monkeypatch):
    world = setup_world(tmp_path, fp_test_folders=["f1"])
    monkeypatch.setattr(sys, "argv", world["argv"])
    with pytest.raises(SystemExit):
        module.main()
    assert not world["lockfile"].exists()


def test_rerun_with_nothing_new_is_a_noop(tmp_path, monkeypatch):
    world = setup_world(tmp_path, fp_test_folders=["f1"])
    world["lockfile"].write_text(json.dumps({"folders": ["f1"]}))
    monkeypatch.setattr(sys, "argv", world["argv"])
    module.main()
    assert read_lock(world) == ["f1"]


def test_every_test_fp_is_appended_in_registry_order(tmp_path, monkeypatch):
    world = setup_world(tmp_path, fp_test_folders=["f1", "f3", "f2"])
    world["lockfile"].write_text(json.dumps({"folders": ["f2"]}))
    monkeypatch.setattr(sys, "argv", world["argv"])
    module.main()
    assert read_lock(world) == ["f2", "f1", "f3"], "frozen first, then the rest"


def test_an_overfull_lockfile_is_never_truncated(tmp_path, monkeypatch):
    world = setup_world(tmp_path, fp_test_folders=["f1"])
    world["lockfile"].write_text(json.dumps({"folders": ["f1", "f2"]}))
    monkeypatch.setattr(sys, "argv", world["argv"])
    with pytest.raises(SystemExit):
        module.main()
    assert read_lock(world) == ["f1", "f2"], "untouched on error"


def test_an_unregistered_frozen_folder_is_an_error(tmp_path, monkeypatch):
    world = setup_world(tmp_path, fp_test_folders=["f1"])
    world["lockfile"].write_text(json.dumps({"folders": ["ghost"]}))
    monkeypatch.setattr(sys, "argv", world["argv"])
    with pytest.raises(SystemExit):
        module.main()


def test_dry_run_writes_nothing(tmp_path):
    world = setup_world(tmp_path, fp_test_folders=["f1"])
    (tmp_path / "built" / "test" / "fp" / "f1").mkdir(parents=True)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *world["argv"][1:], "--dry-run"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not world["lockfile"].exists()
