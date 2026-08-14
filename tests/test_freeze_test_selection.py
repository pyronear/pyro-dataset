"""The freeze command owns the append-only test-negatives lockfile.

Covers bootstrap (seed from the built dataset, never recompute), pinning up
to quota, the two-stage fill of remaining slots, and the hard errors.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
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


def setup_world(
    tmp_path: Path,
    n_wf_test: int,
    fp_test_folders: list[str],
    annotator_fp: tuple[str, ...] = (),
) -> dict:
    """Registries + data dirs for a freeze run. Returns paths and CLI argv."""
    wf_registry = tmp_path / "wf" / "registry.json"
    fp_registry = tmp_path / "fp" / "registry.json"
    fp_data = tmp_path / "fp" / "data"
    write_registry(
        wf_registry,
        [
            {"id": f"wf_{i:08d}", "folder": f"wf{i}", "camera": "c", "split": "test"}
            for i in range(n_wf_test)
        ],
    )
    entries = []
    for i, folder in enumerate(fp_test_folders):
        entry = {"id": f"fp_{i:08d}", "folder": folder, "camera": "c", "split": "test"}
        if folder in annotator_fp:
            entry["source"] = "pyro-annotator"
        entries.append(entry)
        make_seq_dir(fp_data, folder)
    write_registry(fp_registry, entries)
    return {
        "wf_registry": wf_registry,
        "fp_registry": fp_registry,
        "fp_data": fp_data,
        "lockfile": tmp_path / "lock.json",
        "argv": [
            "freeze_test_selection.py",
            "--wf-registry",
            str(wf_registry),
            "--fp-registry",
            str(fp_registry),
            "--fp-data-dir",
            str(fp_data),
            "--embeddings-dir",
            str(tmp_path / "emb"),
            "--lockfile",
            str(tmp_path / "lock.json"),
            "--bootstrap-from",
            str(tmp_path / "built"),
        ],
    }


def read_lock(world: dict) -> list[str]:
    return json.loads(world["lockfile"].read_text())["folders"]


def test_bootstrap_seeds_from_the_built_dataset(tmp_path, monkeypatch):
    world = setup_world(tmp_path, n_wf_test=2, fp_test_folders=["f1", "f2"])
    for name in ("f2", "f1"):
        (tmp_path / "built" / "test" / "fp" / name).mkdir(parents=True)
    monkeypatch.setattr(sys, "argv", world["argv"])
    module.main()
    assert read_lock(world) == ["f1", "f2"], "sorted at bootstrap, exact set"


def test_bootstrap_without_a_built_dataset_is_an_error(tmp_path, monkeypatch):
    world = setup_world(tmp_path, n_wf_test=1, fp_test_folders=["f1"])
    monkeypatch.setattr(sys, "argv", world["argv"])
    with pytest.raises(SystemExit):
        module.main()
    assert not world["lockfile"].exists()


def test_rerun_with_nothing_new_is_a_noop(tmp_path, monkeypatch):
    world = setup_world(tmp_path, n_wf_test=1, fp_test_folders=["f1"])
    world["lockfile"].write_text(json.dumps({"folders": ["f1"]}))
    monkeypatch.setattr(sys, "argv", world["argv"])
    module.main()
    assert read_lock(world) == ["f1"]


def test_annotator_pins_append_up_to_quota(tmp_path, monkeypatch):
    world = setup_world(
        tmp_path,
        n_wf_test=2,
        fp_test_folders=["f1", "p1", "p2"],
        annotator_fp=("p1", "p2"),
    )
    world["lockfile"].write_text(json.dumps({"folders": ["f1"]}))
    monkeypatch.setattr(sys, "argv", world["argv"])
    module.main()
    assert read_lock(world) == ["f1", "p1"], "p2 deferred, quota is 2"


def test_remaining_slots_are_filled_by_two_stage(tmp_path, monkeypatch):
    world = setup_world(tmp_path, n_wf_test=3, fp_test_folders=["f1", "f2", "f3"])
    world["lockfile"].write_text(json.dumps({"folders": ["f1"]}))
    monkeypatch.setattr(
        module,
        "load_embeddings",
        lambda embeddings_dir, split: (
            np.zeros((3, 4), dtype=np.float32),
            [{"sequence_folder": f} for f in ("f1", "f2", "f3")],
        ),
    )
    calls = {}

    def fake_select(**kwargs):
        calls["quota"] = kwargs["quota"]
        calls["folders"] = [it["sequence_folder"] for it in kwargs["items"]]
        return [0, 1]

    monkeypatch.setattr(module, "two_stage_select", fake_select)
    monkeypatch.setattr(sys, "argv", world["argv"])
    module.main()
    assert calls["quota"] == 2
    assert calls["folders"] == ["f2", "f3"], "frozen folders leave the pool"
    assert read_lock(world) == ["f1", "f2", "f3"]


def test_an_overfull_lockfile_is_never_truncated(tmp_path, monkeypatch):
    world = setup_world(tmp_path, n_wf_test=1, fp_test_folders=["f1", "f2"])
    world["lockfile"].write_text(json.dumps({"folders": ["f1", "f2"]}))
    monkeypatch.setattr(sys, "argv", world["argv"])
    with pytest.raises(SystemExit):
        module.main()
    assert read_lock(world) == ["f1", "f2"], "untouched on error"


def test_an_unregistered_frozen_folder_is_an_error(tmp_path, monkeypatch):
    world = setup_world(tmp_path, n_wf_test=1, fp_test_folders=["f1"])
    world["lockfile"].write_text(json.dumps({"folders": ["ghost"]}))
    monkeypatch.setattr(sys, "argv", world["argv"])
    with pytest.raises(SystemExit):
        module.main()


def test_dry_run_writes_nothing(tmp_path):
    world = setup_world(tmp_path, n_wf_test=1, fp_test_folders=["f1"])
    (tmp_path / "built" / "test" / "fp" / "f1").mkdir(parents=True)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *world["argv"][1:], "--dry-run"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not world["lockfile"].exists()
