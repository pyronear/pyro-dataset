"""The sequential build copies the frozen test negatives verbatim.

The lockfile IS the test FP selection: the build performs no selection for
test and errors on any mismatch, so the test set only ever grows
(docs/specs/2026-08-14-annotator-test-growth-design.md §3).

The pinning helpers this file used to cover (`partition_pinned`,
`remaining_quota`) moved to `pyro_dataset.fp.selection` and are tested in
tests/test_fp_pinning.py.
"""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path("scripts/build_sequential_dataset.py")

spec = importlib.util.spec_from_file_location("build_sequential_dataset", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_lock(tmp_path, folders):
    path = tmp_path / "lock.json"
    path.write_text(json.dumps({"folders": folders}))
    return path


def test_frozen_test_selection_returns_lockfile_paths_in_order(tmp_path):
    for name in ("f1", "f2"):
        (tmp_path / "data" / name / "images").mkdir(parents=True)
    lock = make_lock(tmp_path, ["f2", "f1"])
    paths = module.frozen_test_selection(
        lock, quota=2, registered={"f1", "f2"}, data_dir=tmp_path / "data"
    )
    assert [p.name for p in paths] == ["f2", "f1"], "lockfile order, verbatim"


def test_a_missing_lockfile_is_an_error(tmp_path):
    with pytest.raises(SystemExit, match="freeze_test_selection"):
        module.frozen_test_selection(
            tmp_path / "lock.json", quota=1, registered=set(), data_dir=tmp_path
        )


def test_a_quota_mismatch_is_an_error(tmp_path):
    lock = make_lock(tmp_path, ["f1"])
    with pytest.raises(SystemExit, match="quota"):
        module.frozen_test_selection(
            lock, quota=2, registered={"f1"}, data_dir=tmp_path
        )


def test_an_unregistered_or_missing_folder_is_an_error(tmp_path):
    lock = make_lock(tmp_path, ["f1"])
    with pytest.raises(SystemExit, match="f1"):
        module.frozen_test_selection(lock, quota=1, registered=set(), data_dir=tmp_path)
