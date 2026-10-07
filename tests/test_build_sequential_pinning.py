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


def make_extra(tmp_path, wildfire=(), fp=()):
    for kind, names in (("wildfire", wildfire), ("fp", fp)):
        (tmp_path / "extra" / kind).mkdir(parents=True, exist_ok=True)
        for name in names:
            (tmp_path / "extra" / kind / name / "images").mkdir(parents=True)
    return tmp_path / "extra"


def test_extra_test_sequences_lists_both_kinds(tmp_path):
    extra = make_extra(tmp_path, wildfire=["w2", "w1"], fp=["f1"])
    found = module.extra_test_sequences(extra, registered={"other"})
    assert [p.name for p in found["wildfire"]] == ["w1", "w2"]
    assert [p.name for p in found["fp"]] == ["f1"]


def test_no_extra_dir_adds_nothing():
    assert module.extra_test_sequences(None, registered=set()) == {
        "wildfire": [],
        "fp": [],
    }


def test_an_extra_folder_also_registered_is_an_error(tmp_path):
    extra = make_extra(tmp_path, fp=["f1"])
    with pytest.raises(SystemExit, match="f1"):
        module.extra_test_sequences(extra, registered={"f1"})
