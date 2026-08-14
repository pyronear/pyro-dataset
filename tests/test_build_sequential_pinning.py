"""The sequential build must include annotator-sourced FP sequences.

Adding a sequence to data/raw/fp does not put it in the built dataset: the
build picks `quota` representatives by clustering the whole pool and may
simply not choose it. Sequences the importer selected deliberately are pinned.
"""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path("scripts/build_sequential_dataset.py")

spec = importlib.util.spec_from_file_location("build_sequential_dataset", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_partition_pinned_splits_on_source():
    pinned, rest = module.partition_pinned(
        [
            {"folder": "a", "source": "pyro-annotator"},
            {"folder": "b"},
            {"folder": "c", "source": "platform"},
        ]
    )
    assert [s["folder"] for s in pinned] == ["a"]
    assert [s["folder"] for s in rest] == ["b", "c"]


def test_partition_pinned_handles_an_empty_pool():
    assert module.partition_pinned([]) == ([], [])


def test_pinned_sequences_consume_quota_before_clustering():
    pinned, _ = module.partition_pinned(
        [{"folder": f"a{i}", "source": "pyro-annotator"} for i in range(4)]
    )
    assert module.remaining_quota(quota=10, pinned=pinned) == 6


def test_remaining_quota_never_goes_negative():
    pinned, _ = module.partition_pinned(
        [{"folder": f"a{i}", "source": "pyro-annotator"} for i in range(12)]
    )
    assert module.remaining_quota(quota=10, pinned=pinned) == 0


def test_remaining_quota_is_untouched_without_pinned_sequences():
    assert module.remaining_quota(quota=10, pinned=[]) == 10


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
