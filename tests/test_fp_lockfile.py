import pytest

from pyro_dataset.fp.lockfile import (
    extend_with_pins,
    load_lockfile,
    save_lockfile,
    validate_lockfile,
)


def test_load_of_a_missing_file_is_empty(tmp_path):
    assert load_lockfile(tmp_path / "lock.json") == []


def test_round_trips_through_disk(tmp_path):
    path = tmp_path / "lock.json"
    save_lockfile(path, ["b", "a"])
    assert load_lockfile(path) == ["b", "a"], "order is history, never sorted"


def test_pins_append_in_order_up_to_quota():
    assert extend_with_pins(["f1"], ["p1", "p2", "p3"], quota=3) == ["f1", "p1", "p2"]


def test_pins_already_frozen_are_skipped():
    assert extend_with_pins(["p1"], ["p1", "p2"], quota=3) == ["p1", "p2"]


def test_frozen_entries_are_never_removed_or_reordered():
    frozen = ["z", "a", "m"]
    assert extend_with_pins(frozen, [], quota=5)[:3] == ["z", "a", "m"]


def test_a_lockfile_beyond_quota_is_an_error():
    with pytest.raises(ValueError, match="exceeds"):
        extend_with_pins(["f1", "f2"], [], quota=1)


def test_validate_flags_unregistered_and_missing_folders(tmp_path):
    (tmp_path / "on-disk" / "images").mkdir(parents=True)
    problems = validate_lockfile(
        ["on-disk", "not-registered", "registered-but-gone"],
        registered={"on-disk", "registered-but-gone"},
        data_dir=tmp_path,
    )
    assert len(problems) == 2
    assert any("not-registered" in p for p in problems)
    assert any("registered-but-gone" in p for p in problems)


def test_validate_flags_duplicates(tmp_path):
    (tmp_path / "f1" / "images").mkdir(parents=True)
    problems = validate_lockfile(["f1", "f1"], registered={"f1"}, data_dir=tmp_path)
    assert any("duplicate" in p for p in problems)


def test_validate_passes_a_clean_lockfile(tmp_path):
    (tmp_path / "f1" / "images").mkdir(parents=True)
    assert validate_lockfile(["f1"], registered={"f1"}, data_dir=tmp_path) == []
