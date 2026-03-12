import pytest

from pyro_dataset.ingest import (
    assign_split,
    compute_new_assignments,
    extract_camera,
    next_id,
    rebalance_minority_splits,
)


@pytest.mark.parametrize(
    "folder_name, expected",
    [
        ("sdis83_2024-01-15T10-30-00", "sdis83"),
        ("pyro_cam01_2023-06-01T08-00", "pyro_cam01"),
        ("awf_station3_20240315T12-00", "awf_station3"),
        ("no_timestamp_here", "no_timestamp_here"),
    ],
)
def test_extract_camera(folder_name, expected):
    assert extract_camera(folder_name) == expected


def test_assign_split_empty_starts_train():
    counts = {"train": 0, "val": 0, "test": 0}
    assert assign_split(counts) == "train"


def test_assign_split_targets_80_10_10():
    counts = {"train": 0, "val": 0, "test": 0}
    results = []
    for _ in range(10):
        split = assign_split(counts)
        results.append(split)
        counts[split] += 1
    assert results.count("train") == 8
    assert results.count("val") == 1
    assert results.count("test") == 1


def test_assign_split_accounts_for_existing():
    # Already have 8 train, 1 val — next should be test
    counts = {"train": 8, "val": 1, "test": 0}
    assert assign_split(counts) == "test"


def test_next_id_empty():
    assert next_id([], "wf") == 1


def test_next_id_existing():
    sequences = [
        {"id": "wf_00000001", "folder": "a", "camera": "a", "split": "train"},
        {"id": "wf_00000003", "folder": "b", "camera": "b", "split": "val"},
    ]
    assert next_id(sequences, "wf") == 4


def test_compute_new_assignments_ids_and_splits():
    folders = [
        "sdis83_2024-01-01T00-00-00",
        "sdis83_2024-01-02T00-00-00",
        "sdis83_2024-01-03T00-00-00",
        "sdis83_2024-01-04T00-00-00",
        "sdis83_2024-01-05T00-00-00",
        "sdis83_2024-01-06T00-00-00",
        "sdis83_2024-01-07T00-00-00",
        "sdis83_2024-01-08T00-00-00",
        "sdis83_2024-01-09T00-00-00",
        "sdis83_2024-01-10T00-00-00",
    ]
    result = compute_new_assignments(folders, [], start_id=1, prefix="wf")
    assert len(result) == 10
    assert result[0]["id"] == "wf_00000001"
    splits = [r["split"] for r in result]
    assert splits.count("train") == 8
    assert splits.count("val") == 1
    assert splits.count("test") == 1


def test_compute_new_assignments_stable_on_second_run():
    # 10 sequences first batch → 8 train, 1 val, 1 test
    folders_first = [f"cam1_2024-01-{i:02d}T00-00" for i in range(1, 11)]
    first = compute_new_assignments(folders_first, [], start_id=1, prefix="wf")
    first_splits = [r["split"] for r in first]
    assert first_splits.count("train") == 8
    assert first_splits.count("val") == 1
    assert first_splits.count("test") == 1

    # Second batch — existing entries must not be modified
    folders_second = [f"cam1_2024-02-{i:02d}T00-00" for i in range(1, 6)]
    second = compute_new_assignments(folders_second, first, start_id=11, prefix="wf")

    assert second[0]["id"] == "wf_00000011"
    # First batch is untouched (compute_new_assignments returns only new entries)
    assert first_splits == [r["split"] for r in first]


def test_rebalance_swaps_val_to_test():
    # Simulate 10 sequences all on same camera → 8 train, 2 val, 0 test (common edge case)
    new = [
        {"id": f"wf_{i:08d}", "folder": f"cam1_2024-01-{i:02d}T00-00", "camera": "cam1", "split": split}
        for i, split in enumerate(["train"] * 8 + ["val", "val"], 1)
    ]
    result = rebalance_minority_splits(new, [])
    splits = [r["split"] for r in result]
    assert splits.count("val") == 1
    assert splits.count("test") == 1


def test_rebalance_respects_camera_minimum():
    # Camera with only 1 val should not have it swapped away
    new = [
        {"id": "wf_00000001", "folder": "cam1_2024-01-01T00-00", "camera": "cam1", "split": "train"},
        {"id": "wf_00000002", "folder": "cam1_2024-01-02T00-00", "camera": "cam1", "split": "val"},
        # cam2 has surplus val
        {"id": "wf_00000003", "folder": "cam2_2024-01-01T00-00", "camera": "cam2", "split": "train"},
        {"id": "wf_00000004", "folder": "cam2_2024-01-02T00-00", "camera": "cam2", "split": "val"},
        {"id": "wf_00000005", "folder": "cam2_2024-01-03T00-00", "camera": "cam2", "split": "val"},
    ]
    result = rebalance_minority_splits(new, [])
    cam1_splits = [r["split"] for r in result if r["camera"] == "cam1"]
    # cam1's only val should be preserved
    assert cam1_splits.count("val") == 1


def test_compute_new_assignments_multiple_cameras():
    folders = [
        "cam1_2024-01-01T00-00",
        "cam1_2024-01-02T00-00",
        "cam2_2024-01-01T00-00",
        "cam2_2024-01-02T00-00",
    ]
    result = compute_new_assignments(folders, [], start_id=1, prefix="wf")
    by_camera = {}
    for r in result:
        by_camera.setdefault(r["camera"], []).append(r["split"])
    # Each camera should start with train
    assert by_camera["cam1"][0] == "train"
    assert by_camera["cam2"][0] == "train"
