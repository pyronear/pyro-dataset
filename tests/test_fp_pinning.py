"""Both dataset builds must include annotator-sourced FP sequences.

Adding a sequence to data/raw/fp does not put it in a built dataset: the
builds pick `quota` representatives by clustering the whole pool and may
simply not choose it. Sequences the importer selected deliberately are
pinned, and the helpers implementing that live in pyro_dataset.fp.selection
so the YOLO build, the sequential build and the embedding script share them.
Identity for the pinned cohort comes from the recurring-object ledger.
"""

import pytest

from pyro_dataset.fp.selection import (
    folder_to_recurring_object,
    partition_pinned,
    remaining_quota,
)


def test_partition_pinned_splits_on_source():
    pinned, rest = partition_pinned(
        [
            {"folder": "a", "source": "pyro-annotator"},
            {"folder": "b"},
            {"folder": "c", "source": "platform"},
        ]
    )
    assert [s["folder"] for s in pinned] == ["a"]
    assert [s["folder"] for s in rest] == ["b", "c"]


def test_partition_pinned_handles_an_empty_pool():
    assert partition_pinned([]) == ([], [])


def test_pinned_sequences_consume_quota_before_clustering():
    pinned, _ = partition_pinned(
        [{"folder": f"a{i}", "source": "pyro-annotator"} for i in range(4)]
    )
    assert remaining_quota(quota=10, pinned=pinned) == 6


def test_remaining_quota_never_goes_negative():
    pinned, _ = partition_pinned(
        [{"folder": f"a{i}", "source": "pyro-annotator"} for i in range(12)]
    )
    assert remaining_quota(quota=10, pinned=pinned) == 0


def test_remaining_quota_is_untouched_without_pinned_sequences():
    assert remaining_quota(quota=10, pinned=[]) == 10


def test_folder_to_recurring_object_inverts_ingested_folders():
    ledger = {
        "ro_00001": {"ingested_folders": ["cam-a_1"]},
        "ro_00002": {"ingested_folders": ["cam-b_1", "cam-b_2"]},
    }
    assert folder_to_recurring_object(ledger, {"cam-a_1", "cam-b_2"}) == {
        "cam-a_1": "ro_00001",
        "cam-b_2": "ro_00002",
    }


def test_folder_missing_from_ledger_is_a_hard_error():
    ledger = {"ro_00001": {"ingested_folders": ["cam-a_1"]}}
    with pytest.raises(ValueError, match="cam-x_9"):
        folder_to_recurring_object(ledger, {"cam-x_9"})


def test_folder_under_two_objects_is_a_hard_error():
    ledger = {
        "ro_00001": {"ingested_folders": ["dup_1"]},
        "ro_00002": {"ingested_folders": ["dup_1"]},
    }
    with pytest.raises(ValueError, match="dup_1"):
        folder_to_recurring_object(ledger, {"dup_1"})


def test_unrequested_ledger_folders_are_ignored():
    ledger = {"ro_00001": {"ingested_folders": ["staged-never-registered", "cam-a_1"]}}
    assert folder_to_recurring_object(ledger, {"cam-a_1"}) == {"cam-a_1": "ro_00001"}
