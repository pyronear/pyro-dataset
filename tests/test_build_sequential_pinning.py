"""The sequential build must include annotator-sourced FP sequences.

Adding a sequence to data/raw/fp does not put it in the built dataset: the
build picks `quota` representatives by clustering the whole pool and may
simply not choose it. Sequences the importer selected deliberately are pinned.
"""

import importlib.util
from pathlib import Path

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
