from pyro_dataset.annotator.select import rank_objects, select_fp


def test_fooling_objects_rank_above_frequent_ones():
    ranked = rank_objects(
        {
            "ro_00001": {"score": 0.1, "seen": 57},  # very frequent, model is right
            "ro_00002": {"score": 0.9, "seen": 2},  # rare, still fools the model
        },
        threshold=0.5,
    )
    assert ranked == ["ro_00002", "ro_00001"]


def test_within_a_bucket_frequency_decides():
    ranked = rank_objects(
        {
            "ro_00001": {"score": 0.9, "seen": 3},
            "ro_00002": {"score": 0.8, "seen": 30},
        },
        threshold=0.5,
    )
    assert ranked == ["ro_00002", "ro_00001"]


def test_unscored_objects_fall_into_the_second_bucket():
    ranked = rank_objects(
        {
            "ro_00001": {"score": None, "seen": 100},
            "ro_00002": {"score": 0.6, "seen": 1},
        },
        threshold=0.5,
    )
    assert ranked == ["ro_00002", "ro_00001"]


def test_ranking_is_frequency_only_when_nothing_is_scored():
    ranked = rank_objects(
        {
            "ro_00001": {"score": None, "seen": 2},
            "ro_00002": {"score": None, "seen": 9},
        },
        threshold=0.5,
    )
    assert ranked == ["ro_00002", "ro_00001"]


def test_ranking_is_stable_for_equal_objects():
    ranked = rank_objects(
        {
            "ro_00002": {"score": 0.9, "seen": 5},
            "ro_00001": {"score": 0.9, "seen": 5},
        },
        threshold=0.5,
    )
    assert ranked == ["ro_00001", "ro_00002"]


def test_select_takes_one_alert_per_object_up_to_the_quota():
    picked = select_fp(
        ranked=["ro_1", "ro_2", "ro_3"],
        per_object_alerts={
            "ro_1": [{"id": "a", "folder": "f-a"}, {"id": "b", "folder": "f-b"}],
            "ro_2": [{"id": "c", "folder": "f-c"}],
            "ro_3": [{"id": "d", "folder": "f-d"}],
        },
        quota=2,
        max_per_object=1,
        ingested={},
    )
    assert [p["id"] for p in picked] == ["a", "c"]
    assert [p["recurring_object"] for p in picked] == ["ro_1", "ro_2"]


def test_max_per_object_allows_more_than_one():
    picked = select_fp(
        ranked=["ro_1", "ro_2"],
        per_object_alerts={
            "ro_1": [{"id": "a", "folder": "f-a"}, {"id": "b", "folder": "f-b"}],
            "ro_2": [{"id": "c", "folder": "f-c"}],
        },
        quota=3,
        max_per_object=2,
        ingested={},
    )
    assert [p["id"] for p in picked] == ["a", "c", "b"]


def test_the_cap_counts_sequences_from_earlier_imports():
    """Two of three slots are spent, so exactly one more may be taken — and it
    must be an alert not already staged, not a re-pick of one that is."""
    picked = select_fp(
        ranked=["ro_1"],
        per_object_alerts={
            "ro_1": [
                {"id": "a", "folder": "f-a"},
                {"id": "b", "folder": "f-b"},
                {"id": "c", "folder": "f-c"},
            ]
        },
        quota=5,
        max_per_object=3,
        ingested={"ro_1": ["f-a", "f-b"]},
    )
    assert [p["id"] for p in picked] == ["c"]


def test_an_object_below_its_cap_reaches_it_on_a_later_run():
    """Regression: selection used to restart at index 0 and re-pick the alert
    it had already staged. That pick was dropped as an existing folder, burning
    a quota slot, and the object could never reach its cap."""
    picked = select_fp(
        ranked=["ro_1"],
        per_object_alerts={
            "ro_1": [{"id": "a", "folder": "f-a"}, {"id": "b", "folder": "f-b"}]
        },
        quota=2,
        max_per_object=2,
        ingested={"ro_1": ["f-a"]},
    )
    assert [p["id"] for p in picked] == ["b"]


def test_an_object_at_its_cap_contributes_nothing():
    picked = select_fp(
        ranked=["ro_1", "ro_2"],
        per_object_alerts={
            "ro_1": [{"id": "a", "folder": "f-a"}],
            "ro_2": [{"id": "b", "folder": "f-b"}],
        },
        quota=5,
        max_per_object=1,
        ingested={"ro_1": ["f-a"]},
    )
    assert [p["id"] for p in picked] == ["b"]


def test_selection_stops_when_material_runs_out():
    picked = select_fp(
        ranked=["ro_1"],
        per_object_alerts={"ro_1": [{"id": "a", "folder": "f-a"}]},
        quota=10,
        max_per_object=1,
        ingested={},
    )
    assert len(picked) == 1


def test_selection_does_not_mutate_the_caller_s_alerts():
    alerts = {"ro_1": [{"id": "a", "folder": "f-a"}]}
    select_fp(
        ranked=["ro_1"],
        per_object_alerts=alerts,
        quota=1,
        max_per_object=1,
        ingested={},
    )
    assert alerts == {"ro_1": [{"id": "a", "folder": "f-a"}]}
