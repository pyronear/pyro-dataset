from pyro_dataset.annotator.same_fire import group_new_smoke, view_and_time


def folder(camera: str, ts: str) -> str:
    return f"sdis-40_{camera}_61_{ts}"


def test_view_and_time_splits_on_the_last_underscore():
    view, at = view_and_time(folder("laluque-02", "2026-07-06T04-28-05"))
    assert view == "sdis-40_laluque-02_61"
    assert (at.year, at.hour, at.second) == (2026, 4, 5)


def test_two_close_alerts_on_one_view_form_one_group():
    a = folder("laluque-02", "2026-07-06T04-28-05")
    b = folder("laluque-02", "2026-07-06T06-55-00")
    groups = group_new_smoke([a, b], planned={})
    assert groups == [([a, b], None)]


def test_a_gap_beyond_the_window_splits_the_chain():
    a = folder("laluque-02", "2026-07-06T04-00-00")
    b = folder("laluque-02", "2026-07-07T05-00-00")
    groups = group_new_smoke([a, b], planned={}, window_hours=12.0)
    assert groups == [([a], None), ([b], None)]


def test_chaining_is_transitive():
    a = folder("laluque-02", "2026-07-06T00-00-00")
    b = folder("laluque-02", "2026-07-06T10-00-00")
    c = folder("laluque-02", "2026-07-06T20-00-00")
    # a-c gap is 20h > window, but b bridges them.
    groups = group_new_smoke([a, b, c], planned={}, window_hours=12.0)
    assert groups == [([a, b, c], None)]


def test_a_chain_inherits_the_earliest_planned_split():
    planned_early = folder("laluque-02", "2026-07-06T01-00-00")
    planned_late = folder("laluque-02", "2026-07-06T02-00-00")
    new = folder("laluque-02", "2026-07-06T03-00-00")
    groups = group_new_smoke(
        [new], planned={planned_early: "val", planned_late: "train"}
    )
    assert groups == [([new], "val")]


def test_planned_folders_can_bridge_new_ones():
    a = folder("laluque-02", "2026-07-06T00-00-00")
    planned = folder("laluque-02", "2026-07-06T10-00-00")
    b = folder("laluque-02", "2026-07-06T20-00-00")
    groups = group_new_smoke([a, b], planned={planned: "train"}, window_hours=12.0)
    assert groups == [([a, b], "train")]


def test_different_views_never_group():
    a = folder("laluque-02", "2026-07-06T04-00-00")
    b = folder("laluque-03", "2026-07-06T04-05-00")
    groups = group_new_smoke([a, b], planned={})
    assert sorted(g for g, _ in groups) == [[a], [b]]


def test_chains_with_no_new_folder_are_dropped():
    planned = folder("laluque-02", "2026-07-06T04-00-00")
    assert group_new_smoke([], planned={planned: "train"}) == []
