import json
import random

from pyro_dataset.annotator.recurring import Ledger, assign_new_split, main_bbox

ROAD_BOX = {
    "xyxyn": [0.60, 0.46, 0.62, 0.49],
    "smoke_type": None,
    "false_positive_types": ["road"],
    "origin": "engine",
}


def alert_with_boxes(boxes):
    return {
        "objects": [
            {
                "record_kind": "false_positive",
                "frames": [
                    {"detection_id": i, "boxes": [box]} for i, box in enumerate(boxes)
                ],
            }
        ]
    }


def test_main_bbox_returns_none_without_boxes():
    assert main_bbox(alert_with_boxes([])) is None


def test_main_bbox_returns_the_dominant_box():
    assert main_bbox(alert_with_boxes([ROAD_BOX] * 3)) == (0.60, 0.46, 0.62, 0.49)


def test_main_bbox_ignores_degenerate_boxes():
    flat = {**ROAD_BOX, "xyxyn": [0.6, 0.4, 0.6, 0.5]}
    assert main_bbox(alert_with_boxes([flat])) is None


def test_mint_then_match_returns_the_same_id():
    ledger = Ledger()
    ro_id = ledger.mint("cam-01", 285, (0.60, 0.46, 0.62, 0.49), "train", "src:1")
    assert ro_id == "ro_00001"
    assert ledger.match("cam-01", 285, (0.601, 0.461, 0.621, 0.491), 0.3) == ro_id


def test_match_is_scoped_to_camera_and_azimuth():
    ledger = Ledger()
    ledger.mint("cam-01", 285, (0.60, 0.46, 0.62, 0.49), "train", "src:1")
    assert ledger.match("cam-02", 285, (0.60, 0.46, 0.62, 0.49), 0.3) is None
    assert ledger.match("cam-01", 0, (0.60, 0.46, 0.62, 0.49), 0.3) is None


def test_match_returns_none_below_threshold():
    ledger = Ledger()
    ledger.mint("cam-01", 285, (0.10, 0.10, 0.20, 0.20), "train", "src:1")
    assert ledger.match("cam-01", 285, (0.80, 0.80, 0.90, 0.90), 0.3) is None


def test_match_picks_the_best_overlap_when_several_qualify():
    ledger = Ledger()
    ledger.mint("cam-01", 285, (0.10, 0.10, 0.30, 0.30), "train", "src:1")
    near = ledger.mint("cam-01", 285, (0.20, 0.20, 0.40, 0.40), "train", "src:2")
    assert ledger.match("cam-01", 285, (0.21, 0.21, 0.41, 0.41), 0.1) == near


def test_a_matched_object_keeps_its_original_split():
    """The point of the ledger: an artefact cannot change split."""
    ledger = Ledger()
    ro_id = ledger.mint("cam-01", 285, (0.60, 0.46, 0.62, 0.49), "val", "src:1")
    ledger.record_sighting(ro_id, (0.61, 0.47, 0.63, 0.50), "src:2")
    assert ledger.entries[ro_id]["split"] == "val"
    assert ledger.seen(ro_id) == 2


def test_seen_and_ingested_are_independent_counters():
    ledger = Ledger()
    ro_id = ledger.mint("cam-01", 285, (0.60, 0.46, 0.62, 0.49), "train", "src:1")
    ledger.record_sighting(ro_id, (0.60, 0.46, 0.62, 0.49), "src:2")
    ledger.record_ingested(ro_id, "folder-a")
    assert ledger.seen(ro_id) == 2
    assert ledger.ingested(ro_id) == ["folder-a"]


def test_record_sighting_never_moves_the_anchor():
    """Regression: the anchor used to follow each new sighting, so matching
    depended on the order and history of the walk. Replaying one export against
    its own ledger then re-assigned alerts and minted duplicates — each drawing
    a fresh split, which is the leakage the ledger exists to prevent."""
    ledger = Ledger()
    ro_id = ledger.mint("cam-01", 285, (0.60, 0.46, 0.62, 0.49), "train", "src:1")
    ledger.record_sighting(ro_id, (0.61, 0.47, 0.63, 0.50), "src:2")
    assert ledger.entries[ro_id]["bbox_xyxyn"] == [0.60, 0.46, 0.62, 0.49]


def test_matching_is_idempotent_across_a_replay():
    """The same sightings replayed against the resulting ledger must all match
    their original object, minting nothing."""
    ledger = Ledger()
    walk = [
        (0.60, 0.46, 0.62, 0.49),
        (0.61, 0.47, 0.63, 0.50),
        (0.62, 0.48, 0.64, 0.51),
        (0.63, 0.49, 0.65, 0.52),
    ]
    for i, bbox in enumerate(walk):
        ro_id = ledger.match("cam-01", 285, bbox, 0.3)
        if ro_id is None:
            ledger.mint("cam-01", 285, bbox, "train", f"src:{i}")
        else:
            ledger.record_sighting(ro_id, bbox, f"src:{i}")
    minted_first_pass = len(ledger.entries)

    for i, bbox in enumerate(walk):
        assert ledger.match("cam-01", 285, bbox, 0.3) is not None, (
            f"sighting {i} lost its object on replay"
        )
    assert len(ledger.entries) == minted_first_pass


def test_main_bbox_uses_the_dominant_lane_not_a_cross_lane_median():
    """Two artefacts at opposite corners: the median across both lands between
    them, describing a box that exists in neither lane."""
    left = {**ROAD_BOX, "xyxyn": [0.10, 0.10, 0.20, 0.20]}
    right = {**ROAD_BOX, "xyxyn": [0.80, 0.80, 0.90, 0.90]}
    alert = {
        "objects": [
            {
                "record_kind": "false_positive",
                "frames": [{"detection_id": i, "boxes": [left]} for i in range(3)],
            },
            {
                "record_kind": "false_positive",
                "frames": [{"detection_id": 9, "boxes": [right]}],
            },
        ]
    }
    assert main_bbox(alert) == (0.10, 0.10, 0.20, 0.20)


def test_round_trips_through_disk(tmp_path):
    ledger = Ledger()
    ro_id = ledger.mint("cam-01", 285, (0.60, 0.46, 0.62, 0.49), "train", "src:1")
    path = tmp_path / "recurring_objects.json"
    ledger.save(path)
    reloaded = Ledger.load(path)
    assert reloaded.entries[ro_id]["camera"] == "cam-01"
    assert (
        reloaded.mint("cam-02", 0, (0.1, 0.1, 0.2, 0.2), "train", "src:9") == "ro_00002"
    )


def test_seen_counts_distinct_alerts_not_repeat_sightings():
    """The export is a full re-pull, so the same alert is matched on every run.
    Counting instead of identifying would inflate seen by one whole history per
    import and bias the ranking against genuinely new artefacts."""
    ledger = Ledger()
    ro_id = ledger.mint("cam-01", 285, (0.60, 0.46, 0.62, 0.49), "train", "src:1")
    for _ in range(3):
        ledger.record_sighting(ro_id, (0.60, 0.46, 0.62, 0.49), "src:1")
    assert ledger.seen(ro_id) == 1
    ledger.record_sighting(ro_id, (0.60, 0.46, 0.62, 0.49), "src:2")
    assert ledger.seen(ro_id) == 2


def test_record_ingested_is_idempotent_per_folder():
    ledger = Ledger()
    ro_id = ledger.mint("cam-01", 285, (0.60, 0.46, 0.62, 0.49), "train", "src:1")
    ledger.record_ingested(ro_id, "folder-a")
    ledger.record_ingested(ro_id, "folder-a")
    ledger.record_ingested(ro_id, "folder-b")
    assert ledger.ingested(ro_id) == ["folder-a", "folder-b"]


def test_main_bbox_is_the_median_not_the_first_box():
    """NMS cannot discriminate without confidences, so it returned whichever
    box came first in the manifest — an arbitrary first-frame box."""
    outlier = {**ROAD_BOX, "xyxyn": [0.10, 0.10, 0.12, 0.13]}
    steady = {**ROAD_BOX, "xyxyn": [0.60, 0.46, 0.62, 0.49]}
    assert main_bbox(alert_with_boxes([outlier, steady, steady])) == (
        0.60,
        0.46,
        0.62,
        0.49,
    )


def test_load_of_a_missing_file_is_empty(tmp_path):
    assert Ledger.load(tmp_path / "nope.json").entries == {}


def test_assign_new_split_is_ninety_ten_train_val():
    rng = random.Random(0)
    counts = {"train": 0, "val": 0}
    splits = []
    for _ in range(10):
        split = assign_new_split(rng, counts)
        splits.append(split)
        counts[split] += 1
    assert splits.count("train") == 9
    assert splits.count("val") == 1
    assert "test" not in splits


def test_saved_ledger_is_valid_json(tmp_path):
    ledger = Ledger()
    ledger.mint("cam-01", 285, (0.60, 0.46, 0.62, 0.49), "train", "src:1")
    path = tmp_path / "recurring_objects.json"
    ledger.save(path)
    assert json.loads(path.read_text())["ro_00001"]["ingested_folders"] == []
