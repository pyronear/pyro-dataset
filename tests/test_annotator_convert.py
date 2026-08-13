from pyro_dataset.annotator.convert import (
    alert_kind,
    build_meta,
    camera_key,
    folder_name,
    frame_stem,
    label_lines,
)

ALERT = {
    "source_api": "pyronear_french",
    "platform_alert_id": 57057,
    "camera_name": "chateau-eau-milly-la-foret-02",
    "organisation_name": "sdis-91",
    "azimuth": 285,
    "recorded_at": "2026-08-05T13:46:08.069000Z",
    "temporal_model_score": 0.994,
    "temporal_model_version": "0.2.0",
    "objects": [
        {
            "sequence_id": 7236,
            "record_kind": "smoke",
            "smoke_types": ["industrial"],
            "false_positive_types": [],
            "frames": [
                {
                    "detection_id": 116823,
                    "recorded_at": "2026-08-05T13:46:08.069000Z",
                    "bucket_key": "detections/sequence_7236/x.jpg",
                    "image_path": "images/pyronear_french/57057/116823.jpg",
                    "boxes": [
                        {
                            "xyxyn": [0.2, 0.4, 0.4, 0.6],
                            "smoke_type": "industrial",
                            "false_positive_types": None,
                            "origin": "human",
                        }
                    ],
                }
            ],
        }
    ],
}


def test_camera_key_joins_org_camera_azimuth():
    assert camera_key(ALERT) == "sdis-91_chateau-eau-milly-la-foret-02_285"


def test_camera_key_uses_999_when_azimuth_missing():
    assert camera_key({**ALERT, "azimuth": None}).endswith("_999")


def test_folder_name_appends_the_alert_timestamp():
    assert (
        folder_name(ALERT)
        == "sdis-91_chateau-eau-milly-la-foret-02_285_2026-08-05T13-46-08"
    )


def test_frame_stem_uses_the_frame_timestamp_not_the_alert_one():
    frame = {**ALERT["objects"][0]["frames"][0], "recorded_at": "2026-08-05T14:01:02Z"}
    assert frame_stem(ALERT, frame) == f"{camera_key(ALERT)}_2026-08-05T14-01-02"


def test_alert_kind_is_wildfire_when_any_lane_is_smoke():
    assert alert_kind(ALERT) == "wildfire"


def test_alert_kind_is_fp_when_no_lane_is_smoke():
    fp = {
        **ALERT,
        "objects": [{**ALERT["objects"][0], "record_kind": "false_positive"}],
    }
    assert alert_kind(fp) == "fp"


def test_label_lines_convert_xyxyn_to_xywhn_with_class_zero():
    frame = ALERT["objects"][0]["frames"][0]
    assert label_lines(ALERT, frame) == ["0 0.300000 0.500000 0.200000 0.200000"]


def test_label_lines_use_the_sentinel_class_for_fp_lanes():
    fp_alert = {
        **ALERT,
        "objects": [
            {
                **ALERT["objects"][0],
                "record_kind": "false_positive",
                "frames": [
                    {
                        **ALERT["objects"][0]["frames"][0],
                        "boxes": [
                            {
                                "xyxyn": [0.2, 0.4, 0.4, 0.6],
                                "smoke_type": None,
                                "false_positive_types": ["building"],
                                "origin": "engine",
                            }
                        ],
                    }
                ],
            }
        ],
    }
    frame = fp_alert["objects"][0]["frames"][0]
    assert label_lines(fp_alert, frame) == ["99 0.300000 0.500000 0.200000 0.200000"]


def test_label_lines_exclude_boxes_of_the_other_lane_kind():
    """A wildfire alert must not write its FP lane's boxes.

    build_tubes keeps the longest tube regardless of class, so a persistent FP
    track would become the positive sequence's evidence.
    """
    mixed = {
        **ALERT,
        "objects": [
            ALERT["objects"][0],
            {
                "sequence_id": 9999,
                "record_kind": "false_positive",
                "smoke_types": [],
                "false_positive_types": ["building"],
                "frames": [
                    {
                        **ALERT["objects"][0]["frames"][0],
                        "boxes": [
                            {
                                "xyxyn": [0.8, 0.8, 0.9, 0.9],
                                "smoke_type": None,
                                "false_positive_types": ["building"],
                                "origin": "engine",
                            }
                        ],
                    }
                ],
            },
        ],
    }
    frame = mixed["objects"][0]["frames"][0]
    assert label_lines(mixed, frame) == ["0 0.300000 0.500000 0.200000 0.200000"]


def _alert_with_boxes(boxes):
    """ALERT with its single lane's single frame carrying `boxes`."""
    lane = ALERT["objects"][0]
    frame = {**lane["frames"][0], "boxes": boxes}
    return {**ALERT, "objects": [{**lane, "frames": [frame]}]}, frame


def test_label_lines_drop_fp_flagged_boxes_inside_a_smoke_lane():
    alert, frame = _alert_with_boxes(
        [
            {
                "xyxyn": [0.2, 0.4, 0.4, 0.6],
                "smoke_type": None,
                "false_positive_types": ["lens_flare"],
                "origin": "human",
            }
        ]
    )
    assert label_lines(alert, frame) == []


def test_label_lines_are_empty_for_a_gap_frame():
    alert, frame = _alert_with_boxes([])
    assert label_lines(alert, frame) == []


def test_label_lines_union_sibling_lanes_at_one_capture():
    """Sibling lanes carry distinct detection ids for the same capture, so the
    capture's label file must gather every lane's box."""
    lane = ALERT["objects"][0]
    second = {
        **lane,
        "sequence_id": 7237,
        "frames": [
            {
                **lane["frames"][0],
                "detection_id": 116824,
                "boxes": [
                    {
                        "xyxyn": [0.6, 0.6, 0.8, 0.8],
                        "smoke_type": "wildfire",
                        "false_positive_types": None,
                        "origin": "human",
                    }
                ],
            }
        ],
    }
    alert = {**ALERT, "objects": [lane, second]}
    assert label_lines(alert, lane["frames"][0]) == [
        "0 0.300000 0.500000 0.200000 0.200000",
        "0 0.700000 0.700000 0.200000 0.200000",
    ]


def test_build_meta_keeps_every_lane_including_excluded_ones():
    mixed = {
        **ALERT,
        "objects": [
            ALERT["objects"][0],
            {
                "sequence_id": 9999,
                "record_kind": "false_positive",
                "smoke_types": [],
                "false_positive_types": ["building"],
                "frames": ALERT["objects"][0]["frames"],
            },
        ],
    }
    meta = build_meta(mixed, "ro_00042")
    assert meta["platform_alert_id"] == 57057
    assert meta["recurring_object"] == "ro_00042"
    assert meta["temporal_model_score"] == 0.994
    assert [lane["kind"] for lane in meta["lanes"]] == ["smoke", "false_positive"]
    assert meta["lanes"][1]["track"][0]["boxes"], "excluded lane keeps its boxes"
