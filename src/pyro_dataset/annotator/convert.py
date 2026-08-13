"""Turn one alert from a pyro-annotator export manifest into the pieces of a
sequence folder: its name, its per-frame file stems, its YOLO labels and its
provenance sidecar.

Pure functions — the caller owns all I/O.
"""

from typing import Any

from pyro_dataset.constants import CLASS_ID_FALSE_POSITIVE_PROPOSAL, CLASS_ID_SMOKE

UNKNOWN_AZIMUTH = 999


def camera_key(alert: dict[str, Any]) -> str:
    """`{organisation}_{camera}_{azimuth}` — the dataset's camera identity.

    Azimuth falls back to 999, the convention the existing pools use for
    unknown. Neither organisation nor camera may contain an underscore, or the
    result stops matching ingest's folder regex.
    """
    azimuth = alert.get("azimuth")
    azimuth = UNKNOWN_AZIMUTH if azimuth is None else int(azimuth)
    return f"{alert['organisation_name']}_{alert['camera_name']}_{azimuth}"


def _timestamp(iso: str) -> str:
    """`2026-08-05T13:46:08.069000Z` -> `2026-08-05T13-46-08`."""
    return iso[:19].replace(":", "-")


def folder_name(alert: dict[str, Any]) -> str:
    """Sequence folder name — also the alert's identity across re-imports."""
    return f"{camera_key(alert)}_{_timestamp(alert['recorded_at'])}"


def frame_stem(alert: dict[str, Any], frame: dict[str, Any]) -> str:
    """Image/label stem for one frame: camera key plus the frame's own time."""
    return f"{camera_key(alert)}_{_timestamp(frame['recorded_at'])}"


def alert_kind(alert: dict[str, Any]) -> str:
    """`wildfire` if any lane is smoke, else `fp`."""
    if any(obj["record_kind"] == "smoke" for obj in alert["objects"]):
        return "wildfire"
    return "fp"


def label_lines(alert: dict[str, Any], frame: dict[str, Any]) -> list[str]:
    """YOLO `class cx cy w h` lines for one capture, in lane order.

    `frame` identifies the capture; only its timestamp is read. Lanes are
    matched on that timestamp rather than on `detection_id` because sibling
    lanes hold their own copies of a capture under distinct detection ids —
    keying on the id would split one capture's boxes across several label
    files, and a multi-plume alert would lose every plume but one.

    Only the boxes of the lane kind that decided the folder are written: a
    wildfire alert contributes smoke boxes as class 0, an fp alert its
    proposal boxes as class 99. Boxes inside a smoke lane that the annotator
    flagged as false positives are excluded too — build_tubes keeps a single
    class-blind tube per sequence, so a persistent distractor would outlast a
    late-appearing plume and become the positive's evidence.
    """
    kind = alert_kind(alert)
    capture = _timestamp(frame["recorded_at"])
    lines: list[str] = []
    for obj in alert["objects"]:
        lane_is_smoke = obj["record_kind"] == "smoke"
        if kind == "wildfire" and not lane_is_smoke:
            continue
        class_id = CLASS_ID_SMOKE if lane_is_smoke else CLASS_ID_FALSE_POSITIVE_PROPOSAL
        # At most one frame per lane per capture. Stems are second-resolution
        # and a lane can hold two detections inside one second; only the first
        # is written as an image, so taking both here would describe a picture
        # that was never kept.
        matched = [
            lane_frame
            for lane_frame in obj["frames"]
            if _timestamp(lane_frame["recorded_at"]) == capture
        ][:1]
        for lane_frame in matched:
            for box in lane_frame["boxes"]:
                if lane_is_smoke and box.get("smoke_type") is None:
                    continue
                x1, y1, x2, y2 = box["xyxyn"]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                width, height = x2 - x1, y2 - y1
                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}")
    return lines


def build_meta(
    alert: dict[str, Any], recurring_object_id: str | None
) -> dict[str, Any]:
    """Provenance plus every lane's full track.

    Includes the lanes and boxes `label_lines` excluded, which is what makes
    the import lossless: an object-level dataset can be derived later without
    re-exporting or re-annotating.
    """
    return {
        "source_api": alert["source_api"],
        "platform_alert_id": alert["platform_alert_id"],
        "recurring_object": recurring_object_id,
        "temporal_model_score": alert.get("temporal_model_score"),
        "temporal_model_version": alert.get("temporal_model_version"),
        "kind": alert_kind(alert),
        "lanes": [
            {
                "sequence_id": obj["sequence_id"],
                "kind": obj["record_kind"],
                "smoke_types": obj["smoke_types"],
                "false_positive_types": obj["false_positive_types"],
                "track": [
                    {
                        "frame": frame_stem(alert, lane_frame),
                        "detection_id": lane_frame["detection_id"],
                        "boxes": lane_frame["boxes"],
                    }
                    for lane_frame in obj["frames"]
                ],
            }
            for obj in alert["objects"]
        ],
    }
