"""
Pytest configuration file for the pyro-dataset project.
"""

import json
from pathlib import Path

from PIL import Image


def make_alert(alert_id: int, kind: str, camera: str, n_frames: int = 3) -> dict:
    """One export manifest entry with a single lane of `kind`."""
    is_smoke = kind == "smoke"
    box = {
        "xyxyn": [0.2, 0.4, 0.4, 0.6],
        "smoke_type": "wildfire" if is_smoke else None,
        "false_positive_types": None if is_smoke else ["building"],
        "origin": "human" if is_smoke else "engine",
    }
    return {
        "source_api": "pyronear_french",
        "platform_alert_id": alert_id,
        "camera_name": camera,
        "organisation_name": "sdis-91",
        "azimuth": 285,
        "recorded_at": f"2026-08-05T13:{alert_id:02d}:00.000000Z",
        "temporal_model_score": 0.99 if is_smoke else 0.9,
        "temporal_model_version": "0.2.0",
        "objects": [
            {
                "sequence_id": alert_id * 10,
                "record_kind": "smoke" if is_smoke else "false_positive",
                "smoke_types": ["wildfire"] if is_smoke else [],
                "false_positive_types": [] if is_smoke else ["building"],
                "frames": [
                    {
                        "detection_id": alert_id * 100 + i,
                        "recorded_at": f"2026-08-05T13:{alert_id:02d}:{i:02d}.000000Z",
                        "bucket_key": f"detections/sequence_{alert_id}/{i}.jpg",
                        "image_path": f"images/pyronear_french/{alert_id}/{i}.jpg",
                        "boxes": [box],
                    }
                    for i in range(n_frames)
                ],
            }
        ],
    }


def write_export(root: Path, alerts: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "manifest.jsonl").open("w") as fh:
        for alert in alerts:
            fh.write(json.dumps(alert) + "\n")
            for obj in alert["objects"]:
                for frame in obj["frames"]:
                    path = root / frame["image_path"]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (32, 18), "black").save(path)


def small_export(tmp_path: Path) -> Path:
    """One smoke alert and one false positive — the smallest useful export."""
    export = tmp_path / "export"
    write_export(
        export,
        [make_alert(1, "smoke", "cam-a"), make_alert(10, "fp", "cam-b")],
    )
    return export
