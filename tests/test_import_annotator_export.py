import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

SCRIPT = Path("scripts/import_annotator_export.py")


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


def run_import(export: Path, out: Path, ledger: Path, extra: list[str] | None = None):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--export-dir",
            str(export),
            "--output-dir",
            str(out),
            "--ledger",
            str(ledger),
            *(extra or []),
        ],
        capture_output=True,
        text=True,
    )


def small_export(tmp_path: Path) -> Path:
    export = tmp_path / "export"
    write_export(
        export,
        [make_alert(1, "smoke", "cam-a"), make_alert(10, "fp", "cam-b")],
    )
    return export


def test_import_balances_fp_against_the_smoke_count(tmp_path):
    export = tmp_path / "export"
    write_export(
        export,
        [make_alert(1, "smoke", "cam-a"), make_alert(2, "smoke", "cam-b")]
        + [make_alert(10 + i, "fp", f"cam-fp{i}") for i in range(5)],
    )
    out = tmp_path / "staging"
    result = run_import(export, out, tmp_path / "ledger.json")
    assert result.returncode == 0, result.stderr

    assert len(list((out / "wildfire").iterdir())) == 2
    assert len(list((out / "fp").iterdir())) == 2, "FP quota equals the smoke count"


def test_each_folder_has_images_labels_and_meta(tmp_path):
    out = tmp_path / "staging"
    run_import(small_export(tmp_path), out, tmp_path / "ledger.json")

    folder = next((out / "wildfire").iterdir())
    assert len(list((folder / "images").glob("*.jpg"))) == 3
    assert len(list((folder / "labels").glob("*.txt"))) == 3
    meta = json.loads((folder / "meta.json").read_text())
    assert meta["platform_alert_id"] == 1
    assert meta["lanes"][0]["track"], "meta.json carries the full track"


def test_folder_and_frame_names_satisfy_ingest_validation(tmp_path):
    from pyro_dataset.ingest import validate_sequence_folder

    out = tmp_path / "staging"
    run_import(small_export(tmp_path), out, tmp_path / "ledger.json")
    for kind in ("wildfire", "fp"):
        for folder in (out / kind).iterdir():
            result = validate_sequence_folder(folder)
            assert result.is_valid, (
                folder.name,
                result.naming_issues,
                result.structural_issues,
            )


def test_fp_labels_use_the_sentinel_class(tmp_path):
    out = tmp_path / "staging"
    run_import(small_export(tmp_path), out, tmp_path / "ledger.json")

    folder = next((out / "fp").iterdir())
    line = next((folder / "labels").glob("*.txt")).read_text().strip()
    assert line.startswith("99 ")


def test_splits_never_assign_test(tmp_path):
    export = tmp_path / "export"
    write_export(
        export,
        [make_alert(i, "smoke", f"cam-s{i}") for i in range(1, 6)]
        + [make_alert(20 + i, "fp", f"cam-f{i}") for i in range(6)],
    )
    out = tmp_path / "staging"
    run_import(export, out, tmp_path / "ledger.json")
    splits = json.loads((out / "splits.json").read_text())
    assert splits
    assert set(splits.values()) <= {"train", "val"}


def test_every_folder_has_a_split(tmp_path):
    out = tmp_path / "staging"
    run_import(small_export(tmp_path), out, tmp_path / "ledger.json")
    splits = json.loads((out / "splits.json").read_text())
    folders = {p.name for kind in ("wildfire", "fp") for p in (out / kind).iterdir()}
    assert folders == set(splits)


def test_rerun_is_idempotent_and_the_ledger_keeps_splits(tmp_path):
    export = small_export(tmp_path)
    out = tmp_path / "staging"
    ledger_path = tmp_path / "ledger.json"

    run_import(export, out, ledger_path)
    first_splits = json.loads((out / "splits.json").read_text())
    first_ledger = json.loads(ledger_path.read_text())

    run_import(export, out, ledger_path)
    assert json.loads((out / "splits.json").read_text()) == first_splits
    second_ledger = json.loads(ledger_path.read_text())
    assert set(second_ledger) == set(first_ledger)
    assert {k: v["split"] for k, v in second_ledger.items()} == {
        k: v["split"] for k, v in first_ledger.items()
    }


def test_force_train_moves_an_unused_object(tmp_path):
    export = small_export(tmp_path)
    out = tmp_path / "staging"
    ledger_path = tmp_path / "ledger.json"
    run_import(export, out, ledger_path)

    ledger = json.loads(ledger_path.read_text())
    ro_id = next(iter(ledger))
    ledger[ro_id]["split"] = "val"
    ledger[ro_id]["ingested"] = 0
    ledger_path.write_text(json.dumps(ledger))

    result = run_import(export, out, ledger_path, ["--force-train", ro_id])
    assert result.returncode == 0, result.stderr
    assert json.loads(ledger_path.read_text())[ro_id]["split"] == "train"


def test_force_train_refuses_an_object_already_ingested_elsewhere(tmp_path):
    export = small_export(tmp_path)
    out = tmp_path / "staging"
    ledger_path = tmp_path / "ledger.json"
    run_import(export, out, ledger_path)

    ledger = json.loads(ledger_path.read_text())
    ro_id = next(iter(ledger))
    ledger[ro_id]["split"] = "val"
    ledger[ro_id]["ingested"] = 1
    ledger_path.write_text(json.dumps(ledger))

    result = run_import(export, out, ledger_path, ["--force-train", ro_id])
    assert result.returncode != 0
    assert "refused" in result.stderr


def test_force_train_rejects_an_unknown_object(tmp_path):
    export = small_export(tmp_path)
    out = tmp_path / "staging"
    ledger_path = tmp_path / "ledger.json"
    run_import(export, out, ledger_path)

    result = run_import(export, out, ledger_path, ["--force-train", "ro_99999"])
    assert result.returncode != 0
    assert "unknown" in result.stderr


def test_dry_run_writes_nothing(tmp_path):
    export = small_export(tmp_path)
    out = tmp_path / "staging"
    ledger_path = tmp_path / "ledger.json"
    result = run_import(export, out, ledger_path, ["--dry-run"])
    assert result.returncode == 0, result.stderr
    assert not out.exists()
    assert not ledger_path.exists()


def test_an_alert_without_boxes_is_not_a_candidate(tmp_path):
    boxless = make_alert(10, "fp", "cam-b")
    for frame in boxless["objects"][0]["frames"]:
        frame["boxes"] = []
    export = tmp_path / "export"
    write_export(export, [make_alert(1, "smoke", "cam-a"), boxless])
    out = tmp_path / "staging"
    result = run_import(export, out, tmp_path / "ledger.json")
    assert result.returncode == 0, result.stderr
    assert not (out / "fp").exists() or list((out / "fp").iterdir()) == []
