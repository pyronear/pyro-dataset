import json
import subprocess
import sys
from pathlib import Path

from tests.conftest import make_alert, small_export, write_export

PLAN_SCRIPT = Path("scripts/plan_annotator_import.py")
SCRIPT = Path("scripts/materialise_annotator_sequences.py")


def run_materialise(export: Path, plan: Path, out: Path):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--export-dir",
            str(export),
            "--plan",
            str(plan),
            "--output-dir",
            str(out),
        ],
        capture_output=True,
        text=True,
    )


def stage(tmp_path: Path, export: Path) -> Path:
    """Plan the export, then materialise it. Returns the staging directory."""
    plan = tmp_path / "plan.json"
    out = tmp_path / "staging"
    planned = subprocess.run(
        [
            sys.executable,
            str(PLAN_SCRIPT),
            "--export-dir",
            str(export),
            "--plan",
            str(plan),
            "--ledger",
            str(tmp_path / "ledger.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert planned.returncode == 0, planned.stderr
    result = run_materialise(export, plan, out)
    assert result.returncode == 0, result.stderr
    return out


def test_each_folder_has_images_labels_and_meta(tmp_path):
    out = stage(tmp_path, small_export(tmp_path))

    folder = next((out / "wildfire").iterdir())
    assert len(list((folder / "images").glob("*.jpg"))) == 3
    assert len(list((folder / "labels").glob("*.txt"))) == 3
    meta = json.loads((folder / "meta.json").read_text())
    assert meta["platform_alert_id"] == 1
    assert meta["lanes"][0]["track"], "meta.json carries the full track"


def test_folder_and_frame_names_satisfy_ingest_validation(tmp_path):
    from pyro_dataset.ingest import validate_sequence_folder

    out = stage(tmp_path, small_export(tmp_path))
    for kind in ("wildfire", "fp"):
        for folder in (out / kind).iterdir():
            result = validate_sequence_folder(folder)
            assert result.is_valid, (
                folder.name,
                result.naming_issues,
                result.structural_issues,
            )


def test_fp_labels_use_the_sentinel_class(tmp_path):
    out = stage(tmp_path, small_export(tmp_path))

    folder = next((out / "fp").iterdir())
    line = next((folder / "labels").glob("*.txt")).read_text().strip()
    assert line.startswith("99 ")


def test_meta_carries_the_recurring_object_the_plan_assigned(tmp_path):
    out = stage(tmp_path, small_export(tmp_path))

    folder = next((out / "fp").iterdir())
    plan = json.loads((tmp_path / "plan.json").read_text())
    meta = json.loads((folder / "meta.json").read_text())
    assert meta["recurring_object"] == plan[folder.name]["recurring_object"]


def test_splits_json_covers_every_materialised_folder(tmp_path):
    out = stage(tmp_path, small_export(tmp_path))

    splits = json.loads((out / "splits.json").read_text())
    folders = {p.name for kind in ("wildfire", "fp") for p in (out / kind).iterdir()}
    assert folders == set(splits)
    assert set(splits.values()) <= {"train", "val"}


def test_the_splits_match_the_plan(tmp_path):
    out = stage(tmp_path, small_export(tmp_path))

    splits = json.loads((out / "splits.json").read_text())
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert splits == {name: entry["split"] for name, entry in plan.items()}


def test_rerunning_leaves_already_staged_folders_untouched(tmp_path):
    export = small_export(tmp_path)
    out = stage(tmp_path, export)
    staged = [
        path
        for kind in ("wildfire", "fp")
        for path in (out / kind).rglob("*")
        if path.is_file()
    ]
    before = {path: path.stat().st_mtime_ns for path in staged}

    result = run_materialise(export, tmp_path / "plan.json", out)
    assert result.returncode == 0, result.stderr
    assert {path: path.stat().st_mtime_ns for path in staged} == before


def test_a_reannotated_alert_is_materialised_wholly_as_its_planned_kind(tmp_path):
    """The plan pins the kind because the registry is append-only. Pinning only
    the directory would leave a folder registered as a false positive holding
    class-0 labels and a `meta.json` calling itself wildfire — a smoke sequence
    inside the negative pool, and the temporal dataset copies `labels/` as-is."""
    export = tmp_path / "export"
    write_export(
        export, [make_alert(1, "smoke", "cam-a"), make_alert(10, "fp", "cam-b")]
    )
    stage(tmp_path, export)

    # Re-annotated: the same alert now carries a smoke lane.
    write_export(
        export, [make_alert(1, "smoke", "cam-a"), make_alert(10, "smoke", "cam-b")]
    )
    fresh_out = tmp_path / "staging2"
    result = run_materialise(export, tmp_path / "plan.json", fresh_out)
    assert result.returncode == 0, result.stderr

    folder = next((fresh_out / "fp").iterdir())
    line = next((folder / "labels").glob("*.txt")).read_text().strip()
    assert line.startswith("99 "), "still a false-positive proposal box"
    assert json.loads((folder / "meta.json").read_text())["kind"] == "fp"


def test_a_planned_alert_missing_from_the_export_is_skipped(tmp_path):
    """Staging is transient and the export is a full re-pull, so a folder
    already consumed into data/raw can drop out of a later export. Failing
    here would wedge the stage for good."""
    export = tmp_path / "export"
    write_export(
        export, [make_alert(1, "smoke", "cam-a"), make_alert(10, "fp", "cam-b")]
    )
    stage(tmp_path, export)
    plan = tmp_path / "plan.json"
    planned = json.loads(plan.read_text())
    kept = next(name for name, e in planned.items() if e["kind"] == "wildfire")
    gone = next(name for name, e in planned.items() if e["kind"] == "fp")

    write_export(export, [make_alert(1, "smoke", "cam-a")])
    fresh_out = tmp_path / "staging2"
    result = run_materialise(export, plan, fresh_out)
    assert result.returncode == 0, result.stderr

    splits = json.loads((fresh_out / "splits.json").read_text())
    assert set(splits) == {kept}
    assert (fresh_out / "wildfire" / kept).is_dir()
    assert not (fresh_out / "fp" / gone).exists()
