import json
import subprocess
import sys
from pathlib import Path

from tests.conftest import make_alert, small_export, write_export

SCRIPT = Path("scripts/plan_annotator_import.py")


def run_plan(export: Path, plan: Path, ledger: Path, extra: list[str] | None = None):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--export-dir",
            str(export),
            "--plan",
            str(plan),
            "--ledger",
            str(ledger),
            *(extra or []),
        ],
        capture_output=True,
        text=True,
    )


def read_plan(path: Path) -> dict:
    return json.loads(path.read_text())


def kinds(plan: dict, kind: str) -> list[str]:
    return [name for name, entry in plan.items() if entry["kind"] == kind]


def folder_of(plan: dict, alert_id: int) -> str:
    """The planned folder of `make_alert(alert_id, ...)`, found by its minute."""
    return next(name for name in plan if name.endswith(f"T13-{alert_id:02d}-00"))


def test_the_fp_quota_equals_the_smoke_count(tmp_path):
    export = tmp_path / "export"
    write_export(
        export,
        [make_alert(1, "smoke", "cam-a"), make_alert(2, "smoke", "cam-b")]
        + [make_alert(10 + i, "fp", f"cam-fp{i}") for i in range(5)],
    )
    plan_path = tmp_path / "plan.json"
    result = run_plan(export, plan_path, tmp_path / "ledger.json")
    assert result.returncode == 0, result.stderr

    plan = read_plan(plan_path)
    assert len(kinds(plan, "wildfire")) == 2
    assert len(kinds(plan, "fp")) == 2, "FP quota equals the smoke count"


def test_each_entry_carries_kind_split_and_recurring_object(tmp_path):
    plan_path = tmp_path / "plan.json"
    run_plan(small_export(tmp_path), plan_path, tmp_path / "ledger.json")
    plan = read_plan(plan_path)

    smoke = plan[kinds(plan, "wildfire")[0]]
    assert smoke["split"] in {"train", "val", "test"}
    assert smoke["recurring_object"] is None, "smoke has no recurring object"

    fp = plan[kinds(plan, "fp")[0]]
    assert fp["recurring_object"].startswith("ro_")


def test_splits_can_assign_test(tmp_path):
    export = tmp_path / "export"
    write_export(
        export,
        [make_alert(i, "smoke", f"cam-s{i}") for i in range(1, 31)]
        + [make_alert(30 + i, "fp", f"cam-f{i}") for i in range(1, 30)],
    )
    plan_path = tmp_path / "plan.json"
    result = run_plan(export, plan_path, tmp_path / "ledger.json")
    assert result.returncode == 0, result.stderr
    plan = read_plan(plan_path)
    splits = {entry["split"] for entry in plan.values()}
    assert splits <= {"train", "val", "test"}
    assert "test" in splits, "80/10/10 targets must reach test"


def test_same_fire_smoke_alerts_share_a_split(tmp_path):
    export = tmp_path / "export"
    # Same camera => same view; recorded_at differs by one minute.
    write_export(
        export,
        [make_alert(1, "smoke", "cam-a"), make_alert(2, "smoke", "cam-a")]
        + [make_alert(10 + i, "fp", f"cam-f{i}") for i in range(3)],
    )
    plan_path = tmp_path / "plan.json"
    result = run_plan(export, plan_path, tmp_path / "ledger.json")
    assert result.returncode == 0, result.stderr
    plan = read_plan(plan_path)
    assert plan[folder_of(plan, 1)]["split"] == plan[folder_of(plan, 2)]["split"]


def test_same_fire_split_is_inherited_across_runs(tmp_path):
    export = tmp_path / "export"
    plan_path = tmp_path / "plan.json"
    ledger_path = tmp_path / "ledger.json"

    write_export(export, [make_alert(1, "smoke", "cam-a")])
    run_plan(export, plan_path, ledger_path)
    first_plan = read_plan(plan_path)
    first_split = first_plan[folder_of(first_plan, 1)]["split"]

    write_export(
        export, [make_alert(1, "smoke", "cam-a"), make_alert(2, "smoke", "cam-a")]
    )
    run_plan(export, plan_path, ledger_path)
    plan = read_plan(plan_path)
    assert plan[folder_of(plan, 2)]["split"] == first_split


def test_rerun_is_idempotent_and_the_ledger_keeps_splits(tmp_path):
    export = small_export(tmp_path)
    plan_path = tmp_path / "plan.json"
    ledger_path = tmp_path / "ledger.json"

    run_plan(export, plan_path, ledger_path)
    first_plan = read_plan(plan_path)
    first_ledger = json.loads(ledger_path.read_text())

    run_plan(export, plan_path, ledger_path)
    assert read_plan(plan_path) == first_plan
    second_ledger = json.loads(ledger_path.read_text())
    assert set(second_ledger) == set(first_ledger)
    assert {k: v["split"] for k, v in second_ledger.items()} == {
        k: v["split"] for k, v in first_ledger.items()
    }


def test_a_later_export_does_not_double_pay_the_quota(tmp_path):
    """The export is a full re-pull, so smoke is cumulative while already
    planned folders are kept. Without discounting what earlier runs ingested,
    the second run would over-supply false positives and break the balance."""
    plan_path = tmp_path / "plan.json"
    ledger_path = tmp_path / "ledger.json"
    export = tmp_path / "export"

    write_export(
        export,
        [make_alert(1, "smoke", "cam-a"), make_alert(2, "smoke", "cam-b")]
        + [make_alert(10 + i, "fp", f"cam-f{i}") for i in range(6)],
    )
    run_plan(export, plan_path, ledger_path)
    plan = read_plan(plan_path)
    assert len(kinds(plan, "wildfire")) == 2
    assert len(kinds(plan, "fp")) == 2

    # A later pull: the same two smoke alerts are still there, plus two new.
    write_export(
        export,
        [make_alert(1, "smoke", "cam-a"), make_alert(2, "smoke", "cam-b")]
        + [make_alert(3, "smoke", "cam-c"), make_alert(4, "smoke", "cam-d")]
        + [make_alert(10 + i, "fp", f"cam-f{i}") for i in range(6)],
    )
    run_plan(export, plan_path, ledger_path)
    plan = read_plan(plan_path)
    assert len(kinds(plan, "wildfire")) == 4
    assert len(kinds(plan, "fp")) == 4, "cumulative balance, not 2 + 4"


def test_seen_does_not_grow_on_a_repeated_run(tmp_path):
    export = small_export(tmp_path)
    plan_path = tmp_path / "plan.json"
    ledger_path = tmp_path / "ledger.json"
    for _ in range(3):
        run_plan(export, plan_path, ledger_path)
    entry = next(iter(json.loads(ledger_path.read_text()).values()))
    assert entry["seen_alerts"] == ["pyronear_french:10"]


def test_force_train_moves_an_unused_object(tmp_path):
    export = small_export(tmp_path)
    plan_path = tmp_path / "plan.json"
    ledger_path = tmp_path / "ledger.json"
    run_plan(export, plan_path, ledger_path)

    ledger = json.loads(ledger_path.read_text())
    ro_id = next(iter(ledger))
    ledger[ro_id]["split"] = "val"
    ledger[ro_id]["ingested_folders"] = []
    ledger_path.write_text(json.dumps(ledger))

    result = run_plan(export, plan_path, ledger_path, ["--force-train", ro_id])
    assert result.returncode == 0, result.stderr
    assert json.loads(ledger_path.read_text())[ro_id]["split"] == "train"


def test_force_train_rewrites_the_split_of_an_already_planned_folder(tmp_path):
    """The ledger is authoritative: once an object is moved, the folder it
    planned must carry the new split, or add_data would register it under the
    stale one."""
    export = small_export(tmp_path)
    plan_path = tmp_path / "plan.json"
    ledger_path = tmp_path / "ledger.json"
    run_plan(export, plan_path, ledger_path)

    ledger = json.loads(ledger_path.read_text())
    ro_id = next(iter(ledger))
    fp_folder = kinds(read_plan(plan_path), "fp")[0]

    # Put the object in val by hand and forget its folder — the documented
    # escape hatch when the guard would otherwise refuse.
    ledger[ro_id]["split"] = "val"
    ledger[ro_id]["ingested_folders"] = []
    ledger_path.write_text(json.dumps(ledger))

    result = run_plan(export, plan_path, ledger_path, ["--force-train", ro_id])
    assert result.returncode == 0, result.stderr
    assert json.loads(ledger_path.read_text())[ro_id]["split"] == "train"
    assert read_plan(plan_path)[fp_folder]["split"] == "train"


def test_a_stale_plan_entry_is_resynced_to_the_ledger_split(tmp_path):
    """The plan outlives any single run, so an entry could otherwise keep a
    split the ledger has since moved — one artefact in two splits, which is
    exactly the leakage the ledger exists to prevent."""
    export = small_export(tmp_path)
    plan_path = tmp_path / "plan.json"
    ledger_path = tmp_path / "ledger.json"
    run_plan(export, plan_path, ledger_path)

    ledger = json.loads(ledger_path.read_text())
    ro_id = next(iter(ledger))
    fp_folder = kinds(read_plan(plan_path), "fp")[0]
    other = "val" if ledger[ro_id]["split"] == "train" else "train"
    ledger[ro_id]["split"] = other
    ledger_path.write_text(json.dumps(ledger))

    run_plan(export, plan_path, ledger_path)
    assert read_plan(plan_path)[fp_folder]["split"] == other


def test_force_train_is_refused_while_the_object_is_planned(tmp_path):
    """A folder already in the plan is re-recorded into the ledger even when
    nothing new is picked, so the guard sees it."""
    export = small_export(tmp_path)
    plan_path = tmp_path / "plan.json"
    ledger_path = tmp_path / "ledger.json"
    run_plan(export, plan_path, ledger_path)

    ledger = json.loads(ledger_path.read_text())
    ro_id = next(iter(ledger))
    ledger[ro_id]["split"] = "val"
    ledger[ro_id]["ingested_folders"] = []  # ledger forgets, the plan remembers
    ledger_path.write_text(json.dumps(ledger))
    run_plan(export, plan_path, ledger_path)  # re-records from the planned folder

    result = run_plan(export, plan_path, ledger_path, ["--force-train", ro_id])
    assert result.returncode != 0
    assert "refused" in result.stderr


def test_force_train_refuses_an_object_already_ingested_elsewhere(tmp_path):
    export = small_export(tmp_path)
    plan_path = tmp_path / "plan.json"
    ledger_path = tmp_path / "ledger.json"
    run_plan(export, plan_path, ledger_path)

    ledger = json.loads(ledger_path.read_text())
    ro_id = next(iter(ledger))
    ledger[ro_id]["split"] = "val"
    ledger[ro_id]["ingested_folders"] = ["some-folder"]
    ledger_path.write_text(json.dumps(ledger))

    result = run_plan(export, plan_path, ledger_path, ["--force-train", ro_id])
    assert result.returncode != 0
    assert "refused" in result.stderr


def test_force_train_rejects_an_unknown_object(tmp_path):
    export = small_export(tmp_path)
    plan_path = tmp_path / "plan.json"
    ledger_path = tmp_path / "ledger.json"
    run_plan(export, plan_path, ledger_path)

    result = run_plan(export, plan_path, ledger_path, ["--force-train", "ro_99999"])
    assert result.returncode != 0
    assert "unknown" in result.stderr


def test_dry_run_writes_nothing(tmp_path):
    export = small_export(tmp_path)
    plan_path = tmp_path / "plan.json"
    ledger_path = tmp_path / "ledger.json"
    result = run_plan(export, plan_path, ledger_path, ["--dry-run"])
    assert result.returncode == 0, result.stderr
    assert not plan_path.exists()
    assert not ledger_path.exists()


def test_an_alert_without_boxes_is_not_a_candidate(tmp_path):
    boxless = make_alert(10, "fp", "cam-b")
    for frame in boxless["objects"][0]["frames"]:
        frame["boxes"] = []
    export = tmp_path / "export"
    write_export(export, [make_alert(1, "smoke", "cam-a"), boxless])
    plan_path = tmp_path / "plan.json"
    result = run_plan(export, plan_path, tmp_path / "ledger.json")
    assert result.returncode == 0, result.stderr
    assert kinds(read_plan(plan_path), "fp") == []


def test_an_alert_that_changed_kind_keeps_its_planned_kind_and_warns(tmp_path):
    """Re-annotation can flip an alert from fp to wildfire. The folder was
    already registered under the old kind and the registry is append-only, so
    the plan keeps it — but silently changing what a sequence means is exactly
    the kind of thing that must reach a human."""
    plan_path = tmp_path / "plan.json"
    ledger_path = tmp_path / "ledger.json"
    export = tmp_path / "export"
    write_export(
        export, [make_alert(1, "smoke", "cam-a"), make_alert(10, "fp", "cam-b")]
    )
    run_plan(export, plan_path, ledger_path)
    flipped = kinds(read_plan(plan_path), "fp")[0]

    reannotated = make_alert(10, "smoke", "cam-b")
    write_export(export, [make_alert(1, "smoke", "cam-a"), reannotated])
    result = run_plan(export, plan_path, ledger_path)
    assert result.returncode == 0, result.stderr

    assert read_plan(plan_path)[flipped]["kind"] == "fp"
    assert "changed kind" in result.stderr


def test_a_wildfire_folder_is_never_recorded_against_a_recurring_object(tmp_path):
    """The other flip direction. A folder planned as wildfire keeps that kind,
    so recording it as a recurring object's contributed sequence would claim a
    positive as the object's false positive — burning its lifetime slot and
    shorting the quota forever, both silently and both permanently."""
    plan_path = tmp_path / "plan.json"
    ledger_path = tmp_path / "ledger.json"
    export = tmp_path / "export"

    write_export(
        export,
        [make_alert(i, "smoke", f"cam-s{i}") for i in range(1, 4)]
        + [make_alert(20, "fp", "cam-f0")],
    )
    run_plan(export, plan_path, ledger_path)
    flipped = folder_of(read_plan(plan_path), alert_id=3)

    # Alert 3 is re-annotated as a false positive, and the quota has room.
    write_export(
        export,
        [make_alert(i, "smoke", f"cam-s{i}") for i in range(1, 3)]
        + [make_alert(3, "fp", "cam-s3")]
        + [make_alert(20 + i, "fp", f"cam-f{i}") for i in range(3)],
    )
    result = run_plan(export, plan_path, ledger_path)
    assert result.returncode == 0, result.stderr

    assert read_plan(plan_path)[flipped]["kind"] == "wildfire"
    ledger = json.loads(ledger_path.read_text())
    recorded = {f for entry in ledger.values() for f in entry["ingested_folders"]}
    assert flipped not in recorded


def test_an_alert_with_no_image_on_disk_is_never_planned(tmp_path):
    """Deciding this here rather than while copying keeps materialisation a
    total function of the plan — and stops an unusable alert from burning its
    recurring object's one slot."""
    export = tmp_path / "export"
    write_export(
        export,
        [make_alert(1, "smoke", "cam-a")]
        + [make_alert(10 + i, "fp", f"cam-f{i}") for i in range(2)],
    )
    for path in (export / "images" / "pyronear_french" / "10").iterdir():
        path.unlink()

    plan_path = tmp_path / "plan.json"
    result = run_plan(export, plan_path, tmp_path / "ledger.json")
    assert result.returncode == 0, result.stderr

    plan = read_plan(plan_path)
    assert len(kinds(plan, "fp")) == 1
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert len(ledger) == 1, "the imageless alert never became a recurring object"
