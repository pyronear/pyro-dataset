import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from skip_covered_alerts import is_covered

from pyro_dataset.annotator.recurring import Ledger

BBOX = (0.20, 0.40, 0.24, 0.46)
ALERT = {
    "organisation_name": "sdis-91",
    "camera_name": "tigery-02",
    "azimuth": 18,
    "is_wildfire_alertapi": None,
}


def ledger_with(ingested: list[str]) -> tuple[Ledger, str]:
    ledger = Ledger()
    ro_id = ledger.mint("sdis-91_tigery-02", 18, BBOX, "train", "api:1")
    ledger.entries[ro_id]["ingested_folders"] = ingested
    return ledger, ro_id


def test_saturated_object_is_covered():
    ledger, ro_id = ledger_with(["sdis-91_tigery-02_18_2026-08-05T05-08-55"])
    assert is_covered(ledger, ALERT, BBOX, 0.3, 1) == ro_id


def test_object_still_under_the_cap_is_left_alone():
    ledger, _ = ledger_with(["sdis-91_tigery-02_18_2026-08-05T05-08-55"])
    assert is_covered(ledger, ALERT, BBOX, 0.3, max_per_object=2) is None


def test_never_covered_without_an_ingested_sequence():
    ledger, _ = ledger_with([])
    assert is_covered(ledger, ALERT, BBOX, 0.3, 1) is None


def test_smoke_labelled_alerts_are_never_skipped():
    ledger, _ = ledger_with(["sdis-91_tigery-02_18_2026-08-05T05-08-55"])
    smoke = {**ALERT, "is_wildfire_alertapi": "wildfire_smoke"}
    assert is_covered(ledger, smoke, BBOX, 0.3, 1) is None


def test_other_view_of_the_same_geometry_is_not_covered():
    ledger, _ = ledger_with(["sdis-91_tigery-02_18_2026-08-05T05-08-55"])
    elsewhere = {**ALERT, "azimuth": 22}
    assert is_covered(ledger, elsewhere, BBOX, 0.3, 1) is None


def test_no_geometry_means_no_skip():
    ledger, _ = ledger_with(["sdis-91_tigery-02_18_2026-08-05T05-08-55"])
    assert is_covered(ledger, ALERT, None, 0.3, 1) is None
