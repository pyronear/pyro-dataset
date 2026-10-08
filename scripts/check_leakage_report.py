"""
Fail when the committed leakage report is stale or failing.

`dvc repro test_data_leakage` writes data/reporting/leakage_results.xml, which is
committed because CI has no data. CI runs this instead: the report must pass,
must come from a run with every dataset present, and the inputs recorded for
the stage in dvc.lock must be the ones the repository points at now — the .dvc
pointer for a raw pool, the producing stage's output for a processed dataset,
the file itself for code and the git-tracked state files. A mismatch means the
data moved since the report was produced.

Usage:
    python scripts/check_leakage_report.py   # from the repo root
"""

import hashlib
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

REPORT = Path("data/reporting/leakage_results.xml")
LOCK = Path("dvc.lock")
STAGE = "test_data_leakage"


def current_hashes(lock: dict) -> dict[str, str]:
    """What each path hashes to now, from the pointers committed beside the lock."""
    hashes = {
        out["path"]: out["md5"]
        for stage in lock["stages"].values()
        for out in stage.get("outs", [])
    }
    for pointer in Path("data").rglob("*.dvc"):
        for out in yaml.safe_load(pointer.read_text())["outs"]:
            hashes[str(pointer.parent / out["path"])] = out["md5"]
    return hashes


def stale_deps(lock: dict) -> list[str]:
    hashes = current_hashes(lock)
    stale = []
    for dep in lock["stages"][STAGE]["deps"]:
        path = dep["path"]
        if path in hashes:
            now = hashes[path]
        elif Path(path).is_file():
            now = hashlib.md5(Path(path).read_bytes()).hexdigest()
        else:
            now = None
        if now != dep["md5"]:
            stale.append(path)
    return stale


def main() -> int:
    if not REPORT.is_file():
        print(f"ERROR: {REPORT} not found. Run `dvc repro {STAGE}` and commit it.")
        return 1
    stale = stale_deps(yaml.safe_load(LOCK.read_text()))
    if stale:
        print(f"ERROR: the report predates changes to {stale}.")
        print(f"Run `dvc repro {STAGE}` and commit dvc.lock with the report.")
        return 1

    # pytest wraps its <testsuite> in a <testsuites> root that carries no
    # counts; reading the root alone reports 0 tests and always passes.
    root = ET.parse(REPORT).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))

    def total(key: str) -> int:
        return sum(int(s.attrib.get(key, 0)) for s in suites)

    failures, errors = total("failures"), total("errors")
    tests, skipped = total("tests"), total("skipped")
    print(f"Tests: {tests}, Skipped: {skipped}, Failures: {failures}, Errors: {errors}")

    # A skip means the data was missing when the report was produced.
    if not tests or skipped:
        print("ERROR: the report must come from a run with every dataset present.")
        return 1
    if failures or errors:
        for tc in root.iter("testcase"):
            for bad in list(tc.iter("failure")) + list(tc.iter("error")):
                print(f"FAIL {tc.attrib.get('name')}: {bad.text}")
        return 1
    print("All leakage tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
