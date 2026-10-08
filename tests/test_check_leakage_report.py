import subprocess
import sys
from pathlib import Path

SCRIPT = Path("scripts/check_leakage_report.py").resolve()

REPORT = (
    '<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest" '
    'errors="0" failures="0" skipped="0" tests="1"><testcase name="t" /></testsuite>'
    "</testsuites>"
)
LOCK = """stages:
  build:
    cmd: x
    outs:
    - path: data/processed/built
      md5: aaa.dir
  test_data_leakage:
    cmd: x
    deps:
    - path: data/raw/wildfire
      md5: bbb.dir
    - path: data/processed/built
      md5: aaa.dir
    - path: tests/test_data_leakage.py
      md5: 9e107d9d372bb6826bd81d3542a419d6
"""


def make_repo(root: Path, pool_md5: str = "bbb.dir") -> None:
    (root / "data" / "reporting").mkdir(parents=True)
    (root / "data" / "reporting" / "leakage_results.xml").write_text(REPORT)
    (root / "data" / "raw").mkdir()
    (root / "data" / "raw" / "wildfire.dvc").write_text(
        f"outs:\n- md5: {pool_md5}\n  path: wildfire\n"
    )
    (root / "tests").mkdir()
    (root / "tests" / "test_data_leakage.py").write_text(
        "The quick brown fox jumps over the lazy dog"
    )
    (root / "dvc.lock").write_text(LOCK)


def run(root: Path):
    return subprocess.run(
        [sys.executable, str(SCRIPT)], cwd=root, capture_output=True, text=True
    )


def test_a_report_matching_the_lock_passes(tmp_path):
    make_repo(tmp_path)
    result = run(tmp_path)
    assert result.returncode == 0, result.stdout


def test_a_pool_that_moved_since_the_report_fails(tmp_path):
    make_repo(tmp_path, pool_md5="ccc.dir")
    result = run(tmp_path)
    assert result.returncode == 1
    assert "predates changes to ['data/raw/wildfire']" in result.stdout
