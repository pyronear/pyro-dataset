"""Moving a sequence between pools keeps every registry field but the id.

Dropping `source` silently unpins an annotator sequence from the builders.
"""

import importlib.util
import json
from pathlib import Path

import pytest


def load(script: str):
    spec = importlib.util.spec_from_file_location(script, Path(f"scripts/{script}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "script,src,dst,new_id",
    [
        ("move_fp_to_wildfire", "fp", "wildfire", "wf_00000001"),
        ("move_wf_to_fp", "wildfire", "fp", "fp_00000001"),
    ],
)
def test_move_keeps_registry_fields(tmp_path, monkeypatch, script, src, dst, new_id):
    module = load(script)
    entry = {
        "id": "x_00000007",
        "folder": "seq",
        "camera": "cam",
        "split": "test",
        "source": "pyro-annotator",
    }
    for pool, sequences in ((src, [entry]), (dst, [])):
        (tmp_path / pool / "data").mkdir(parents=True)
        (tmp_path / pool / "registry.json").write_text(
            json.dumps({"sequences": sequences})
        )
    (tmp_path / src / "data" / "seq").mkdir()
    for name in ("FP_DIR", "WF_DIR"):
        pool = tmp_path / ("fp" if name == "FP_DIR" else "wildfire")
        monkeypatch.setattr(module, name, pool)
    monkeypatch.setattr(module, "FP_REGISTRY", tmp_path / "fp" / "registry.json")
    monkeypatch.setattr(module, "WF_REGISTRY", tmp_path / "wildfire" / "registry.json")

    module.move_sequences(["seq"])

    moved = json.loads((tmp_path / dst / "registry.json").read_text())["sequences"]
    assert moved == [{**entry, "id": new_id}]
    assert (tmp_path / dst / "data" / "seq").is_dir()
