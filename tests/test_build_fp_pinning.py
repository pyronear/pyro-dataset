"""The YOLO FP build pins one background image per recurring object.

Annotator labels are score-less (five fields, class 99), so every frame ties:
the representative must be chosen by a deterministic rule, not directory
traversal order. Within an object: lexicographically first folder; within a
folder: highest label score, ties broken by filename.
"""

import importlib.util
from pathlib import Path

SCRIPT = Path("scripts/build_fp_yolo_dataset.py")

spec = importlib.util.spec_from_file_location("build_fp_yolo_dataset", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

BOX = "99 0.5 0.5 0.1 0.1\n"
SCORED = "0 0.5 0.5 0.1 0.1 0.42\n"


def make_seq(tmp_path, folder, labels):
    seq = tmp_path / folder
    (seq / "images").mkdir(parents=True)
    (seq / "labels").mkdir()
    for stem, text in labels.items():
        (seq / "images" / f"{stem}.jpg").write_bytes(b"jpg")
        (seq / "labels" / f"{stem}.txt").write_text(text)
    return seq


def test_scoreless_labels_pick_the_first_frame_by_name(tmp_path):
    make_seq(tmp_path, "seq", {"b": BOX, "a": BOX, "c": BOX})
    assert module.best_pinned_image(tmp_path / "seq").name == "a.jpg"


def test_a_scored_label_outranks_name_order(tmp_path):
    make_seq(tmp_path, "seq", {"a": BOX, "z": SCORED})
    assert module.best_pinned_image(tmp_path / "seq").name == "z.jpg"


def test_sequence_without_labeled_frames_yields_none(tmp_path):
    seq = tmp_path / "seq"
    (seq / "images").mkdir(parents=True)
    (seq / "labels").mkdir()
    assert module.best_pinned_image(seq) is None


def test_one_image_per_recurring_object(tmp_path):
    make_seq(tmp_path, "cam_2", {"x": BOX})
    make_seq(tmp_path, "cam_1", {"y": BOX})
    make_seq(tmp_path, "other_1", {"z": BOX})
    ledger = {
        "ro_00001": {"ingested_folders": ["cam_1", "cam_2"]},
        "ro_00002": {"ingested_folders": ["other_1"]},
    }
    pinned = [{"folder": "cam_1"}, {"folder": "cam_2"}, {"folder": "other_1"}]
    images = module.select_pinned_images(pinned, ledger, tmp_path)
    # ro_00001 is represented by its lexicographically first folder, cam_1
    assert [i.name for i in images] == ["y.jpg", "z.jpg"]


def test_object_falls_to_its_next_folder_when_the_first_has_no_labels(tmp_path):
    empty = tmp_path / "cam_1"
    (empty / "images").mkdir(parents=True)
    (empty / "labels").mkdir()
    make_seq(tmp_path, "cam_2", {"x": BOX})
    ledger = {"ro_00001": {"ingested_folders": ["cam_1", "cam_2"]}}
    images = module.select_pinned_images(
        [{"folder": "cam_1"}, {"folder": "cam_2"}], ledger, tmp_path
    )
    assert [i.name for i in images] == ["x.jpg"]
