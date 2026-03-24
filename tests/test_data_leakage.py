"""
Tests to detect data leakage between train/val/test splits in processed datasets.

Two types of leakage are checked:

1. Image-level: the same image filename appears in multiple splits.
2. Sequence-level (sequential datasets only): the same sequence folder appears
   in multiple splits. Each folder corresponds to one labelled event — images
   from the same sequence must not span splits.

Note: same-camera images across splits are acceptable (a camera can contribute
to both train and test on different events). Only exact duplicates (images) or
the same sequence event (sequential datasets) are flagged.

Datasets covered:
- wildfire_yolo      – merged YOLO dataset (images/train, images/val, images/test)
- yolo_train_val     – YOLO train/val (images/train, images/val)  \\ checked against
- yolo_test          – YOLO test      (images/test)               //
- sequential_train_val + sequential_test – sequential datasets for wildfire and fp
"""

from pathlib import Path

import pytest

PROCESSED = Path(__file__).parent.parent / "data" / "processed"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _yolo_images(split_dir: Path) -> set[str]:
    """Return the set of image stems in a flat YOLO split directory."""
    return {p.stem for p in split_dir.glob("*.jpg")}


def _seq_folders(category_dir: Path) -> set[str]:
    """Return the set of sequence folder names in a sequential category dir."""
    return {p.name for p in category_dir.iterdir() if p.is_dir()}


def _seq_images(category_dir: Path) -> set[str]:
    """Return the set of image stems across all sequences in a sequential dir."""
    return {p.stem for p in category_dir.glob("*/images/*.jpg")}


def _skip_if_missing(*dirs: Path) -> None:
    missing = [str(d) for d in dirs if not d.exists()]
    if missing:
        pytest.skip(f"Data not available (run `dvc pull`): {missing}")


# ---------------------------------------------------------------------------
# wildfire_yolo – image leakage between splits
# ---------------------------------------------------------------------------

WILDFIRE_YOLO_IMAGES = PROCESSED / "wildfire_yolo" / "images"

YOLO_SPLIT_PAIRS = [
    ("train", "test"),
    ("val", "test"),
    ("train", "val"),
]


@pytest.mark.parametrize("split_a,split_b", YOLO_SPLIT_PAIRS)
def test_wildfire_yolo_no_image_leakage(split_a: str, split_b: str) -> None:
    """No image filename should appear in two splits of wildfire_yolo."""
    dir_a = WILDFIRE_YOLO_IMAGES / split_a
    dir_b = WILDFIRE_YOLO_IMAGES / split_b
    _skip_if_missing(dir_a, dir_b)

    overlap = _yolo_images(dir_a) & _yolo_images(dir_b)
    assert not overlap, (
        f"wildfire_yolo image leakage ({split_a} ∩ {split_b}): "
        f"{len(overlap)} duplicate(s), e.g. {sorted(overlap)[:3]}"
    )


# ---------------------------------------------------------------------------
# yolo_train_val vs yolo_test – image leakage
# ---------------------------------------------------------------------------

YOLO_TRAIN_VAL_IMAGES = PROCESSED / "yolo_train_val" / "images"
YOLO_TEST_IMAGES = PROCESSED / "yolo_test" / "images" / "test"

YOLO_CROSS_PAIRS = [
    (YOLO_TRAIN_VAL_IMAGES / "train", "train"),
    (YOLO_TRAIN_VAL_IMAGES / "val", "val"),
]


@pytest.mark.parametrize("train_val_dir,tv_name", YOLO_CROSS_PAIRS)
def test_yolo_datasets_no_image_leakage(train_val_dir: Path, tv_name: str) -> None:
    """No image in yolo_test should duplicate an image in yolo_train_val."""
    _skip_if_missing(train_val_dir, YOLO_TEST_IMAGES)

    overlap = _yolo_images(train_val_dir) & _yolo_images(YOLO_TEST_IMAGES)
    assert not overlap, (
        f"yolo image leakage ({tv_name} ∩ test): "
        f"{len(overlap)} duplicate(s), e.g. {sorted(overlap)[:3]}"
    )


# ---------------------------------------------------------------------------
# Sequential datasets – sequence-folder and image leakage
# ---------------------------------------------------------------------------

SEQ_TRAIN_VAL = PROCESSED / "sequential_train_val"
SEQ_TEST = PROCESSED / "sequential_test"

SEQ_CATEGORIES = ["wildfire", "fp"]

SEQ_SPLIT_PAIRS = [
    (SEQ_TRAIN_VAL / "train", SEQ_TRAIN_VAL / "val", "train", "val"),
    (SEQ_TRAIN_VAL / "train", SEQ_TEST / "test", "train", "test"),
    (SEQ_TRAIN_VAL / "val", SEQ_TEST / "test", "val", "test"),
]


@pytest.mark.parametrize("category", SEQ_CATEGORIES)
@pytest.mark.parametrize("dir_a,dir_b,name_a,name_b", SEQ_SPLIT_PAIRS)
def test_sequential_no_sequence_leakage(
    dir_a: Path, dir_b: Path, name_a: str, name_b: str, category: str
) -> None:
    """No sequence folder should appear in two sequential splits."""
    cat_a = dir_a / category
    cat_b = dir_b / category
    _skip_if_missing(cat_a, cat_b)

    overlap = _seq_folders(cat_a) & _seq_folders(cat_b)
    assert not overlap, (
        f"Sequential [{category}] sequence leakage ({name_a} ∩ {name_b}): "
        f"{len(overlap)} sequence(s) shared, e.g. {sorted(overlap)[:3]}"
    )


@pytest.mark.parametrize("category", SEQ_CATEGORIES)
@pytest.mark.parametrize("dir_a,dir_b,name_a,name_b", SEQ_SPLIT_PAIRS)
def test_sequential_no_image_leakage(
    dir_a: Path, dir_b: Path, name_a: str, name_b: str, category: str
) -> None:
    """No image filename should appear in two sequential splits."""
    cat_a = dir_a / category
    cat_b = dir_b / category
    _skip_if_missing(cat_a, cat_b)

    overlap = _seq_images(cat_a) & _seq_images(cat_b)
    assert not overlap, (
        f"Sequential [{category}] image leakage ({name_a} ∩ {name_b}): "
        f"{len(overlap)} image(s) shared, e.g. {sorted(overlap)[:3]}"
    )
