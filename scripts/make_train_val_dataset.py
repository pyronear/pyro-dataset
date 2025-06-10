"""
CLI Script to generate the wildfire dataset (train and val).

The folder structure follows the ultralytics scaffolding template.

Usage:
    python make_train_val_dataset.py --save-dir <save_directory> --dir-data-split-smoke <smoke_data_directory> --dir-data-split-false-positives <false_positives_directory> --random-seed <seed> [--percentage-background-images <percentage>] [-log <loglevel>]

Arguments:
    --save-dir: Directory to save the wildfire dataset.
    --dir-data-split-smoke: Directory containing the part of the wildfire dataset with fire smokes only (no background images).
    --dir-data-split-false-positives: Directory containing the false positives (only background images).
    --random-seed: Random seed for reproducibility.
    --percentage-background-images: Percentage of background images to add to the dataset (default is 0.1).
    -log, --loglevel: Provide logging level (default is info).
"""

import argparse
import logging
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from pyro_dataset.constants import CLASS_ID_SMOKE, CLASS_SMOKE_LABEL
from pyro_dataset.utils import combine_ultralytics_datasets, yaml_write
from pyro_dataset.yolo.utils import annotation_to_txt, parse_yolo_annotation_txt_file


@dataclass
class DataSplit:
    """
    Simple class to store the result of the datasplit.

    Attributes:
        train: contains a list of image filepaths and makes the train split.
        val: contains a list of image filepaths and makes the val split.
        test: contains a list of image filepaths and makes the test split.
    """

    train: list[Path]
    val: list[Path]
    test: list[Path]


def make_cli_parser() -> argparse.ArgumentParser:
    """
    Make the CLI parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-dir",
        help="directory to save the wildfire dataset.",
        type=Path,
        default=Path("./data/processed/wildfire/"),
    )
    parser.add_argument(
        "--dir-data-split-smoke",
        help="directory containing the part of the wildfire dataset with fire smokes only (no background images).",
        type=Path,
        default=Path("./data/interim/data-split/smoke/wildfire/"),
    )
    parser.add_argument(
        "--dir-data-split-false-positives",
        help="directory containing the false positives (only background images).",
        type=Path,
        default=Path("./data/interim/data-split/false_positives/FP_2024/wise_wolf/"),
    )
    parser.add_argument(
        "--dirs-ultralytics-datasets",
        help="list of directories containing ultralytics datasets to incorporate.",
        type=Path,
        nargs="+",
        default=[],
    )
    parser.add_argument(
        "--random-seed",
        help="random seed",
        type=int,
        required=True,
        default=0,
    )
    parser.add_argument(
        "--percentage-background-images",
        help="percentage of background images to add to the dataset",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "-log",
        "--loglevel",
        default="info",
        help="Provide logging level. Example --loglevel debug, default=warning",
    )
    return parser


def validate_parsed_args(args: dict) -> bool:
    """
    Return whether the parsed args are valid.
    """
    if not args["dir_data_split_smoke"].exists():
        logging.error(
            f"invalid --dir-data-split-smoke, dir {args['dir_data_split_smoke']} does not exist"
        )
        return False
    elif not args["dir_data_split_false_positives"].exists():
        logging.error(
            f"invalid --dir-data-split-false-positives, dir {args['dir_data_split_false_positives']} does not exist"
        )
        return False

    for dir_dataset in args["dirs_ultralytics_datasets"]:
        if not dir_dataset.exists() and dir_dataset.is_dir():
            logging.error(
                f"The provided ultralytics dataset dir is invalid {dir_dataset}"
            )
            return False

    return True


def list_ultralytics_images(dir_dataset: Path) -> list[Path]:
    """
    List all images in a ultralytics YOLO dataset.

    Returns:
        filepaths (list[Path])
    """
    return list(dir_dataset.glob("**/*.jpg"))


def list_ultralytics_labels(dir_dataset: Path) -> list[Path]:
    """
    List all filepaths in a ultralytics YOLO dataset.

    Returns:
        filepaths (list[Path])
    """
    return list(dir_dataset.glob("**/*.txt"))


def add_percentage_background_images(
    data_split_smoke: DataSplit,
    data_split_background: DataSplit,
    percentage_background: float,
    random_seed: int,
) -> DataSplit:
    """
    Create a DataSplit that contains at most `percentage_background_images` for
    the train and val splits.

    Returns:
        data_split (DataSplit): a new data_split containg at most
        `percentage_background` based on the data_split_background.
    """

    rng = random.Random(random_seed)
    train_sampled_false_positives = rng.sample(
        data_split_background.train,
        min(
            len(data_split_background.train),
            int(len(data_split_smoke.train) * percentage_background),
        ),
    )
    val_sampled_false_positives = rng.sample(
        data_split_background.val,
        min(
            len(data_split_background.val),
            int(len(data_split_smoke.val) * percentage_background),
        ),
    )
    return DataSplit(
        train=data_split_smoke.train.copy() + train_sampled_false_positives,
        val=data_split_smoke.val.copy() + val_sampled_false_positives,
        test=[],
    )


def load_data_split_from_ultralytics(dir_dataset: Path) -> DataSplit:
    """
    Load a DataSplit from a directory following the ultralytics scaffolding
    template.
    """
    filepaths_images = list_ultralytics_images(dir_dataset)
    train_filepaths = []
    val_filepaths = []
    test_filepaths = []

    for filepath_image in tqdm(filepaths_images):
        split = filepath_image.parts[-2]
        if split == "train":
            train_filepaths.append(filepath_image)
        elif split == "val":
            val_filepaths.append(filepath_image)
        elif split == "test":
            test_filepaths.append(filepath_image)

    return DataSplit(
        train=train_filepaths,
        val=val_filepaths,
        test=test_filepaths,
    )


def filepath_image_to_filepath_label(filepath_image: Path) -> Path:
    """
    Given a filepath_image it returns its associated filepath_label.

    Returns:
        filepath_label (Path): the associated label filepath.
    """
    split = filepath_image.parts[-2]
    return (
        filepath_image.parent.parent.parent
        / "labels"
        / split
        / f"{filepath_image.stem}.txt"
    )


def persist_data_split(
    data_split: DataSplit,
    save_dir: Path,
) -> None:
    """
    Persist the data_split in save_dir, structure it in a regular yolo folder structure.

    Returns:
        None
    """
    for split, split_xs in [("train", data_split.train), ("val", data_split.val)]:
        for filepath_image in tqdm(split_xs):
            filepath_label = filepath_image_to_filepath_label(filepath_image)
            filepath_image_destination = (
                save_dir / "images" / split / filepath_image.name
            )
            filepath_label_destination = (
                save_dir / "labels" / split / filepath_label.name
            )
            filepath_image_destination.parent.mkdir(parents=True, exist_ok=True)
            filepath_label_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(filepath_image, filepath_image_destination)
            if filepath_label.exists():
                shutil.copy(filepath_label, filepath_label_destination)
            else:
                filepath_label_destination.touch()


def normalize_class_id(save_dir: Path, class_id: int) -> None:
    """
    Rewrite instance ids to the specified `instance_id` to avoid discrepencies
    between datasets (0 or 1 would map to fire smoke).

    It parses the annotation and sets the class_id to thew provided_one when
    overwriting the label content.

    Note: It seems that the annotated data from pyro-sdis uses 1 as a class_id
    for the fire smokes.
    """
    filepaths_labels = list_ultralytics_labels(save_dir)
    for filepath_label in filepaths_labels:
        with open(filepath_label, "r") as fd:
            txt_content = fd.read()
            yolo_object_detection_annotations = parse_yolo_annotation_txt_file(
                txt_content
            )

            lines = []
            for annotation in yolo_object_detection_annotations:
                annotation.class_id = class_id
                line_content_label = annotation_to_txt(annotation)
                lines.append(line_content_label)

            content_label = "\n".join(lines)

        with open(filepath_label, "w") as fd:
            fd.write(content_label)


def write_data_yaml(yaml_filepath: Path) -> None:
    """Writes the data.yaml file used by the yolo models."""
    content = {
        "train": "./images/train",
        "val": "./images/val",
        "nc": 1,
        "names": [CLASS_SMOKE_LABEL],
    }
    yaml_write(to=yaml_filepath, data=content)


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        exit(1)
    else:
        logging.info(args)
        save_dir = args["save_dir"]
        dir_data_split_smoke = args["dir_data_split_smoke"]
        dir_data_split_false_positives = args["dir_data_split_false_positives"]
        dirs_ultralytics_datasets = args["dirs_ultralytics_datasets"]
        random_seed = args["random_seed"]
        percentage_background_images = args["percentage_background_images"]

        logger.info(f"save the wildfire dataset in {save_dir}")
        save_dir.mkdir(parents=True, exist_ok=True)
        filepaths_smoke_images = list_ultralytics_images(dir_data_split_smoke)
        filepaths_false_positive_images = list_ultralytics_images(
            dir_data_split_false_positives
        )
        logger.info(
            f"found {len(filepaths_smoke_images)} images in {dir_data_split_smoke}"
        )
        logger.info(
            f"found {len(filepaths_false_positive_images)} images in {dir_data_split_false_positives}"
        )
        data_split_smoke = load_data_split_from_ultralytics(dir_data_split_smoke)
        data_split_false_positives = load_data_split_from_ultralytics(
            dir_data_split_false_positives
        )
        logger.info(
            f"add {percentage_background_images} percentage of background images."
        )
        data_split = add_percentage_background_images(
            data_split_smoke=data_split_smoke,
            data_split_background=data_split_false_positives,
            percentage_background=percentage_background_images,
            random_seed=random_seed,
        )
        logger.info(f"persist the data split in {save_dir}")
        persist_data_split(data_split=data_split, save_dir=save_dir)
        logger.info(f"normalize class id for all annotations to {CLASS_ID_SMOKE}")
        normalize_class_id(save_dir=save_dir, class_id=CLASS_ID_SMOKE)
        filepath_data_yaml = save_dir / "data.yaml"
        logger.info(f"save data.yaml in {filepath_data_yaml}")
        write_data_yaml(save_dir / "data.yaml")

        combine_ultralytics_datasets(
            dirs_dataset=dirs_ultralytics_datasets,
            dir_save=save_dir,
            splits=["train", "val"],
        )

        exit(0)
