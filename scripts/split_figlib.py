"""
CLI Script to perform a data split from the FIGLIB-ANNOTATED-RESIZED dataset to assign images into a train, val, test split.

The folder structure will follow a ultralytics YOLO scaffolding.
"""

import argparse
import logging
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from pyro_dataset.constants import DATE_FORMAT_OUTPUT

# Date format used in the naming of files in FIGLIB_ANNOTATED_RESIZED
DATE_FORMAT_INPUT = "%Y_%m_%dT%H_%M_%S"


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


@dataclass
class ObservationMetadata:
    """
    Simple class to store the metadata of an observation.

    Attributes:
        reference_id (str): the camera reference (where it was taken)
        datetime (datetime): when it was taken
    """

    reference_id: str
    datetime: datetime


def parse_filepath_image(filepath_image: Path) -> ObservationMetadata:
    """
    Given a filepath_image, it returns an ObservationMetadata containing some
    key information about the location and time the picture was taken.


    Returns:
        observation_metadata (ObservationMetadata)
    """
    reference_id = "_".join(filepath_image.stem.split("_")[:3])
    datetime_str = "_".join(filepath_image.stem.split("_")[3:])
    return ObservationMetadata(
        reference_id=reference_id,
        datetime=datetime.strptime(datetime_str, DATE_FORMAT_INPUT),
    )


def make_cli_parser() -> argparse.ArgumentParser:
    """
    Make the CLI parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-dir",
        help="directory to save the filtered dataset.",
        type=Path,
        default=Path("./data/interim/data-split/smoke/FIGLIB_ANNOTATED_RESIZED/"),
    )
    parser.add_argument(
        "--dir-dataset",
        help="directory containing the pyro-sdis dataset.",
        type=Path,
        default=Path("./data/interim/filtered/smoke/FIGLIB_ANNOTATED_RESIZED/"),
    )
    parser.add_argument(
        "--random-seed",
        help="Random Seed to perform the data split",
        type=int,
        required=True,
        default=0,
    )
    parser.add_argument(
        "--ratio-train-val",
        help="Ratio for splitting train and val splits",
        type=float,
        default=0.9,
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
    if not args["dir_dataset"].exists():
        logging.error(
            f"invalid --dir-dataset, dir {args['dir_dataset']} does not exist"
        )
        return False
    else:
        return True


def list_directories(dir_dataset: Path) -> list[Path]:
    """
    List all directories in the `dir_dataset`.

    Returns:
        dirs (list[Path]): list of directories.
    """
    return [d for d in dir_dataset.iterdir() if d.is_dir()]


def filepath_image_to_filepath_label(filepath_image: Path) -> Path:
    """
    Given a filepath_image it returns its associated filepath_label.

    Returns:
        filepath_label (Path): the associated label filepath.
    """
    return filepath_image.parent / "labels" / f"{filepath_image.stem}.txt"


def make_data_split(
    dir_dataset: Path,
    random_seed: float,
    ratio_train_val: float,
) -> DataSplit:
    """
    Perform the data split for the FIGLIB_ANNOTATED_RESIZED dataset using the
    provided `random_seed` and the `ratio_train_val`

    __Note__: Prevent train/val leakage by splitting at the folder level.

    Returns:
        data_split (DataSplit): the data split.
    """
    rng = random.Random(random_seed)
    folders = list_directories(dir_dataset)
    folders_shuffled = rng.sample(folders, len(folders))
    number_folders = len(folders)

    filepaths_images_train = []
    filepaths_images_val = []

    for idx, folder in enumerate(folders_shuffled):
        filepaths_images = list(folder.glob("**/*.jpg"))
        if idx < number_folders * ratio_train_val:
            filepaths_images_train.extend(filepaths_images)
        else:
            filepaths_images_val.extend(filepaths_images)
    return DataSplit(train=filepaths_images_train, val=filepaths_images_val, test=[])


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
            shutil.copy(filepath_label, filepath_label_destination)


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
        dir_dataset = args["dir_dataset"]
        random_seed = args["random_seed"]
        ratio_train_val = args["ratio_train_val"]
        logger.info(f"filtering smokes and saving results in {save_dir}")
        save_dir.mkdir(parents=True, exist_ok=True)
        folders = list_directories(dir_dataset)
        logger.info(f"found {len(folders)} directories in {dir_dataset}")

        logger.info("split the data in train, val")
        data_split = make_data_split(
            dir_dataset=dir_dataset,
            random_seed=random_seed,
            ratio_train_val=ratio_train_val,
        )
        logger.info(
            f"datasplit: {len(data_split.train)} images in train - {len(data_split.val)} images in val"
        )
        logger.info(f"persist the data split in {save_dir}.")
        persist_data_split(data_split=data_split, save_dir=save_dir)
        exit(0)
