"""
CLI Script to analyze and make a reporting of the
generated dataset.
"""

import argparse
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any
from enum import Enum
from datetime import datetime
from pyro_dataset.utils import yaml_read


class DatasetOrigin(Enum):
    """
    All the possible dataset origins.

    Note: UNKNOW is used to account for new datasets that have not yet
    been added to this enum.
    """
    PYRONEAR = "pyronear"
    ADF = "adf"
    AWF = "awf"
    HPWREN = "hpwren"
    RANDOM_SMOKE = "random"
    UNKNOWN = "unknown"


class DataSplit(Enum):
    """
    All the possible datasplits.
    """

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass
class DetectionDetails:
    dataset_origin: DatasetOrigin
    details: dict[str, Any]


@dataclass
class SplitSummary:
    n_images: int
    n_labels: int
    n_background_images: int
    detection_details: list[DetectionDetails]


def make_cli_parser() -> argparse.ArgumentParser:
    """
    Make the CLI parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-dir",
        help="directory to save the dataset analysis",
        type=Path,
        default=Path("./data/reporting/wildfire/"),
    )
    parser.add_argument(
        "--filepath-data-yaml-train-val", help="filepath containing the data.yaml for the train and val splits of the dataset.",
        type=Path,
        default=Path("./data/processed/wildfire/data.yaml"),
    )
    parser.add_argument(
        "--filepath-data-yaml-test",
        help="filepath containing the data.yaml for the test split of the dataset.",
        type=Path,
        default=Path("./data/processed/wildfire_test/data.yaml"),
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
    if not args["filepath_data_yaml_train_val"].exists():
        logging.error(
            f"invalid --filepath-data-yaml-train-val, filepath {args['filepath_data_yaml_train_val']} does not exist"
        )
        return False
    elif not args["filepath_data_yaml_test"].exists():
        logging.error(
            f"invalid --filepath-data-yaml-test, filepath {args['filepath_data_yaml_test']} does not exist"
        )
        return False
    else:
        return True


def is_background(filepath_label: Path) -> bool:
    """
    Is the `filepath_label` a background image - no smokes in it.

    Returns:
        is_background? (bool): whether or not the filepath has a smoke detected in it.
    """
    return (
        filepath_label.exists()
        and filepath_label.is_file()
        and filepath_label.stat().st_size == 0
    )

def parse_details_pyronear(stem: str) -> dict[str, Any]:
    return {"test": "pyro"}

def parse_details_adf(stem: str) -> dict[str, Any]:
    return {"test": "adf"}

def parse_details_awf(stem: str) -> dict[str, Any]:
    return {"test": "awf"}

def parse_details_random_smoke(stem: str) -> dict[str, Any]:
    return {"test": "awf"}

def parse_details(stem: str, dataset_origin: DatasetOrigin) -> dict[str, Any]:
    match dataset_origin:
        case DatasetOrigin.PYRONEAR:
            return parse_details_pyronear(stem)
        case DatasetOrigin.ADF:
            return parse_details_adf(stem)
        case DatasetOrigin.AWF:
            return parse_details_awf(stem)
        case DatasetOrigin.HPWREN:
            return parse_details_awf(stem)
        case DatasetOrigin.RANDOM_SMOKE:
            return parse_details_random_smoke(stem)
        case _:
          return {}

def parse_dataset_origin(stem: str) -> DatasetOrigin:
    parts = stem.lower().split("_")
    origin_str = parts[0]
    if origin_str in [do.value for do in DatasetOrigin]:
        return DatasetOrigin(value=parts[0])
    # Fix the inconsitencies between the test dataset naming and the val/train naming
    elif origin_str in ["sdis-77", "sdis-07", "force-06", "marguerite-282", "pyro", "ardeche"]:
        return DatasetOrigin.PYRONEAR
    # Fix the inconsitencies between the test dataset naming and the val/train naming
    elif origin_str in ["axis", "2023"]:
        return DatasetOrigin.AWF
    else:
        print(f"unknow: {origin_str}")
        return DatasetOrigin.UNKNOWN


def parse_filepath_stem(stem: str) -> DetectionDetails:
    """
    Parse the filepath stem and return a Detection Details with as much
    extracted details as possible.
    """
    dataset_origin = parse_dataset_origin(stem=stem)
    dataset_details = parse_details(stem=stem, dataset_origin=dataset_origin)
    return DetectionDetails(
        dataset_origin=dataset_origin,
        details=dataset_details,
    )

def split_summary(filepath_data_yaml: Path, data_split: DataSplit) -> SplitSummary:
    """
    Create the split summary for a given data_yaml filepath and a DataSplit.
    """
    data_yaml = yaml_read(filepath_data_yaml)
    dir_images = filepath_data_yaml.parent / data_yaml[data_split.value]
    dir_labels = filepath_data_yaml.parent / data_yaml[data_split.value].replace("images", "labels")
    filepaths_images = list(dir_images.glob("*.jpg"))
    filepaths_labels = list(dir_labels.glob("*.txt"))
    filepaths_labels_background = [fp for fp in filepaths_labels if is_background(fp)]
    detection_details = [parse_filepath_stem(fp.stem) for fp in filepaths_images]

    return SplitSummary(
        n_images=len(filepaths_images),
        n_labels=len(filepaths_labels),
        n_background_images=len(filepaths_labels_background),
        detection_details=detection_details,
    )


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        exit(1)
    else:
        logging.info(args)
        filepath_data_yaml_train_val = args["filepath_data_yaml_train_val"]
        filepath_data_yaml_test = args["filepath_data_yaml_test"]
        data_yaml_train_val = yaml_read(filepath_data_yaml_train_val)
        data_yaml_test = yaml_read(filepath_data_yaml_test)
        logger.info(f"data_yaml test split: {data_yaml_test}")
        logger.info(f"data_yaml train and val splits: {data_yaml_train_val}")

        # Train
        print("TRAIN")
        split_summary_train = split_summary(filepath_data_yaml_train_val, data_split=DataSplit.TRAIN)
        # print(split_summary_train)
        # Val
        print("VAL")
        split_summary_val = split_summary(filepath_data_yaml_train_val, data_split=DataSplit.VAL)
        # print(split_summary_val)

        # Test
        print("TEST")
        split_summary_test = split_summary(filepath_data_yaml_test, data_split=DataSplit.TEST)
        # print(split_summary_test)

        # print(filepaths_train_images[:10])
        # detection_details = [parse_filepath_stem(fp.stem) for fp in filepaths_train_images]
        # print(detection_details[:10])
