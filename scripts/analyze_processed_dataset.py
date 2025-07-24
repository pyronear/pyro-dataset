"""
CLI Script to analyze and generate reports for the processed wildfire dataset.

This script provides functionalities to:
- Parse command-line arguments for input and output file paths.
- Compute file content hashes for images in different dataset splits (train, val, test).
- Summarize dataset statistics, including counts of images, labels, and background images.
- Detect potential data leakage between dataset splits.
- Generate visual plots to analyze the dataset breakdown by origin, year, and month.

Usage:
    python analyze_processed_dataset.py [OPTIONS]

Options:
    --save-dir                Directory to save the dataset analysis.
    --filepath-data-yaml-train-val
                             Path to the data.yaml file for train and val splits.
    --filepath-data-yaml-test  Path to the data.yaml file for the test split.
    -log, --loglevel          Set the logging level (default: info).
"""

import argparse
import hashlib
import logging
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from bokeh.io import show
from bokeh.plotting import output_file
from tqdm import tqdm

from pyro_dataset.plots.report import (
    make_figure_for_data_splits_breakdown,
    make_figure_for_data_splits_month_breakdown,
    make_figure_for_data_splits_year_breakdown,
    make_figure_for_ratio_background_images,
    make_plot_data_for_data_splits_breakdown,
    make_plot_data_for_data_splits_month_breakdown,
    make_plot_data_for_data_splits_year_breakdown,
    make_plot_data_for_ratio_background_images,
)
from pyro_dataset.utils import yaml_read, yaml_write
import pyro_dataset.parsers as parsers


class DataSplit(Enum):
    """
    All the possible datasplits.
    """

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass
class DataLeakageSummary:
    """
    Simple dataclass to represent data leakage summary.
    """

    train_val_file_content: list[Path]
    train_test_file_content: list[Path]
    val_test_file_content: list[Path]


@dataclass
class SplitSummary:
    """
    Generic Split Summary for a given split.
    """

    children: list["SplitSummary"]
    name: str
    n_images: int
    n_labels: int
    n_background_images: int
    detection_details: list[parsers.DetectionDetails]
    frequencies_origins: Counter[parsers.DatasetOrigin]
    frequencies_years: Counter[int]
    frequencies_months: Counter[int]
    frequencies_year_months: Counter[str]

    def __repr__(self):
        return f"""SplitSummary(
        name={self.name},
        n_images={self.n_images},
        n_labels={self.n_labels},
        n_background_images={self.n_background_images},
        children={self.children})"""


@dataclass
class SplitFilepaths:
    """
    list the different Filepaths for a given split.
    """

    images: list[Path]
    labels: list[Path]
    labels_background: list[Path]


def normalize_frequencies(freqs: dict[Any, int]) -> dict[Any, float]:
    """
    Normalize a frequency dict.
    """
    total = sum(v for v in freqs.values())
    return {k: v / total for k, v in freqs.items()}


def compute_file_content_hash(filepath: Path) -> str:
    """
    Compute the file content hash of the `filepath`.

    Returns:
        hexdigest (str)
    """
    # Create a hash object
    hash_sha256 = hashlib.sha256()

    # Open the file in binary mode
    with open(filepath, "rb") as f:
        # Read the file in chunks to avoid using too much memory
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)

    # Return the hexadecimal digest of the hash
    return hash_sha256.hexdigest()


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
        "--filepath-data-yaml-train-val",
        help="filepath containing the data.yaml for the train and val splits of the dataset.",
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


def get_summary_frequencies(
    detection_details: list[parsers.DetectionDetails],
) -> dict[str, Counter]:
    """
    Return a dict mapping keys to frequencies Counter.

    Returns:
      frequencies_year_months (Counter)
      frequencies_origins (Counter)
      frequencies_years (Counter)
      frequencies_months (Counter)
    """
    frequencies_origins = Counter([dd.dataset_origin for dd in detection_details])
    frequencies_year_months = Counter(
        [
            dd.details["datetime"].strftime("%Y-%m")
            for dd in detection_details
            if dd.details["datetime"]
        ]
    )
    frequencies_years = Counter(
        [
            dd.details["datetime"].year
            for dd in detection_details
            if dd.details["datetime"]
        ]
    )
    frequencies_months = Counter(
        [
            dd.details["datetime"].month
            for dd in detection_details
            if dd.details["datetime"]
        ]
    )
    return {
        "frequencies_origins": frequencies_origins,
        "frequencies_months": frequencies_months,
        "frequencies_years": frequencies_years,
        "frequencies_year_months": frequencies_year_months,
    }


def split_filepaths(filepath_data_yaml: Path, data_split: DataSplit) -> SplitFilepaths:
    data_yaml = yaml_read(filepath_data_yaml)
    dir_images = filepath_data_yaml.parent / data_yaml[data_split.value]
    dir_labels = filepath_data_yaml.parent / data_yaml[data_split.value].replace(
        "images", "labels"
    )
    filepaths_images = list(dir_images.glob("*.jpg"))
    filepaths_labels = list(dir_labels.glob("*.txt"))
    filepaths_labels_background = [fp for fp in filepaths_labels if is_background(fp)]
    return SplitFilepaths(
        images=filepaths_images,
        labels=filepaths_images,
        labels_background=filepaths_labels_background,
    )


def split_summary(filepath_data_yaml: Path, data_split: DataSplit) -> SplitSummary:
    """
    Create the split summary for a given data_yaml filepath and a DataSplit.
    """
    data_yaml = yaml_read(filepath_data_yaml)
    dir_labels = filepath_data_yaml.parent / data_yaml[data_split.value].replace(
        "images", "labels"
    )
    filepaths = split_filepaths(
        filepath_data_yaml=filepath_data_yaml,
        data_split=data_split,
    )
    detection_details = [
        parsers.parse_filepath(fp) for fp in filepaths.images
    ]
    frequencies = get_summary_frequencies(detection_details=detection_details)

    children = []
    for dataset_origin, _ in frequencies["frequencies_origins"].most_common(n=5):
        detection_details_filtered = [
            dd for dd in detection_details if dd.dataset_origin == dataset_origin
        ]
        filepaths_labels_filtered = [
            dir_labels / f"{dd.filepath.stem}.txt"
            for dd in detection_details_filtered
            if (dir_labels / f"{dd.filepath.stem}.txt").exists()
        ]
        filepaths_labels_background_filtered = [
            fp for fp in filepaths_labels_filtered if is_background(fp)
        ]
        frequencies_filtered = get_summary_frequencies(
            detection_details=detection_details_filtered
        )
        sub_split_summary = SplitSummary(
            name=dataset_origin.value,
            children=[],
            n_images=len(detection_details_filtered),
            n_labels=len(filepaths_labels_filtered),
            n_background_images=len(filepaths_labels_background_filtered),
            detection_details=detection_details_filtered,
            frequencies_origins=frequencies_filtered["frequencies_origins"],
            frequencies_years=frequencies_filtered["frequencies_years"],
            frequencies_months=frequencies_filtered["frequencies_months"],
            frequencies_year_months=frequencies_filtered["frequencies_year_months"],
        )
        children.append(sub_split_summary)

    return SplitSummary(
        name="all",
        children=children,
        n_images=len(filepaths.images),
        n_labels=len(filepaths.labels),
        n_background_images=len(filepaths.labels_background),
        detection_details=detection_details,
        frequencies_origins=frequencies["frequencies_origins"],
        frequencies_years=frequencies["frequencies_years"],
        frequencies_months=frequencies["frequencies_months"],
        frequencies_year_months=frequencies["frequencies_year_months"],
    )


def split_summary_to_dict(split_summary: SplitSummary) -> dict:
    """
    Turn a split summary into a python dictionnary that can be saved as yaml
    for instance.
    """
    result = {}
    result["origin"] = split_summary.name
    result["statistics"] = {
        "n_images": split_summary.n_images,
        "n_labels": split_summary.n_labels,
        "n_background_images": split_summary.n_background_images,
    }
    result["frequencies"] = {
        "origins": dict(
            sorted(
                [
                    (dataset_origin.value, count)
                    for dataset_origin, count in split_summary.frequencies_origins.items()
                ],
                key=lambda e: e[1],
                reverse=True,
            )
        ),
        "years": dict(
            sorted(
                split_summary.frequencies_years.items(),
                key=lambda e: e[1],
                reverse=True,
            )
        ),
        "months": dict(
            sorted(
                split_summary.frequencies_months.items(),
                key=lambda e: e[1],
                reverse=True,
            )
        ),
        "year_months": dict(
            sorted(
                split_summary.frequencies_year_months.items(),
                key=lambda e: e[1],
                reverse=True,
            )
        ),
    }
    result["sub_splits"] = []

    for child in split_summary.children:
        result_child = split_summary_to_dict(split_summary=child)
        result["sub_splits"].append(result_child.copy())

    if len(result["sub_splits"]) == 0:
        del result["sub_splits"]

    return result


def to_data_leakage_summary(hashes: dict[str, dict]) -> DataLeakageSummary:
    hashes_file_content_leakage_train_val = set(hashes["train"].keys()).intersection(
        set(hashes["val"].keys())
    )
    hashes_file_content_leakage_train_test = set(hashes["train"].keys()).intersection(
        set(hashes["test"].keys())
    )
    hashes_file_content_leakage_val_test = set(hashes["val"].keys()).intersection(
        set(hashes["test"].keys())
    )
    train_val_file_content = [
        v
        for k, v in hashes["train"].items()
        if k in hashes_file_content_leakage_train_val
    ] + [
        v
        for k, v in hashes["test"].items()
        if k in hashes_file_content_leakage_train_val
    ]

    # TODO: finish implementing leakage from other splits
    return DataLeakageSummary(
        train_val_file_content=train_val_file_content,
        train_test_file_content=[],
        val_test_file_content=[],
    )


def write_report_yaml(
    split_summary_train: SplitSummary,
    split_summary_val: SplitSummary,
    split_summary_test: SplitSummary,
    filepath_output_yaml: Path,
    data_leakage_summary: DataLeakageSummary,
) -> None:
    dict_train = split_summary_to_dict(split_summary_train)
    dict_val = split_summary_to_dict(split_summary_val)
    dict_test = split_summary_to_dict(split_summary_test)
    dict_data = {
        "data_leakage": {
            "train_val": {
                "file_content_hash_check": {
                    "is_leakage": len(data_leakage_summary.train_val_file_content) > 0,
                    "filepaths": data_leakage_summary.train_val_file_content,
                }
            },
            "train_test": {
                "file_content_hash_check": {
                    "is_leakage": len(data_leakage_summary.train_test_file_content) > 0,
                    "filepaths": data_leakage_summary.train_test_file_content,
                }
            },
            "val_test": {
                "file_content_hash_check": {
                    "is_leakage": len(data_leakage_summary.val_test_file_content) > 0,
                    "filepaths": data_leakage_summary.val_test_file_content,
                }
            },
        },
        "summary": {
            "split": {
                "train": dict_train,
                "val": dict_val,
                "test": dict_test,
            },
        },
    }
    filepath_output_yaml.parent.mkdir(parents=True, exist_ok=True)
    yaml_write(to=filepath_output_yaml, data=dict_data)


def compute_all_hashes(
    filepath_data_yaml_train_val: Path,
    filepath_data_yaml_test: Path,
) -> dict[str, dict]:
    """
    Compute all file content hashes for the images in the different splits.

    Returns:
        train (dict[str, Path]): hashes for the train images.
        val (dict[str, Path]): hashes for the val images.
        test (dict[str, Path]): hashes for the test images.
    """
    filepaths_train = split_filepaths(
        filepath_data_yaml=filepath_data_yaml_train_val,
        data_split=DataSplit.TRAIN,
    )
    filepaths_val = split_filepaths(
        filepath_data_yaml=filepath_data_yaml_train_val,
        data_split=DataSplit.VAL,
    )
    filepaths_test = split_filepaths(
        filepath_data_yaml=filepath_data_yaml_test,
        data_split=DataSplit.TEST,
    )
    logging.info(
        f"compute file content hash for {len(filepaths_train.images)} images in train split"
    )
    hashes_train = {
        compute_file_content_hash(fp): fp for fp in tqdm(filepaths_train.images)
    }
    logging.info(
        f"compute file content hash for {len(filepaths_val.images)} images in val split"
    )
    hashes_val = {
        compute_file_content_hash(fp): fp for fp in tqdm(filepaths_val.images)
    }
    logging.info(
        f"compute file content hash for {len(filepaths_test.images)} images in test split"
    )
    hashes_test = {
        compute_file_content_hash(fp): fp for fp in tqdm(filepaths_test.images)
    }

    return {
        "train": hashes_train,
        "val": hashes_val,
        "test": hashes_test,
    }


def make_analysis_plots(filepath_report_yaml: Path) -> None:
    """
    Generate the different plots for analyzing the generated datasets.
    """

    logging.info(f"Loading the report to generate visual plots {filepath_report_yaml}")
    report = yaml_read(filepath_report_yaml)

    filepath_data_splits_origin_breakdown = (
        save_dir / "plots" / "data_splits_origin_breakdown.html"
    )
    filepath_data_splits_origin_breakdown.parent.mkdir(parents=True, exist_ok=True)
    logging.info(
        f"Generating plot for data splits origin breakdown in {filepath_data_splits_origin_breakdown}"
    )
    origins = ["pyronear", "hpwren", "awf", "random", "adf", "unknown"]
    data = make_plot_data_for_data_splits_breakdown(
        report=report,
        origins=origins,
    )
    figure_data_splits_breakdown = make_figure_for_data_splits_breakdown(data=data)
    output_file(filepath_data_splits_origin_breakdown)
    show(figure_data_splits_breakdown)

    filepath_image_ratios_breakdown = (
        save_dir / "plots" / "data_splits_background_images_ratios_breakdown.html"
    )
    logging.info(
        f"Generating plot for ratio image/background breakdown in {filepath_image_ratios_breakdown}"
    )
    data = make_plot_data_for_ratio_background_images(report)
    figure_image_ratios_breakdown = make_figure_for_ratio_background_images(data)
    output_file(filepath_image_ratios_breakdown)
    show(figure_image_ratios_breakdown)

    filepath_years_breakdown = save_dir / "plots" / "data_splits_years_breakdown.html"
    logging.info(f"Generating plot for years breakdown in {filepath_years_breakdown}")
    data = make_plot_data_for_data_splits_year_breakdown(report)
    figure_year_breakdown = make_figure_for_data_splits_year_breakdown(data)
    output_file(filepath_years_breakdown)
    show(figure_year_breakdown)

    filepath_months_breakdown = save_dir / "plots" / "data_splits_months_breakdown.html"
    logging.info(f"Generating plot for months breakdown in {filepath_months_breakdown}")
    data = make_plot_data_for_data_splits_month_breakdown(report)
    figure_month_breakdown = make_figure_for_data_splits_month_breakdown(data)
    output_file(filepath_months_breakdown)
    show(figure_month_breakdown)


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
        filepath_data_yaml_train_val = args["filepath_data_yaml_train_val"]
        filepath_data_yaml_test = args["filepath_data_yaml_test"]
        data_yaml_train_val = yaml_read(filepath_data_yaml_train_val)
        data_yaml_test = yaml_read(filepath_data_yaml_test)
        logger.info(f"data_yaml test split: {data_yaml_test}")
        logger.info(f"data_yaml train and val splits: {data_yaml_train_val}")

        split_summary_train = split_summary(
            filepath_data_yaml_train_val, data_split=DataSplit.TRAIN
        )
        split_summary_val = split_summary(
            filepath_data_yaml_train_val, data_split=DataSplit.VAL
        )
        split_summary_test = split_summary(
            filepath_data_yaml_test, data_split=DataSplit.TEST
        )
        logger.info(f"train split summary: {split_summary_train}")
        logger.info(f"val split summary: {split_summary_val}")
        logger.info(f"test split summary: {split_summary_test}")

        hashes = compute_all_hashes(
            filepath_data_yaml_train_val=filepath_data_yaml_train_val,
            filepath_data_yaml_test=filepath_data_yaml_test,
        )

        data_leakage_summary = to_data_leakage_summary(hashes=hashes)
        logger.info(f"data leakage summary: {data_leakage_summary}")

        filepath_output_yaml = save_dir / "report.yaml"
        logger.info(f"Saving report in {filepath_output_yaml}")

        write_report_yaml(
            split_summary_train=split_summary_train,
            split_summary_val=split_summary_val,
            split_summary_test=split_summary_test,
            filepath_output_yaml=filepath_output_yaml,
            data_leakage_summary=data_leakage_summary,
        )

        logger.info(
            f"Make some visualization plots based on the report.yaml file {filepath_output_yaml}"
        )
        make_analysis_plots(filepath_report_yaml=filepath_output_yaml)


## REPL DRIVEN
# args = {
#     "save_dir": Path("data/reporting/wildfire"),
#     "filepath_data_yaml_train_val": Path("data/processed/wildfire/data.yaml"),
#     "filepath_data_yaml_test": Path("data/processed/wildfire_test/data.yaml"),
#     "loglevel": "info",
# }
#
# save_dir = args["save_dir"]
# filepath_data_yaml_train_val = args["filepath_data_yaml_train_val"]
# filepath_data_yaml_test = args["filepath_data_yaml_test"]
# data_yaml_train_val = yaml_read(filepath_data_yaml_train_val)
# data_yaml_test = yaml_read(filepath_data_yaml_test)
#
# data_yaml_train_val
# data_yaml_test
#
# split_summary_train = split_summary(
#     filepath_data_yaml_train_val, data_split=DataSplit.TRAIN
# )
# split_summary_val = split_summary(
#     filepath_data_yaml_train_val, data_split=DataSplit.VAL
# )
# split_summary_test = split_summary(filepath_data_yaml_test, data_split=DataSplit.TEST)
# split_summary_test
#
# parse_details(
#     stem="pyronear_valbonne_3_2023_11_02T07_05_56.jpg",
#     dataset_origin=DatasetOrigin.PYRONEAR,
# )
# parse_details(
#     stem="pyronear_serre-de-barre-310_2024-09-02T13-12-23.jpg",
#     dataset_origin=DatasetOrigin.PYRONEAR,
# )
# parse_details(
#     stem="pyronear_serre-de-barre-250_2024-08-25T11-19-22.jpg",
#     dataset_origin=DatasetOrigin.PYRONEAR,
# )
#
# s1 = "pyronear_valbonne_3_2023_11_02T07_05_56.jpg"
# s2 = "pyronear_serre-de-barre-310_2024-09-02T13-12-23.jpg"
# s3 = "pyronear_serre-de-barre-250_2024-08-25T11-19-22.jpg"
# s4 = "Pyronear_test_DS_00000328.jpg"
# s5 = "pyronear_st_peray_2_2023_07_12T13_24_14.jpg"
#
# parse_pyronear_camera_details(s1)
# parse_pyronear_camera_details(s2)
# parse_pyronear_camera_details(s3)
# parse_pyronear_camera_details(s4)
# parse_pyronear_camera_details(s5)
