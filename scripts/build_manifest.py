"""
CLI script to generate a manifest file for the generated datasets.

This script provides a command-line interface (CLI) for creating a manifest
file that contains metadata about the wildfire datasets used in training,
validation, and testing. The manifest includes statistics about the datasets
and information from the DVC lock file.

Usage:
    python build_manifest.py --save-dir <directory> \
        --dir-wildfire-dataset-train-val <train_val_directory> \
        --dir-wildfire-dataset-test <test_directory> \
        [--loglevel <level>]

Arguments:
    --save-dir: Directory to save the generated manifest file.
    --dir-wildfire-dataset-train-val: Directory containing the wildfire dataset
        used for training and validation.
    --dir-wildfire-dataset-test: Directory containing the wildfire test dataset.
    --loglevel: Set the logging level (default is 'info').

The generated manifest file will be saved as 'manifest.yaml' in the specified
save directory.
"""

import argparse
import logging
from pathlib import Path

from pyro_dataset.utils import yaml_read, yaml_write


def make_cli_parser() -> argparse.ArgumentParser:
    """
    Make the CLI parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-dir",
        help="directory to save the manifest.",
        type=Path,
        default=Path("./data/reporting/manifest/"),
    )
    parser.add_argument(
        "--dir-wildfire-dataset-train-val",
        help="directory containing the wildfire dataset used for training and validation.",
        type=Path,
        default=Path("./data/processed/wildfire/"),
    )
    parser.add_argument(
        "--dir-wildfire-dataset-test",
        help="directory containing the wildfire_test dataset.",
        type=Path,
        default=Path("./data/processed/wildfire_test/"),
    )
    parser.add_argument(
        "--dir-wildfire-temporal-dataset-train-val",
        help="directory containing the wildfire temporal dataset used for training and validation.",
        type=Path,
        default=Path("./data/processed/wildfire_temporal/"),
    )
    parser.add_argument(
        "--dir-wildfire-temporal-dataset-test",
        help="directory containing the wildfire temporal dataset used for testing.",
        type=Path,
        default=Path("./data/processed/wildfire_temporal_test/"),
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
    if not args["dir_wildfire_dataset_train_val"].exists():
        logging.error(
            f"invalid --dir-wildfire-dataset-train-val, does not exist {args['dir_wildfire_dataset_train_val']}"
        )
        return False
    elif not args["dir_wildfire_dataset_test"].exists():
        logging.error(
            f"invalid --dir-wildfire-dataset-test, does not exist {args['dir_wildfire_dataset_test']}"
        )
        return False
    if not args["dir_wildfire_temporal_dataset_train_val"].exists():
        logging.error(
            f"invalid --dir-wildfire-temporal-dataset-train-val, does not exist {args['dir_wildfire_temporal_dataset_train_val']}"
        )
        return False
    elif not args["dir_wildfire_temporal_dataset_test"].exists():
        logging.error(
            f"invalid --dir-wildfire-temporal-dataset-test, does not exist {args['dir_wildfire_temporal_dataset_test']}"
        )
        return False
    return True


def find_image_files(dir: Path) -> list[Path]:
    return list(dir.rglob("*.jpg"))


def find_sequence_folders(dir: Path) -> list[Path]:
    """
    Extract and return a list of sequence folders from annotated sequence directories.

    This function searches through the specified directory for subdirectories that
    contain sequence information. It includes directories that match the pattern '_sequence-<id>',
    where <id> is a numeric value.

    Args:
        dir_annotated_sequences (Path): The directory containing annotated sequence
        subdirectories to search for sequence folders.

    Returns:
        set[Path]: A set of unique Path objects representing the directories found in the directory.
    """
    return [
        seq_dir
        for seq_dir in dir.rglob("**/*")
        if seq_dir.is_dir()
        and "_sequence-" in seq_dir.name
        and seq_dir.name.split("_sequence-")[-1].isdigit()
    ]


def get_temporal_stats(dir_dataset: Path) -> dict:
    result = {}
    result = {"train": {}, "val": {}, "test": {}}
    dir_images = dir_dataset / "images"
    splits = [dir.name for dir in dir_images.iterdir() if dir.is_dir()]
    for split in splits:
        for class_str in ["smoke", "background"]:
            dir_split = dir_images / split / class_str
            dirs_sequences = find_sequence_folders(dir_split)
            filepaths_images = find_image_files(dir_split)
            file_count = len(filepaths_images)
            result[split][class_str] = {
                "number_images": file_count,
                "number_sequences": len(dirs_sequences),
            }
    return result


def find_empty_txt_filepaths(dir: Path) -> list[Path]:
    """
    Find and return a list of empty .txt file paths in the specified directory.

    Args:
        dir (Path): The directory to search for empty .txt files.

    Returns:
        list[Path]: A list of Path objects for empty .txt files found in the directory.
    """
    return [txt_file for txt_file in dir.rglob("*.txt") if txt_file.stat().st_size == 0]


def get_stats(dir_dataset: Path) -> dict:
    """
    Return a dictionnary containg stats about the different splits.
    """
    result = {}
    dir_images = dir_dataset / "images"
    splits = [dir.name for dir in dir_images.iterdir() if dir.is_dir()]
    for split in splits:
        dir_split = dir_images / split
        filepaths_images = find_image_files(dir_split)
        file_count = len(filepaths_images)
        filepaths_empty_txt = find_empty_txt_filepaths(
            Path(str(dir_split).replace("images", "labels"))
        )
        n_backgrounds = len(filepaths_empty_txt)
        n_smokes = file_count - n_backgrounds
        result[split] = {
            "number_images": file_count,
            "number_backgrounds": n_backgrounds,
            "number_smokes": n_smokes,
            "ratio_backgrounds": n_backgrounds / (n_backgrounds + n_smokes),
            "ratio_smokes": n_smokes / (n_backgrounds + n_smokes),
        }
    return result


def make_data_manifest(
    dir_wildfire_dataset_train_val: Path,
    dir_wildfire_dataset_test: Path,
    dir_wildfire_temporal_dataset_train_val: Path,
    dir_wildfire_temporal_dataset_test: Path,
) -> dict:
    """
    Make the data manifest.yaml file.

    It is meant to persist as much information as possible about the dataset
    inputs.
    """
    filepath_dvc_lock = Path("./dvc.lock")
    dvc_lock = yaml_read(filepath_dvc_lock)
    stats_for_datasets = {
        str(dir): get_stats(dir)
        for dir in [
            dir_wildfire_dataset_train_val,
            dir_wildfire_dataset_test,
        ]
    }
    stats_for_temporal_datasets = {
        str(dir): get_temporal_stats(dir)
        for dir in [
            dir_wildfire_temporal_dataset_test,
            dir_wildfire_temporal_dataset_train_val,
        ]
    }

    return {
        "stats": {**stats_for_datasets, **stats_for_temporal_datasets},
        "dvc_lock": dvc_lock,
    }


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        exit(1)
    else:
        logger.info(args)
        dir_wildfire_dataset_train_val = args["dir_wildfire_dataset_train_val"]
        dir_wildfire_dataset_test = args["dir_wildfire_dataset_test"]
        dir_wildfire_temporal_dataset_test = args["dir_wildfire_temporal_dataset_test"]
        dir_wildfire_temporal_dataset_train_val = args[
            "dir_wildfire_temporal_dataset_train_val"
        ]

        save_dir = args["save_dir"]
        save_dir.mkdir(parents=True, exist_ok=True)
        filepath_manifest = save_dir / "manifest.yaml"
        data_manifest = make_data_manifest(
            dir_wildfire_dataset_train_val=dir_wildfire_dataset_train_val,
            dir_wildfire_dataset_test=dir_wildfire_dataset_test,
            dir_wildfire_temporal_dataset_train_val=dir_wildfire_temporal_dataset_train_val,
            dir_wildfire_temporal_dataset_test=dir_wildfire_temporal_dataset_test,
        )
        logger.info(f"saving manifest.yaml file in {save_dir}")
        yaml_write(to=filepath_manifest, data=data_manifest)
        exit(0)
