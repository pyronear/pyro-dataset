"""
CLI script to convert the data split from split_selected_sequences.py into an ultralytics dataset.

This script is designed to generate a dataset in the ultralytics format, which is commonly used for
training and evaluating object detection models. It organizes images and their corresponding label
files based on specified parameters, allowing for easy integration with ultralytics training pipelines.

Usage:
    python make_ultralytics_dataset.py --dir-save <path_to_save_dataset> --dir-data-split <path_to_data_split> --loglevel <log_level>

Arguments:
    --dir-save: Directory to save the ultralytics dataset (default: ./data/interim/pyronear-platform/sequences-ultralytics/).
    --dir-data-split: Directory for the sequences data split (default: ./data/interim/pyronear-platform/sequences-data-split/).
    -log, --loglevel: Provide logging level (default: info).
"""

import argparse
import logging
import shutil
from pathlib import Path

from pyro_dataset.yolo.utils import (
    annotation_to_label_txt,
    parse_yolo_prediction_txt_file,
    yolo_prediction_to_annotation,
)


def make_cli_parser() -> argparse.ArgumentParser:
    """
    Make the CLI parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir-save",
        help="Directory to save the ultralytics dataset",
        type=Path,
        default=Path("./data/interim/pyronear-platform/sequences-ultralytics/"),
    )
    parser.add_argument(
        "--dir-data-split",
        help="Directory for the sequences-data-split",
        type=Path,
        default=Path("./data/interim/pyronear-platform/sequences-data-split/"),
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
    if not args["dir_data_split"].exists():
        logging.error(f"Invalid --dir-data-split, directory does not exist")
        return False
    return True


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


def find_false_positive_folders(dir: Path) -> list[Path]:
    """
    Find and return a list of nested directories that end with 'false-positives'.

    Args:
        dir_annotated_sequences (Path): The directory to search for false-positive folders.

    Returns:
        list[Path]: A list of Path objects representing the directories that end with 'false-positives'.
    """
    return [
        seq_dir
        for seq_dir in dir.rglob("*")
        if seq_dir.is_dir() and seq_dir.name.endswith("false-positives")
    ]


def find_true_positive_folders(dir: Path) -> list[Path]:
    """
    Find and return a list of nested directories that end with 'true-positives'.

    Args:
        dir_annotated_sequences (Path): The directory to search for true-positive folders.

    Returns:
        list[Path]: A list of Path objects representing the directories that end with 'true-positives'.
    """
    return [
        seq_dir
        for seq_dir in dir.rglob("*")
        if seq_dir.is_dir() and seq_dir.name.endswith("true-positives")
    ]


def find_train_val_test_dirs_sequences(dir: Path) -> dict[str, list[Path]]:
    """
    Find and categorize sequence folders into training, validation, and test sets.

    This function searches for sequence folders within the specified directory and categorizes them
    based on their naming convention. It identifies folders that are specifically designated for
    training, validation, and testing. If any sequence folders do not fit into these categories,
    a warning is logged.

    Args:
        dir (Path): The directory containing the sequence folders to categorize.

    Returns:
        dict[str, list[Path]]: A dictionary with keys 'train', 'val', and 'test', each containing a
        list of Path objects representing the corresponding sequence folders found in the directory.
    """
    dirs_sequences = find_sequence_folders(dir=dir)
    train_xs = [seq for seq in dirs_sequences if "train" in str(seq)]
    val_xs = [seq for seq in dirs_sequences if "val" in str(seq)]
    test_xs = [seq for seq in dirs_sequences if "test" in str(seq)]

    for seq in dirs_sequences:
        if "train" not in str(seq) and "val" not in str(seq) and "test" not in str(seq):
            logging.warning(f"No split for sequence {seq}")

    return {
        "train": train_xs,
        "val": val_xs,
        "test": test_xs,
    }


def read_file_content(filepath: Path) -> str:
    """
    Read the content of a file and return it as a string.

    Args:
        filepath (Path): The path to the file to be read.

    Returns:
        str: The content of the file as a string.
    """
    with open(filepath, "r", encoding="utf-8") as file:
        content = file.read()
    return content


def make_ultralytics_format(
    dir_save: Path,
    split: str,
    dir_sequence: Path,
    is_background: bool,
) -> dict:
    """
    Convert sequence data into the ultralytics format.

    This function prepares the dataset in the ultralytics format by organizing
    images and their corresponding label files based on the specified parameters.
    It reads YOLO prediction files, converts them to annotations, and structures
    the output for both images and labels.

    Args:
        dir_save (Path): The directory where the ultralytics formatted dataset will be saved.
        split (str): The dataset split type (e.g., 'train', 'val', 'test').
        dir_sequence (Path): The directory containing the sequence data to be processed.
        is_background (bool): Flag indicating whether the sequences are background (false positives).

    Returns:
        dict: A dictionary containing the formatted dataset with images and labels.
    """
    filepaths_labels = list(dir_sequence.glob("**/*.txt"))
    filepaths_images = [
        fp for fp in dir_sequence.glob("**/*.jpg") if "images" in str(fp)
    ]

    dict_labels = {}
    for filepath_label in filepaths_labels:
        if is_background:
            dict_labels[filepath_label] = ""
        else:
            yolo_prediction = parse_yolo_prediction_txt_file(
                read_file_content(filepath=filepath_label)
            )
            yolo_annotation = yolo_prediction_to_annotation(
                yolo_prediction=yolo_prediction
            )
            label_content = annotation_to_label_txt(yolo_annotation=yolo_annotation)
            dict_labels[filepath_label] = label_content

    return {
        "images": [
            {
                "filepath_source": fp,
                "filepath_destination": dir_save / "images" / split / fp.name,
            }
            for fp in filepaths_images
        ],
        "labels": [
            {
                "filepath_destination": dir_save / "labels" / split / fp.name,
                "content": content,
            }
            for fp, content in dict_labels.items()
        ],
    }


def run(dict_ultralytics_format: dict) -> None:
    """
    Copy images and write label files to the specified destination.

    This function processes the given dictionary containing information about
    images and labels in the ultralytics format. It copies the source images
    to their corresponding destination paths and writes the content of the
    labels to the specified label files.

    Args:
        dict_ultralytics_format (dict): A dictionary containing 'images' and 'labels'
        with their respective file paths and content.
    """
    for record in dict_ultralytics_format["images"]:
        filepath_source = record["filepath_source"]
        filepath_destination = record["filepath_destination"]
        filepath_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src=filepath_source, dst=filepath_destination)

    for record in dict_ultralytics_format["labels"]:
        filepath_destination = record["filepath_destination"]
        content = record["content"]
        filepath_destination.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath_destination, "w") as f:
            f.write(content)


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        exit(1)
    else:
        dir_save = args["dir_save"]
        dir_data_split = args["dir_data_split"]

        dirs_fp = find_false_positive_folders(dir=dir_data_split)
        dirs_tp = find_true_positive_folders(dir=dir_data_split)

        dir_save.mkdir(parents=True, exist_ok=True)

        logger.info(f"Handling the false positives and setting them as backgrounds")
        for dir_fp in dirs_fp:
            dict_dirs_sequences = find_train_val_test_dirs_sequences(dir=dir_fp)
            for split in dict_dirs_sequences.keys():
                for dir_sequence in dict_dirs_sequences[split]:
                    run(
                        make_ultralytics_format(
                            dir_save=dir_save,
                            split=split,
                            dir_sequence=dir_sequence,
                            is_background=True,  # Set to true for False Positives
                        )
                    )

        logger.info(f"Handling the true positives")
        for dir_tp in dirs_tp:
            dict_dirs_sequences = find_train_val_test_dirs_sequences(dir=dir_tp)
            for split in dict_dirs_sequences.keys():
                for dir_sequence in dict_dirs_sequences[split]:
                    run(
                        make_ultralytics_format(
                            dir_save=dir_save,
                            split=split,
                            dir_sequence=dir_sequence,
                            is_background=False,  # Set to False for True Positives
                        )
                    )
