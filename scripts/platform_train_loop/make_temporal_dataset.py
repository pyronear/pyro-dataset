"""
CLI script for creating a temporal dataset from the Pyronear sequences.

This script processes annotated Pyronear sequences, organizes them into
specified directories for images and labels based on the dataset split
(train, validation, test), and handles both smoke and background sequences.
It supports command-line arguments for configuration, including paths for
saving the dataset and specifying the data splits.

Usage:
    python make_temporal_dataset.py [OPTIONS]

Options:
    --save-dir <path>                          Directory to save the temporal dataset.
    --dir-sequences-data-split <path>          Directory containing the data splits of the Pyronear sequences.
    --dir-platform-annotated-sequences <path>  Directory containing the raw annotated Pyronear sequences.
    --loglevel <level>                         Logging level to set for the script execution (default: info).
    -h, --help                                 Show this help message and exit.

Arguments
    --save-dir: Directory to save the temporal dataset.
    --dir-sequences-data-split: Directory containing the data splits of the Pyronear sequences.
    --dir-platform-annotated-sequences: Directory containing the raw annotated Pyronear sequences.
    --loglevel: Logging level to set for the script execution.
"""

import argparse
import logging
import shutil
from pathlib import Path

from tqdm import tqdm


def make_cli_parser() -> argparse.ArgumentParser:
    """
    Make the CLI parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir-save",
        help="directory to save the temporal dataset build from the pyronear sequences.",
        type=Path,
        default=Path("./data/interim/pyronear-platform/sequences-temporal"),
    )
    parser.add_argument(
        "--dir-sequences-data-split",
        help="directory containing the data splits of the pyronear sequences.",
        type=Path,
        default=Path("./data/interim/pyronear-platform/sequences-data-split/"),
    )
    parser.add_argument(
        "--dir-platform-annotated-sequences",
        help="directory containing the raw annotated pyronear sequences.",
        type=Path,
        default=Path("./data/raw/pyronear-platform-annotated-sequences/"),
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
    if not args["dir_sequences_data_split"].exists():
        logging.error(
            f"invalid --dir-sequences-data-split, dir {args['dir_sequences_data_split']} does not exist"
        )
        return False

    if not args["dir_platform_annotated_sequences"].exists():
        logging.error(
            f"invalid --dir-platform-annotated-sequences, dir {args['dir_platform_annotated_sequences']} does not exist"
        )
        return False

    return True


def find_dir_annotated_sequence_by_reference(
    sequence_reference: str,
    dir: Path,
) -> None | Path:
    """
    Find the directory of the annotated sequence by its reference name.

    Parameters:
        sequence_reference (str): The reference name of the sequence to search for.
        dir (Path): The directory path to search within.

    Returns:
        None | Path: Returns the path of the annotated sequence directory if found, otherwise None.
    """
    result = [
        seq_dir
        for seq_dir in dir.rglob("**/*")
        if seq_dir.is_dir() and seq_dir.name == sequence_reference
    ]

    if result:
        return result[0]
    else:
        return None


def find_dirs_sequences_for_split(dir: Path, split: str) -> list[Path]:
    """
    Find directories containing sequences for a specific dataset split.

    Parameters:
        dir (Path): The directory path to search within for sequences.
        split (str): The dataset split to filter directories by (e.g., 'train', 'val', 'test').

    Returns:
        list[Path]: A list of Path objects representing the directories found for the specified split.
    """
    dirs_split = [
        seq_dir
        for seq_dir in dir.rglob("**/*")
        if seq_dir.is_dir() and seq_dir.name.endswith(split)
    ]

    return [path for dir in dirs_split for path in dir.glob("*") if dir.is_dir()]


def get_dirs(
    is_smoke: bool,
    dir_sequence: Path,
    split: str,
) -> dict[str, Path]:
    """
    Generate directories for images and labels based on the sequence type and split.

    Parameters:
      is_smoke (bool): Flag indicating if the sequence is a smoke sequence.
      dir_sequence (Path): The directory path for the sequence.
      split (str): The dataset split (e.g., train, val, test).

    Returns:
      dict[str, Path]: A dictionary containing paths for images and labels.
    """

    class_str = "smoke" if is_smoke else "background"
    dir_sequence_images = dir_save / "images" / split / class_str / dir_sequence.name
    dir_sequence_labels = dir_save / "labels" / split / class_str / dir_sequence.name

    return {
        "dir_images": dir_sequence_images,
        "dir_labels": dir_sequence_labels,
    }


def handle_dir_sequence(dir_sequence: Path, split: str, is_smoke: bool = True) -> None:
    """
    Handle the processing of a single sequence directory for a given dataset split.

    This function locates the corresponding annotated sequence directory, creates
    necessary directories for images and labels, and copies the relevant files from
    the annotated sequence directory to the appropriate locations based on the split.

    Parameters:
        dir_sequence (Path): The directory path of the sequence to handle.
        split (str): The dataset split to process (e.g., 'train', 'val', 'test').
        is_smoke (bool): Flag indicating if the sequence is related to smoke (default is True).
    """

    sequence_reference = dir_sequence.name

    dir_annotated_sequence = find_dir_annotated_sequence_by_reference(
        sequence_reference=sequence_reference,
        dir=dir_platform_annotated_sequences,
    )

    if dir_annotated_sequence:
        dirs = get_dirs(
            is_smoke=is_smoke,
            dir_sequence=dir_annotated_sequence,
            split=split,
        )
        dir_sequence_images = dirs["dir_images"]
        dir_sequence_labels = dirs["dir_labels"]
        dir_sequence_images.mkdir(exist_ok=True, parents=True)
        dir_sequence_labels.mkdir(exist_ok=True, parents=True)

        for filepath_image in (dir_annotated_sequence / "images").glob("*.jpg"):
            shutil.copy(
                src=filepath_image,
                dst=dir_sequence_images / filepath_image.name,
            )

            # FIXME: this should look into ground truth instead of labels
            for filepath_label in (dir_annotated_sequence / "labels").glob("*.txt"):
                if is_smoke:
                    shutil.copy(
                        src=filepath_label,
                        dst=dir_sequence_labels / filepath_label.name,
                    )
                else:
                    (dir_sequence_labels / filepath_label.name).touch()


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        exit(1)
    else:
        logging.info(args)
        dir_save = args["dir_save"]
        dir_platform_annotated_sequences = args["dir_platform_annotated_sequences"]
        dir_sequences_data_split = args["dir_sequences_data_split"]

        for split in ["train", "val", "test"]:
            logger.info(f"Generating the {split} split")
            dirs_seq = find_dirs_sequences_for_split(
                dir_sequences_data_split,
                split=split,
            )

            dirs_fp_seq = [dir for dir in dirs_seq if "false-positives" in str(dir)]
            dirs_tp_seq = [dir for dir in dirs_seq if "true-positives" in str(dir)]

            logger.info(f"{split} split: smoke sequences")
            for dir_tp_seq in tqdm(dirs_tp_seq):
                handle_dir_sequence(
                    dir_sequence=dir_tp_seq,
                    is_smoke=True,
                    split=split,
                )

            logger.info(f"{split} split: background sequences")
            for dir_fp_seq in tqdm(dirs_fp_seq):
                handle_dir_sequence(
                    dir_sequence=dir_fp_seq,
                    is_smoke=False,
                    split=split,
                )
