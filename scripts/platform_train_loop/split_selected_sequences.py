"""
CLI script to perform data splitting into train, validation, and test sets for selected detection sequences.

This script allows users to specify directories containing annotated sequences and split them into three subsets:
training, validation, and test. The user can customize the ratios for each subset and set a random seed for reproducibility.

Usage:
    python split_selected_sequences.py --dir-save <path> --dir-platform-selected-sequences <path> --random-seed <int> --ratio-train <float> --ratio-val-test <float> --loglevel <level>

Arguments:
    --dir-save: Directory where the split data will be saved.
    --dir-platform-selected-sequences: Directory containing the selected detection sequences.
    --random-seed: Seed for random number generation.
    --ratio-train: Proportion of data to be used for training (default is 0.8).
    --ratio-val-test: Proportion of data to be used for validation and testing, split evenly (default is 0.5).
    -log, --loglevel: Logging level (default is 'info').
"""

import argparse
import logging
import random
import shutil
from pathlib import Path


def make_cli_parser() -> argparse.ArgumentParser:
    """
    Make the CLI parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir-save",
        help="Directory to save the selection",
        type=Path,
        default=Path("./data/interim/pyronear-platform/sequences-data-split/"),
    )
    parser.add_argument(
        "--dir-platform-selected-sequences",
        help="Directory of the selected sequences from the Pyronear Platform",
        type=Path,
        default="./data/interim/pyronear-platform/sequences/",
    )
    parser.add_argument(
        "--random-seed",
        help="Random seed for shuffling",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--ratio-train",
        help="Training ratio",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--ratio-val-test",
        help="Validation and test ratio",
        type=float,
        default=0.5,
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
    if not args["dir_platform_selected_sequences"].exists():
        logging.error(
            f"Invalid --dir-platform-selected-sequences, directory does not exist"
        )
        return False
    return True


def find_false_positive_folders(dir_annotated_sequences: Path) -> list[Path]:
    """
    Find and return a list of nested directories that end with 'false-positives'.

    Args:
        dir_annotated_sequences (Path): The directory to search for false-positive folders.

    Returns:
        list[Path]: A list of Path objects representing the directories that end with 'false-positives'.
    """
    return [
        seq_dir
        for seq_dir in dir_annotated_sequences.rglob("*")
        if seq_dir.is_dir() and seq_dir.name.endswith("false-positives")
    ]


def find_true_positive_folders(dir_annotated_sequences: Path) -> list[Path]:
    """
    Find and return a list of nested directories that end with 'true-positives'.

    Args:
        dir_annotated_sequences (Path): The directory to search for true-positive folders.

    Returns:
        list[Path]: A list of Path objects representing the directories that end with 'true-positives'.
    """
    return [
        seq_dir
        for seq_dir in dir_annotated_sequences.rglob("*")
        if seq_dir.is_dir() and seq_dir.name.endswith("true-positives")
    ]


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


def train_val_test_split(
    dir: Path,
    random_seed: int,
    train_ratio: float,
    val_test_ratio: float,
) -> dict[str, list[Path]]:
    """
    Split the sequence folders into training, validation, and test sets.

    This function takes a directory containing annotated sequence folders and splits them into three subsets:
    training, validation, and test. The split is performed based on specified ratios, and the directories are
    shuffled to ensure randomness. The function also includes assertions to verify that the splits do not
    overlap and that all directories are accounted for.

    Args:
        dir (Path): The directory containing the sequence folders to be split.
        random_seed (int): The seed used for random number generation to ensure reproducibility.

    Returns:
        dict[str, list[Path]]: A dictionary containing three keys: 'train', 'val', and 'test',
        each mapping to a list of Path objects representing the corresponding directories.
    """
    assert 0 <= train_ratio <= 1, "train_ratio must be between 0 and 1."
    assert 0 <= val_test_ratio <= 1, "val_test_ratio must be between 0 and 1."

    dirs_sequences = find_sequence_folders(dir)

    # Shuffle the directories to ensure random distribution
    rgn = random.Random(random_seed)
    dirs_sequences_shuffled = rgn.sample(dirs_sequences, len(dirs_sequences))

    # Calculate the split indices
    train_size = int(len(dirs_sequences) * train_ratio)
    val_size = int(len(dirs_sequences) * (1 - train_ratio) * val_test_ratio)
    test_size = len(dirs_sequences) - train_size - val_size

    # Assert check to ensure all directories are used in train_dirs, val_dirs, and test_dirs
    assert train_size + val_size + test_size == len(
        dirs_sequences
    ), "Not all directories are accounted for in the splits."

    # Split the directories into train, validation, and test sets
    train_dirs = dirs_sequences_shuffled[:train_size]
    val_dirs = dirs_sequences_shuffled[train_size : train_size + val_size]
    test_dirs = dirs_sequences_shuffled[train_size + val_size :]

    # Assert check to ensure all directories are used in train_dirs, val_dirs, and test_dirs
    assert len(train_dirs) + len(val_dirs) + len(test_dirs) == len(
        dirs_sequences
    ), "Not all directories are accounted for in the splits."

    # Assert check to ensure there is no overlap between train_dirs and val_dirs
    assert not set(train_dirs) & set(
        val_dirs
    ), "There is an overlap between training and validation directories."

    # Assert check to ensure there is no overlap between train_dirs and test_dirs
    assert not set(train_dirs) & set(
        test_dirs
    ), "There is an overlap between training and test directories."

    # Assert check to ensure there is no overlap between val_dirs and test_dirs
    assert not set(val_dirs) & set(
        test_dirs
    ), "There is an overlap between validation and test directories."

    return {
        "train": train_dirs,
        "val": val_dirs,
        "test": test_dirs,
    }


def get_filepaths(dir_sequence: Path, stem: str) -> dict[str, Path]:
    """
    Construct file paths for labels, images, and detection files based on the provided directory sequence and stem.

    Args:
        dir_sequence (Path): The directory containing the sequence data.
        stem (str): The stem of the filename used to construct the paths.

    Returns:
        dict[str, Path]: A dictionary containing the file paths for labels, images, and detections.
    """
    return {
        "filepath_label": dir_sequence / "labels" / f"{stem}.txt",
        "filepath_image": dir_sequence / "images" / f"{stem}.jpg",
        "filepath_detection": dir_sequence / "detections" / f"{stem}.jpg",
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
        random_seed = args["random_seed"]
        random_seed = 1
        dir_save = args["dir_save"]
        dir_platform_selected_sequences = args["dir_platform_selected_sequences"]
        train_ratio = args["ratio_train"]
        val_test_ratio = args["ratio_val_test"]

        dirs_fp = find_false_positive_folders(dir_platform_selected_sequences)
        dirs_tp = find_true_positive_folders(dir_platform_selected_sequences)

        logger.info(f"ramdom Seed: {random_seed}")
        logger.info(f"Handling the false positives from {dirs_fp}")
        for dir_fp in dirs_fp:
            dirs_sequences_fp = find_sequence_folders(dir_fp)
            logger.info(
                f"Found {len(dirs_sequences_fp)} false positive sequences in {dir_fp}"
            )
            data_split = train_val_test_split(
                dir=dir_fp,
                random_seed=random_seed,
                train_ratio=train_ratio,
                val_test_ratio=val_test_ratio,
            )
            for split in ["train", "val", "test"]:
                dirs_sequences = data_split[split]
                for dir_sequence in dirs_sequences:
                    dir_dst = (
                        dir_save
                        / dir_sequence.relative_to(
                            dir_platform_selected_sequences
                        ).parent
                        / split
                        / dir_sequence.name
                    )
                    dir_dst.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(src=dir_sequence, dst=dir_dst, dirs_exist_ok=True)

        logger.info(f"Handling the true positives from {dirs_tp}")
        for dir_tp in dirs_tp:
            dirs_sequences_tp = find_sequence_folders(dir_tp)
            logger.info(
                f"Found {len(dirs_sequences_tp)} true positive sequences in {dir_tp}"
            )
            data_split = train_val_test_split(
                dir=dir_tp,
                random_seed=random_seed,
                train_ratio=train_ratio,
                val_test_ratio=val_test_ratio,
            )
            for split in ["train", "val", "test"]:
                dirs_sequences = data_split[split]
                for dir_sequence in dirs_sequences:
                    dir_dst = (
                        dir_save
                        / dir_sequence.relative_to(
                            dir_platform_selected_sequences
                        ).parent
                        / split
                        / dir_sequence.name
                    )
                    dir_dst.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(src=dir_sequence, dst=dir_dst, dirs_exist_ok=True)

        logger.info(f"Data saved in {dir_save}")
