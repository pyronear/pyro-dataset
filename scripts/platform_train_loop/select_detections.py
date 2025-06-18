"""
CLI script for selecting and processing detection data from annotated
sequences. It allows users to specify directories for saving selected
detections, loading annotated sequences, and defining parameters for true and
false positive detections.

Usage:
    python select_detections.py --dir-save <directory_to_save>
                                  --dir-platform-annotated-sequences <directory_of_annotated_sequences>
                                  [--number-detections-per-sequence-true-positive <number>]
                                  [--number-detections-per-sequence-false-positive <number>]
                                  [-log <loglevel>]

Arguments:
    --dir-save: Directory to save the selected detections (default: './data/interim/pyronear-platform/sequences/')
    --dir-platform-annotated-sequences: Required. Directory containing annotated sequences.
    --number-detections-per-sequence-true-positive: Number of true positive detections to process (default: 3).
    --number-detections-per-sequence-false-positive: Number of false positive detections to process (default: 3).
    -log, --loglevel: Set the logging level (default: 'info').
"""

import argparse
import logging
import shutil
from pathlib import Path

from pyro_dataset.yolo.utils import parse_yolo_prediction_txt_file


def make_cli_parser() -> argparse.ArgumentParser:
    """
    Make the CLI parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir-save",
        help="Directory to save the selection",
        type=Path,
        default=Path("./data/interim/pyronear-platform/sequences/"),
    )
    parser.add_argument(
        "--dir-platform-annotated-sequences",
        help="Directory of the annotated sequences from the Pyronear Platform",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--number-detections-per-sequence-true-positive",
        help="Number of detections per sequence for true positives",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--number-detections-per-sequence-false-positive",
        help="Number of detections per sequence for false positives",
        type=int,
        default=3,
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
    if not args["dir_platform_annotated_sequences"].exists():
        logging.error(
            f"Invalid --dir-platform-annotated-sequences, directory does not exist"
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


def find_sequence_folders(dir_annotated_sequences: Path) -> list[Path]:
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
        for seq_dir in dir_annotated_sequences.rglob("**/*")
        if seq_dir.is_dir()
        and "_sequence-" in seq_dir.name
        and seq_dir.name.split("_sequence-")[-1].isdigit()
    ]


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
        "filepath_label_ground_truth": dir_sequence
        / "labels_ground_truth"
        / f"{stem}.txt",
        "filepath_label_prediction": dir_sequence
        / "labels_predictions"
        / f"{stem}.txt",
        "filepath_image": dir_sequence / "images" / f"{stem}.jpg",
        "filepath_detection": dir_sequence / "detections" / f"{stem}.jpg",
    }


def select_best_false_positives(
    dir_sequence: Path,
    number_detections_per_sequence: int,
) -> list[dict[str, Path]]:
    """
    Select the best false positives from the detection records.

    This function processes the detection records for a given sequence,
    filtering them based on confidence scores and selecting the top
    results according to the specified number of detections.

    Args:
        dir_sequence (Path): The directory containing the sequence data.
        number_detections_per_sequence (int): The number of top detections to select.

    Returns:
        list[dict[str, Path]]: A list of dictionaries containing the file paths
        for the selected detections, excluding the prediction details.
    """
    records = [
        {
            "prediction": prediction,
            **(get_filepaths(dir_sequence=dir_sequence, stem=fp.stem)),
        }
        for fp in (dir_sequence.glob("**/labels_predictions/*.txt"))
        for prediction in parse_yolo_prediction_txt_file(read_file_content(fp))
    ]
    records_filtered = [
        record for record in records if record["prediction"].confidence > 0
    ]
    records_selected = sorted(
        records_filtered, key=lambda x: x["prediction"].confidence, reverse=True
    )[:number_detections_per_sequence]
    result = [
        {key: value for key, value in r.items() if key != "prediction"}
        for r in records_selected
    ]
    return result


def select_best_true_positives(
    dir_sequence: Path,
    number_detections_per_sequence: int,
) -> list[dict[str, Path]]:
    """
    Select the best true positives from the detection records.

    This function processes the detection records for a given sequence,
    filtering them based on confidence scores and selecting the top
    results according to the specified number of detections.

    Args:
        dir_sequence (Path): The directory containing the sequence data.
        number_detections_per_sequence (int): The number of top detections to select.

    Returns:
        list[dict[str, Path]]: A list of dictionaries containing the file paths
        for the selected detections, excluding the prediction details.
    """
    # TODO: leverage the labels_ground_truth to pick the ones that have not
    # been at all detected instead or pick the lowest scores as we do now
    records = [
        {
            "prediction": prediction,
            **(get_filepaths(dir_sequence=dir_sequence, stem=fp.stem)),
        }
        for fp in (dir_sequence.glob("**/labels_predictions/*.txt"))
        for prediction in parse_yolo_prediction_txt_file(read_file_content(fp))
    ]
    records_filtered = [
        record for record in records if record["prediction"].confidence > 0
    ]
    records_selected = sorted(
        records_filtered, key=lambda x: x["prediction"].confidence, reverse=False
    )[:number_detections_per_sequence]
    result = [
        {key: value for key, value in r.items() if key != "prediction"}
        for r in records_selected
    ]
    return result


def copy_over(
    dir_sequence: Path,
    dir_platform_annotated_sequences: Path,
    dir_save: Path,
    records: list[dict],
) -> None:
    """
    Copy image, detection, and label files from the source directory to the destination directory.

    This function takes the records of selected detections and copies the corresponding
    image, detection, and label files from the source directory to the specified
    destination directory, maintaining the directory structure.

    Args:
        dir_sequence (Path): The directory containing the sequence data.
        dir_platform_annotated_sequences (Path): The base directory of the annotated sequences.
        dir_save (Path): The directory where the files will be saved.
        records (list[dict]): A list of dictionaries containing file paths for the selected detections.
    """
    for record in records:
        filepath_source_image = record["filepath_image"]
        filepath_destination_image = (
            dir_save
            / dir_sequence.relative_to(dir_platform_annotated_sequences)
            / "images"
            / filepath_source_image.name
        )
        filepath_source_detection = record["filepath_detection"]
        filepath_destination_detection = (
            dir_save
            / dir_sequence.relative_to(dir_platform_annotated_sequences)
            / "detections"
            / filepath_source_detection.name
        )
        filepath_source_label_prediction = record["filepath_label_prediction"]
        filepath_destination_label_prediction = (
            dir_save
            / dir_sequence.relative_to(dir_platform_annotated_sequences)
            / "labels_predictions"
            / filepath_source_label_prediction.name
        )
        filepath_source_label_ground_truth = record["filepath_label_ground_truth"]
        filepath_destination_label_ground_truth = (
            dir_save
            / dir_sequence.relative_to(dir_platform_annotated_sequences)
            / "labels_ground_truth"
            / filepath_source_label_ground_truth.name
        )
        filepath_destination_image.parent.mkdir(parents=True, exist_ok=True)
        filepath_destination_detection.parent.mkdir(parents=True, exist_ok=True)
        # filepath_destination_label.parent.mkdir(parents=True, exist_ok=True)
        filepath_destination_label_prediction.parent.mkdir(parents=True, exist_ok=True)
        filepath_destination_label_ground_truth.parent.mkdir(
            parents=True, exist_ok=True
        )
        shutil.copy(src=filepath_source_image, dst=filepath_destination_image)
        shutil.copy(src=filepath_source_detection, dst=filepath_destination_detection)
        # shutil.copy(src=filepath_source_label, dst=filepath_destination_label)
        shutil.copy(
            src=filepath_source_label_prediction,
            dst=filepath_destination_label_prediction,
        )

        if filepath_destination_label_ground_truth.exists():
            shutil.copy(
                src=filepath_source_label_ground_truth,
                dst=filepath_destination_label_ground_truth,
            )


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        exit(1)
    else:
        logger.info(args)
        dir_save = args["dir_save"]
        dir_platform_annotated_sequences = args["dir_platform_annotated_sequences"]
        number_detections_per_sequence_false_positive = args[
            "number_detections_per_sequence_false_positive"
        ]
        number_detections_per_sequence_true_positive = args[
            "number_detections_per_sequence_true_positive"
        ]
        dirs_fp = find_false_positive_folders(dir_platform_annotated_sequences)
        dirs_tp = find_true_positive_folders(dir_platform_annotated_sequences)

        logger.info(f"Selecting the false positives from {dirs_fp}")
        for dir_fp in dirs_fp:
            dirs_sequences_fp = find_sequence_folders(dir_fp)
            print(
                f"Found {len(dirs_sequences_fp)} false positive sequences in {dir_fp}"
            )
            for dir_sequence_fp in dirs_sequences_fp:
                records = select_best_false_positives(
                    dir_sequence_fp,
                    number_detections_per_sequence=number_detections_per_sequence_false_positive,
                )
                copy_over(
                    dir_sequence=dir_sequence_fp,
                    dir_platform_annotated_sequences=dir_platform_annotated_sequences,
                    dir_save=dir_save,
                    records=records,
                )

        logger.info(f"Selecting the true positives from {dirs_tp}")
        for dir_tp in dirs_tp:
            dirs_sequences_tp = find_sequence_folders(dir_tp)
            print(f"Found {len(dirs_sequences_tp)} true positive sequences in {dir_tp}")
            for dir_sequence_tp in dirs_sequences_tp:
                records = select_best_true_positives(
                    dir_sequence_tp,
                    number_detections_per_sequence=number_detections_per_sequence_true_positive,
                )
                copy_over(
                    dir_sequence=dir_sequence_tp,
                    dir_platform_annotated_sequences=dir_platform_annotated_sequences,
                    dir_save=dir_save,
                    records=records,
                )
