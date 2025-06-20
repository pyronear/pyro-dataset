"""
CLI script to perform a sanity check on the dir-platform-annotated-sequences

Usage:
    python sanity_check_annotated_sequences.py [--dir-platform-annotated-sequences DIR] [-log LOGLEVEL]

Arguments:
    --dir-platform-annotated-sequences DIR
        Directory of the annotated sequences from the Pyronear Platform. Default is './data/raw/pyronear-platform-annotated-sequences'.

    -log LOGLEVEL, --loglevel LOGLEVEL
        Provide logging level. Example: --loglevel debug. Default is 'info'.
"""

import argparse
import json
import logging
from pathlib import Path


def make_cli_parser() -> argparse.ArgumentParser:
    """
    Make the CLI parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir-platform-annotated-sequences",
        help="Directory of the annotated sequences from the Pyronear Platform",
        default=Path("./data/raw/pyronear-platform-annotated-sequences"),
        type=Path,
    )
    parser.add_argument(
        "-log",
        "--loglevel",
        default="info",
        help="Provide logging level. Example --loglevel debug, default=warning",
    )
    return parser


def find_sequence_dirs(dir: Path) -> list[Path]:
    """
    Recursively find all directories containing 'sequence-' in their name.
    """
    return list(dir.rglob("*sequence-*"))


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


def count_files_in_dir(dir: Path, suffix: str) -> int:
    """
    Count the number of files in the given directory.
    """
    if dir.is_dir():
        return sum(1 for _ in dir.glob(f"*.{suffix}") if _.is_file())
    else:
        return 0


def check_dir_sequence(dir_sequence: Path, check_ground_truth: bool) -> dict:
    results = {
        "detections": {},
        "images": {},
        "labels_predictions": {},
        "labels_ground_truth": {},
    }
    dir_detections = dir_sequence / "detections"
    dir_images = dir_sequence / "images"
    dir_labels_predictions = dir_sequence / "labels_predictions"
    dir_labels_ground_truth = dir_sequence / "labels_ground_truth"
    n_detections = count_files_in_dir(dir_detections, suffix="jpg")
    n_images = count_files_in_dir(dir_images, suffix="jpg")
    n_labels_predictions = count_files_in_dir(dir_labels_predictions, suffix="txt")
    n_labels_ground_truth = count_files_in_dir(dir_labels_ground_truth, suffix="txt")
    results["detections"]["count"] = n_detections
    results["images"]["count"] = n_images
    results["labels_predictions"]["count"] = n_labels_predictions
    results["labels_ground_truth"]["count"] = n_labels_ground_truth

    if check_ground_truth and not (
        n_detections == n_images == n_labels_predictions == n_labels_ground_truth
    ):
        logging.error(f"Mismatch of number of files in {dir_sequence} - {results}")
    if not check_ground_truth and not (
        n_detections == n_images == n_labels_predictions
    ):
        logging.error(f"Mismatch of number of files in {dir_sequence} - {results}")

    if not dir_detections.exists() or not dir_detections.is_dir():
        results["detections"] = "missing folder"
        logging.error(f"Missing folder {dir_detections}")
    if not dir_images.exists() or not dir_images.is_dir():
        results["images"] = "missing folder"
        logging.error(f"Missing folder {dir_images}")
    if not dir_labels_predictions.exists() or not dir_labels_predictions.is_dir():
        results["labels_predictions"] = "missing folder"
        logging.error(f"Missing folder {dir_labels_predictions}")
    if check_ground_truth and (
        not dir_labels_ground_truth.exists() or not dir_labels_ground_truth.is_dir()
    ):
        results["labels_ground_truth"] = "missing folder"
        logging.error(f"Missing folder {dir_labels_ground_truth}")

    return results


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        exit(1)
    else:
        dir_platform_annotated_sequences = args["dir_platform_annotated_sequences"]
        filepath_sequences_csv = dir_platform_annotated_sequences / "sequences.csv"
        dirs = [d for d in dir_platform_annotated_sequences.iterdir() if d.is_dir()]
        results = {}
        for dir in dirs:
            results[dir.name] = {"false-positives": {}, "true-positives": {}}
            dir_false_positives = dir / "false-positives"
            dir_true_positives = dir / "true-positives"

            if not dir_false_positives.exists():
                logger.error(f"Missing `false-positives` folder in {dir}")
                results[dir.name]["false-positives"] = "missing folder"
            else:
                dirs_sequences = find_sequence_dirs(dir=dir_true_positives)
                for dir_sequence in dirs_sequences:
                    results[dir.name]["false-positives"][dir_sequence.name] = (
                        check_dir_sequence(
                            dir_sequence=dir_sequence,
                            check_ground_truth=False,
                        )
                    )

            if not dir_true_positives.exists():
                logger.error(f"Missing `true-positives` folder in {dir}")
                results[dir.name]["true-positives"] = "missing folder"
            else:
                dirs_sequences = find_sequence_dirs(dir=dir_true_positives)
                for dir_sequence in dirs_sequences:
                    results[dir.name]["true-positives"][dir_sequence.name] = (
                        check_dir_sequence(
                            dir_sequence=dir_sequence,
                            check_ground_truth=True,
                        )
                    )
        # logger.info(json.dumps(results, indent=4))
        logger.info("Check done ✅")
