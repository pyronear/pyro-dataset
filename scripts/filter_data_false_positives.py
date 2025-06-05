"""
CLI Script to filter data from the FP_2024 dataset to only keep images with
false positives.

This script processes a dataset by filtering out images that do not contain
false positives as identified by predictions from a model. The resulting
filtered dataset retains the original folder structure, ensuring that only
images associated with false positive predictions are saved in the specified
output directory.

Usage:
    python filter_data_false_positives.py --save-dir <output_directory>
    --dir-dataset <input_dataset_directory> --dir-predictions <predictions_directory>

Arguments:
    --save-dir: Directory where the filtered dataset will be saved.
    --dir-dataset: Directory containing the raw FP_2024 dataset.
    --dir-predictions: Directory containing the model predictions.
    --loglevel: Set the logging level (default: info).
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
        "--save-dir",
        help="directory to save the filtered dataset.",
        type=Path,
        default=Path("./data/interim/filtered/false_positives/FP_2024/wise_wolf/"),
    )
    parser.add_argument(
        "--dir-dataset",
        help="directory containing the pyro-sdis dataset.",
        type=Path,
        default=Path("./data/raw/FP_2024/"),
    )
    parser.add_argument(
        "--dir-predictions",
        help="directory containing the predictions made by a model on the --dir-dataset.",
        type=Path,
        default=Path("./data/interim/FP_2024/predictions/wise_wolf/"),
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
    elif not args["dir_predictions"].exists():
        logging.error(
            f"invalid --dir-predictions, dir {args['dir_predictions']} does not exist"
        )
        return False
    else:
        return True


def list_ultralytics_labels(dir_dataset: Path) -> list[Path]:
    """
    List all filepaths in a ultralytics YOLO dataset.

    Returns:
        filepaths (list[Path])
    """
    return list(dir_dataset.glob("**/*.txt"))


def has_smoke(filepath_label: Path) -> bool:
    """
    Does the `filepath_label` contain a smoke?

    Returns:
        has_smoke? (bool): whether or not the filepath has a smoke detected in it.
    """
    return (
        filepath_label.exists()
        and filepath_label.is_file()
        and filepath_label.stat().st_size > 0
    )


def list_images(dir_dataset: Path) -> list[Path]:
    """
    List all images under `dir_dataset`

    Returns:
        filepaths (list[Path]): all image filepaths
    """

    return list(dir_dataset.glob("**/*.jpg"))


def filepath_image_to_filepath_label(
    filepath_image: Path, dir_predictions: Path
) -> Path:
    """
    Given a filepath_image it returns its associated filepath_label.

    Returns:
        filepath_label (Path): the associated label filepath.
    """
    return dir_predictions / f"{filepath_image.stem}.txt"


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
        dir_predictions = args["dir_predictions"]
        logger.info(
            f"filtering false positives from predictions located in {dir_predictions} and saving results in {save_dir}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        filepaths_images = list_images(dir_dataset)
        filepaths_predictions = list_ultralytics_labels(dir_predictions)
        logger.info(
            f"found {len(filepaths_predictions)} prediction files in {dir_predictions}"
        )
        filepaths_predictions_with_smoke = {
            fp for fp in filepaths_predictions if has_smoke(fp)
        }
        logger.info(
            f"found {len(filepaths_predictions_with_smoke)} prediction files containing false positives"
        )
        for filepath_image in tqdm(filepaths_images):
            filepath_prediction = filepath_image_to_filepath_label(
                filepath_image,
                dir_predictions=dir_predictions,
            )
            if filepath_prediction in filepaths_predictions_with_smoke:
                filepath_image_destination = save_dir / filepath_image.relative_to(
                    dir_dataset
                )
                filepath_image_destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(filepath_image, filepath_image_destination)

        exit(0)
