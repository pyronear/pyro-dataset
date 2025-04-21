"""
CLI Script to merge all different data split sources containg fire smoke
without `background` images.

Note:
    A background image is an image without associated detections.
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
        default=Path("./data/interim/data-split/smoke/wildfire/"),
    )
    parser.add_argument(
        "--dir-datasets",
        help="directory containing the datasets in ultralytics format to join.",
        nargs="+",
        type=Path,
        default=[
            Path("./data/interim/data-split/smoke/FIGLIB_ANNOTATED_RESIZED/"),
            Path("./data/interim/filtered/smoke/pyro-sdis/"),
            Path("./data/interim/filtered/smoke/pyronear_ds_03_2024/"),
        ],
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
    for dir_dataset in args["dir_datasets"]:
        if not dir_dataset.exists() and dir_dataset.is_dir():
            logging.error(
                f"invalid --dir-datasets, dir {args['dir_dataset']} is not valid"
            )
            return False
    else:
        return True


def list_ultralytics_images(dir_dataset: Path) -> list[Path]:
    """
    List all images in a ultralytics YOLO dataset.

    Returns:
        filepaths (list[Path])
    """
    return list(dir_dataset.glob("**/*.jpg"))


def list_ultralytics_labels(dir_dataset: Path) -> list[Path]:
    """
    List all filepaths in a ultralytics YOLO dataset.

    Returns:
        filepaths (list[Path])
    """
    return list(dir_dataset.glob("**/*.txt"))


def to_filepath_destination(dir_dataset: Path, save_dir: Path, filepath: Path) -> Path:
    """
    Return the filepath_destination to store the filepath.

    Returns:
        filepath_destination (Path): where to copy filepath to.
    """
    if dir_dataset.parts[-1] == "pyro-sdis":
        return (
            save_dir
            / filepath.relative_to(dir_dataset).parent
            / f"pyronear_{filepath.name}"
        )
    else:
        return save_dir / filepath.relative_to(dir_dataset)


def merge_ultralytics_dataset(dir_dataset: Path, save_dir: Path) -> None:
    """
    Merge the dataset from dir_dataset into save_dir which can contain already
    existing ultralytics datasets.
    """

    filepaths_images = list_ultralytics_images(dir_dataset)
    filepaths_labels = list_ultralytics_labels(dir_dataset)
    filepaths = filepaths_images + filepaths_labels

    for filepath in tqdm(filepaths):
        filepath_destination = to_filepath_destination(
            dir_dataset=dir_dataset,
            save_dir=save_dir,
            filepath=filepath,
        )
        filepath_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(filepath, filepath_destination)


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        exit(1)
    else:
        logger.info(args)
        save_dir = args["save_dir"]
        dir_datasets = args["dir_datasets"]
        logger.info(
            f"joining {len(dir_datasets)} datasets in {save_dir}: {dir_datasets}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        for dir_dataset in dir_datasets:
            logging.info(f"merge dataset {dir_dataset} into {save_dir}")
            merge_ultralytics_dataset(dir_dataset=dir_dataset, save_dir=save_dir)

        exit(0)
