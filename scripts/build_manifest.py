"""
CLI script to generate a manifest file for the generated datasets.
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

    return True


def get_stats(dir_dataset: Path) -> dict:
    """
    Return a dictionnary containg stats about the different splits.
    """
    result = {}
    dir_images = dir_dataset / "images"
    splits = [dir.name for dir in dir_images.iterdir() if dir.is_dir()]
    for split in splits:
        dir_split = dir_images / split
        file_count = sum(1 for item in dir_split.iterdir() if item.is_file())
        result[split] = {"number_images": file_count}
    return result


def make_data_manifest(
    dir_wildfire_dataset_train_val: Path,
    dir_wildfire_dataset_test: Path,
) -> dict:
    """
    Make the data manifest.yaml file.

    It is meant to persist as much information as possible about the dataset
    inputs.
    """
    filepath_dvc_lock = Path("./dvc.lock")
    dvc_lock = yaml_read(filepath_dvc_lock)
    return {
        "stats": {
            str(dir): get_stats(dir)
            for dir in [dir_wildfire_dataset_train_val, dir_wildfire_dataset_test]
        },
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
        save_dir = args["save_dir"]
        save_dir.mkdir(parents=True, exist_ok=True)
        filepath_manifest = save_dir / "manifest.yaml"
        data_manifest = make_data_manifest(
            dir_wildfire_dataset_train_val=dir_wildfire_dataset_train_val,
            dir_wildfire_dataset_test=dir_wildfire_dataset_test,
        )
        logger.info(f"saving manifest.yaml file in {save_dir}")
        yaml_write(to=filepath_manifest, data=data_manifest)
        exit(0)
