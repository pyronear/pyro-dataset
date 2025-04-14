"""
CLI Script to install pyronear datasets that are saved locally (parquet files).

__Note__:
    The dataset rows must contain the following keys:
      - image: image byte data
      - image_name: name of the image
      - annotations: yolov8 txt format annotations
"""

import argparse
import logging
import os
from pathlib import Path

from datasets import load_dataset


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-dir",
        help="directory to save the extracted dataset.",
        type=Path,
        default=Path("./data/raw/pyro-sdis-testset/"),
    )
    parser.add_argument(
        "--data-dir",
        help="directory where the dataset files are located.",
        type=Path,
        default=Path("./data/raw/pyro-sdis-testset-parquet/"),
    )
    parser.add_argument(
        "-log",
        "--loglevel",
        default="info",
        help="Provide logging level. Example --loglevel debug, default=warning",
    )
    return parser


def validate_parsed_args(args: dict) -> bool:
    return True


def save_ultralytics_format(
    dataset_split,
    split: str,
    dir_images: Path,
    dir_labels: Path,
) -> None:
    """
    Save a dataset split into the Ultralytics format.

    Args:
        dataset_split: The dataset split (e.g., dataset["train"])
        split: "train" or "val"
    """
    for example in dataset_split:
        # Save the image to the appropriate folder
        image = example["image"]  # PIL.Image.Image
        image_name = example["image_name"]  # Original file name
        # output_image_path = os.path.join(dir_images, split, image_name)
        output_image_path = dir_images / split / image_name

        # Save the image object to disk
        image.save(output_image_path)

        # Save label
        annotations = example["annotations"]
        label_name = image_name.replace(".jpg", ".txt").replace(".png", ".txt")
        # output_label_path = os.path.join(dir_labels, split, label_name)
        output_label_path = dir_labels / split / label_name

        with open(output_label_path, "w") as label_file:
            label_file.write(annotations)


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
        data_dir = args["data_dir"]
        # dataset = load_dataset("parquet", data_dir="./data/raw/pyro-sdis-testset/")
        dataset = load_dataset("parquet", data_dir=data_dir)

        dir_images = save_dir / "images"
        dir_labels = save_dir / "labels"

        # Create the directory structure
        for split in dataset.keys():
            (dir_images / split).mkdir(parents=True, exist_ok=True)
            (dir_labels / split).mkdir(parents=True, exist_ok=True)

        for split in dataset.keys():
            logging.info(f"save dataset in ultralytics format for split {split}")
            save_ultralytics_format(
                dataset_split=dataset[split],
                split=split,
                dir_images=dir_images,
                dir_labels=dir_labels,
            )

        logging.info(f"dataset from {data_dir} succesfully exported to {save_dir}")

        exit(0)
