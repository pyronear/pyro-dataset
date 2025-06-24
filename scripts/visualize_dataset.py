"""
CLI script to overlay the labels (smoke bounding boxes) onto images in
different datasets.

This script processes images in a specified dataset directory, overlays ground
truth labels
onto the images, and saves the visualized images in a designated output
directory.

Usage:
    python visualize_dataset.py --dir-save /path/to/save --dir-dataset /path/to/dataset --loglevel debug

Arguments:
    --dir-save: Directory to save the filtered dataset (default: ./data/processed/viz/wildfire_test/).
    --dir-dataset: Directory containing the dataset to visualize (default: ./data/processed/wildfire_test).
    --loglevel: Provide logging level (default: info, options: debug, info, warning, error, critical).

"""

import argparse
import logging
from pathlib import Path

import cv2
import tqdm

from pyro_dataset.yolo.utils import overlay_ground_truth, parse_yolo_annotation_txt_file


def make_cli_parser() -> argparse.ArgumentParser:
    """
    Make the CLI parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir-save",
        help="directory to save the filtered dataset.",
        type=Path,
        default=Path("./data/reporting/viz/wildfire_test/"),
    )
    parser.add_argument(
        "--dir-dataset",
        help="directory containing the dataset to visualize.",
        type=Path,
        default=Path("./data/processed/wildfire_test"),
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
    else:
        return True


def find_images_filepaths(dir: Path) -> list[Path]:
    """
    Recursively find all image file paths in the specified directory.

    This function searches for all JPG image files in the provided directory
    and its subdirectories. It returns a list of paths to the found image files.

    Args:
        dir (Path): The directory path to search for images.

    Returns:
        list[Path]: A list of paths to the JPG image files found in the directory.
    """
    return list(dir.rglob("**/*.jpg"))


def to_label_ground_truth(filepath_image: Path) -> Path:
    """
    Convert the image file path to its corresponding label file path.

    This function takes the file path of an image and constructs the file path of
    the associated label file by replacing the "images" directory with "labels"
    and changing the file extension from ".jpg" to ".txt".

    Args:
        filepath_image (Path): The file path of the image.

    Returns:
        Path: The file path of the corresponding label file.
    """
    return Path(str(filepath_image).replace("images", "labels").replace("jpg", "txt"))


def run(dir_save: Path, dir_dataset: Path) -> None:
    """
    Process images in the specified dataset directory, overlay ground truth labels,
    and save the visualized images in the designated output directory.

    This function finds all JPG images in the given dataset directory, retrieves
    their corresponding label files, overlays the labels onto the images, and
    saves the processed images in the specified save directory.

    Args:
        dir_save (Path): Directory where the processed images will be saved.
        dir_dataset (Path): Directory containing the dataset of images to be processed.
    """

    filepaths_images = find_images_filepaths(dir=dir_dataset)
    dir_save.mkdir(parents=True, exist_ok=True)

    for filepath_image in tqdm.tqdm(filepaths_images):
        try:
            filepath_label = to_label_ground_truth(filepath_image)
            assert filepath_image.exists()
            assert filepath_label.exists()

            array_image = cv2.imread(str(filepath_image))
            filepath_save = dir_save / Path(
                str(filepath_image.relative_to(dir_dataset)).replace(
                    "images", "images_ground_truth"
                )
            )
            filepath_save.parent.mkdir(parents=True, exist_ok=True)

            with open(filepath_label, "r") as fd:
                txt_content = fd.read()
                annotations = parse_yolo_annotation_txt_file(txt_content=txt_content)
                if len(annotations) == 0:
                    cv2.imwrite(str(filepath_save), array_image)
                else:
                    scene = overlay_ground_truth(
                        array_image=array_image, annotations=annotations
                    )
                    cv2.imwrite(str(filepath_save), scene)
        except Exception as e:
            print(f"ERROR with filepath_image {filepath_image}")
            print(e)


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
        dir_dataset = args["dir_dataset"]
        dir_save.mkdir(parents=True, exist_ok=True)
        logger.info(f"Visualizing dataset in {dir_save}")
        run(dir_save=dir_save, dir_dataset=dir_dataset)
