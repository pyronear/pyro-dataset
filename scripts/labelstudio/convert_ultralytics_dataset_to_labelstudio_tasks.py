"""
CLI script to generate LS tasks from a Ultralytics Dataset.

Usage:
    python convert_ultralytics_dataset_to_labelstudio_tasks.py --dir-save <directory> --dir-ultralytics-dataset <directory> --url-root-images <url> --loglevel <level>

Arguments:
    --dir-save: Directory to save the LS tasks.
    --dir-ultralytics-dataset: Directory containing a Ultralytics dataset.
    --url-root-images: Root of the URL that serves the images, can be an S3 URL, a local server URL, etc.
    -log, --loglevel: Provide logging level. Example --loglevel debug, default=warning.
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from pyro_dataset.yolo.utils import (
    YOLOObjectDetectionAnnotation,
    parse_yolo_annotation_txt_file,
)


def make_cli_parser() -> argparse.ArgumentParser:
    """
    Make the CLI parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir-save",
        help="Directory to save the LS tasks",
        type=Path,
        default=Path("./data/raw/labelstudio/tasks/"),
    )
    parser.add_argument(
        "--dir-ultralytics-dataset",
        help="Directory containing a Ultralytics dataset",
        type=Path,
        default=Path("./data/raw/Test_dataset_2025/"),
    )
    parser.add_argument(
        "--url-root-images",
        help="Root of the url that serves the images, can be an s3 url, a local server url, etc.",
        type=str,
        default="http://localhost:8000/pyronear/test/",
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
    if (
        not args["dir_ultralytics_dataset"].exists()
        or not args["dir_ultralytics_dataset"].is_dir()
    ):
        logging.error("Invalid --dir-ultralytics-dataset")
        return False
    return True


def yolo_annotation_to_labelstudio_annotation(
    yolo_annotation: YOLOObjectDetectionAnnotation,
    width_image: int,
    height_image: int,
    class_id_to_name_mapping: dict[int, str],
) -> dict:
    """
    Convert a yolo_annotation into a labelstudio annotation.
    """
    xn, yn, wn, hn = yolo_annotation.xywhn
    class_name = class_id_to_name_mapping.get(yolo_annotation.class_id, "unknown")
    return {
        "from_name": "label",
        "image_rotation": 0,
        "original_height": height_image,
        "original_width": width_image,
        "to_name": "image",
        "type": "rectanglelabels",
        "value": {
            "height": hn.item() * 100,
            "width": wn.item() * 100,
            "rectanglelabels": [class_name],
            "rotation": 0,
            "x": (xn.item() - wn.item() / 2) * 100,
            "y": (yn.item() - hn.item() / 2) * 100,
        },
    }


def make_ls_task(
    filepath_image: Path,
    image_url: str,
    yolo_annotation_list: list[YOLOObjectDetectionAnnotation],
    model_version: str = "yolo",
    random_seed: int = 0,
    class_id_to_name_mapping: dict[int, str] = {},
) -> dict:
    """
    Create a LS formatted task that can be imported into LabelStudio.
    Returns a dict that can be stored as json.
    """
    pil_image = Image.open(filepath_image)
    width_image, height_image = pil_image.size
    rng = random.Random(random_seed)

    ls_predictions_result = []
    for yolo_annotation in yolo_annotation_list:
        id_random = f"{rng.getrandbits(128):032x}"
        ls_pred = yolo_annotation_to_labelstudio_annotation(
            yolo_annotation=yolo_annotation,
            width_image=width_image,
            height_image=height_image,
            class_id_to_name_mapping=class_id_to_name_mapping,
        )
        ls_pred["id"] = id_random
        ls_predictions_result.append(ls_pred)

    return {
        "data": {
            "filepath": f"{filepath_image}",
            "image": f"{image_url}",
            "number_smokes": len(yolo_annotation_list),
        },
        "predictions": [
            {
                "model_version": model_version,
                "origin": model_version,
                "result": ls_predictions_result,
            }
        ],
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
        dir_save = args["dir_save"]
        dir_ultralytics_dataset = args["dir_ultralytics_dataset"]
        url_root_images = args["url_root_images"]
        filepaths_images = list(dir_ultralytics_dataset.rglob("*.jpg"))
        logger.info(
            f"Found {len(filepaths_images)} images in {dir_ultralytics_dataset}"
        )
        dir_save.mkdir(parents=True, exist_ok=True)
        class_id_to_name_mapping = {0: "smoke", 1: "smoke"}

        logger.info("Generating the LS tasks ")
        for filepath_image in tqdm(filepaths_images):
            filepath_label = Path(
                str(filepath_image).replace("images", "labels").replace("jpg", "txt")
            )
            image_url = os.path.join(url_root_images, filepath_image.name)
            yolo_annotation_list = []
            if filepath_label.exists():
                with open(filepath_label, "r") as f:
                    txt_content = f.read()
                    yolo_annotation_list = parse_yolo_annotation_txt_file(txt_content)

            ls_task = make_ls_task(
                filepath_image=filepath_image,
                image_url=image_url,
                yolo_annotation_list=yolo_annotation_list,
                class_id_to_name_mapping=class_id_to_name_mapping,
                model_version="yolo",
            )

            filepath_ls_task_json = dir_save / f"{filepath_image.stem}.json"
            logging.debug(f"Saving LS task in {filepath_ls_task_json}")
            with open(filepath_ls_task_json, "w") as f:
                json.dump(ls_task, f)
