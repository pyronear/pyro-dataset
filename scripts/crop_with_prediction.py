"""
CLI Script to generate image crops based on the predicted bounding boxes from the
models. This script processes images and their corresponding YOLO prediction files,
crops the images based on the provided bounding boxes, and saves the crops in specified
directories. Additionally, it logs the details of the cropping process and generates a
summary CSV file containing metadata about each crop operation.

Usage:
    python crop_with_prediction.py [options]

Options:
    --save-dir          Directory to save the cropped images.
    --target-size       Size of the crops to generate (default: 224).
    --dir-images        Directory containing the images to process.
    --dir-predictions   Directory containing the model predictions.
    --loglevel          Logging level (default: info).
"""

import argparse
import logging
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from pyro_dataset.yolo.utils import (
    clip_xyxy,
    crop_xyxy,
    expand_xyxy,
    parse_yolo_prediction_txt_file,
    xyxyn2xyxy,
)


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-dir",
        help="directory to save the predictions.",
        type=Path,
        default=Path("./data/interim/pyro-sdis/val/crops/"),
    )
    parser.add_argument(
        "--target-size",
        help="size of the crops to generate.",
        type=int,
        default=224,
    )
    parser.add_argument(
        "--dir-images",
        help="directory containing the images.",
        type=Path,
        default=Path("./data/raw/pyro-sdis/images/val/"),
    )
    parser.add_argument(
        "--dir-predictions",
        help="directory containing the model predictions.",
        type=Path,
        default=Path("./data/interim/pyro-sdis/val/predictions/wise_wolf"),
    )
    parser.add_argument(
        "-log",
        "--loglevel",
        default="info",
        help="Provide logging level. Example --loglevel debug, default=warning",
    )
    return parser


def validate_parsed_args(args: dict) -> bool:
    """Returns whether the parsed args are valid."""
    if not args["dir_images"].exists():
        logging.error(f"invalid --dir-images, dir {args['dir_images']} does not exist")
        return False
    elif not args["dir_predictions"].exists():
        logging.error(
            f"invalid --dir-predictions, file {args['dir_predictions']} does not exist"
        )
        return False

    else:
        return True


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
        dir_images = args["dir_images"]
        dir_predictions = args["dir_predictions"]
        target_height, target_width = args["target_size"], args["target_size"]
        target_size = (target_width, target_height)
        save_dir.mkdir(parents=True, exist_ok=True)

        # List all nested txt and jpg files in the dir_images and
        # dir_predictions
        filepaths_predictions = list(dir_predictions.glob("**/*.txt"))
        filepaths_images = list(dir_images.glob("**/*.jpg"))
        logging.info(
            f"found {len(filepaths_predictions)} txt prediction files in {dir_predictions}"
        )
        logging.info(f"found {len(filepaths_images)} images in {dir_images}")
        filepath_summary_csv = save_dir / "summary.csv"
        summary_records = []

        for filepath_image in tqdm(filepaths_images):
            filepath_label = (
                dir_predictions / f"{filepath_image.relative_to(dir_images).stem}.txt"
            )
            if (
                filepath_label.exists()
                and filepath_image.exists()
                and filepath_label.is_file()
                and filepath_label.stat().st_size > 0
            ):
                array_image = cv2.imread(filepath_image)
                h, w, _ = array_image.shape
                with open(filepath_label, "r") as fd:
                    txt_content = fd.read()
                    yolo_predictions = parse_yolo_prediction_txt_file(
                        txt_content=txt_content
                    )
                    for idx, yolo_prediction in enumerate(yolo_predictions):
                        xyxy = xyxyn2xyxy(yolo_prediction.xyxyn, w=w, h=h)
                        xyxy_clipped = clip_xyxy(xyxy=xyxy, w=w, h=h)
                        x_min, y_min, x_max, y_max = xyxy_clipped
                        crop = array_image[y_min:y_max, x_min:x_max]
                        crop = crop_xyxy(
                            xyxy=xyxy_clipped,
                            array_image=array_image,
                        )
                        xyxy_expanded = expand_xyxy(
                            xyxy=xyxy_clipped,
                            array_image=array_image,
                            target_width=target_width,
                            target_height=target_height,
                        )
                        crop_expanded = crop_xyxy(
                            xyxy=xyxy_expanded,
                            array_image=array_image,
                        )
                        filepath_crop_original = (
                            save_dir / "raw" / f"{filepath_image.stem}_crop_{idx}.jpg"
                        )
                        filepath_crop_expanded = (
                            save_dir
                            / f"{target_width}x{target_height}"
                            / f"{filepath_image.stem}_crop_{idx}.jpg"
                        )
                        filepath_crop_original.parent.mkdir(parents=True, exist_ok=True)
                        filepath_crop_expanded.parent.mkdir(parents=True, exist_ok=True)

                        # data that describes the different crop operations
                        record = {
                            "image_height": h,
                            "image_width": w,
                            "crop_height": crop.shape[0],
                            "crop_width": crop.shape[1],
                            "crop_expanded_height": crop_expanded.shape[0],
                            "crop_expanded_width": crop_expanded.shape[1],
                            "filepath_image": str(filepath_image),
                            "filepath_label": str(filepath_label),
                            "filepath_crop_original": str(filepath_crop_original),
                            "filepath_crop_expanded": str(filepath_crop_expanded),
                            "crop_index": idx,
                            "prediction_xywhn": str(yolo_prediction.xywhn),
                            "prediction_xyxyn": str(yolo_prediction.xyxyn),
                            "prediction_xyxy": str(xyxy),
                            "prediction_xyxy_clipped": str(xyxy_clipped),
                            "prediction_confidence": yolo_prediction.confidence,
                            "prediction_class_id": yolo_prediction.class_id,
                        }

                        if crop_expanded.shape[:2] != target_size:
                            crop_expanded_resized = cv2.resize(
                                crop_expanded,
                                target_size,
                            )
                            cv2.imwrite(filepath_crop_original, crop)
                            cv2.imwrite(filepath_crop_expanded, crop_expanded_resized)
                            record["crop_expanded_is_resized"] = True
                        else:

                            cv2.imwrite(filepath_crop_original, crop)
                            cv2.imwrite(filepath_crop_expanded, crop_expanded)
                            record["crop_expanded_is_resized"] = False

                        summary_records.append(record)

            df_summary = pd.DataFrame(summary_records)
            df_summary.to_csv(filepath_summary_csv, index=False)
