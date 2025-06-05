"""
This module provides functions for processing detection data from the API,
including converting detection and metadata into structured records,
formatting bounding boxes, downloading images, and saving labels. It also
handles the creation of file paths for storing images and labels, as well as
processing dataframes containing sequences and detections.

Key functionalities include:
- Converting detection metadata into records.
- Formatting API datetime strings and bounding boxes.
- Downloading images and saving detection labels.
- Processing dataframes to generate YOLO formatted labels and overlayed images.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from pyro_dataset.yolo.utils import (
    overlay_predictions,
    parse_yolo_prediction_txt_file,
    xyxyn2xywhn,
)


def to_record(
    detection: dict,
    camera: dict,
    organization: dict,
    sequence: dict,
) -> dict:
    """
    Convert detection, camera, organization, and sequence data into a structured record.

    Parameters:
        detection (dict): Information about the detection including metadata.
        camera (dict): Information about the camera that captured the detection.
        organization (dict): Information about the organization managing the camera.
        sequence (dict): Information about the sequence of detections.

    Returns:
        dict: A structured record containing relevant metadata for the detection.
    """

    return {
        # Organization metadata
        "organization_id": camera["organization_id"],
        "organization_name": organization["name"],
        # Camera metadata
        "camera_id": sequence["camera_id"],
        "camera_name": camera["name"],
        "camera_lat": camera["lat"],
        "camera_lon": camera["lon"],
        "camera_is_trustable": camera["is_trustable"],
        "camera_angle_of_view": camera["angle_of_view"],
        # Sequence metadata
        "sequence_id": sequence["id"],
        "sequence_is_wildfire": sequence["camera_id"],
        "sequence_started_at": sequence["started_at"],
        "sequence_last_seen_at": sequence["last_seen_at"],
        "sequence_azimuth": sequence["azimuth"],
        # Detection metadata
        "detection_id": detection["id"],
        "detection_created_at": detection["created_at"],
        "detection_azimuth": detection["azimuth"],
        "detection_url": detection["url"],
        "detection_bboxes": detection["bboxes"],
        "detection_bucket_key": detection["bucket_key"],
    }


def _format_api_datetime_str(datetime_str: str) -> str:
    """
    Format a datetime string returned by the API into a format suitable for use in file paths.

    This function replaces colons with hyphens and removes microseconds, ensuring the resulting
    string is compliant with typical file naming conventions.

    Example:
      >>> _format_api_datetime_str("2025-05-30T10:02:55.981732")
      '2025-05-30T10-02-55'
    """
    return datetime_str.replace(":", "-").split(".")[0]


def _format_api_bboxes_ultralytics(detection_bboxes: str, class_id: int = 0) -> str:
    """
    Format the bounding boxes (bboxes) returned by the platform API into the
    ultralytics format, which includes the class ID and normalized coordinates.

    Parameters:
        detection_bboxes (str): A string representation of the bounding boxes,
                                typically in a list format where each box
                                contains coordinates and confidence score.
        class_id (int, optional): The class identifier for the bounding boxes.
                                   Defaults to 0.

    Returns:
        str: A string containing the formatted bounding boxes, each on a new line,
             ready for use with the ultralytics model.
    """
    xs = eval(detection_bboxes)

    def format_bbox(x):
        conf = x[4]
        xyxyn = x[0:4]
        xywhn = xyxyn2xywhn(np.array(xyxyn))
        return f"{class_id} {' '.join(map(str, xywhn.tolist()))} {conf}"

    return "\n".join(map(format_bbox, xs))


def _get_filepaths(
    save_dir: Path,
    organization_name: str,
    camera_name: str,
    sequence_id: int,
    sequence_azimuth: float,
    sequence_started_at: str,
    detection_created_at: str,
    detection_azimuth: float,
) -> dict[str, Path]:
    """
    Return a set of filepaths to store the detection, image and label
    provided the parameters.

    Returns:
      filepath_image (Path): path to the detection image (raw without bounding boxes)
      filepath_label (Path): path to the detection label txt file
      filepath_prediction (Path): path to the image with bouding box drawn on top
    """
    dir_sequence = (
        save_dir
        / organization_name
        / f"{_format_api_datetime_str(sequence_started_at)}_{camera_name}-{int(sequence_azimuth)}_sequence-{sequence_id}"
    )
    filepath_stem = f"{organization_name}_{camera_name}-{int(detection_azimuth)}_{_format_api_datetime_str(detection_created_at)}"
    return {
        "filepath_image": dir_sequence / "images" / f"{filepath_stem}.jpg",
        "filepath_label": dir_sequence / "labels" / f"{filepath_stem}.txt",
        "filepath_prediction": dir_sequence / "predictions" / f"{filepath_stem}.jpg",
    }


def save_label(bboxes: str, filepath_label: Path) -> None:
    """
    Persist the label txt files using the bounding boxes (bboxes) as a string
    obtained from the API responses. This function formats the bboxes into a
    suitable structure for YOLO and saves it to the specified file path.

    Parameters:
        bboxes (str): A string representation of the bounding boxes, typically
                      in a list format where each box contains coordinates and
                      a confidence score.
        filepath_label (Path): The path where the label txt file will be saved.

    Returns:
        None
    """
    label_content = _format_api_bboxes_ultralytics(bboxes, class_id=0)
    filepath_label.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath_label, "w") as file:
        file.write(label_content)


def download_image(url: str, filepath_destination: Path, force: bool = False) -> None:
    """
    Download the image located at `url` and save it locally at
    `filepath_destination`. If the image has already been downloaded, it will
    skip the download unless the `force` flag is set to True, in which case
    it will download the image again regardless of its current presence or size.

    Parameters:
        url (str): The URL from which to download the image.
        filepath_destination (Path): The local file path where the image will be saved.
        force (bool, optional): A flag indicating whether to force re-download the image
                                if it already exists. Defaults to False.

    Returns:
        None
    """
    if (
        not force
        and filepath_destination.exists()
        and filepath_destination.is_file()
        and filepath_destination.stat().st_size > 0
    ):
        logging.info(f"skipping downloading again {url}")
    else:
        filepath_destination.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(url)
        if response.status_code == 200:
            with open(filepath_destination, "wb") as f:
                f.write(response.content)
        else:
            logging.warning(f"failed to download image from {url}")


def process_dataframe(df: pd.DataFrame, save_dir: Path) -> None:
    """
    Process the dataframe containing sequences and detections information.

    This function performs the following tasks:
    1. Downloads associated images based on the detection URLs.
    2. Creates YOLO formatted label files using the provided bounding boxes.
    3. Generates overlaid images with predictions visualized on top of the original images.
    4. Saves the dataframe with additional paths to the labels, images, and predictions.

    Parameters:
        df (pd.DataFrame): The input dataframe containing detection and sequence information.
        save_dir (Path): The directory where images, labels, and predictions will be stored.

    Returns:
        None
    """
    records = []
    for _, row in tqdm(df.iterrows()):

        dict_filepaths = _get_filepaths(
            save_dir=save_dir,
            organization_name=row["organization_name"],
            camera_name=row["camera_name"],
            sequence_id=row["sequence_id"],
            sequence_azimuth=row["sequence_azimuth"],
            sequence_started_at=row["sequence_started_at"],
            detection_created_at=row["detection_created_at"],
            detection_azimuth=row["detection_azimuth"],
        )
        save_label(
            bboxes=row["detection_bboxes"],
            filepath_label=dict_filepaths["filepath_label"],
        )
        download_image(
            url=row["detection_url"],
            filepath_destination=dict_filepaths["filepath_image"],
            force=False,
        )
        array_image = cv2.imread(str(dict_filepaths["filepath_image"]))

        with open(dict_filepaths["filepath_label"], "r") as file:
            txt_content = file.read()
            yolo_predictions = parse_yolo_prediction_txt_file(txt_content)
            array_image_overlayed_with_predictions = overlay_predictions(
                array_image=array_image, predictions=yolo_predictions
            )
            dict_filepaths["filepath_prediction"].parent.mkdir(
                parents=True, exist_ok=True
            )
            cv2.imwrite(
                str(dict_filepaths["filepath_prediction"]),
                array_image_overlayed_with_predictions,
            )

        record = {**row, **dict_filepaths}
        records.append(record)

    df_extra = pd.DataFrame(records)
    filepath_dataframe = save_dir / "sequences.csv"
    logging.info(f"Saving the generated dataframe in {filepath_dataframe}")
    df_extra.to_csv(filepath_dataframe, index=False)
