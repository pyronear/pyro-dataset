"""
CLI script to fetch sequences from the Pyronear platform API.

Parameters:
  --save-dir (Path): directory to save the sequences
  --date-from (date): date in YYYY-MM-DD format to start the sequence fetching
  --date-end (date): date in YYYY-MM-DD format to end the sequence fetching, defaults to now()
  --loglevel (str): provide logging level for the script

Environment variables required:
  PLATFORM_API_ENDPOINT (str): API url endpoint. eg https://alertapi.pyronear.org
  PLATFORM_LOGIN (str): login
  PLATFORM_PASSWORD (str): password
  PLATFORM_ADMIN_LOGIN (str): admin login - useful to access /api/v1/organizations endpoints
  PLATFORM_ADMIN_PASSWORD (str): admin password - useful to access /api/v1/organizations endpoints
"""

import argparse
import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

import pyro_dataset.platform.api as api
from pyro_dataset.yolo.utils import (
    overlay_predictions,
    parse_yolo_prediction_txt_file,
    xyxyn2xywhn,
)


def valid_date(s: str):
    """
    Datetime parser for the CLI.
    """
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        msg = "not a valid date: {0!r}".format(s)
        raise argparse.ArgumentTypeError(msg)


def make_cli_parser() -> argparse.ArgumentParser:
    """
    Make the CLI parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-dir",
        help="Directory to save the sequences",
        type=Path,
        default=Path("./data/raw/pyronear-platform/sequences/"),
    )
    parser.add_argument(
        "--date-from",
        help="Date in YYYY-MM-DD format",
        type=valid_date,
        required=True,
    )
    parser.add_argument(
        "--date-end",
        help="Date in YYYY-MM-DD format, defaults to now.",
        type=valid_date,
        default=datetime.now().date(),
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
    if args["date_from"] > args["date_end"]:
        logging.error(f"Invalid combination of --date-from and --date-end parameters")
        return False
    return True


def validate_available_env_variables() -> bool:
    """
    Check whether the environment variables required for
    hitting the API are properly set.

    PLATFORM_API_ENDPOINT (str): API url endpoint
    PLATFORM_LOGIN (str): login
    PLATFORM_PASSWORD (str): password
    PLATFORM_ADMIN_LOGIN (str): admin login
    PLATFORM_ADMIN_PASSWORD (str): admin password
    """
    platform_api_endpoint = os.getenv("PLATFORM_API_ENDPOINT")
    platform_login = os.getenv("PLATFORM_LOGIN")
    platform_password = os.getenv("PLATFORM_LOGIN")
    platform_admin_login = os.getenv("PLATFORM_ADMIN_LOGIN")
    platform_admin_password = os.getenv("PLATFORM_ADMIN_PASSWORD")
    if not platform_api_endpoint:
        logging.error(
            f"PLATFORM_API_ENDPOINT is not set. eg. https://alertapi.pyronear.org"
        )
        return False
    elif not platform_login:
        logging.error(f"PLATFORM_LOGIN is not set")
        return False
    elif not platform_password:
        logging.error(f"PLATFORM_PASSWORD is not set")
        return False
    elif not platform_admin_login:
        logging.error(f"PLATFORM_ADMIN_LOGIN is not set")
        return False
    elif not platform_admin_password:
        logging.error(f"PLATFORM_ADMIN_PASSWORD is not set")
        return False
    else:
        return True


def index_by(xs: list[dict], key: str) -> dict[str, dict]:
    """
    Index a collection of dicts `xs` by the provided `key`.
    """
    return {x[key]: x for x in xs}


def get_dates_within(date_from: date, date_end: date) -> list[date]:
    """
    Collect all the days between `date_from` and `date_end` as
    datetimes.
    """
    assert date_from <= date_end, f"date_from should be < date_end"
    result = []
    date = date_from
    while date < date_end:
        result.append(date)
        date = date + timedelta(days=1)
    return result


def fetch_all_sequences_within(
    date_from: date,
    date_end: date,
    api_endpoint: str,
    access_token: str,
    access_token_admin: str,
) -> pd.DataFrame:
    """
    Fetch all sequences and detections between `date_from` and
    `date_end`

    Returns
        df (pd.DataFrame): dataframe containing all the details
    """
    cameras = api.list_cameras(api_endpoint=api_endpoint, access_token=access_token)
    indexed_cameras = index_by(cameras, key="id")
    organizations = api.list_organizations(
        api_endpoint=api_endpoint,
        access_token=access_token_admin,
    )
    indexed_organizations = index_by(organizations, key="id")

    logging.info(
        f"Fetching sequences between {date_from:%Y-%m-%d} and {date_end:%Y-%m-%d}"
    )
    sequences = []
    dates = get_dates_within(date_from=date_from, date_end=date_end)
    for date in tqdm(dates):
        xs = api.list_sequences_for_date(
            api_endpoint=api_endpoint,
            date=date,
            limit=1000,
            offset=0,
            access_token=access_token,
        )
        sequences.extend(xs)

    logging.info(
        f"Collected {len(sequences)} sequences between {date_from:%Y-%m-%d} and {date_end:%Y-%m-%d}"
    )

    # Creating records for making the dataframe
    records = []
    for sequence in tqdm(sequences):
        detections = api.list_sequence_detections(
            api_endpoint=api_endpoint,
            sequence_id=sequence["id"],
            access_token=access_token,
        )
        for detection in detections:
            camera = indexed_cameras[sequence["camera_id"]]
            organization = indexed_organizations[camera["organization_id"]]
            record = {
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
            records.append(record)
    df = pd.DataFrame(records)
    logger.info(df.head())
    return df


def _format_api_datetime_str(datetime_str: str) -> str:
    """
    Format a datetime string returned by the API in a format that can be
    used in filepaths for instance.

    Eg.
      >>> _format_api_datetime_str(2025-05-30T10:02:55.981732)
      2025-05-30T10-02-55
    """
    return datetime_str.replace(":", "-").split(".")[0]


def _format_api_bboxes_ultralytics(detection_bboxes: str, class_id: int = 0) -> str:
    """
    Format the bboxes returned by the platform API into ultralytics format.
    """
    xs = eval(detection_bboxes)

    def format_bbox(x):
        conf = x[4]
        xyxyn = x[0:4]
        xywhn = xyxyn2xywhn(np.array(xyxyn))
        return f"{class_id} {' '.join(map(str, xywhn.tolist()))} {conf}"

    return "\n".join(map(format_bbox, xs))


def _get_local_filepaths(
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
    Return a set of filepaths to store the detection image and label
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
    Persist the label txt files using the bboxes (string) from the API responses
    """
    label_content = _format_api_bboxes_ultralytics(bboxes, class_id=0)
    filepath_label.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath_label, "w") as file:
        file.write(label_content)


def download_image(url: str, filepath_destination: Path, force: bool = False) -> None:
    """
    Download the image located at `url` and save it locally at
    `filepath_destination`. If already downloaded, it skips it unless the
    `force` flag is set to True.
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
    Process the dataframe of sequences and detections information.

    It does the following:
    1. Downloads the associated images
    2. Create the yolo txt label files
    3. Generate the overlaid predictions
    4. Persist the dataframe with added paths to the labels, images and predictions.
    """
    records = []
    for _, row in df.iterrows():

        dict_filepaths = _get_local_filepaths(
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


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        exit(1)
    elif not validate_available_env_variables():
        exit(1)
    else:
        logger.info(args)
        platform_api_endpoint = os.getenv("PLATFORM_API_ENDPOINT")
        api_url = f"{platform_api_endpoint}/api/v1"
        platform_login = os.getenv("PLATFORM_LOGIN")
        platform_password = os.getenv("PLATFORM_PASSWORD")
        platform_admin_login = os.getenv("PLATFORM_ADMIN_LOGIN")
        platform_admin_password = os.getenv("PLATFORM_ADMIN_PASSWORD")
        if (
            not platform_login
            or not platform_password
            or not platform_api_endpoint
            or not platform_admin_login
            or not platform_admin_password
        ):
            logger.error("Missing platform credentials...")
            exit(1)

        save_dir = args["save_dir"]
        date_from = args["date_from"]
        date_end = args["date_end"]
        logger.info(
            f"Fetching sequences from {date_from:%Y-%m-%d} until {date_end:%Y-%m-%d} and storing data in {save_dir} from the platform API {api_url}"
        )
        logger.info("Fetching an access token to authenticate API requests...")
        access_token = api.get_api_access_token(
            api_endpoint=platform_api_endpoint,
            username=platform_login,
            password=platform_password,
        )
        logger.info("Succesfully fetched an acess token to authenticate API requests ✔️")
        access_token_admin = api.get_api_access_token(
            api_endpoint=platform_api_endpoint,
            username=platform_admin_login,
            password=platform_admin_password,
        )
        logger.info(
            "Succesfully fetched an admin acess token to authenticate API requests ✔️"
        )
        headers = api.make_request_headers(access_token=access_token)
        df = fetch_all_sequences_within(
            date_from=date_from,
            date_end=date_end,
            api_endpoint=platform_api_endpoint,
            access_token=access_token,
            access_token_admin=access_token_admin,
        )

        filepath_api_results_csv = save_dir / "api_results.csv"
        logger.info(f"Save API results to {filepath_api_results_csv}")
        save_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath_api_results_csv, index=False)

        process_dataframe(df=df, save_dir=save_dir)
        logger.info(f"Done ✅")
