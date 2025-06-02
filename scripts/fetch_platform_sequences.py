"""
CLI script to release the pyronear datasets.

Environment variables required:
  PLATFORM_API_ENDPOINT (str): API url endpoint. eg https://alertapi.pyronear.org
  PLATFORM_LOGIN (str): login
  PLATFORM_PASSWORD (str): password
"""


import argparse
import logging
import os
import sys
import cv2
from datetime import datetime, timedelta, date
from pathlib import Path
import shutil
from tqdm import tqdm

import pandas as pd
import requests
import argparse
from pathlib import Path

from pyro_dataset.yolo.utils import parse_yolo_prediction_txt_file, overlay_predictions

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
        help='Directory to save the sequences',
        type=Path,
        default=Path("./data/raw/pyronear-platform/sequences/"),
    )
    parser.add_argument(
        "--date-from",
        help='Date in YYYY-MM-DD format',
        type=valid_date,
        required=True,
    )
    parser.add_argument(
        "--date-end",
        help='Date in YYYY-MM-DD format, defaults to now.',
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
    return True


def validate_available_env_variables() -> bool:
    """
    Check whether the environment variables required for
    hitting the API are properly set.

    PLATFORM_API_ENDPOINT (str): API url endpoint
    PLATFORM_LOGIN (str): login
    PLATFORM_PASSWORD (str): password
    """
    platform_api_endpoint = os.getenv("PLATFORM_API_ENDPOINT")
    platform_login = os.getenv("PLATFORM_LOGIN")
    platform_password = os.getenv("PLATFORM_LOGIN")
    if not platform_api_endpoint:
        logging.error(f"PLATFORM_API_ENDPOINT is not set. eg. https://alertapi.pyronear.org")
        return False
    elif not platform_login:
        logging.error(f"PLATFORM_LOGIN is not set")
        return False
    elif not platform_password:
        logging.error(f"PLATFORM_PASSWORD is not set")
        return False
    else:
        return True


def get_api_access_token(api_endpoint: str, username: str, password: str) -> str:
    """
    Fetch an access token that is used to authenticate and authorize
    subsequent API requests.
    """
    url = f"{api_endpoint}/api/v1/login/creds"
    response = requests.post(
        url,
        data={"username": username, "password": password},
        timeout=5,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def api_get(route: str, headers: dict[str, str]):
    """
    Issue a GET request against the API route with the provided headers.

    Returns:
        response (dict): JSON response

    Raises:
        Exception: when the API request fails
    """
    logging.info(f"Making an HTTP request to route {route}")
    response = requests.get(route, headers=headers)
    try:
        return response.json()
    except:
        raise Exception(f"API Error: {response.status_code} {response.text}")


def list_cameras(api_endpoint: str, headers: dict[str, str]) -> list[dict]:
    """
    List all cameras using the platform API.
    """
    url = f"{api_endpoint}/api/v1/cameras/"
    return api_get(route=url, headers=headers)


def get_camera(api_endpoint: str, camera_id: int, headers: dict[str, str]) -> dict:
    """
    Fetch the information of a specific camera `camera_id`.
    """
    url = f"{api_endpoint}/api/v1/cameras/{camera_id}"
    return api_get(route=url, headers=headers)


def list_organizations(api_endpoint: str, headers: dict[str, str]) -> list[dict]:
    url = f"{api_endpoint}/api/v1/organizations/"
    return api_get(route=url, headers=headers)


def list_sequences_for_date(api_endpoint: str, date: date, limit: int, offset: int, headers: dict[str, str]) -> list[dict]:
    """
    List sequences for a specified date, limit the result to `limit` and
    use an `offset` if the results are paginated."""
    url = f"{api_endpoint}/api/v1/sequences/all/fromdate?from_date={date:%Y-%m-%d}&limit={limit}&offset={offset}"
    return api_get(route=url, headers=headers)


def get_detections(api_endpoint: str, detection_id: int, headers: dict[str, str]) -> dict:
    """
    Fetch the information of a specific detection.
    """
    url = f"{api_endpoint}/api/v1/detections/{detection_id}"
    return api_get(route=url, headers=headers)


def list_sequence_detections(api_endpoint: str, sequence_id: int, headers: dict[str, str]) -> list[dict]:
    url = f"{api_endpoint}/api/v1/sequences/{sequence_id}/detections"
    return api_get(route=url, headers=headers)


def make_request_headers(access_token: str) -> dict[str, str]:
    """
    Make the HTTP request headers.
    """
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }


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
        headers: dict[str, str],
) -> pd.DataFrame:
    """
    Fetch all sequences and detections between `date_from` and
    `date_end`

    Returns
        df (pd.DataFrame): dataframe containing all the details
    """
    cameras = list_cameras(api_endpoint=api_endpoint, headers=headers)
    indexed_cameras = index_by(cameras, key="id")
    organizations = list_organizations(api_endpoint=api_endpoint, headers=headers)
    indexed_organizations = index_by(organizations, key="id")

    logging.info(f"Fetching sequences between {date_from:%Y-%m-%d} and {date_end:%Y-%m-%d}")
    sequences = []
    dates = get_dates_within(date_from=date_from, date_end=date_end)
    for date in tqdm(dates):
        xs = list_sequences_for_date(
            api_endpoint=api_endpoint,
            date=date,
            limit=1000,
            offset=0,
            headers=headers
        )
        sequences.extend(xs)

    logging.info(f"Collected {len(sequences)} sequences between {date_from:%Y-%m-%d} and {date_end:%Y-%m-%d}")

    # Creating records for making the dataframe
    records = []
    for sequence in tqdm(sequences):
        detections = list_sequence_detections(
            api_endpoint=api_endpoint,
            sequence_id=sequence["id"],
            headers=headers,
        )
        for detection in detections:
            print("detection:")
            print(detection)
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
    return "\n".join(f"{class_id} " + " ".join(map(str, bbox)) for bbox in xs)


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
    dir_sequence = save_dir / organization_name / f"{_format_api_datetime_str(sequence_started_at)}_{camera_name}-{int(sequence_azimuth)}_sequence-{sequence_id}"
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


def download_image(url: str, filepath_destination: Path, force_download: bool = False) -> None:
    """
    Download the image located at `url` and save it locally at
    `filepath_destination`. If already downloaded, it will skip it
    unless the `force_download` flag is set to True.
    """
    if not force_download and filepath_destination.exists() and filepath_destination.is_file() and filepath_destination.stat().st_size > 0:
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
        save_label(bboxes=row["detection_bboxes"], filepath_label=dict_filepaths["filepath_label"])
        download_image(
            url=row["detection_url"],
            filepath_destination=dict_filepaths["filepath_image"],
            force_download=False,
        )
        array_image = cv2.imread(str(dict_filepaths["filepath_image"]))

        with open(dict_filepaths["filepath_label"], "r") as file:
            txt_content = file.read()
            yolo_predictions = parse_yolo_prediction_txt_file(txt_content)
            array_image_overlayed_with_predictions = overlay_predictions(array_image=array_image, predictions=yolo_predictions)
            dict_filepaths["filepath_prediction"].parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dict_filepaths["filepath_prediction"]), array_image_overlayed_with_predictions)

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
        if not platform_login or not platform_password or not platform_api_endpoint:
            logger.error("Missing platform credentials...")
            exit(1)

        save_dir = args["save_dir"]
        date_from = args["date_from"]
        date_end = args["date_end"]
        logger.info(f"Fetching sequences from {date_from:%Y-%m-%d} until {date_end:%Y-%m-%d} and storing data in {save_dir} from the platform API {api_url}")
        logger.info("Fetching an access token to authenticate API requests...")
        # access_token = get_api_access_token(
        #     api_endpoint=platform_api_endpoint,
        #     username=platform_login,
        #     password=platform_password,
        # )
        # logger.info("Succesfully fetched an acess token to authenticate API requests ✔️")
        # headers = make_request_headers(access_token=access_token)
        # df = fetch_all_sequences_within(date_from=date_from, date_end=date_end, api_endpoint=platform_api_endpoint, headers=headers)
        #
        # save_dir.mkdir(parents=True, exist_ok=True)
        # df.to_csv(save_dir / "api_results.csv", index=False)
        # logger.info(f"Process the dataframe")

        # FIXME: Remove
        df = pd.read_csv(save_dir / "api_results.csv")
        df.info()

        process_dataframe(df=df, save_dir=save_dir)
