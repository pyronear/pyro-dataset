"""
CLI script to release the pyronear datasets.

Environment variables required:
    PLATFORM_API_ENDPOINT (str): API url endpoint. eg https://alertapi.pyronear.org
    PLATFORM_LOGIN (str): login
    PLATFORM_PASSWORD (str): password

TODO:
- [ ] Use dates instead of datetimes in date_from date_end
"""



import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, date
from pathlib import Path
import shutil
from tqdm import tqdm

import pandas as pd
import requests
import argparse
from pathlib import Path

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

# from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
# load_dotenv(override=True)

# logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")


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


# FIXME
def download_image(url: str, path: str):
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, "wb") as f:
            f.write(response.content)
    else:
        logging.warning(f"Failed to download image from {url}")


# ----------- Config ------------
# api_url = os.environ.get("API_URL") + "/api/v1"
# superuser_login = os.environ.get("SUPERADMIN_LOGIN")
# superuser_pwd = os.environ.get("SUPERADMIN_PWD")
# -------------------------------

# auth_headers = {
#     "Authorization": f"Bearer {get_token(api_url, superuser_login, superuser_pwd)}",
#     "Content-Type": "application/json",
# }

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


def fetch_sequences_for_date(
    date_str: str,
    api_url: str,
    api_endpoint: str,
    headers: dict[str, str],
    save_dir: Path,
) -> None:
    logging.info(f"Fetching sequences for date: {date_str}")

    # Step 1: Get cameras and build camera_id -> org_id mapping
    url_cameras_list = f"{api_endpoint}"
    camera_list = api_get(f"{api_url}/cameras/", headers)
    logging.info(f"Found {len(camera_list)}: {camera_list}")
    # print(f"camera list: {camera_list}")
    camera_map = {}
    for cam in camera_list:
        camera_map[cam["id"]] = cam["organization_id"]

    # Step 2: Get all sequences for the given date
    # sequences = api_get(
    #     f"{api_url}/sequences/all/fromdate?from_date={date_str}&limit=1000&offset=0",
    #     auth_headers,
    # )
    # print(f"sequences: {sequences}")
    #
    # # Organize directory structure
    # for sequence in sequences:
    #     sequence_id = sequence["id"]
    #     org_id = camera_map.get(sequence["camera_id"])
    #     if org_id is None:
    #         logging.warning(f"Camera ID {sequence['camera_id']} not found in map.")
    #         continue
    #
    #     org_dir = os.path.join(output_root, f"sdis_{org_id}")
    #     os.makedirs(org_dir, exist_ok=True)
    #
    #     csv_path = os.path.join(org_dir, "sequences.csv")
    #
    #     # ✅ Load existing sequence IDs from CSV (if exists)
    #     existing_ids = set()
    #     if os.path.exists(csv_path):
    #         try:
    #             existing_df = pd.read_csv(csv_path, usecols=["sequence_id"])
    #             existing_ids = set(existing_df["sequence_id"].astype(int))
    #         except Exception as e:
    #             logging.warning(f"Failed to load existing CSV: {csv_path} ({e})")
    #
    #     if sequence_id in existing_ids:
    #         logging.info(f"Sequence {sequence_id} already exists in CSV, skipping.")
    #         continue
    #
    #     # --- Proceed only if not already stored ---
    #     seq_dir = os.path.join(org_dir, f"sequence_{sequence['id']}")
    #     os.makedirs(seq_dir, exist_ok=True)
    #
    #     csv_path = os.path.join(org_dir, "sequences.csv")
    #
    #     # Step 3: Get detections for the sequence
    #     detections = api_get(
    #         f"{api_url}/sequences/{sequence['id']}/detections", auth_headers
    #     )
    #
    #     # Step 4: Save images and metadata
    #     rows = []
    #     for det in detections:
    #         image_url = det["url"]
    #         image_filename = f"detection_{det['id']}.jpg"
    #         image_path = os.path.join(seq_dir, image_filename)
    #         download_image(image_url, image_path)
    #
    #         rows.append(
    #             {
    #                 "sequence_id": sequence["id"],
    #                 "camera_id": sequence["camera_id"],
    #                 "organization_id": org_id,
    #                 "is_wildfire": sequence.get("is_wildfire", None),
    #                 "started_at": sequence["started_at"],
    #                 "last_seen_at": sequence["last_seen_at"],
    #                 "detection_id": det["id"],
    #                 "image_path": image_path,
    #                 "created_at": det["created_at"],
    #                 "azimuth": det["azimuth"],
    #                 "bucket_key": det["bucket_key"],
    #                 "bboxes": det["bboxes"],
    #             }
    #         )
    #
    #     # Append to CSV file
    #     df = pd.DataFrame(rows)
    #     if not df.empty:
    #         if os.path.exists(csv_path):
    #             df.to_csv(csv_path, mode="a", index=False, header=False)
    #         else:
    #             df.to_csv(csv_path, index=False)
    #
    #     print(rows)
    # logging.info("✅ Done!")


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
            camera = indexed_cameras[sequence["camera_id"]]
            organization = indexed_organizations[camera["organization_id"]]
            record = {
                "sequence_id": sequence["id"],
                "camera_id": sequence["camera_id"],
                "camera_name": camera["name"],
                "organization_id": camera["organization_id"],
                "organization_name": organization["name"],
                "camera_lat": camera["lat"],
                "camera_lon": camera["lon"],
                "camera_is_trustable": camera["is_trustable"],
                "camera_angle_of_view": camera["angle_of_view"],
                "detection_id": detection["id"],
                "is_wildfire": sequence["camera_id"],
                "sequence_started_at": sequence["started_at"],
                "sequence_last_seen_at": sequence["last_seen_at"],
                "sequence_azimuth": sequence["azimuth"],
                "detection_azimuth": detection["azimuth"],
                "detection_url": detection["url"],
                "detection_bboxes": detection["bboxes"],
                "detection_bucket_key": detection["bucket_key"],
            }
            records.append(record)
    df = pd.DataFrame(records)
    logger.info(df.head())
    return df


def process_dataframe(df: pd.DataFrame, save_dir: Path) -> None:
    """
    Process the dataframe of sequences and detections information. Downloading
    the associated images and creating the yolo txt label files.
    """
    return None


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
        access_token = get_api_access_token(
            api_endpoint=platform_api_endpoint,
            username=platform_login,
            password=platform_password,
        )
        logger.info("Succesfully fetched an acess token to authenticate API requests ✔️")
        headers = make_request_headers(access_token=access_token)
        df = fetch_all_sequences_within(date_from=date_from, date_end=date_end, api_endpoint=platform_api_endpoint, headers=headers)

        save_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_dir / "results.csv", index=False)
        logger.info(f"Process the dataframe")
        process_dataframe(df=df, save_dir=save_dir)

        # Download the images into the right folder structure
        # Create the label files into the right folder stucture (alongside the images?)
