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
import pyro_dataset.platform.utils as platform_utils
from pyro_dataset.utils import index_by, yaml_write
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


def get_dates_within(date_from: date, date_end: date) -> list[date]:
    """
    Collect all the days between `date_from` and `date_end` as
    datetime objects. This function generates a list of dates
    starting from `date_from` up to, but not including, `date_end`.

    Parameters:
        date_from (date): The starting date for the range.
        date_end (date): The ending date for the range.

    Returns:
        list[date]: A list of date objects representing each day in the range.

    Raises:
        AssertionError: If `date_from` is greater than `date_end`.
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
            record = platform_utils.to_record(
                detection=detection,
                camera=camera,
                organization=organization,
                sequence=sequence,
            )
            records.append(record)
    df = pd.DataFrame(records)
    logger.info(df.head())
    return df


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

        platform_utils.process_dataframe(df=df, save_dir=save_dir)
        args_content = {
            "date-from": str(args["date_from"]),
            "date-end": str(args["date_end"]),
            "save-dir": str(args["save_dir"]),
            "platform-login": platform_login,
            "platform-admin-login": platform_admin_login,
        }
        yaml_write(to=save_dir / "args.yaml", data=args_content)
        logger.info(f"Done ✅")
