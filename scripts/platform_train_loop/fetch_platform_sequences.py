"""
CLI script to fetch sequences from the Pyronear platform API.

Usage:
  python fetch_platform_sequences.py --save-dir <path> --date-from <date> --date-end <date> --detections-limit <int> --detections-order-by <str> --loglevel <str>

Arguments:
  --save-dir (Path): directory to save the sequences
  --date-from (date): date in YYYY-MM-DD format to start the sequence fetching
  --date-end (date): date in YYYY-MM-DD format to end the sequence fetching, defaults to now()
  --detections-limit (int): maximum number of detections to fetch, defaults to 10
  --detections-order-by (str): whether to order the detections by created_at in descending or ascending order, defaults to ascending
  --loglevel (str): provide logging level for the script

Environment variables required:
  PLATFORM_API_ENDPOINT (str): API url endpoint. eg https://alertapi.pyronear.org
  PLATFORM_LOGIN (str): login
  PLATFORM_PASSWORD (str): password
  PLATFORM_ADMIN_LOGIN (str): admin login - useful to access /api/v1/organizations endpoints
  PLATFORM_ADMIN_PASSWORD (str): admin password - useful to access /api/v1/organizations endpoints
"""

import argparse
import concurrent.futures
import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import pyro_dataset.platform.api as api
import pyro_dataset.platform.utils as platform_utils
from pyro_dataset.utils import index_by


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
        "--detections-limit",
        help="Maximum number of detections to fetch",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--detections-order-by",
        help="Whether to order the detections by created_at in descending or ascending order",
        choices=["desc", "asc"],
        type=str,
        default="asc",
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
        logging.error("Invalid combination of --date-from and --date-end parameters")
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
            "PLATFORM_API_ENDPOINT is not set. eg. https://alertapi.pyronear.org"
        )
        return False
    elif not platform_login:
        logging.error("PLATFORM_LOGIN is not set")
        return False
    elif not platform_password:
        logging.error("PLATFORM_PASSWORD is not set")
        return False
    elif not platform_admin_login:
        logging.error("PLATFORM_ADMIN_LOGIN is not set")
        return False
    elif not platform_admin_password:
        logging.error("PLATFORM_ADMIN_PASSWORD is not set")
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
    assert date_from <= date_end, "date_from should be < date_end"
    result = []
    date = date_from
    while date < date_end:
        result.append(date)
        date = date + timedelta(days=1)
    return result


def _process_sequence(
    api_endpoint: str,
    sequence: dict,
    detections_limit: int,
    detections_order_by: str,
    indexed_cameras: dict,
    indexed_organizations: dict,
    access_token: str,
):
    """
    Process a single sequence to extract detections and convert them into records.

    Parameters:
        api_endpoint (str): The API endpoint to fetch data from.
        sequence (dict): The sequence data containing information about the sequence.
        indexed_cameras (dict): A dictionary of indexed cameras for easy access.
        indexed_organizations (dict): A dictionary of indexed organizations for easy access.
        access_token (str): The access token for authenticating API requests.

    Returns:
        list: A list of records extracted from the sequence detections.
    """
    detections = api.list_sequence_detections(
        api_endpoint=api_endpoint,
        sequence_id=sequence["id"],
        access_token=access_token,
        limit=detections_limit,
        desc=True if detections_order_by == "desc" else False,
    )
    records = []
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
    return records


def _fetch_sequences_for_date(api_endpoint: str, date: date, access_token: str) -> list:
    """
    Fetch sequences for a specific date.

    Parameters:
        api_endpoint (str): The API endpoint to fetch data from.
        date (date): The specific date to fetch sequences for.
        access_token (str): The access token for authenticating API requests.

    Returns:
        list: A list of sequences fetched for the specified date.
    """
    return api.list_sequences_for_date(
        api_endpoint=api_endpoint,
        date=date,
        limit=1000,
        offset=0,
        access_token=access_token,
    )


def fetch_all_sequences_within(
    date_from: date,
    date_end: date,
    detections_limit: int,
    detections_order_by: str,
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
    if len(dates) < 2:
        logging.info(f"Found {len(dates)} days: {dates}")
    else:
        logging.info(
            f"Found {len(dates)} days between {date_from:%Y-%m-%d} and {date_end:%Y-%m-%d}: [{dates[0]:%Y-%m-%d}, {dates[1]:%Y-%m-%d},..., {dates[-2]:%Y-%m-%d}, {dates[-1]:%Y-%m-%d}]"
        )

    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_date = {
            executor.submit(
                _fetch_sequences_for_date, api_endpoint, mdate, access_token
            ): mdate
            for mdate in dates
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_to_date), total=len(future_to_date)
        ):
            sequences.extend(future.result())

    logging.info(
        f"Collected {len(sequences)} sequences between {date_from:%Y-%m-%d} and {date_end:%Y-%m-%d}"
    )

    logging.info(
        f"Fetching all detections for the {len(sequences)} sequences between {date_from:%Y-%m-%d} and {date_end:%Y-%m-%d}"
    )
    # Creating records for making the dataframe
    records = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_sequence = {
            executor.submit(
                _process_sequence,
                api_endpoint,
                sequence,
                detections_limit,
                detections_order_by,
                indexed_cameras,
                indexed_organizations,
                access_token,
            ): sequence
            for sequence in sequences
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_to_sequence),
            total=len(future_to_sequence),
        ):
            records.extend(future.result())

    logging.info(f"Processed {len(records)} detections")

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
        detections_limit = args["detections_limit"]
        detections_order_by = args["detections_order_by"]
        logger.info(
            f"Fetching sequences from {date_from:%Y-%m-%d} until {date_end:%Y-%m-%d} and storing data in {save_dir} from the platform API {api_url}"
        )
        logger.info("Fetching an access token to authenticate API requests...")
        access_token = api.get_api_access_token(
            api_endpoint=platform_api_endpoint,
            username=platform_login,
            password=platform_password,
        )
        logger.info(
            "Succesfully fetched an access token to authenticate API requests ✔️"
        )
        access_token_admin = api.get_api_access_token(
            api_endpoint=platform_api_endpoint,
            username=platform_admin_login,
            password=platform_admin_password,
        )
        logger.info(
            "Succesfully fetched an admin access token to authenticate API requests ✔️"
        )
        headers = api.make_request_headers(access_token=access_token)
        df = fetch_all_sequences_within(
            date_from=date_from,
            date_end=date_end,
            detections_limit=detections_limit,
            detections_order_by=detections_order_by,
            api_endpoint=platform_api_endpoint,
            access_token=access_token,
            access_token_admin=access_token_admin,
        )

        filepath_api_results_csv = save_dir / "api_results.csv"
        logger.info(f"Save API results to {filepath_api_results_csv}")
        save_dir.mkdir(parents=True, exist_ok=True)
        platform_utils.append_dataframe_to_csv(
            df=df, filepath_csv=filepath_api_results_csv
        )

        logger.info("Processing all the detections")
        platform_utils.process_dataframe(df=df, save_dir=save_dir)
        args_content = {
            "date-from": str(args["date_from"]),
            "date-end": str(args["date_end"]),
            "date-now": datetime.now().strftime("%Y-%m-%d"),
            "save-dir": str(args["save_dir"]),
            "platform-login": platform_login,
            "platform-admin-login": platform_admin_login,
        }
        filepath_args_yaml = save_dir / "args.yaml"
        logger.info(f"Saving args run in {filepath_args_yaml}")
        platform_utils.append_yaml_run(filepath=filepath_args_yaml, data=args_content)
        logger.info("Done ✅")
