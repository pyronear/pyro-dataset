"""
CLI script to fetch one sequence from the Pyronear platform API.

Parameters:
  --save-dir (Path): directory to save the sequences
  --sequence-id (int): sequence id to fetch
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
from pathlib import Path

import pandas as pd

import pyro_dataset.platform.api as api
import pyro_dataset.platform.utils as platform_utils
from pyro_dataset.utils import yaml_write


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
        "--sequence-id",
        help="sequence id to be fetched",
        type=int,
        required=True,
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


def fetch_sequence(
    sequence_id: int,
    api_endpoint: str,
    access_token: str,
    access_token_admin: str,
) -> list[dict]:
    """
    Fetch a sequence of detections from the Pyronear platform API.

    Parameters:
        sequence_id (int): The ID of the sequence to be fetched.
        api_endpoint (str): The API endpoint URL for the Pyronear platform.
        access_token (str): The access token for authenticating API requests.
        access_token_admin (str): The admin access token for accessing organization information.

    Returns:
        list[dict]: A list of dictionaries containing detection records, each including metadata about the organization, camera, and detection details.
    """
    sequence = api.get_sequence(
        api_endpoint=api_endpoint,
        sequence_id=sequence_id,
        access_token=access_token,
    )
    detections = api.list_sequence_detections(
        api_endpoint=api_endpoint,
        sequence_id=sequence_id,
        access_token=access_token,
    )
    camera = api.get_camera(
        api_endpoint=api_endpoint,
        camera_id=sequence["camera_id"],
        access_token=access_token,
    )
    organization = api.get_organization(
        api_endpoint=api_endpoint,
        organization_id=camera["organization_id"],
        access_token=access_token_admin,
    )
    return [
        platform_utils.to_record(
            detection=detection,
            camera=camera,
            organization=organization,
            sequence=sequence,
        )
        for detection in detections
    ]


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
        sequence_id = args["sequence_id"]
        logger.info(
            f"Fetching sequence id {sequence_id}  and storing data in {save_dir} from the platform API {api_url}"
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
        df = pd.DataFrame(
            fetch_sequence(
                sequence_id=sequence_id,
                api_endpoint=platform_api_endpoint,
                access_token=access_token,
                access_token_admin=access_token_admin,
            )
        )
        platform_utils.process_dataframe(df=df, save_dir=save_dir)
        args_content = {
            "sequence-id": args["sequence_id"],
            "save-dir": str(args["save_dir"]),
            "platform-login": platform_login,
            "platform-admin-login": platform_admin_login,
        }
        yaml_write(to=save_dir / "args.yaml", data=args_content)
        logger.info(f"Done ✅")
