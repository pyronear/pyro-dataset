"""
CLI script to release the pyronear datasets.
"""

import argparse
import logging
import os
import shutil
from pathlib import Path

import boto3
import requests
from botocore.exceptions import NoCredentialsError


def make_cli_parser() -> argparse.ArgumentParser:
    """
    Make the CLI parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        help="server version scheme, always incrementing",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--github-owner",
        help="Github Owner",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--github-repo",
        help="Github Owner",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--s3-bucket",
        help="S3 bucket to save the release",
        default="pyro-datasets",
        type=str,
    )
    parser.add_argument(
        "-log",
        "--loglevel",
        default="info",
        help="Provide logging level. Example --loglevel debug, default=warning",
    )
    return parser


def is_valid_version(version: str) -> bool:
    """
    Check whether the `version` is valid.
    """
    parts_version = version.split(".")
    return len(parts_version) == 3


def validate_parsed_args(args: dict) -> bool:
    """
    Return whether the parsed args are valid.
    """
    if not is_valid_version(args["version"]):
        logging.error(f"invalid --version, should follow semver eg. v1.0.2")
        return False
    return True


def create_archive(source_folder: Path, archive_name: str) -> Path:
    """
    Create an archive tar.gz file of the source_folder and using `archive_name`
    to name it.
    """
    # Create a tar.gz archive of the source folder
    archive_path = Path("/tmp") / archive_name
    shutil.make_archive(
        str(archive_path).replace(".tar.gz", ""), "gztar", source_folder
    )
    return archive_path


def upload_to_s3(filepath: Path, bucket_name: str, s3_folder: str) -> None:
    """
    Upload the filepath to the `bucket_name/s3_folder/`
    """
    s3_client = boto3.client("s3")
    s3_key = f"{s3_folder}/{filepath.name}"
    logging.info(f"S3 key: {s3_key}")
    try:
        s3_client.upload_file(
            filepath,
            bucket_name,
            s3_key,
            ExtraArgs={"ACL": "private"},
        )
        logging.info(f"Uploaded {filepath} to s3://{s3_key}")
    except FileNotFoundError:
        logging.error(f"The file {filepath} was not found.")
    except NoCredentialsError:
        logging.error("Credentials not available.")


def create_release(
    owner: str,
    repo: str,
    version: str,
    body: str,
    github_access_token: str,
) -> dict:
    """
    Create a release using the Github API.
    Returns the response as a dict.
    """

    url = f"https://api.github.com/repos/{owner}/{repo}/releases"

    # Set the headers
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {github_access_token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # Define the data to be sent in the POST request
    data = {
        "tag_name": version,
        "target_commitish": "main",
        "name": f"Pyro Datasets {version}",
        "body": body,
        "draft": False,
        "prerelease": False,
        "generate_release_notes": False,
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()


def upload_asset(
    owner: str,
    repo: str,
    filepath_asset: Path,
    name: str,
    release_id: int,
    github_access_token: str,
) -> dict:
    """
    Upload asset for a given release identified with `release_id`.
    """
    url = f"https://uploads.github.com/repos/{owner}/{repo}/releases/{release_id}/assets?name={name}"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {github_access_token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/octet-stream",
    }
    # Open the file in binary mode and make the POST request
    with open(filepath_asset, "rb") as file:
        response = requests.post(url, headers=headers, data=file)
        return response.json()


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        exit(1)
    else:
        logger.info(args)
        version = args["version"]
        owner = args["github_owner"]
        repo = args["github_repo"]
        s3_bucket = args["s3_bucket"]
        github_access_token = os.getenv("GITHUB_ACCESS_TOKEN")
        archive_name = f"pyro-datasets-{version}.tar.gz"
        s3_folder = "github-releases"
        assert github_access_token, "You must set the env variable GITHUB_ACCESS_TOKEN"
        body_release = f"Pyronear datasets for early forest fire detection 🔥 \n\nDownload the datasets from the following S3 location: {s3_bucket}/{s3_folder}/{archive_name}"
        response_release = create_release(
            owner=owner,
            repo=repo,
            body=body_release,
            version=version,
            github_access_token=github_access_token,
        )
        release_id = response_release["id"]
        logger.info(f"release created: {response_release}")
        logger.info(f"release_id: {release_id}")
        dir_datasets = Path("./data/processed/")
        logger.info(
            f"Creating an archive named {archive_name} of the datasets located in {dir_datasets}"
        )
        filepath_archive = create_archive(
            source_folder=dir_datasets,
            archive_name=archive_name,
        )
        logger.info(f"uploading dataset archive to S3")
        upload_to_s3(
            filepath=filepath_archive,
            bucket_name=s3_bucket,
            s3_folder="github-releases",
        )
        logger.info(f"uploading associated report and plots")
        logger.info(f"uploading report.yaml")
        upload_asset(
            owner=owner,
            repo=repo,
            release_id=release_id,
            filepath_asset=Path("./data/reporting/wildfire/report.yaml"),
            name="report.yaml",
            github_access_token=github_access_token,
        )
        filepaths_plot = list(Path("./data/reporting/wildfire/plots/").glob("*.html"))
        for filepath_plot in filepaths_plot:
            logger.info(f"uploading plot asset: {filepath_plot}")
            upload_asset(
                owner=owner,
                repo=repo,
                release_id=release_id,
                filepath_asset=filepath_plot,
                name=filepath_plot.name,
                github_access_token=github_access_token,
            )
        logger.info(f"Removing local archive {filepath_archive}")
        filepath_archive.unlink()
        logger.info(f"Done ✅")
        exit(0)
