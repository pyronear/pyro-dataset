"""
API client functions to interact with the Pyronear platform.

This module provides functions to authenticate with the Pyronear API,
retrieve access tokens, and perform various operations related to
cameras, organizations, sequences, and detections.

Functions:
- get_api_access_token: Retrieve an API access token using credentials.
- api_get: Make a GET request to a specified API route.
- list_cameras: Retrieve a list of all cameras.
- get_camera: Fetch information for a specific camera.
- list_organizations: Retrieve a list of all organizations.
- get_organization: Fetch information for a specific organization.
- list_sequences_for_date: List sequences for a specific date.
- get_detections: Fetch information for a specific detection.
- get_sequence: Fetch information for a specific sequence.
- list_sequence_detections: List detections for a specific sequence.
"""

import logging
from datetime import date

import requests


def get_api_access_token(api_endpoint: str, username: str, password: str) -> str:
    """
    Retrieve the API access token using the provided credentials.

    Args:
        api_endpoint (str): The base URL of the API endpoint.
        username (str): The username for API authentication.
        password (str): The password for API authentication.

    Returns:
        str: The access token for API authentication.

    Raises:
        Exception: If the request to retrieve the access token fails.
    """
    url = f"{api_endpoint}/api/v1/login/creds"
    response = requests.post(
        url,
        data={"username": username, "password": password},
        timeout=5,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def api_get(route: str, access_token: str):
    """
    Issue a GET request against the API route with the provided headers.

    Returns:
        response (dict): JSON response

    Raises:
        Exception: when the API request fails
    """
    headers = make_request_headers(access_token=access_token)
    logging.debug(f"Making an HTTP request to route {route}")
    response = requests.get(route, headers=headers)
    try:
        return response.json()
    except:
        raise Exception(f"API Error: {response.status_code} {response.text}")


def make_request_headers(access_token: str) -> dict[str, str]:
    """
    Create headers for API requests.

    Args:
        access_token (str): The access token for API authentication.

    Returns:
        dict[str, str]: A dictionary containing the headers for the API request.
    """
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }


def list_cameras(api_endpoint: str, access_token: str) -> list[dict]:
    """
    List all cameras using the platform API.

    Args:
        api_endpoint (str): The base URL for the API endpoint.
        access_token (str): The access token for API authentication.

    Returns:
        list[dict]: A list of dictionaries containing camera information.
    """
    url = f"{api_endpoint}/api/v1/cameras/"
    return api_get(route=url, access_token=access_token)


def get_camera(api_endpoint: str, camera_id: int, access_token: str) -> dict:
    """
    Fetch the information of a specific camera.

    Args:
        api_endpoint (str): The base URL for the API endpoint.
        camera_id (int): The ID of the camera to fetch.
        access_token (str): The access token for API authentication.

    Returns:
        dict: A dictionary containing the camera information.
    """
    url = f"{api_endpoint}/api/v1/cameras/{camera_id}"
    return api_get(route=url, access_token=access_token)


def list_organizations(api_endpoint: str, access_token: str) -> list[dict]:
    """
    List all organizations using the platform API.

    Args:
        api_endpoint (str): The base URL for the API endpoint.
        access_token (str): The access token for API authentication.

    Returns:
        list[dict]: A list of dictionaries containing organization information.
    """
    url = f"{api_endpoint}/api/v1/organizations/"
    return api_get(route=url, access_token=access_token)


def get_organization(
    api_endpoint: str,
    organization_id: int,
    access_token: str,
) -> dict:
    """
    Fetch the information of a specific organization `organization_id`.

    Args:
        api_endpoint (str): The base URL for the API endpoint.
        organization_id (int): The ID of the organization to fetch.
        access_token (str): The access token for API authentication.

    Returns:
        list[dict]: A dictionary containing the organization information.
    """
    url = f"{api_endpoint}/api/v1/organizations/{organization_id}"
    return api_get(route=url, access_token=access_token)


def list_sequences_for_date(
    api_endpoint: str,
    date: date,
    limit: int,
    offset: int,
    access_token: str,
) -> list[dict]:
    """
    List all sequences for a given date using the platform API.

    Args:
        api_endpoint (str): The base URL for the API endpoint.
        date (date): The date for which to list sequences.
        limit (int): The maximum number of sequences to return.
        offset (int): The number of sequences to skip before starting to collect the result set.
        access_token (str): The access token for API authentication.

    Returns:
        list[dict]: A list of dictionaries containing sequence information.
    """
    url = f"{api_endpoint}/api/v1/sequences/all/fromdate?from_date={date:%Y-%m-%d}&limit={limit}&offset={offset}"
    return api_get(route=url, access_token=access_token)


def get_detections(api_endpoint: str, detection_id: int, access_token: str) -> dict:
    """
    Fetch the information of a specific detection.

    Args:
        api_endpoint (str): The base URL for the API endpoint.
        detection_id (int): The ID of the detection to fetch.
        access_token (str): The access token for API authentication.

    Returns:
        dict: A dictionary containing the detection information.
    """
    url = f"{api_endpoint}/api/v1/detections/{detection_id}"
    return api_get(route=url, access_token=access_token)


def get_sequence(api_endpoint: str, sequence_id: int, access_token: str) -> dict:
    """
    Fetch the information of a specific sequence.

    Args:
        api_endpoint (str): The base URL for the API endpoint.
        sequence_id (int): The ID of the sequence to fetch.
        access_token (str): The access token for API authentication.

    Returns:
        dict: A dictionary containing the sequence information.
    """
    url = f"{api_endpoint}/api/v1/sequences/{sequence_id}"
    return api_get(route=url, access_token=access_token)


def list_sequence_detections(
    api_endpoint: str,
    sequence_id: int,
    access_token: str,
) -> list[dict]:
    """
    List all detections for a given sequence ID using the platform API.

    Args:
        api_endpoint (str): The base URL for the API endpoint.
        sequence_id (int): The ID of the sequence for which to list detections.
        access_token (str): The access token for API authentication.

    Returns:
        list[dict]: A list of dictionaries containing detection information.
    """
    url = f"{api_endpoint}/api/v1/sequences/{sequence_id}/detections"
    return api_get(route=url, access_token=access_token)
