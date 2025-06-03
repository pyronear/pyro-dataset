"""
API client functions to interact with the Pyronear platform.
"""

import logging
from datetime import date

import requests


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


def api_get(route: str, access_token: str):
    """
    Issue a GET request against the API route with the provided headers.

    Returns:
        response (dict): JSON response

    Raises:
        Exception: when the API request fails
    """
    headers = make_request_headers(access_token=access_token)
    logging.info(f"Making an HTTP request to route {route}")
    response = requests.get(route, headers=headers)
    try:
        return response.json()
    except:
        raise Exception(f"API Error: {response.status_code} {response.text}")


def make_request_headers(access_token: str) -> dict[str, str]:
    """
    Make the HTTP request headers.
    """
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }


def list_cameras(api_endpoint: str, access_token: str) -> list[dict]:
    """
    List all cameras using the platform API.
    """
    url = f"{api_endpoint}/api/v1/cameras/"
    return api_get(route=url, access_token=access_token)


def get_camera(api_endpoint: str, camera_id: int, access_token: str) -> dict:
    """
    Fetch the information of a specific camera `camera_id`.
    """
    url = f"{api_endpoint}/api/v1/cameras/{camera_id}"
    return api_get(route=url, access_token=access_token)


def list_organizations(api_endpoint: str, access_token: str) -> list[dict]:
    url = f"{api_endpoint}/api/v1/organizations/"
    return api_get(route=url, access_token=access_token)


def get_organization(
    api_endpoint: str, organization_id: int, access_token: str
) -> list[dict]:
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
    List sequences for a specified date, limit the result to `limit` and
    use an `offset` if the results are paginated."""
    url = f"{api_endpoint}/api/v1/sequences/all/fromdate?from_date={date:%Y-%m-%d}&limit={limit}&offset={offset}"
    return api_get(route=url, access_token=access_token)


def get_detections(api_endpoint: str, detection_id: int, access_token: str) -> dict:
    """
    Fetch the information of a specific detection.
    """
    url = f"{api_endpoint}/api/v1/detections/{detection_id}"
    return api_get(route=url, access_token=access_token)


def list_sequence_detections(
    api_endpoint: str, sequence_id: int, access_token: str
) -> list[dict]:
    url = f"{api_endpoint}/api/v1/sequences/{sequence_id}/detections"
    return api_get(route=url, access_token=access_token)
