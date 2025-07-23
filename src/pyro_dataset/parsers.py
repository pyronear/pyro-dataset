"""
Parser utils to parse filepaths and stems from the datasets.
"""

import re
from datetime import datetime
from enum import Enum
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DatasetOrigin(Enum):
    """
    All the possible dataset origins.

    Note: UNKNOW is used to account for new datasets that have not yet
    been added to this enum.
    """

    PYRONEAR = "pyronear"
    ADF = "adf"
    AWF = "awf"
    HPWREN = "hpwren"
    RANDOM_SMOKE = "random"
    UNKNOWN = "unknown"


@dataclass
class DetectionDetails:
    dataset_origin: DatasetOrigin
    stem: str
    details: dict[str, Any]


def parse_datetime(stem: str) -> datetime | None:
    """
    Parse the datetime from the filepath stem.

    Note: Try different datetime string formats and normalize it.
    """
    pattern = (
        r"(\d{4}[-_]\d{2}[-_]\d{2}T\d{2}[-_]\d{2}[-_]\d{2}|\d{4}[-_]\d{2}[-_]\d{2})"
    )

    match = re.search(pattern, stem)

    if match:
        # Extract the matched string
        datetime_str = match.group(0)
        # Replace underscores with dashes for consistency
        datetime_str = datetime_str.replace("_", "-")
        # Try to parse the datetime
        try:
            # Check for full datetime
            return datetime.strptime(datetime_str, "%Y-%m-%dT%H-%M-%S")
        except ValueError:
            return None
    else:
        return None


def parse_pyronear_camera_details(stem: str) -> dict | None:
    try:
        has_datetime = parse_datetime(stem)
        if not has_datetime:
            logging.warning(f"no datetime in stem {stem}")
            return None

        stem_without_origin_prefix = (
            stem.replace("Pyronear_", "").replace("pyronear_", "").replace("pyro_", "")
        )

        string_datetime_example = "2023_07_12T13_24_14"
        length_datetime_string = len(string_datetime_example)
        string_station_with_details = "_".join(
            stem_without_origin_prefix[: -length_datetime_string - 1].split("_")
        )
        if "-" in string_station_with_details:
            azimuth = int(string_station_with_details.split("-")[-1])
            camera_name = "-".join(string_station_with_details.split("-")[:-1])
            return {"name": camera_name, "azimuth": azimuth}
        else:
            camera_name = "_".join(string_station_with_details.split("_")[0:-1])
            camera_number = int(string_station_with_details.split("_")[-1])
            return {"name": camera_name, "number": camera_number}
    except Exception as e:
        logging.warning(f"Error parsing camera details from {stem} - {e}")
        return None


def parse_dataset_origin(stem: str) -> DatasetOrigin:
    """
    Parse the dataset origin from a filepath stem.
    """
    parts = stem.lower().split("_")
    origin_str = parts[0]
    if origin_str in [do.value for do in DatasetOrigin]:
        return DatasetOrigin(value=parts[0])
    # Fix the inconsitencies between the test dataset naming and the val/train naming
    elif origin_str in [
        "sdis-77",
        "sdis-07",
        "force-06",
        "marguerite-282",
        "pyro",
        "ardeche",
    ]:
        return DatasetOrigin.PYRONEAR
    # Fix the inconsitencies between the test dataset naming and the val/train naming
    elif origin_str in ["axis", "2023"]:
        return DatasetOrigin.AWF
    else:
        logging.warning(f"unknown dataset origin: {origin_str}")
        return DatasetOrigin.UNKNOWN


def parse_details_pyronear(stem: str) -> dict[str, Any]:
    return {
        "datetime": parse_datetime(stem),
        "camera": parse_pyronear_camera_details(stem),
    }


def parse_details_adf(stem: str) -> dict[str, Any]:
    return {
        "datetime": parse_datetime(stem),
    }


def parse_details_awf(stem: str) -> dict[str, Any]:
    return {
        "datetime": parse_datetime(stem),
    }


def parse_details_hpwren(stem: str) -> dict[str, Any]:
    return {
        "datetime": parse_datetime(stem),
    }


def parse_details_random_smoke(stem: str) -> dict[str, Any]:
    return {
        "datetime": parse_datetime(stem),
    }


def parse_details(stem: str, dataset_origin: DatasetOrigin) -> dict[str, Any]:
    """
    Parse details from the stem - mostly datetime if possible to extract it.
    """
    match dataset_origin:
        case DatasetOrigin.PYRONEAR:
            return parse_details_pyronear(stem)
        case DatasetOrigin.ADF:
            return parse_details_adf(stem)
        case DatasetOrigin.AWF:
            return parse_details_awf(stem)
        case DatasetOrigin.HPWREN:
            return parse_details_hpwren(stem)
        case DatasetOrigin.RANDOM_SMOKE:
            return parse_details_random_smoke(stem)
        case _:
            return {"datetime": parse_datetime(stem)}


def parse_filepath_stem(stem: str) -> DetectionDetails:
    """
    Parse the filepath stem and return a Detection Details with as much
    extracted details as possible.
    """
    dataset_origin = parse_dataset_origin(stem=stem)
    dataset_details = parse_details(stem=stem, dataset_origin=dataset_origin)
    return DetectionDetails(
        dataset_origin=dataset_origin,
        details=dataset_details,
        stem=stem,
    )

def parse_filepath(filepath: Path) -> DetectionDetails:
    """
    Parse the filepath stem and return a Detection Details with as much
    extracted details as possible.
    """
    dataset_origin = parse_dataset_origin(stem=filepath.stem)
    dataset_details = parse_details(stem=filepath.stem, dataset_origin=dataset_origin)
    return DetectionDetails(
        dataset_origin=dataset_origin,
        details=dataset_details,
        stem=filepath.stem,
    )
