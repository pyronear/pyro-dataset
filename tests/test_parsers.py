"""
Tests for the parser utility functions.
"""

import pytest

from pyro_dataset.parsers import parse_pyronear_camera_details

s1 = "pyronear_valbonne_3_2023_11_02T07_05_56.jpg"
s2 = "pyronear_serre-de-barre-310_2024-09-02T13-12-23.jpg"
s3 = "pyronear_serre-de-barre-250_2024-08-25T11-19-22.jpg"
s4 = "Pyronear_test_DS_00000328.jpg"
s5 = "pyronear_st_peray_2_2023_07_12T13_24_14.jpg"


@pytest.mark.parametrize(
    "stem,expected",
    [
        ("marguerite-282_2024-09-09T14-57-20", {"name": "marguerite", "azimuth": 282}),
        (
            "pyronear_valbonne_3_2023_11_02T07_05_56",
            {"name": "valbonne", "number": 3},
        ),
        (
            "pyronear_serre-de-barre-310_2024-09-02T13-12-23",
            {"name": "serre-de-barre", "azimuth": 310},
        ),
        (
            "pyronear_serre-de-barre-250_2024-08-25T11-19-22",
            {"name": "serre-de-barre", "azimuth": 250},
        ),
        ("Pyronear_test_DS_00000328.jpg", None),
        ("pyronear_st_peray_2_2023_07_12T13_24_14", {"name": "st_peray", "number": 2}),
    ],
)
def test_parse_pyronear_camera_details(stem, expected):
    camera_details = parse_pyronear_camera_details(stem)
    assert camera_details == expected
