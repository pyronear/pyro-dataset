"""
Utility functions to make plots and graphs based on the report.yaml file
generated when analyzing the processed datasets.
"""

from collections import defaultdict
from typing import Any

from bokeh.palettes import Category10, Category20, MediumContrast6
from bokeh.plotting import figure


def normalize_frequencies(freqs: dict[Any, int]) -> dict[Any, float]:
    """
    Normalize a frequency dict.
    """
    total = sum(v for v in freqs.values())
    return {k: v / total for k, v in freqs.items()}


def merge_frequencies(*freq_dicts):
    """
    Merge frequency dicts.
    """
    merged_freq = defaultdict(int)
    for freq in freq_dicts:
        for year, count in freq.items():
            merged_freq[year] += count
    return dict(merged_freq)


def month_index_to_month_str(idx: int) -> str:
    """
    Turn a month index from 1 to 12 into its associated string.

    Raises:
        assertError when the `idx` is out of range.
    """
    assert 1 <= idx <= 12, f"idx should be in (1,12)"

    month_strs = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ]
    return month_strs[idx - 1]


def make_plot_data_for_data_splits_breakdown(report: dict, origins: list[str]) -> dict:
    """
    Create the data dict needed for the bokeh viz layer.
    """
    splits = list(report["summary"]["split"].keys())
    train = report["summary"]["split"]["train"]
    val = report["summary"]["split"]["val"]
    test = report["summary"]["split"]["test"]
    frequencies_origins_normalized_train = normalize_frequencies(
        train["frequencies"]["origins"]
    )
    frequencies_origins_normalized_val = normalize_frequencies(
        val["frequencies"]["origins"]
    )
    frequencies_origins_normalized_test = normalize_frequencies(
        test["frequencies"]["origins"]
    )
    dict_origin_values = {
        origin: [
            frequencies_origins_normalized_train.get(origins[idx], 0),
            frequencies_origins_normalized_val.get(origins[idx], 0),
            frequencies_origins_normalized_test.get(origins[idx], 0),
        ]
        for idx, origin in enumerate(origins)
    }

    data = {"splits": splits, **dict_origin_values}
    return data


def make_figure_for_data_splits_breakdown(data: dict) -> figure:
    """
    Make the figure based on the provided data.
    """

    splits = data["splits"]
    tmp = data.copy()
    del tmp["splits"]
    stacks = list(tmp.keys())

    p = figure(
        x_range=splits,
        height=450,
        title="Data splits origin breakdown",
        toolbar_location=None,
        tools="hover",
        tooltips="$name @splits: @$name{0.0%}",
    )

    color = MediumContrast6[: len(data.keys()) - 1]

    p.vbar_stack(
        stacks,
        x="splits",
        width=0.7,
        color=color,
        source=data,
        legend_label=stacks,
    )

    p.title_location = "above"
    p.title.align = "center"
    p.y_range.start = 0
    p.x_range.range_padding = 0.1
    p.xgrid.grid_line_color = None
    p.axis.minor_tick_line_color = None
    p.xaxis.axis_label = "Data Splits"
    p.yaxis.axis_label = "Breakdown"
    p.outline_line_color = None
    p.legend.location = "bottom_right"
    p.legend.orientation = "vertical"

    return p


def make_plot_data_for_ratio_background_images(report: dict) -> dict:
    """
    Create the data dict needed for the bokeh viz layer.
    """

    splits = list(report["summary"]["split"].keys())

    train = report["summary"]["split"]["train"]
    val = report["summary"]["split"]["val"]
    test = report["summary"]["split"]["test"]

    train_n_images = train["statistics"]["n_images"]
    train_n_background_images = train["statistics"]["n_background_images"]
    val_n_images = val["statistics"]["n_images"]
    val_n_background_images = val["statistics"]["n_background_images"]
    test_n_images = test["statistics"]["n_images"]
    test_n_background_images = test["statistics"]["n_background_images"]

    ratio_train_background = train_n_background_images / train_n_images
    ratio_val_background = val_n_background_images / val_n_images
    ratio_test_background = test_n_background_images / test_n_images

    data = {
        "splits": splits,
        "ratio_smoke_images": [
            1 - ratio_train_background,
            1 - ratio_val_background,
            1 - ratio_test_background,
        ],
        "ratio_background_images": [
            ratio_train_background,
            ratio_val_background,
            ratio_test_background,
        ],
    }

    return data


def make_figure_for_ratio_background_images(data: dict) -> figure:
    """
    Make the figure based on the provided data.
    """

    splits = data["splits"]
    tmp = data.copy()
    del tmp["splits"]
    stacks = list(tmp.keys())

    p = figure(
        x_range=splits,
        height=450,
        title="Smoke vs Background Images breakdown",
        toolbar_location=None,
        tools="hover",
        tooltips="$name @splits: @$name{0.%}",
    )

    color = MediumContrast6[: len(data) - 1]

    p.vbar_stack(
        stacks,
        x="splits",
        width=0.7,
        color=color,
        source=data,
        legend_label=stacks,
    )

    p.title_location = "above"
    p.title.align = "center"
    p.y_range.start = 0
    p.x_range.range_padding = 0.1
    p.xgrid.grid_line_color = None
    p.axis.minor_tick_line_color = None
    p.xaxis.axis_label = "Data Splits"
    p.yaxis.axis_label = "Breakdown"
    p.outline_line_color = None
    p.legend.location = "bottom_right"
    p.legend.orientation = "horizontal"

    return p


def make_plot_data_for_data_splits_year_breakdown(report: dict) -> dict:
    """
    Create the data dict needed for the bokeh viz layer.
    """
    splits = list(report["summary"]["split"].keys())
    train = report["summary"]["split"]["train"]
    val = report["summary"]["split"]["val"]
    test = report["summary"]["split"]["test"]
    frequencies_years_normalized_train = normalize_frequencies(
        train["frequencies"]["years"]
    )
    frequencies_years_normalized_val = normalize_frequencies(
        val["frequencies"]["years"]
    )
    frequencies_years_normalized_test = normalize_frequencies(
        test["frequencies"]["years"]
    )
    frequencies_years = merge_frequencies(
        train["frequencies"]["years"],
        val["frequencies"]["years"],
        test["frequencies"]["years"],
    )
    years = sorted(
        frequencies_years.keys(), key=lambda k: frequencies_years[k], reverse=True
    )

    dict_origin_values = {
        str(year): [
            frequencies_years_normalized_train.get(years[idx], 0),
            frequencies_years_normalized_val.get(years[idx], 0),
            frequencies_years_normalized_test.get(years[idx], 0),
        ]
        for idx, year in enumerate(years)
    }

    data = {"splits": splits, **dict_origin_values}
    return data


def make_figure_for_data_splits_year_breakdown(data: dict) -> figure:
    """
    Make the figure based on the provided data.
    """

    splits = data["splits"]
    tmp = data.copy()
    del tmp["splits"]
    stacks = list(tmp.keys())

    p = figure(
        x_range=splits,
        height=450,
        title="Data splits Years breakdown",
        toolbar_location=None,
        tools="hover",
        tooltips="$name @splits: @$name{0.0%}",
    )

    color = Category10[len(data.keys()) - 1]

    p.vbar_stack(
        stacks,
        x="splits",
        width=0.7,
        color=color,
        source=data,
        legend_label=stacks,
    )

    p.title_location = "above"
    p.title.align = "center"
    p.y_range.start = 0
    p.x_range.range_padding = 0.1
    p.xgrid.grid_line_color = None
    p.axis.minor_tick_line_color = None
    p.xaxis.axis_label = "Data Splits"
    p.yaxis.axis_label = "Breakdown"
    p.outline_line_color = None
    p.legend.location = "bottom_right"
    p.legend.orientation = "vertical"

    return p


def make_plot_data_for_data_splits_month_breakdown(report: dict) -> dict:
    """
    Create the data dict needed for the bokeh viz layer.
    """
    splits = list(report["summary"]["split"].keys())
    train = report["summary"]["split"]["train"]
    val = report["summary"]["split"]["val"]
    test = report["summary"]["split"]["test"]
    frequencies_months_normalized_train = normalize_frequencies(
        train["frequencies"]["months"]
    )
    frequencies_months_normalized_val = normalize_frequencies(
        val["frequencies"]["months"]
    )
    frequencies_months_normalized_test = normalize_frequencies(
        test["frequencies"]["months"]
    )
    frequencies_months = merge_frequencies(
        train["frequencies"]["months"],
        val["frequencies"]["months"],
        test["frequencies"]["months"],
    )
    months = sorted(
        frequencies_months.keys(), key=lambda k: frequencies_months[k], reverse=True
    )

    dict_month_values = {
        f"{month_index_to_month_str(int(month))}": [
            frequencies_months_normalized_train.get(months[idx], 0),
            frequencies_months_normalized_val.get(months[idx], 0),
            frequencies_months_normalized_test.get(months[idx], 0),
        ]
        for idx, month in enumerate(months)
    }

    data = {"splits": splits, **dict_month_values}
    return data


def make_figure_for_data_splits_month_breakdown(data: dict) -> figure:
    """
    Make the figure based on the provided data.
    """

    splits = data["splits"]
    tmp = data.copy()
    del tmp["splits"]
    stacks = list(tmp.keys())

    p = figure(
        x_range=splits,
        height=450,
        title="Data splits Months breakdown",
        toolbar_location=None,
        tools="hover",
        tooltips="$name @splits: @$name{0.0%}",
    )

    color = Category20[len(data.keys()) - 1]

    p.vbar_stack(
        stacks,
        x="splits",
        width=0.7,
        color=color,
        source=data,
        legend_label=stacks,
    )

    p.title_location = "above"
    p.title.align = "center"
    p.y_range.start = 0
    p.x_range.range_padding = 0.1
    p.xgrid.grid_line_color = None
    p.axis.minor_tick_line_color = None
    p.xaxis.axis_label = "Data Splits"
    p.yaxis.axis_label = "Breakdown"
    p.outline_line_color = None
    p.legend.location = "bottom_right"
    p.legend.orientation = "vertical"

    return p
