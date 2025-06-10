"""
CLI script to persist the sequence informations for the annotated sequences.

Usage:
    To run the script, execute the module from the command line with the following
    arguments:

    --dir-save <path>                   Directory to save the filtered sequences.
                                        Default is './data/raw/pyronear-platform/sequences/'.

    --dir-platform-annotated-sequences <path>
                                        Directory containing annotated sequences.

    --dir-platform-sequences <path>     Directory of fetched platform sequences.

    -log, --loglevel <level>            Set the logging level (default is 'info').
                                        Accepted values are 'debug', 'info', 'warning',
                                        'error', and 'critical'.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd


def find_sequences_csv_files(dir_platform_sequences: Path) -> list[Path]:
    """
    Find and return a list of all 'sequences.csv' files within the specified
    directory and its subdirectories.

    This function recursively searches through the given directory for any files
    named 'sequences.csv' and returns their paths as a list. It is useful for
    collecting all sequence files for further processing.

    Args:
        dir_platform_sequences (Path): The directory in which to search for
        'sequences.csv' files.

    Returns:
        list[Path]: A list of Path objects representing the found 'sequences.csv' files.
    """
    return list(dir_platform_sequences.rglob("sequences.csv"))


def read_csv_files_to_dataframe(filepaths: list[Path]) -> pd.DataFrame:
    """
    Read a list of CSV files and concatenate them into a single DataFrame.

    Args:
        filepaths (list[Path]): A list of Path objects representing the CSV files to read.

    Returns:
        pd.DataFrame: A DataFrame containing the concatenated data from all CSV files.
    """
    df_list = [pd.read_csv(filepath) for filepath in filepaths]
    return pd.concat(df_list, ignore_index=True)


def find_sequence_ids(dir_annotated_sequences: Path) -> set[int]:
    """
    Extract and return a set of unique sequence IDs from annotated sequence directories.

    This function searches through the specified directory for subdirectories that
    contain sequence information. It extracts sequence IDs from directory names
    that match the pattern '_sequence-<id>', where <id> is a numeric value.

    Args:
        dir_annotated_sequences (Path): The directory containing annotated sequence
        subdirectories to search for sequence IDs.

    Returns:
        set[int]: A set of unique integer sequence IDs found in the directory.
    """
    return {
        int(seq_dir.name.split("_sequence-")[-1])
        for seq_dir in dir_annotated_sequences.rglob("**/*")
        if seq_dir.is_dir()
        and "_sequence-" in seq_dir.name
        and seq_dir.name.split("_sequence-")[-1].isdigit()
    }


def filter_dataframe_by_sequence_ids(
    df: pd.DataFrame,
    sequence_ids: set[int],
) -> pd.DataFrame:
    """
    Filter the given DataFrame to include only rows where the 'sequence_id'
    is in the provided set of sequence IDs.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        sequence_ids (set[int]): A set of sequence IDs to filter the DataFrame by.

    Returns:
        pd.DataFrame: A DataFrame containing only the rows with matching sequence IDs.
    """
    return df[df["sequence_id"].isin(sequence_ids)]


def make_cli_parser() -> argparse.ArgumentParser:
    """
    Make the CLI parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir-save",
        help="Directory to save the sequences",
        type=Path,
        default=Path("./data/raw/pyronear-platform/sequences/"),
    )
    parser.add_argument(
        "--dir-platform-annotated-sequences",
        help="Directory of the annotated sequences from the Pyronear Platform",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--dir-platform-sequences",
        help="Directory of the fetched platform sequences - usually via the scripts fetch_platform_sequences.py",
        type=Path,
        default="",
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
    if not args["dir_platform_annotated_sequences"].exists():
        logging.error(
            f"Invalid --dir-platform-annotated-sequences, directory does not exist"
        )
        return False
    if not args["dir_platform_sequences"].exists():
        logging.error(f"Invalid --dir-platform-sequences, directory does not exist")
        return False
    return True


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        exit(1)
    else:
        logger.info(args)
        dir_save = args["dir_save"]
        dir_platform_sequences = args["dir_platform_sequences"]
        dir_platform_annotated_sequences = args["dir_platform_annotated_sequences"]

        logger.info(f"Find all sequences.csv file from {dir_platform_sequences}")
        sequences_csv = find_sequences_csv_files(dir_platform_sequences)
        logger.info(f"Found the following sequences.csv files: {sequences_csv}")
        df_all_sequences = read_csv_files_to_dataframe(sequences_csv)
        logger.info(df_all_sequences.head())
        sequence_ids = find_sequence_ids(dir_platform_annotated_sequences)
        logger.info(
            f"Found {len(sequence_ids)} sequence ids in {dir_platform_annotated_sequences}"
        )
        df_sequences = filter_dataframe_by_sequence_ids(
            df_all_sequences, sequence_ids=sequence_ids
        )
        logger.info(
            f"Saving {len(df_sequences)} rows from the original sequences.csv files"
        )
        logger.info(df_sequences.head())
        dir_save.mkdir(parents=True, exist_ok=True)
        df_sequences.to_csv(dir_save / "sequences.csv", index=False)
