"""
CLI Script to generate the __test__ wildfire temporal dataset.

Usage:
    python make_temporal_test_dataset.py --random-seed <seed> [options]

Arguments:
    --dir-save <path>                         Directory to save the temporal test wildfire dataset. (default: ./data/processed/wildfire_temporal_test/)
    --dir-platform-sequence-temporal <path>   Directory containing the pyronear platform temporal sequences. (default: ./data/interim/pyronear-platform/sequences-temporal/)
    --dir-selection-sequence-temporal <path>  Directory containing a manually curated set of temporal sequences. (default: ./data/raw/wildfire_temporal_test_selection/)
    --random-seed <int>                       Random seed (required).
    --ratio-background <float>                Ratio of background sequences to add to the dataset. (default: 0.5)
    -log, --loglevel <level>                  Provide logging level. Example: --loglevel debug (default: warning)
"""

import argparse
import logging
import shutil
from pathlib import Path


def make_cli_parser() -> argparse.ArgumentParser:
    """
    Make the CLI parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir-save",
        help="directory to save the test wildfire temporal dataset.",
        type=Path,
        default=Path("./data/processed/wildfire_temporal_test/"),
    )
    parser.add_argument(
        "--dir-platform-sequence-temporal",
        help="directory containing the pyronear platform temporal sequences.",
        type=Path,
        default=Path("./data/interim/pyronear-platform/sequences-temporal/"),
    )
    parser.add_argument(
        "--dir-selection-sequence-temporal",
        help="directory containing a manually curated set of temporal sequences.",
        type=Path,
        default=Path("./data/raw/wildfire_temporal_test_selection/"),
    )
    parser.add_argument(
        "--random-seed",
        help="random seed",
        type=int,
        required=True,
        default=0,
    )
    parser.add_argument(
        "--ratio-background",
        help="ratio of background sequences to add to the dataset",
        type=float,
        default=0.5,
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
    if not args["dir_platform_sequence_temporal"].exists():
        logging.error(
            f"invalid --dir-platform-sequence-temporal, dir {args['dir_platform_sequence_temporal']} does not exist"
        )
        return False
    elif not args["dir_selection_sequence_temporal"].exists():
        logging.error(
            f"invalid --dir-selection-sequence-temporal, dir {args['dir_selection_sequence_temporal']} does not exist"
        )
        return False

    return True


def find_sequence_folders(dir: Path) -> list[Path]:
    """
    Extract and return a list of sequence folders from annotated sequence directories.

    This function searches through the specified directory for subdirectories that
    contain sequence information. It includes directories that match the pattern '_sequence-<id>',
    where <id> is a numeric value.

    Args:
        dir_annotated_sequences (Path): The directory containing annotated sequence
        subdirectories to search for sequence folders.

    Returns:
        set[Path]: A set of unique Path objects representing the directories found in the directory.
    """
    return [
        seq_dir
        for seq_dir in dir.rglob("**/*")
        if seq_dir.is_dir()
        and "_sequence-" in seq_dir.name
        and seq_dir.name.split("_sequence-")[-1].isdigit()
    ]


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        exit(1)
    else:
        logging.info(args)
        random_seed = args["random_seed"]
        ratio_background = args["ratio_background"]
        dir_save = args["dir_save"]
        dir_platform_sequence_temporal = args["dir_platform_sequence_temporal"]
        dir_selection_sequence_temporal = args["dir_selection_sequence_temporal"]
        shutil.copytree(src=dir_selection_sequence_temporal, dst=dir_save)
        dirs_sequences_background_platform = find_sequence_folders(
            dir=dir_platform_sequence_temporal / "images" / "test" / "background"
        )
        print(
            (
                dir_selection_sequence_temporal / "images" / "test" / "background"
            ).exists()
        )
        print(
            find_sequence_folders(
                dir_selection_sequence_temporal / "images" / "test" / "background"
            )
        )
        dirs_sequences_background_selected = find_sequence_folders(
            dir=dir_selection_sequence_temporal / "images" / "test" / "background"
        )
        dirs_sequences_smoke_platform = find_sequence_folders(
            dir=dir_platform_sequence_temporal / "images" / "test" / "smoke"
        )
        dirs_sequences_smoke_selected = find_sequence_folders(
            dir=dir_selection_sequence_temporal / "images" / "test" / "smoke"
        )

        n_background_selected = len(dirs_sequences_background_selected)
        n_background_platform = len(dirs_sequences_background_platform)
        n_background_total = n_background_selected + n_background_platform
        n_smoke_selected = len(dirs_sequences_smoke_selected)
        n_smoke_platform = len(dirs_sequences_smoke_platform)
        n_smoke_total = n_smoke_selected + n_smoke_platform
        n_total = n_background_total + n_smoke_total
        print(
            "\n"
            f"{'Sequence Type':<20} {'Selected':<10} {'Platform':<10} {'Total':<10}\n"
            f"{'-' * 20} {'-' * 10} {'-' * 10} {'-' * 10}\n"
            f"{'Background Sequences':<20} {n_background_selected:<10} {n_background_platform:<10} {n_background_total:<10}\n"
            f"{'Smoke Sequences':<20} {n_smoke_selected:<10} {n_smoke_platform:<10} {n_smoke_total:<10}\n"
            f"{'Total':<20} {n_background_selected + n_smoke_selected:<10} {n_background_platform + n_smoke_platform:<10} {n_total:<10}\n"
        )
        # TODO: handle the ratio_background
        shutil.copytree(
            src=dir_platform_sequence_temporal, dst=dir_save, dirs_exist_ok=True
        )
