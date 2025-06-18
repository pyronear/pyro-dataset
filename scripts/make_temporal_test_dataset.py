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
        # TODO: add the sequences from dir_platform_sequence_temporal
        logger.info(f"Done generating the temporal dataset in {dir_save} ✅")
