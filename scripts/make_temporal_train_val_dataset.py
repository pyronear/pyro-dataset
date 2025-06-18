"""
CLI Script to generate the __train__ and __val__ wildfire temporal dataset.

Usage:
    python make_temporal_train_val_dataset.py --dir-save <directory> --dir-platform-sequence-temporal <directory> --random-seed <integer> [--ratio-background <float>] [-log <level>]

Arguments:
    --dir-save: directory to save the train/val wildfire temporal dataset.
    --dir-platform-sequence-temporal: directory containing the pyronear platform temporal sequences.
    --random-seed: random seed (required).
    --ratio-background: ratio of background sequences to add to the dataset (default is 0.5).
    -log, --loglevel: Provide logging level. Example --loglevel debug, default=warning.
"""

import argparse
import logging
import random
import shutil
from pathlib import Path


def make_cli_parser() -> argparse.ArgumentParser:
    """
    Make the CLI parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir-save",
        help="directory to save the train/val wildfire temporal dataset.",
        type=Path,
        default=Path("./data/processed/wildfire_temporal/"),
    )
    parser.add_argument(
        "--dir-platform-sequence-temporal",
        help="directory containing the pyronear platform temporal sequences.",
        type=Path,
        default=Path("./data/interim/pyronear-platform/sequences-temporal/"),
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

    return True


def handle_sequence(dir: Path, split: str, is_smoke: bool, dir_save: Path) -> None:
    class_str = "smoke" if is_smoke else "background"
    dst_images = dir_save / "images" / split / class_str / dir.name
    dst_labels = dir_save / "labels" / split / class_str / dir.name
    src_labels = Path(str(dir).replace("images", "labels"))
    dst_images.parent.mkdir(exist_ok=True, parents=True)
    dst_labels.parent.mkdir(exist_ok=True, parents=True)
    shutil.copytree(src=dir, dst=dst_images)
    shutil.copytree(src=src_labels, dst=dst_labels)


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
        rng = random.Random(random_seed)

        for split in ["train", "val"]:
            logger.info(f"Handling split {split}")
            dir_smoke = dir_platform_sequence_temporal / "images" / split / "smoke"
            dir_background = (
                dir_platform_sequence_temporal / "images" / split / "background"
            )
            dir_smoke_sequences = list(dir_smoke.iterdir())
            n_smoke = len(dir_smoke_sequences)
            dir_background_sequences = list(dir_background.iterdir())
            n_background = len(dir_background_sequences)
            k = int((1 - ratio_background) / ratio_background * n_smoke)
            dir_selected_background_sequences = rng.sample(
                population=dir_background_sequences,
                k=k,
            )
            logger.info(
                f"{split} split: {len(dir_smoke_sequences)} smoke sequences - {len(dir_background_sequences)} background sequences"
            )
            logger.info(
                f"selecting randomly {k} background sequences to account for ratio-background of {ratio_background}"
            )
            for dir in dir_smoke_sequences:
                handle_sequence(dir=dir, split=split, is_smoke=True, dir_save=dir_save)

            for dir in dir_selected_background_sequences:
                handle_sequence(dir=dir, split=split, is_smoke=False, dir_save=dir_save)

        logger.info(f"Done generating temporal dataset in {dir_save} ✅")
