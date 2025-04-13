"""
CLI Script to generate predictions using a trained model and save them as
YOLOv8 TXT format.
"""

from pathlib import Path
import logging
from tqdm import tqdm
import argparse

from pyro_dataset.yolo.main import load_model, ultralytics_results_to_yolo_txt


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-dir",
        help="directory to save the predictions.",
        type=Path,
        default=Path("./data/interim/pyro-sdis/predictions/wise_wolf/"),
    )
    parser.add_argument(
        "--dir-images",
        help="directory containing the images to run inference on.",
        type=Path,
        default=Path("./data/raw/pyro-sdis/images/val/"),
    )
    parser.add_argument(
        "--filepath-weights",
        help="filepath to the model weights.",
        type=Path,
        default=Path("./data/external/models/wise_wolf/weights/yolov11s.pt"),
    )
    parser.add_argument(
        "-log",
        "--loglevel",
        default="warning",
        help="Provide logging level. Example --loglevel debug, default=warning",
    )
    return parser


def validate_parsed_args(args: dict) -> bool:
    """Returns whether the parsed args are valid."""
    if not args["dir_images"].exists():
        logging.error(
            f"invalid --dir-images, dir {args['dir_images']} does not exist"
        )
        return False
    elif not args["filepath_weights"].exists():
        logging.error(
            f"invalid --filepath-weights, file {args['filepath_weights']} does not exist"
        )
        return False

    else:
        return True


def predict_and_save(
    dir_images: Path,
    filepath_weights: Path,
    save_dir: Path,
) -> list[Path]:
    """
    Predict and save the YOLOv8 TXT files on all images found in `dir_images`.

    Note: The directory structure of `dir_images` is preserved.
    """
    logging.info(f"loading the model from {filepath_weights}")
    model = load_model(filepath_weights)
    logging.info("model loaded ✔️")
    filepaths = list(dir_images.glob("*.jpg"))
    logging.info(f"found {len(filepaths)} images in {dir_images}")
    label_filepaths = []

    for filepath in tqdm(filepaths[:10]):
        results = model.predict(filepath, verbose=False)
        label_content = ultralytics_results_to_yolo_txt(results)
        label_filepath = save_dir / f"{filepath.relative_to(dir_images).stem}.txt"
        label_filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(label_filepath, "w") as fd:
            fd.write(label_content)
        label_filepaths.append(label_filepath)

    return label_filepaths


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        exit(1)
    else:
        logging.info(args)
        save_dir = args["save_dir"]
        dir_images = args["dir_images"]
        filepath_weights = args["filepath_weights"]
        predict_and_save(dir_images=dir_images, filepath_weights=filepath_weights, save_dir=save_dir)
        exit(0)
