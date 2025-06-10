import hashlib
import logging
import shutil
from pathlib import Path

import yaml


def compute_file_content_sha256(filepath: Path) -> str:
    """
    Compute the file content hash of the `filepath`.

    Returns:
        hexdigest (str)
    """
    # Create a hash object
    hash_sha256 = hashlib.sha256()

    # Open the file in binary mode
    with open(filepath, "rb") as f:
        # Read the file in chunks to avoid using too much memory
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)

    # Return the hexadecimal digest of the hash
    return hash_sha256.hexdigest()


class MyDumper(yaml.Dumper):
    """Formatter for dumping yaml."""

    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)


def yaml_read(path: Path) -> dict:
    """Returns yaml content as a python dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def yaml_write(to: Path, data: dict, dumper=MyDumper) -> None:
    """Writes a `data` dictionnary to the provided `to` path."""
    with open(to, "w") as f:
        yaml.dump(
            data=data,
            stream=f,
            Dumper=dumper,
            default_flow_style=False,
            sort_keys=False,
        )


def index_by(xs: list[dict], key: str) -> dict[str, dict]:
    """
    Index a collection of dicts `xs` by the provided `key`.
    """
    return {x[key]: x for x in xs}


def ultralytics_dataset_info(dir_dataset: Path) -> dict:
    """
    Retrieve information about the Ultralytics dataset from the specified directory.

    Args:
        dir_dataset (Path): The path to the directory containing the Ultralytics dataset.

    Returns:
        dict: A dictionary containing the dataset information, including:
            - data_yaml: The contents of the data.yaml file as a dictionary.
            - splits: A dictionary containing the images and labels for each split (train, val, test).
    """
    filepath_data_yaml = dir_dataset / "data.yaml"
    data_yaml = yaml_read(filepath_data_yaml)
    all_splits = ["train", "val", "test"]
    splits = list(set(data_yaml.keys()).intersection(set(all_splits)))

    info = {
        "data_yaml": data_yaml,
        "splits": {
            "train": {"images": [], "labels": []},
            "val": {"images": [], "labels": []},
            "test": {"images": [], "labels": []},
        },
    }

    for split in splits:

        dir_images = dir_dataset / data_yaml[split]
        dir_labels = Path(str(dir_images).replace("images", "labels"))
        filepaths_images = list(dir_images.glob("*.jpg"))
        filepaths_labels = list(dir_labels.glob("*.txt"))

        assert len(filepaths_images) == len(
            filepaths_labels
        ), "Should be the same length"
        info["splits"][split] = {"images": filepaths_images, "labels": filepaths_labels}

    return info


def copy_ultralytics_dataset(
    dir_dataset: Path,
    dir_save: Path,
    splits: list[str] = ["train", "val", "test"],
) -> None:
    """
    Copy images and labels from the specified Ultralytics dataset directory to
    a new directory.

    Args:
        dir_dataset (Path): The path to the source dataset directory.
        dir_save (Path): The path to the destination directory where the dataset will be copied.
        splits (list[str], optional): A list of splits to copy (default is ["train", "val", "test"]).
    """
    assert (
        len(splits) >= 1
    ), "`splits` should at least contain one split from {train, val, test}"

    info = ultralytics_dataset_info(dir_dataset=dir_dataset)
    dir_save.mkdir(parents=True, exist_ok=True)
    filepath_data_yaml = dir_save / "data.yaml"

    # Remove train/val/test keys from data.yaml if not provided in splits
    data_yaml = info["data_yaml"]
    for s in ["train", "val", "test"]:
        if not s in splits and s in data_yaml:
            del data_yaml[s]

    yaml_write(to=filepath_data_yaml, data=data_yaml)

    for split in splits:
        for filepath_image in info["splits"][split]["images"]:
            filepath_destination = dir_save / "images" / split / filepath_image.name
            filepath_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src=filepath_image, dst=filepath_destination)

        for filepath_label in info["splits"][split]["labels"]:
            filepath_destination = dir_save / "labels" / split / filepath_label.name
            filepath_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src=filepath_label, dst=filepath_destination)


def combine_ultralytics_datasets(
    dirs_dataset: list[Path],
    dir_save: Path,
    splits: list[str] = ["train", "val", "test"],
) -> None:
    """
    Combine multiple Ultralytics datasets into a single dataset directory.

    This function copies images and labels from multiple Ultralytics dataset directories
    into a specified save directory. The datasets are merged based on the provided splits.

    Args:
        dirs_dataset (list[Path]): A list of paths to the source Ultralytics dataset directories.
        dir_save (Path): The path to the destination directory where the combined dataset will be saved.
        splits (list[str], optional): A list of dataset splits to be included in the combined dataset.
                                       Default is ["train", "val", "test"].
    """
    dir_save.mkdir(parents=True, exist_ok=True)
    for dir_dataset in dirs_dataset:
        logging.info(f"Add ultralytics dataset from {dir_dataset}")
        copy_ultralytics_dataset(
            dir_dataset=dir_dataset,
            dir_save=dir_save,
            splits=splits,
        )
