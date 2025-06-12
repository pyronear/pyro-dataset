import shutil
from pathlib import Path

dir_root = Path("./data/raw/pyronear-platform-annotated-sequences/sdis-77/")
dir_root.exists()


def get_predictions_subdirs(dir_path):
    """
    Recursively gets all subdirectories named 'predictions' within the specified directory.

    Args:
        dir_path (Path): The root directory to search within.

    Returns:
        List[Path]: A list of Paths to subdirectories named 'predictions'.
    """
    predictions_subdirs = []
    for path in dir_path.rglob("predictions"):
        if path.is_dir():
            predictions_subdirs.append(path)
    return predictions_subdirs


dirs = get_predictions_subdirs(dir_root)

dirs[:10]


def move_predictions_to_detections(predictions_dir):
    """
    Moves a directory named 'predictions' and all its files to a new folder named 'detections'.

    Args:
        predictions_dir (Path): The directory to be moved.
    """
    detections_dir = predictions_dir.parent / (
        predictions_dir.name.replace("predictions", "detections")
    )
    shutil.copytree(predictions_dir, detections_dir)
    # shutil.rmtree(predictions_dir)


dirs[0]
