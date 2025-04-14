import logging
from pathlib import Path

from ultralytics import YOLO

# Create a logger
logger = logging.getLogger(__name__)


class YOLOModelLoadingError(Exception):
    pass


def load_model(filepath_weights: Path) -> YOLO:
    """
    Load the YOLO model from the weight filepath.

    Returns:
        model (YOLO): loaded model
    Throws:
        YOLOModelLoadingError: When the model cannot be loaded from the
        provided weight filepath.
    """
    if not filepath_weights.exists():
        logger.error(f"Cannot load the model from weights: {filepath_weights}")
        raise YOLOModelLoadingError
    else:
        try:
            model = YOLO(filepath_weights)
            model.info()
            return model
        except Exception:
            raise YOLOModelLoadingError


def ultralytics_results_to_yolo_txt(ultralytics_results: list) -> str:
    """
    Turn a ultralytics results into a YOLOv8 TXT format string.

    Returns:
        results (str): YOLOv8 TXT results as a string.

    Documentation: https://roboflow.com/formats/yolov8-pytorch-txt
    """
    if not ultralytics_results:
        return ""
    else:
        lines = [
            f"{int(class_id)} {' '.join(str(c) for c in xywh)} {prob}"
            for (xywh, prob, class_id) in zip(
                ultralytics_results[0].boxes.xywh.cpu().tolist(),
                ultralytics_results[0].boxes.conf.cpu().tolist(),
                ultralytics_results[0].boxes.cls.cpu().tolist(),
            )
        ]
        return "\n".join(lines)
