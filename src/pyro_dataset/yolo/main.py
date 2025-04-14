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
        results (str): YOLOv8 TXT results as a string. It uses xywhn format to
        store coordinates.

    Eg.
        1 0.617 0.3594420600858369 0.114 0.17381974248927037
        1 0.094 0.38626609442060084 0.156 0.23605150214592274

    Documentation: https://roboflow.com/formats/yolov8-pytorch-txt
    """
    if not ultralytics_results:
        return ""
    else:
        lines = [
            f"{int(class_id)} {' '.join(str(c) for c in xywhn)} {prob}"
            for (xywhn, prob, class_id) in zip(
                ultralytics_results[0].boxes.xywhn.cpu().tolist(),
                ultralytics_results[0].boxes.conf.cpu().tolist(),
                ultralytics_results[0].boxes.cls.cpu().tolist(),
            )
        ]
        return "\n".join(lines)
