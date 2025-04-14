from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class YOLOObjectDetectionPrediction:
    """
    Dataclass for representing a YOLO Prediction made by a model
    """

    class_id: int
    xywhn: NDArray[np.float16]
    xyxyn: NDArray[np.float16] = field(init=False)
    confidence: float

    def __post_init__(self):
        self.xyxyn = xywhn2xyxyn(self.xywhn)


def xywhn2xyxyn(bbox: NDArray[np.float16]) -> NDArray[np.float16]:
    """
    Convert a xywhn bbox into a xyxyn bbox format.
    """
    y = np.copy(bbox)
    y[..., 0] = bbox[..., 0] - bbox[..., 2] / 2  # top left x
    y[..., 1] = bbox[..., 1] - bbox[..., 3] / 2  # top left y
    y[..., 2] = bbox[..., 0] + bbox[..., 2] / 2  # bottom right x
    y[..., 3] = bbox[..., 1] + bbox[..., 3] / 2  # bottom right y
    return y.astype("float")


def parse_yolo_prediction_txt_file(
    txt_content: str,
) -> list[YOLOObjectDetectionPrediction]:
    """
    Parse the `txt_content` of a YOLOv8 TXT format and return a list of
    structured yolo predictions.
    """
    lines = txt_content.split("\n")
    result = []
    for line in lines:
        numbers = np.array(line.split(" ")).astype("float16")
        yolo_prediction = YOLOObjectDetectionPrediction(
            class_id=int(numbers[0]),
            xywhn=numbers[1:-1],
            confidence=numbers[-1].item(),
        )
        result.append(yolo_prediction)
    return result


def clip_xyxy(xyxy: NDArray[np.int_], w: int, h: int) -> NDArray[np.int_]:
    """
    Clip an xyxy bbox onto the actual size (w and h) of the image.
    """
    x_min, y_min, x_max, y_max = xyxy.astype(float)
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)

    return np.array([x_min, y_min, x_max, y_max]).astype("int")


def xyxyn2xyxy(xyxyn: NDArray[np.float16], w: int, h: int) -> NDArray[np.int_]:
    """
    Convert a xyxyn box into a xyxy box.
    """
    xyxy = xyxyn.copy()
    xyxy[::2] *= w
    xyxy[1::2] *= h
    return xyxy.astype("int")


def crop_xyxy(
    xyxy: NDArray[np.int_],
    array_image: NDArray[np.int_],
) -> NDArray[np.int_]:
    """
    Crop the `array_image` using the xyxy box.
    """
    x_min, y_min, x_max, y_max = xyxy
    crop = array_image[y_min:y_max, x_min:x_max]
    return crop


def expand_xyxy(
    xyxy: NDArray[np.int_],
    array_image: NDArray[np.int_],
    target_width: int,
    target_height: int,
) -> NDArray[np.int_]:
    """
    Expand the xyxy box to match a `target_width` and `target_height`.

    Returns:
      - box (NDArray[np.int_]): new xyxy expanded to match the target size.
    """
    h_image, w_image = array_image.shape[:2]
    x_min, y_min, x_max, y_max = xyxy
    w_box = x_max - x_min
    h_box = y_max - y_min

    # Adjust for small bboxes by expanding
    if w_box < target_width or h_box < target_height:
        # Try to expand symmetrically
        x_min = max(0, x_min - (target_width - w_box) // 2)
        x_max = min(w_image, x_min + target_width)

        y_min = max(0, y_min - (target_height - h_box) // 2)
        y_max = min(h_image, y_min + target_height)

        # If still too small and touching the border, align to the border
        if x_max - x_min < target_width:
            if x_min == 0:  # Align right if touching left border
                x_max = min(w_image, target_width)
            else:  # Align left if touching right border
                x_min = max(0, w_image - target_width)
                x_max = w_image

        if y_max - y_min < target_height:
            if y_min == 0:  # Align bottom if touching top border
                y_max = min(h_image, target_height)
            else:  # Align top if touching bottom border
                y_min = max(0, h_image - target_height)
                y_max = h_image

    return np.array([x_min, y_min, x_max, y_max])
