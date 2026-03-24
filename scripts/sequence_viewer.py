"""Streamlit app to visualize sequences from data/raw with bounding boxes."""

from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"

st.set_page_config(page_title="Sequence Viewer", layout="wide")
st.title("Sequence Viewer")


# --- NMS (from pyro-engine/pyroengine/utils.py) ---

def box_iou(box1: np.ndarray, box2: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Pairwise IoU for boxes in xyxy format. Returns (N, M) array."""
    (a1, a2), (b1, b2) = np.split(box1, 2, 1), np.split(box2, 2, 1)
    inter = (np.minimum(a2, b2[:, None, :]) - np.maximum(a1, b1[:, None, :])).clip(0).prod(2)
    return inter / ((a2 - a1).prod(1) + (b2 - b1).prod(1)[:, None] - inter + eps)


def nms(boxes: np.ndarray, overlap_thresh: float = 0.0) -> np.ndarray:
    """Non-maximum suppression. boxes: (N, 5) in (x1,y1,x2,y2,conf) format."""
    if len(boxes) == 0:
        return boxes
    boxes = boxes[boxes[:, -1].argsort()]
    indices = np.arange(len(boxes))
    rr = box_iou(boxes[:, :4], boxes[:, :4])
    for i in range(len(boxes)):
        temp_indices = indices[indices != i]
        if np.any(rr[i, temp_indices] > overlap_thresh):
            indices = indices[indices != i]
    return boxes[indices]


def xywhn2xyxy(cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float]:
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


# --- Data helpers ---

def find_sequence(seq_name: str) -> Path | None:
    matches = list(DATA_RAW.glob(f"*/data/{seq_name}"))
    return matches[0] if matches else None


def load_labels(label_path: Path) -> list[list[float]]:
    """Return list of [cls, cx, cy, w, h] rows (YOLO xywhn)."""
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().strip().splitlines():
        if line.strip():
            boxes.append([float(p) for p in line.strip().split()])
    return boxes


def draw_boxes(
    img: np.ndarray,
    boxes: list[list[float]],
    highlight: np.ndarray | None = None,
    color: tuple = (255, 80, 0),
    highlight_color: tuple = (0, 220, 80),
) -> np.ndarray:
    """Draw xywhn boxes. If highlight (xyxy normalized) given, draw it in a different color."""
    h, w = img.shape[:2]
    img = img.copy()
    for box in boxes:
        _, cx, cy, bw, bh = box[:5]
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, "smoke", (x1, max(y1 - 6, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    if highlight is not None:
        hx1, hy1, hx2, hy2 = highlight
        x1, y1 = int(hx1 * w), int(hy1 * h)
        x2, y2 = int(hx2 * w), int(hy2 * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), highlight_color, 3)
        cv2.putText(img, "main", (x1, max(y1 - 6, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, highlight_color, 2)
    return img


def compute_main_bboxes(
    image_files: list[Path],
    labels_dir: Path,
) -> tuple[np.ndarray, list[tuple[Path, list[list[float]]]]]:
    """
    Aggregate all boxes across frames, run NMS (iou=0) to get main bboxes.
    Returns:
        main_boxes: (M, 5) array of main bboxes in xyxy+conf format (normalized)
        frame_data: list of (img_path, boxes_xywhn) for all frames
    """
    all_xyxy = []  # [x1, y1, x2, y2, conf, frame_idx, box_idx]
    frame_data = []

    for frame_idx, img_path in enumerate(image_files):
        label_path = labels_dir / img_path.with_suffix(".txt").name
        boxes = load_labels(label_path)
        frame_data.append((img_path, boxes))
        for box in boxes:
            _, cx, cy, bw, bh = box[:5]
            x1, y1, x2, y2 = xywhn2xyxy(cx, cy, bw, bh)
            all_xyxy.append([x1, y1, x2, y2, 1.0])

    if not all_xyxy:
        return np.empty((0, 5)), frame_data

    arr = np.array(all_xyxy, dtype=float)
    main_boxes = nms(arr, overlap_thresh=0.0)
    return main_boxes, frame_data


def frames_for_main_box(
    main_box: np.ndarray,
    frame_data: list[tuple[Path, list[list[float]]]],
) -> list[tuple[Path, list[list[float]]]]:
    """Return frames that have at least one box overlapping (iou > 0) with main_box."""
    mb = main_box[:4].reshape(1, 4)
    matching = []
    for img_path, boxes in frame_data:
        if not boxes:
            continue
        frame_xyxy = np.array([xywhn2xyxy(*b[1:5]) for b in boxes])
        ious = box_iou(mb, frame_xyxy)[:, 0]  # (K,) — iou of each frame box vs main box
        if np.any(ious > 0):
            matching.append((img_path, boxes))
    return matching


def remove_zone_from_labels(
    main_box: np.ndarray,
    frame_data: list[tuple[Path, list[list[float]]]],
    labels_dir: Path,
) -> int:
    """Remove boxes overlapping with main_box from all label files. Returns number of files modified."""
    mb = main_box[:4].reshape(1, 4)
    modified = 0
    for img_path, boxes in frame_data:
        if not boxes:
            continue
        frame_xyxy = np.array([xywhn2xyxy(*b[1:5]) for b in boxes])
        ious = box_iou(mb, frame_xyxy)[:, 0]
        kept = [box for box, iou in zip(boxes, ious) if iou <= 0]
        if len(kept) == len(boxes):
            continue  # nothing overlapped, skip
        label_path = labels_dir / img_path.with_suffix(".txt").name
        if kept:
            content = "\n".join(" ".join(str(v) for v in box) for box in kept)
            label_path.write_text(content + "\n")
        else:
            label_path.write_text("")
        modified += 1
    return modified


# --- Sidebar ---
with st.sidebar:
    st.header("Search")
    all_seq = sorted(p.name for p in DATA_RAW.glob("*/data/*") if p.is_dir())
    seq_name = st.text_input(
        "Sequence name",
        placeholder="pyronear-sdis-07_brison_110_2024-03-27T15-58-18",
    )
    st.markdown(f"**{len(all_seq):,}** sequences available")

    if seq_name:
        suggestions = [s for s in all_seq if seq_name.lower() in s.lower()][:20]
        if suggestions:
            st.markdown("**Suggestions:**")
            for s in suggestions:
                st.code(s, language=None)

# --- Main ---
if not seq_name:
    st.info("Enter a sequence name in the sidebar to get started.")
    st.stop()

seq_path = find_sequence(seq_name)
if seq_path is None:
    st.error(f"Sequence `{seq_name}` not found under `data/raw/*/data/`.")
    st.stop()

images_dir = seq_path / "images"
labels_dir = seq_path / "labels"
image_files = sorted(images_dir.glob("*.jpg"))

if not image_files:
    st.warning(f"No images found in `{images_dir}`.")
    st.stop()

dataset_tag = seq_path.parent.parent.name
st.success(f"Found in **{dataset_tag}** — {len(image_files)} frames")

# Compute main bboxes
main_boxes, frame_data = compute_main_bboxes(image_files, labels_dir)

if len(main_boxes) == 0:
    st.info("No annotations found in this sequence.")

    # Show all frames without boxes
    cols = st.columns(3)
    for i, (img_path, _) in enumerate(frame_data):
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        with cols[i % 3]:
            st.image(img, caption=img_path.name, width="stretch")
    st.stop()

st.markdown(f"**{len(main_boxes)} main bbox{'es' if len(main_boxes) != 1 else ''}** found via NMS (iou=0)")

# One tab per main bbox
tab_labels = [f"Zone {i + 1}" for i in range(len(main_boxes))]
tabs = st.tabs(tab_labels)

for i, (tab, main_box) in enumerate(zip(tabs, main_boxes)):
    with tab:
        mb_xyxy = main_box[:4]
        st.caption(
            f"Main bbox (xyxy norm): "
            f"({mb_xyxy[0]:.3f}, {mb_xyxy[1]:.3f}, {mb_xyxy[2]:.3f}, {mb_xyxy[3]:.3f})"
        )

        matching_frames = frames_for_main_box(main_box, frame_data)
        st.markdown(f"**{len(matching_frames)} frame{'s' if len(matching_frames) != 1 else ''}** with this zone")

        if not matching_frames:
            st.info("No frames found for this zone.")
            continue

        debug_mode = st.checkbox("Debug mode (dry run)", value=True, key=f"debug_{i}")

        if st.button(f"🗑 Remove zone from all {len(matching_frames)} label files", key=f"remove_{i}"):
            if debug_mode:
                mb = main_box[:4].reshape(1, 4)
                lines_out = []
                for img_path, boxes in frame_data:
                    if not boxes:
                        continue
                    frame_xyxy = np.array([xywhn2xyxy(*b[1:5]) for b in boxes])
                    ious = box_iou(mb, frame_xyxy)[:, 0]
                    removed = [box for box, iou in zip(boxes, ious) if iou > 0]
                    if removed:
                        label_path = labels_dir / img_path.with_suffix(".txt").name
                        for box in removed:
                            lines_out.append(f"{label_path.name}: remove {' '.join(str(v) for v in box)}")
                st.code("\n".join(lines_out) if lines_out else "(nothing to remove)", language=None)
            else:
                n_modified = remove_zone_from_labels(main_box, frame_data, labels_dir)
                st.success(f"Removed from {n_modified} label file{'s' if n_modified != 1 else ''}.")
                st.rerun()

        cols = st.columns(3)
        for j, (img_path, boxes) in enumerate(matching_frames):
            img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            img = draw_boxes(img, boxes, highlight=mb_xyxy)
            n = len(boxes)
            with cols[j % 3]:
                st.image(img, caption=f"{img_path.name} — {n} box{'es' if n != 1 else ''}", width="stretch")
