"""Compute one DINOv2-base embedding per FP sequence using the highest-scoring
labeled frame. Used by the FP YOLO dataset build pipeline (val + train).

For each sequence in the chosen split(s):
  1. Pick the labeled frame with the highest detection score (6th column of
     the YOLO-style label). This is the same frame that the dataset build
     stage selects, so the embedding is consistent with the chosen image.
  2. Build a square crop around the highest-scoring bbox (side = max(w, h)
     × (1 + 2*padding), padded with black if outside image), resized to 224.
  3. Forward through facebook/dinov2-base, take the CLS token, L2-normalize.

Outputs per split:
  data/interim/fp_sequence_embeddings/<split>/embeddings_dinov2.npz   (n × 768)
  data/interim/fp_sequence_embeddings/<split>/embeddings_meta.json    (per-item metadata)

Usage:
  uv run python scripts/embed_fp_for_selection.py --splits val train
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


HF_REPO = "facebook/dinov2-base"
PADDING = 0.2
OUT_SIZE = 224


def make_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--registry", type=Path, default=Path("data/raw/fp/registry.json"))
    p.add_argument("--data-dir", type=Path, default=Path("data/raw/fp/data"))
    p.add_argument(
        "--splits",
        nargs="+",
        default=["val", "train", "test"],
        help="Which splits to embed. Defaults to val + train + test "
        "(YOLO test uses round-robin and ignores embeddings; sequential test uses them).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/fp_sequence_embeddings"),
        help="Output root. Per-split subfolders will be created.",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default=None, help="cpu | cuda | mps. Default: auto.")
    return p


def best_scoring_frame(seq_dir: Path):
    """Return (score, image_path, bbox_xyxyn) for the highest-scoring labeled
    frame in a sequence; None if no scored bbox is found."""
    labels_dir = seq_dir / "labels"
    images_dir = seq_dir / "images"
    if not labels_dir.is_dir() or not images_dir.is_dir():
        return None
    best = None  # (score, image_path, bbox_xyxyn)
    for lbl in labels_dir.glob("*.txt"):
        try:
            text = lbl.read_text()
        except OSError:
            continue
        for line in text.splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cx, cy, w, h = (float(parts[i]) for i in (1, 2, 3, 4))
            except ValueError:
                continue
            if w <= 0 or h <= 0:
                continue
            score = 1.0
            if len(parts) >= 6:
                try:
                    score = float(parts[5])
                except ValueError:
                    pass
            if best is not None and score <= best[0]:
                continue
            img = images_dir / (lbl.stem + ".jpg")
            if not img.exists():
                continue
            bbox = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            best = (score, img, bbox)
    return best


def square_crop_224(img_path: Path, bbox, padding: float, out_size: int = 224):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = (x2 - x1) * W
    bh = (y2 - y1) * H
    if bw <= 0 or bh <= 0:
        return None
    cx = (x1 + x2) / 2.0 * W
    cy = (y1 + y2) / 2.0 * H
    side = max(bw, bh) * (1.0 + 2.0 * padding)
    half = side / 2
    xs = int(round(cx - half))
    ys = int(round(cy - half))
    si = int(round(side))
    canvas = np.zeros((si, si, 3), dtype=np.uint8)
    sx0 = max(0, xs); sy0 = max(0, ys)
    sx1 = min(W, xs + si); sy1 = min(H, ys + si)
    if sx1 <= sx0 or sy1 <= sy0:
        return None
    dx0 = sx0 - xs; dy0 = sy0 - ys
    canvas[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = img[sy0:sy1, sx0:sx1]
    interp = cv2.INTER_AREA if si > out_size else cv2.INTER_CUBIC
    return cv2.resize(canvas, (out_size, out_size), interpolation=interp)


def detect_device(arg: str | None) -> str:
    if arg:
        return arg
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def embed_split(split: str, sequences: list[dict], data_dir: Path, model, proc, device: str, batch_size: int):
    metadata: list[dict] = []
    crops: list[Image.Image] = []
    skipped = 0
    for seq in tqdm(sequences, desc=f"prepare crops [{split}]"):
        seq_dir = data_dir / seq["folder"]
        if not seq_dir.is_dir():
            skipped += 1
            continue
        bf = best_scoring_frame(seq_dir)
        if bf is None:
            skipped += 1
            continue
        score, img_path, bbox = bf
        crop = square_crop_224(img_path, bbox, PADDING, out_size=OUT_SIZE)
        if crop is None:
            skipped += 1
            continue
        crops.append(Image.fromarray(crop))
        metadata.append(
            {
                "sequence_id": seq["id"],
                "sequence_folder": seq["folder"],
                "camera": seq["camera"],
                "split": seq["split"],
                "image_name": img_path.name,
                "score": float(score),
                "bbox_xyxyn": list(bbox),
            }
        )
    print(f"[{split}] crops ready: {len(crops)}  skipped: {skipped}")

    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for i in tqdm(range(0, len(crops), batch_size), desc=f"embed [{split}]"):
            batch = crops[i:i + batch_size]
            inp = proc(images=batch, return_tensors="pt").to(device)
            out = model(**inp)
            chunks.append(out.last_hidden_state[:, 0, :].float().cpu().numpy())
    feats = np.concatenate(chunks, axis=0).astype(np.float32) if chunks else np.empty((0, 768), dtype=np.float32)
    norms = np.linalg.norm(feats, axis=1, keepdims=True).clip(min=1e-12)
    feats = feats / norms
    return feats, metadata


def main() -> None:
    args = make_cli_parser().parse_args()

    sequences = json.loads(args.registry.read_text())["sequences"]
    by_split: dict[str, list[dict]] = {}
    for s in sequences:
        by_split.setdefault(s["split"], []).append(s)

    device = detect_device(args.device)
    print(f"device: {device}")

    proc = AutoImageProcessor.from_pretrained(HF_REPO)
    model = AutoModel.from_pretrained(HF_REPO).eval().to(device)

    for split in args.splits:
        seqs = by_split.get(split, [])
        if not seqs:
            print(f"[{split}] no sequences; skipping")
            continue
        feats, meta = embed_split(split, seqs, args.data_dir, model, proc, device, args.batch_size)

        out_dir = args.output / split
        out_dir.mkdir(parents=True, exist_ok=True)
        out_npz = out_dir / "embeddings_dinov2.npz"
        out_meta = out_dir / "embeddings_meta.json"
        np.savez(out_npz, embeddings=feats)
        out_meta.write_text(
            json.dumps(
                {
                    "split": split,
                    "model": HF_REPO,
                    "feature_dim": int(feats.shape[1]),
                    "n_items": len(meta),
                    "padding": PADDING,
                    "input_size": OUT_SIZE,
                    "frame_strategy": "highest_score",
                    "items": meta,
                },
                indent=2,
            )
        )
        print(f"[{split}] saved {out_npz} (shape={feats.shape}) and {out_meta}")


if __name__ == "__main__":
    main()
