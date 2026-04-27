"""Two-stage FP sequence selection used by both the YOLO and the sequential
dataset builders.

Stage 1 — bbox-overlap atoms (per-camera, deterministic):
  For each FP sequence, NMS top-1 main bbox across all predicted frames.
  Within a camera, sequences whose main boxes have IoU > `match_iou` are
  linked via union-find. Connected components ≡ recurring artefact at the
  same place on the same physical camera.

Stage 2 — KMeans on the per-atom representatives:
  Pick the atom rep = sequence whose embedding is closest to the atom
  centroid. Run KMeans(k = quota) on the rep embeddings. For each KMeans
  cluster, the final selected sequence = the rep closest to the cluster
  centroid.

Inputs are aligned: `embeddings[i]` corresponds to `items[i]`.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans


def iou_xyxyn(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_seq_boxes(seq_dir: Path):
    """Read all valid prediction boxes from labels/*.txt for a sequence.

    Returns a list of (x1, y1, x2, y2, conf) tuples in normalized coords.
    Handles both 5-col GT and 6-col prediction labels (conf defaults to 1.0).
    """
    lp = seq_dir / "labels"
    if not lp.is_dir():
        return []
    boxes: list[tuple[float, float, float, float, float]] = []
    for f in lp.glob("*.txt"):
        try:
            text = f.read_text()
        except OSError:
            continue
        for line in text.splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cx, cy, w, h = (float(p) for p in parts[1:5])
            except ValueError:
                continue
            if cx == 0 and cy == 0 and w == 0 and h == 0:
                continue
            if w <= 0 or h <= 0:
                continue
            conf = 1.0
            if len(parts) >= 6:
                try:
                    conf = float(parts[5])
                except ValueError:
                    pass
            boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, conf))
    return boxes


def nms_top1(boxes, iou_thr: float) -> tuple[float, float, float, float] | None:
    """Greedy NMS, return the highest-conf surviving box as xyxyn or None."""
    if not boxes:
        return None
    ordered = sorted(boxes, key=lambda b: b[4], reverse=True)
    kept: list = []
    for box in ordered:
        if all(iou_xyxyn(box, k) < iou_thr for k in kept):
            kept.append(box)
    return tuple(kept[0][:4]) if kept else None


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def two_stage_select(
    items: list[dict],
    embeddings: np.ndarray,
    data_dir: Path,
    quota: int,
    nms_iou: float = 0.3,
    match_iou: float = 0.7,
    seed: int = 0,
) -> list[int]:
    """Run the two-stage selection and return the chosen item indices.

    items: list of per-sequence metadata (must include 'sequence_folder' and
        'camera'). Aligned with `embeddings` row-by-row.
    embeddings: (n, d) L2-normalized vectors.
    data_dir: base dir where each sequence folder lives.
    quota: target number of selected items.
    Returns: list of indices into `items`/`embeddings`.
    """
    n = len(items)
    if n == 0 or quota <= 0:
        return []

    # Stage 1: bbox-overlap atoms per camera
    main_boxes: dict[int, tuple] = {}
    cameras: dict[int, str] = {}
    for idx, it in enumerate(items):
        seq_dir = data_dir / it["sequence_folder"]
        mb = nms_top1(load_seq_boxes(seq_dir), nms_iou)
        if mb is None:
            continue
        main_boxes[idx] = mb
        cameras[idx] = it["camera"]

    by_cam: dict[str, list[int]] = defaultdict(list)
    for idx, c in cameras.items():
        by_cam[c].append(idx)

    valid = sorted(main_boxes.keys())
    pos_of = {i: p for p, i in enumerate(valid)}
    uf = UnionFind(len(valid))
    for idxs in by_cam.values():
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                a, b = idxs[i], idxs[j]
                if iou_xyxyn(main_boxes[a], main_boxes[b]) > match_iou:
                    uf.union(pos_of[a], pos_of[b])

    members_by_atom: dict[int, list[int]] = defaultdict(list)
    for idx in valid:
        members_by_atom[uf.find(pos_of[idx])].append(idx)
    next_aid = (max(members_by_atom.keys()) + 1) if members_by_atom else 0
    for idx in range(n):
        if idx not in main_boxes:
            members_by_atom[next_aid] = [idx]
            next_aid += 1

    # Atom rep = closest-to-centroid sequence within the atom
    atom_rep: dict[int, int] = {}
    for aid, mem in members_by_atom.items():
        if len(mem) == 1:
            atom_rep[aid] = mem[0]
            continue
        sub = embeddings[mem]
        c = sub.mean(axis=0)
        c = c / max(np.linalg.norm(c), 1e-12)
        atom_rep[aid] = int(mem[int(np.argmax(sub @ c))])
    rep_indices = list(atom_rep.values())
    rep_emb = embeddings[rep_indices]

    # Stage 2: KMeans on rep embeddings
    n_clusters = int(min(quota, len(rep_indices)))
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    labels = km.fit_predict(rep_emb)
    centroids = km.cluster_centers_

    # Pick one final rep per cluster (closest to cluster centroid)
    selected: list[int] = []
    seen: set[int] = set()
    for cid in range(n_clusters):
        pos = np.where(labels == cid)[0]
        if len(pos) == 0:
            continue
        c = centroids[cid] / max(np.linalg.norm(centroids[cid]), 1e-12)
        sims = rep_emb[pos] @ c
        chosen = rep_indices[int(pos[int(np.argmax(sims))])]
        if chosen in seen:
            continue
        seen.add(chosen)
        selected.append(chosen)
    return selected


def load_embeddings(embeddings_dir: Path, split: str) -> tuple[np.ndarray, list[dict]]:
    """Load DINOv2 embeddings and the matching items metadata for a split."""
    base = embeddings_dir / split
    npz = base / "embeddings_dinov2.npz"
    meta = base / "embeddings_meta.json"
    if not npz.exists() or not meta.exists():
        raise FileNotFoundError(
            f"missing embeddings for split={split}: expected {npz} and {meta}. "
            f"Run scripts/embed_fp_for_selection.py first."
        )
    emb = np.load(npz)["embeddings"]
    items = json.loads(meta.read_text())["items"]
    if len(items) != emb.shape[0]:
        raise ValueError(
            f"meta/emb size mismatch for split={split}: "
            f"{len(items)} items vs {emb.shape[0]} embeddings"
        )
    return emb, items
