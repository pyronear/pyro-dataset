"""Recurring-object identity for annotator imports.

A recurring object is one artefact on one camera view — the road that fires
every evening, the antenna on the horizon. Two jobs depend on recognising it:
never importing the same artefact repeatedly, and keeping all of its sequences
inside one split.

Its identity is a surrogate id held in a ledger, never a value derived from
geometry. The representative bbox drifts as sightings accumulate, so a
content-derived key would rename the object and let it land in a second split —
exactly the leakage the ledger exists to prevent. Geometry is only the matching
signal.
"""

import json
import random
from pathlib import Path
from typing import Any

from pyro_dataset.fp.selection import iou_xyxyn, nms_top1

Bbox = tuple[float, float, float, float]

SPLIT_TARGETS = {"train": 0.9, "val": 0.1}


def main_bbox(alert: dict[str, Any], nms_iou: float = 0.3) -> Bbox | None:
    """The alert's dominant box across all frames of all lanes.

    Boxes carry no confidence in the export, so every box enters NMS with the
    same score and ties resolve by appearance order — deterministic, because
    the manifest's lane and frame order is stable.
    """
    boxes: list[tuple[float, float, float, float, float]] = []
    for obj in alert["objects"]:
        for frame in obj["frames"]:
            for box in frame["boxes"]:
                x1, y1, x2, y2 = box["xyxyn"]
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append((x1, y1, x2, y2, 1.0))
    if not boxes:
        return None
    return nms_top1(boxes, nms_iou)


def assign_new_split(rng: random.Random, counts: dict[str, int]) -> str:
    """Greedy 90/10 train/val — whichever split is furthest below its target.

    Test is never a candidate: annotator sequences do not enter it, which is
    what keeps the test set untouched by an import.
    """
    total = sum(counts.get(split, 0) for split in SPLIT_TARGETS) + 1
    deficits = {
        split: SPLIT_TARGETS[split] - (counts.get(split, 0) / total)
        for split in SPLIT_TARGETS
    }
    best = max(deficits.values())
    candidates = sorted(split for split, gap in deficits.items() if gap == best)
    return candidates[0] if len(candidates) == 1 else rng.choice(candidates)


class Ledger:
    """recurring object id -> camera, azimuth, bbox, split, seen, ingested."""

    def __init__(self, entries: dict[str, dict[str, Any]] | None = None) -> None:
        self.entries: dict[str, dict[str, Any]] = entries if entries is not None else {}

    @classmethod
    def load(cls, path: Path | str) -> "Ledger":
        path = Path(path)
        if not path.is_file():
            return cls()
        return cls(json.loads(path.read_text()))

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.entries, indent=2, sort_keys=True) + "\n")

    def match(self, camera: str, azimuth: int, bbox: Bbox, iou: float) -> str | None:
        """Best-IoU existing object on the same view, or None.

        Objects are never merged, even when a new sighting bridges two of them:
        they may hold different splits, and merging would force a sequence to
        change split, which the append-only registry forbids.
        """
        best_id: str | None = None
        best_iou = iou
        for ro_id in sorted(self.entries):
            entry = self.entries[ro_id]
            if entry["camera"] != camera or entry["azimuth"] != azimuth:
                continue
            score = iou_xyxyn(
                tuple(entry["bbox_xyxyn"]) + (1.0,),
                tuple(bbox) + (1.0,),
            )
            if score > best_iou:
                best_id, best_iou = ro_id, score
        return best_id

    def mint(self, camera: str, azimuth: int, bbox: Bbox, split: str) -> str:
        """Register a newly seen artefact. This is the only moment a split is
        ever chosen for it."""
        ro_id = f"ro_{len(self.entries) + 1:05d}"
        self.entries[ro_id] = {
            "camera": camera,
            "azimuth": azimuth,
            "bbox_xyxyn": list(bbox),
            "split": split,
            "seen": 1,
            "ingested": 0,
        }
        return ro_id

    def record_sighting(self, ro_id: str, bbox: Bbox) -> None:
        """Another alert matched this object: count it, refresh the bbox."""
        entry = self.entries[ro_id]
        entry["seen"] += 1
        entry["bbox_xyxyn"] = list(bbox)

    def record_ingested(self, ro_id: str) -> None:
        """One of this object's sequences entered the dataset."""
        self.entries[ro_id]["ingested"] += 1
