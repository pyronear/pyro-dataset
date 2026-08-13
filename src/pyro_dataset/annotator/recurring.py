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
from statistics import median
from typing import Any

from pyro_dataset.fp.selection import iou_xyxyn

Bbox = tuple[float, float, float, float]

SPLIT_TARGETS = {"train": 0.9, "val": 0.1}


def main_bbox(alert: dict[str, Any]) -> Bbox | None:
    """The alert's representative box: the per-coordinate median of the boxes
    of its **dominant lane** — the lane contributing the most boxes.

    Per lane, not across lanes: an alert may hold two spatially separate
    artefacts, and a median spanning both lands between them, describing a box
    that exists nowhere and anchoring the recurring object on a phantom.

    NMS is useless here — the export carries no confidence, so every box would
    enter with the same score and the winner would just be whichever appeared
    first in the manifest, an arbitrary first-frame box. The median is robust
    to a drifting plume or a jittering detection, and mirrors how pyro-annotator
    derives its own group representative.
    """
    per_lane: list[list[tuple[float, float, float, float]]] = []
    for obj in alert["objects"]:
        boxes = [
            tuple(box["xyxyn"])
            for frame in obj["frames"]
            for box in frame["boxes"]
            if box["xyxyn"][2] > box["xyxyn"][0] and box["xyxyn"][3] > box["xyxyn"][1]
        ]
        if boxes:
            per_lane.append(boxes)
    if not per_lane:
        return None
    dominant = max(per_lane, key=len)
    return (
        median(b[0] for b in dominant),
        median(b[1] for b in dominant),
        median(b[2] for b in dominant),
        median(b[3] for b in dominant),
    )


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

    def mint(
        self, camera: str, azimuth: int, bbox: Bbox, split: str, alert: str
    ) -> str:
        """Register a newly seen artefact. This is the only moment a split is
        ever chosen for it."""
        ro_id = f"ro_{len(self.entries) + 1:05d}"
        self.entries[ro_id] = {
            "camera": camera,
            "azimuth": azimuth,
            "bbox_xyxyn": list(bbox),
            "split": split,
            "seen_alerts": [alert],
            "ingested_folders": [],
        }
        return ro_id

    def record_sighting(self, ro_id: str, bbox: Bbox, alert: str) -> None:
        """Another alert matched this object: record it.

        Identified by alert rather than counted, because the export is a full
        re-pull: counting would inflate `seen` by one whole history per import
        and bias the frequency ranking against genuinely new artefacts.

        **The anchor bbox is deliberately never moved.** Updating it to each
        new sighting made matching depend on the order and history of the walk:
        replaying the same export against its own ledger re-assigned alerts to
        different objects and minted duplicates, each duplicate then drawing a
        fresh split — the one-artefact-in-two-splits leakage this ledger exists
        to prevent. A frozen anchor makes `match` a pure function of the
        recorded geometry, so a re-pull is a no-op. `bbox` stays in the
        signature because the caller has it and a future scheme may want it.
        """
        entry = self.entries[ro_id]
        if alert not in entry["seen_alerts"]:
            entry["seen_alerts"].append(alert)

    def record_ingested(self, ro_id: str, folder: str) -> None:
        """One of this object's sequences entered the dataset.

        Recording the folder, not a tally, is what lets a later import pick a
        *different* alert of the same artefact when the cap allows another.
        """
        folders = self.entries[ro_id]["ingested_folders"]
        if folder not in folders:
            folders.append(folder)

    def seen(self, ro_id: str) -> int:
        """Distinct alerts ever matched to this object."""
        return len(self.entries[ro_id]["seen_alerts"])

    def ingested(self, ro_id: str) -> list[str]:
        """Folders of this object already staged into the dataset."""
        return list(self.entries[ro_id]["ingested_folders"])
