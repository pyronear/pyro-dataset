"""Rank recurring objects and pick which of their alerts to import.

Hard negatives first: an object whose alerts still fool the deployed temporal
model is worth more than one that merely fires often. Within a bucket,
frequency decides.
"""

from typing import Any


def rank_objects(
    objects: dict[str, dict[str, Any]], threshold: float = 0.5
) -> list[str]:
    """Recurring object ids, best first.

    Bucket 1 holds objects whose best `temporal_model_score` reaches
    `threshold` — false positives the deployed model believes. Bucket 2 holds
    the rest, including objects with no scored alert. Bucketing rather than
    blending keeps the order explainable and robust to score noise, and with
    no scores at all it degrades to frequency-only.
    """

    def sort_key(ro_id: str) -> tuple[int, int, str]:
        entry = objects[ro_id]
        score = entry.get("score")
        fooling = 1 if score is not None and score >= threshold else 0
        return (-fooling, -entry["seen"], ro_id)

    return sorted(objects, key=sort_key)


def select_fp(
    ranked: list[str],
    per_object_alerts: dict[str, list[dict[str, Any]]],
    quota: int,
    max_per_object: int,
    ingested: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """Walk the ranking, taking alerts until the quota is met.

    One alert per object per pass, so the quota spreads across distinct
    artefacts before it deepens any of them. `ingested` maps each object to the
    folders it already contributed, making `max_per_object` a lifetime cap.

    Those folders are skipped rather than merely counted. Counting alone made
    an object re-pick its already-staged alert on the next run, which was then
    silently dropped as an existing folder — burning a quota slot and leaving
    the object permanently short of its cap.

    Alerts within an object are taken in the order the caller supplied — the
    importer sorts them best-first — and are copied, not mutated.
    """
    picked: list[dict[str, Any]] = []
    fresh: dict[str, list[dict[str, Any]]] = {
        ro_id: [a for a in alerts if a["folder"] not in set(ingested.get(ro_id, []))]
        for ro_id, alerts in per_object_alerts.items()
    }
    taken = {ro_id: len(folders) for ro_id, folders in ingested.items()}
    used_this_run: dict[str, int] = {}

    for _pass in range(max_per_object):
        for ro_id in ranked:
            if len(picked) >= quota:
                return picked
            if taken.get(ro_id, 0) >= max_per_object:
                continue
            alerts = fresh.get(ro_id, [])
            index = used_this_run.get(ro_id, 0)
            if index >= len(alerts):
                continue
            picked.append({**alerts[index], "recurring_object": ro_id})
            taken[ro_id] = taken.get(ro_id, 0) + 1
            used_this_run[ro_id] = index + 1
    return picked
