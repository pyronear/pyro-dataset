"""Park annotator alerts whose artefact the dataset already holds.

The classify queue is full of alerts the import can never use: with
`--max-per-object 1`, a recurring object that already contributed a sequence is
done for life, so every further sighting of it costs annotation time and yields
nothing. This walks the queue, matches each pending alert against the ledger the
same way `plan_annotator_import.py` does, and skips the ones landing on a
saturated object.

Skipping is recoverable — `DELETE /sequences/alert/skip` puts an alert back in
whichever queue its stages qualify it for — but it is still a write to the
annotator, so the default is a dry run.

Alerts the platform labelled as smoke are never skipped: every smoke alert is
imported regardless of recurring objects, and the ledger match only speaks about
where the detector fired, not about what is burning there.

    uv run python scripts/skip_covered_alerts.py --api-url http://host:5050
    uv run python scripts/skip_covered_alerts.py --api-url http://host:5050 --apply
"""

import argparse
import json
import logging
import os
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median
from typing import Any

from pyro_dataset.annotator.recurring import Bbox, Ledger

ROOT = Path(__file__).parent.parent
LEDGER = ROOT / "data" / "raw" / "pyro-annotator" / "recurring_objects.json"
SMOKE_LABELS = {"wildfire_smoke", "other_smoke"}


def request(
    url: str,
    token: str | None = None,
    payload: dict | None = None,
    method: str | None = None,
) -> Any:
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=60) as response:
        body = response.read()
    return json.loads(body) if body else None


def login(api: str, user: str, password: str) -> str:
    payload = {"username": user, "password": password}
    return request(f"{api}/api/v1/auth/login", payload=payload)["access_token"]


def classify_queue(api: str, token: str) -> list[dict]:
    """Every alert still waiting to be classified."""
    items: list[dict] = []
    page = 1
    while True:
        got = request(
            f"{api}/api/v1/sequences/classify-queue?size=100&page={page}", token
        )
        items += got["items"]
        if page >= got["pages"] or not got["items"]:
            return items
        page += 1


def engine_bbox(api: str, token: str, sequence_id: int) -> Bbox | None:
    """Median of the engine's predicted boxes over the sequence.

    The annotated equivalent is `recurring.main_bbox`; before annotation the
    engine's own predictions are the only geometry there is, and the median is
    robust to the same jitter.
    """
    got = request(f"{api}/api/v1/detections/?sequence_id={sequence_id}&size=100", token)
    boxes = [
        prediction["xyxyn"]
        for item in got["items"]
        for prediction in (
            (item.get("algo_predictions") or {}).get("predictions") or []
        )
        if prediction["xyxyn"][2] > prediction["xyxyn"][0]
        and prediction["xyxyn"][3] > prediction["xyxyn"][1]
    ]
    if not boxes:
        return None
    x1, y1, x2, y2 = (median(box[index] for box in boxes) for index in range(4))
    return (x1, y1, x2, y2)


def is_covered(
    ledger: Ledger, alert: dict, bbox: Bbox | None, iou: float, max_per_object: int
) -> str | None:
    """The saturated object this alert would land on, or None.

    None covers every reason to leave an alert alone: no geometry to match on, a
    smoke label from the platform, no matching object, or an object that still
    has room under the cap.
    """
    if bbox is None or alert.get("is_wildfire_alertapi") in SMOKE_LABELS:
        return None
    camera = f"{alert['organisation_name']}_{alert['camera_name']}"
    ro_id = ledger.match(camera, int(alert["azimuth"]), bbox, iou)
    if ro_id is None:
        return None
    if len(ledger.entries[ro_id]["ingested_folders"]) < max_per_object:
        return None
    return ro_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", required=True, help="Annotation API base URL")
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument(
        "--iou", type=float, default=0.3, help="Match IoU (default: 0.3)"
    )
    parser.add_argument("--max-per-object", type=int, default=1)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--apply", action="store_true", help="Actually skip; default is a dry run"
    )
    parser.add_argument("--loglevel", default="info")
    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel.upper(), format="%(message)s")

    user = os.getenv("MAIN_ANNOTATION_LOGIN") or os.getenv("ANNOTATOR_LOGIN", "admin")
    password = os.getenv("MAIN_ANNOTATION_PASSWORD") or os.getenv("ANNOTATOR_PASSWORD")
    if not password:
        sys.exit("Set MAIN_ANNOTATION_PASSWORD (or ANNOTATOR_PASSWORD).")

    api = args.api_url.rstrip("/")
    token = login(api, user, password)
    ledger = Ledger.load(args.ledger)
    saturated = sum(
        1
        for entry in ledger.entries.values()
        if len(entry["ingested_folders"]) >= args.max_per_object
    )
    pending = classify_queue(api, token)
    logging.info(
        f"{len(pending)} alertes en attente; ledger: {len(ledger.entries)} objets, "
        f"{saturated} saturés à {args.max_per_object} séquence(s)"
    )

    def covered(alert: dict) -> tuple[dict, str | None]:
        try:
            bbox = engine_bbox(api, token, alert["primary_sequence_id"])
        except (urllib.error.URLError, KeyError, ValueError) as error:
            logging.warning(f"{alert['platform_alert_id']}: {error}")
            return alert, None
        return alert, is_covered(ledger, alert, bbox, args.iou, args.max_per_object)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        hits = [(alert, ro) for alert, ro in pool.map(covered, pending) if ro]

    for alert, ro_id in hits:
        logging.info(
            f"  {alert['source_api']}:{alert['platform_alert_id']:<8} "
            f"{alert['organisation_name']}_{alert['camera_name']} "
            f"az={int(alert['azimuth'])} -> {ro_id}"
        )
    if not args.apply:
        logging.info(
            f"\nDRY RUN — {len(hits)} alertes seraient skippées. --apply pour écrire."
        )
        return

    for alert, ro_id in hits:
        request(
            f"{api}/api/v1/sequences/alert/skip",
            token,
            payload={
                "source_api": alert["source_api"],
                "platform_alert_id": alert["platform_alert_id"],
                "note": f"couvert par {ro_id}, déjà dans le dataset",
            },
        )
    logging.info(
        f"\n{len(hits)} alertes skippées (réversibles via DELETE /sequences/alert/skip)."
    )


if __name__ == "__main__":
    main()
