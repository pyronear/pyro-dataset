"""Streamlit app to browse the annotator recurring-object ledger.

Reads `data/raw/pyro-annotator/` — the ledger plus the export it was minted
from — so it needs no `dvc pull` of the built pools: an object's ingested
sequence is the same folder the export already holds on disk.
"""

import itertools
import json
from pathlib import Path

import cv2
import streamlit as st

from pyro_dataset.annotator.convert import folder_name

ROOT = Path(__file__).parent.parent
ANNOTATOR = ROOT / "data" / "raw" / "pyro-annotator"
EXPORT = ANNOTATOR / "export"

st.set_page_config(page_title="Recurring objects", layout="wide")


@st.cache_data
def load() -> tuple[dict, dict, dict]:
    """Ledger, alerts keyed as the ledger references them, folder -> alert."""
    ledger = json.loads((ANNOTATOR / "recurring_objects.json").read_text())
    alerts, by_folder = {}, {}
    with (EXPORT / "manifest.jsonl").open() as lines:
        for line in lines:
            alert = json.loads(line)
            alerts[f"{alert['source_api']}:{alert['platform_alert_id']}"] = alert
            by_folder[folder_name(alert)] = alert
    return ledger, alerts, by_folder


def fp_types(alert: dict) -> str:
    types = {t for obj in alert["objects"] for t in (obj["false_positive_types"] or [])}
    return ", ".join(sorted(types)) or "—"


def render(frame: dict, anchor: list[float] | None, zoom: bool = False) -> "cv2.Mat":
    """One frame: annotated boxes in red, the object's frozen anchor in green.

    `zoom` crops around the anchor — these artefacts occupy a couple of percent
    of the frame, so a full view shows a landscape and no object.
    """
    img = cv2.imread(str(EXPORT / frame["image_path"]))
    if img is None:
        return None
    height, width = img.shape[:2]
    for box in frame["boxes"]:
        x1, y1, x2, y2 = box["xyxyn"]
        cv2.rectangle(
            img,
            (int(x1 * width), int(y1 * height)),
            (int(x2 * width), int(y2 * height)),
            (0, 0, 255),
            2,
        )
    if anchor:
        x1, y1, x2, y2 = anchor
        cv2.rectangle(
            img,
            (int(x1 * width), int(y1 * height)),
            (int(x2 * width), int(y2 * height)),
            (0, 255, 0),
            1,
        )
    if zoom and anchor:
        x1, y1, x2, y2 = anchor
        pad = max(x2 - x1, y2 - y1) * 3 + 0.03
        left, right = int(max(x1 - pad, 0) * width), int(min(x2 + pad, 1) * width)
        top, bottom = int(max(y1 - pad, 0) * height), int(min(y2 + pad, 1) * height)
        img = img[top:bottom, left:right]
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


ledger, alerts, by_folder = load()
mode = st.sidebar.radio("Vue", ["Objets récurrents", "Doublons suspects"])
zoom = st.sidebar.checkbox("Zoom sur l'objet", value=True)

if mode == "Objets récurrents":
    order = sorted(ledger, key=lambda ro: -len(ledger[ro]["seen_alerts"]))
    labels = {
        ro: f"{ro} · {ledger[ro]['camera']} az={ledger[ro]['azimuth']} "
        f"· {len(ledger[ro]['seen_alerts'])} vues"
        for ro in order
    }
    ro_id = st.sidebar.radio(
        f"{len(ledger)} objets",
        order,
        format_func=labels.get,
        label_visibility="visible",
    )
    entry = ledger[ro_id]
    sightings = [alerts[a] for a in entry["seen_alerts"] if a in alerts]

    st.title(f"{ro_id} — {entry['camera']} az={entry['azimuth']}")
    columns = st.columns(4)
    columns[0].metric("Sightings", len(entry["seen_alerts"]))
    columns[1].metric("Séquences au dataset", len(entry["ingested_folders"]))
    columns[2].metric("Split", entry["split"])
    columns[3].metric("Type", fp_types(sightings[0]) if sightings else "—")
    st.caption(
        f"Ancre figée (vert) : {[round(v, 3) for v in entry['bbox_xyxyn']]} — "
        "les boîtes annotées sont en rouge."
    )

    ingested = set(entry["ingested_folders"])
    # Ingested first: the point of the page is comparing what entered the
    # dataset against everything the same artefact also produced.
    sightings.sort(key=lambda a: (folder_name(a) not in ingested, a["recorded_at"]))
    for alert in sightings:
        folder = folder_name(alert)
        in_dataset = folder in ingested
        title = ("✅ DANS LE DATASET — " if in_dataset else "") + (
            f"{folder} · {alert['recorded_at'][:19]} · {fp_types(alert)}"
        )
        with st.expander(title, expanded=in_dataset):
            frames = [f for obj in alert["objects"] for f in obj["frames"]]
            for row in range(0, min(len(frames), 8), 4):
                for column, frame in zip(st.columns(4), frames[row : row + 4]):
                    img = render(frame, entry["bbox_xyxyn"], zoom)
                    if img is not None:
                        column.image(img, caption=frame["recorded_at"][11:19])
            st.caption(f"{len(frames)} frames — {len(frames[:8])} affichées")

else:
    st.title("Doublons suspects")
    st.caption(
        "Deux objets sur la même caméra à moins de 5° l'un de l'autre : "
        "l'appariement exige l'égalité stricte de l'azimut, donc un même "
        "artefact filmé à un pan légèrement différent devient deux objets — "
        "et peut atterrir dans deux splits."
    )
    pairs = [
        (a, b)
        for a, b in itertools.combinations(sorted(ledger), 2)
        if ledger[a]["camera"] == ledger[b]["camera"]
        and 0 < abs(int(ledger[a]["azimuth"]) - int(ledger[b]["azimuth"])) <= 5
    ]
    st.metric("Paires", len(pairs))
    for left, right in pairs:
        st.subheader(f"{ledger[left]['camera']} — {left} vs {right}")
        for column, ro_id in zip(st.columns(2), (left, right)):
            entry = ledger[ro_id]
            column.write(
                f"**{ro_id}** az={entry['azimuth']} · split=`{entry['split']}` · "
                f"{len(entry['seen_alerts'])} vues"
            )
            folders = entry["ingested_folders"] or [
                folder_name(alerts[a]) for a in entry["seen_alerts"][:1] if a in alerts
            ]
            alert = by_folder.get(folders[0]) if folders else None
            if alert:
                frames = [f for obj in alert["objects"] for f in obj["frames"]]
                img = render(frames[0], entry["bbox_xyxyn"], zoom) if frames else None
                if img is not None:
                    column.image(img, caption=folders[0])
