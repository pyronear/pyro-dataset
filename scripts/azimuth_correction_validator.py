"""Streamlit app to validate azimuth correction proposals.

For each pair (site, azimuth_a, azimuth_b) with enough SIFT inliers, show the
two representative frames with matched keypoints overlaid and let the user:
  - Accept the match (and choose which azimuth to keep — defaults to the
    azimuth whose most recent sequence is the most recent overall)
  - Reject the match
  - Skip / revisit later

Decisions are persisted to data/interim/camera_kp_matches/<split>/decisions.json.

Run after match_cameras_by_keypoints.py:
  uv run streamlit run scripts/azimuth_correction_validator.py
"""

import csv
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).parent.parent
MATCHES_ROOT = ROOT / "data" / "interim" / "camera_kp_matches"
DEFAULT_MIN_INLIERS = 30


def pair_id(p: dict) -> str:
    return f"{p['site']}|{p['azimuth_a']}|{p['azimuth_b']}"


@st.cache_data
def load_pairs(split: str):
    base = MATCHES_ROOT / split
    pairs = []
    with (base / "pairs.csv").open() as f:
        for r in csv.DictReader(f):
            def _f(v):
                if v is None or v == "":
                    return None
                try:
                    return float(v)
                except ValueError:
                    return None
            pairs.append(
                {
                    "site": r["site"],
                    "azimuth_a": int(r["azimuth_a"]),
                    "azimuth_b": int(r["azimuth_b"]),
                    "n_kp_a": int(r["n_kp_a"]),
                    "n_kp_b": int(r["n_kp_b"]),
                    "n_good": int(r["n_good"]),
                    "n_inliers": int(r["n_inliers"]),
                    "dcx": _f(r.get("dcx")),
                    "h_scale": _f(r.get("h_scale")),
                    "s_area": _f(r.get("s_area")),
                    "d_area": _f(r.get("d_area")),
                }
            )
    pairs.sort(key=lambda r: -r["n_inliers"])

    reprs: dict[tuple[str, int], dict] = {}
    with (base / "representatives.csv").open() as f:
        for r in csv.DictReader(f):
            reprs[(r["site"], int(r["azimuth"]))] = {
                "image_path": r["image_path"],
                "latest_timestamp": r.get("latest_timestamp", ""),
                "n_sequences": int(r.get("n_sequences", 0) or 0),
            }
    return pairs, reprs


def load_decisions(split: str) -> dict:
    fp = MATCHES_ROOT / split / "decisions.json"
    if fp.exists():
        return json.loads(fp.read_text())
    return {}


def save_decisions(split: str, decisions: dict) -> None:
    fp = MATCHES_ROOT / split / "decisions.json"
    fp.write_text(json.dumps(decisions, indent=2))


@st.cache_data
def compute_overlay(
    img_a_path: str, img_b_path: str, n_features: int, max_lines: int
):
    img_a = cv2.imread(img_a_path)
    img_b = cv2.imread(img_b_path)
    if img_a is None or img_b is None:
        return None
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=n_features)
    kp_a, desc_a = sift.detectAndCompute(gray_a, None)
    kp_b, desc_b = sift.detectAndCompute(gray_b, None)
    if desc_a is None or desc_b is None or len(desc_a) < 4 or len(desc_b) < 4:
        return None
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(desc_a, desc_b, k=2)
    good = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 4:
        return None
    src = np.float32([kp_a[m.queryIdx].pt for m in good])
    dst = np.float32([kp_b[m.trainIdx].pt for m in good])
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if mask is None:
        return None
    inlier_idx = [i for i, ok in enumerate(mask.ravel()) if ok]
    if max_lines and len(inlier_idx) > max_lines:
        rng = np.random.default_rng(0)
        inlier_idx = list(rng.choice(inlier_idx, size=max_lines, replace=False))
    inlier_matches = [good[i] for i in inlier_idx]
    overlay = cv2.drawMatches(
        img_a,
        kp_a,
        img_b,
        kp_b,
        inlier_matches,
        None,
        matchColor=(0, 220, 0),
        singlePointColor=(120, 120, 120),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


# ---------- UI ----------

st.set_page_config(page_title="Azimuth validator", layout="wide")
st.title("Azimuth correction validator")

available = (
    [p.name for p in MATCHES_ROOT.iterdir() if p.is_dir() and (p / "pairs.csv").exists()]
    if MATCHES_ROOT.is_dir()
    else []
)
if not available:
    st.error("No matches found. Run scripts/match_cameras_by_keypoints.py first.")
    st.stop()

split = st.sidebar.selectbox(
    "split", sorted(available), index=sorted(available).index("all") if "all" in available else 0
)
pairs, reprs = load_pairs(split)
decisions = load_decisions(split)


def status_of(p: dict) -> str:
    return decisions.get(pair_id(p), {}).get("status", "pending")


max_inliers_total = max((p["n_inliers"] for p in pairs), default=0) or 1
min_inliers = st.sidebar.slider(
    "min inliers", 10, max_inliers_total, min(DEFAULT_MIN_INLIERS, max_inliers_total)
)
status_filter = st.sidebar.radio(
    "status filter", ["pending", "accepted", "rejected", "all"]
)
sites = sorted({p["site"] for p in pairs})
site_filter = st.sidebar.selectbox("site filter", ["(all)"] + sites)
max_lines = st.sidebar.slider("max match lines drawn", 20, 400, 80)
n_features = st.sidebar.slider("SIFT n_features", 500, 4000, 2000, step=500)


def passes(p: dict) -> bool:
    if p["n_inliers"] < min_inliers:
        return False
    if status_filter != "all" and status_of(p) != status_filter:
        return False
    if site_filter != "(all)" and p["site"] != site_filter:
        return False
    return True


filtered = [p for p in pairs if passes(p)]

# Counts for global progress
n_total_eligible = sum(1 for p in pairs if p["n_inliers"] >= min_inliers)
n_acc = sum(1 for p in pairs if p["n_inliers"] >= min_inliers and status_of(p) == "accepted")
n_rej = sum(1 for p in pairs if p["n_inliers"] >= min_inliers and status_of(p) == "rejected")
n_pen = n_total_eligible - n_acc - n_rej

st.sidebar.markdown(
    f"**progress**  \n"
    f"eligible (≥{min_inliers}): {n_total_eligible}  \n"
    f"✅ accepted: {n_acc}  \n"
    f"❌ rejected: {n_rej}  \n"
    f"⏳ pending: {n_pen}"
)

if not filtered:
    st.info("No pairs to show with current filters.")
    st.stop()

idx = st.session_state.get("pair_idx", 0)
idx = max(0, min(idx, len(filtered) - 1))
p = filtered[idx]
pid = pair_id(p)
decision = decisions.get(pid, {})

a_meta = reprs[(p["site"], p["azimuth_a"])]
b_meta = reprs[(p["site"], p["azimuth_b"])]

# Default "keep" choice: most recent timestamp wins; on ties, more sequences wins.
def _score(meta: dict):
    return (meta.get("latest_timestamp", ""), meta.get("n_sequences", 0))

if decision.get("status") == "accepted" and decision.get("kept_azimuth") is not None:
    default_keep = int(decision["kept_azimuth"])
elif _score(a_meta) >= _score(b_meta):
    default_keep = p["azimuth_a"]
else:
    default_keep = p["azimuth_b"]

# Header
left, right = st.columns([3, 2])
with left:
    st.subheader(
        f"#{idx + 1}/{len(filtered)}  ·  {p['site']}  ·  azimuth "
        f"**{p['azimuth_a']}** ↔ **{p['azimuth_b']}**"
    )
    def _fmt(v, prec=2, default="—"):
        return default if v is None else f"{v:.{prec}f}"

    st.caption(
        f"inliers=**{p['n_inliers']}** · good={p['n_good']} · "
        f"kp_a={p['n_kp_a']} kp_b={p['n_kp_b']}  |  "
        f"Δcx={_fmt(p['dcx'])} · H_scale={_fmt(p['h_scale'])} · "
        f"s_area={_fmt(p['s_area'])} · d_area={_fmt(p['d_area'])}"
    )
with right:
    if decision.get("status") and decision["status"] != "pending":
        st.info(
            f"current decision: **{decision['status']}**"
            + (
                f" · keep azimuth **{decision.get('kept_azimuth')}**"
                if decision["status"] == "accepted"
                else ""
            )
        )

# Overlay
overlay = compute_overlay(a_meta["image_path"], b_meta["image_path"], n_features, max_lines)
if overlay is None:
    st.warning("could not compute overlay")
else:
    st.image(overlay, caption=f"left: az {p['azimuth_a']}  ·  right: az {p['azimuth_b']}", width="stretch")

# Metadata per side
m1, m2 = st.columns(2)
with m1:
    st.markdown(
        f"**azimuth {p['azimuth_a']}**  \n"
        f"latest: `{a_meta['latest_timestamp']}`  \n"
        f"n_sequences: {a_meta['n_sequences']}"
    )
with m2:
    st.markdown(
        f"**azimuth {p['azimuth_b']}**  \n"
        f"latest: `{b_meta['latest_timestamp']}`  \n"
        f"n_sequences: {b_meta['n_sequences']}"
    )

# Decision controls
keep = st.radio(
    "keep which azimuth?",
    [p["azimuth_a"], p["azimuth_b"]],
    index=0 if default_keep == p["azimuth_a"] else 1,
    horizontal=True,
    key=f"keep_{pid}",
)


def _record(status: str, kept):
    decisions[pid] = {
        "status": status,
        "kept_azimuth": int(kept) if kept is not None else None,
        "discarded_azimuth": (
            p["azimuth_a"] if kept == p["azimuth_b"] else p["azimuth_b"]
        )
        if kept is not None
        else None,
        "n_inliers": p["n_inliers"],
        "decided_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_decisions(split, decisions)
    st.session_state["pair_idx"] = idx + 1


def _skip():
    st.session_state["pair_idx"] = idx + 1


def _back():
    st.session_state["pair_idx"] = max(0, idx - 1)


def _clear():
    decisions.pop(pid, None)
    save_decisions(split, decisions)


c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
c1.button("✅ Accept", on_click=_record, args=("accepted", keep), width="stretch")
c2.button("❌ Reject", on_click=_record, args=("rejected", None), width="stretch")
c3.button("⏭ Skip", on_click=_skip, width="stretch")
c4.button("⏮ Previous", on_click=_back, width="stretch")
c5.button("↺ Clear", on_click=_clear, width="stretch", help="Remove existing decision for this pair.")
