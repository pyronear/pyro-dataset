"""Match cameras with the same site name across different azimuths via SIFT.

Pulls sequences from one or more raw datasets (fp + wildfire by default), so a
(site, azimuth) seen in either source contributes to the analysis. For each
multi-azimuth site:
  1. Pick one representative full frame per (site, azimuth) — the highest-scoring
     frame from the most recent sequence at that azimuth, regardless of source.
  2. Run SIFT on the frame.
  3. For every pair of azimuths at the same site, BF-match descriptors,
     apply Lowe's ratio test, then RANSAC homography to count inliers.
  4. Save pairs to data/interim/camera_kp_matches/<split>/pairs.csv.
  5. Save representative metadata to representatives.csv with
     (site, azimuth, image_path, latest_timestamp, n_sequences, source).

Usage:
  uv run python scripts/match_cameras_by_keypoints.py
  uv run python scripts/match_cameras_by_keypoints.py --sources data/raw/fp
  uv run python scripts/match_cameras_by_keypoints.py --split test
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


CAM_RE = re.compile(r"^(.+)_(\d+)$")
TS_RE = re.compile(r"_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})$")


def make_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sources",
        type=Path,
        nargs="+",
        default=[Path("data/raw/fp")],
        help="Raw dataset roots (each must contain registry.json and data/).",
    )
    p.add_argument(
        "--split",
        default="all",
        help="all (default) | train | val | test",
    )
    p.add_argument("--output", type=Path, default=Path("data/interim/camera_kp_matches"))
    p.add_argument("--n-features", type=int, default=2000)
    p.add_argument("--ratio", type=float, default=0.75, help="Lowe's ratio test threshold.")
    p.add_argument("--ransac-thresh", type=float, default=5.0, help="RANSAC reprojection threshold (pixels).")
    p.add_argument(
        "--max-dcx",
        type=float,
        default=0.50,
        help="Hard-reject if |Δcx| (horizontal shift between inlier centroids, "
        "normalized by image width) exceeds this. Catches rotational neighbours "
        "where opposite edges overlap.",
    )
    p.add_argument(
        "--scale-min",
        type=float,
        default=0.60,
        help="Hard-reject if the homography scale (det of 2x2 linear part) is "
        "below this. Catches degenerate / inconsistent transformations.",
    )
    p.add_argument(
        "--scale-max",
        type=float,
        default=1.70,
        help="Hard-reject if the homography scale exceeds this.",
    )
    return p


def extract_timestamp(folder_name: str) -> str | None:
    m = TS_RE.search(folder_name)
    return m.group(1) if m else None


def split_camera(cam: str) -> tuple[str, int | None]:
    m = CAM_RE.match(cam)
    if not m:
        return cam, None
    return m.group(1), int(m.group(2))


def best_frame(seq_dir: Path) -> Path | None:
    """Return the image path of the most prominent frame in a sequence.

    Picks the frame with the highest detection score when labels have a 6th
    column (class cx cy w h score, FP-style), and falls back to the frame with
    the largest bounding box area when labels are 5-column GT (wildfire-style).
    Empty labels are skipped.
    """
    labels_dir = seq_dir / "labels"
    images_dir = seq_dir / "images"
    if not labels_dir.is_dir() or not images_dir.is_dir():
        return None
    best_score: float = -1.0
    best_img: Path | None = None
    best_area: float = -1.0
    best_img_area: Path | None = None
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
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                continue
            if w <= 0 or h <= 0:
                continue
            img = images_dir / (lbl.stem + ".jpg")
            if not img.exists():
                continue
            if len(parts) >= 6:
                try:
                    score = float(parts[5])
                except ValueError:
                    score = None
                if score is not None and score > best_score:
                    best_score = score
                    best_img = img
            area = w * h
            if area > best_area:
                best_area = area
                best_img_area = img
    return best_img if best_img is not None else best_img_area


def main():
    args = make_cli_parser().parse_args()
    out_dir = args.output / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load sequences from every source root
    all_sequences: list[dict] = []
    for src in args.sources:
        reg = src / "registry.json"
        data_dir = src / "data"
        if not reg.exists() or not data_dir.is_dir():
            print(f"  skipping {src}: missing registry.json or data/")
            continue
        seqs = json.loads(reg.read_text())["sequences"]
        for s in seqs:
            s["__data_dir"] = str(data_dir)
            s["__source"] = src.name
            all_sequences.append(s)
    if args.split != "all":
        all_sequences = [s for s in all_sequences if s["split"] == args.split]
    sources_summary = ", ".join(str(s) for s in args.sources)
    print(f"loaded {len(all_sequences)} sequences from {{ {sources_summary} }}  split={args.split}")

    # Group sequences by (site, azimuth) and pick the most-recent one as representative
    by_key: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for s in all_sequences:
        site, azimuth = split_camera(s["camera"])
        if azimuth is None:
            continue
        by_key[(site, azimuth)].append(s)

    seen: dict[tuple[str, int], Path] = {}
    latest_ts: dict[tuple[str, int], str] = {}
    n_seqs: dict[tuple[str, int], int] = {}
    source_per_key: dict[tuple[str, int], str] = {}
    for key, seqs in by_key.items():
        seqs_sorted = sorted(
            seqs,
            key=lambda s: extract_timestamp(s["folder"]) or "",
            reverse=True,
        )
        n_seqs[key] = len(seqs_sorted)
        for s in seqs_sorted:
            seq_dir = Path(s["__data_dir"]) / s["folder"]
            if not seq_dir.is_dir():
                continue
            bf = best_frame(seq_dir)
            if bf is not None:
                seen[key] = bf
                latest_ts[key] = extract_timestamp(s["folder"]) or ""
                source_per_key[key] = s["__source"]
                break

    by_site: dict[str, list[int]] = defaultdict(list)
    for site, azimuth in seen.keys():
        by_site[site].append(azimuth)
    multi_sites = {site: sorted(az) for site, az in by_site.items() if len(az) >= 2}
    print(f"sites total: {len(by_site)}  multi-azimuth: {len(multi_sites)}")
    n_pairs = sum(len(az) * (len(az) - 1) // 2 for az in multi_sites.values())
    print(f"pairs to compare: {n_pairs}")

    # SIFT per (site, azimuth) — only for multi-azimuth sites
    sift = cv2.SIFT_create(nfeatures=args.n_features)
    keypoints: dict[tuple[str, int], tuple[list, np.ndarray]] = {}
    image_size: dict[tuple[str, int], tuple[int, int]] = {}  # (W, H)
    repr_csv_rows: list[tuple[str, int, str, str, int, str]] = []
    for site, azs in tqdm(multi_sites.items(), desc="SIFT"):
        for az in azs:
            img_path = seen[(site, az)]
            key = (site, az)
            repr_csv_rows.append(
                (
                    site,
                    az,
                    str(img_path),
                    latest_ts.get(key, ""),
                    n_seqs.get(key, 0),
                    source_per_key.get(key, ""),
                )
            )
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                keypoints[(site, az)] = ([], np.empty((0, 128), dtype=np.float32))
                image_size[(site, az)] = (1, 1)
                continue
            h_img, w_img = img.shape[:2]
            image_size[(site, az)] = (w_img, h_img)
            kp, desc = sift.detectAndCompute(img, None)
            if desc is None:
                desc = np.empty((0, 128), dtype=np.float32)
            keypoints[(site, az)] = (kp, desc)

    bf_matcher = cv2.BFMatcher(cv2.NORM_L2)

    pair_rows: list[dict] = []
    n_dropped = {"dcx": 0, "scale": 0, "degenerate": 0}
    for site, azs in tqdm(multi_sites.items(), desc="match pairs"):
        for i in range(len(azs)):
            for j in range(i + 1, len(azs)):
                a, b = azs[i], azs[j]
                kp_a, desc_a = keypoints[(site, a)]
                kp_b, desc_b = keypoints[(site, b)]
                if len(desc_a) < 4 or len(desc_b) < 4:
                    continue
                raw = bf_matcher.knnMatch(desc_a, desc_b, k=2)
                good = []
                for pair in raw:
                    if len(pair) < 2:
                        continue
                    m, n = pair
                    if m.distance < args.ratio * n.distance:
                        good.append(m)
                n_inliers = 0
                dcx = 0.0
                h_scale = float("nan")
                s_area = 0.0
                d_area = 0.0
                if len(good) >= 4:
                    src = np.float32([kp_a[m.queryIdx].pt for m in good])
                    dst = np.float32([kp_b[m.trainIdx].pt for m in good])
                    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, args.ransac_thresh)
                    if mask is not None:
                        m1 = mask.ravel().astype(bool)
                        n_inliers = int(m1.sum())
                        if n_inliers >= 4:
                            in_src = src[m1]
                            in_dst = dst[m1]
                            wa, ha = image_size.get((site, a), (1, 1))
                            wb, hb = image_size.get((site, b), (1, 1))
                            sx0, sy0 = float(in_src[:, 0].min()) / wa, float(in_src[:, 1].min()) / ha
                            sx1, sy1 = float(in_src[:, 0].max()) / wa, float(in_src[:, 1].max()) / ha
                            dx0, dy0 = float(in_dst[:, 0].min()) / wb, float(in_dst[:, 1].min()) / hb
                            dx1, dy1 = float(in_dst[:, 0].max()) / wb, float(in_dst[:, 1].max()) / hb
                            s_area = (sx1 - sx0) * (sy1 - sy0)
                            d_area = (dx1 - dx0) * (dy1 - dy0)
                            dcx = ((dx0 + dx1) - (sx0 + sx1)) / 2.0
                            if H is not None:
                                det = float(H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0])
                                h_scale = float(np.sqrt(det)) if det > 0 else float("nan")

                # Hard-reject filters: only drop when we are confident
                if not np.isnan(h_scale):
                    if abs(dcx) > args.max_dcx:
                        n_dropped["dcx"] += 1
                        continue
                    if h_scale < args.scale_min or h_scale > args.scale_max:
                        n_dropped["scale"] += 1
                        continue
                else:
                    if n_inliers >= 4:
                        n_dropped["degenerate"] += 1
                        continue

                pair_rows.append(
                    {
                        "site": site,
                        "azimuth_a": a,
                        "azimuth_b": b,
                        "n_kp_a": len(kp_a),
                        "n_kp_b": len(kp_b),
                        "n_good": len(good),
                        "n_inliers": n_inliers,
                        "dcx": round(dcx, 4),
                        "h_scale": round(h_scale, 4) if not np.isnan(h_scale) else "",
                        "s_area": round(s_area, 4),
                        "d_area": round(d_area, 4),
                    }
                )

    pair_rows.sort(key=lambda r: -r["n_inliers"])
    print(
        f"hard-rejected: {sum(n_dropped.values())} pairs  "
        f"(|dcx|>{args.max_dcx}: {n_dropped['dcx']}, "
        f"scale outside [{args.scale_min}, {args.scale_max}]: {n_dropped['scale']}, "
        f"degenerate H: {n_dropped['degenerate']})"
    )

    repr_csv = out_dir / "representatives.csv"
    with repr_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["site", "azimuth", "image_path", "latest_timestamp", "n_sequences", "source"])
        w.writerows(repr_csv_rows)

    pairs_csv = out_dir / "pairs.csv"
    with pairs_csv.open("w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "site",
                "azimuth_a",
                "azimuth_b",
                "n_kp_a",
                "n_kp_b",
                "n_good",
                "n_inliers",
                "dcx",
                "h_scale",
                "s_area",
                "d_area",
            ],
        )
        w.writeheader()
        w.writerows(pair_rows)

    print(f"\nwrote {repr_csv}  ({len(repr_csv_rows)} representatives)")
    print(f"wrote {pairs_csv}   ({len(pair_rows)} pairs)")
    print("\ntop-15 pairs by inliers:")
    for r in pair_rows[:15]:
        print(
            f"  {r['site']:<40} {r['azimuth_a']:>4} ↔ {r['azimuth_b']:>4}  "
            f"good={r['n_good']:>4}  inliers={r['n_inliers']:>4}"
        )


if __name__ == "__main__":
    main()
