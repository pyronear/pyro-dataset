"""Resolve accepted azimuth-pair decisions into per-site clusters.

Reads decisions.json (output of azimuth_correction_validator.py) and produces
the final discard→keep mapping. Pairs are clustered per site via union-find.
Within each cluster the canonical azimuth is chosen by:
  1. votes (number of accepted pairs in the cluster where this azimuth was
     marked as 'kept_azimuth' by the user)
  2. latest sequence timestamp (most recent wins)
  3. n_sequences (more wins)
  4. azimuth value (deterministic last-resort)

Output: azimuth_remap.csv with one row per (site, discard_azimuth) →
keep_azimuth, plus per-split sequence counts and the cluster id.

Usage:
  uv run python scripts/resolve_azimuth_clusters.py
  uv run python scripts/resolve_azimuth_clusters.py --split test
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


CAM_RE = re.compile(r"^(.+)_(\d+)$")


def make_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--matches-dir", type=Path, default=Path("data/interim/camera_kp_matches"))
    p.add_argument("--split", default="all", help="Which split's decisions.json to consume.")
    p.add_argument("--registry", type=Path, default=Path("data/raw/fp/registry.json"))
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <matches-dir>/<split>/azimuth_remap.csv.",
    )
    return p


class UnionFind:
    def __init__(self) -> None:
        self.p: dict = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def main() -> None:
    args = make_cli_parser().parse_args()
    base = args.matches_dir / args.split

    decisions = json.loads((base / "decisions.json").read_text())
    reprs: dict[tuple[str, int], dict] = {}
    with (base / "representatives.csv").open() as f:
        for r in csv.DictReader(f):
            reprs[(r["site"], int(r["azimuth"]))] = {
                "latest_timestamp": r.get("latest_timestamp", ""),
                "n_sequences": int(r.get("n_sequences", 0) or 0),
            }

    # Sequence inventory: (site, azimuth) → list of registry entries
    inventory: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for s in json.loads(args.registry.read_text())["sequences"]:
        m = CAM_RE.match(s["camera"])
        if m:
            inventory[(m.group(1), int(m.group(2)))].append(s)

    # Per-site union-find on accepted pairs
    sites_pairs: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
    for pid, dec in decisions.items():
        if dec["status"] != "accepted":
            continue
        site, a, b = pid.split("|")
        sites_pairs[site].append(
            (int(a), int(b), int(dec["kept_azimuth"]), int(dec["n_inliers"]))
        )

    rows: list[dict] = []
    cluster_counter = 0
    for site in sorted(sites_pairs):
        uf = UnionFind()
        for a, b, _, _ in sites_pairs[site]:
            uf.union(a, b)
        comps: dict[int, set[int]] = defaultdict(set)
        for a, b, _, _ in sites_pairs[site]:
            comps[uf.find(a)].update([a, b])
        for root, members in comps.items():
            cluster_id = f"c{cluster_counter:03d}"
            cluster_counter += 1
            votes: Counter[int] = Counter()
            evidence: list[tuple[int, int, int, int]] = []
            for a, b, kept, ninl in sites_pairs[site]:
                if uf.find(a) == root:
                    votes[kept] += 1
                    evidence.append((a, b, kept, ninl))
            canonical = max(
                members,
                key=lambda az: (
                    votes[az],
                    reprs.get((site, az), {}).get("latest_timestamp", ""),
                    reprs.get((site, az), {}).get("n_sequences", 0),
                    az,
                ),
            )
            for m in sorted(members):
                if m == canonical:
                    continue
                seqs = inventory.get((site, m), [])
                splits = Counter(s["split"] for s in seqs)
                rows.append(
                    {
                        "site": site,
                        "cluster_id": cluster_id,
                        "discard_azimuth": m,
                        "keep_azimuth": canonical,
                        "n_sequences": len(seqs),
                        "n_train": splits.get("train", 0),
                        "n_val": splits.get("val", 0),
                        "n_test": splits.get("test", 0),
                        "votes_keep": votes[canonical],
                        "votes_discard": votes[m],
                        "cluster_size": len(members),
                        "cluster_members": ",".join(str(x) for x in sorted(members)),
                        "evidence_pairs": " | ".join(
                            f"{a}↔{b}(keep={k},inl={n})" for a, b, k, n in sorted(evidence, key=lambda e: -e[3])
                        ),
                    }
                )

    out_path = args.out or (base / "azimuth_remap.csv")
    fieldnames = [
        "site",
        "cluster_id",
        "discard_azimuth",
        "keep_azimuth",
        "n_sequences",
        "n_train",
        "n_val",
        "n_test",
        "votes_keep",
        "votes_discard",
        "cluster_size",
        "cluster_members",
        "evidence_pairs",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {out_path} — {len(rows)} azimuths to remap")
    n_seq_total = sum(r["n_sequences"] for r in rows)
    splits_total = {
        s: sum(r[f"n_{s}"] for r in rows) for s in ("train", "val", "test")
    }
    n_clusters = len({r["cluster_id"] for r in rows})
    print(f"  clusters: {n_clusters}")
    print(f"  sequences impacted: {n_seq_total}  ({splits_total})")


if __name__ == "__main__":
    main()
