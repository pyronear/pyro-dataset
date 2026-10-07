# pyro-annotator

Everything that comes from [pyro-annotator](https://github.com/pyronear/pyro-annotator),
the annotation UI where humans classify and localize alerts.

```
data/raw/pyro-annotator/
├── export/                    # DVC-tracked — pulled from the annotation API
│   ├── manifest.jsonl         #   one JSON line per finished alert
│   └── images/{source_api}/{platform_alert_id}/{detection_id}.jpg
├── recurring_objects.json     # git-tracked — the recurring-object ledger
├── import_plan.json           # git-tracked — which alerts to import, and where
├── testbed_provenance.json    # git-tracked — the one-off May 2026 testbed import
└── README.md
```

## `export/` — the input

Produced by `make export-alerts` in the pyro-annotator repo (the target lives
in `annotation_api/Makefile`; exact invocation in
`docs/runbooks/annotator-import.md`, step 1), which walks
`GET /api/v1/export/alerts` and downloads the images. It holds only **finished**
alerts: every lane annotated, unsure lanes omitted, skipped alerts excluded.

Each alert carries its camera, organisation, azimuth, `recorded_at`, its
`temporal_model_score` from the deployed temporal model, and its objects (lanes)
with per-frame boxes. One alert may hold several lanes — several smoke plumes, or
several distinct false positives — and sibling lanes share the same frames under
different detection ids.

Every pull is a **full re-pull**, not a delta: the manifest is rewritten from
scratch and only missing images are downloaded. Re-running `dvc add` after a pull
records the new version; the `.dvc` file in git says which one a commit used.

Snapshot in this repo: 892 alerts (154 smoke, 738 false positive), 20,380
frames, 2.4 GB, pulled 2026-08-14.

## `recurring_objects.json` — the ledger

A **recurring object** is one artefact on one camera view: the road that fires
every evening, the antenna on the horizon. `scripts/plan_annotator_import.py`
recomputes them geometrically and records each one here:

```json
{"ro_00042": {"camera": "sdis-91_sdis-tigery-02", "azimuth": 285,
              "bbox_xyxyn": [0.61, 0.47, 0.62, 0.49], "split": "train",
              "seen_alerts": ["pyronear_french:50397"],
              "ingested_folders": ["sdis-91_sdis-tigery-02_285_2026-07-06T04-28-05"]}}
```

**This file is authoritative and must not be lost.** It is what keeps every
sequence of one artefact inside a single split — lose it and the next import
mints fresh ids, free to place the same object in train and val, which is exactly
the leakage it exists to prevent. That is why it lives in git rather than DVC: it
is small, diffable, and its history is worth reading.

The ids are surrogate keys, never derived from the geometry: an object's
representative bbox drifts as sightings accumulate, so a content-derived key would
rename it. `seen_alerts` and `ingested_folders` are identity lists rather than
counts because the export is a full re-pull — counting would re-count the whole
history on every import.

Commit it together with the `registry.json` change from the same import, so the
two never disagree about which split an artefact went to.

## `import_plan.json` — what to import, and where

The decisions the ledger implies, one entry per folder:

```json
{"sdis-91_sdis-tigery-02_285_2026-07-06T04-28-05":
  {"kind": "fp", "split": "train", "recurring_object": "ro_00042"}}
```

Like the ledger it accumulates across imports and lives in git, and for the same
reason: it is the record of what was decided, not something derivable from the
export alone — smoke folders carry a split that exists nowhere else. Planning
re-reads the ledger's split onto every entry, so the ledger stays authoritative
and a stale entry cannot put one artefact in two splits.

It is also what makes staging disposable: with the plan in git, materialisation
rebuilds `data/interim/pyro-annotator/sequences/` from the export at any time.

## `testbed_provenance.json` — the May 2026 testbed

A one-off import, separate from the export flow above. temporal-model used to
compare releases on its own fixed testbed: 332 pyro-annotator sequences from 8
sdis-77 cameras, 2026-05-05 to 2026-05-11, labeled smoke / fp / unknown, with
images only. These sequences are now registered in `raw/wildfire` and `raw/fp`
with `source: pyro-annotator-testbed`, so temporal-model can evaluate on
`sequential_test` alone.

- **Labels** come from the testbed only. The 15 `unknown` sequences are excluded.
- **Platform API**, metadata only: the camera azimuth, the frame timestamps, and
  the detection box used to pick the right YOLO box. Testbed ids are platform
  sequence ids, and the images are byte-identical.
- **Boxes** come from `pyronear/yolov11s` v8.2.0: per frame, the box that best
  overlaps the platform box. Smoke boxes are class 0; FP boxes are class 99 plus
  their score. They are detector output, not human boxes.
- **Splits are 40/10/50** by groups, so one source never spans two splits. Smoke
  is grouped by camera view, or by a start within 30 min on any camera. FP is
  grouped by camera view plus box IoU >= 0.3. Sequences that share an image
  are always grouped, compared by content (MD5), across smoke and FP. The
  platform opens one sequence per detected area, so concurrent sequences share
  frames, and it stores identical frames under different timestamps, which
  file names do not reveal. Result: 17/4/21 smoke and 102/25/127 FP.
- **Left out** (`skipped`): 19 FP sequences the detector no longer fires on, and
  2 smoke sequences dropped after a visual review.

The FP sequences are not `pyro-annotator`-sourced, so they are embedded and
selected like any pool FP, not pinned through the ledger.

## Using it

The step-by-step operator guide — every command, what it prints, what to check,
and how to recover when something goes wrong — is
[`docs/runbooks/annotator-import.md`](../../../docs/runbooks/annotator-import.md).
The condensed shape of an import:

```bash
uv run python scripts/plan_annotator_import.py --dry-run   # inspect the plan
uv run python scripts/plan_annotator_import.py             # write the plan + ledger
dvc repro materialise_annotator_sequences                  # write staging folders
uv run python scripts/add_data.py ... --splits-from ...    # register both halves
dvc repro compute_fp_embeddings                            # freeze needs fresh ones
uv run python scripts/freeze_test_selection.py             # grow the frozen test set
dvc repro                                                  # rebuild + leakage checks
```

Staging folders land in `data/interim/pyro-annotator/sequences/` — `wildfire/`,
`fp/` and a `splits.json` — mirroring `data/interim/pyronear-platform/sequences/`
from the platform loop. They are transient: `scripts/add_data.py` copies them into
the pools and writes the registry, after which the staging directory can be
deleted.

What makes this import different from the platform loop is summarised in
`CLAUDE.md`; the design and its rationale are in
`docs/specs/2026-08-13-annotator-export-sequential-import-design.md` and
`docs/specs/2026-08-14-annotator-test-growth-design.md`.
