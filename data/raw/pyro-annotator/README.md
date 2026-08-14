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
