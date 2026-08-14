# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

```bash
# Install dependencies
uv sync

# Run the full DVC pipeline
dvc repro

# Run a single DVC stage
dvc repro <stage_name>

# Pull DVC-tracked data
dvc pull

# Run tests
uv run pytest tests/

# Run a single test file
uv run pytest tests/test_parsers.py

# Run linting (ruff auto-fixes imports and formatting)
uv run ruff check --fix --select I .
uv run ruff format .

# Install pre-commit hooks
uv run pre-commit install

# Run pre-commit on all files
uv run pre-commit run --all-files
```

## Architecture Overview

This is a **multi-stage DVC pipeline** that builds wildfire smoke detection datasets for training YOLO models.

### Data Flow

```
data/raw/          →   data/interim/    →   data/processed/
(DVC-tracked)          (filtered,           (final YOLO datasets)
                        split, merged)
```

The pipeline has 23 DVC stages defined in `dvc.yaml`. Key stages in order:
1. Install HuggingFace datasets to YOLO format
2. Run model inference (wise_wolf yolov11s, legendary_field yolov8s) to generate predictions
3. Filter datasets to extract smoke-only images and false positives
4. Split datasets into train/val/test
5. Merge smoke datasets and build final YOLO-format datasets (`wildfire`, `wildfire_test`)
6. Build temporal datasets from platform sequences (`wildfire_temporal`, `wildfire_temporal_test`)
7. Analyze and generate HTML reports

### Package Structure (`src/pyro_dataset/`)

- `constants.py` — `CLASS_ID_SMOKE=0`, `CLASS_SMOKE_LABEL="smoke"`, `DATE_FORMAT_OUTPUT`
- `yolo/` — YOLO bbox conversions (`xywhn2xyxyn`, `xyxyn2xywhn`) and pipeline logic
- `platform/` — Pyronear API client and utilities for fetching sequences
- `filepaths/` — Filename parsers for Pyronear camera naming conventions
- `plots/report.py` — Bokeh-based HTML report generation

### Scripts (`scripts/`)

All pipeline stages call scripts via `uv run python scripts/<name>.py`. Key scripts:
- `platform_train_loop/` — Iterative workflow for fetching new sequences from the Pyronear platform API, annotating them, and incorporating into the dataset
- `analyze_processed_dataset.py` — Data validation (leakage checks, distribution analysis)
- `release.py` — GitHub release automation with S3 upload

### Platform Train Loop Workflow

The `scripts/platform_train_loop/` subdirectory implements an iterative annotation loop:
1. Fetch sequences from Pyronear API (`fetch_platform_sequences.py`)
2. Manually annotate sequences
3. Place in `data/raw/pyronear-platform-annotated-sequences/`
4. Run selection/splitting scripts
5. `dvc repro` to regenerate datasets
6. Review reports, push data, retrain model, create GitHub release

### Annotator Export Import

Brings human-annotated alerts from pyro-annotator into the raw pools. Design:
`docs/specs/2026-08-13-annotator-export-sequential-import-design.md`.
Step-by-step operator runbook: `docs/runbooks/annotator-import.md`.

```bash
# 1. Decide what to import (writes import_plan.json + the ledger, copies nothing)
uv run python scripts/plan_annotator_import.py   # reads data/raw/pyro-annotator/export

# 2. Copy the planned alerts into staging folders (a DVC stage)
dvc repro materialise_annotator_sequences

# 3. Register both halves — splits come from the file, not per-camera assignment
uv run python scripts/add_data.py --src data/interim/pyro-annotator/sequences/wildfire \
  --type wildfire --splits-from data/interim/pyro-annotator/sequences/splits.json
uv run python scripts/add_data.py --src data/interim/pyro-annotator/sequences/fp \
  --type fp --splits-from data/interim/pyro-annotator/sequences/splits.json

# 4. Refresh embeddings, then grow the frozen test negatives to the new quota
dvc repro compute_fp_embeddings
uv run python scripts/freeze_test_selection.py
```

Run step 1 with `--dry-run` first: it reports the quota, how many recurring objects
were found and how many still fool the temporal model, without writing anything.

**The two halves split at the state boundary.** Planning carries accumulated state —
the ledger, and the plan itself — so it stays a manual command: a stage regenerates
its outputs from its deps, which would let `dvc repro` rebuild the ledger and destroy
the split pinning it exists to provide. Materialisation is a pure function of
`export/` + `import_plan.json`, so it caches and reproduces correctly.

Because it is a stage, a full `dvc repro` now needs the annotator export on disk:
run `dvc pull data/raw/pyro-annotator/export` (2.2 GB) first, or expect that one
stage to fail. It sits last in `dvc.yaml` so the rest of the pipeline still runs —
nothing downstream consumes its output, `add_data.py` does, by hand.

What differs from the platform loop:

- **Every smoke alert is imported**, and that count sets the false-positive quota,
  so the pools stay balanced. One sequence per recurring object
  (`--max-per-object` raises the lifetime cap), ranked hard-negatives-first by
  `temporal_model_score`, then by how often the artefact fires.
- **Splits come from `data/raw/pyro-annotator/recurring_objects.json`**, the recurring-object
  ledger, not from per-camera stratification: every sequence of one artefact must
  share a split, or the model meets the same object on both sides. The ledger is
  **committed to git** (52 KB, diffable) rather than tracked by DVC, because it is
  authoritative metadata — losing it means later imports re-mint objects and can
  place one artefact in a second split, which is the leakage it exists to prevent.
  `import_plan.json` sits beside it and is committed for the same reason. Commit
  both with the registry change from the same import. `--force-train <ro_id>`
  moves an artefact that is hurting in production into train, unless it already has
  sequences elsewhere.
- **Test grows append-only.** Imports assign 80/10/10, and the FP half of
  `sequential_test` is frozen in `data/raw/sequential_test_lock.json` —
  git-committed, appended to only by `scripts/freeze_test_selection.py`, and
  copied verbatim by the build, which errors on any mismatch instead of
  re-selecting. Every release's test set is a superset of the previous one,
  so models stay comparable across releases. Commit the lockfile with the
  ledger and plan from the same import. Any ingest that adds test WF
  sequences — annotator or not — grows the quota and needs a freeze before
  the next `dvc repro`, or the build errors on the stale lockfile.
  Design: `docs/specs/2026-08-14-annotator-test-growth-design.md`.
- **False-positive boxes are class 99**, not `0`: they mark where the detector
  fired, not smoke. Annotator-sourced FP sequences are never embedded — their
  identity comes from `recurring_objects.json` — and in train and val both
  builders pin them ahead of clustering: the sequential build takes every
  sequence, the YOLO build one image per recurring object. In test they reach
  `sequential_test` only through `freeze_test_selection.py`, up to the quota —
  surplus pins are deferred until new test smoke opens slots. See
  `docs/specs/2026-08-14-recurring-object-fp-identity-design.md`.
- Each folder carries a `meta.json` with every lane's full track, including boxes
  left out of `labels/`, so an object-level dataset can be derived later without
  re-exporting.

### Bounding Box Formats

YOLO uses `xywhn` (center_x, center_y, width, height, normalized). Internally we also use `xyxyn` (x1, y1, x2, y2, normalized). Conversion utilities are in `src/pyro_dataset/yolo/utils.py`.

## Branch Policy

Direct commits to `main` are blocked by pre-commit hook. Always work on a feature branch.
