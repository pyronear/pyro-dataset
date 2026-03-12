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

### Bounding Box Formats

YOLO uses `xywhn` (center_x, center_y, width, height, normalized). Internally we also use `xyxyn` (x1, y1, x2, y2, normalized). Conversion utilities are in `src/pyro_dataset/yolo/utils.py`.

## Branch Policy

Direct commits to `main` are blocked by pre-commit hook. Always work on a feature branch.
