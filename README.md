# Pyro Dataset

This repository contains all the code and data necessary to build the wildfire
dataset. This dataset is then used to train our ML models.

## Setup

### 🐍 Python dependencies

Install `uv` with `pipx`:

```sh
pipx install uv
```

Create a virtualenv and install the dependencies with `uv`:

```sh
uv sync
```

Activate the `uv` virutalenv:

```sh
source .venv/bin/activate
```

### 🍜 Data dependencies

Get the wildfire datasets with `dvc`:

```sh
dvc get . data/processed
```

Pull all the data with `dvc`:

```sh
dvc pull
```

__Note__: One needs to configure their dvc remote and get access to our remote
data storage. Please ask somebody from the team to give you access.

Run the pipeline to build the dataset:

```sh
dvc repro
```

## Adding Data

Before running the DVC pipeline, you can add new sequences to the raw datasets.

### 1. Pull existing raw data

```sh
dvc pull
```

### 2. Add new sequences

```sh
# Add new wildfire sequences
uv run python scripts/add_data.py --src /path/to/new/wildfire/sequences --type wildfire

# Add new false positive sequences
uv run python scripts/add_data.py --src /path/to/new/fp/sequences --type fp
```

This copies the folders into `data/raw/<type>/data/`, validates naming and structure, assigns stable train/val/test splits (80/10/10 per camera), and updates `data/raw/<type>/registry.json`.

Use `--dry-run` to preview without writing anything.

### 3. Track and push the new data

Re-add the updated folder(s) to DVC and push to remote storage:

```sh
# For wildfire
uv run dvc add data/raw/wildfire
git add data/raw/wildfire.dvc
dvc push data/raw/wildfire

# For false positives
uv run dvc add data/raw/fp
git add data/raw/fp.dvc
dvc push data/raw/fp
```

### 4. Run the pipeline

```sh
dvc repro
```

---

## Data Pipeline

### Stages

- **build_wf_yolo_dataset**: Samples up to 10 labeled images per wildfire sequence and copies them into a YOLO-format dataset (`data/processed/wildfire_yolo/`), split into train/val/test according to `registry.json`.
- **build_fp_yolo_dataset**: Samples false positive images using round-robin by max detection score. Quotas: 10% FP for train/val, 50% FP for test. Outputs to `data/processed/fp_yolo/`.
- **merge_yolo_dataset**: Merges wildfire and FP images into two final datasets — `data/processed/yolo_train_val/` and `data/processed/yolo_test/`.

---

## Dataset Versioning

All dataset versions are tracked via Git tags. Each tag points to a specific `dvc.lock`, which records the exact content hashes of every output.

### Release a new version

```sh
# 1. Produce datasets
uv run dvc repro

# 2. Push data to remote
uv run dvc push

# 3. Commit and tag
git add dvc.lock data/raw/wildfire.dvc data/raw/fp.dvc
git commit -m "dataset: release v1.0.0"
git tag v1.0.0
git push && git push --tags
```

### Use a specific version in another repo

```sh
# Import locked to a tag (reproducible, updatable)
dvc import https://github.com/pyronear/pyro-dataset data/processed/yolo_train_val --rev v1.0.0
dvc import https://github.com/pyronear/pyro-dataset data/processed/yolo_test --rev v1.0.0

# Update to latest
dvc update yolo_train_val.dvc
dvc update yolo_test.dvc
```

---

