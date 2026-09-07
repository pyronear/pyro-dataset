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

### Importing a pyro-annotator export

Human-annotated alerts from
[pyro-annotator](https://github.com/pyronear/pyro-annotator) follow a different
path: their splits come from the recurring-object ledger rather than per-camera
stratification, so `add_data.py` is called with `--splits-from`. Every sequence
of one artefact has to land in the same split, or the model meets the same
object on both sides of the evaluation.

The export itself is produced by `make export-alerts` in the pyro-annotator
repository, not here.

Follow [docs/runbooks/annotator-import.md](docs/runbooks/annotator-import.md) —
it is the source of truth for the whole flow, from the export to the release
tag.

---

## Data Pipeline

### Stages

`dvc.yaml` is authoritative; this list is a map of what each stage is for.

- **build_wf_yolo_dataset**: Samples up to 10 labeled images per wildfire sequence and copies them into a YOLO-format dataset (`data/processed/wildfire_yolo/`), split into train/val/test according to `registry.json`.
- **compute_fp_embeddings**: DINOv2 embeddings of the false-positive sequences, used to cluster them when selecting negatives. Refresh it after any ingest that adds FP sequences, or the new ones stay invisible to the selection.
- **build_fp_yolo_dataset**: Samples false positive images using round-robin by max detection score. Quotas: 10% FP for train/val, 50% FP for test. Outputs to `data/processed/fp_yolo/`.
- **merge_yolo_dataset**: Merges wildfire and FP images into two final datasets — `data/processed/yolo_train_val/` and `data/processed/yolo_test/`.
- **build_sequential_dataset**: Builds the temporal datasets — `data/processed/sequential_train_val/` and `data/processed/sequential_test/` — at 50% FP in every split. The test half is copied verbatim from `data/raw/sequential_test_lock.json` and the stage errors rather than re-selecting when that lockfile is stale.
- **test_data_leakage**: Runs `tests/test_data_leakage.py` against the real data: split leakage, recurring-object pinning, and lockfile-vs-quota consistency.
- **build_toy_dataset**: A 5% sample of both datasets, for smoke-testing a training loop without moving 9 GB.
- **visualize_yolo_train_val** / **visualize_yolo_test**: Render annotated samples into `data/reporting/viz/`.
- **materialise_annotator_sequences**: Copies the alerts selected by `import_plan.json` out of the annotator export into staging folders. A full `dvc repro` therefore needs the export on disk — `dvc pull data/raw/pyro-annotator/export`, or expect this one stage to fail.

---

## Dataset Versioning

All dataset versions are tracked via Git tags. Each tag points to a specific `dvc.lock`, which records the exact content hashes of every output.

### Release a new version

Work on a feature branch: direct commits to `main` are blocked, and a release
is a tag on the merged commit.

```sh
# 1. Refresh the embeddings, then grow the frozen test negatives to the new
#    quota. Any ingest that added test wildfire sequences opens slots, and
#    build_sequential_dataset refuses to run against a stale lockfile.
uv run dvc repro compute_fp_embeddings
uv run python scripts/freeze_test_selection.py --dry-run   # then without it

# 2. Produce datasets
uv run dvc repro

# 3. Commit every piece of state the build depends on, together
git add dvc.lock \
        data/raw/wildfire.dvc data/raw/fp.dvc \
        data/raw/sequential_test_lock.json \
        data/raw/pyro-annotator/export.dvc \
        data/raw/pyro-annotator/import_plan.json \
        data/raw/pyro-annotator/recurring_objects.json
git commit
git push -u origin <branch>

# 4. Push the data, and check it actually landed — a tag whose outputs are
#    missing from the remote is a release nobody can consume. `dvc push`
#    routes each output to its own remote; verification has to ask both,
#    because the test datasets live on `awspyronear-private` and a bare
#    `dvc status --cloud` only answers for the default remote.
uv run dvc push
uv run dvc status --cloud
uv run dvc status --cloud -r awspyronear-private \
        data/processed/sequential_test data/processed/yolo_test

# 5. Tag the merged commit, once the pull request is merged
git checkout main && git pull
git tag vX.Y.Z        # `git tag` lists the last one; new data = minor bump
git push origin vX.Y.Z
```

The ledger, the plan and the test lockfile are accumulated state: they are
committed to git rather than tracked by DVC, and they only make sense together
with the registries they were computed against. Commit them in the same commit
— that is what makes a tag reproducible.

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
