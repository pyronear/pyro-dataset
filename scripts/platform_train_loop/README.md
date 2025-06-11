# Platfrom Train Loop

We deploy new stations or we want to reduce the amount of false positives from
the existing ones. This document explains a proven workflow designed to augment
the dataset with current bad predictions from the running model. The goal being
that retraining the model on a dataset that includes its current mistakes will
make it a more performant model that generates fewer false positives.

## Retraining the model with a new dataset

Create a new git branch named: `<username>/train-best-dataset-<dataset-version>`

Copy the generated `wildfire` into the `mlops` repository into `./data/01_raw/` - Remove the old one if necessary.
Tell dvc to track the new dataset with 

```bash
dvc commit ./data/01_raw/wildfire
```


Run all the `yolo_best` stages with the dvc command:


```bash
dvc repro --glob "*yolo_best"
```


Push the new best model with dvc if you are happy with the result:

```bash
dvc push
```

Open a Github PR and merge to main when accepted.

## Make a release

Run the script to release the new model (set the right version number and pic a `release-name`)

```bash
export GITHUB_ACCESS_TOKEN=XXX
uv run python ./scripts/release.py \
  --version v2.0.0 \
  --release-name "brave badger" \
  --github-owner earthtoolsmaker \
  --github-repo pyronear-mlops
```
