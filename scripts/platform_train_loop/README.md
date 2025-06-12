# Platfrom Train Loop

We deploy new stations or we want to reduce the amount of false positives from
the existing ones. This document explains a proven workflow designed to augment
the dataset with current bad predictions from the running model. The goal being
that retraining the model on a dataset that includes its current mistakes will
make it a more performant model that generates fewer false positives.

## Incorporating new sequences into the dataset

### Using data from the Pyronear Platform

Use the python scripts `fetch_platform_sequences` and
`fetch_platform_sequence_id` to fetch full sequences with label files.

Eg. Fetch the sequences for 10 days between 2025-05-01 and 2025-05-10 for `sdis-07`:

```bash
export PLATFORM_API_ENDPOINT="https://alertapi.pyronear.org"
export PLATFORM_LOGIN=sdis-07
export PLATFORM_PASSWORD=XXX
export PLATFORM_ADMIN_LOGIN=XXX
export PLATFORM_ADMIN_PASSWORD=XXX

uv run python ./scripts/fetch_platform_sequences.py \
  --save-dir ./data/raw/pyronear-platform/sequences/sdis-07/ \
  --date-from 2025-05-01 \
  --date-end 2025-05-10
```

__Note__: Make sure to use an admin login/password as well as a regular
login/password. The admin level access is needed to fetch information about the
organizations and properly name the detection images locally.

It will automatically name and organize sequences that can be added to our
train/val/test datasets.

Open a file browser to check the sequences you downloaded and add them
following the folder structure outlined in
`data/raw/pyronear-platform-annotated-sequences/sdis-template/`

Make sure to add diversity and do not repeat the same false positives too
frequently in your selection. Also keep track of some true positives.
If you are unsure about a sequence, do not copy it over. Focus first on the
false positives that seem easy for the model to fix, avoid the low clouds for
instance.

Once you have added the new sequences, run the following script to keep track
of some metadata of the sequences - this will add to the `sequences.csv` file
the sequences that were selected.

```bash
uv run python ./scripts/platform_train_loop/copy_annotated_sequences_details.py \
  --dir-save ./data/raw/pyronear-platform-annotated-sequences \
  --dir-platform-annotated-sequences ./data/raw/pyronear-platform-annotated-sequences \
  --dir-platform-sequences ./data/raw/pyronear-platform/sequences/
```

Use dvc to commit the new data change:

```bash
dvc commit ./data/raw/pyronear-platform-annotated-sequences
```

Regenerate a dataset which includes the new sequences:

```bash
dvc repro
```

__Note__: One can check beforehand which stages will be rerun with the followin
dvc command `dvc status`

Check that the reports that are generated from the wildfire datasets look good
(HTML plots) and once you are happy with it, you can push the dvc data to the
remote:

```bash
dvc push
```

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
