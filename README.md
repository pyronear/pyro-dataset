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

## Data Pipeline

The whole repository is organized as a data pipeline that can be run to
generate the different datasets.

The Data pipeline is organized with a [dvc.yaml](./dvc.yaml) file.

### DVC stages

This section list and describes all the DVC stages that are defined in the
[dvc.yaml](./dvc.yaml) file:

#### ⛱️ Data Preparation

- __data_pyro_sdis_testset__: Turn the parquet files of the
__pyro-sdis-testset__ dataset into a regular ultralytics folder structure.

#### 🧠 Model Inference

- __predictions_wise_wolf_pyro_sdis_val__: Run inference on all images from the
pyro-sdis val split with the `wise_wolf` model.
- __predictions_legendary_field_pyro_sdis_val__: Run inference on all images
from the pyro-sdis val split with the `legendary_field` model.
- __predictions_wise_wolf_FP_2024__: Run inference on all images from the
FP_2024 dataset with the `wise_wolf` model.
- __crops_wise_wolf_pyro_sdis_val__: Generate crops from the predictions of the
`wise_wolf` model on the pyro-sdis val split.
- __crops_wise_wolf_FP_2024__: Generate crops from the predictions of the
`wise_wolf` model on the FP_2024 dataset.

#### 🚭 Filtering

- __filter_data_pyrosdis_smoke__: Keep only the fire smokes from the
`pyro-sdis` dataset - remove the background images.
- __filter_data_figlib_smoke__: Keep only the fire smokes from the
`FIGLIB_ANNOTATED_RESIZED` dataset - remove the background images.
- __filter_data_pyronear_ds_smoke__: Keep only the fire smokes from the
`pyronear-ds-03-2024` dataset - remove the background images.
- __filter_data_false_positives_FP_2024__: Keep only the false positives that
the `wise_wolf` has made on the `FP_2024` dataset.

#### 🍞 Data Splitting

- __split_data_figlib__: Split the `FIGLIB_ANNOTATED_RESIZED` dataset into
train/val/test sets.
- __split_data_false_positives_FP_2024__: Split the false postives dataset into
train/val/test sets.
- __merge_smoke_datasets__: Merge the different data sources of fire smokes and
split into the train/val/test sets.

#### 🧬 Dataset Creation

- __make_train_val_wildfire_dataset__: Make the train/val `wildfire` dataset
using the previous stages.
- __make_test_wildfire_dataset__: Make the test `wildfire` dataset using the
previous stages.

#### 🔎 Dataset Analysis

- __analyze_wildfire_dataset__: Run some analyses on the generated dataset to
check for data leakage, data distribution, and background images. Some
interactive plots are also generated and exported.

## Data

### Raw

The datasets below are the foundation of our data pipeline and are the source
of truth.

- __FIGLIB_ANNOTATED_RESIZED__: re-annotated dataset from the [Fire Ignition
images Library](https://www.hpwren.ucsd.edu/FIgLib/).
- __DS_fp__: All the collected false positives of the Pyronear System before 2024.
- __FP_2024__: All the collected false positives of the Pyronear System in 2024.
- [__pyro-sdis__](https://huggingface.co/datasets/pyronear/pyro-sdis):
Pyro-SDIS is a dataset designed for wildfire smoke detection using AI models.
It is developed in collaboration with the Fire and Rescue Services (SDIS) in
France and the dedicated volunteers of the Pyronear association. It contains
only detected fires by the Pyronear System.
- __pyronear-ds-03-2024__: Dataset of fire smokes as a mix of different public
datasets and synthetic images. It also includes temporal sequences of fire
events.
- [__pyro-sdis-testset__](https://huggingface.co/datasets/pyronear/pyro-sdis-testset):
Private dataset used for evaluating the final performances of the ML models.
- __Test_dataset_2025__: built from Test_DS by adding extra false positives.
- __Test_DS__: The initial and curated test dataset.

### Interim

All the folders located in `./data/interim/` are intermediary results needed to
build up the final datasets. They are versioned with DVC.

Many artifacts and datasets can be found here: from cropped image areas, to
filtered datasets to focus on false positives for instance.

### Processed

The final datasets are located in `./data/processed/`:

- 🔥 __wildfire__: the train/val dataset used to train our ML models. It
follows the ultralytics format.
-  🔦 __wildfire_test__: the test dataset used to evaluate the performance of
our ML models.

### Reporting

Once the datasets are generated and stored in the `./data/processed/`
directory, various reports are created to visualize the data. These reports
break down the datasets across different dimensions, allowing for a quick
assessment of whether the various data splits are logical and meaningful.

These reports live under `./data/reporting/`.

## 🧠 Models

- 🌈 __legendary_field__: yolov8s object detection model, first performant model
trained in 2019. to detect fire smoke.
- 🐺 __wise_wolf__: yolov11s object detection model, trained on `2024-04-26` using
a larger dataset.
