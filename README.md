# Pyro Dataset

This repository contains all the code and data necessary to build the wildfire
dataset. This dataset is then used to train our ML models.

## Setup

### Python dependencies

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

### Data dependencies

Pull the data with `dvc`:

```sh
dvc pull
```

__Note__: One needs to configure their dvc remote and get access to our remote
data storage. Please ask somebody from the team to give you access.

Run the pipeline to build the dataset:

```sh
dvc repro
```
