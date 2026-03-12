# Processed Data

This folder contains the Pyronear datasets produced by the DVC pipeline defined in [dvc.yaml](../../dvc.yaml).

## YOLO Datasets

For training and evaluating YOLO object detection models (Ultralytics format).

- **wildfire_yolo** — intermediate: all wildfire sequences, up to 10 labeled images per sequence, split into train/val/test.
- **fp_yolo** — intermediate: false-positive sequences sampled by max detection score (10% FP for train/val, 50% for test).
- **yolo_train_val** — final: merged wildfire + FP images for training and validation (`images/train/`, `images/val/`, `labels/`, `data.yaml`).
- **yolo_test** — final: merged wildfire + FP images for evaluation (`images/test/`, `labels/`, `data.yaml`).

## Sequential Datasets

For training and evaluating temporal/sequential models. Each sequence is preserved as a folder (`images/` + `labels/`) with 50% FP balance at the sequence level.

- **sequential_train_val** — train and val sequences (`train/`, `val/`).
- **sequential_test** — test sequences (`test/`).

## Notes

- Versioning all datasets within the same repository prevents data leakage between splits across different dataset versions.
- Use Git tags + `dvc import --rev <tag>` to reproduce any past dataset version.
