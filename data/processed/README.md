# Processed Data

This folder contains the Pyronear datasets available for training and
evaluation purposes. These datasets represent the final results of the data
pipeline, as detailed in the [dvc.yaml](../../dvc.yaml) file.

The following datasets have been generated:

- **wildfire**: An Ultralytics dataset utilized for training YOLO object detection models, encompassing both training and validation splits. This dataset can be widely distributed, enabling users to develop improved models.
- **wildfire_test**: An Ultralytics dataset designated for evaluating YOLO object detection models.
- **wildfire_temporal**: A dataset intended for training temporal models, including both training and validation splits. Similar to the wildfire dataset, it can be broadly shared to foster model enhancement.
- **wildfire_temporal_test**: A dataset meant for evaluating temporal models and the Pyronear engine.

**Notes**

- Versioning all datasets within the same repository helps prevent data leakage between training and test sets across different splits and datasets.
