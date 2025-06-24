# __raw__ data

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
- __pyronear-platform-annotated-sequences__: Contains annotated sequences from
the Pyronear platform, distinguishing between true positives, false positives,
and providing ground truth labels.
