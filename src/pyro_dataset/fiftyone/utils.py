"""
Helper functions to work with fiftyone.
"""

from pathlib import Path

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz
import pandas as pd
from fiftyone.core.dataset import Dataset

DIR_DATASET_FP_2024 = Path("./data/interim/FP_2024/crops/wise_wolf/raw/")
NAME_DATASET_FP_2024 = "wise_wolf_FP_2024_crops_raw"
TAGS_EXPORT_ALLOWED = [
    "antenna",
    "cliff",
    "cloud_high",
    "dark",
    "forest",
    "glare",
    "large_area",
    "light",
    "paragliding",
    "rainbow",
    "river",
    "sky",
    "track",
]
ABSOLUTE_PATH_PROJECT = Path(
    "/media/data/ssd_1/earthtoolsmaker/projects/pyronear/pyro-dataset/"
)
FILEPAH_SAVE_CSV = Path(
    "./data/interim/fiftyone/datasets/FP_2024/wise_wolf/raw/annotated_with_tags_dataset.csv"
)


def filter_by_tag(dataset: Dataset, tag: str) -> list:
    """
    Filter dataset samples by a given `tag`.
    """
    return [sample for sample in dataset if tag in sample.tags]


def export_tags(
    dataset: Dataset,
    filepath_save_csv: Path,
    tags_allowed: list[str] = TAGS_EXPORT_ALLOWED,
) -> None:
    records = []
    for sample in dataset:
        if len(sample.tags) > 0:
            record = {
                "filepath": Path(sample.filepath).relative_to(ABSOLUTE_PATH_PROJECT)
            }
            for tag in tags_allowed:
                record[f"is_{tag}"] = tag in sample.tags
            records.append(record)
    df = pd.DataFrame(records)
    filepath_save_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath_save_csv, index=False)


def load_dataset_FP_2024() -> Dataset:
    """
    Load the FP_2024 Dataset.
    """
    try:
        # Create the dataset if it does not exist
        return fo.Dataset.from_dir(
            dataset_dir=str(DIR_DATASET_FP_2024),
            dataset_type=fo.types.ImageDirectory,
            name=NAME_DATASET_FP_2024,
            persistent=True,
        )
    except Exception:
        # Load it from the database
        return fo.load_dataset(NAME_DATASET_FP_2024)


def embed_and_compute_visualizations(
    dataset: Dataset,
    model_embedder_name: str = "open-clip-torch",
    brain_key: str = "img_viz_raw_open_clip_torch",
) -> dict:
    """
    Embed all samples from the `dataset` using the `model_embedder_name`.
    Compute the visualizations under `brain_key` - that can be picked from
    fiftyone UI.

    Returns:
        dict containing the loaded model_embedder, the computed embeddings and
        the visualizations object.

    __Note__:
        Get a list of available models with `foz.list_zoo_models()`
    """
    print(f"Loading the model {model_embedder_name} from the model zoo.")
    model = foz.load_zoo_model(model_embedder_name)
    print(f"Computing embeddings with {model_embedder_name} on dataset {dataset.name}")
    embeddings = dataset.compute_embeddings(model)
    visualizations = fob.compute_visualization(
        dataset,
        embeddings=embeddings,
        brain_key=brain_key,
    )
    return {
        "model_embedder": model,
        "embeddings": embeddings,
        "visualizations": visualizations,
    }


## REPL

# dataset_crops_raw = load_dataset_FP_2024()
# filter_by_tag(dataset=dataset_crops_raw, tag="glare")
#
# result_embed = embed_and_compute_visualizations(
#     dataset=dataset_crops_raw,
#     model_embedder_name="open-clip-torch",
#     brain_key="img_viz_raw_open_clip_torch_1",
# )
# result_embed = embed_and_compute_visualizations(
#     dataset=dataset,
#     model_embedder_name="open-clip-torch",
#     brain_key="img_viz_raw_open_clip_torch_1",
# )
# result_embed2 = embed_and_compute_visualizations(
#     dataset=dataset_crops_raw,
#     model_embedder_name="dinov2-vitb14-torch",
#     # brain_key="img_viz_raw_dinov2-vitb14-torch",
#     brain_key="img_viz_raw_dino_torch",
# )
#
# export_tags(
#     dataset=dataset_crops_raw,
#     filepath_save_csv=Path(
#         "./data/interim/fiftyone/datasets/FP_2024/wise_wolf/raw2/annotated_dataset.csv"
#     ),
# )

# export_tags(
#     dataset=dataset,
#     filepath_save_csv=FILEPAH_SAVE_CSV,
# )

if __name__ == "__main__":
    # Ensures that the App processes are safely launched on Windows
    dataset = load_dataset_FP_2024()
    session = fo.launch_app(dataset)
    session.wait()
