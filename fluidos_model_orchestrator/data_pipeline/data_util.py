from pathlib import Path

import pandas as pd

from fluidos_model_orchestrator.data_pipeline.augmentation.augmentation_utils import AUGMENTATION_TYPES
from fluidos_model_orchestrator.model.utils import D_TYPE  # type: ignore
from fluidos_model_orchestrator.model.utils import FLUIDOS_COL_NAMES  # type: ignore


class FLUIDOS_DATASETS:
    TEMPLATE_DATASET = "template_dataset"
    # GCT = "GCT"
    GCT = "gct"
    BITBRAINS = "bitbrains"
    MATERNA = "materna"
    MOVIE_LENS = "movieLens"


class PIPELINE_FILES:
    POD_TEMPLATE_RESOURCE_ASSIGNMENTS = "pod_template_resource_assignments.csv"
    TEMPLATE_RESOURCE_RESOURCES = "template_resource_resources.csv"
    DATASET_METADATA = "dataset_metadata.json"
    POD_2_TEMPLATE = "pod2template_resource.json"
    EVALUATION_RESULTS = "evaluation_results.json"

    TORCH_TRAIN_DATASET = "train_augmented.ptd"
    TORCH_VAL_DATASET = "val_augmented.ptd"
    CG_MODEL_CONFIG_DATASET_SPECIFIC = "cg_model_config.json"

    IMAGE_FEATURES_NAME = "image_features.json"
    TEMPLATE_RESOURCES_TO_CLASS_ID = "template_resources2id.json"


def get_target_column(augmentation_type: str) -> str:
    if augmentation_type == AUGMENTATION_TYPES.PERFORMANCE_RATING:
        return FLUIDOS_COL_NAMES.TARGET_PERFORMANCE_RESOURCES_AUGMENTATION_COL
    elif augmentation_type == AUGMENTATION_TYPES.FEEDBACK_LOOP:
        return FLUIDOS_COL_NAMES.TARGET_MOST_OPTIMAL_TEMPLATE_ID
    else:
        raise Exception(f"Couldn't find what augmentation type you were referring to with: {augmentation_type}")


def load_ml_ready_df(ml_ready_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:  # TODO ml_ready_path check everywhere for the path
    """_summary_

        Args:
            dataset_path (_type_): _description_

        Returns:
            _type_: _description_
    """

    print(f"Loading dataset: {ml_ready_path}")
    # ,pod_filename,pod_manifest,pod_cpu,pod_memory,machine_id,machine_cpu,machine_memory,machine_location,machine_throughput
    # '/workspaces/fluidos-model-orchestrator/fluidos_model_orchestrator/tmp_fluidos_dataset_ml/df/pod_machine_assignments.csv'

    if not ml_ready_path.exists():
        raise Exception(f"Path to ml_ready dataset does not exist. Path was: {ml_ready_path}")

    pods_assigment_df = pd.read_csv(
        ml_ready_path.joinpath(PIPELINE_FILES.POD_TEMPLATE_RESOURCE_ASSIGNMENTS).as_posix(),
        dtype={
            data_name: D_TYPE[data_name]['type'].__name__.replace("str", "bytes") for data_name in D_TYPE
        },
    )

    pods_assigment_df = pods_assigment_df.drop("Unnamed: 0", axis=1)
    if ml_ready_path.joinpath(PIPELINE_FILES.TEMPLATE_RESOURCE_RESOURCES).exists():
        template_resources_df = pd.read_csv(
            ml_ready_path.joinpath(PIPELINE_FILES.TEMPLATE_RESOURCE_RESOURCES).as_posix(),
            header="infer",
            dtype={
                FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_ID: "bytes",
                FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_CPU: "int64",
                FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_MEMORY: "int64",
                FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_LOCATION: "bytes",
                FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_THROUGHPUT: "float64",
            },
        )
        template_resources_df = template_resources_df.drop("Unnamed: 0", axis=1)
    else:
        template_resources_df = None
    return pods_assigment_df, template_resources_df
