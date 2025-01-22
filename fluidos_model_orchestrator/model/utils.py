from enum import Enum
import pandas as pd
from pathlib import Path
from ..common import KnownIntent


class DATA_DEPENDENCY(Enum):
    DEPENDENCY_INPUTS = "inputs"
    DEPENDENCY_TARGET = "target"


class FLUIDOS_COL_NAMES:
    POD_FILE_NAME = "pod_filename"
    TEMPLATE_RESOURCE_ID = "template_resource_id"
    POD_CPU = "pod_cpu"
    POD_GPU = "pod_gpu"
    TEMPLATE_RESOURCE_CPU = "template_resource_cpu"
    TEMPLATE_RESOURCE_GPU = "template_resource_gpu"
    POD_MEMORY = "pod_memory"
    TEMPLATE_RESOURCE_MEMORY = "template_resource_memory"
    POD_MANIFEST = "pod_manifest"
    TEMPLATE_RESOURCE_THROUGHPUT = "template_resource_throughput"
    OUTPUT = "output_speed"
    TEMPLATE_RESOURCE_LOCATION = "template_resource_location"
    POD_LOCATION = "pod_location"
    POD_THROUGHPUT = "pod_throughput"
    TEMPLATE_RESOURCE_CANDIDATE_ID = "template_resource_config_id"
    ACCEPTABLE_CANDIDATES = 'acceptable_configs'
    NON_ACCEPTABLE_CANDIDATES = "non_acceptable_configs"
    TARGET_BASIC_RESOURCE_AVAIL_AUGMENTATION_COL = "basic_resource_avail"
    TARGET_MOST_OPTIMAL_TEMPLATE_ID = "best_candidate"
    TARGET_PERFORMANCE_RESOURCES_AUGMENTATION_COL = "performance_resources"
    MSPL_INTENT = "mspl_intent"
    
  
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
                FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_GPU: "int64",
            },
        )
        template_resources_df = template_resources_df.drop("Unnamed: 0", axis=1)
    else:
        template_resources_df = None
    return pods_assigment_df, template_resources_df



class FLUIDOS_INPUT_OUTPUT_NAME:
    POD_NAME = "pod"
    TEMPLATE_RESOURCE_NAME = "template_resource"


KNOWN_INTENT_TO_POD_INTENT: dict[str, str] = {
    KnownIntent.cpu.name: FLUIDOS_COL_NAMES.POD_CPU,
    KnownIntent.memory.name: FLUIDOS_COL_NAMES.POD_MEMORY,
    KnownIntent.location.name: FLUIDOS_COL_NAMES.POD_LOCATION,
    KnownIntent.throughput.name: FLUIDOS_COL_NAMES.POD_THROUGHPUT,
    KnownIntent.mspl.name: FLUIDOS_COL_NAMES.MSPL_INTENT,
}


D_TYPE: dict[str, dict[str, type]] = {
    FLUIDOS_COL_NAMES.POD_FILE_NAME: {"type": str},
    FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_ID: {
        "type": str,
    },
    FLUIDOS_COL_NAMES.POD_MANIFEST: {"type": str},
    FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_CPU: {"type": int},
    FLUIDOS_COL_NAMES.POD_CPU: {"type": int},
    FLUIDOS_COL_NAMES.POD_MEMORY: {"type": int},
    FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_MEMORY: {"type": int},
    FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_THROUGHPUT: {"type": float},
    FLUIDOS_COL_NAMES.POD_THROUGHPUT: {"type": float},
    FLUIDOS_COL_NAMES.OUTPUT: {"type": int},
    FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_LOCATION: {"type": str},
    FLUIDOS_COL_NAMES.POD_LOCATION: {"type": str},
    FLUIDOS_COL_NAMES.TARGET_BASIC_RESOURCE_AVAIL_AUGMENTATION_COL: {"type": float},
    FLUIDOS_COL_NAMES.MSPL_INTENT: {"type": str},
}

D_UNITS = {
    FLUIDOS_COL_NAMES.POD_CPU: ["m"],
    FLUIDOS_COL_NAMES.POD_MEMORY: ["Mi"],
    FLUIDOS_COL_NAMES.POD_THROUGHPUT: ["Ks"],
    FLUIDOS_COL_NAMES.POD_LOCATION: [""],
    FLUIDOS_COL_NAMES.TARGET_BASIC_RESOURCE_AVAIL_AUGMENTATION_COL: ["%"],

    FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_CPU: ["m"],
    FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_MEMORY: ["Mi"],
    FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_THROUGHPUT: ["Ks"],
    FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_LOCATION: [""],

    FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_ID: [""],
    
    FLUIDOS_COL_NAMES.MSPL_INTENT: [""],

}


class MODEL_TYPES:
    CG = "model_cg"
    CG_LEGACY = "model_cg_legacy"
    BASIC_RANKER = "pytorch_ranker"
    TEMPLATE_MODEL = "model_template"
