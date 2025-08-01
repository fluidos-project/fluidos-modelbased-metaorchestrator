from enum import Enum

from fluidos_model_orchestrator.common.intent import KnownIntent


class DATA_DEPENDENCY(Enum):
    DEPENDENCY_INPUTS = "inputs"
    DEPENDENCY_TARGET = "target"


class MODEL_TYPES:
    CG = "model_cg"
    CG_75 = "model_cg_75"
    CG_LEGACY = "model_cg_legacy"
    BASIC_RANKER = "pytorch_ranker"
    TEMPLATE_MODEL = "model_template"


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
    TARGET_PERFORMANCE_RESOURCES_AUGMENTATION_COL = "performance_resources"
    TARGET_MOST_OPTIMAL_TEMPLATE_ID = "best_candidate"
    # TARGET_BASIC_RESOURCE_AVAIL_AUGMENTATION_COL = "performance_resources"
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
    # FLUIDOS_COL_NAMES.TARGET_BASIC_RESOURCE_AVAIL_AUGMENTATION_COL: {"type": float},
    FLUIDOS_COL_NAMES.TARGET_PERFORMANCE_RESOURCES_AUGMENTATION_COL: {"type": float},
    FLUIDOS_COL_NAMES.MSPL_INTENT: {"type": str},
}

D_UNITS = {
    FLUIDOS_COL_NAMES.POD_CPU: ["m", "n", ''],
    FLUIDOS_COL_NAMES.POD_MEMORY: ["Ki", "Mi", "Gi", "Ti", "Ei", "Pi", "G", "M", "K", "T", "P", "E"],
    FLUIDOS_COL_NAMES.POD_THROUGHPUT: ["Ks"],
    FLUIDOS_COL_NAMES.POD_LOCATION: [""],
    FLUIDOS_COL_NAMES.POD_GPU: [""],
    FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_GPU: [""],
    FLUIDOS_COL_NAMES.TARGET_PERFORMANCE_RESOURCES_AUGMENTATION_COL: ["%"],

    FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_CPU: ["m", "n", ''],
    FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_MEMORY: ["Ki", "Mi", "Gi", "Ti", "Ei", "Pi", "G", "M", "K", "T", "P", "E"],
    FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_THROUGHPUT: ["Ks"],
    FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_LOCATION: [""],

    FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_ID: [""],
    FLUIDOS_COL_NAMES.POD_MANIFEST: [""],

    FLUIDOS_COL_NAMES.MSPL_INTENT: [""],

}


def convert_memory_to_Ki(memory_value: str, memory_tag: str) -> int | float | str:
    memory_unit: str | None = None

    for unit in D_UNITS[memory_tag]:
        if memory_value[-(len(unit)):] == unit:
            memory_unit = unit
            break
    memory_type: type = D_TYPE[memory_tag]['type']

    if not memory_unit:
        if memory_value.isnumeric():
            return memory_type(round(float(memory_value) / 1000))
        else:
            raise ValueError(f"Incorrect value {memory_value}")
    else:
        match memory_unit:
            case "K":
                return memory_type(float(memory_value[:-len(memory_unit)]) * 1024 / 1000)
            case "Ki":
                return memory_type(memory_value[:-len(memory_unit)])
            case "M":
                return memory_type(round(float(memory_value[:-len(memory_unit)]) * 1024 * 1024 / 1000))
            case "Mi":
                return memory_type(float(memory_value[:-len(memory_unit)]) * 1000)
            case "Gi":
                return memory_type(float(memory_value[:-len(memory_unit)]) * 1000 * 1000)

            case "G":
                return memory_type(round(float(memory_value[:-len(memory_unit)]) * 1024 * 1024 * 1024 / 1000))
            case "T":
                return memory_type(round(float(memory_value[:-len(memory_unit)]) * 1024 * 1024 * 1024 * 1024 / 1000))
            case "Ti":
                return memory_type(round(float(memory_value[:-len(memory_unit)]) * 1000 * 1000 * 1000))
            case "P":
                return memory_type(round(float(memory_value[:-len(memory_unit)]) * 1024 * 1024 * 1024 * 1024 * 1024 / 1000))
            case "Pi":
                return memory_type(round(float(memory_value[:-len(memory_unit)]) * 1000 * 1000 * 1000 * 1000))
            case "E":
                return memory_type(round(float(memory_value[:-len(memory_unit)]) * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 / 1000))
            case "Ei":
                return memory_type(round(float(memory_value[:-len(memory_unit)]) * 1000 * 1000 * 1000 * 1000 * 1000))
    return memory_type(memory_value)


def convert_memory_to_Mi(memory_value: str, memory_tag: str) -> int | float | str:
    memory_unit: str | None = None

    for unit in D_UNITS[memory_tag]:
        if memory_value[-(len(unit)):] == unit:
            memory_unit = unit
            break
    memory_type: type = D_TYPE[memory_tag]['type']

    if not memory_unit:
        if memory_value.isnumeric():
            return memory_type(round(float(memory_value) / 1000 / 1000))
        else:
            raise ValueError(f"Incorrect value {memory_value}")
    else:
        match memory_unit:
            case "K":
                return memory_type(float(memory_value[:-len(memory_unit)]) * 1024 / 1000 / 1000)
            case "Ki":
                return memory_type(float(memory_value[:-len(memory_unit)]) / 1000)
            case "M":
                return memory_type(round(float(memory_value[:-len(memory_unit)]) * 1024 * 1024 / 1000))
            case "Mi":
                return memory_type(memory_value[:-len(memory_unit)])
            case "Gi":
                return memory_type(float(memory_value[:-len(memory_unit)]) * 1000)
            case "G":
                return memory_type(round(float(memory_value[:-len(memory_unit)]) * 1024 * 1024 / 1000))
            case "T":
                return memory_type(round(float(memory_value[:-len(memory_unit)]) * 1024 * 1024 * 1024 / 1000))
            case "Ti":
                return memory_type(round(float(memory_value[:-len(memory_unit)]) * 1000 * 1000))
            case "P":
                return memory_type(round(float(memory_value[:-len(memory_unit)]) * 1024 * 1024 * 1024 * 1024 / 1000))
            case "Pi":
                return memory_type(round(float(memory_value[:-len(memory_unit)]) * 1000 * 1000 * 1000))
            case "E":
                return memory_type(round(float(memory_value[:-len(memory_unit)]) * 1024 * 1024 * 1024 * 1024 * 1024 / 1000))
            case "Ei":
                return memory_type(round(float(memory_value[:-len(memory_unit)]) * 1000 * 1000 * 1000 * 1000))
            case _:
                raise
    return memory_type(memory_value)


def convert_cpu_to_m(cpu_value: str, cpu_tag: str) -> int | float | str:
    cpu_unit: str | None = None

    for unit in D_UNITS[cpu_tag]:
        if cpu_value[-(len(unit)):] == unit:
            cpu_unit = unit
            break
    cpu_type: type = D_TYPE[cpu_tag]['type']

    if not cpu_unit:
        if cpu_value.isnumeric():
            return cpu_type(int(cpu_value) * 1000)
        else:
            raise ValueError(f"Incorrect value {cpu_value}")
    match cpu_unit:
        case "m":
            return cpu_type(cpu_value[:-len(cpu_unit)])
        case "n":
            return cpu_type(float(cpu_value[:-len(cpu_unit)]) / 1000)
        case _:
            raise ValueError(f"Incorrect unit {cpu_unit}")
    return cpu_type(cpu_value)


def convert_cpu_to_n(cpu_value: str, cpu_tag: str) -> int | float | str:
    cpu_unit: str | None = None

    for unit in D_UNITS[cpu_tag]:
        if cpu_value[-(len(unit)):] == unit:
            cpu_unit = unit
            break
    cpu_type: type = D_TYPE[cpu_tag]['type']

    if not cpu_unit:
        if cpu_value.isnumeric():
            return cpu_type(int(cpu_value) * 1000 * 1000)
        else:
            raise ValueError(f"Incorrect value {cpu_value}")
    match cpu_unit:
        case "n":
            return cpu_type(cpu_value[:-len(cpu_unit)])
        case "m":
            return cpu_type(float(cpu_value[:-len(cpu_unit)]) * 1000)
        case _:
            raise ValueError(f"Incorrect unit {cpu_unit}")
    return cpu_type(cpu_value)


RESOURCE_TYPES: dict[str, dict[str, str]] = {
    MODEL_TYPES.CG: {
        FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_CPU: "m",
        FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_MEMORY: "Ki",
        FLUIDOS_COL_NAMES.POD_CPU: "m",
        FLUIDOS_COL_NAMES.POD_MEMORY: "Ki",
        FLUIDOS_COL_NAMES.POD_GPU: "",
        FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_GPU: "",
        FLUIDOS_COL_NAMES.POD_THROUGHPUT: "Ks",
        FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_THROUGHPUT: "Ks",
    },
    MODEL_TYPES.CG_75: {
        FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_CPU: "m",
        FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_MEMORY: "Ki",
        FLUIDOS_COL_NAMES.POD_CPU: "m",
        FLUIDOS_COL_NAMES.POD_MEMORY: "Ki",
        FLUIDOS_COL_NAMES.POD_GPU: "",
        FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_GPU: "",
        FLUIDOS_COL_NAMES.POD_THROUGHPUT: "Ks",
        FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_THROUGHPUT: "Ks",
    },
    MODEL_TYPES.CG_LEGACY: {
        FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_CPU: "m",
        FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_MEMORY: "Mi",
        FLUIDOS_COL_NAMES.POD_CPU: "m",
        FLUIDOS_COL_NAMES.POD_MEMORY: "Mi",
        FLUIDOS_COL_NAMES.POD_THROUGHPUT: "Ks",
        FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_THROUGHPUT: "Ks",
    },
}
