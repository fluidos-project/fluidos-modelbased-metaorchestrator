from pathlib import Path

from fluidos_model_orchestrator.model.utils import FLUIDOS_COL_NAMES
from fluidos_model_orchestrator.model.utils import MODEL_TYPES
from fluidos_model_orchestrator.model_pipeline.model_cg.model_trainer import CGModelTrainer
from fluidos_model_orchestrator.model_pipeline.model_trainer import BaseModelTrainer
# from fluidos_model_orchestrator.model_pipeline.model_2t.model_trainer import TwoTowerModelTrainer as TwoTowerModelTrainer
# from fluidos_model_orchestrator.model_pipeline.model_basic_ranker.model_trainer import ModelTrainer as BasicRankerModelTrainer
# from fluidos_model_orchestrator.model_pipeline.model_fluidos_ranker.model_trainer import ModelTrainer as FluidOSRankerModelTrainer
# from fluidos_model_orchestrator.model_pipeline.model_small_tf.model_trainer import ModelTrainer as SmallTFModelTrainer


def model_target_is_valid(model_type: str, col_target: str) -> bool:
    if model_type == MODEL_TYPES.TWO_TOWER or model_type == MODEL_TYPES.SMALL_TF or model_type == MODEL_TYPES.BASIC_RANKER:
        if col_target in [FLUIDOS_COL_NAMES.TARGET_PERFORMANCE_RESOURCES_AUGMENTATION_COL]:
            return True
    elif model_type == MODEL_TYPES.CG:
        if col_target in [FLUIDOS_COL_NAMES.TARGET_MOST_OPTIMAL_TEMPLATE_ID]:
            return True
    else:
        raise ValueError(f"Can't find what model type {model_type} is referring to")
    return False


class ModelTrainerFactory:

    @staticmethod
    def create_model_trainer(
        model_type: str,
        dataset_path: Path,
        output_dir: Path,
        target_column: str,
        epochs: int = 5,
        max_pod: int = -1,
        load_from_generated: bool = False,
        model_name: str | None = None
    ) -> BaseModelTrainer:
        print("Creating model trainer")
        if model_name is not None:
            trainer_specific_output_dir = output_dir.joinpath(model_name)
        else:
            trainer_specific_output_dir = output_dir

        if model_type == MODEL_TYPES.CG:
            return CGModelTrainer(
                dataset_path=dataset_path,
                output_dir=trainer_specific_output_dir,
                target_column=target_column,
                dataset_max_size=max_pod,
                epochs=epochs,
                load_from_generated=load_from_generated
            )
        else:
            raise ValueError(f"Can't find what model type {model_type} is referring to")
