import json
import os
import pickle  # nosec
import time
from pathlib import Path
from pprint import pprint
from typing import Any

import matplotlib  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import numpy.typing as npt  # type: ignore
import pandas as pd

from fluidos_model_orchestrator.data_pipeline.data_util import FLUIDOS_COL_NAMES
from fluidos_model_orchestrator.data_pipeline.data_util import PIPELINE_FILES
from fluidos_model_orchestrator.model.common import MODEL_FILE_NAME

matplotlib.use('Agg')


class BaseModelTrainer:
    def __init__(
        self,
        model_name: str,
        dataset_path: Path,
        output_dir: Path,
        epochs: int,
        dataset_max_size: int,
        validation_freq: int,
        learning_rate: float,
        target_column: str,
        repo_id: str
    ) -> None:
        self.model_name = model_name
        self.model_dir = output_dir.joinpath("model")
        self.weights_path = Path(os.path.join(self.model_dir, "model_main_dir/model_weights"))
        self.checkpoint_dir = self.model_dir.joinpath("training_checkpoints")
        self.checkpoint_path = self.checkpoint_dir.joinpath("cp-{epoch:04d}.ckpt")
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.evaluation_dir = output_dir.joinpath("evaluation")
        os.makedirs(self.model_dir.as_posix(), exist_ok=True)
        os.makedirs(self.evaluation_dir.as_posix(), exist_ok=True)
        self.epochs = epochs
        self.dataset_max_size = dataset_max_size
        self.validation_freq = validation_freq
        self.learning_rate = learning_rate
        self.loaded_model = None
        self.callbacks: list[Any] = []
        self.target_column = target_column
        self.model_file_name = MODEL_FILE_NAME
        self.repo_id = repo_id

    def _save_sample_queries(self) -> None:
        df_query_inputs = self.pods_assigment_df[[FLUIDOS_COL_NAMES.POD_FILE_NAME,
                                                  FLUIDOS_COL_NAMES.POD_MANIFEST,
                                                  FLUIDOS_COL_NAMES.POD_CPU,
                                                  FLUIDOS_COL_NAMES.POD_MEMORY,
                                                  FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_ID,
                                                  FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_CPU,
                                                  FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_MEMORY]].head(100)
        dict_values = df_query_inputs.to_dict('list')
        # paired_items =  list(zip(dict_values[FLUIDOS_DATA_NAMES.POD_FILE_NAME], dict_values[FLUIDOS_DATA_NAMES.POD_MANIFEST], dict_values[FLUIDOS_DATA_NAMES.TEMPLATE_RESOURCE_ID]))
        # combined_dict = [{FLUIDOS_DATA_NAMES.POD_FILE_NAME: value1, FLUIDOS_DATA_NAMES.POD_MANIFEST:value2, FLUIDOS_DATA_NAMES.TEMPLATE_RESOURCE_ID:value3} for (value1, value2, value3) in paired_items]
        with open(self.output_dir.joinpath('sampleQueryInputs.pickle'), 'wb') as file:
            pickle.dump(dict_values, file)

    def _load_sample_queries(self) -> dict[str, list[Any]]:
        with open(self.output_dir.joinpath('sampleQueryInputs.pickle'), 'rb') as file:
            queries = pickle.load(file)  # nosec
            return queries

    def prepare_dataset(self, pods_assigment_df: pd.DataFrame, machine_df: pd.DataFrame, test_mode: bool = False, training_size_ratio: float = 0.80,
                        # model_tag: str = "model_source", dataset_type: str = "train") -> tuple[Any, Any]:
                        model_tag: str = "model_source", dataset_type: str = "train") -> Any:
        if test_mode:
            print("----------- TEST_MODE ")
            self.dataset_max_size = 10_000
            self.epochs = 3

        self.pods_assigment_df = pods_assigment_df
        self.template_resource_df = machine_df
        if self.dataset_max_size > 0:
            self.pods_assigment_df = self.pods_assigment_df[: self.dataset_max_size]

        self._save_sample_queries()
        if dataset_type == "train":
            self.cached_train, _ = self._prepare_dataset_model_specific(test_mode, training_size_ratio, model_tag, dataset_type)
            return self.cached_train
        else:
            self.cached_test, _ = self._prepare_dataset_model_specific(test_mode, training_size_ratio, model_tag, dataset_type)
            return self.cached_test

    def _prepare_dataset_model_specific(self, test_mode: bool = False, training_size_ratio: float = 0.8,
                                        model_tag: str = "model_source", dataset_type: str = "train") -> tuple[Any, Any]:
        raise NotImplementedError(" method is not implemented")

    def check_model_dataset_depencies(self) -> None:
        dependencies = self.model.get_columns_dependencies()
        d_list = []
        for keys in dependencies.keys():
            if isinstance(dependencies[keys], list):
                d_list.extend(dependencies[keys])
            else:
                d_list.append(dependencies[keys])

        for column_dependency in d_list:
            if column_dependency not in list(self.pods_assigment_df.columns):
                raise Exception(f"The dataset is missing column: {column_dependency} needed to train model: {self.model_name}")

    def build_model(self, ) -> Any:
        self.model = self._build_model_specific()
        return self.model

    def _build_model_specific(self) -> Any:
        raise NotImplementedError(" method is not implemented")

    def train_model(self) -> dict[str, list[float]]:
        training_history = self._train_model_specific()

        with open(
            self.evaluation_dir.joinpath("training_history.json"), "w"
        ) as file:
            json.dump(training_history, file, indent=4)
        return training_history

    def validate_model(self) -> Any:
        return self._validate_model_specific()

    def _validate_model_specific(self) -> Any:
        raise NotImplementedError(" method is not implemented")

    def _train_model_specific(self) -> dict[str, list[float]]:
        raise NotImplementedError(" method is not implemented")

    def save_model(self, saving_path: Path | None = None, upload_to_hugging_face: bool = False) -> Path | str | None:

        if saving_path is not None:
            self.weights_path = saving_path
        os.makedirs(self.weights_path, exist_ok=True)

        data = self._export_hyper_params()
        with open(
            self.output_dir.joinpath("hyper_params.json").as_posix(), "w"
        ) as file:
            json.dump(data, file, indent=4)

        return self._save_model_specific(upload_to_hugging_face)

    def _save_model_specific(self, upload_to_hugging_face: bool = False) -> Path | str | None:
        raise NotImplementedError(" method is not implemented")

    def _upload_to_hugging_face_model_specific(self) -> None:
        raise NotImplementedError(" method is not implemented")

    def load_model(self, path: Path | str | None = None, load_from_hugging_face: bool = False, load_from_checkpoint: bool = False) -> Any:
        if path is None:
            if load_from_checkpoint is True:
                path = self.checkpoint_path
            else:
                path = self.weights_path
        if path is None:
            raise Exception("No weights are available to load this model")
        # self.loaded_model = self._load_model_specific(model_path, load_from_hugging_face)
        # return self.loaded_model
        return self._load_model_specific(path, load_from_hugging_face, load_from_checkpoint)

    def _load_model_specific(self, model_path: Path | str, load_from_hugging_face: bool = False, load_from_checkpoint: bool = False) -> Any:
        raise NotImplementedError(" method is not implemented")

    def prepare_directories(self) -> None:
        print("Preparing directories")
        os.makedirs(self.output_dir.as_posix(), exist_ok=True)
        os.makedirs(self.model_dir.as_posix(), exist_ok=True)

        # data = self._export_hyper_params()
        # with open(
        #     self.output_dir.joinpath("hyper_params.json").as_posix(), "w"
        # ) as file:
        #     json.dump(data, file, indent=4)

    def _export_hyper_params(self) -> dict[str, Any]:
        model_hyper_params = self._export_hyper_params_model_specific()
        model_hyper_params["model_name"] = self.model_name
        model_hyper_params["dataset_path"] = self.dataset_path.as_posix()
        model_hyper_params["epochs"] = self.epochs
        model_hyper_params["dataset_max_size"] = self.dataset_max_size
        model_hyper_params["validation_freq"] = self.validation_freq
        model_hyper_params["learning_rate"] = self.learning_rate
        return model_hyper_params

    def _export_hyper_params_model_specific(self) -> dict[str, Any]:
        raise NotImplementedError(" method is not implemented")

    def evaluate(self) -> dict[str, float]:
        evaluation_results = self._evaluate_model_specific()
        os.makedirs(self.output_dir.as_posix(), exist_ok=True)
        with open(
            self.output_dir.joinpath(PIPELINE_FILES.EVALUATION_RESULTS).as_posix(), "w"
        ) as file:
            json.dump(evaluation_results, file, indent=4)

        pprint(f"Evaluation results: {evaluation_results}")
        return evaluation_results

    def _evaluate_model_specific(self) -> dict[str, float]:
        raise NotImplementedError(" method is not implemented")

    def _plot_history(self, training_history: dict[str, list[float]], data_to_plot_list: list[str], title: str, y_label: str, x_label: str) -> Path:
        for data in data_to_plot_list:
            plt.plot(training_history[data])

        plt.title(title)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(data_to_plot_list, loc='upper left')
        figure_path = self.evaluation_dir.joinpath(f'{title.replace(" ", "-")}.png')
        plt.savefig(figure_path)
        return figure_path

    def run_training_pipeline(
        self, pods_assigment_df: pd.DataFrame, machine_df: pd.DataFrame, epochs: int = 6, test_mode: bool = False, upload_to_hugging_face: bool = False
    ) -> None:
        start = time.time()
        print("====== PREPARING DATASET FOR MODEL CONSUMPTION ======")
        cached_train = self.prepare_dataset(pods_assigment_df[:int(len(pods_assigment_df) * 0.7)], machine_df, test_mode, dataset_type="train")
        cached_test = self.prepare_dataset(pods_assigment_df[int(len(pods_assigment_df) * 0.7):], machine_df, test_mode, dataset_type="val")
        self.cached_train = cached_train
        self.cached_test = cached_test
        print("====== BUILDING MODEL ======")
        self.model = self.build_model()
        print("====== CHECKING MODEL DATASET DEPENDENCIES ======")
        self.check_model_dataset_depencies()
        print("====== TRAINING MODEL ======")
        self.train_model()
        print("====== SAVING MODEL ======")
        self.save_model(upload_to_hugging_face=upload_to_hugging_face)
        print("====== EVALUATING MODEL ======")
        self.evaluate()
        total_time = time.time() - start
        print(
            f"Total Training Pipeline Time: {round(total_time / 3600)}h:{round(total_time / 60)}min"
        )

    def predict(self, input_dict: dict[str, Any] | None = None, load_from_hugging_face: bool = False) -> npt.NDArray[np.float32]:
        if self.loaded_model is None:  # type: ignore
            self.loaded_model = self.load_model(load_from_hugging_face=load_from_hugging_face)  # type: ignore
        if input_dict is None:
            input_dict = self._load_sample_queries()
        return self._predict_model_specific(input_dict)

    # TODO this method should ultimately be implemented fully in this class not the sub classes
    def _predict_model_specific(self, input_dict: dict[str, list[Any]]) -> npt.NDArray[np.float32]:
        raise NotImplementedError(" method is not implemented")
