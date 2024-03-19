from __future__ import annotations

import tensorflow as tf
import pandas as pd
from pathlib import Path

from ...common import (
    # Intent,
    KnownIntent,
    Resource,
    ModelInterface,
    ModelPredictRequest,
    ModelPredictResponse,
)

import logging

logger = logging.getLogger(__name__)


class TwoTowerOrchestrator(ModelInterface):
    def __init__(
        self,
        model_path: Path = Path(Path(__file__).parent, "resources"),
        model_name: str = "model_2t_v1",
        device: str = "cpu"
    ) -> None:
        model_index = str(Path(model_path, "model_index").absolute())
        logger.info(f"Retrieving model index from {model_index}")
        self.loaded_index = tf.saved_model.load(
            # str(model_path.joinpath(f"{model_name}/model_index"))
            model_index
        )

        machine_data_path = str(Path(model_path, "machine_resources.csv").absolute())

        logger.info(f"Retrieving model_data from: {machine_data_path}")

        machine_df = pd.read_csv(
            # str(model_path.joinpath(f"{model_name}/machine_resources.csv")),
            machine_data_path,
            header="infer",
            dtype={"machine_id": "bytes", "cpu": "int64", "memory": "int64"},
        )

        self.machine_df = machine_df.drop("Unnamed: 0", axis=1)

    def predict(self, data: ModelPredictRequest) -> ModelPredictResponse:
        input_data = {
            "pod_id": tf.constant([data.id]),
            "machine_id": tf.constant(["machine_id"]),
        }

        # TODO use the actual pod yaml file in the future
        input_data["pod_manifest"] = tf.constant([str(data.pod_request)])

        # TODO this should be simplified once intent names are standardised
        for intent in data.intents:
            if intent.name == KnownIntent.cpu:
                input_data["cpu"] = tf.constant([int(intent.value.replace("m", ""))])
            if intent.name == KnownIntent.memory:
                input_data["memory"] = tf.constant(
                    [int(intent.value.replace("Mi", ""))]
                )
        _, candidate_machine_ids = self.loaded_index(input_data)
        machine_id = candidate_machine_ids[0][0].numpy().decode("utf-8")
        try:
            # cpu, mem = get_machine_resources(machine_df, candidate_machine_id)
            row = self.machine_df[self.machine_df["machine_id"] == machine_id]
            if row.shape[0] == 0:
                raise Exception(f"Couldn't find data for machineId: {machine_id}")
            cpu = row["cpu"].iloc[0].item()
            memory = row["memory"].iloc[0].item()
        except Exception:
            cpu = memory = -1
        # predicted_config = {"cpu": cpu, "mem": mem, "fluidos-intent-throughput": -1}

        return ModelPredictResponse(
            data.id,
            resource_profile=Resource(
                id=data.id,
                # region="dummyRegion",
                cpu=cpu,
                memory=memory,
                # architecture="arm64",
            ),
        )
