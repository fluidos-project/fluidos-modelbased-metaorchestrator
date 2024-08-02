from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np   # type: ignore
import pandas as pd   # type: ignore
import tensorflow as tf  # type: ignore
import tensorflow_recommenders as tfrs  # type: ignore

from fluidos_model_orchestrator.common import ModelPredictRequest
from fluidos_model_orchestrator.common import ModelPredictResponse
from fluidos_model_orchestrator.common import OrchestratorInterface
from fluidos_model_orchestrator.common import Resource
from fluidos_model_orchestrator.common import ResourceProvider
from fluidos_model_orchestrator.model.common import download_model_from_hf
from fluidos_model_orchestrator.model.common import MODEL_FILE_NAME
from fluidos_model_orchestrator.model.common import ModelInterface
from fluidos_model_orchestrator.model.utils import DATA_DEPENDENCY
from fluidos_model_orchestrator.model.utils import FLUIDOS_COL_NAMES

REPO_ID = "fluidos/basic_ranker"


class RankingModel(tf.keras.Model):
    def __init__(self, unique_user_ids: np.array, unique_movie_titles: np.array) -> None:
        super().__init__()
        embedding_dimension = 32
        self.unique_user_ids = unique_user_ids
        self.unique_movie_titles = unique_movie_titles

        # Compute embeddings for users.
        self.user_embeddings = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=self.unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(self.unique_user_ids) + 1, embedding_dimension)])

        # Compute embeddings for movies.
        self.movie_embeddings = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=self.unique_movie_titles, mask_token=None),
            tf.keras.layers.Embedding(len(self.unique_movie_titles) + 1, embedding_dimension)])

        # Compute predictions.
        self.ratings = tf.keras.Sequential([
            # Learn multiple dense layers.
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            # Make rating predictions in the final layer.
            tf.keras.layers.Dense(1)])

    def call(self, inputs: tuple[Any, Any]) -> Any:
        user_id, movie_title = inputs
        user_embedding = self.user_embeddings(user_id)
        movie_embedding = self.movie_embeddings(movie_title)
        return self.ratings(tf.concat([user_embedding, movie_embedding], axis=1))

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "unique_user_ids": self.unique_user_ids,
                "unique_movie_titles": self.unique_movie_titles,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> type[RankingModel]:
        # Note that you can also use `keras.saving.deserialize_keras_object` here

        # config["pod_manifest_text"] = tf.keras.layers.deserialize(config["pod_manifest_text"])
        config["unique_user_ids"] = tf.keras.layers.deserialize(config["unique_user_ids"])
        config["unique_movie_titles"] = tf.keras.layers.deserialize(
            config["unique_movie_titles"]
        )

        return cls(**config)

# RankingModel()((["42"], ["One Flew Over the Cuckoo's Nest (1975)"]))


class BasicRankerModel(tfrs.models.Model, ModelInterface):
    def __init__(self, unique_user_ids: np.array,
                 unique_movie_titles: np.array,
                 target_column: str):
        super().__init__()
        self.ranking_model: tf.keras.Model = RankingModel(unique_user_ids, unique_movie_titles)
        self.loss = tf.keras.losses.MeanSquaredError()
        self.metrics_k = {
            "root_mean_squared": tf.keras.metrics.RootMeanSquaredError()
        }
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(loss=self.loss, metrics=list(self.metrics_k.values()))
        self.target_column = target_column

    def get_columns_dependencies(self) -> dict[str, Any]:
        return {
            DATA_DEPENDENCY.DEPENDENCY_INPUTS.name: [FLUIDOS_COL_NAMES.POD_FILE_NAME, FLUIDOS_COL_NAMES.POD_MANIFEST],
            DATA_DEPENDENCY.DEPENDENCY_TARGET.name: self.target_column
        }

    @staticmethod
    def create_sample_request() -> ModelPredictRequest:

        #PODS should be identical
        pod_request = {FLUIDOS_COL_NAMES.POD_FILE_NAME: ['pod_1.yaml'],
                       FLUIDOS_COL_NAMES.POD_MANIFEST: ["{'apiVersion': 'v1', 'kind': 'Pod', 'metadata': {'name': 'nginx'}, 'spec': {'containers': [{'image': 'nginx:1.14.2', 'name': 'nginx', 'ports': [{'containerPort': 80}], 'resources': {'requests': {'cpu': '225m', 'memory': '686Mi'}"]}

        return ModelPredictRequest(
            id="dummyId",
            namespace="namespace",
            pod_request=pod_request,
            intents=[],
            container_image_embeddings=[]
        )

    def call(self, features: dict[str, tf.Tensor]) -> tf.Tensor:
        return self.ranking_model((features[FLUIDOS_COL_NAMES.POD_FILE_NAME], features[FLUIDOS_COL_NAMES.POD_MANIFEST]))

    def compute_loss(self, features: dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        labels = features.pop(self.target_column)
        rating_predictions = self(features)
        # The task computes the loss and the metrics.
        return self.task(labels=labels, predictions=rating_predictions)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                # "pod_manifest_text": self.pod_manifest_text,
                "unique_pod_ids": self.unique_pod_ids,
                "embedding_output_dimension": self.embedding_output_dimension,
                "max_tokens": self.max_tokens,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Any:
        # Note that you can also use `keras.saving.deserialize_keras_object` here

        # config["pod_manifest_text"] = tf.keras.layers.deserialize(config["pod_manifest_text"])
        config["unique_pod_ids"] = tf.keras.layers.deserialize(config["unique_pod_ids"])
        config["embedding_output_dimension"] = tf.keras.layers.deserialize(
            config["embedding_output_dimension"]
        )
        config["max_tokens"] = tf.keras.layers.deserialize(config["max_tokens"])

        return cls(**config)


class Orchestrator(OrchestratorInterface):

    def __init__(self) -> None:

        self.model = self.load()

        unique_template_id_resources = Path('fluidos_model_orchestrator/model/model_basic_ranker/gct_template_id_unique_resources.csv')
        if unique_template_id_resources.exists():
            self.template_id_resources_df = pd.read_csv(unique_template_id_resources)

    def load(self) -> Any:
        unzipped_model_path = download_model_from_hf(REPO_ID, MODEL_FILE_NAME)
        return tf.saved_model.load(unzipped_model_path.as_posix())

    def predict(self, request: ModelPredictRequest, architecture: str = "amd64") -> ModelPredictResponse:

        input_dict = request.pod_request

        input_df = pd.DataFrame({
            FLUIDOS_COL_NAMES.POD_FILE_NAME: input_dict[FLUIDOS_COL_NAMES.POD_FILE_NAME],
            FLUIDOS_COL_NAMES.POD_MANIFEST: input_dict[FLUIDOS_COL_NAMES.POD_MANIFEST]})

        input_df_extended = pd.DataFrame()
        print(input_df_extended.shape)
        for template_id_resources_id in range(self.template_id_resources_df.shape[0]):
            tmp_df = input_df
            tmp_df[FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_ID] = self.template_id_resources_df[FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_ID].iloc[template_id_resources_id]
            tmp_df[FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_CPU] = self.template_id_resources_df[FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_CPU].iloc[template_id_resources_id]
            tmp_df[FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_MEMORY] = self.template_id_resources_df[FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_MEMORY].iloc[template_id_resources_id]
            input_df_extended = pd.concat([input_df_extended, tmp_df], axis=0, ignore_index=True)
            print(input_df_extended.shape)

        input_tmp = {FLUIDOS_COL_NAMES.POD_FILE_NAME: list(input_df_extended[FLUIDOS_COL_NAMES.POD_FILE_NAME]),
                     FLUIDOS_COL_NAMES.POD_MANIFEST: list(input_df_extended[FLUIDOS_COL_NAMES.POD_MANIFEST]),
                     FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_ID: list(input_df_extended[FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_ID])}
        prediction = self.model(input_tmp).numpy()

        top_prediction = np.argmax(prediction)

        selected_resource = Resource(
            id=request.id,
            region='dummyLocation',
            cpu=input_df_extended[FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_CPU][top_prediction],
            memory=input_df_extended[FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_MEMORY][top_prediction])
        # resource = self._retrieve_template_resources(request.id, input_df_extended[FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_ID][top_prediction])

        return ModelPredictResponse(
            request.id,
            resource_profile=selected_resource)

    def rank_resource(self, providers: list[ResourceProvider], prediction: ModelPredictResponse) -> list[ResourceProvider]:
        return providers
