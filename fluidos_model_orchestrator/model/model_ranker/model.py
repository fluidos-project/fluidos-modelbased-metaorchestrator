from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from fluidos_model_orchestrator.common import ModelPredictRequest
from fluidos_model_orchestrator.common import ModelPredictResponse
from fluidos_model_orchestrator.common import OrchestratorInterface
from fluidos_model_orchestrator.common import Resource
from fluidos_model_orchestrator.common import ResourceProvider
from fluidos_model_orchestrator.model.common import ModelInterface
from fluidos_model_orchestrator.model.utils import D_UNITS
from fluidos_model_orchestrator.model.utils import DATA_DEPENDENCY
from fluidos_model_orchestrator.model.utils import FLUIDOS_COL_NAMES

REPO_ID = "fluidos/basic_ranker"


class BasicRankerModel(nn.Module, ModelInterface):
    def __init__(self, unique_pod_ids: np.array, unique_pod_location_ids: np.array, unique_template_ids: np.array, unique_template_location_ids: np.array, target_column: str):
        super().__init__()
        embedding_dimension = 32
        self.unique_pod_ids = unique_pod_ids
        self.unique_template_ids = unique_template_ids
        self.unique_pod_location_ids = unique_pod_location_ids
        self.unique_template_location_ids = unique_template_location_ids

        self.pod_id_lookup = nn.Embedding(len(self.unique_pod_ids) + 1, embedding_dimension)
        self.template_id_lookup = nn.Embedding(len(self.unique_template_ids) + 1, embedding_dimension)
        self.pod_location_lookup = nn.Embedding(len(self.unique_pod_location_ids) + 1, embedding_dimension)
        self.template_location_lookup = nn.Embedding(len(self.unique_template_location_ids) + 1, embedding_dimension)

        # pod manifest
        # Adding a linear transformation for the pod_manifest embedding
        self.pod_manifest_transform = nn.Linear(512, embedding_dimension)  # Assuming pod_manifest embeddings are 512-dimensional

        self.pod_cpu_feature_transform = nn.Linear(1, embedding_dimension)

        # Dense layers for rating prediction
        self.total_embeddings_to_concat = 9
        self.ratings = nn.Sequential(
            nn.Linear(embedding_dimension * self.total_embeddings_to_concat, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.integer_feature_transform = nn.Linear(1, embedding_dimension)  # Transform to match embedding dimension

        # self.ranking_model = RankingModel(unique_user_ids, unique_movie_titles)
        self.loss_fn = nn.MSELoss()

        #Not used anymorej
        self.metrics = {
            "root_mean_squared": nn.MSELoss()
        }
        self.target_column = target_column

        # self.task = CustomRanking(self.loss_fn, metrics=list(self.metrics.values()))
        # metrics_arg = list(self.metrics.values())
        # self.metrics = metrics_arg if metrics_arg is not None else {}

    def forward(self, inputs: Any) -> torch.Tensor:
        # user_id = features[FLUIDOS_COL_NAMES.POD_FILE_NAME]
        # movie_title = features[FLUIDOS_COL_NAMES.POD_MANIFEST]
        # user_id, movie_title = inputs

        pod_id, pod_cpu, pod_mem, pod_location, pod_manifest, template_resource_id, template_cpu, template_mem, template_location = inputs

        pod_id_embedding = self.pod_id_lookup(pod_id)
        pod_location_embedding = self.pod_location_lookup(pod_location)
        template_location_embedding = self.template_location_lookup(template_location)

        # Transform the pod_manifest embeddings to the target dimension
        pod_manifest_embedding = self.pod_manifest_transform(pod_manifest)

        #TODO shouldn't that be moved to the preprocessing? maybe not since this can be done on the fly with input values only
        def min_max_normalize(tensor: Any, eps=1e-8) -> Any:
            min_val = tensor.min(dim=0, keepdim=True).values
            max_val = tensor.max(dim=0, keepdim=True).values
            return (tensor - min_val) / (max_val - min_val + eps)

        # Transform the integer feature to match the embedding dimension
        pod_cpu = pod_cpu.view(-1, 1)  # Ensure it's the right shape (batch_size, 1)
        pod_cpu_normalized = min_max_normalize(pod_cpu)
        pod_cpu_embedding = self.integer_feature_transform(pod_cpu_normalized.float())

        pod_mem = pod_mem.view(-1, 1)  # Ensure it's the right shape (batch_size, 1)
        pod_mem_normalized = min_max_normalize(pod_mem)
        pod_mem_embedding = self.integer_feature_transform(pod_mem_normalized.float())

        template_resource_id_embedding = self.template_id_lookup(template_resource_id)

        template_cpu = template_cpu.view(-1, 1)  # Ensure it's the right shape (batch_size, 1)
        template_cpu_normalized = min_max_normalize(template_cpu)
        template_cpu_embedding = self.integer_feature_transform(template_cpu_normalized.float())

        template_mem = template_mem.view(-1, 1)  # Ensure it's the right shape (batch_size, 1)
        template_mem_normalized = min_max_normalize(template_mem)
        template_mem_embedding = self.integer_feature_transform(template_mem_normalized.float())

        # Concatenate all embeddings
        concatenated_embeddings = [pod_id_embedding, pod_cpu_embedding, pod_mem_embedding, pod_location_embedding, pod_manifest_embedding, template_resource_id_embedding, template_cpu_embedding, template_mem_embedding, template_location_embedding]
        assert len(concatenated_embeddings) == self.total_embeddings_to_concat

        combined_embeddings = torch.cat(concatenated_embeddings, dim=1)
        # combined_embeddings = torch.cat([pod_id_embedding, pod_manifest_embedding, template_resource_id_embedding], dim=1)

        # return self.ratings(torch.cat([user_embedding, movie_embedding], dim=1))
        ret = self.ratings(combined_embeddings)

        return ret

        # return self.ranking_model((features[FLUIDOS_COL_NAMES.POD_FILE_NAME], features[FLUIDOS_COL_NAMES.POD_MANIFEST]))

    # def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
    #     user_id = features[FLUIDOS_COL_NAMES.POD_FILE_NAME]
    #     movie_title = features[FLUIDOS_COL_NAMES.POD_MANIFEST]
    #     # user_id, movie_title = inputs
    #     user_embedding = self.user_lookup(user_id)
    #     movie_embedding = self.movie_lookup(movie_title)
    #     return self.ratings(torch.cat([user_embedding, movie_embedding], dim=1))
    #     # return self.ranking_model((features[FLUIDOS_COL_NAMES.POD_FILE_NAME], features[FLUIDOS_COL_NAMES.POD_MANIFEST]))

    @staticmethod
    def load_from_hugging_face(model_name: str = None) -> Any:
        from huggingface_hub import hf_hub_download

        model_to_load = model_name if model_name else "model.pt"
        # Download the model file from Hugging Face
        downloaded_model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=model_to_load,
            repo_type="model"
        )
        # Load the model from the downloaded path
        return torch.load(downloaded_model_path)

    @staticmethod
    def prepare_model_input(input_dict):
        from fluidos_model_orchestrator.model.model_ranker.pt_dataset import get_tensor_input
        from fluidos_model_orchestrator.model.model_ranker.pt_dataset import RankerDataset
        input_df = pd.DataFrame(input_dict)
        dataset = RankerDataset(input_df)

        input_tensor_sample_batch = get_tensor_input(input_df, dataset)
        return input_tensor_sample_batch

    def compute_loss(self, features: dict[str, torch.Tensor], training: bool = True) -> torch.Tensor:
        labels = features.pop(self.target_column)
        rating_predictions = self.forward(features)
        return self.loss_fn(rating_predictions, labels)

    def get_columns_dependencies(self) -> dict[str, Any]:
        return {
            DATA_DEPENDENCY.DEPENDENCY_INPUTS.name: [FLUIDOS_COL_NAMES.POD_FILE_NAME, FLUIDOS_COL_NAMES.POD_MANIFEST],
            DATA_DEPENDENCY.DEPENDENCY_TARGET.name: self.target_column
        }


class Orchestrator(OrchestratorInterface):

    def __init__(self) -> None:

        self.model = self.load()

        unique_template_id_resources = Path('fluidos_model_orchestrator/model/model_ranker/gct_template_id_unique_resources.csv')
        if unique_template_id_resources.exists():
            self.template_id_resources_df = pd.read_csv(unique_template_id_resources)
            pass

    def load(self) -> Any:
        return BasicRankerModel.load_from_hugging_face(model_name="model_2025_01_16_dataset_full.pt")

    def create_sample_request() -> Any:
        pod_info_path = Path('fluidos_model_orchestrator/model/model_ranker/sampleOrchestratorQueryInputs.json')

        # Load the dictionary from JSON
        with open(pod_info_path.as_posix()) as file:
            pod_info_dict = json.load(file)

        return ModelPredictRequest(
            id="dummyId",
            namespace="namespace",
            pod_request=pod_info_dict,
            intents=[],
            container_image_embeddings=[]
        )

    def predict(self, request: ModelPredictRequest, architecture: str = "amd64") -> ModelPredictResponse:
        pod_info_dict = request.pod_request

        input_df = pd.DataFrame(pod_info_dict)

        # Create extended input dataframe by duplicating input_df rows
        input_df_extended = pd.concat([input_df] * len(self.template_id_resources_df), ignore_index=True)

        # Add template resources columns
        for column in self.template_id_resources_df.columns:
            input_df_extended[column] = self.template_id_resources_df[column].values

        input_dict = input_df_extended.to_dict('list')
        # input_df = pd.DataFrame(input_dict)
        # dataset = RankerDataset(input_df)

        # input_tensor_sample_batch = get_tensor_input(input_df, dataset)

        # tensor_input_dict = get_rand_sample_input(input_df, dataset)
        # self.loaded_model
        # rating_output = self.model(input_tensor_sample_batch)

        input_tensor_sample_batch = BasicRankerModel.prepare_model_input(input_dict)
        rating_output = self.model(input_tensor_sample_batch)
        # Get the index of the highest rating
        best_match_index = torch.argmax(rating_output).item()

        return ModelPredictResponse(
            request.id,
            resource_profile=Resource(
                id=request.id,
                region=input_df_extended.iloc[best_match_index][FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_LOCATION],
                cpu=f"{input_df_extended.iloc[best_match_index][FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_CPU]}{D_UNITS[FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_CPU][0]}",
                memory=f"{input_df_extended.iloc[best_match_index][FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_MEMORY]}{D_UNITS[FLUIDOS_COL_NAMES.TEMPLATE_RESOURCE_MEMORY][0]}",
                architecture=architecture)
        )

    def rank_resource(self, providers: list[ResourceProvider], prediction: ModelPredictResponse, request: ModelPredictRequest) -> list[ResourceProvider]:
        return providers
