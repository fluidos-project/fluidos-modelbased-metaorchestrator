from __future__ import annotations

import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Dict, Any
import pkg_resources
import ast

from sentence_transformers import SentenceTransformer

from ...common import Intent, Resource, ModelInterface, ModelPredictRequest, ModelPredictResponse

import logging

logger = logging.getLogger(__name__)


def compute_embedding_for_sentence(sentence: str, sentence_transformer: SentenceTransformer):
    embeddings = sentence_transformer.encode(sentence)
    return torch.tensor(embeddings).unsqueeze(0)


class EmbeddingAggregation(nn.Module):
    def __init__(self, aggregation_mode: str = "mean"):
        super().__init__()
        if aggregation_mode not in ['sum', "mean"]:
            raise NotImplementedError(f"mode {aggregation_mode} not implemented!")
        self.aggregation_mode = aggregation_mode

    def forward(self, x):
        if self.aggregation_mode == 'sum':
            aggregated = torch.sum(x, dim=1)  # axis
        elif self.aggregation_mode == 'mean':
            aggregated = torch.mean(x, dim=1)  # axis

        return aggregated


# https://theiconic.tech/implementing-the-youtube-recommendations-paper-in-tensorflow-part-1-d1e1299d5622
# https://storage.googleapis.com/pub-tools-public-publication-data/pdf/45530.pdf
class OrchestrationModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(OrchestrationModel, self).__init__()

        self.n_configs: int = config['num_configs']  # – number of configurations in total
        self.config_embedding_size: int = config['config_embedding_size']  # the size of  config embedding
        self.config_embedding_dropout: int = config['config_embedding_dropout']  # dropout of cnfig embedding

        self.pod_embedding_size: int = config['pod_embedding_size']  # the size of input pod embedding vector
        self.dict_size: int = config['dict_size']  # TODO remove
        self.image_embedding_size: int = config['image_embedding_size']  # TODO remove
        self.pod_embedding_dropout: int = config['pod_embedding_dropout']   # TODO remove

        self.config_embedding = nn.Sequential(
            nn.Embedding(num_embeddings=self.n_configs, embedding_dim=self.config_embedding_size),
            nn.Dropout(p=self.config_embedding_dropout))  #

        self.pod_embedding = nn.Sequential(nn.Embedding(num_embeddings=self.dict_size,
                                                        embedding_dim=self.pod_embedding_size),
                                           nn.Dropout(p=self.pod_embedding_dropout))  # TODO remove

        self.fc1_size = config['fc1_size']
        self.fc2_size = config['fc2_size']
        self.fc3_size = config['fc3_size']

        self.dropout_val1 = config['dropout1']
        self.dropout_val2 = config['dropout2']
        self.dropout_val3 = config['dropout3']

        self.embedding_aggregator = EmbeddingAggregation(aggregation_mode=config["aggregation_mode"])  # component-wise average

        self.linear1 = nn.Linear(self.pod_embedding_size + 1 + 2 * self.config_embedding_size, self.fc1_size)  # 2 - scalar input
        self.activation1 = nn.ReLU(inplace=True)
        self.batch_norm1 = nn.BatchNorm1d(self.fc1_size)
        self.dropout1 = nn.Dropout(p=self.dropout_val1)
        self.fc_layer1 = nn.Sequential(self.linear1, self.activation1, self.batch_norm1, self.dropout1)

        self.linear2 = nn.Linear(self.fc1_size, self.fc2_size)
        self.activation2 = nn.ReLU(inplace=True)
        self.batch_norm2 = nn.BatchNorm1d(self.fc2_size)
        self.dropout2 = nn.Dropout(p=self.dropout_val2)
        self.fc_layer2 = nn.Sequential(self.linear2, self.activation2, self.batch_norm2, self.dropout2)

        self.linear3 = nn.Linear(self.fc2_size, self.fc3_size)
        self.activation3 = nn.ReLU(inplace=True)
        self.batch_norm = nn.BatchNorm1d(self.fc3_size)
        self.dropout3 = nn.Dropout(p=self.dropout_val3)
        self.fc_layer3 = nn.Sequential(self.linear3, self.activation3, self.batch_norm, self.dropout3)

        self.head = torch.nn.Linear(in_features=self.fc3_size, out_features=self.n_configs)

    def forward(self, input):
        """
        Predicts relevant config id
        config = (cpu, memory, location)

        Args:
            input (List): input features list, 3 items
            input[0] (torch.Tensor): 0..385 for all-MiniLM-L6-v2 sentence transformer model,
                0:384, pod_embeddings, 384:385 pod_location (1-9 normalized by 9, 9 -->1)
            input[1] (torch.Tensor): list of relevant configuration ids
            input[2] (torch.Tensor): list of non relevant configuration ids

        Returns:
            torch.Tensor: logits (tensor with maximum index refers to predicted configuration id)
        """
        # Embedding preprocessing
        x_in = input[0]
        x_rel = input[1]
        x_non_rel = input[2]

        pod_embedding = x_in[:, :self.pod_embedding_size]
        pod_embedding = F.normalize(pod_embedding)
        # avg_pod_embedding = self.embedding_aggregator(pod_embedding) # TODO enable if few images in a single pod

        scalar_features = x_in[:, self.pod_embedding_size:self.pod_embedding_size + 1]  # pod location
        # scalar_features = F.normalize(scalar_features)

        rel_config_embedding = self.config_embedding(x_rel.type(torch.int))  # rel lists of configs
        rel_config_embedding = F.normalize(rel_config_embedding)
        rel_config_embedding = self.embedding_aggregator(rel_config_embedding)

        non_rel_config_embedding = self.config_embedding(x_non_rel.type(torch.int))  # non rel lists of configs
        non_rel_config_embedding = F.normalize(non_rel_config_embedding)
        non_rel_config_embedding = self.embedding_aggregator(non_rel_config_embedding)

        x = torch.cat((pod_embedding, rel_config_embedding, non_rel_config_embedding, scalar_features), dim=1)  # check batch mode

        x = self.fc_layer1(x)
        x = self.fc_layer2(x)
        x = self.fc_layer3(x)
        x = self.head(x)
        x = F.softmax(x, dim=1)
        return x


class Orchestrator(ModelInterface):
    embedding_model_name: str = "all-MiniLM-L6-v2"  # TODO read from metadata

    def __init__(self, device: str = "cpu") -> None:
        self.sentence_transformer = SentenceTransformer(self.embedding_model_name)
        self.device = device
        metadata_path = pkg_resources.resource_filename(__name__, "metadata.json")  # TODO replace pkg_resources with importlib
        try:
            with open(metadata_path, "r") as f:
                self.metadata: Dict[str, int] = json.load(f)
        except:  # noqa
            raise FileNotFoundError()

        self.orchestrator: OrchestrationModel = OrchestrationModel(self.metadata["training_setup"])
        model_path = pkg_resources.resource_filename(__name__, "orchestrator.pt")  # TODO replace pkg_resources with importlib
        self.orchestrator.load_state_dict(torch.load(model_path, map_location=self.device)["model_state_dict"])

    def generate_configs_feature_set(self, region: str) -> tuple[torch.Tensor, torch.Tensor]:
        relevant_configs: List[int] = []
        non_relevant_configs: List[int] = []

        # relevant configs - some of those which are in the target region ?
        # non_relevant configs - those which are not the target region

        for configuration in self.metadata["configuration2id"]:
            config_region = ast.literal_eval(configuration)["region"]
            if region != config_region:
                non_relevant_configs.append(self.metadata["configuration2id"][configuration])
            else:
                relevant_configs.append(self.metadata["configuration2id"][configuration])

        # relevant_configs.pop(random.randint(0, len(relevant_configs))) # TODO return back

        relevant_configs = torch.tensor(relevant_configs, device=self.device, dtype=torch.float32).unsqueeze(0)
        non_relevant_configs = torch.tensor(non_relevant_configs, device=self.device, dtype=torch.float32).unsqueeze(0)

        return relevant_configs, non_relevant_configs

    def predict(self, data: ModelPredictRequest) -> ModelPredictResponse:
        region = "a"  # TODO read from input
        # location encoding from [a, b, ..] to 1..9 / 9
        location = torch.tensor((self.metadata["location_dict"][region] + 1) / (len(self.metadata["location_dict"]) + 1))

        logger.info("pod embedding generation")
        pod_embedding = compute_embedding_for_sentence(str(data.pod_request), self.sentence_transformer)
        feature_input = torch.cat(([pod_embedding.squeeze(0), location.unsqueeze(0)])).unsqueeze(0)
        relevant_configs, non_relevant_configs = self.generate_configs_feature_set(region)

        # model input feature vector
        logger.info("model input feature vector")
        model_input = [feature_input, relevant_configs, non_relevant_configs]

        self.orchestrator.eval()
        logits = self.orchestrator.forward(model_input)
        predicted_configuration_id = logits.detach().numpy().argmax()

        predicted_config = self.metadata["id2configuration"][str(predicted_configuration_id)]
        predicted_config_dict = ast.literal_eval(predicted_config)

        return ModelPredictResponse(
            data.id,
            resource_profile=Resource(
                id=data.id, region=_extract_region(data.intents), cpu=f"{predicted_config_dict['cpu']}n",
                memory=f"{predicted_config_dict['memory']}Ki", architecture="arm64")
        )


def _extract_region(intents: list[Intent]) -> str:
    for intent in intents:
        if intent.name == "location":
            return intent.value
    return "dublin"
