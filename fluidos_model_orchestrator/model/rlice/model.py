import json
import logging
from pathlib import Path
from typing import Any,cast

import pandas as pd
import numpy as np
import pkg_resources
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
from huggingface_hub import PyTorchModelHubMixin  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore

from fluidos_model_orchestrator.model.utils import convert_cpu_to_n
from fluidos_model_orchestrator.model.utils import convert_memory_to_Ki
from fluidos_model_orchestrator.common import cpu_to_int
from fluidos_model_orchestrator.common import memory_to_int
from fluidos_model_orchestrator.common import FlavorK8SliceData
from fluidos_model_orchestrator.common import ModelPredictRequest
from fluidos_model_orchestrator.common import ModelPredictResponse
from fluidos_model_orchestrator.common import OrchestratorInterface
from fluidos_model_orchestrator.common import ResourceProvider
from fluidos_model_orchestrator.common import Resource
from fluidos_model_orchestrator.model.utils import FLUIDOS_COL_NAMES

logger = logging.getLogger(__name__)
REPO_ID = "fluidos/rlice"

class EquivariantLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Gamma = nn.Linear(in_channels, out_channels, bias=False)
        self.Lambda = nn.Linear(in_channels, out_channels, bias=False)
        nn.init.kaiming_uniform_(self.Gamma.weight, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.Lambda.weight, nonlinearity='linear')
    
    def forward(self, x: torch.Tensor):
   
        xm, _ = torch.max(x, dim=1, keepdim=True)
  
        out = self.Lambda(x) - self.Gamma(xm)
       
        return out


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
                EquivariantLayer(5, 128),
                nn.ReLU(),
                EquivariantLayer(128, 64),
                nn.ReLU(),
                EquivariantLayer(64, 1),
            )

    def forward(self, x):
        
        out = self.network(x)
        return torch.squeeze(out, dim=-1)

class OrchestrationModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super().__init__()
        self.dqn = QNetwork()
        self.load_from_hugging_face(model_name="deepset_dqn.cleanrl_model")
        
        
    def load_from_hugging_face(self, model_name: str | None = None) -> Any:
        from huggingface_hub import hf_hub_download  # type: ignore

        model_to_load = model_name if model_name else "model.pt"
        # Download the model file from Hugging Face
        downloaded_model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=model_to_load,
        )
        # Load the model from the downloaded path
        self.dqn.load_state_dict(torch.load(downloaded_model_path))
        self.dqn.eval()
        

    def forward(self, obs: list[torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            q_values = self.dqn(torch.Tensor(obs).unsqueeze(0))
            print(q_values)
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
            node_id = actions[0]
        return node_id
        

class RliceOrchestrator(OrchestratorInterface):

    def __init__(self) -> None:

        self.model = OrchestrationModel()
        
    def min_max_normalization(self, state: list) -> torch.Tensor:
        
        num_nodes = len(state)
        min_vals = np.tile([0,0,0,0,0],(num_nodes,1))
        max_vals = np.tile([1000*1000,64*1024*1024,2,1000*1000,64*1024*1024],(num_nodes,1))
        normalized_state = (state - min_vals) / (max_vals - min_vals)
        normalized_state = np.clip(normalized_state, 0, 1)

        return normalized_state


    def predict(self, request: ModelPredictRequest, architecture: str = "amd64") -> ModelPredictResponse:
        
        return None
        #return ModelPredictResponse(id='pod-1', resource_profile=Resource(id='pod-1', cpu='2000m', memory='8192Ki', architecture='amd64', gpu=None, storage=None, region=None, pods=None), delay=0)

    def rank_resources(self, providers: list[ResourceProvider], prediction: ModelPredictResponse,
                       request: ModelPredictRequest) -> list[ResourceProvider]:

        logger.debug(f"ModelPredictRequest pod_request: {request.pod_request}")

        nodes_features = []
        cpuRequest,ramRequest=None,None

        # build input 
        modified_pod_manifest = request.pod_request[FLUIDOS_COL_NAMES.POD_MANIFEST]
        if "requests" in modified_pod_manifest['spec']['containers'][0]["resources"]:
            cpuRequest = convert_cpu_to_n(modified_pod_manifest['spec']['containers'][0]['resources']['requests']['cpu'], FLUIDOS_COL_NAMES.POD_CPU)
            ramRequest = convert_memory_to_Ki(modified_pod_manifest['spec']['containers'][0]['resources']['requests']['memory'], FLUIDOS_COL_NAMES.POD_MEMORY)

        if cpuRequest == None or cpuRequest <= 0:
            logging.exception("CPU request must be provided greater than 0")
            return []

        if ramRequest == None or ramRequest <= 0:
            logging.exception("RAM request must be provided greater than 0")
            return []

        for provider in providers:

            flavor = provider.flavor
            type_data = cast(FlavorK8SliceData, flavor.spec.flavor_type.type_data)
            logger.debug(f"provider ID: {provider.id}")
            logger.debug(f"flavor ID: {flavor.metadata.name}")
            logger.debug(f"flavor characteristics: {type_data.characteristics}")
            logger.debug(f"flavor price_dict: {flavor.spec.price}")

            price_dict = flavor.spec.price

            if len(price_dict.keys()) == 0:
                logging.exception(f"Skipping flavor {provider.flavor.metadata.name} from provider {provider.id} as with no price information")
                continue

            amount = float(price_dict['amount'])
            period = price_dict['period']

            if period == "weekly":
                hourly_price = amount/(24*7)
            elif period == "monthly":
                hourly_price = amount/(24*30)
            elif period == "hourly":
                hourly_price = amount
            else:
                logging.exception(f"Skipping flavor {provider.flavor.metadata.name} from provider {provider.id} as period {period} is not supported")
                continue

            cpu = cpu_to_int(type_data.characteristics.cpu)
            mem = memory_to_int(type_data.characteristics.memory)
            
            row = [cpu,mem,hourly_price,cpuRequest,ramRequest]
            nodes_features.append(row)

        model_input = self.min_max_normalization(nodes_features)
        index = self.model.forward(model_input)



        
           

