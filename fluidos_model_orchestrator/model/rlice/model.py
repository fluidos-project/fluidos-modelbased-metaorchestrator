import json
import logging
from pathlib import Path
from typing import Any,cast

import pandas as pd
import pkg_resources
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
from huggingface_hub import PyTorchModelHubMixin  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore

from fluidos_model_orchestrator.common import cpu_to_int
from fluidos_model_orchestrator.common import FlavorK8SliceData
from fluidos_model_orchestrator.common import KnownIntent
from fluidos_model_orchestrator.common import memory_to_int
from fluidos_model_orchestrator.common import ModelPredictRequest
from fluidos_model_orchestrator.common import ModelPredictResponse
from fluidos_model_orchestrator.common import OrchestratorInterface
from fluidos_model_orchestrator.common import ResourceProvider
from fluidos_model_orchestrator.common import Resource

logger = logging.getLogger(__name__)

class RliceOrchestrator(OrchestratorInterface):

    def predict(self, request: ModelPredictRequest, architecture: str = "amd64") -> ModelPredictResponse:
        
        #return None
        return ModelPredictResponse(id='pod-1', resource_profile=Resource(id='pod-1', cpu='2000m', memory='8192Ki', architecture='amd64', gpu=None, storage=None, region=None, pods=None), delay=0)

    def rank_resources(self, providers: list[ResourceProvider], prediction: ModelPredictResponse,
                       request: ModelPredictRequest) -> list[ResourceProvider]:

        logger.debug(f"ModelPredictRequest pod_request: {request.pod_request}")
        nodes_features = []
        for intent in request.intents:
            if intent.name is KnownIntent.cpu:
                cpuRequest = cpu_to_int(intent.value)
                logger.debug(f"Found cpu request from intent file: {cpuRequest}")
            elif intent.name is KnownIntent.memory:
                ramRequest = memory_to_int(intent.value)
                logger.debug(f"Found memory request from intent file: {ramRequest}")

        for provider in providers:
            flavor = provider.flavor
            type_data = cast(FlavorK8SliceData, flavor.spec.flavor_type.type_data)
           
            logger.debug(f"provider ID: {provider.id}")
            logger.debug(f"flavor ID: {provider.flavor.metadata.name}")
            logger.debug(f"flavor characteristics: {type_data.characteristics}")
            logger.debug(f"flavor optional_fields: {type_data.properties}")

            cpu = cpu_to_int(type_data.characteristics.cpu)
            mem = memory_to_int(type_data.characteristics.memory)
            bw = type_data.properties.get("additionalProperties", {}).get("bandwidth", {})
            row = [cpu,mem,bw,cpuRequest,ramRequest]
            nodes_features.append(row)

        
           

