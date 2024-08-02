from typing import Any

import pandas as pd

from fluidos_model_orchestrator.common import OrchestratorInterface
from fluidos_model_orchestrator.model.candidate_generation import Orchestrator as OrchestratorV1
from fluidos_model_orchestrator.model.model_basic_ranker.model import Orchestrator as BasicRankerOrchestrator
from fluidos_model_orchestrator.model.utils import MODEL_TYPES


class OrchestratorFactory:

    @staticmethod
    def create_orchestrator(
        model_type: str
    ) -> OrchestratorInterface:
        print("Creating orchestrator")
        if model_type == MODEL_TYPES.BASIC_RANKER:
            return BasicRankerOrchestrator()
        elif model_type == MODEL_TYPES.CG:
            return OrchestratorV1(device="cpu")
        else:
            raise ValueError(f"Can't find what model type {model_type} is referring to")
