import pytest  # type: ignore

from fluidos_model_orchestrator.common.model import ModelPredictRequest
from fluidos_model_orchestrator.common.model import ModelPredictResponse
from fluidos_model_orchestrator.model.candidate_generation.model import Orchestrator as CGOrchestrator
from fluidos_model_orchestrator.model.model_ranker.model import Orchestrator as BasicRankerOrchestrator
from fluidos_model_orchestrator.model.orchestrator_factory import OrchestratorFactory
from fluidos_model_orchestrator.model.utils import MODEL_TYPES


ORCHESTRATOR_MODELS_TO_TEST = [MODEL_TYPES.CG, MODEL_TYPES.CG_LEGACY, MODEL_TYPES.CG_75]  # , MODEL_TYPES.BASIC_RANKER]


def create_orchestrator_sample_request(model_type: str) -> ModelPredictRequest:
    if model_type == MODEL_TYPES.CG:
        return CGOrchestrator.create_sample_request()
    elif model_type == MODEL_TYPES.CG_LEGACY:
        return CGOrchestrator.create_sample_request_legacy()
    elif model_type == MODEL_TYPES.CG_75:
        return CGOrchestrator.create_sample_request_75()
    elif model_type == MODEL_TYPES.BASIC_RANKER:
        return BasicRankerOrchestrator.create_sample_request()
    else:
        raise ValueError(f"Can't find what model type {model_type} is referring to")


@pytest.mark.parametrize("model_type", ORCHESTRATOR_MODELS_TO_TEST)
def test_orchestrator_loading(model_type: str) -> None:
    assert OrchestratorFactory.create_orchestrator(model_type)


@pytest.mark.parametrize("model_type", ORCHESTRATOR_MODELS_TO_TEST)
def test_orchestrator_predict(model_type: str) -> None:
    orchestrator = OrchestratorFactory.create_orchestrator(model_type)
    response = orchestrator.predict(create_orchestrator_sample_request(model_type))
    assert isinstance(response, ModelPredictResponse)
    assert response is not None


def test_cg_orchestrator_predict_values() -> None:
    orchestrator = OrchestratorFactory.create_orchestrator(MODEL_TYPES.CG)
    response = orchestrator.predict(create_orchestrator_sample_request(MODEL_TYPES.CG))
    assert response is not None
    assert response.resource_profile.region is None

    assert response.resource_profile.cpu == "1000m"
    assert response.resource_profile.memory == "509Ki"


def test_cg_orchestrator_75_predict_values() -> None:
    orchestrator = OrchestratorFactory.create_orchestrator(MODEL_TYPES.CG_75)
    response = orchestrator.predict(create_orchestrator_sample_request(MODEL_TYPES.CG_75))
    assert response is not None
    assert response.resource_profile.region is None
    assert response.resource_profile.cpu == "1000m"
    assert response.resource_profile.memory == "124923Ki"


def test_cg_legacy_orchestrator_predict_values() -> None:
    orchestrator = OrchestratorFactory.create_orchestrator(MODEL_TYPES.CG_LEGACY)
    response = orchestrator.predict(create_orchestrator_sample_request(MODEL_TYPES.CG_LEGACY))
    assert response is not None

    assert response.resource_profile.region is None
    assert response.resource_profile.cpu == "1000m"
    assert response.resource_profile.memory == "509Mi"
