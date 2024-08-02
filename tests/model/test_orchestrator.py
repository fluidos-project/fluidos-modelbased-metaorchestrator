import importlib.resources

import pytest  # type: ignore
import yaml

from fluidos_model_orchestrator.common import ModelPredictRequest
from fluidos_model_orchestrator.common import ModelPredictResponse
from fluidos_model_orchestrator.model import convert_to_model_request
from fluidos_model_orchestrator.model.candidate_generation import Orchestrator as OrchestratorV1
from fluidos_model_orchestrator.model.candidate_generation import Orchestrator
from fluidos_model_orchestrator.model.model_basic_ranker.model import BasicRankerModel
from fluidos_model_orchestrator.model.orchestrator_factory import OrchestratorFactory
from fluidos_model_orchestrator.model.utils import MODEL_TYPES

ORCHESTRATOR_MODELS_TO_TEST = [MODEL_TYPES.BASIC_RANKER]


def create_orchestrator_sample_request(model_type: str) -> ModelPredictRequest:
    if model_type == MODEL_TYPES.BASIC_RANKER:
        return BasicRankerModel.create_sample_request()
    elif model_type == MODEL_TYPES.CG:
        raise NotImplementedError("Not implemented: abstract method")
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


#TODO should merge this test with any of the tests above
@pytest.mark.xfail(reason="No CGV1 model is avilable, to be replaced with HF calls")
def test_orchestration_model_with_throughput_old():
    ref = importlib.resources.files(__package__).joinpath("k8s/pod_throughput_location.yaml")
    with ref.open('rb') as pod_stream:
        pod_dict = yaml.safe_load(pod_stream)

    request = convert_to_model_request(pod_dict, "fluidos")
    assert request is not None
    model = OrchestratorV1(device="cpu")
    response = model.predict(request)

    assert response
    assert response.resource_profile.region == 'a'
    assert response.resource_profile.cpu == "1000m"
    # assert response.resource_profile.memory == "155Mi"
    assert response.resource_profile.memory == "509Mi"


#TODO should merge this test with any of the tests above
@pytest.mark.xfail(reason="No CGV2 model is avilable for the moment, to be replaced with HF calls")
def test_orchestration_model_with_throughput():
    ref = importlib.resources.files(__package__).joinpath("k8s/pod_throughput_location.yaml")
    with ref.open('rb') as pod_stream:
        pod_dict = yaml.safe_load(pod_stream)

    request = convert_to_model_request(pod_dict, "fluidos")
    assert request is not None
    model = Orchestrator(device="cpu")
    response = model.predict(request)

    assert response
    assert response.resource_profile.region == 'bitbrains_a'
    assert response.resource_profile.cpu == "1000m"
    assert response.resource_profile.memory == "686Mi"
