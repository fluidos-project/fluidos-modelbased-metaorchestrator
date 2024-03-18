import yaml
import pkg_resources
# from fluidos_model_orchestrator.model.v1.orchestrator import Orchestrator
import pytest
# import importlib_resources
import pkg_resources
import os
from pathlib import Path
from fluidos_model_orchestrator.model import (
    ModelPredictRequest,
    convert_to_model_request,
)

from fluidos_model_orchestrator.model.two_tower_v1.orchestrator import (
    TwoTowerOrchestrator,
)

models_path = Path(
    os.path.dirname(os.path.abspath(__file__)),
    "model_resources/",
)


def models_missing():
    return False if models_path.exists() else True

@pytest.mark.skip(reason="outdated model version")
def test_orchestration_model():
    with pkg_resources.resource_stream(__name__, "k8s/pod.yaml") as pod_stream:
        pod_dict = yaml.safe_load(pod_stream)

    pod_string = str(pod_dict)

    request = ModelPredictRequest(id="pod", pod_request=pod_string)
    model = OrchestratorV1(device="cpu")
    response = model.predict(request)

    assert response
    assert response.resource_profile.region == "a"
    assert response.resource_profile.cpu == "28407n"
    assert response.resource_profile.memory == "174848Ki"


@pytest.mark.skipif(
    models_missing(),
    reason="Skip this test if the test models do not exist",
)
def test_orchestration_model_two_towers():
    with pkg_resources.resource_stream(__name__, "k8s/pod_throughput_location.yaml") as pod_stream:
        pod_dict = yaml.safe_load(pod_stream)

    request = convert_to_model_request(pod_dict)

    model = TwoTowerOrchestrator(models_path, model_name="model_2t_v1", device="cpu")
    response = model.predict(request)

    assert response
    # TODO regions are not yet supported by model
    assert response.resource_profile.region == "dummyRegion"
    assert response.resource_profile.cpu == 4000
    assert response.resource_profile.memory == 16312
