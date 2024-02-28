import yaml
import importlib_resources
from fluidos_model_orchestrator.model import convert_to_model_request
from fluidos_model_orchestrator.model.candidate_generation import Orchestrator

import pkg_resources
import pytest
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


def test_orchestration_model_with_throughput():
    ref = importlib_resources.files(__name__).joinpath(
        "k8s/pod_throughput_location.yaml"
    )
    with ref.open("rb") as pod_stream:
        pod_dict = yaml.safe_load(pod_stream)

    request = convert_to_model_request(pod_dict)
    model = Orchestrator(device="cpu")
    response = model.predict(request)

    assert response
    assert response.resource_profile.region == "a"
    assert response.resource_profile.cpu == "1000m"
    assert response.resource_profile.memory == "509Mi"


@pytest.mark.skipif(
    models_missing(),
    reason="Skip this test if the test models do not exist",
)
def test_orchestration_model_two_towers():
    ref = importlib_resources.files(__name__).joinpath(
        "k8s/pod_throughput_location.yaml"
    )
    with ref.open("rb") as pod_stream:
        pod_dict = yaml.safe_load(pod_stream)

    request = convert_to_model_request(pod_dict)

    model = TwoTowerOrchestrator(models_path, model_name="model_2t_v1", device="cpu")
    response = model.predict(request)

    assert response
    # TODO regions are not yet supported by model
    assert response.resource_profile.region == "dummyRegion"
    assert response.resource_profile.cpu == 4000
    assert response.resource_profile.memory == 16312
