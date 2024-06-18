from os import environ

import pkg_resources
import pytest
import yaml

from fluidos_model_orchestrator.model import convert_to_model_request
from fluidos_model_orchestrator.model.two_tower_v1.orchestrator import TwoTowerOrchestrator


@pytest.mark.xfail(environ.get("CI", "false") == "true", reason="Not always running on Travis")
def test_orchestration_model_two_towers():
    with pkg_resources.resource_stream(__name__, "k8s/pod_throughput_location.yaml") as pod_stream:
        pod_dict = yaml.safe_load(pod_stream)

    request = convert_to_model_request(pod_dict, "fluidos")
    assert request is not None

    model = TwoTowerOrchestrator()
    response = model.predict(request)

    assert response
    assert response.resource_profile.cpu == 4000
    assert response.resource_profile.memory == 16312
