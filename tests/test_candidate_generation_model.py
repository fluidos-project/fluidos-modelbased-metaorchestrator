import yaml
import importlib_resources
from fluidos_model_orchestrator.model import convert_to_model_request
from fluidos_model_orchestrator.model.candidate_generation import Orchestrator


def test_orchestration_model_with_throughput():
    ref = importlib_resources.files(__name__).joinpath("k8s/pod_throughput_location.yaml")
    with ref.open('rb') as pod_stream:
        pod_dict = yaml.safe_load(pod_stream)

    request = convert_to_model_request(pod_dict)
    model = Orchestrator(device="cpu")
    response = model.predict(request)

    assert response
    assert response.resource_profile.region == 'a'
    assert response.resource_profile.cpu == "1000m"
    # assert response.resource_profile.memory == "155Mi"
    assert response.resource_profile.memory == "509Mi"
