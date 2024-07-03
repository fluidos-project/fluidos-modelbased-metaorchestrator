import pkg_resources
import yaml

from fluidos_model_orchestrator.model import convert_to_model_request
from fluidos_model_orchestrator.model import get_model_object
from fluidos_model_orchestrator.model.carbon_aware.orchestrator import CarbonAwareOrchestrator


def test_something_is_returned_even_with_no_intents():
    with pkg_resources.resource_stream(__name__, "k8s/deployment.yaml") as stream:
        spec = convert_to_model_request(yaml.safe_load(stream), "fluidos")
    assert spec is not None

    assert type(get_model_object(spec)) is not CarbonAwareOrchestrator


def test_returns_most_matching():
    with pkg_resources.resource_stream(__name__, "k8s/pod_throughput_location.yaml") as stream:
        spec = convert_to_model_request(yaml.safe_load(stream), "fluidos")
    assert spec is not None

    assert type(get_model_object(spec)) is not CarbonAwareOrchestrator
