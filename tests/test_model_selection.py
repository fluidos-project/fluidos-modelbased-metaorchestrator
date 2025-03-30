import pkg_resources
import yaml

from fluidos_model_orchestrator.model import convert_to_model_request
from fluidos_model_orchestrator.model import get_model_object
from fluidos_model_orchestrator.model.candidate_generation.model import Orchestrator as CandidateGeneration
from fluidos_model_orchestrator.model.carbon_aware.orchestrator import CarbonAwareOrchestrator
from fluidos_model_orchestrator.model.ensemble import FluidosModelEnsemble


def test_something_is_returned_even_with_no_intents():
    with pkg_resources.resource_stream(__name__, "k8s/deployment.yaml") as stream:
        spec = convert_to_model_request(yaml.safe_load(stream), "fluidos")
    assert spec is not None

    assert type(get_model_object(spec)) is not CarbonAwareOrchestrator


def test_returns_most_matching():
    with pkg_resources.resource_stream(__name__, "k8s/pod_throughput_location.yaml") as stream:
        spec = convert_to_model_request(yaml.safe_load(stream), "fluidos")
    assert spec is not None

    assert type(get_model_object(spec)) is CandidateGeneration


def test_returns_ensamble():
    with pkg_resources.resource_stream(__name__, "k8s/rse-example.yaml") as stream:
        spec = convert_to_model_request(
            yaml.safe_load(stream)["spec"],
            "fluidos"
        )
    assert spec is not None

    m = get_model_object(spec)

    assert type(m) is FluidosModelEnsemble
