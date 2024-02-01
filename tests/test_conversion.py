import pkg_resources
import pytest
from yaml import load
from yaml import Loader
from fluidos_model_orchestrator.model import convert_to_model_request


def test_when_no_intents_specified_none_injected_pod():
    with pkg_resources.resource_stream(__name__, "k8s/pod.yaml") as stream:
        specs = load(stream, Loader=Loader)
    request = convert_to_model_request(specs)

    assert request is not None
    assert not len(request.intents)


def test_when_no_intents_specified_none_injected_deployment():
    with pkg_resources.resource_stream(__name__, "k8s/deployment.yaml") as stream:
        specs = load(stream, Loader=Loader)
    request = convert_to_model_request(specs)

    assert request is not None
    assert not len(request.intents)


@pytest.mark.skip("Not supported yet")
def test_when_no_intents_specified_none_injected_replica_set():
    with pkg_resources.resource_stream(__name__, "k8s/replica_set.yaml") as stream:
        specs = load(stream, Loader=Loader)
    request = convert_to_model_request(specs)

    assert request is not None
    assert not len(request.intents)


def test_intents_correctly_parsed():
    with pkg_resources.resource_stream(__name__, "examples/test-pod-w-intent.yaml") as stream:
        specs = load(stream, Loader=Loader)["spec"]
    request = convert_to_model_request(specs)

    assert request is not None
    assert len(request.intents)
