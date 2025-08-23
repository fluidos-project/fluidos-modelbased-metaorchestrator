import importlib.resources

import pytest  # type: ignore
from yaml import load
from yaml import Loader

from fluidos_model_orchestrator.model import convert_to_model_request


def test_when_no_intents_specified_none_injected_pod():
    with (importlib.resources.files(__package__) / "k8s/pod.yaml").open("r") as stream:
        specs = load(stream, Loader=Loader)
    request = convert_to_model_request(specs, "fluidos")

    assert request is not None
    assert not len(request.intents)


def test_when_no_intents_specified_none_injected_deployment():
    with (importlib.resources.files(__package__) / "k8s/deployment.yaml").open("r") as stream:
        specs = load(stream, Loader=Loader)
    request = convert_to_model_request(specs, "fluidos")

    assert request is not None
    assert not len(request.intents)


@pytest.mark.skip("ReplicaSet not supported yet")
def test_when_no_intents_specified_none_injected_replica_set():
    with (importlib.resources.files(__package__) / "k8s/replica_set.yaml").open("r") as stream:
        specs = load(stream, Loader=Loader)
    request = convert_to_model_request(specs, "fluidos")

    assert request is not None
    assert not len(request.intents)


def test_intents_correctly_parsed():
    with (importlib.resources.files(__package__) / "examples/test-pod-w-intent.yaml").open("r") as stream:
        specs = load(stream, Loader=Loader)["spec"]
    request = convert_to_model_request(specs, "fluidos")

    assert request is not None
    assert len(request.intents)
