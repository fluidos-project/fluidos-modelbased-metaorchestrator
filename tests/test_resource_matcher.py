from fluidos_model_orchestrator.common import Flavor
from fluidos_model_orchestrator.common import FlavorCharacteristics
from fluidos_model_orchestrator.common import FlavorType
from fluidos_model_orchestrator.common import Resource


def test_request_satisfied():
    res = Resource(id="foo", cpu="2n", memory="10Mi")

    flavor = Flavor(
        id="bar",
        type=FlavorType.K8SLICE,
        characteristics=FlavorCharacteristics(architecture="amd64", cpu="2000000n", memory="100Gi", gpu="0"),
        owner={})

    assert res.can_run_on(flavor)


def test_request_not_sastisfied():
    res = Resource(id="foo", cpu="2n", memory="10Mi", gpu="1")

    assert not res.can_run_on(Flavor(
        id="bar",
        type=FlavorType.K8SLICE,
        characteristics=FlavorCharacteristics(architecture="amd64", cpu="2000000n", memory="100Gi", gpu="0"),
        owner={})), "Missing GPU"
    assert not res.can_run_on(Flavor(
        id="bar",
        type=FlavorType.K8SLICE,
        characteristics=FlavorCharacteristics(architecture="amd64", cpu="1n", memory="100Gi", gpu="1"),
        owner={})), "Not enough CPU"
