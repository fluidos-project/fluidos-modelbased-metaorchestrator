from fluidos_model_orchestrator.common import Resource
from fluidos_model_orchestrator.common import Flavour


def test_request_satisfied():
    res = Resource(id="foo", cpu="2n", memory="10Mi")

    flavour = Flavour(id="bar", architecture="amd64", cpu="2000000n", memory="100Gi", gpu="0")

    assert res.can_run_on(flavour)


def test_request_not_sastisfied():
    res = Resource(id="foo", cpu="2n", memory="10Mi", gpu="1")

    assert not res.can_run_on(Flavour(id="bar", architecture="amd64", cpu="2000000n", memory="100Gi", gpu="0")), "Missing GPU"
    assert not res.can_run_on(Flavour(id="bar", architecture="amd64", cpu="1n", memory="100Gi", gpu="1")), "Not enough CPU"
