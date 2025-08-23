import importlib.resources

import yaml

from fluidos_model_orchestrator.model import convert_to_model_request
from fluidos_model_orchestrator.model import get_model_object


def test_rse():
    f = importlib.resources.files(__package__) / "k8s/rse-example.yaml"
    with f.open("r") as stream:
        spec = convert_to_model_request(
            yaml.safe_load(stream)["spec"],
            "fluidos"
        )
    assert spec is not None

    m = get_model_object(spec)

    r = m.predict(spec)

    assert r is not None
