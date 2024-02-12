from fluidos_model_orchestrator.model import get_model_object
from fluidos_model_orchestrator.model import DummyOrchestrator


def test_defaults_to_dummy():
    assert type(get_model_object(None)) is DummyOrchestrator
