from fluidos_model_orchestrator.model import get_model_object
from fluidos_model_orchestrator.model.dummy import DummyOrchestrator
from fluidos_model_orchestrator.model.candidate_generation import Orchestrator as CGOrchestrator
import pytest


def test_defaults_to_dummy():
    assert type(get_model_object(None)) is DummyOrchestrator


@pytest.mark.skip(reason="skip until get_model_object is updated")
def test_cg_model_selection():
    assert type(get_model_object("candidate_generation")) is CGOrchestrator
