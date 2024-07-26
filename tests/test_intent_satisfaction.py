from fluidos_model_orchestrator.common import Flavor
from fluidos_model_orchestrator.common import FlavorCharacteristics
from fluidos_model_orchestrator.common import FlavorType
from fluidos_model_orchestrator.common import Intent
from fluidos_model_orchestrator.common import KnownIntent
from fluidos_model_orchestrator.resources.rear.local_resource_provider import LocalResourceProvider


def test_satisfaction_hardware_resources():
    provider_ok = LocalResourceProvider("ok", Flavor(
        "foo",
        FlavorType.K8SLICE,
        FlavorCharacteristics(
            cpu="10n",
            memory="32Gi",
            architecture="amd64",
            gpu="1",
        ),
        {},
        "provider_id",
        {},
        {}
    ))

    provider_not_ok = LocalResourceProvider("not ok", Flavor(
        "foo",
        FlavorType.K8SLICE,
        FlavorCharacteristics(
            cpu="1n",
            memory="1Gi",
            architecture="arm64",
            gpu="0",
        ),
        {},
        "provider_id",
        {},
        {}
    ))

    intents = [
        Intent(KnownIntent.cpu, "10n"),
        Intent(KnownIntent.memory, "2Gi"),
        Intent(KnownIntent.gpu, "1"),
        Intent(KnownIntent.architecture, "amd64"),
    ]

    for intent in intents:
        assert intent.validates(provider_ok), intent

    for intent in intents:
        assert not intent.validates(provider_not_ok), intent
