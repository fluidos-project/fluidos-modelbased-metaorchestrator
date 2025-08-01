from fluidos_model_orchestrator.common.intent import KnownIntent


def test_intent_values():
    expected_intents = {
        "fluidos-intent-architecture",
        "fluidos-intent-battery",
        "fluidos-intent-compliance",
        "fluidos-intent-cpu",
        "fluidos-intent-max-delay",
        "fluidos-intent-carbon-aware",
        "fluidos-intent-energy",
        "fluidos-intent-gpu",
        "fluidos-intent-latency",
        "fluidos-intent-location",
        "fluidos-intent-memory",
        "fluidos-intent-resource",
        "fluidos-intent-service",
        "fluidos-intent-throughput",
        "fluidos-intent-bandwidth-against",
        "fluidos-intent-tee-readiness",
        "fluidos-intent-mspl",
        "fluidos-intent-vm-type",
        "fluidos-intent-sensor",
        "fluidos-intent-hardware",
        "fluidos-intent-cyber-deception",
        "fluidos-intent-magi",
    }

    for intent in KnownIntent:
        assert intent.to_intent_key() in expected_intents


def test_iternal_or_external():
    external_intents = {
        "fluidos-intent-service",
    }

    internal_intents = {
        "fluidos-intent-architecture",
        "fluidos-intent-battery",
        "fluidos-intent-compliance",
        "fluidos-intent-cpu",
        "fluidos-intent-max-delay",
        "fluidos-intent-carbon-aware",
        "fluidos-intent-energy",
        "fluidos-intent-gpu",
        "fluidos-intent-latency",
        "fluidos-intent-location",
        "fluidos-intent-memory",
        "fluidos-intent-resource",
        "fluidos-intent-throughput",
        "fluidos-intent-bandwidth-against",
        "fluidos-intent-tee-readiness",
        "fluidos-intent-mspl",
        "fluidos-intent-vm-type",
        "fluidos-intent-sensor",
        "fluidos-intent-hardware",
        "fluidos-intent-cyber-deception",
        "fluidos-intent-magi",
    }

    for intent in KnownIntent:
        key = intent.to_intent_key()
        assert (key in internal_intents and key not in external_intents) or (key in external_intents and key not in internal_intents), key
        assert (intent.is_external_requirement() and key in external_intents) or (key in internal_intents), key


def test_intent_validated():
    valid_intents = [
        intent.to_intent_key() for intent in KnownIntent
    ]

    for valid in valid_intents:
        assert KnownIntent.is_supported(valid)

    invalid_intents = [
        "asdfasdf-dfaklsjfa",
        "my-funny-intent",
        "fluidos-intent-not-yet-defined"
    ]

    for invalid in invalid_intents:
        assert not KnownIntent.is_supported(invalid)


def test_intents_requiring_monitoring():
    requires_monitoring = {
        "fluidos-intent-bandwidth-against",
        "fluidos-intent-battery",
        "fluidos-intent-energy",
        "fluidos-intent-latency",
        "fluidos-intent-throughput",
    }

    not_requires_monitoring = {
        "fluidos-intent-architecture",
        "fluidos-intent-carbon-aware",
        "fluidos-intent-compliance",
        "fluidos-intent-cpu",
        "fluidos-intent-cyber-deception",
        "fluidos-intent-gpu",
        "fluidos-intent-hardware",
        "fluidos-intent-location",
        "fluidos-intent-magi",
        "fluidos-intent-max-delay",
        "fluidos-intent-memory",
        "fluidos-intent-mspl",
        "fluidos-intent-resource",
        "fluidos-intent-sensor",
        "fluidos-intent-service",
        "fluidos-intent-tee-readiness",
        "fluidos-intent-vm-type",
    }

    for intent in KnownIntent:
        key = intent.to_intent_key()
        assert not (
            key in requires_monitoring and key in not_requires_monitoring
        ), f"{key} is in both"
        assert key in requires_monitoring or key in not_requires_monitoring, f"{key} is in neither"

        assert intent.needs_monitoring() or key in not_requires_monitoring, f"{key} not requiring monitoring but listed in requires monitoring"
