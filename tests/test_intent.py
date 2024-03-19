from fluidos_model_orchestrator.common import KnownIntent


def test_intent_values():
    expected_intents = [
        "fluidos-intent-latency",
        "fluidos-intent-location",
        "fluidos-intent-throughput",
        "fluidos-intent-compliance",
        "fluidos-intent-energy",
        "fluidos-intent-battery",
        "fluidos-intent-service",
        "fluidos-intent-cpu",
        "fluidos-intent-memory"
    ]

    for intent in KnownIntent:
        assert intent.to_intent_key() in expected_intents


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


def test_intent_converted():
    expected = [
        ("fluidos-intent-latency", KnownIntent.latency),
        ("fluidos-intent-location", KnownIntent.location),
        ("fluidos-intent-throughput", KnownIntent.throughput),
        ("fluidos-intent-compliance", KnownIntent.compliance),
        ("fluidos-intent-energy", KnownIntent.energy),
        ("fluidos-intent-battery", KnownIntent.battery),
        ("fluidos-intent-service", KnownIntent.service),
        ("fluidos-intent-cpu", KnownIntent.cpu),
        ("fluidos-intent-memory", KnownIntent.memory),
    ]

    for key, intent in expected:
        assert intent == KnownIntent.get_intent(key)
