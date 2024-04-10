from fluidos_model_orchestrator.common import KnownIntent
# from pytest import fail


def test_intent_values():
    expected_intents = set([
        "fluidos-intent-cpu",
        "fluidos-intent-memory",
        "fluidos-intent-latency",
        "fluidos-intent-location",
        "fluidos-intent-resource",
        "fluidos-intent-throughput",
        "fluidos-intent-compliance",
        "fluidos-intent-energy",
        "fluidos-intent-battery",
        "fluidos-intent-service",
    ])

    for intent in KnownIntent:
        assert intent.to_intent_key() in expected_intents


def test_iternal_or_external():
    external_intents = set([
        "fluidos-intent-service",
    ])

    internal_intents = set([
        "fluidos-intent-cpu",
        "fluidos-intent-memory",
        "fluidos-intent-latency",
        "fluidos-intent-location",
        "fluidos-intent-resource",
        "fluidos-intent-throughput",
        "fluidos-intent-compliance",
        "fluidos-intent-energy",
        "fluidos-intent-battery",
    ])

    for intent in KnownIntent:
        key = intent.to_intent_key()
        assert (key in internal_intents and key not in external_intents) or (key in external_intents and key not in internal_intents)
        assert (intent.has_external_requirement() and key in external_intents) or (key in internal_intents)


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
