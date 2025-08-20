from fluidos_model_orchestrator.common.flavor import Flavor
from fluidos_model_orchestrator.common.flavor import FlavorCharacteristics
from fluidos_model_orchestrator.common.flavor import FlavorK8SliceData
from fluidos_model_orchestrator.common.flavor import FlavorMetadata
from fluidos_model_orchestrator.common.flavor import FlavorSpec
from fluidos_model_orchestrator.common.flavor import FlavorType
from fluidos_model_orchestrator.common.flavor import FlavorTypeData
from fluidos_model_orchestrator.common.flavor import GPUData
from fluidos_model_orchestrator.common.intent import Intent
from fluidos_model_orchestrator.common.intent import KnownIntent
from fluidos_model_orchestrator.common.intent import validate_location
from fluidos_model_orchestrator.resources.rear.local_resource_provider import LocalResourceProvider


def test_satisfaction_hardware_resources():
    provider_ok = LocalResourceProvider("ok", Flavor(
        metadata=FlavorMetadata(
            name="foo",
            owner_references={}
        ),
        spec=FlavorSpec(
            availability=True,
            flavor_type=FlavorTypeData(
                type_identifier=FlavorType.K8SLICE,
                type_data=FlavorK8SliceData(
                    characteristics=FlavorCharacteristics(
                        cpu="10n",
                        memory="32Gi",
                        architecture="amd64",
                        gpu=GPUData(
                            cores=1,
                            memory="",
                            model=""
                        )
                    ),
                    policies={},
                    properties={}
                )
            ),
            location={},
            network_property_type="",
            owner={},
            providerID="provider_id",
            price={}
        )
    ))

    provider_not_ok = LocalResourceProvider("not_ok", Flavor(
        metadata=FlavorMetadata(
            name="foo",
            owner_references={}
        ),
        spec=FlavorSpec(
            availability=True,
            flavor_type=FlavorTypeData(
                type_identifier=FlavorType.K8SLICE,
                type_data=FlavorK8SliceData(
                    characteristics=FlavorCharacteristics(
                        cpu="1n",
                        memory="1Gi",
                        architecture="arm",
                        gpu=GPUData(
                            cores=0,
                            memory="",
                            model=""
                        )
                    ),
                    policies={},
                    properties={}
                )
            ),
            location={},
            network_property_type="",
            owner={},
            providerID="provider_id",
            price={}
        )
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


def test_validation_of_location():
    provider = LocalResourceProvider("test", Flavor(
        metadata=FlavorMetadata(
            name="foo",
            owner_references={}
        ),
        spec=FlavorSpec(
            availability=True,
            flavor_type=FlavorTypeData(
                type_identifier=FlavorType.K8SLICE,
                type_data=FlavorK8SliceData(
                    characteristics=FlavorCharacteristics(cpu="1n", memory="1Gi", architecture="arm", gpu=GPUData(cores=0, memory="", model="")),
                    policies={},
                    properties={}
                )
            ),
            location={
                "city": "Dublin",
                "country": "Ireland",
            },
            network_property_type="",
            owner={},
            providerID="provider_id",
            price={}
        )
    ))

    assert validate_location(provider, "Dublin")
    assert validate_location(provider, "Ireland")
    assert not validate_location(provider, "Turin")
    assert not validate_location(provider, "Italy")


def test_validate_bandwidth_against_satisfaction() -> None:
    provider = LocalResourceProvider("test", Flavor(
        metadata=FlavorMetadata(
            name="foo",
            owner_references={}
        ),
        spec=FlavorSpec(
            availability=True,
            flavor_type=FlavorTypeData(
                type_identifier=FlavorType.K8SLICE,
                type_data=FlavorK8SliceData(
                    characteristics=FlavorCharacteristics(cpu="1n", memory="1Gi", architecture="arm", gpu=GPUData(cores=0, memory="", model="")),
                    policies={},
                    properties={
                        "additionalProperties": {
                            "bandwidth": {
                                "POINT-A": "500ms",
                                "POINT-B": "200ms",
                                "AZURE": "100ms",
                            }
                        }
                    }
                )
            ),
            location={
                "city": "Dublin",
                "country": "Ireland",
            },
            network_property_type="",
            owner={},
            providerID="provider_id",
            price={}
        )
    ))

    assert Intent(KnownIntent.bandwidth_against, "<= 200ms POINT-B").validates(provider)
    assert Intent(KnownIntent.bandwidth_against, "< 300ms POINT-B").validates(provider)
    assert not Intent(KnownIntent.bandwidth_against, "< 100ms POINT-B").validates(provider)
    assert Intent(KnownIntent.bandwidth_against, "<= 200ms AZURE").validates(provider)


def test_validate_tee_readiness() -> None:
    bad_no_tee = LocalResourceProvider("test", Flavor(
        metadata=FlavorMetadata(
            name="foo",
            owner_references={}
        ),
        spec=FlavorSpec(
            availability=True,
            flavor_type=FlavorTypeData(
                type_identifier=FlavorType.K8SLICE,
                type_data=FlavorK8SliceData(
                    characteristics=FlavorCharacteristics(cpu="1n", memory="1Gi", architecture="arm", gpu=GPUData(cores=0, memory="", model="")),
                    policies={},
                    properties={
                        "additionalProperties": {
                            "TEE": False
                        }
                    }
                )
            ),
            location={
                "city": "Dublin",
                "country": "Ireland",
            },
            network_property_type="",
            owner={},
            providerID="provider_id",
            price={}
        )
    ))
    bad_no_information = LocalResourceProvider("test", Flavor(
        metadata=FlavorMetadata(
            name="foo",
            owner_references={}
        ),
        spec=FlavorSpec(
            availability=True,
            flavor_type=FlavorTypeData(
                type_identifier=FlavorType.K8SLICE,
                type_data=FlavorK8SliceData(
                    characteristics=FlavorCharacteristics(cpu="1n", memory="1Gi", architecture="arm", gpu=GPUData(cores=0, memory="", model="")),
                    policies={},
                    properties={}
                )
            ),
            location={
                "city": "Dublin",
                "country": "Ireland",
            },
            network_property_type="",
            owner={},
            providerID="provider_id",
            price={}
        )
    ))
    good = LocalResourceProvider("test", Flavor(
        metadata=FlavorMetadata(
            name="foo",
            owner_references={}
        ),
        spec=FlavorSpec(
            availability=True,
            flavor_type=FlavorTypeData(
                type_identifier=FlavorType.K8SLICE,
                type_data=FlavorK8SliceData(
                    characteristics=FlavorCharacteristics(cpu="1n", memory="1Gi", architecture="arm", gpu=GPUData(cores=0, memory="", model="")),
                    policies={},
                    properties={
                        "additionalProperties": {
                            "TEE": True
                        }
                    }
                )
            ),
            location={
                "city": "Dublin",
                "country": "Ireland",
            },
            network_property_type="",
            owner={},
            providerID="provider_id",
            price={}
        )
    ))

    intent1 = Intent(KnownIntent.tee_readiness, "True")
    intent2 = Intent(KnownIntent.tee_readiness, "true")

    assert intent1.validates(good)
    assert not intent1.validates(bad_no_tee)
    assert not intent1.validates(bad_no_information)

    assert intent2.validates(good)
    assert not intent2.validates(bad_no_tee)
    assert not intent2.validates(bad_no_information)


def test_validate_vm_type_satisfaction() -> None:
    provider_bad_wrong_value = LocalResourceProvider("test", Flavor(
        metadata=FlavorMetadata(
            name="foo",
            owner_references={}
        ),
        spec=FlavorSpec(
            availability=True,
            flavor_type=FlavorTypeData(
                type_identifier=FlavorType.K8SLICE,
                type_data=FlavorK8SliceData(
                    characteristics=FlavorCharacteristics(cpu="1n", memory="1Gi", architecture="arm", gpu=GPUData(cores=0, memory="", model="")),
                    policies={},
                    properties={
                        "additionalProperties": {
                            "vm-type": "the-wrong-vm-type"
                        }
                    }
                )
            ),
            location={
                "city": "Dublin",
                "country": "Ireland",
            },
            network_property_type="",
            owner={},
            providerID="provider_id",
            price={}
        )
    ))
    provider_bad_no_value = LocalResourceProvider("test", Flavor(
        metadata=FlavorMetadata(
            name="foo",
            owner_references={}
        ),
        spec=FlavorSpec(
            availability=True,
            flavor_type=FlavorTypeData(
                type_identifier=FlavorType.K8SLICE,
                type_data=FlavorK8SliceData(
                    characteristics=FlavorCharacteristics(cpu="1n", memory="1Gi", architecture="arm", gpu=GPUData(cores=0, memory="", model="")),
                    policies={},
                    properties={
                        "additionalProperties": {
                        }
                    }
                )
            ),
            location={
                "city": "Dublin",
                "country": "Ireland",
            },
            network_property_type="",
            owner={},
            providerID="provider_id",
            price={}
        )
    ))
    provider_good = LocalResourceProvider("test", Flavor(
        metadata=FlavorMetadata(
            name="foo",
            owner_references={}
        ),
        spec=FlavorSpec(
            availability=True,
            flavor_type=FlavorTypeData(
                type_identifier=FlavorType.K8SLICE,
                type_data=FlavorK8SliceData(
                    characteristics=FlavorCharacteristics(cpu="1n", memory="1Gi", architecture="arm", gpu=GPUData(cores=0, memory="", model="")),
                    policies={},
                    properties={
                        "additionalProperties": {
                            "vm-type": "my-vm-type"
                        }
                    }
                )
            ),
            location={
                "city": "Dublin",
                "country": "Ireland",
            },
            network_property_type="",
            owner={},
            providerID="provider_id",
            price={}
        )
    ))

    intent = Intent(KnownIntent.vm_type, "my-vm-type")

    assert intent.validates(provider_good)
    assert not intent.validates(provider_bad_wrong_value)
    assert not intent.validates(provider_bad_no_value)


def test_validate_sensor_availability() -> None:
    provider_bad_wrong_value = LocalResourceProvider("test", Flavor(
        metadata=FlavorMetadata(
            name="foo",
            owner_references={}
        ),
        spec=FlavorSpec(
            availability=True,
            flavor_type=FlavorTypeData(
                type_identifier=FlavorType.K8SLICE,
                type_data=FlavorK8SliceData(
                    characteristics=FlavorCharacteristics(cpu="1n", memory="1Gi", architecture="arm", gpu=GPUData(cores=0, memory="", model="")),
                    policies={},
                    properties={
                        "additionalProperties": {
                            "sensors": [
                                "CO2",
                                "Something else",
                                "Still not what we want",
                            ]
                        }
                    }
                )
            ),
            location={
                "city": "Dublin",
                "country": "Ireland",
            },
            network_property_type="",
            owner={},
            providerID="provider_id",
            price={}
        )
    ))
    provider_bad_no_value = LocalResourceProvider("test", Flavor(
        metadata=FlavorMetadata(
            name="foo",
            owner_references={}
        ),
        spec=FlavorSpec(
            availability=True,
            flavor_type=FlavorTypeData(
                type_identifier=FlavorType.K8SLICE,
                type_data=FlavorK8SliceData(
                    characteristics=FlavorCharacteristics(cpu="1n", memory="1Gi", architecture="arm", gpu=GPUData(cores=0, memory="", model="")),
                    policies={},
                    properties={
                        "additionalProperties": {
                        }
                    }
                )
            ),
            location={
                "city": "Dublin",
                "country": "Ireland",
            },
            network_property_type="",
            owner={},
            providerID="provider_id",
            price={}
        )
    ))
    provider_good = LocalResourceProvider("test", Flavor(
        metadata=FlavorMetadata(
            name="foo",
            owner_references={}
        ),
        spec=FlavorSpec(
            availability=True,
            flavor_type=FlavorTypeData(
                type_identifier=FlavorType.K8SLICE,
                type_data=FlavorK8SliceData(
                    characteristics=FlavorCharacteristics(cpu="1n", memory="1Gi", architecture="arm", gpu=GPUData(cores=0, memory="", model="")),
                    policies={},
                    properties={
                        "additionalProperties": {
                            "sensors": {
                                "humidity"
                            }
                        }
                    }
                )
            ),
            location={
                "city": "Dublin",
                "country": "Ireland",
            },
            network_property_type="",
            owner={},
            providerID="provider_id",
            price={}
        )
    ))

    intent = Intent(KnownIntent.sensor, "humidity")

    assert intent.validates(provider_good)
    assert not intent.validates(provider_bad_wrong_value)
    assert not intent.validates(provider_bad_no_value)


def test_validate_additional_hardware() -> None:
    provider_bad_wrong_value = LocalResourceProvider("test", Flavor(
        metadata=FlavorMetadata(
            name="foo",
            owner_references={}
        ),
        spec=FlavorSpec(
            availability=True,
            flavor_type=FlavorTypeData(
                type_identifier=FlavorType.K8SLICE,
                type_data=FlavorK8SliceData(
                    characteristics=FlavorCharacteristics(cpu="1n", memory="1Gi", architecture="arm", gpu=GPUData(cores=0, memory="", model="")),
                    policies={},
                    properties={
                        "additionalProperties": {
                            "additional-hardware": [
                                "actuator1",
                                "actuator2",
                                "sprinkler",
                                "icecream machine",
                            ]
                        }
                    }
                )
            ),
            location={
                "city": "Dublin",
                "country": "Ireland",
            },
            network_property_type="",
            owner={},
            providerID="provider_id",
            price={}
        )
    ))
    provider_bad_no_value = LocalResourceProvider("test", Flavor(
        metadata=FlavorMetadata(
            name="foo",
            owner_references={}
        ),
        spec=FlavorSpec(
            availability=True,
            flavor_type=FlavorTypeData(
                type_identifier=FlavorType.K8SLICE,
                type_data=FlavorK8SliceData(
                    characteristics=FlavorCharacteristics(cpu="1n", memory="1Gi", architecture="arm", gpu=GPUData(cores=0, memory="", model="")),
                    policies={},
                    properties={
                        "additionalProperties": {
                        }
                    }
                )
            ),
            location={
                "city": "Dublin",
                "country": "Ireland",
            },
            network_property_type="",
            owner={},
            providerID="provider_id",
            price={}
        )
    ))
    provider_good = LocalResourceProvider("test", Flavor(
        metadata=FlavorMetadata(
            name="foo",
            owner_references={}
        ),
        spec=FlavorSpec(
            availability=True,
            flavor_type=FlavorTypeData(
                type_identifier=FlavorType.K8SLICE,
                type_data=FlavorK8SliceData(
                    characteristics=FlavorCharacteristics(cpu="1n", memory="1Gi", architecture="arm", gpu=GPUData(cores=0, memory="", model="")),
                    policies={},
                    properties={
                        "additionalProperties": {
                            "additional-hardware": [
                                "actuator1",
                                "coffee machine",
                                "actuator2",
                                "sprinkler",
                                "icecream machine",
                                "coffee machine",
                            ]
                        }
                    }
                )
            ),
            location={
                "city": "Dublin",
                "country": "Ireland",
            },
            network_property_type="",
            owner={},
            providerID="provider_id",
            price={}
        )
    ))

    intent = Intent(KnownIntent.hardware, "coffee machine")

    assert intent.validates(provider_good)
    assert not intent.validates(provider_bad_wrong_value)
    assert not intent.validates(provider_bad_no_value)


def test_validate_latency_in_flavor() -> None:
    provider_bad_wrong_value = LocalResourceProvider("test", Flavor(
        metadata=FlavorMetadata(
            name="foo",
            owner_references={}
        ),
        spec=FlavorSpec(
            availability=True,
            flavor_type=FlavorTypeData(
                type_identifier=FlavorType.K8SLICE,
                type_data=FlavorK8SliceData(
                    characteristics=FlavorCharacteristics(cpu="1n", memory="1Gi", architecture="arm", gpu=GPUData(cores=0, memory="", model="")),
                    policies={},
                    properties={
                        "additionalProperties": {
                            "latency": "500"
                        }
                    }
                )
            ),
            location={
                "city": "Dublin",
                "country": "Ireland",
            },
            network_property_type="",
            owner={},
            providerID="provider_id",
            price={}
        )
    ))
    provider_good_no_value = LocalResourceProvider("test", Flavor(
        metadata=FlavorMetadata(
            name="foo",
            owner_references={}
        ),
        spec=FlavorSpec(
            availability=True,
            flavor_type=FlavorTypeData(
                type_identifier=FlavorType.K8SLICE,
                type_data=FlavorK8SliceData(
                    characteristics=FlavorCharacteristics(cpu="1n", memory="1Gi", architecture="arm", gpu=GPUData(cores=0, memory="", model="")),
                    policies={},
                    properties={
                        "additionalProperties": {
                        }
                    }
                )
            ),
            location={
                "city": "Dublin",
                "country": "Ireland",
            },
            network_property_type="",
            owner={},
            providerID="provider_id",
            price={}
        )
    ))
    provider_good = LocalResourceProvider("test", Flavor(
        metadata=FlavorMetadata(
            name="foo",
            owner_references={}
        ),
        spec=FlavorSpec(
            availability=True,
            flavor_type=FlavorTypeData(
                type_identifier=FlavorType.K8SLICE,
                type_data=FlavorK8SliceData(
                    characteristics=FlavorCharacteristics(cpu="1n", memory="1Gi", architecture="arm", gpu=GPUData(cores=0, memory="", model="")),
                    policies={},
                    properties={
                        "additionalProperties": {
                            "latency": "100"
                        }
                    }
                )
            ),
            location={
                "city": "Dublin",
                "country": "Ireland",
            },
            network_property_type="",
            owner={},
            providerID="provider_id",
            price={}
        )
    ))

    intent = Intent(KnownIntent.latency, "100")

    assert intent.validates(provider_good)
    assert intent.validates(provider_good_no_value)
    assert not intent.validates(provider_bad_wrong_value)
