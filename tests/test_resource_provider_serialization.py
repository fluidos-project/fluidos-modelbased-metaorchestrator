import yaml

from fluidos_model_orchestrator.common.flavor import build_flavor
from fluidos_model_orchestrator.resources.rear.local_resource_provider import LocalResourceProvider
from fluidos_model_orchestrator.resources.rear.remote_resource_provider import RemoteResourceProvider


EXAMPLE_FLAVOR = """apiVersion: nodecore.fluidos.eu/v1alpha1
kind: Flavor
metadata:
  creationTimestamp: "2025-07-31T21:43:09Z"
  generation: 2
  name: consumer.fluidos.eu-k8slice-edff
  namespace: fluidos
  ownerReferences:
    - apiVersion: nodecore.fluidos.eu/v1alpha1
      kind: Node
      name: consumer-worker
      uid: b9b92efb-3542-4cd3-9601-3bea2779bc11
  resourceVersion: "1776"
  uid: c4604f47-e12f-4c58-8548-7488ce2c156c
spec:
  availability: true
  flavorType:
    typeData:
      characteristics:
        architecture: arm64
        cpu: 4334709917n
        gpu:
          cores: "0"
          memory: "0"
          model: ""
        memory: 18586800Ki
        pods: "110"
        storage: "0"
      policies:
        partitionability:
          cpuMin: "0"
          cpuStep: "1"
          gpuMin: "0"
          gpuStep: "0"
          memoryMin: "0"
          memoryStep: 100Mi
          podsMin: "0"
          podsStep: "0"
      properties: {}
    typeIdentifier: K8Slice
  location:
    additionalNotes: None
    city: Dublin
    country: Ireland
    latitude: "53.3498"
    longitude: "6.2603"
  networkPropertyType: networkProperty
  owner:
    domain: consumer.fluidos.eu
    ip: 10.89.0.47:30000
    nodeID: u8vftaal4l
  price:
    amount: ""
    currency: ""
    period: ""
  providerID: u8vftaal4l
"""


def test_flavor_serialization() -> None:
    flavor = build_flavor(yaml.safe_load(EXAMPLE_FLAVOR))

    json_version = flavor.to_json()

    assert json_version


def test_local_resource_provider() -> None:
    flavor = build_flavor(yaml.safe_load(EXAMPLE_FLAVOR))

    provider = LocalResourceProvider("foo-bar", flavor)

    assert provider

    assert provider.to_json()


def test_remote_resource_provider() -> None:
    flavor = build_flavor(yaml.safe_load(EXAMPLE_FLAVOR))

    provider = RemoteResourceProvider("foo-bar", flavor, "peering_candidate", "reserviation", None, {})

    assert provider

    provider.remote_cluster_id = "remote_cluster_id"

    assert provider.to_json()
