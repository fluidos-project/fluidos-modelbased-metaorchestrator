from typing import cast

import yaml

from fluidos_model_orchestrator.common.flavor import build_flavor
from fluidos_model_orchestrator.common.flavor import FlavorK8SliceData


def test_flavor_deserialization() -> None:
    example = """apiVersion: nodecore.fluidos.eu/v1alpha1
kind: Flavor
metadata:
  creationTimestamp: "2025-07-21T09:03:56Z"
  generation: 3
  name: consumer.fluidos.eu-k8slice-b57f
  namespace: fluidos
  ownerReferences:
  - apiVersion: nodecore.fluidos.eu/v1alpha1
    kind: Node
    name: consumer-worker
    uid: 9bab0ef6-8b9e-404d-b1e3-76f1019706fe
  resourceVersion: "91464"
  uid: 7120a20a-730f-469e-83c0-654849d9ce24
spec:
  availability: true
  flavorType:
    typeData:
      characteristics:
        architecture: arm64
        cpu: 4395468769n
        gpu:
          cores: "0"
          memory: "0"
          model: ""
        memory: 18645328Ki
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
      properties:
        additionalProperties:
          hardware:
          - coffee machine
          - a more proper actuator
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
    ip: 10.89.0.21:30000
    nodeID: ps3nqbieve
  price:
    amount: ""
    currency: ""
    period: ""
  providerID: ps3nqbieve
"""
    flavor = build_flavor(yaml.safe_load(example))

    assert flavor

    assert cast(FlavorK8SliceData, flavor.spec.flavor_type.type_data).properties["additionalProperties"]
