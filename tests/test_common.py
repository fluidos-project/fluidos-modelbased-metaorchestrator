from typing import Any

import yaml

from fluidos_model_orchestrator.common.contract import build_contract
from fluidos_model_orchestrator.common.flavor import build_flavor


def test_flavor_construction() -> None:
    example = """apiVersion: nodecore.fluidos.eu/v1alpha1
kind: Flavor
metadata:
  name: fluidos.eu-k8slice-2f1640fd4dfb151b4d81ad9590dcb6cc
  ownerReferences:
  - apiVersion: nodecore.fluidos.eu/v1alpha1
    kind: Node
    name: fluidos-consumer-1-worker
    uid: bcbc5b65-3de6-434a-aef9-a329317a189d
  resourceVersion: "1956"
  uid: 1dbd9361-a36b-4ee6-adff-958b5e0647bf
spec:
  availability: true
  flavorType:
    typeData:
      characteristics:
        architecture: amd64
        cpu: 1947481697n
        gpu:
          cores: "0"
          memory: "0"
          model: ""
        memory: 3735836Ki
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
    city: Turin
    country: Italy
    latitude: "10"
    longitude: "58"
  networkPropertyType: networkProperty
  owner:
    domain: fluidos.eu
    ip: 172.18.0.7:30000
    nodeID: ekvjnuvsel
  price:
    amount: ""
    currency: ""
    period: ""
  providerID: ekvjnuvsel
"""

    data: dict[str, Any] = yaml.safe_load(example)

    flavor = build_flavor(data)

    assert flavor is not None


def test_service_flavor_construction() -> None:
    example = """apiVersion: nodecore.fluidos.eu/v1alpha1
kind: Flavor
metadata:
  creationTimestamp: "2024-10-10T17:49:40Z"
  generation: 1
  name: provider.fluidos.eu-service-df0f
  namespace: fluidos
  ownerReferences:
  - apiVersion: nodecore.fluidos.eu/v1alpha1
    kind: ServiceBlueprint
    name: mq-rabbitmq
    uid: b1c9bf29-935b-4f61-af30-392f7d3c50b9
  resourceVersion: "4966"
  uid: e52f9585-a8e6-457e-8741-e50f759e34bb
spec:
  availability: true
  flavorType:
    typeData:
      category: message-queue
      configurationTemplate:
        $schema: http://json-schema.org/draft-07/schema#
        properties:
          password:
            type: string
          username:
            type: string
        required:
        - username
        - password
        type: object
      description: A message queue service blueprint
      hostingPolicies:
      - Consumer
      name: mq
      tags:
      - message-queue
      - rabbitmq
      - mqtt
    typeIdentifier: Service
  location:
    additionalNotes: None
    city: Turin
    country: Italy
    latitude: "10"
    longitude: "58"
  networkPropertyType: networkProperty
  owner:
    domain: provider.fluidos.eu
    ip: 10.89.0.54:30001
    nodeID: nx5w2u96xt
  price:
    amount: ""
    currency: ""
    period: ""
  providerID: nx5w2u96xt"""
    data: dict[str, Any] = yaml.safe_load(example)

    flavor = build_flavor(data)

    assert flavor is not None


def test_building_of_contract() -> None:
    data = """apiVersion: reservation.fluidos.eu/v1alpha1
kind: Contract
metadata:
  creationTimestamp: "2025-06-29T21:06:49Z"
  generation: 1
  name: contract-it-provider-fluidos-eu-k8slice-4e87-e756
  namespace: fluidos
  resourceVersion: "65608"
  uid: 91d6bc6e-86dd-4e88-8d0f-5e55b6c1a537
spec:
  buyer:
    additionalInformation:
      liqoID: quiet-mountain
    domain: consumer.fluidos.eu
    ip: 10.89.0.55:30000
    nodeID: rcaedor2qn
  buyerClusterID: quiet-mountain
  configuration:
    data:
      cpu: "1"
      memory: 1Gi
      pods: "110"
    type: K8Slice
  expirationTime: "2026-06-29T21:06:49Z"
  flavor:
    apiVersion: nodecore.fluidos.eu/v1alpha1
    kind: Flavor
    metadata:
      name: it.provider.fluidos.eu-k8slice-4e87
      namespace: fluidos
    spec:
      availability: true
      flavorType:
        typeData:
          characteristics:
            architecture: arm64
            cpu: 4894139596n
            gpu:
              cores: "0"
              memory: "0"
              model: ""
            memory: 18564928Ki
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
        city: Turin
        country: Italy
        latitude: "10"
        longitude: "58"
      networkPropertyType: networkProperty
      owner:
        domain: it.provider.fluidos.eu
        ip: 10.89.0.52:30001
        nodeID: yjxveoz3cc
      price:
        amount: ""
        currency: ""
        period: ""
      providerID: yjxveoz3cc
    status:
      creationTime: ""
      expirationTime: ""
      lastUpdateTime: ""
  peeringTargetCredentials:
    kubeconfig: YXBpVmVyc2lvbjogdjEKY2x1c3RlcnM6Ci0gY2x1c3RlcjoKICAgIGNlcnRpZmljYXRlLWF1dGhvcml0eS1kYXRhOiBMUzB0TFMxQ1JVZEpUaUJEUlZKVVNVWkpRMEZVUlMwdExTMHRDazFKU1VSQ1ZFTkRRV1V5WjBGM1NVSkJaMGxKUmxWRmIzTnlNSEZWSzJ0M1JGRlpTa3R2V2tsb2RtTk9RVkZGVEVKUlFYZEdWRVZVVFVKRlIwRXhWVVVLUVhoTlMyRXpWbWxhV0VwMVdsaFNiR042UVdWR2R6QjVUbFJCTWsxcVkzaE5lbFY2VFdwQ1lVWjNNSHBPVkVFeVRXcFZlRTE2VlRSTmFrSmhUVUpWZUFwRmVrRlNRbWRPVmtKQlRWUkRiWFF4V1cxV2VXSnRWakJhV0UxM1oyZEZhVTFCTUVkRFUzRkhVMGxpTTBSUlJVSkJVVlZCUVRSSlFrUjNRWGRuWjBWTENrRnZTVUpCVVVSUU1uTkNNMVZHWTJkSFZIUjJTRWhoTldwRGFFNVVjVWxoS3pkWlNGazBTWFJNUTJkbWJYRTVhMEV3YUVSTWFXeGtaemQyUTBoRVYxRUthMmhtWTB0aGJtTlVSbk5CT1hGa09HVlRPR2MxVjNkTmJsbFFhM2szV0hOSk1FRnNPRlpWTkRkRlpVRnZORVJhY21WbVJVOVNORWhUY0RGQ2FuZEpTd3BxYm10YVprNTJVVlppY1U5bmVFWjVhbWwxVDFSMlpGRm5ibXh4U1ZCaVlXTkpkVlkzY25OUk9FWnlRbk1yY2s1SVVFODROVXhITUVOa1pHVkROVU01Q21WV1R6RkhaVXh1ZDI0d1IzSnBTWEZqVVhGdmFrVllkMWhQVm5KRUx6SktjbEJZVURaQlJVMTRNVUZLZEZKWmJFbEVTek5yVEhoR1JXVlpiVmt2U21ZS1RDOVpWMng1TDFKcFRGbzFka1ZtVkZsYWEzZDFTRkJYYkhGMFV6Sm1kMlYwTURFMVFrTXdlVWx4Y0RkQ2RscDJTSFU1YVd0dE5tRmFSekV4WkhoSlF3cFFNRWxXZW1Gc1RtbFZhelZqZVROVGIwbFFTRE5WWW5neFNHUjZRV2ROUWtGQlIycFhWRUpZVFVFMFIwRXhWV1JFZDBWQ0wzZFJSVUYzU1VOd1JFRlFDa0puVGxaSVVrMUNRV1k0UlVKVVFVUkJVVWd2VFVJd1IwRXhWV1JFWjFGWFFrSlVhek12TmtSTE1qRnVWVWRQVjI1WFR6Sk5hVFI2YVRsSlRVcEVRVllLUW1kT1ZraFNSVVZFYWtGTloyZHdjbVJYU214amJUVnNaRWRXZWsxQk1FZERVM0ZIVTBsaU0wUlJSVUpEZDFWQlFUUkpRa0ZSUTJWa2VFUkdWWE5aVWdwRGJHdFZkMVpSUTBFNGFrZGxSVUlyZG1ST1VXSkdibTExUVhwdlJqbEVWVzVsUlhoQ2EyZFZlVFZ2Y0RFeGJuZEZlVzVtYzFkNVZFOTRjMWxCYmpodkNreFVSbk40VFdoNGMzUkJhMUI2VTFaaVZWZE1VekpTVDBkWlFrcDRSeTlPYVNzeVIwNVJTazgxYkZoMmFYcEljbUpwUlhadVYyNWpRMlIyZUVOS2NtOEtiWGhGUkdSTU5FUmtaVTVNSzNjd01uZEpRamQwVlZGRlpXWnJVSFZaSzBzMlNqbHlRWEUzYVhWMVZtZzBXVlpXVFhBeVVrTTNSalJ6ZW5GWVlXNVplZ294WlhRMFZHNVZOR3RvVlVjelNpdGFaSGhIZFhRd05rcEdUVE52Tmk5eFNYaFlPVGhGV21KdmRWRnBUalpRVFd3M2QxRnZja2xJY21kVmIxQjVVMVI2Q2xVcmQzbDFUSGhWTlhGTmRYVXhOeTh5YkVVcldTdHBlVGx1YW0xQlIyTm9SRWc0ZERsNlNqazFWMFI2UVRBelVuUmFaa28yVkd0amJsTTVabkZOVnprS05XMVFWbTVKU1V4aGVqZHdDaTB0TFMwdFJVNUVJRU5GVWxSSlJrbERRVlJGTFMwdExTMEsKICAgIHNlcnZlcjogaHR0cHM6Ly8xMC44OS4wLjUyOjY0NDMKICBuYW1lOiBtdWRkeS1oYXplCmNvbnRleHRzOgotIGNvbnRleHQ6CiAgICBjbHVzdGVyOiBtdWRkeS1oYXplCiAgICB1c2VyOiBtdWRkeS1oYXplCiAgbmFtZTogbXVkZHktaGF6ZQpjdXJyZW50LWNvbnRleHQ6IG11ZGR5LWhhemUKa2luZDogQ29uZmlnCnByZWZlcmVuY2VzOiB7fQp1c2VyczoKLSBuYW1lOiBtdWRkeS1oYXplCiAgdXNlcjoKICAgIHRva2VuOiBleUpoYkdjaU9pSlNVekkxTmlJc0ltdHBaQ0k2SW5vME5HTkpUV1ZMWVZGeWRGbG9VMlJ3VldwRFNFczJTVk54U1cxSlgxOHpYMFJzWm5CQmFVZzVVMVVpZlEuZXlKcGMzTWlPaUpyZFdKbGNtNWxkR1Z6TDNObGNuWnBZMlZoWTJOdmRXNTBJaXdpYTNWaVpYSnVaWFJsY3k1cGJ5OXpaWEoyYVdObFlXTmpiM1Z1ZEM5dVlXMWxjM0JoWTJVaU9pSnNhWEZ2SWl3aWEzVmlaWEp1WlhSbGN5NXBieTl6WlhKMmFXTmxZV05qYjNWdWRDOXpaV055WlhRdWJtRnRaU0k2SW14cGNXOHRZMngxYzNSbGNpMXRkV1JrZVMxb1lYcGxJaXdpYTNWaVpYSnVaWFJsY3k1cGJ5OXpaWEoyYVdObFlXTmpiM1Z1ZEM5elpYSjJhV05sTFdGalkyOTFiblF1Ym1GdFpTSTZJbXhwY1c4dFkyeDFjM1JsY2kxdGRXUmtlUzFvWVhwbElpd2lhM1ZpWlhKdVpYUmxjeTVwYnk5elpYSjJhV05sWVdOamIzVnVkQzl6WlhKMmFXTmxMV0ZqWTI5MWJuUXVkV2xrSWpvaVlqWTBORGd3WVdNdE5qVTBNaTAwT0dVMExXRTRPV1F0TnpOa09EUXdNV1UwWW1Oa0lpd2ljM1ZpSWpvaWMzbHpkR1Z0T25ObGNuWnBZMlZoWTJOdmRXNTBPbXhwY1c4NmJHbHhieTFqYkhWemRHVnlMVzExWkdSNUxXaGhlbVVpZlEucHNJUDB5OEctOTBTWG03MHJpdFBDbDJXMzBSMmlLQzJpa0h1dW01dTFnaF9LTDZJUno4cFpaM3BKLWdjMmQwQkRNdFFvSEhUZnpXY1dEM19SX2RmYmVlQlIzNXB2MW9ScHRUZTFPOEdDUXowamJ6SDFjcmhwNzF4Z2VMZ2JKR3ZVU2ItODlXN3ppd3NmQm1neGlPcTNER2pXdVhpWHhtdlZCZlNqUU1BQXY5YmxYR2dUSkhRNS1aTjBUcUtBOVdqQXNmV3J6UEt4YWRFbGZFYmllNUlHQmh2TkFMLTFVWVBIRmhjdlExQU9KR1k0NWVBSjBCWXJBM3ZpaXpGWmRCNjZOWFR6cDZkQkhpcmNuVjA1N1dpX3ZwQ2l3N0RVQ2VEX0pUVVk4SmMxck9yRlRzU20tWTY4YjV6bmFsWUJkdDcwaV9TampiS2hxYms2TFBTYzhXMm1nCg==
    liqoID: muddy-haze
  seller:
    additionalInformation:
      liqoID: muddy-haze
    domain: it.provider.fluidos.eu
    ip: 10.89.0.52:30001
    nodeID: yjxveoz3cc
  transactionID: 0cfdf309521482c414e938a12bfe1d25-1751231198522898934
"""
    contract = build_contract(yaml.safe_load(data)["spec"])

    assert contract
