# mypy: ignore-errors
# flake8: noqa
from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AlgorithmName(_message.Message):
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class ResourceQuantity(_message.Message):
    __slots__ = ("value", "format")
    VALUE_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    value: str
    format: str
    def __init__(self, value: _Optional[str] = ..., format: _Optional[str] = ...) -> None: ...

class Data(_message.Message):
    __slots__ = ("workload", "infrastructure")
    WORKLOAD_FIELD_NUMBER: _ClassVar[int]
    INFRASTRUCTURE_FIELD_NUMBER: _ClassVar[int]
    workload: Workload
    infrastructure: Infrastructure
    def __init__(self, workload: _Optional[_Union[Workload, _Mapping]] = ..., infrastructure: _Optional[_Union[Infrastructure, _Mapping]] = ...) -> None: ...

class Infrastructure(_message.Message):
    __slots__ = ("regions", "links")
    REGIONS_FIELD_NUMBER: _ClassVar[int]
    LINKS_FIELD_NUMBER: _ClassVar[int]
    regions: _containers.RepeatedCompositeFieldContainer[Region]
    links: _containers.RepeatedCompositeFieldContainer[Link]
    def __init__(self, regions: _Optional[_Iterable[_Union[Region, _Mapping]]] = ..., links: _Optional[_Iterable[_Union[Link, _Mapping]]] = ...) -> None: ...

class Region(_message.Message):
    __slots__ = ("id", "location", "resource_cost", "nodes")
    ID_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_COST_FIELD_NUMBER: _ClassVar[int]
    NODES_FIELD_NUMBER: _ClassVar[int]
    id: str
    location: str
    resource_cost: int
    nodes: _containers.RepeatedCompositeFieldContainer[Node]
    def __init__(self, id: _Optional[str] = ..., location: _Optional[str] = ..., resource_cost: _Optional[int] = ..., nodes: _Optional[_Iterable[_Union[Node, _Mapping]]] = ...) -> None: ...

class Node(_message.Message):
    __slots__ = ("name", "cpu_used", "mem_used", "cpu_cap", "mem_cap")
    NAME_FIELD_NUMBER: _ClassVar[int]
    CPU_USED_FIELD_NUMBER: _ClassVar[int]
    MEM_USED_FIELD_NUMBER: _ClassVar[int]
    CPU_CAP_FIELD_NUMBER: _ClassVar[int]
    MEM_CAP_FIELD_NUMBER: _ClassVar[int]
    name: str
    cpu_used: ResourceQuantity
    mem_used: ResourceQuantity
    cpu_cap: ResourceQuantity
    mem_cap: ResourceQuantity
    def __init__(self, name: _Optional[str] = ..., cpu_used: _Optional[_Union[ResourceQuantity, _Mapping]] = ..., mem_used: _Optional[_Union[ResourceQuantity, _Mapping]] = ..., cpu_cap: _Optional[_Union[ResourceQuantity, _Mapping]] = ..., mem_cap: _Optional[_Union[ResourceQuantity, _Mapping]] = ...) -> None: ...

class Link(_message.Message):
    __slots__ = ("id", "endpoint_a", "endpoint_b", "bandwidth", "latency", "bandwidth_used")
    ID_FIELD_NUMBER: _ClassVar[int]
    ENDPOINT_A_FIELD_NUMBER: _ClassVar[int]
    ENDPOINT_B_FIELD_NUMBER: _ClassVar[int]
    BANDWIDTH_FIELD_NUMBER: _ClassVar[int]
    LATENCY_FIELD_NUMBER: _ClassVar[int]
    BANDWIDTH_USED_FIELD_NUMBER: _ClassVar[int]
    id: str
    endpoint_a: str
    endpoint_b: str
    bandwidth: ResourceQuantity
    latency: ResourceQuantity
    bandwidth_used: ResourceQuantity
    def __init__(self, id: _Optional[str] = ..., endpoint_a: _Optional[str] = ..., endpoint_b: _Optional[str] = ..., bandwidth: _Optional[_Union[ResourceQuantity, _Mapping]] = ..., latency: _Optional[_Union[ResourceQuantity, _Mapping]] = ..., bandwidth_used: _Optional[_Union[ResourceQuantity, _Mapping]] = ...) -> None: ...

class Workload(_message.Message):
    __slots__ = ("microservices", "data_flows", "placements")
    MICROSERVICES_FIELD_NUMBER: _ClassVar[int]
    DATA_FLOWS_FIELD_NUMBER: _ClassVar[int]
    PLACEMENTS_FIELD_NUMBER: _ClassVar[int]
    microservices: _containers.RepeatedCompositeFieldContainer[Microservice]
    data_flows: _containers.RepeatedCompositeFieldContainer[DataFlow]
    placements: _containers.RepeatedCompositeFieldContainer[Placement]
    def __init__(self, microservices: _Optional[_Iterable[_Union[Microservice, _Mapping]]] = ..., data_flows: _Optional[_Iterable[_Union[DataFlow, _Mapping]]] = ..., placements: _Optional[_Iterable[_Union[Placement, _Mapping]]] = ...) -> None: ...

class Microservice(_message.Message):
    __slots__ = ("name", "replicas", "cpu_required", "mem_required")
    NAME_FIELD_NUMBER: _ClassVar[int]
    REPLICAS_FIELD_NUMBER: _ClassVar[int]
    CPU_REQUIRED_FIELD_NUMBER: _ClassVar[int]
    MEM_REQUIRED_FIELD_NUMBER: _ClassVar[int]
    name: str
    replicas: int
    cpu_required: ResourceQuantity
    mem_required: ResourceQuantity
    def __init__(self, name: _Optional[str] = ..., replicas: _Optional[int] = ..., cpu_required: _Optional[_Union[ResourceQuantity, _Mapping]] = ..., mem_required: _Optional[_Union[ResourceQuantity, _Mapping]] = ...) -> None: ...

class DataFlow(_message.Message):
    __slots__ = ("name", "bandwidth_required", "latency_required", "vertices")
    NAME_FIELD_NUMBER: _ClassVar[int]
    BANDWIDTH_REQUIRED_FIELD_NUMBER: _ClassVar[int]
    LATENCY_REQUIRED_FIELD_NUMBER: _ClassVar[int]
    VERTICES_FIELD_NUMBER: _ClassVar[int]
    name: str
    bandwidth_required: ResourceQuantity
    latency_required: ResourceQuantity
    vertices: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, name: _Optional[str] = ..., bandwidth_required: _Optional[_Union[ResourceQuantity, _Mapping]] = ..., latency_required: _Optional[_Union[ResourceQuantity, _Mapping]] = ..., vertices: _Optional[_Iterable[str]] = ...) -> None: ...

class Placement(_message.Message):
    __slots__ = ("microservice_name", "order", "replica_scores", "node_selected")
    MICROSERVICE_NAME_FIELD_NUMBER: _ClassVar[int]
    ORDER_FIELD_NUMBER: _ClassVar[int]
    REPLICA_SCORES_FIELD_NUMBER: _ClassVar[int]
    NODE_SELECTED_FIELD_NUMBER: _ClassVar[int]
    microservice_name: str
    order: int
    replica_scores: _containers.RepeatedCompositeFieldContainer[ReplicaScores]
    node_selected: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, microservice_name: _Optional[str] = ..., order: _Optional[int] = ..., replica_scores: _Optional[_Iterable[_Union[ReplicaScores, _Mapping]]] = ..., node_selected: _Optional[_Iterable[str]] = ...) -> None: ...

class ReplicaScores(_message.Message):
    __slots__ = ("scores",)
    SCORES_FIELD_NUMBER: _ClassVar[int]
    scores: _containers.RepeatedCompositeFieldContainer[Score]
    def __init__(self, scores: _Optional[_Iterable[_Union[Score, _Mapping]]] = ...) -> None: ...

class Score(_message.Message):
    __slots__ = ("node", "score")
    NODE_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    node: str
    score: int
    def __init__(self, node: _Optional[str] = ..., score: _Optional[int] = ...) -> None: ...
