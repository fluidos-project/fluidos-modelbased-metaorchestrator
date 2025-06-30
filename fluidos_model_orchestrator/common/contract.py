from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from fluidos_model_orchestrator.common.flavor import build_flavor
from fluidos_model_orchestrator.common.flavor import Flavor


@dataclass(kw_only=True)
class PeeringTargetCredentials:
    kubeconfig: str
    liqoID: str


@dataclass(kw_only=True)
class Contract:
    buyer: dict[str, Any]
    buyerClusterID: str
    configuration: dict[str, Any]
    expirationTime: datetime
    flavor: Flavor
    peeringTargetCredentials: PeeringTargetCredentials
    seller: dict[str, Any]
    transactionID: str


def build_contract(spec: dict[str, Any]) -> Contract:
    return Contract(
        buyer=spec["buyer"],
        buyerClusterID=spec["buyerClusterID"],
        configuration=spec["configuration"],
        expirationTime=spec["expirationTime"],
        flavor=build_flavor(spec["flavor"]),
        peeringTargetCredentials=PeeringTargetCredentials(
            kubeconfig=spec["peeringTargetCredentials"]["kubeconfig"],
            liqoID=spec["peeringTargetCredentials"]["liqoID"],
        ),
        seller=spec["seller"],
        transactionID=spec["transactionID"],
    )
