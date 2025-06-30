from logging import Logger
from typing import Any
from typing import cast

import kopf  # type: ignore

from fluidos_model_orchestrator.common.contract import build_contract
from fluidos_model_orchestrator.common.flavor import FlavorK8SliceData
from fluidos_model_orchestrator.common.flavor import FlavorType
from fluidos_model_orchestrator.configuration import CONFIGURATION


@kopf.on.create("contracts")  # type: ignore
async def monitor_contracts(spec: dict[str, Any], name: str, namespace: str, logger: Logger, errors: kopf.ErrorsMode = kopf.ErrorsMode.PERMANENT, **kwargs: str) -> Any:
    logger.info("Monitoring for contracts being created")
    if not CONFIGURATION.monitor_contracts:
        logger.info("But stopping as not requested")
        return

    # check if we are the provider in this contract
    logger.info("Building contract object")
    contract = build_contract(spec)

    if not CONFIGURATION.check_identity(contract.seller):
        logger.info("Ignoring this one as we are not the seller")
        return

    flavor = contract.flavor

    if flavor.spec.flavor_type.type_identifier is not FlavorType.K8SLICE:
        logger.info("Ignoring as not about Kubernetes Slice")
        return

    flavor_data = cast(FlavorK8SliceData, flavor.spec.flavor_type.type_data)

    if not flavor_data.properties.get("additionalProperties", {}).get("require-instantiation", False):
        logger.info("Ignoring, this flavor does not require node creation")
        return

    vm_type = flavor_data.properties.get("additionalProperties", {}).get("vm-type", CONFIGURATION.default_vm_type)

    # DO YOUR MAGIC
    logger.info(f"Creating node for {vm_type}")
