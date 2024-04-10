from typing import Any

import kopf

from .resources import ResourceProvider

from kubernetes import utils

from .common import CONFIGURATION, Intent

import logging


logger = logging.getLogger()


def deploy(spec: dict[str, Any], provider: ResourceProvider, expanding_resources: list[tuple[ResourceProvider, Intent]]) -> bool:
    spec_dict = {
        k: v for k, v in spec.items()
    }

    if not provider.acquire():
        return False

    enrich(spec_dict, provider)

    k8s_client = CONFIGURATION.k8s_client

    kopf.adopt(spec_dict)

    reference = utils.create_from_dict(k8s_client=k8s_client, data=spec_dict)

    return reference is not None


def enrich(spec: dict[str, Any], provider: ResourceProvider) -> bool:
    logger.info("Enriching workload manifest with resource provider information")

    label = provider.get_label()

    nodeSelector: dict[str, Any]

    if spec["kind"] == "Deployment":
        logger.debug("Workload manifest of type Deployment")
        if "nodeSelector" not in spec["spec"]["template"]["spec"]:
            spec["spec"]["template"]["spec"]["nodeSelector"] = {}
        nodeSelector = spec["spec"]["template"]["spec"]["nodeSelector"]

    elif spec["kind"] == "Pod":
        logger.debug("Workload manifest of type Pod")
        if "nodeSelector" not in spec["spec"]:
            spec["spec"]["nodeSelector"] = {}
        nodeSelector = spec["spec"]["nodeSelector"]
    else:
        logger.warning(f"Unsupported manifest type: {spec['kind']}")
        raise ValueError(f"Unsupported manifest kind: {spec['kind']}")

    if label:
        nodeSelector[CONFIGURATION.remote_node_key] = label
    else:
        nodeSelector[CONFIGURATION.local_node_key] = "true"

    return True
