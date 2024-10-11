import asyncio
import logging
from typing import Any

import kopf  # type: ignore
from kubernetes.utils import create_from_dict  # type: ignore

from .common import Intent
from .common import ModelPredictResponse
from .common import ResourceProvider
from .common import ServiceResourceProvider
from .configuration import CONFIGURATION


logger = logging.getLogger()


def expand(spec: dict[str, Any], expanding: tuple[ServiceResourceProvider, Intent]) -> bool:
    # retrieve credentials as secret named f"credentials-{contact_name}"
    # stored in namespaced called f"{contact_name}"
    # to be decoded

    return True


async def deploy(
        spec: dict[str, Any],
        provider: ResourceProvider,
        expanding_resources: list[tuple[ServiceResourceProvider, Intent]],
        response: ModelPredictResponse,
        namespace: str) -> bool:
    spec_dict = {
        k: v for k, v in spec.items()
    }

    enrich(spec_dict, provider)

    for expanding_resource in expanding_resources:
        expand(spec_dict, expanding_resource)

    delay_time = response.delay * 60 * 60
    logger.info(f"Waiting to deploy {delay_time=}")
    await asyncio.sleep(delay_time)

    k8s_client = CONFIGURATION.k8s_client

    kopf.adopt(spec_dict)

    reference = create_from_dict(k8s_client=k8s_client, data=spec_dict, namespace=namespace)

    return reference is not None


def enrich(spec: dict[str, Any], provider: ResourceProvider) -> bool:
    logger.info("Enriching workload manifest with resource provider information")

    labels = provider.get_label()

    if not len(labels):
        logger.info("No label provided, deploying randomly")
        return True

    nodeSelector: dict[str, str] = _get_node_selector(spec)

    nodeSelector.update(labels)

    # if label:
    #     nodeSelector[CONFIGURATION.remote_node_key] = label
    # else:
    #     nodeSelector[CONFIGURATION.local_node_key] = "true"

    return True


def _get_node_selector(spec: dict[str, Any]) -> dict[str, str]:
    if spec["kind"] == "Deployment":
        logger.debug("Workload manifest of type Deployment")
        spec = spec.get("spec", {}).get("template", {})
    elif spec["kind"] == "Pod":
        logger.debug("Workload manifest of type Pod")
    else:
        logger.warning(f"Unsupported manifest type: {spec['kind']}")
        raise ValueError(f"Unsupported manifest kind: {spec['kind']}")

    if "nodeSelector" not in spec["spec"]:
        spec["spec"]["nodeSelector"] = {}

    return spec["spec"]["nodeSelector"]
