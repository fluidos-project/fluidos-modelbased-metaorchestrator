from typing import Any

from kopf import Spec
import kopf

from .resources import ResourceProvider

from kubernetes import config
from kubernetes import client
from kubernetes import utils

from .common import configuration

import logging


logger = logging.getLogger()


def deploy(spec: Spec, provider: ResourceProvider) -> bool:
    spec_dict = {
        k: v for k, v in spec.items()
    }

    if not provider.acquire():
        return False

    enrich(spec_dict, provider)

    my_config = client.Configuration()
    config.load_config(client_configuration=my_config)
    k8s_client = client.ApiClient(my_config)

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
        nodeSelector[configuration.remote_node_key] = label
    else:
        nodeSelector[configuration.local_node_key] = "true"

    return True
