import logging
from dataclasses import dataclass
from dataclasses import field
from typing import Any

import kopf  # type: ignore
from kubernetes import client  # type: ignore
from kubernetes.client import CoreV1Api  # type: ignore
from kubernetes.client import V1ConfigMapList
from kubernetes.client.exceptions import ApiException  # type: ignore


@dataclass
class Configuration:
    local_node_key: str = "fluidos.eu/resource-node"
    remote_node_key: str = "liqo.io/remote-cluster-id"
    namespace: str = "fluidos"
    k8s_client: client.ApiClient | None = None
    identity: dict[str, str] = field(default_factory=dict)
    api_keys: dict[str, str] = field(default_factory=dict)
    DAEMON_SLEEP_TIME: float = 60. * 60.  # 1h in seconds

    def check_identity(self, identity: dict[str, str]) -> bool:
        return all(
            self.identity[key] == identity.get(key, "")
            for key in self.identity.keys()
        )


def enrich_configuration(config: Configuration,
                         settings: kopf.OperatorSettings,
                         param: Any,
                         memo: Any,
                         kwargs: dict[str, Any],
                         logger: logging.Logger,
                         my_config: client.Configuration) -> None:
    logger.info("Enrich default configuration with user provided information")

    config.k8s_client = _build_k8s_client(my_config)
    config.identity = _retrieve_node_identity(config, logger)
    config.api_keys = _retrieve_api_key(config, logger)


def _retrieve_api_key(config: Configuration, logger: logging.Logger) -> dict[str, str]:
    logger.info("Retrieving API KEYS from config map")
    api_endpoint = CoreV1Api(config.k8s_client)

    try:
        config_maps: V1ConfigMapList = api_endpoint.list_config_map_for_all_namespaces()
        if len(config_maps.items):
            for item in config_maps.items:
                if item.metadata.name == "fluidos-mbmo-configmap":
                    logger.info("ConfigMap identified")
                    data: dict[str, str] = item.data
                    return {
                        key: value
                        for key, value in data.items() if key.endswith("_API_KEY")
                    }
    except ApiException as e:
        logger.error(f"Unable to retrieve config map {e=}")

    logger.error("Something went wrong while retrieving config map")
    raise ValueError("Unable to retrieve config map")


def _build_k8s_client(config: client.Configuration) -> client.ApiClient:
    return client.ApiClient(config)


def _retrieve_node_identity(config: Configuration, logger: logging.Logger) -> dict[str, str]:
    logger.info("Retrieving node id from config map, or generate a new one if not existing (aka debug mode)")
    api_endpoint = CoreV1Api(config.k8s_client)

    try:
        config_maps: V1ConfigMapList = api_endpoint.list_config_map_for_all_namespaces()
        if len(config_maps.items):
            for item in config_maps.items:
                if item.metadata.name == "fluidos-network-manager-identity":
                    logger.info("ConfigMap identified")
                    logger.debug(f"Returning {item.data}")
                    return item.data

    except ApiException as e:
        logger.error(f"Unable to retrieve configmap {e=}")

    logger.error("Something went wrong while retrieving node identity")
    raise ValueError("Unable to retrieve node identity")


CONFIGURATION = Configuration()
