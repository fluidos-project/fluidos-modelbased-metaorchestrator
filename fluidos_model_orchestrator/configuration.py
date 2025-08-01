import logging
from dataclasses import dataclass
from dataclasses import field
from typing import Any

import kopf  # type: ignore
from kubernetes.client import CoreV1Api  # type: ignore
from kubernetes.client import V1ConfigMapList  # type: ignore
from kubernetes.client import V1SecretList  # type: ignore
from kubernetes.client.api_client import ApiClient  # type: ignore
from kubernetes.client.configuration import Configuration as K8SConfig  # type: ignore
from kubernetes.client.exceptions import ApiException  # type: ignore


logger = logging.getLogger(__name__)


@dataclass
class Configuration:
    local_node_key: str = "node-role.fluidos.eu/resources"
    remote_node_key: str = "liqo.io/remote-cluster-id"
    namespace: str = "fluidos"
    k8s_client: ApiClient | None = None
    identity: dict[str, str] = field(default_factory=dict)
    api_keys: dict[str, str] = field(default_factory=dict)
    DAEMON_SLEEP_TIME: float = 60. * 60.  # 1h in seconds
    UPDATE_FLAVORS: bool = True
    FLAVOR_UPDATE_SLEEP_TIME: float = 60. * 60.  # 1h in seconds
    architecture: str = "amd64"
    n_try: int = 25
    API_SLEEP_TIME: float = 0.1  # 100 ms
    SOLVER_SLEEPING_TIME: float = .5  # 500ms
    MSPL_ENDPOINT: str = ""
    monitor_enabled: bool = False
    local_prometheus: str = "localhost:9090"  # TODO: should be loaded from configuration
    MONITOR_SLEEP_TIME: float = 5.  # 5 seconds

    monitor_contracts: bool = False
    default_vm_type: str = "default-vm-type"

    def check_identity(self, identity: dict[str, str]) -> bool:
        return all([
            identity["domain"] == self.identity["domain"],
            identity["ip"] == self.identity["ip"] or identity["ip"] == f"{self.identity['ip']}:{self.identity.get('port', 3000)}",
            identity["nodeID"] == self.identity["nodeID"]
        ])


def enrich_configuration(config: Configuration,
                         settings: kopf.OperatorSettings,
                         param: Any,
                         memo: Any,
                         kwargs: dict[str, Any],
                         logger: logging.Logger,
                         my_config: K8SConfig) -> None:
    logger.info("Enrich default configuration with user provided information")

    config.k8s_client = _build_k8s_client(my_config)
    config.identity = _retrieve_node_identity(config, logger)
    config.architecture = _retrieve_architecture(config, logger)
    config.MSPL_ENDPOINT = _retrieve_mspl_endpoint(config, logger)
    config.UPDATE_FLAVORS, config.FLAVOR_UPDATE_SLEEP_TIME = _retrieve_update_flavor(config, logger)
    config.api_keys = _retrieve_api_key(config, logger)
    config.monitor_contracts = _retrieve_monitoring_contracts(config, logger)
    config.default_vm_type = _retrieve_default_vm_type(config, logger)
    config.monitor_enabled, config.MONITOR_SLEEP_TIME, config.local_prometheus = _retrieve_monitor_information(config, logger)


def _retrieve_monitor_information(config: Configuration, logger: logging.Logger) -> tuple[bool, float, str]:
    logger.info("Retrieve monitor fluidosdeployments from config map")
    api_endpoint = CoreV1Api(config.k8s_client)

    try:
        config_maps: V1ConfigMapList = api_endpoint.list_namespaced_config_map(config.namespace)
        if len(config_maps.items):
            for item in config_maps.items:
                if item.metadata is None:
                    continue

                if item.metadata.name == "fluidos-mbmo-configmap":
                    logger.debug("ConfigMap identified")
                    if item.data is None:
                        raise ValueError("ConfigMap data missing.")

                    data: dict[str, str] = item.data
                    return (
                        data.get("monitor_enabled", "False").casefold() == "True".casefold(),  # disable by default
                        float(data.get("monitor_interval", 5.)),  # 5 seconds
                        str(data.get("prometheus_endpoint", "localhost:9090"))
                    )
    except ApiException as e:
        logger.error(f"Unable to retrieve config map {e=}")

    logger.error("Something went wrong while retrieving config map")
    raise ValueError("Unable to retrieve config map")


def _retrieve_update_flavor(config: Configuration, logger: logging.Logger) -> tuple[bool, float]:
    logger.info("Retrieving update flavors from config map")
    api_endpoint = CoreV1Api(config.k8s_client)

    try:
        config_maps: V1ConfigMapList = api_endpoint.list_namespaced_config_map(config.namespace)
        if len(config_maps.items):
            for item in config_maps.items:
                if item.metadata is None:
                    continue

                if item.metadata.name == "fluidos-mbmo-configmap":
                    logger.debug("ConfigMap identified")
                    if item.data is None:
                        raise ValueError("ConfigMap data missing.")

                    data: dict[str, str] = item.data
                    return (
                        data.get("UPDATE_FLAVORS", "False").casefold() == "True".casefold(),  # disable by default
                        float(data.get("FLAVOR_UPDATE_SLEEP_TIME", 60. * 60.))  # 1h in seconds
                    )
    except ApiException as e:
        logger.error(f"Unable to retrieve config map {e=}")

    logger.error("Something went wrong while retrieving config map")
    raise ValueError("Unable to retrieve config map")


def _retrieve_api_key_from_secret(config: Configuration, logger: logging.Logger) -> dict[str, str]:
    logger.info("Retrieving API KEYS from secret")
    api_endpoint = CoreV1Api(config.k8s_client)

    try:
        secrets: V1SecretList = api_endpoint.list_secret_for_all_namespaces()
        if len(secrets.items):
            for secret in secrets.items:
                if secret.metadata is None:
                    continue

                if secret.metadata.name == "electricity-map":
                    logger.info("Secret identified")
                    if secret.data is None:
                        raise ValueError("Secret data missing.")

                    data: dict[str, str] = secret.data
                    return {
                        "ELECTRICITY_MAP_API_KEY": data.get("KEY", "PLACEHOLDER_NOT_VALID")
                    }
    except ApiException as e:
        logger.error(f"Unable to retrieve secrets {e=}")

    logger.error("Something went wrong while retrieving secrets")
    raise ValueError("Unable to retrieve secrets")


def _retrieve_api_key(config: Configuration, logger: logging.Logger) -> dict[str, str]:
    logger.info("Retrieving API KEYS from config map")
    api_endpoint = CoreV1Api(config.k8s_client)

    try:
        config_maps: V1ConfigMapList = api_endpoint.list_config_map_for_all_namespaces()
        if len(config_maps.items):
            for config_map in config_maps.items:
                if config_map.metadata is None:
                    continue

                if config_map.metadata.name == "fluidos-mbmo-configmap":
                    logger.debug("ConfigMap identified")
                    if config_map.data is None:
                        logger.error("Unable to retrieve API Keys. ConfigMap data missing.")
                        raise ValueError("Unable to retrieve API Keys. ConfigMap data missing.")

                    data: dict[str, str] = config_map.data
                    return {
                        key: value
                        for key, value in data.items() if key.endswith("_API_KEY")
                    }
    except ApiException as e:
        logger.error(f"Unable to retrieve config map {e=}")

    logger.error("Something went wrong while retrieving config map")
    raise ValueError("Unable to retrieve config map")


def _build_k8s_client(config: K8SConfig) -> ApiClient:
    return ApiClient(config)


def _retrieve_mspl_endpoint(config: Configuration, logger: logging.Logger) -> str:
    logger.info("Retrieving MSPL endpoint information from config map")
    api_endpoint = CoreV1Api(config.k8s_client)

    try:
        config_maps: V1ConfigMapList = api_endpoint.list_namespaced_config_map(config.namespace)
        if len(config_maps.items):
            for item in config_maps.items:
                if item.metadata is None:
                    continue

                if item.metadata.name == "fluidos-mbmo-configmap":
                    logger.debug("ConfigMap identified")
                    if item.data is None:
                        logger.error("Unable to retrieve MSPL endpoint. ConfigMap data missing.")
                        raise ValueError("Unable to retrieve MSPL endpoint. ConfigMap data missing.")

                    data: dict[str, str] = item.data

                    return data.get("MSPL_ENDPOINT", "http://155.54.210.136:8002/meservice")
    except ApiException as e:
        logger.error(f"Unable to retrieve config map {e=}")

    logger.error("Something went wrong while retrieving config map")
    raise ValueError("Unable to retrieve config map")


def _retrieve_architecture(config: Configuration, logger: logging.Logger) -> str:
    logger.info("Retrieving architecture from config map")
    api_endpoint = CoreV1Api(config.k8s_client)

    try:
        config_maps: V1ConfigMapList = api_endpoint.list_namespaced_config_map(config.namespace)
        if len(config_maps.items):
            for item in config_maps.items:
                if item.metadata is None:
                    continue

                if item.metadata.name == "fluidos-mbmo-configmap":
                    logger.debug("ConfigMap identified")
                    if item.data is None:
                        logger.error("Unable to retrieve architecture. ConfigMap data missing.")
                        raise ValueError("Unable to retrieve architecture. ConfigMap data missing.")

                    data: dict[str, str] = item.data

                    return data.get("architecture", "amd64")
    except ApiException as e:
        logger.error(f"Unable to retrieve config map {e=}")

    logger.error("Something went wrong while retrieving config map")
    raise ValueError("Unable to retrieve config map")


def _retrieve_node_identity(config: Configuration, logger: logging.Logger) -> dict[str, str]:
    logger.info("Retrieving node id from config map, or generate a new one if not existing (aka debug mode)")
    api_endpoint = CoreV1Api(config.k8s_client)

    try:
        config_maps: V1ConfigMapList = api_endpoint.list_namespaced_config_map(config.namespace)
        if len(config_maps.items):
            for item in config_maps.items:
                if item.metadata is None:
                    continue

                if item.metadata.name == "fluidos-node-identity":
                    logger.debug("ConfigMap identified")

                    if item.data is None:
                        raise ValueError("ConfigMap data missing.")

                    logger.debug(f"Returning {item.data}")
                    return item.data

    except ApiException as e:
        logger.error(f"Unable to retrieve configmap {e=}")
        raise e

    logger.error("Something went wrong while retrieving node identity. Check that meta-orchestrator connected to a FLUIDOS Node.")
    raise ValueError("Something went wrong while retrieving node identity. Check that the meta-orchestrator connected to a FLUIDOS Node.")


def _retrieve_monitoring_contracts(config: Configuration, logger: logging.Logger) -> bool:
    logger.info("Checking if user wants contract monitoring")
    api_endpoint = CoreV1Api(config.k8s_client)

    try:
        config_maps: V1ConfigMapList = api_endpoint.list_namespaced_config_map(config.namespace)
        if len(config_maps.items):
            for item in config_maps.items:
                if item.metadata is None:
                    continue

                if item.metadata.name == "fluidos-mbmo-configmap":
                    logger.debug("ConfigMap identified")
                    if item.data is None:
                        logger.error("Unable to retrieve flag to enable contract monitoring. ConfigMap data missing.")
                        raise ValueError("Unable to retrieve flag to enable contract monitoring. ConfigMap data missing.")

                    data: dict[str, str] = item.data

                    return data.get("MONITOR_CONTRACTS", "False").casefold() == "true"
    except ApiException as e:
        logger.error(f"Unable to retrieve configmap {e=}")
        raise e

    logger.error("Something went wrong while retrieving MIMO config map.")
    return False


def _retrieve_default_vm_type(config: Configuration, logger: logging.Logger) -> str:
    logger.info("Checking if user wants contract monitoring")
    api_endpoint = CoreV1Api(config.k8s_client)

    try:
        config_maps: V1ConfigMapList = api_endpoint.list_namespaced_config_map(config.namespace)
        if len(config_maps.items):
            for item in config_maps.items:
                if item.metadata is None:
                    continue

                if item.metadata.name == "fluidos-mbmo-configmap":
                    logger.debug("ConfigMap identified")
                    if item.data is None:
                        logger.error("Unable to retrieve default vm type. ConfigMap data missing.")
                        raise ValueError("Unable to retrieve default vm type. ConfigMap data missing.")

                    data: dict[str, str] = item.data

                    return data.get("DEFAULT_VM_TYPE", "")
    except ApiException as e:
        logger.error(f"Unable to retrieve configmap {e=}")
        raise e

    logger.error("Something went wrong while retrieving MIMO config map.")
    return ""


CONFIGURATION = Configuration()
