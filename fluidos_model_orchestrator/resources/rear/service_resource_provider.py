import logging
from base64 import b64decode
from typing import Any

from kubernetes.client import CoreV1Api  # type: ignore
from kubernetes.client.api_client import ApiClient  # type: ignore
from kubernetes.client.exceptions import ApiException  # type: ignore

from fluidos_model_orchestrator.common import ServiceResourceProvider


logger = logging.getLogger(__name__)


class REARServiceResourceProvider(ServiceResourceProvider):
    def __init__(self, endpoints: str, username: str, password: str) -> None:
        # for the demo only!!
        self.endpoints = endpoints
        self.username = username
        self.password = password

    def enrich(self, container: dict[str, Any]) -> None:

        if "env" not in container:
            container["env"] = {}

        env: dict[str, str] = container["env"]

        env["FLUIDOS_MQTT_ENDPOINTS"] = self.endpoints
        env["FLUIDOS_MQTT_USERNAME"] = self.username
        env["FLUIDOS_MQTT_PASSWORD"] = self.password


def build_REARServiceResourceProvider(api_client: ApiClient | None, allocation: dict[str, Any]) -> REARServiceResourceProvider:
    if api_client is None:
        raise ValueError("api_client is None")

    client = CoreV1Api(api_client=api_client)

    secret_name = allocation["status"]["resourceRef"]["name"]
    namespace = allocation["status"]["resourceRef"]["namespace"]

    try:
        secret = client.read_namespaced_secret(secret_name, namespace)

        if secret.data is None:
            raise ValueError("Unexpected None value in secret data")

        return REARServiceResourceProvider(
            endpoints=b64decode(secret.data["endpoints"]).decode(),
            username=b64decode(secret.data["username"]).decode(),
            password=b64decode(secret.data["password"]).decode(),
        )

    except ApiException as e:
        logger.error("Unable to retrieve sercret")
        raise e
