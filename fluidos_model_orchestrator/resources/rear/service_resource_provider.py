import logging
from typing import Any

from kubernetes import client  # type: ignore

from fluidos_model_orchestrator.common import ServiceResourceProvider


logger = logging.getLogger(__name__)


class REARServiceResourceProvider(ServiceResourceProvider):
    def __init__(self, api_client: client.CustomObjectsApi, allocation: dict[str, Any]) -> None:
        self.api_client = api_client
        self.allocation = allocation

    def enrich(self, spec: dict[str, Any]) -> None:
        super().enrich(spec)
