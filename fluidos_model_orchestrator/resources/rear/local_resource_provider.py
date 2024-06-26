from typing import Any

from fluidos_model_orchestrator.common import ResourceProvider


class LocalResourceProvider(ResourceProvider):
    def __init__(self, id: str, owner: dict[str, Any]) -> None:
        super().__init__(id)
        self.owner = owner

    def get_label(self) -> str:
        return ""
