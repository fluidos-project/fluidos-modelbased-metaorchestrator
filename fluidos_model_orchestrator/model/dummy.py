from uuid import uuid4

from fluidos_model_orchestrator.common.model import ModelPredictRequest
from fluidos_model_orchestrator.common.model import ModelPredictResponse
from fluidos_model_orchestrator.common.model import OrchestratorInterface
from fluidos_model_orchestrator.common.resource import Resource


class DummyOrchestrator(OrchestratorInterface):
    def predict(self, req: ModelPredictRequest, architecture: str = "amd64") -> ModelPredictResponse:
        return ModelPredictResponse(
            req.id,
            Resource(id=f"dummy-{str(uuid4())}", cpu="2n", memory="20Mi")
        )
