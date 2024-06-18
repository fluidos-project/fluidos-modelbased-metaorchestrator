from uuid import uuid4

from ..common import ModelInterface
from ..common import ModelPredictRequest
from ..common import ModelPredictResponse
from ..common import Resource


class DummyOrchestrator(ModelInterface):
    def predict(self, req: ModelPredictRequest, architecture: str = "amd64") -> ModelPredictResponse:
        return ModelPredictResponse(
            req.id,
            Resource(id=f"dummy-{str(uuid4())}", cpu="2n", memory="20Mi")
        )
