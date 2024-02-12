from ..common import ModelInterface, Resource
from ..common import ModelPredictRequest
from ..common import ModelPredictResponse


class DummyOrchestrator(ModelInterface):
    def predict(self, req: ModelPredictRequest) -> ModelPredictResponse:
        return ModelPredictResponse(
            req.id,
            Resource(cpu="2n", memory="20Mi")
        )
