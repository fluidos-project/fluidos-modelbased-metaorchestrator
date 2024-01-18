from ..common import ModelInterface
from ..common import ModelPredictRequest
from ..common import ModelPredictResponse


class DummyOrchestrator(ModelInterface):
    def predict(self, req: ModelPredictRequest) -> ModelPredictResponse:
        return None
