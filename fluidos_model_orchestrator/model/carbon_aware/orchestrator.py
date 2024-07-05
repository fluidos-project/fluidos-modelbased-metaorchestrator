from uuid import uuid4

from fluidos_model_orchestrator.common import ModelInterface
from fluidos_model_orchestrator.common import ModelPredictRequest
from fluidos_model_orchestrator.common import ModelPredictResponse
from fluidos_model_orchestrator.common import Resource

class CarbonAwareOrchestrator(ModelInterface):
    def predict(self, req: ModelPredictRequest, architecture: str = "arm64") -> ModelPredictResponse:

        res = ModelPredictResponse(
            req.id,
            Resource(id=f"carbonaware-{str(uuid4())}", cpu="2n", memory="20Mi", architecture="arm64"))

        #update_local_flavours_forecasted_data(req.namespace)

        return res
