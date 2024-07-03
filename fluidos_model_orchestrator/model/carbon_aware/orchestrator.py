from uuid import uuid4

from fluidos_model_orchestrator.common import ModelInterface
from fluidos_model_orchestrator.common import ModelPredictRequest
from fluidos_model_orchestrator.common import ModelPredictResponse
from fluidos_model_orchestrator.common import Resource
from fluidos_model_orchestrator.resources import get_resource_finder

class CarbonAwareOrchestrator(ModelInterface):
    def predict(self, req: ModelPredictRequest, architecture: str = "arm64") -> ModelPredictResponse:

        res = ModelPredictResponse(
            req.id,
            Resource(id=f"carbonaware-{str(uuid4())}", cpu="2n", memory="20Mi", architecture="arm64"))

        #resourcesFinder = get_resource_finder(res, req)

        return res
