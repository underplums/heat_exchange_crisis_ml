from src.pipelines.inference_pipeline import InferencePipeline


class InferenceService:

    def __init__(self):

        self.pipeline = InferencePipeline()


    def predict(self, scenario_dict: dict):

        result = self.pipeline.predict_scenario(scenario_dict)

        return result