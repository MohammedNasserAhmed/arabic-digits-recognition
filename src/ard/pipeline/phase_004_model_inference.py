from ard.config.configuration import ConfigurationManager
from ard.components.tf_model_inference import ModelInference
from ard import logger

PHASE_ID = "Model Inference"


class ModelInferencePipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_tf_inference_config, data_tf_inference_params = config.get_data_tf_inference_config()
        data_tf_inference = ModelInference(config=data_tf_inference_config, params=data_tf_inference_params)
        data_tf_inference.predict()




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> phase {PHASE_ID} started <<<<<<")
        obj = ModelInferencePipeline()
        obj.main()
        logger.info(f">>>>>> phase {PHASE_ID} done successfully <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e