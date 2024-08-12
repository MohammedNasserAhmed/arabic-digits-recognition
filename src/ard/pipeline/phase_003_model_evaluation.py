from ard.config.configuration import ConfigurationManager
from ard.components.tf_model_evaluation import ModelEvaluation
from ard import logger

PHASE_ID = "Model Evaluation"


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_tf_evaluation_config = config.get_data_tf_evaluation_config()
        data_tf_evaluation = ModelEvaluation(config=data_tf_evaluation_config)
        data_tf_evaluation.predict()




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> phase {PHASE_ID} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> phase {PHASE_ID} done successfully <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e