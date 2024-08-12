from adr.config.configuration import ConfigurationManager
from adr.components.tf_model_training import ModelTraining
from adr import logger

PHASE_ID = "Model Trainig"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_tf_training_config, data_tf_training_params = config.get_data_tf_training_config()
        data_tf_training = ModelTraining(config=data_tf_training_config, params=data_tf_training_params)
        data_tf_training.train()




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> phase {PHASE_ID} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> phase {PHASE_ID} done successfully <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e