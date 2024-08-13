from adr.config.configuration import ConfigurationManager
from adr.components.data_training import DataTraining
from adr import logger

PHASE_ID = "Data Training"


class DataTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_training_config = config.get_data_training_config()
        data_training = DataTraining(config=data_training_config)
        data_training.process()



if __name__ == '__main__':
    try:
        logger.info(f">>>>>> phase {PHASE_ID} started <<<<<<")
        obj = DataTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> phase {PHASE_ID} done successfully <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e