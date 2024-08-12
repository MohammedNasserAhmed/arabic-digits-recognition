from adr.config.configuration import ConfigurationManager
from adr.components.data_preprocessing import DataPreprocessing
from adr import logger

PHASE_ID = "Data Preprocessing"


class DataPreprocessPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_preprocess_config = config.get_data_preprocess_config()
        data_preprocess = DataPreprocessing(config=data_preprocess_config)
        dataset = data_preprocess.get_data()
        dataset.save(compress=True)




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> phase {PHASE_ID} started <<<<<<")
        obj = DataPreprocessPipeline()
        obj.main()
        logger.info(f">>>>>> phase {PHASE_ID} done successfully <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e