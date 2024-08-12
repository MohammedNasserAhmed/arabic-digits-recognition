from ard.config.configuration import ConfigurationManager
from ard.components.tf_data_ingestion import DataTFIngestion
from ard import logger

PHASE_ID = "Data Ingestion"


class DataTFIngestionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_tf_ingestion_config = config.get_data_tf_ingestion_config()
        data_tf_ingestion = DataTFIngestion(config=data_tf_ingestion_config)
        data_tf_ingestion.load_and_preprocess_data()




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> phase {PHASE_ID} started <<<<<<")
        obj = DataTFIngestionPipeline()
        obj.main()
        logger.info(f">>>>>> phase {PHASE_ID} done successfully <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e