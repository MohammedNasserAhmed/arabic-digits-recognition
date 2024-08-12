from adr.config.configuration import ConfigurationManager
from adr.components.data_ingestion import DataIngestion
from adr import logger

PHASE_ID = "Data Ingestion"


class DataIngestionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config, data_ingestion_params = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config, params=data_ingestion_params)
        dataset = data_ingestion.Load()
        dataset.save(compress=True)
        data_ingestion.save()




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> phase {PHASE_ID} started <<<<<<")
        obj = DataIngestionPipeline()
        obj.main()
        logger.info(f">>>>>> phase {PHASE_ID} done successfully <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e