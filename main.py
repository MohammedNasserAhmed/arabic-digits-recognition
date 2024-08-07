from ard import logger
from ard.pipeline.phase_01_data_ingestion import DataIngestionPipeline
from ard.pipeline.phase_02_data_preprocessing import DataPreprocessPipeline

PHASE_ID = "Data Ingestion"
try:
   logger.info(f">>>>>> phase {PHASE_ID} started <<<<<<") 
   data_ingestion = DataIngestionPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> phase {PHASE_ID} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

PHASE_ID = "Data Preprocessing"

try:
   logger.info(f">>>>>> phase {PHASE_ID} started <<<<<<")
   obj = DataPreprocessPipeline()
   obj.main()
   logger.info(f">>>>>> phase {PHASE_ID} done successfully <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e