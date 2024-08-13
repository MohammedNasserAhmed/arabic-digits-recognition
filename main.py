from adr import logger
from adr.pipeline.phase_01_data_ingestion import DataIngestionPipeline
from adr.pipeline.phase_02_data_preprocessing import DataPreprocessPipeline
from adr.pipeline.phase_03_data_training import DataTrainingPipeline

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

PHASE_ID = "Data Training"
try:
   logger.info(f">>>>>> phase {PHASE_ID} started <<<<<<")
   obj = DataTrainingPipeline()
   obj.main()
   logger.info(f">>>>>> phase {PHASE_ID} done successfully <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e

