from adr import logger
from adr.pipeline.phase_01_data_ingestion import DataIngestionPipeline
from adr.pipeline.phase_02_data_preprocessing import DataPreprocessPipeline
from adr.pipeline.phase_001_tf_data_ingestion import DataTFIngestionPipeline
from adr.pipeline.phase_002_tf_model_training import ModelTrainingPipeline
from adr.pipeline.phase_003_model_evaluation import ModelEvaluationPipeline
from adr.pipeline.phase_004_model_inference import ModelInferencePipeline

""" PHASE_ID = "Data Ingestion"
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
   raise e """
PHASE_ID = "Data Ingestion" 
try:
   logger.info(f">>>>>> phase {PHASE_ID} started <<<<<<")
   obj = DataTFIngestionPipeline()
   obj.main()
   logger.info(f">>>>>> phase {PHASE_ID} done successfully <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e


PHASE_ID = "Model Training" 
try:
   logger.info(f">>>>>> phase {PHASE_ID} started <<<<<<")
   obj = ModelTrainingPipeline()
   obj.main()
   logger.info(f">>>>>> phase {PHASE_ID} done successfully <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e

PHASE_ID = "Model Evaluation" 
try:
   logger.info(f">>>>>> phase {PHASE_ID} started <<<<<<")
   obj = ModelEvaluationPipeline()
   obj.main()
   logger.info(f">>>>>> phase {PHASE_ID} done successfully <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e

PHASE_ID = "Model Inference" 
try:
   logger.info(f">>>>>> phase {PHASE_ID} started <<<<<<")
   obj = ModelInferencePipeline()
   obj.main()
   logger.info(f">>>>>> phase {PHASE_ID} done successfully <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e

