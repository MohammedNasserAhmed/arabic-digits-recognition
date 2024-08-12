from ard.constants import *
from ard.utils.help import read_yaml, create_directories
from ard.entity.config_entity import (DataIngestionConfig, DataPreprocessConfig, 
                                      DataTFIngestionConfig, DataTFTrainConfig, DataTFEvaluationConfig,
                                      DataTFInferenceConfig)
class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_path=config.source_path,
            arclasses_path= config.arclasses_path,
            data_file=config.data_file,
            csv_file=config.csv_file
        )

        return data_ingestion_config, self.params
    
    def get_data_tf_ingestion_config(self) -> DataTFIngestionConfig:
        config = self.config.data_tf_ingestion

        create_directories([config.root_dir])

        data_tf_ingestion_config = DataTFIngestionConfig(
            root_dir=config.root_dir,
            source_path=config.source_path,
            dst_path=config.dst_path
           
        )

        return data_tf_ingestion_config
    
    def get_data_tf_training_config(self) -> DataTFTrainConfig:
        config = self.config.data_tf_training

        create_directories([config.root_dir])

        data_tf_training_config = DataTFTrainConfig(
            root_dir=config.root_dir,
            dst_path=config.dst_path
           
        )

        return data_tf_training_config, self.params
    
    def get_data_tf_evaluation_config(self) -> DataTFEvaluationConfig:
        config = self.config.data_tf_evaluation

        create_directories([config.root_dir])

        data_tf_evaluation_config = DataTFEvaluationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_path=config.model_path
            
           
        )

        return data_tf_evaluation_config
    
    def get_data_tf_inference_config(self) -> DataTFInferenceConfig:
        config = self.config.data_tf_inference

        create_directories([config.root_dir])

        data_tf_inference_config = DataTFInferenceConfig(
            root_dir=config.root_dir,
            audio_path=config.audio_path,
            model_path=config.model_path
           
        )

        return data_tf_inference_config, self.params
    def get_data_preprocess_config(self) -> DataPreprocessConfig:
        config = self.config.data_preprocessing

        create_directories([config.root_dir])

        data_preprocess_config = DataPreprocessConfig(
            root_dir=config.root_dir,
            source_path=config.source_path,
            data_file=config.data_file,
        )

        return data_preprocess_config