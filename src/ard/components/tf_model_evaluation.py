from ard.entity.config_entity import DataTFEvaluationConfig
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from ard import logger
from ard.utils.tf_utils import build_dataset, finalize_dataset

tf.get_logger().setLevel('ERROR')  # Stop tf WARNINGS


class ModelEvaluation:
    def __init__(self, config: DataTFEvaluationConfig):
        self.config = config
        
        
    def predict(self):
        # Prepare to collect test audio and labels
        test_files = pd.read_csv(os.path.join(self.config.data_path))
        logger.info(f"Total Test files: {len(test_files)}")
        test = self._create_dataset(test_files['path'].tolist())
        model = tf.keras.models.load_model(self.config.model_path)
        test_audio = []
        test_labels = []

        for audio, label in test:
            # Convert tensors to numpy arrays
            audio_np = audio.numpy()  
            test_audio.append(audio_np)
            test_labels.append(label.numpy()) 

        # Concatenate the batches into a single array
        test_audio = np.concatenate(test_audio, axis=0)  
        test_labels = np.concatenate(test_labels, axis=0) 

        # Now you can evaluate the model
        loss, accuracy = model.evaluate(test_audio, test_labels)
        logger.info(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')
        
    def _create_dataset(self, files):
        AUTOTUNE = tf.data.AUTOTUNE

        test_ds = build_dataset(files, AUTOTUNE)

        return finalize_dataset(test_ds, AUTOTUNE)