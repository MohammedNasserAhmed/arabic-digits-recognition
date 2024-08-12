from adr.entity.config_entity import DataTFInferenceConfig
import tensorflow as tf
import numpy as np
from typing import Dict, Any
import tensorflow_io as tfio
from adr import logger
import pandas as pd

tf.get_logger().setLevel('ERROR')  # Stop tf WARNINGS
class ModelInference:
    def __init__(self, config: DataTFInferenceConfig, params: Dict[str, Any]):
        self.config = config
        self.params=params
        self.model = tf.keras.models.load_model(self.config.model_path)
        self._target_sample_rate = params['TARGET_SAMPLE_RATE']
    def preprocess_audio(self):
        test_data = pd.read_csv(self.config.audio_path)
    
        # Select a random row from the specified column
        audio_path = test_data["path"].sample(n=1).values[0]  # Get a 
        logger.info(str(audio_path))
        audio_data = tf.io.read_file(audio_path)
        audio, sample_rate = tf.audio.decode_wav(audio_data, desired_channels=1)
        audio = tf.squeeze(audio, axis=-1)
        sample_rate = tf.cast(sample_rate, dtype=tf.int64)
        audio =  tfio.audio.resample(audio,rate_in=sample_rate, rate_out=16000)
        input_len = 10000
        if tf.shape(audio)[0] < input_len:
            zero_padding = tf.zeros(
                [10000] - tf.shape(audio),
                dtype=tf.float32)
            audio = tf.cast(audio, dtype=tf.float32)
            equal_length = tf.concat([audio, zero_padding], 0)
        else:
            equal_length = audio[:input_len]
            
        spectrogram = tf.signal.stft(
            equal_length, frame_length=self.params.FRAME_LENGTH, 
            frame_step=self.params.FRAME_STEP, window_fn = tf.signal.hamming_window)

        spectrogram = tf.abs(spectrogram)
        spectrogram = spectrogram[..., tf.newaxis]
        logger.info(f"Shape of WAV is {spectrogram.shape}")
        spectrogram = tf.expand_dims(spectrogram, axis=0)  # Add batch dimension


        return spectrogram

    def predict(self):
        audio_data = self.preprocess_audio()
        predictions = self.model.predict(audio_data)
        predicted_label = np.argmax(predictions, axis=1)
        logger.info(f"Prediction score of the givin digit is:{predicted_label}")
        

