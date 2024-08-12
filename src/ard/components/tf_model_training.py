import tensorflow as tf
import pandas as pd
from typing import Dict, Any, Tuple
import os
from ard import logger
from ard.utils.tf_utils import build_dataset, finalize_dataset
from ard.entity.config_entity import DataTFTrainConfig

tf.get_logger().setLevel('ERROR')  # Stop tf WARNINGS


class ModelTraining:
    def __init__(self, config: DataTFTrainConfig, params: Dict[str, Any]):
        self.config = config
        self.params = params
        self.labels = params['LABELS']
    
    
    def load_and_preprocess_data(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        train_files, val_files = self._read_csv_to_list()
        for files in [train_files, val_files]:
            self._log_audio_info(files)
        return self._create_datasets(train_files, val_files)


    def _read_csv_to_list(self):

        train_files = pd.read_csv(os.path.join(self.config.root_dir, 'train_metadata.csv'))
        val_files = pd.read_csv(os.path.join(self.config.root_dir, 'val_metadata.csv'))
        logger.info(f"Total files: {len(train_files)+ len(val_files)}")
        logger.info(f"Training files: {len(train_files)} ({len(train_files)/(len(train_files)+ len(val_files)):.2%})")
        logger.info(f"Validation files: {len(val_files)} ({len(val_files)/(len(train_files)+ len(val_files)):.2%})")
        # Convert the 'path' column to a list
        return train_files['path'].tolist(), val_files['path'].tolist()
        

    def _log_audio_info(self, files, desc=None):
        num_samples = len(files)
        logger.info(f'Number of total examples in {desc}: {num_samples}')
        
        monos, stereos = self._count_channels(files)
        logger.info(f"Mono audio files: {len(monos)}, Stereo audio files: {len(stereos)} in {desc} dataset")

       

    def _count_channels(self, files):
        monos, stereos = [], []
        for file in files:
            wav_contents = tf.io.read_file(file)
            wav, _ = tf.audio.decode_wav(contents=wav_contents)
            (monos if wav.shape[1] == 1 else stereos).append(file)
        return monos, stereos

    def _create_datasets(self, train_files, val_files):
        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = build_dataset(train_files, AUTOTUNE)
        val_ds = build_dataset(val_files, AUTOTUNE)

        return finalize_dataset(train_ds, AUTOTUNE), finalize_dataset(val_ds, AUTOTUNE)

   
    def build_and_train_model(self, train_ds, val_ds):
        input_shape = self._get_input_shape(train_ds)[1:]
        logger.info(f'Input shape: {input_shape}')
        model = self._create_model(input_shape, len(self.params.LABELS))
        self._compile_model(model)
        self._train_model(model, train_ds, val_ds)

        model.save(self.config.dst_path)

    def _get_input_shape(self, dataset):
        for spectrogram, _ in dataset.take(1):
            input_shape = spectrogram.shape
        return input_shape

    def _create_model(self, input_shape, num_labels):
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_labels)
        ])

    def _compile_model(self, model):
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

    def _train_model(self, model, train_ds, val_ds):
        EPOCHS = 100
        early_stopping = tf.keras.callbacks.EarlyStopping(verbose=1, patience=10)
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=[early_stopping]
        )
        return history
        
    def train(self):
        train, validation = self.load_and_preprocess_data()
        self.build_and_train_model(train, validation)