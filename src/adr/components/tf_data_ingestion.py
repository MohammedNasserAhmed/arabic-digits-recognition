import tensorflow as tf
import numpy as np
from pathlib import Path
import pandas as pd
import random
import os
from adr import logger
from adr.entity.config_entity import DataTFIngestionConfig

tf.get_logger().setLevel('ERROR') 

class DataTFIngestion:
    def __init__(self, config: DataTFIngestionConfig):
        self.config = config
        self.labels = None

    def load_and_preprocess_data(self) -> None:
        self.labels = self._get_labels()
        files = self._read_and_shuffle_files()
        return self._split_and_save_files(files)

    def _get_labels(self) -> np.ndarray:
        labels = np.array(tf.io.gfile.listdir(str(self.config.source_path)))
        logger.info(f'Labels: {labels}')
        return labels

    def _get_shuffled_files(self) -> tf.Tensor:
        _files = tf.io.gfile.glob(str(self.config.source_path) + '*/*')
        files = tf.random.shuffle(_files)
        self._log_audio_info(files)
        return files

    def _log_audio_info(self, files):
        num_samples = len(files)
        logger.info(f'Number of total examples: {num_samples}')
        
        monos, stereos = self._count_channels(files)
        logger.info(f"Mono audio files: {len(monos)}, Stereo audio files: {len(stereos)}")

        for index, item in enumerate(self.labels):
            count = len(tf.io.gfile.listdir(Path(self.config.source_path, self.labels[index])))
            logger.info(f'Number of examples for {item}: {count}')

    def _count_channels(self, files):
        monos, stereos = [], []
        for file in files:
            wav_contents = tf.io.read_file(file)
            wav, _ = tf.audio.decode_wav(contents=wav_contents)
            (monos if wav.shape[1] == 1 else stereos).append(file)
        return monos, stereos
    
        

    def _split_and_save_files(self, files):
        total_files = len(files)
        train_size = int(0.80 * total_files)
        val_size = int(0.10 * total_files)
        
        train_files = files[:train_size]
        val_files = files[train_size:train_size + val_size]
        test_files = files[train_size + val_size:]
        
        
        logger.info(f"Total files: {total_files}")
        logger.info(f"Training files: {len(train_files)} ({len(train_files)/total_files:.2%})")
        logger.info(f"Validation files: {len(val_files)} ({len(val_files)/total_files:.2%})")
        logger.info(f"Testing files: {len(test_files)} ({len(test_files)/total_files:.2%})")
        metadata = self._extract_labels_from_paths(train_files)
        self._save_metadata_to_csv(metadata, os.path.join(self.config.dst_path, 'train_metadata.csv'))
        metadata = self._extract_labels_from_paths(val_files)
        self._save_metadata_to_csv(metadata, os.path.join(self.config.dst_path, 'val_metadata.csv'))
        metadata = self._extract_labels_from_paths(test_files)
        self._save_metadata_to_csv(metadata, os.path.join(self.config.dst_path, 'test_metadata.csv'))
        
        logger.info(f"Training files saved as train_metadata.csv")
        logger.info(f"Validation files saved as val_metadata.csv")
        logger.info(f"Testing files saved as test_metadata.csv")

        return 
    
        
    def _extract_labels_from_paths(self, file_paths):
        metadata = []
        i = 0
        for file_path in file_paths:
            # Get the parent directory name (label)
            label = os.path.basename(os.path.dirname(file_path))
            # Append the file path and label to the metadata list
            metadata.append({
                'path': file_path,
                'label': label
            })
            i=i+1
        return metadata

    def _save_metadata_to_csv(self, metadata, filename):
        # Create a DataFrame and save to CSV
        df = pd.DataFrame(metadata)
        df.to_csv(filename, index=False)
        

    def _read_and_shuffle_files(self):
        # Get a list of all subdirectories in the source path
        subdirs = [os.path.join(self.config.source_path, d) for d in os.listdir(self.config.source_path) if os.path.isdir(os.path.join(self.config.source_path, d))]
        
        # Initialize an empty list to store all file paths
        all_files = []
        
        # Iterate through each subdirectory and collect audio files
        for subdir in subdirs:
            for filename in os.listdir(subdir):
                if filename.endswith('.wav'):  # Adjust this based on your audio file format
                    file_path = os.path.join(subdir, filename)
                    all_files.append(file_path)
        
        # Shuffle the list of file paths
        random.shuffle(all_files)
        
        return all_files

