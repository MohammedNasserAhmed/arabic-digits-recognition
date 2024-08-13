"""This module provides a class to load audio files into Signal objects."""

from adr import logger
from pathlib import Path
import os
import csv
from contextlib import contextmanager
import copy
from typing import Dict, List, Tuple, Optional
import librosa
import numpy as np
from tqdm import tqdm
import time
import tensorflow as tf
from adr.constants import PARAMS_FILE_PATH
from adr.utils.help import read_yaml

params = read_yaml(PARAMS_FILE_PATH)
class WVLoader:
    """A class for loading audio files into Signal objects."""

    DEFAULT_EXTENSIONS = [".wav", ".mp3"]
   

    def __init__(
        self,
        mono: bool = True,
        allowed_extensions: Optional[List[str]] = None
    ):
        """
        Initialize the WVLoader.

        Args:
            mono (bool): Whether to load audio as mono. Defaults to True.
            data_type (DTypeLike): The data type for the waveform. Defaults to np.float32.
            allowed_extensions (Optional[List[str]]): List of allowed audio file extensions. 
                                                      Defaults to [".wav", ".mp3"].
        """
        self.mono = mono
        self._audio_file_extensions = allowed_extensions or self.DEFAULT_EXTENSIONS
        logger.info(
            "WVLoader is initializing"
        )

    def load(self, file: str):
        """
        Load an audio file and return a Signal object.

        Args:
            file (str): Path to the audio file to load.
            sample_rate (int): The desired sample rate for the audio.

        Returns:
            Signal: A Signal object containing the loaded audio data.

        Raises:
            FileExtensionError: If the file extension is not allowed.
            FileNotFoundError: If the file does not exist.
        """
        file_path = Path(file)
        if not file_path.exists():
            raise FileNotFoundError(f"The file {file} does not exist.")

        self._validate_file_extension(file_path)
        
        waveform, sample_rate = self._load_audio(file_path)
        
        #logger.info("Loaded audio file: %s", file.split('\\')[-1])
        return waveform, sample_rate

    def _validate_file_extension(self, file_path: Path) -> None:
        """
        Validate that the file has an allowed extension.

        Args:
            file_path (Path): Path to the audio file.

        Raises:
            FileExtensionError: If the file extension is not allowed.
        """
        extension = file_path.suffix.lower()
        if extension not in self._audio_file_extensions:
            raise FileExtensionError(f"File extension '{extension}' is not allowed. "
                                     f"Allowed extensions are: {', '.join(self._audio_file_extensions)}")

    def _load_audio(self, file_path: Path) -> np.array:
        """
        Load the audio file using librosa.

        Args:
            file_path (Path): Path to the audio file.
            sample_rate (int): The desired sample rate.

        Returns:
            np.ndarray: The waveform.

        Raises:
            LibrosaError: If there's an error loading the audio file with librosa.
        """
        try:
            waveform, loaded_sample_rate = librosa.load(
                file_path,
                mono=self.mono,
                dtype=np.float32
            )
            # if loaded_sample_rate != sample_rate:
            #     logger.warning(
            #         "Loaded sample rate (%d) differs from requested sample rate (%d)",
            #         loaded_sample_rate, sample_rate
            #     )
           
            return waveform, loaded_sample_rate
        except librosa.LibrosaError as e:
            logger.error("Error loading audio file: %s", str(e))
            raise

class MFCCExtractor:
    
    """
    
    Objective:
             The class is designed to extract Mel-frequency cepstral coefficients
             (MFCCs) from an audio signal using the 'mfcc' method.
             MFCCs are commonly used in audio signal processing for
             tasks such as speech recognition and music genre classification.
             The class utilizes the 'librosa' library to compute the MFCCs.
    """
    
    def __init__(
        self):
        pass        
    
    
    def mfcc(self, audio, sample_rate):
        
        """
        Objective:
                   Computes the MFCCs of an audio signal using 
                   the 'librosa' library. The method takes in 
                   an audio signal and optional spectrogram parameters
                   as input and returns the computed MFCCs.
                   
        """
        #audio = self.frame_audio(audio)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc = params['NUM_MFCC'])
        
       
        return mfcc
    
    def frame_audio(self, audio):
        input_len = 10000
        if tf.shape(audio)[0] < input_len:
            zero_padding = tf.zeros(
                [10000] - tf.shape(audio),
                dtype=tf.float32)
            audio = tf.cast(audio, dtype=tf.float32)
            audio = tf.concat([audio, zero_padding], 0)
        else:
            audio = audio[:input_len]
        
        return np.asarray(audio)


class SaveAsCSV:
    def __init__(self, func):
        """
        Initializes the class with a function as input.

        Args:
            func (callable): The function to be decorated.
        """
        self._func = func

    def __call__(self, audio_dir: str, csv_path: str) -> None:
        """
        Modifies the input function to add metadata to the audio files and save it as a CSV file.

        Args:
            audio_dir (str): Directory containing audio files.
            csv_path (str): Path to save the CSV file.
            metadata (Dict[str, List]): Initial metadata dictionary.
            digits (Dict[str, int]): Mapping of digit words to numbers.
        """
        metadata = {
        "filename": [],
        "arclasses": [],
        "classes": []
        }
        digits= {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9
        }
        metadata_dict, total_files = self.metadata(audio_dir, metadata, digits)
        return self._func(csv_path, metadata_dict, total_files)

    @staticmethod
    def metadata(audio_dir: str, metadata: Dict[str, List], digits: Dict[str, int]) -> Tuple[Dict[str, List], int]:
        """
        Collects metadata for each audio file in the directory.

        Args:
            audio_dir (str): Directory containing audio files.
            metadata (Dict[str, List]): Initial metadata dictionary.
            digits (Dict[str, int]): Mapping of digit words to numbers.

        Returns:
            Tuple[Dict[str, List], int]: Updated metadata dictionary and total number of files.
        """
        metadata_dict = copy.deepcopy(metadata)
        total_files = 0
        
        for dirpath, _, filenames in os.walk(audio_dir):
            if dirpath != audio_dir:
                dclass = os.path.basename(dirpath)
               
                with tqdm(total=len(filenames), colour="blue", desc=f"Storing Metadata:{dclass}", 
                  bar_format="{l_bar}{bar} [time spent: {elapsed}]",
                  leave=True) as pbar:
                    for f in filenames:
                        if f.lower().endswith(".wav"):
                            total_files += 1
                            arclass = os.path.basename(dirpath)
                            metadata_dict["filename"].append(f)
                            label = f.split('-')[0]
                            label = digits.get(label, label)  # Use get() to handle potential KeyError
                            metadata_dict["arclasses"].append(arclass)
                            metadata_dict["classes"].append(label)
                            pbar.update(1)
                            time.sleep(0.01)

        return metadata_dict, total_files

@SaveAsCSV
def save_metadata(csv_path: str, data_dict: Dict[str, List], total_files: int) -> None:
    """
    Saves the metadata to a CSV file.

    Args:
        audio_dir (str): Directory containing audio files.
        csv_path (str): Path to save the CSV file.
        data_dict (Dict[str, List]): Metadata dictionary.
        digits (Dict[str, int]): Mapping of digit words to numbers.
        total_files (int): Total number of files processed.
    """
    headers = list(data_dict.keys())
    with managed_save_file(csv_path) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        for i in range(total_files):
            writer.writerow({header: data_dict[header][i] for header in headers})

@contextmanager
def managed_save_file(file_path: str):
    """
    Context manager for safely opening and closing a file.

    Args:
        file_path (str): Path to the file to be opened.

    Yields:
        file object: The opened file object.
    """
    try:
        csv_file = open(file_path, mode='w', encoding='utf-8', newline='')
        yield csv_file
    finally:
        csv_file.close()




class FileExtensionError(Exception):
    """Exception raised when a file with an invalid extension is encountered."""


class LibrosaError(Exception):
    """Exception raised when there's an error loading audio with librosa."""