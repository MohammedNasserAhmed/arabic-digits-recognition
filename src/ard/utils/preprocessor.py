import numpy as np
from src.ard.utils.common import validate_types, ArdArray
import librosa
from abc import ABC, abstractmethod


class saver(ABC):

    def __init__(self, extension: str =None):
        self.extension = extension

    @abstractmethod
    def save(self,
             file: str,
             array: ArdArray):
        """Store array to disk.
        :param file: Path where to save file without extension
        :param array: Numpy array to store
        """
class loader:
    
    """
    Objective:
            The class is designed to load audio files in the WAV format 
            and convert them into a numerical representation that can be 
            used for further processing. It uses the 'librosa' library 
            to read the audio file and resample it if necessary to match 
            the desired sample rate. The resulting numerical representation
            is returned as a numpy array.
    
    """

    def __init__(self):
        pass
    
   
    @staticmethod
    @validate_types(str, int)
    def wav(file_path, sample_rate):
        
        """
        Objective:
                   A static method that takes in the file 
                   path and desired sample rate as arguments, 
                   reads the audio file using 'librosa', 
                   resamples it if necessary, and returns 
                   the numerical representation as a numpy array.

        """
        signal, sr = librosa.load(file_path,
                              sr=sample_rate,
                              mono=True,
                              dtype=np.float32)
        
        assert isinstance(signal, ArdArray)
        if sample_rate is not None and sr != sample_rate:
                signal = librosa.resample(signal, orig_sr=sr, 
                                          target_sr=sample_rate, 
                                          res_type='kaiser_best')    
        return signal
    
    
class padder:
    """
    
    Objective:
            The class is responsible for padding and fitting 
            a numpy array to a specified length. It can either 
            truncate the array if it is longer than the specified 
            length or pad it with zeros if it is shorter. 
            The class assumes that the time-axis is the first 
            axis of the array.
    """
    def __init__(self):
        pass
    
    @staticmethod
    @validate_types(ArdArray, int)
    def fix_length(signal, num_samples):
        
        """
        
        Objective:
                  A static method that takes in a numpy array 
                  and a desired length and returns the padded 
                  or truncated array. If the array is longer 
                  than the desired length, it truncates the noise.
                  If it is shorter, it pads the array with zero.

       
        """
        assert(len(signal.shape) == 2)
        diff = abs(len(signal) - num_samples)
        if len(signal) > num_samples:
            # Truncate noise
            signal = signal[diff // 2 : -((diff + 1) // 2)]
        elif len(signal) < num_samples:
            # Assume the time-axis is the first: (Time, Channel)
            pad_width = [(diff // 2, (diff + 1) // 2)] + [
                         (0, 0) for _ in range(signal.ndim - 1)
                         ]
            
            signal = np.pad(signal, pad_width=pad_width, constant_values=0, mode="constant")
        return signal 
    
  
class extractor:
    
    """
    
    Objective:
             The class is designed to extract Mel-frequency cepstral coefficients
             (MFCCs) from an audio signal using the 'mfcc' method.
             MFCCs are commonly used in audio signal processing for
             tasks such as speech recognition and music genre classification.
             The class utilizes the 'librosa' library to compute the MFCCs.
    """
    
    def __init__(self):
        pass
        
    @staticmethod
    @validate_types(ArdArray, dict)
    def mfcc(signal, spec_kwargs = None):
        
        """
        Objective:
                   Computes the MFCCs of an audio signal using 
                   the 'librosa' library. The method takes in 
                   an audio signal and optional spectrogram parameters
                   as input and returns the computed MFCCs.
                   
        """
        mfcc = librosa.feature.mfcc(y=signal, **spec_kwargs).T
        return mfcc
            
