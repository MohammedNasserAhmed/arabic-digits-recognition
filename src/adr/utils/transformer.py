from __future__ import annotations

import logging
import librosa
import numpy as np
from src.adr.utils.common import Signal
from typing import List

from abc import ABC, abstractmethod
from enum import Enum

logger = logging.getLogger(__name__)


class TransformType(Enum):
    """Enumeration class with all available transforms."""

    MINMAXSCALER = "minmaxscaler"
    MFCC = "mfcc"
    STANDARDSCALER = "standardscaler"
    PCA = "pca"


class Transform(ABC):

    def __init__(self, name: TransformType):
        self.name = name
        logger.info("Instantiated %s transform", self.name)

    @abstractmethod
    def process(self, signal: Signal) -> Signal:
        """This method is responsible to apply a transforms to the incoming
        signal.
        :param signal: Signal object to be manipulated
        :return: New signal object with transformed values
        """

    def _prepend_transform_name(self, string):
        return self.name.value + "_" + string
    
    
    
class Scaler(Transform):
    
    def process(self, signal: Signal) -> Signal:
        signal.name = self._prepend_transform_name(signal.name)
        signal.data = self._scale(signal.data)
        #logger.info("Loaded and Applied %s on %s", self.name, signal.filepath.split("\\")[-1])
        return signal

    @abstractmethod
    def _scale(self, array: np.array):
        """Concrete Scalers must implement this method. In this method,
        the specific scaling strategy must be implemented.
        :param array: Array to scale
        :return: Scaled array
        """

class MFCC(Transform):


    def __init__(self,
                 num_mfcc: int = 13,
                 frame_length: int = 2048,
                 hop_length: int = 1024,
                 win_length: int = 2048,
                 window: str = "hann"):
        
        super().__init__(TransformType.MFCC)
        self.num_mfcc = num_mfcc
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window

    def process(self, signal: Signal) -> Signal:
        """Extract MFCCs and modify signal.
        :param signal: Signal object. Note: this transforms works only with
            waveform data
        :return: Modified signal
        """
        signal.name = self._prepend_transform_name(signal.name)
        signal.data = librosa.feature.mfcc(signal.data,
                                           sr=signal.samplerate,
                                           n_fft=self.frame_length,
                                           hop_length=self.hop_length,
                                           win_length=self.win_length,
                                           window=self.window)
        logger.info("Applied %s to %s", self.name.value, signal.filepath.split("\\")[-1])
        return signal
    

class MinMaxScaler(Scaler):
    
    def __init__(self, min: float = 0., max: float = 1.):
        super().__init__(TransformType.MINMAXSCALER)
        self.min_val = min
        self.max_val = max

    def _scale(self, array: np.array):
        scaled_array = (array - array.min()) / (array.max() - array.min())
        scaled_array = scaled_array * (self.max_val - self.min_val) + self.min_val
        return scaled_array
    
class Standardiser(Scaler):

    def __init__(self):
        super().__init__(TransformType.STANDARDSCALER)
        self._min_std = 1e-7

    def _scale(self, array: np.array):
        array_mean = np.mean(array)
        array_std = np.std(array)
        scaled_array = (array - array_mean) / np.maximum(array_std, self._min_std)
        return scaled_array

class TransformsChain():
    """Apply multiple transforms on a signal in a sequential manner."""

    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms

    @property
    def transforms_names(self):
        transform_names = [transform.name.value for transform in
                           self.transforms]
        return transform_names

    def process(self, signal: Signal) -> Signal:
        """Apply multiple transforms sequentially to a signal.
        :param signal: Signal to transform
        :return: Modified signal
        """
        for transform in self.transforms:
            signal = transform.process(signal)
        return signal