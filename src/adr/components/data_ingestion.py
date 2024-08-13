from typing import Optional, Union, IO
import os
import random
import numpy as np 
from src.adr.utils.dataset import SeqDataset
from tqdm import tqdm
import time
from src.adr.utils.common import get_wav_path, get_wav_label,Signal
from src.adr.utils.preprocess import WVLoader, MFCCExtractor
from src.adr.utils.transformer import Standardiser, TransformsChain
from tensorflow.keras.preprocessing.sequence import pad_sequences
from adr import logger
from adr.entity.config_entity import DataIngestionConfig

class DataIngestion:
    
    def __init__(self,
                config : DataIngestionConfig,
                params : None
                
                 ):
        
        self.loader = WVLoader()
        self.extractor= MFCCExtractor()
        self.params=params
        self._target_sample_rate = self.params.TARGET_SAMPLE_RATE
        self._random_state = self.params.RANDOM_STATE
        self.config=config
        self._source_path = self.config.source_path
        self._data_file = self.config.data_file
        self.standardiser = Standardiser()
        self.transform_chain = TransformsChain(transforms=[self.standardiser])

        assert self._target_sample_rate > 0, "Sample rate must be a positive integer"
    
    def Load(self): 
        files_list, _inputs, _targets, _lengths = [], [],[], []
        classes = range(10)
        for file in os.listdir(self._source_path):
            if file.lower().endswith(".wav"):
                files_list.append(file)
        
        if files_list:
            random.shuffle(files_list)
            
        with tqdm(total=len(files_list), colour="green", desc="Processing MFCC ", 
                  bar_format="{l_bar}{bar} [time spent: {elapsed}]",
                  leave=True) as pbar:
            for file_name in files_list:
                wav_path = get_wav_path(file_name, self._source_path)
                label = get_wav_label(file_name)
                waveform, sr = self.loader.load(file = wav_path)
                mfcc_signal = self.extractor.mfcc(audio=waveform, sample_rate=sr)
                signal = Signal(name = file_name.split('\\')[-1], data=mfcc_signal, samplerate=sr, filepath=wav_path)
                signal = self.transform_chain.process(signal)
                # Append the MFCC features to the list
              
               
                _inputs.append(signal.data)
                _targets.append(label)
                _lengths.append(signal.data.shape[0])
                pbar.update(1)
                time.sleep(0.01)
        max_length = max(mfcc.shape[1] for mfcc in _inputs) 
        sequences = pad_sequences([mfcc.T for mfcc in _inputs], maxlen=max_length, padding='post', dtype='float32')
        logger.info(f"Padded MFCC shape: {sequences.shape}")
        logger.info(f"Labels shape: {np.array(_targets).shape}")          
        #sequences = np.array(_inputs, dtype=object)
        lengths = np.array(_lengths, dtype=int)
        idx = np.argwhere(np.isin(_targets, classes)).flatten()
        return SeqDataset(features= sequences[idx], targets = np.array(_targets)[idx],
                          lengths = lengths[idx], classes = classes, path =self._data_file, 
                          random_state = self._random_state)
    
    