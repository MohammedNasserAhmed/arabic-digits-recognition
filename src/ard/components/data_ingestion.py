import os
import random
import numpy as np 
from ard.utils.dataset import SeqDataset
from tqdm import tqdm
import time
from ard.utils.common import get_wav_path, get_wav_label, Signal
from ard.utils.preprocess import WVLoader, MFCCExtractor, save_metadata
from ard.utils.transformer import  MinMaxScaler, Standardiser, TransformsChain, MFCC
from ard.entity.config_entity import DataIngestionConfig

class DataIngestion:
    
    def __init__(self,
                config : DataIngestionConfig,
                params : None
                
                 ):
        
        self.loader = WVLoader()
        self.extractor= MFCCExtractor()
        self.params=params
        self._target_sample_rate = self.params.TARGET_SAMPLE_RATE
        self._num_samples = self.params.NUM_SAMPLES
        self._random_state = self.params.RANDOM_STATE
        self._transform_kwargs = self.params.SPEC_KWARGS
        self.config=config
        self._source_path = self.config.source_path
        self._arclasses_path = self.config.arclasses_path
        self._data_file = self.config.data_file
        self._csv_file = self.config.csv_file
        self.minmax_scaler = MinMaxScaler(min=0, max=1)
        self.standardiser = Standardiser()
        self.transform_chain = TransformsChain(transforms=[self.minmax_scaler, self.standardiser])

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
                waveform = self.loader.load(file = wav_path,sample_rate= self._target_sample_rate)
                mfcc_signal = self.extractor.mfcc(data=waveform, spec_kwargs=self._transform_kwargs)
                signal = Signal(name = file_name.split('\\')[-1], data=mfcc_signal, samplerate=self._target_sample_rate, filepath=wav_path)
                scaled_signal = self.transform_chain.process(signal)
                _inputs.append(scaled_signal.data)
                _targets.append(label)
                _lengths.append(scaled_signal.data.shape[0])
                pbar.update(1)
                time.sleep(0.01)
        sequences = np.array(_inputs, dtype=object)
        lengths = np.array(_lengths, dtype=int)
        idx = np.argwhere(np.isin(_targets, classes)).flatten()
        
        return SeqDataset(features= np.vstack(sequences[idx]), targets = np.array(_targets)[idx],
                          lengths = lengths[idx], classes = classes, path =self._data_file, 
                          random_state = self._random_state)
    def save(self):
        save_metadata(self._arclasses_path, self._csv_file)