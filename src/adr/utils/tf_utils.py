# -*- coding: utf-8 -*- 

import tensorflow as tf
import tensorflow_io as tfio
import os
from adr.constants import *
from adr.utils.help import read_yaml

params = read_yaml(PARAMS_FILE_PATH)
def _GET_LABELs(file_path):
  slices = tf.strings.split(
      input=file_path,
      sep=os.path.sep)
  #tf.print("The parent or label of this audio is :", slices[-2])
  return slices[-2]

def _LOAD_in16KH_MONO(audio_path):
    audio_data = tf.io.read_file(audio_path)
    audio, sample_rate = tf.audio.decode_wav(audio_data, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    audio =  tfio.audio.resample(audio,rate_in=sample_rate, rate_out=16000)
    return audio, sample_rate

def _GET_WAVEFORM_LABEL(file_path, display = False):
  label = _GET_LABELs(file_path)
  waveform, _ = _LOAD_in16KH_MONO(file_path)
  return waveform, label

def _GET_SPECTROGRAM(audio):
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
      equal_length, frame_length=params['FRAME_LENGTH'], 
      frame_step=params['FRAME_STEP'], window_fn = tf.signal.hamming_window)

  spectrogram = tf.abs(spectrogram)
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram


def _GET_SPECTROGRAM_LABEL_ID(audio, label):
  
  labels = params['LABELS']
  spectrogram = _GET_SPECTROGRAM(audio)
  label_id = tf.math.argmax(label == labels)
  return spectrogram, label_id


def build_dataset(files, AT):
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  wf_ds = files_ds.map(
      map_func=_GET_WAVEFORM_LABEL,
      num_parallel_calls=AT)
  spec_ds = wf_ds.map(
      map_func=_GET_SPECTROGRAM_LABEL_ID,
      num_parallel_calls=AT)
  return spec_ds

def finalize_dataset(data, AUTOTUNE):
        batch_size = 24
        buffer_size = 1000

        data = data.shuffle(buffer_size).batch(batch_size).cache().prefetch(AUTOTUNE)

        return data
