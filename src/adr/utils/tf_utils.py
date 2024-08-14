from typing import Tuple
import tensorflow as tf
import tensorflow_io as tfio
import os
from adr.constants import PARAMS_FILE_PATH
from adr.utils.help import read_yaml

params = read_yaml(PARAMS_FILE_PATH)
def extract_label_from_path(file_path):
    """
    Get the parent or label of the audio file.

    Args:
        file_path (str): The path of the audio file.

    Returns:
        str: The parent or label of the audio file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File path {file_path} does not exist.")
    slices = tf.strings.split(
        input=file_path,
        sep=os.path.sep)
    return slices[-2]

def load_in_16khz_mono(audio_path):
    if not isinstance(audio_path, str) or not os.path.isfile(audio_path):
        raise ValueError("`audio_path` must be a valid file path.")
    audio_data = tf.io.read_file(audio_path)
    audio, sample_rate = tf.audio.decode_wav(audio_data, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    audio =  tfio.audio.resample(audio,rate_in=sample_rate, rate_out=16000)
    return audio, sample_rate

def _GET_WAVEFORM_LABEL(file_path: str) -> Tuple[tf.Tensor, str]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File path {file_path} does not exist.")
    label = extract_label_from_path(file_path)
    waveform, _ = load_in_16khz_mono(file_path)
    return waveform, label

def _GET_SPECTROGRAM(audio):
    input_len = params['INPUT_LENGTH']
    audio = tf.cast(audio, dtype=tf.float32)
    if tf.shape(audio)[0] < input_len:
        zero_padding = tf.zeros(
            [input_len] - tf.shape(audio),
            dtype=tf.float32)
        equal_length = tf.concat([audio, zero_padding], 0)
    else:
        equal_length = audio[:input_len]
    spectrogram = tf.signal.frame(
        equal_length, frame_length=params['FRAME_LENGTH'], 
        frame_step=params['FRAME_STEP'])

    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


def _GET_SPECTROGRAM_LABEL_ID(audio, label):
  """
  Compute the spectrogram and label ID for the given audio and label.

  Parameters:
  audio (Tensor): Input audio data.
  label: Input label data.

  Returns:
  Tuple: A tuple containing the spectrogram and label ID.
  """
  labels = params['LABELS']
  spectrogram = _GET_SPECTROGRAM(audio)
  label_id = tf.math.argmax(label == labels)
  return spectrogram, label_id


def build_dataset(files, AT):
  # Create a dataset from the list of files
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  
  # Map the _GET_WAVEFORM_LABEL function to each file in parallel
  wf_ds = files_ds.map(
      map_func=_GET_WAVEFORM_LABEL,
      num_parallel_calls=AT)
  
  # Map the _GET_SPECTROGRAM_LABEL_ID function to each waveform in parallel
  spec_ds = wf_ds.map(
      map_func=_GET_SPECTROGRAM_LABEL_ID,
      num_parallel_calls=AT)
  
  return spec_ds

def finalize_dataset(data: tf.data.Dataset, AUTOTUNE) -> tf.data.Dataset:
    """
    Finalize the dataset by shuffling, batching, caching, and prefetching.

    Args:
        data: The input dataset to be processed.
        AUTOTUNE: The parameter for prefetching.

    Returns:
        The finalized dataset ready for training.
    """
    batch_size: int = params['BATCH_SIZE']
    buffer_size: int = params['BUFFER_SIZE']

    data = data.shuffle(buffer_size).batch(batch_size).cache().prefetch(AUTOTUNE)

    return data
