from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_path: str
    arclasses_path: Path
    data_file: Path
    csv_file: Path

@dataclass(frozen=True)
class DataTFIngestionConfig:
    root_dir: Path
    source_path: str
    dst_path:str
    
@dataclass(frozen=True)
class DataTFTrainConfig:
    root_dir: Path
    dst_path: str
    
@dataclass(frozen=True)
class DataTFEvaluationConfig:
    root_dir: Path
    data_path: str
    model_path:str

@dataclass(frozen=True)
class DataTFInferenceConfig:
    root_dir: Path
    audio_path: str
    model_path: str
    
@dataclass(frozen=True)
class DataPreprocessConfig:
    root_dir: Path
    source_path: str
    data_file: Path