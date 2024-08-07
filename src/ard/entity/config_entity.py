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
class DataPreprocessConfig:
    root_dir: Path
    source_path: str
    data_file: Path