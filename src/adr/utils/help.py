import os
import yaml
from adr import logger
import json
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (Path): path like input

    Raises:
        ValueError: if yaml file is empty
        FileNotFoundError: if the file is not found
        yaml.YAMLError: if there is an error loading the YAML file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            if content is None or not content:
                raise ValueError("yaml file is empty")
            logger.info(f"yaml file: {path_to_yaml} loaded successfully. Content size: {len(content)}")
            return ConfigBox(content)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path_to_yaml}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error loading YAML file: {e}")
    


@ensure_annotations
def create_directories(path_to_directories: list, log_creation: bool = True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        log_creation (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    total_dirs = len(path_to_directories)
    for path in path_to_directories:
        try:
            os.makedirs(path, mode=0o770, exist_ok=True)
        except PermissionError as e:
            logger.error(f"Permission error while creating directory at: {path}. Error: {e}")
        except Exception as e:
            logger.error(f"Error creating directory at: {path}. Error: {e}")

        if log_creation and total_dirs > 1:
            logger.info(f"created directory at: {path}")

    logger.info(f"Total directories created: {total_dirs}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """


    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except IOError as e:
        logger.error(f"Error saving json file at {path}: {e}")
        return

    logger.info(f"json file saved at: {path}")




@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)

