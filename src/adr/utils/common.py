from dataclasses import dataclass
import numpy as np
from pathlib import Path
import os
from functools import wraps
from typing import Any, Callable, Type, TypeVar, Generic, Union, IO
from types import MappingProxyType

T = TypeVar('T', bound=np.dtype)

class ArdArray(np.ndarray, Generic[T]):
    """
    A custom array class that enforces a specific dtype.
    
    This class is a subclass of numpy.ndarray with added type checking.
    It allows for the creation of arrays with a specified inner type.
    """
    
    inner_type: Type[T]
    
    def __new__(cls, input_array: Any) -> 'ArdArray[T]':
        """Create a new ArdArray instance."""
        obj = np.asarray(input_array, dtype=cls.inner_type).view(cls)
        return obj
    
    @classmethod
    def __class_getitem__(cls, item: Type[T]) -> Type['ArdArray[T]']:
        """Create a new ArdArray class with the specified inner type."""
        return type(f'ArdArray[{item.__name__}]', (cls,), {'inner_type': item})
    
    @classmethod
    def __get_validators__(cls):
        """Yield the validate_type method for Pydantic compatibility."""
        yield cls.validate_type
    
    @classmethod
    def validate_type(cls, val: Any) -> 'ArdArray[T]':
        """Validate and convert the input to a ArdArray with the correct dtype."""
        return cls(val)
    
    def __call__(self, *args, **kwargs):
        """Make the ArdArray instance callable."""
        # Define the behavior when the instance is called
        # For example, you might want to return a specific element or perform an operation
        return self  # Or any other desired operation

def validate_types(*types: Type) -> Callable:
    """
    A decorator that validates the types of function arguments.
    
    Args:
        *types: The expected types for each function argument.
    
    Returns:
        A decorator function that performs type checking.
    
    Raises:
        TypeError: If an argument doesn't match the expected type.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            import inspect
            bound_args = inspect.signature(func).bind(*args, **kwargs)
            for (name, value), expected_type in zip(bound_args.arguments.items(), types):
                if not isinstance(value, expected_type):
                    raise TypeError(f"Argument '{name}' must be of type {expected_type.__name__}, "
                                    f"not {type(value).__name__}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Example usage
@validate_types(int, float, ArdArray[np.float64])
def example_function(a: int, b: float, c: ArdArray[np.float64]) -> None:
    print(f"a: {a}, b: {b}, c: {c}")

import numpy as np

def to_categorical(y, num_classes=None):
    """Convert a class vector (integers) to binary class matrix.

    Args:
        y (array-like): Array of integer labels to be converted into a matrix
                        (must be in range 0 to num_classes - 1).
        num_classes (int): Total number of classes. If None, this will be inferred
                           from the highest label in y.

    Returns:
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical
class convert_digits: 
    """
    The convert_digits class provides methods to convert digits
    between their word and numerical representations. 
    It initializes a read-only dictionary of digit mappings 
    using the MappingProxyType from the types module. 
    The class can be used to convert a word representation of a digit to 
    its numerical value or vice versa.
    """
    def __init__(self) :
        
        """
        initializes the class by creating a read-only dictionary of 
        digit mappings using the MappingProxyType from the types module.
        
        """
        digits = {"zero":0,"one": 1, "two": 2, "three":3, "four":4, "five":5, "six":6, "seven":7, "eight":8, "nine":9}
        ar_to_en = {
            "واحد": "One",
            "اثنان": "Two",
            "ثلاثة": "Three",
            "أربعة": "Four",
            "خمسة": "Five",
            "ستة": "Six",
            "سبعة": "Seven",
            "ثمانية": "Eight",
            "تسعة": "Nine"
        }
        self.digits = MappingProxyType(digits)
        self.ar_to_en = MappingProxyType(ar_to_en)

    def convert_word_2_num(self, dig):
        """

        Args:
            dig (str): takes a word representation of a digit as input 

        Returns:
            int : returns the digit numerical value.
        """
        return self.digits[dig]
    
    def convert_num_2_word(self, dig):
        """

        Args:
            dig (int): takes a numerical value of a digit as input.

        Returns:
            str:  returns the digit word representation.
        """
        digits = {v: k for k, v in self.digits.items()}
        return digits[dig]
    
    def ardig_to_endig(self, arabic_number):
        """Converts Arabic number words (واحد to تسعة) to English digits (0 to 9).

        Args:
            arabic_number: The Arabic number word to convert.

        Returns:
            The equivalent English digit as a string, or None if the input is not a valid Arabic number word.
        """

        return self.ar_to_en[arabic_number]


    
    

def get_wav_path(file_name, aud_dir):
        path = os.path.join(aud_dir, file_name)
        return path
    
        
def get_wav_label(file_name):
    digit = file_name.split('-')[0]
    label = convert_digits().convert_word_2_num(digit)
    return label


def extract_extension_from_file(file : str):
        return Path(file).suffix.lower()[:]
    
def add_extension_to_file(file: str,
                          extension: str) -> str:
    """Add extension to a file.
    :param file: File path to which to add an extension
    :param extension: Extension to append to file (e.g., mp3)
    :return: File with extension
    """
    return file + "." + extension


def remove_extension_from_file(file: str) -> str:
    """
    :param file: File path to which to remove extension
    :return: File without extension
    """
    return str(Path(file).with_suffix(""))


def create_dir_hierarchy(dir: str):
    """Creates a hierarchy of directories, if it doesn't exists. Else,
    it doesn't do anything.
    :param dir: Path with directory hierarchy which should be created, if it
        doesn't exist
    """
    Path(dir).mkdir(parents=True, exist_ok=True)


def create_dir_hierarchy_from_file(file: str):
    """Creates a hierarchy of directories for a file, if it doesn't exists.
    Else, it doesn't do anything.
    :param file: File for which to create the relative dir hierarchy
    """
    dir = os.path.dirname(file)
    create_dir_hierarchy(dir)
    
    

@dataclass
class Signal:
    name: str
    samplerate: int
    data: np.array
    filepath: Union[str, Path, IO] 
    

