from __future__ import annotations
from adr import logger
import warnings
from typing import Optional, Tuple, Union, IO
import pathlib
import numpy as np
from sklearn.model_selection import train_test_split
class SeqDataset:
    """Utility wrapper for a generic sequential dataset."""

    def __init__(
        self,
        features: np.array,
        targets: Optional[np.array] = None,
        classes: Optional[np.array[int]] = None,
        path: Optional[Union[str, pathlib.Path, IO]] = None,
        lengths: Optional[np.array[int]] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        )-> SeqDataset:
        self._features = features
        self._targets = targets
        self._classes = classes
        self._lengths = lengths
        self._path=path
        self._random_state =random_state

        self._features_targets = (self._features, self._targets)
        self._features_lengths = (self._features, self._lengths)
        self._features_targets_lengths = (self._features, self._targets, self._lengths)
        
        self._idxs = self._get_idxs(self._lengths)
    def shape(self)-> Tuple[int, int, int]:
        """
        Get the shape of the dataset.

        Returns
        -------
        shape: Tuple[int, int, int]
            A tuple containing (num_samples, max_sequence_length, num_features).
        """
        #num_samples = self._features
        #max_sequence_length = max(self._lengths) if self._lengths is not None else self._features.shape[1]
        #num_features = self._features.shape[-1] if self._features.ndim > 2 else 1

        return self._features.shape

    def __str__(self) -> str:
        """
        String representation of the dataset.

        Returns
        -------
        str
            A string describing the dataset shape and other relevant information.
        """
        shape = self.shape()
        return (f"SeqDataset(samples={shape[0]}, max_length={shape[1]}, "
                f"features={shape[2]}, classes={len(self._classes) if self._classes is not None else 'None'})")
    @staticmethod
    def _get_idxs(lengths):
        ends = np.cumsum(lengths)
        starts = np.zeros_like(ends)
        starts[1:] = ends[:-1]
        return np.c_[starts, ends]
    
    def __len__(self):
        return len(self._targets)

    def __getitem__(self, dx):
        return self._features[dx], self._targets[dx], self._lengths[dx]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def _get_data(self):
        """Fetch the instances and labels.

        Returns
        -------
        X: array-like
            Data instances.

        y: array-like
            Labels corresponding to data instances.
        """
        return self._features, self._targets, self._lengths, self._classes
    
    @staticmethod
    def _iter_X(X, idxs):
        for start, end in idxs:
            yield X[start:end] 
            
    def iterator(self):
        """Generator for iterating through instances partitioned by class.

        Returns
        -------
        instances: generator yielding ``(instances, class)``
            Instances belong to each class.
        """
        if len(self._features) == len(self._targets):
            # Case when features is a 1D array of sequences
            _features_np = np.array(self._features, dtype=object)
            for clas in self._classes:
                yield _features_np[self._targets == clas].tolist(), clas
        else:
            # Case when features is a 2D array
            feature_length = self._features.shape[1] if self._features.ndim > 1 else 1
            start = 0
            for length, target in zip(self._lengths, self._targets):
                end = start + length
                yield self._features[start:end].reshape(-1, feature_length).tolist(), target
                start = end
        
    

    def split_data(self,
                   split_size: Optional[Union[int, float]] = None,
                   shuffle: bool = True,
                   stratify: bool = False)->Tuple[SeqDataset ,SeqDataset]:
        """Splits the dataset into two smaller :class:`Dataset` objects.

        Parameters
        ----------
        split_size: 0 < float < 1
            Proportion of instances to be allocated to the second split.

        stratify: bool
            Whether or not stratify the split on the labels such that each split
            has a similar distribution of labels.

        shuffle: bool
            Whether or not to shuffle the data before partitioniing it.

        Returns
        -------
        split_1: :class:`Dataset`
            First dataset split.

        split_2: :class:`Dataset`
            Second dataset split.
        """
        _train_size = 1 - split_size
        _test_size = split_size
        
        if stratify and self._targets is None:
            logger.warning('Cannot stratify with no provided outputs')
            stratify = None
        else:
            if stratify:
                if self._classes is None:
                    logger.warning('Cannot stratify on non-categorical outputs')
                    stratify = None
                else:
                    stratify = self._targets
            else:
                stratify = None

        idxs = np.arange(len(self._lengths))
        train_idxs, test_idxs = train_test_split(
            idxs,
            test_size= _test_size,
            train_size=_train_size,
            random_state=self._random_state,
            shuffle=shuffle,
            stratify=stratify
        )

        if self._targets is None:
            X_train, y_train = self[train_idxs], None
            X_test, y_test = self[test_idxs], None
        else:
            X_train, y_train, lengths_train = self[train_idxs]
            X_test, y_test, lengths_test = self[test_idxs]
        
        classes = self._classes

        data_train = SeqDataset(features= X_train, targets=y_train, lengths=lengths_train, classes=classes)
        data_test = SeqDataset(features= X_test, targets=y_test, lengths=lengths_test, classes=classes)

        return data_train, data_test

        
    def save(self, compress: bool = True):
        arrs = {
            'features': self._features,
            'lengths': self._lengths
        }

        if self._targets is not None:
            arrs['targets'] = self._targets

        if self._classes is not None:
            arrs['classes'] = self._classes

        save_fun = np.savez_compressed if compress else np.savez
        save_fun(self._path, **arrs)
        logger.info(
            "A npz file has been saved"
        )
        
        

   