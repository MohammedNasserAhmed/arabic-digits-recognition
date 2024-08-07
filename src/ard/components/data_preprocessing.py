import numpy as np
from src.ard.utils.dataset import SeqDataset
from sklearn.decomposition import PCA
from operator import itemgetter
from ard.entity.config_entity import DataPreprocessConfig
class DataPreprocessing:
    def __init__(self,
                 config : DataPreprocessConfig
    ):
        self.config=config
        self._source_path = self.config.source_path
        self._data_file = self.config.data_file
    def get_data(self, digits : list = range(10)):
        data = np.load(self._source_path)

        # Fetch arrays from loaded file
        features, targets, lengths = itemgetter('features', 'targets', 'lengths')(data)
        
        pca = PCA(n_components=13)
        pca.fit(features)
        features = pca.transform(features)
        idx = np.argwhere(np.isin(targets, digits)).flatten()
        ranges = SeqDataset._get_idxs(lengths)[idx]
       
        return SeqDataset(
            features = np.vstack(np.array([x for x in SeqDataset._iter_X(features, ranges)], dtype=object)),
            targets = targets[idx],
            lengths= lengths[idx],
            classes=digits, path =self._data_file)
        
        
 