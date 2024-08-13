import numpy as np
from adr.utils.dataset import SeqDataset
from sklearn.decomposition import PCA
from operator import itemgetter
from adr import logger
from adr.entity.config_entity import DataPreprocessConfig
class DataPreprocessing:
    def __init__(self,
                 config : DataPreprocessConfig
    ):
        self.config=config
        self._source_path = self.config.source_path
        self._data_file = self.config.data_file
    def get_data(self):
        data = np.load(self._source_path, allow_pickle=True)

        # Fetch arrays from loaded file
        features, targets, lengths, classes = itemgetter('features', 'targets', 'lengths', 'classes')(data)
        # Assuming features is of shape (n_samples, n_timesteps, n_features)
        _, n_timesteps, _ = features.shape
        
        pca = PCA(n_components=40)
        # Apply PCA to each time step
        features = np.array([
            pca.fit_transform(features[:, t, :]) for t in range(n_timesteps)
        ]).transpose(1, 0, 2)  # Reshape back to original structure if needed
        
        idx = np.argwhere(np.isin(targets, classes)).flatten()
        logger.info(f"features shape :{features.shape} type:{type(features)}")
        return SeqDataset(
            features = features[idx],
            targets = targets[idx],
            lengths= lengths[idx],
            classes=classes, path =self._data_file)
        
        
 