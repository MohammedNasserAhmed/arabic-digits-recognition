import numpy as np
from adr.utils.dataset import SeqDataset
from adr import logger
from operator import itemgetter
import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
from adr.entity.config_entity import DataTrainingConfig

# results = {
#     'dataset_size': [],
#     'X_train_shape': [],
#     'X_test_shape': [],
#     'y_train_shape': [],
#     'y_test_shape': [],
#     'model_metrics': [],
#     'model_scores': [],
#     'model_evaluation': []
# }
class DataTraining:
    def __init__(self,
                 config : DataTrainingConfig
    ):
        self.config=config
        self._origin_data_path = self.config.origin_data_path
        self.num_classes = None
        self.input_shape=None
    def load_npz_data(self, data_path):
        """
        Load data from an NPZ file.

        Args:
            file_path (str): Path to the NPZ file.

        Returns:
            SeqDataset: Loaded dataset.
        """
        try:
            data = np.load(data_path, allow_pickle=True)
           
            features, targets, lengths, classes = itemgetter('features', 'targets', 'lengths','classes')(data)
            logger.info(f"features shape: {features.shape}")
            idx = np.argwhere(np.isin(targets, classes)).flatten()
        
            return SeqDataset(
                features = features[idx],
                targets = targets[idx],
                lengths= lengths[idx],
                classes=classes)
        
            
        except Exception as e:
            logger.error(f"Error loading NPZ file: {e}")
            raise

    def prepare_data(self):
        """
        Prepare data for a CNN model.

        Args:
            dataset (SeqDataset): The dataset to prepare.
            train_shape (tuple): The desired shape for training data.
            test_shape (tuple): The desired shape for testing data.

        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        logger.info("Loading data from NPZ file...")
        dataset = self.load_npz_data(self._origin_data_path)
        
        logger.info(f"Dataset loaded")
        
        # Split the data
        logger.info("Preparing data for training...")
        train_data, test_data = dataset.split_data(split_size=0.2, shuffle=True, stratify=True)
       
        X_train, y_train,_,_ = train_data._get_data()
        X_test, y_test,_,_ = test_data._get_data()
        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"y_train shape: {y_train.shape}")
        # Reshape for CNN (add channel dimension)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],X_train.shape[2], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1],X_test.shape[2] , 1)
       
        
        

        # Convert labels to categorical
        self.num_classes = len(np.unique(dataset._targets))
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=self.num_classes)
        y_test = tf.keras.utils.to_categorical(y_test,  num_classes=self.num_classes)
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Testing data shape: {X_test.shape}")
        logger.info(f"Training labels shape: {y_train.shape}")
        logger.info(f"Testing labels shape: {y_test.shape}")
        
        # results['dataset_size'].append(len(dataset._features))
        # results['X_train_shape'].append(X_train.shape)
        # results['X_test_shape'].append(X_test.shape)
        # results['y_train_shape'].append(y_train.shape)
        # results['y_test_shape'].append(y_test.shape)

        return X_train, X_test, y_train, y_test
    
    def build_and_train_model(self, X_train, X_test, y_train, y_test):
        self.model = self._create_model()
        self._compile_model(self.model)
        self._train_model(self.model, X_train, X_test, y_train, y_test)
        # results['model_metrics'].append(self.model.metrics_names)
        # results['model_scores'].append(self.model.evaluate(X_test, y_test))
        # results['model_evaluation'].append(self.model.history.history)  # Store the training history
        self.model.save(self.config.dst_path)

    def _create_model(self, filters=32, kernel_size=(3, 3), dense_units=256, dropout_rate=0.2):
        model = models.Sequential([
            layers.Conv2D(filters, kernel_size, activation='relu', padding='valid', input_shape=self.input_shape),  
            layers.MaxPooling2D(2, padding='same'),
            layers.Conv2D(128, kernel_size, activation='relu', padding='valid'),
            layers.MaxPooling2D(2, padding='same'),
            layers.Dropout(dropout_rate),
            layers.Conv2D(128, kernel_size, activation='relu', padding='valid'),
            layers.MaxPooling2D(2, padding='same'),
            layers.Dropout(dropout_rate),
            layers.GlobalAveragePooling2D(),
            layers.Dense(dense_units, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model


    def _compile_model(self, model):
        model.compile(
            loss = 'categorical_crossentropy', optimizer = 'adam', metrics = 'acc'
        )
        model.summary()
        

    def _train_model(self, model, X_train, X_test, y_train, y_test):
       
        EPOCHS = 100
        batch_size = 12
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, 
                                                          verbose=1, mode='auto',baseline=None,
                                                          restore_best_weights=True)
        model.fit(X_train,y_train ,
            validation_data=(X_test,y_test),
            epochs=100,
            callbacks = [early_stopping],batch_size=batch_size)
    
    def get_predictions(self, X_test):
        # Get the predicted probabilities
        y_pred_proba = self.model.predict(X_test)
        
        # Convert probabilities to class labels
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        return y_pred, y_pred_proba
    def process(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.input_shape=X_train.shape[1:]
        logger.info(f'Input shape: {self.input_shape}')
        self.build_and_train_model( X_train, X_test, y_train, y_test)
        
       