import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CryptoTransformer:
    """
    Lightweight Transformer model for cryptocurrency price prediction.
    Implements a simplified transformer architecture for time series forecasting.
    """
    
    def __init__(self, 
                seq_length=24,
                n_features=None,
                d_model=64,
                num_heads=4,
                dropout_rate=0.1,
                ff_dim=128,
                num_transformer_blocks=2):
        """
        Initialize the CryptoTransformer model.
        
        Args:
            seq_length: Number of time steps in each input sequence
            n_features: Number of features at each time step
            d_model: Dimensionality of the transformer model
            num_heads: Number of attention heads
            dropout_rate: Dropout rate
            ff_dim: Hidden layer size in feed forward network inside transformer
            num_transformer_blocks: Number of transformer blocks to stack
        """
        self.seq_length = seq_length
        self.n_features = n_features
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.model = None
    
    def _create_positional_encoding(self, position, d_model):
        """Create positional encoding for transformer input."""
        def get_angles(pos, i, d_model):
            angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
            return pos * angle_rates
        
        angle_rads = get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        
        # Apply sin to even indices in the array
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        
        # Apply cos to odd indices in the array
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def _transformer_encoder(self, inputs):
        """Create a transformer encoder block."""
        # Normalization and attention
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        attention_output = layers.MultiHeadAttention(
            key_dim=self.d_model // self.num_heads,
            num_heads=self.num_heads,
            dropout=self.dropout_rate
        )(x, x)
        attention_output = layers.Dropout(self.dropout_rate)(attention_output)
        x1 = layers.Add()([attention_output, inputs])
        
        # Feed-forward network
        x2 = layers.LayerNormalization(epsilon=1e-6)(x1)
        x2 = layers.Conv1D(filters=self.ff_dim, kernel_size=1, activation='relu')(x2)
        x2 = layers.Dropout(self.dropout_rate)(x2)
        x2 = layers.Conv1D(filters=self.d_model, kernel_size=1)(x2)
        x2 = layers.Dropout(self.dropout_rate)(x2)
        
        return layers.Add()([x1, x2])
    
    def build_model(self):
        """
        Build and compile the transformer model.
        
        Returns:
            Compiled Keras model
        """
        if self.n_features is None:
            raise ValueError("Number of features (n_features) must be specified")
        
        # Input layer
        inputs = keras.Input(shape=(self.seq_length, self.n_features))
        
        # Embedding layer to convert inputs to d_model dimensions
        embedding = layers.Conv1D(filters=self.d_model, kernel_size=1, activation='relu')(inputs)
        
        # Add positional encoding
        pos_encoding = self._create_positional_encoding(self.seq_length, self.d_model)
        x = embedding + pos_encoding
        
        # Dropout for regularization
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Stack transformer blocks
        for _ in range(self.num_transformer_blocks):
            x = self._transformer_encoder(x)
        
        # Global average pooling to get fixed-size output
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layer for processing pooled features
        x = layers.Dense(self.d_model, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Multiple output heads
        # 1. Direction prediction (binary classification)
        direction_output = layers.Dense(1, activation='sigmoid', name='direction')(x)
        
        # 2. Category prediction (multi-class)
        # 5 categories: Large_Down, Medium_Down, Small, Medium_Up, Large_Up
        category_output = layers.Dense(5, activation='softmax', name='category')(x)
        
        # 3. Percentage change prediction (regression)
        regression_output = layers.Dense(1, name='regression')(x)
        
        # Create model with multiple outputs
        self.model = keras.Model(
            inputs=inputs,
            outputs=[direction_output, category_output, regression_output]
        )
        
        # Compile model with appropriate loss functions
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss={
                'direction': 'binary_crossentropy',
                'category': 'categorical_crossentropy',
                'regression': 'mse'
            },
            metrics={
                'direction': ['accuracy'],
                'category': ['accuracy'],
                'regression': ['mae']
            },
            loss_weights={
                'direction': 1.0,
                'category': 1.0,
                'regression': 0.5
            }
        )
        
        logger.info("Transformer model built and compiled")
        logger.info(f"Input shape: {inputs.shape}")
        logger.info(f"Model parameters: {self.model.count_params():,}")
        
        return self.model
    
    def fit(self, X, y, validation_split=0.2, epochs=50, batch_size=32, callbacks=None):
        """
        Train the model.
        
        Args:
            X: Input features (shape: [samples, seq_length, n_features])
            y: Dictionary of targets with keys 'direction', 'category', 'regression'
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Batch size
            callbacks: Optional list of Keras callbacks
            
        Returns:
            Training history
        """
        if self.model is None:
            self.n_features = X.shape[2]
            self.build_model()
        
        if callbacks is None:
            # Define default callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )
            ]
        
        # Training
        history = self.model.fit(
            X,
            {
                'direction': y['direction'],
                'category': y['category'],
                'regression': y['regression']
            },
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """
        Make predictions with the model.
        
        Args:
            X: Input features (shape: [samples, seq_length, n_features])
            
        Returns:
            Dictionary of predictions with keys 'direction', 'category', 'regression'
        """
        if self.model is None:
            raise ValueError("Model must be built and trained before making predictions")
        
        # Get raw predictions
        direction_pred, category_pred, regression_pred = self.model.predict(X)
        
        # Process predictions
        direction_binary = (direction_pred > 0.5).astype(int)
        category_classes = np.argmax(category_pred, axis=1)
        
        # Map category indices to labels
        category_map = {
            0: 'Large_Down',    # <= -5%
            1: 'Medium_Down',   # -5% to -1%
            2: 'Small',         # -1% to 1%
            3: 'Medium_Up',     # 1% to 5%
            4: 'Large_Up'       # >= 5%
        }
        category_labels = np.array([category_map[i] for i in category_classes])
        
        # Return dictionary of predictions
        predictions = {
            'direction': direction_binary,
            'direction_prob': direction_pred,
            'category': category_labels,
            'category_prob': category_pred,
            'price_change_pct': regression_pred.flatten()
        }
        
        return predictions
    
    def save(self, filepath):
        """Save the model to a file."""
        if self.model is None:
            raise ValueError("Model must be built before saving")
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load the model from a file."""
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
        
        # Update model parameters based on the loaded model
        input_shape = self.model.input_shape
        self.seq_length = input_shape[1]
        self.n_features = input_shape[2]
        
        return self.model


# Example usage
if __name__ == "__main__":
    # Synthetic data for demonstration
    seq_length = 24
    n_features = 50
    n_samples = 1000
    
    # Generate random sequences
    X = np.random.randn(n_samples, seq_length, n_features)
    
    # Generate random targets
    y_direction = np.random.randint(0, 2, size=(n_samples,))
    y_category = np.random.randint(0, 5, size=(n_samples,))
    y_category_onehot = np.eye(5)[y_category]
    y_regression = np.random.randn(n_samples,) * 5
    
    y = {
        'direction': y_direction,
        'category': y_category_onehot,
        'regression': y_regression
    }
    
    # Create and train model
    model = CryptoTransformer(seq_length=seq_length, n_features=n_features)
    model.build_model()
    
    # Train for just a few epochs with synthetic data
    history = model.fit(X, y, epochs=5, batch_size=32)
    
    # Make predictions
    predictions = model.predict(X[:5])
    
    # Print sample predictions
    for i in range(5):
        logger.info(f"Sample {i+1}:")
        logger.info(f"  Direction: {'Up' if predictions['direction'][i] == 1 else 'Down'} ({predictions['direction_prob'][i][0]:.4f})")
        logger.info(f"  Category: {predictions['category'][i]} (Probs: {predictions['category_prob'][i]})")
        logger.info(f"  Price Change %: {predictions['price_change_pct'][i]:.2f}%")
