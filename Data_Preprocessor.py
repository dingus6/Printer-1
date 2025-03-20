import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class CryptoDataPreprocessor:
    def __init__(self, candle_data_path, fear_greed_data_path, liquidation_data_path=None):
        """
        Initialize the preprocessor with file paths.
        
        Parameters:
        -----------
        candle_data_path : str
            Path to the candle/price data CSV
        fear_greed_data_path : str
            Path to the fear & greed index CSV
        liquidation_data_path : str, optional
            Path to the liquidation data CSV
        """
        self.candle_data_path = candle_data_path
        self.fear_greed_data_path = fear_greed_data_path
        self.liquidation_data_path = liquidation_data_path
        self.scaler = StandardScaler()
        
    def compute_rsi(self, series, window):
        """
        Compute the Relative Strength Index (RSI) for a given series and window.
        """
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean()
        # Use a small constant to avoid division by zero
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def load_and_process_data(self):
        """
        Load and preprocess all data sources, then merge them.
        """
        # Load all data sources
        candle_df = self.process_candle_data()
        fear_greed_df = self.process_fear_greed_data(candle_df.index)
        
        # Merge data
        data = candle_df.merge(fear_greed_df, left_index=True, right_index=True, 
                              how='left', suffixes=('', '_fg'))
        
        # Add liquidation data if provided
        if self.liquidation_data_path:
            liq_df = self.process_liquidation_data(candle_df.index)
            data = data.merge(liq_df, left_index=True, right_index=True, 
                             how='left', suffixes=('', '_liq'))
        
        # Fill missing values with forward fill only
        data.fillna(method='ffill', inplace=True)
        data.dropna(inplace=True)  # Drop any remaining NAs
        
        return data
    
    def process_candle_data(self):
        """
        Process the candle/price data with feature engineering.
        """
        # Load data
        candle_df = pd.read_csv(self.candle_data_path)
        
        # Convert timestamp to datetime and set as index
        candle_df['datetime'] = pd.to_datetime(candle_df['timestamp'], unit='s')
        candle_df.set_index('datetime', inplace=True)
        candle_df.sort_index(inplace=True)
        
        # Feature Engineering
        candle_df['PriceRange'] = candle_df['high'] - candle_df['low']
        candle_df['Volatility'] = candle_df['PriceRange'] / candle_df['low']
        candle_df['PriceChangePercent'] = (candle_df['close'] - candle_df['open']) / candle_df['open']
        candle_df['IsUp'] = (candle_df['close'] > candle_df['open']).astype(int)
        candle_df['Returns'] = candle_df['close'].pct_change()
        candle_df['VolumeChange'] = candle_df['volume'].pct_change()
        
        # Moving Averages with different windows
        candle_df['MA_3h'] = candle_df['close'].rolling(window=3, min_periods=1).mean()
        candle_df['MA_6h'] = candle_df['close'].rolling(window=6, min_periods=1).mean()
        candle_df['MA_12h'] = candle_df['close'].rolling(window=12, min_periods=1).mean()
        candle_df['MA_24h'] = candle_df['close'].rolling(window=24, min_periods=1).mean()
        
        # Forward fill any NaN values from rolling calculations
        candle_df.fillna(method='ffill', inplace=True)
        
        # RSI calculations
        candle_df['RSI_6h'] = self.compute_rsi(candle_df['close'], 6)
        candle_df['RSI_12h'] = self.compute_rsi(candle_df['close'], 12)
        candle_df['RSI_24h'] = self.compute_rsi(candle_df['close'], 24)
        
        # Add spread estimate (from second code)
        candle_df['SpreadEstimate'] = (1 / candle_df['volume']) * candle_df['close'] * 10000
        
        return candle_df
    
    def process_fear_greed_data(self, hourly_index):
        """
        Process fear & greed data and upsample to hourly frequency.
        """
        # Load data
        fear_greed_df = pd.read_csv(self.fear_greed_data_path)
        
        # Convert date column and set as index
        fear_greed_df['date'] = pd.to_datetime(fear_greed_df['date'], dayfirst=True)
        fear_greed_df.set_index('date', inplace=True)
        fear_greed_df.sort_index(inplace=True)
        
        # One-hot encode the classification (from second code)
        if 'fng_classification' in fear_greed_df.columns:
            classification_dummies = pd.get_dummies(fear_greed_df['fng_classification'], prefix='fng')
            fear_greed_df = pd.concat([fear_greed_df, classification_dummies], axis=1)
            fear_greed_df.drop('fng_classification', axis=1, inplace=True)
        
        # Upsample to hourly frequency
        hourly_fear_greed = fear_greed_df.reindex(hourly_index, method='ffill')
        
        return hourly_fear_greed
    
    def process_liquidation_data(self, hourly_index):
        """
        Process liquidation data and upsample to hourly frequency.
        """
        # Load data
        liq_df = pd.read_csv(self.liquidation_data_path)
        
        # Convert datetime column and set as index
        liq_df['datetime'] = pd.to_datetime(liq_df['datetime'])
        liq_df.set_index('datetime', inplace=True)
        liq_df.sort_index(inplace=True)
        
        # Calculate additional features
        liq_df['OI_Change'] = liq_df['open_interest'].pct_change()
        
        # Upsample to hourly frequency
        liq_hourly = liq_df.reindex(hourly_index, method='ffill')
        
        return liq_hourly
    
    def prepare_data(self, sequence_length=24, forecast_horizon=1, train_ratio=0.8):
        """
        Prepare data for model training with flexible sequence length and horizon.
        
        Parameters:
        -----------
        sequence_length : int
            Length of input sequences (lookback window)
        forecast_horizon : int
            How far ahead to predict
        train_ratio : float
            Ratio for train/test split
        
        Returns:
        --------
        dict: Contains train and test data loaders and other metadata
        """
        # Get processed data
        data = self.load_and_process_data()
        
        # Create multiple target variables (from second code)
        data['Target_Return'] = data['Returns'].shift(-forecast_horizon)
        data['Target_Volatility'] = data['Volatility'].shift(-forecast_horizon)
        data['Target_PriceChange'] = data['PriceChangePercent'].shift(-forecast_horizon)
        data['Target_IsUp'] = data['IsUp'].shift(-forecast_horizon)
        
        # Drop rows with NA targets
        data.dropna(inplace=True)
        
        # Define feature columns (excluding targets and other non-feature columns)
        target_cols = ['Target_Return', 'Target_Volatility', 'Target_PriceChange', 'Target_IsUp']
        exclude_cols = target_cols + ['timestamp'] if 'timestamp' in data.columns else target_cols
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        # Train-test split (chronological)
        split_idx = int(len(data) * train_ratio)
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        # Fit scaler on training data
        self.scaler.fit(train_data[feature_cols])
        
        # Transform features
        train_features = self.scaler.transform(train_data[feature_cols])
        test_features = self.scaler.transform(test_data[feature_cols])
        
        # Prepare target variables
        train_targets = {
            'return': train_data['Target_Return'].values,
            'volatility': train_data['Target_Volatility'].values,
            'price_change': train_data['Target_PriceChange'].values,
            'is_up': train_data['Target_IsUp'].values
        }
        
        test_targets = {
            'return': test_data['Target_Return'].values,
            'volatility': test_data['Target_Volatility'].values,
            'price_change': test_data['Target_PriceChange'].values,
            'is_up': test_data['Target_IsUp'].values
        }
        
        # Create dataset objects and data loaders
        train_dataset = CryptoDataset(
            train_features, train_targets, sequence_length
        )
        
        test_dataset = CryptoDataset(
            test_features, test_targets, sequence_length
        )
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        return {
            'train_loader': train_loader,
            'test_loader': test_loader,
            'feature_dim': len(feature_cols),
            'feature_names': feature_cols,
            'scaler': self.scaler
        }

class CryptoDataset(Dataset):
    """
    PyTorch Dataset for cryptocurrency data.
    """
    def __init__(self, features, targets, seq_length):
        """
        Initialize dataset with features and targets.
        
        Parameters:
        -----------
        features : numpy.ndarray
            Scaled feature values
        targets : dict
            Dictionary of target variables
        seq_length : int
            Length of input sequences
        """
        self.features = features
        self.targets = targets
        self.seq_length = seq_length
        
        # Create sequences
        self.X, self.y = self._create_sequences()
        
    def _create_sequences(self):
        """
        Create sequences for time series modeling.
        """
        X_seq = []
        y_dict = {k: [] for k in self.targets.keys()}
        
        for i in range(len(self.features) - self.seq_length):
            # Input sequence
            X_seq.append(self.features[i:i+self.seq_length])
            
            # Target values (at the end of the sequence)
            for k in self.targets.keys():
                y_dict[k].append(self.targets[k][i+self.seq_length])
        
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32)
        
        y_tensors = {
            k: torch.tensor(np.array(v), dtype=torch.float32).unsqueeze(1)
            for k, v in y_dict.items()
        }
        
        return X_tensor, y_tensors
    
    def __len__(self):
        """Return the number of sequences in the dataset."""
        return len(self.X)
    
    def __getitem__(self, idx):
        """Get sequence and targets at index idx."""
        return self.X[idx], {k: v[idx] for k, v in self.y.items()}

    # Example usage:
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = CryptoDataPreprocessor(
        candle_data_path='C:\\root\\printer1\\candle_data.csv',
        fear_greed_data_path='C:\\root\\printer1\\fng.csv',
        liquidation_data_path='C:\\root\\printer1\\liquidation_oi_data.csv'
    )
    
    # Prepare data with custom sequence length and forecast horizon
    data_dict = preprocessor.prepare_data(sequence_length=24, forecast_horizon=1)
    
    # Access data loaders and other information
    train_loader = data_dict['train_loader']
    test_loader = data_dict['test_loader']
    feature_dim = data_dict['feature_dim']
    
    print(f"Feature dimension: {feature_dim}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Testing batches: {len(test_loader)}")
    
    # Example of accessing a batch
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        print(f"Batch {batch_idx} shape: {inputs.shape}")
        print(f"Target shapes: {[(k, v.shape) for k, v in targets.items()]}")
        break
