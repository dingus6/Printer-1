import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

class FinancialDataPreprocessor:
    def __init__(self, hourly_data_path, fear_greed_data_path, export_path=None):
        self.hourly_data_path = hourly_data_path
        self.fear_greed_data_path = fear_greed_data_path
        self.hourly_scaler = StandardScaler()
        self.fear_greed_scaler = StandardScaler()
        self.export_path = export_path
        
        self.hourly_data = self._load_hourly_data()
        self.fear_greed_data = self._load_fear_greed_data()
        
    def _load_hourly_data(self):
        df = pd.read_csv(self.hourly_data_path)
        
        # Drop the MarketCap column if it exists (not needed by model)
        if 'MarketCap' in df.columns:
            df = df.drop(columns=['MarketCap'])
            
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
        df.set_index('Timestamp', inplace=True)
        
        # Add time-based features
        df['hour_of_day'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Calculate hour of day as sine and cosine components for cyclical nature
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day']/24)
        
        # Calculate day of week as sine and cosine components
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
        
        df['PriceRange'] = df['High'] - df['Low']
        df['Volatility'] = df['PriceRange'] / df['Low']
        df['PriceChangePercent'] = (df['Close'] - df['Open']) / df['Open']
        df['IsUp'] = (df['Close'] > df['Open']).astype(int)
        df['VolumeChange'] = df['Volume'].pct_change().fillna(0)
        df['Returns'] = df['Close'].pct_change().fillna(0)
        
        df['MA_3h'] = df['Close'].rolling(window=3).mean().ffill().bfill()
        df['MA_6h'] = df['Close'].rolling(window=6).mean().ffill().bfill()
        df['MA_12h'] = df['Close'].rolling(window=12).mean().ffill().bfill()
        
        df['RSI_6h'] = self._calculate_rsi(df['Close'], 6)
        df['RSI_12h'] = self._calculate_rsi(df['Close'], 12)
        
        # Calculate spread estimate from volume (inverse relationship with volume)
        # Higher volume typically means lower spread
        df['SpreadEstimate'] = (1 / df['Volume']) * df['Close'] * 10000  # Scale for readability
        
        return df
    
    def _calculate_rsi(self, prices, period):
        delta = prices.diff()
        gain = delta.clip(lower=0).fillna(0)
        loss = -delta.clip(upper=0).fillna(0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss.replace(0, 1e-9)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)  # Fill NaN with neutral value
    
    def _load_fear_greed_data(self):
        df = pd.read_csv(self.fear_greed_data_path)
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        classification_dummies = pd.get_dummies(df['fng_classification'], prefix='fng')
        df = pd.concat([df, classification_dummies], axis=1)
        df.drop('fng_classification', axis=1, inplace=True)
        
        return df
    
    def prepare_data(self, window_size=4, horizon=1, train_ratio=0.8, export_data=True):
        merged_data = self._merge_hourly_and_fear_greed()
        
        # Export the combined data before additional processing
        if export_data and self.export_path:
            merged_data.reset_index().to_csv(f"{self.export_path}/combined_data.csv", index=False)
            print(f"Combined data exported to {self.export_path}/combined_data.csv")
        
        # Add any additional processing here if needed
        
        # Export the fully processed data with all calculated columns
        if export_data and self.export_path:
            merged_data.reset_index().to_csv(f"{self.export_path}/processed_data.csv", index=False)
            print(f"Processed data exported to {self.export_path}/processed_data.csv")
        
        X, y = self._create_sequences(merged_data, window_size, horizon)
        
        split_idx = int(len(X) * train_ratio)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'merged_data': merged_data  # Return the merged data for further inspection
        }
    
    def _merge_hourly_and_fear_greed(self):
        hourly_data_with_date = self.hourly_data.reset_index().copy()
        hourly_data_with_date['date'] = hourly_data_with_date['Timestamp'].dt.date
        hourly_data_with_date['date'] = pd.to_datetime(hourly_data_with_date['date'])
        
        fear_greed_data = self.fear_greed_data.reset_index().copy()
        
        merged_data = pd.merge(
            hourly_data_with_date,
            fear_greed_data,
            left_on='date',
            right_on='date',
            how='left'
        )
        
        merged_data.set_index('Timestamp', inplace=True)
        merged_data.drop('date', axis=1, inplace=True, errors='ignore')
        
        # Forward fill missing fear/greed values
        cols_to_fill = [col for col in merged_data.columns if col not in self.hourly_data.columns]
        merged_data[cols_to_fill] = merged_data[cols_to_fill].ffill()
        
        return merged_data
    
    def _create_sequences(self, data, window_size, horizon):
        X = []
        y = []
        
        target_cols = ['IsUp', 'Returns', 'PriceChangePercent', 'Volatility', 'SpreadEstimate']
        feature_cols = [col for col in data.columns if col not in target_cols]
        
        self.feature_columns = feature_cols
        print(f"Creating sequences with {len(feature_cols)} features: {feature_cols}")
        
        # Split data into train and validation before normalization
        train_size = int(len(data) * 0.8)
        train_data = data.iloc[:train_size]
        val_data = data.iloc[train_size:]
        
        # Fit scalers on training data only
        self.hourly_scaler.fit(train_data[feature_cols].values)
        self.target_scaler = StandardScaler()
        self.target_scaler.fit(train_data[target_cols[1:]].values)
        
        # Transform both train and validation data
        train_features = self.hourly_scaler.transform(train_data[feature_cols].values)
        val_features = self.hourly_scaler.transform(val_data[feature_cols].values)
        
        train_targets = self.target_scaler.transform(train_data[target_cols[1:]].values)
        val_targets = self.target_scaler.transform(val_data[target_cols[1:]].values)
        
        # Create normalized dataframes
        train_normalized = pd.DataFrame(train_features, index=train_data.index, columns=feature_cols)
        val_normalized = pd.DataFrame(val_features, index=val_data.index, columns=feature_cols)
        
        # Add normalized targets
        train_normalized[target_cols[0]] = train_data[target_cols[0]]  # Keep IsUp as is
        val_normalized[target_cols[0]] = val_data[target_cols[0]]
        
        for i, col in enumerate(target_cols[1:]):
            train_normalized[col] = train_targets[:, i]
            val_normalized[col] = val_targets[:, i]
        
        # Export normalized data if export path is provided
        if self.export_path:
            train_normalized.reset_index().to_csv(f"{self.export_path}/normalized_train_data.csv", index=False)
            val_normalized.reset_index().to_csv(f"{self.export_path}/normalized_val_data.csv", index=False)
            print(f"Normalized data exported to {self.export_path}/normalized_train_data.csv and {self.export_path}/normalized_val_data.csv")
        
        # Create sequences for both train and validation
        for df in [train_normalized, val_normalized]:
            for i in range(len(df) - window_size - horizon + 1):
                window_data = df.iloc[i:i+window_size][feature_cols].values
                target_idx = i + window_size + horizon - 1
                
                if np.isnan(window_data).any() or target_idx >= len(df):
                    continue
                    
                X.append(window_data)
                
                y.append([
                    df.iloc[target_idx]['IsUp'],
                    df.iloc[target_idx]['Volatility'],
                    df.iloc[target_idx]['PriceChangePercent'],
                    df.iloc[target_idx]['SpreadEstimate']
                ])
        
        X_array = np.array(X)
        y_array = np.array(y)
        
        print(f"Created {len(X_array)} sequences with shape {X_array.shape}")
        
        return X_array, y_array
    
    def get_dataloaders(self, window_size=4, horizon=1, batch_size=64, train_ratio=0.8, export_data=True):
        data_dict = self.prepare_data(window_size, horizon, train_ratio, export_data)
        
        train_dataset = FinancialDataset(
            data_dict['X_train'],
            data_dict['y_train']
        )
        
        val_dataset = FinancialDataset(
            data_dict['X_val'],
            data_dict['y_val']
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=4
        )
        
        return train_loader, val_loader, data_dict['merged_data']
    
    def get_feature_dim(self):
        if hasattr(self, 'feature_columns'):
            return len(self.feature_columns)
        else:
            # Estimate feature dimension before sequence creation
            target_cols = ['IsUp', 'Returns', 'PriceChangePercent', 'Volatility', 'SpreadEstimate']
            merged_data = self._merge_hourly_and_fear_greed()
            feature_cols = [col for col in merged_data.columns if col not in target_cols]
            
            print(f"Estimated feature dimension: {len(feature_cols)}")
            return len(feature_cols)
    
    def export_data_sample(self, sample_size=100):
        """
        Export a sample of the data at different processing stages for validation
        """
        if not self.export_path:
            print("Export path not specified. Unable to export data samples.")
            return
            
        # Export sample of raw hourly data
        self.hourly_data.head(sample_size).reset_index().to_csv(
            f"{self.export_path}/sample_hourly_raw.csv", index=False
        )
        
        # Export sample of raw fear/greed data
        self.fear_greed_data.head(sample_size).reset_index().to_csv(
            f"{self.export_path}/sample_fear_greed_raw.csv", index=False
        )
        
        # Export sample of merged data
        merged_data = self._merge_hourly_and_fear_greed()
        merged_data.head(sample_size).reset_index().to_csv(
            f"{self.export_path}/sample_merged_data.csv", index=False
        )
        
        print(f"Data samples exported to {self.export_path}")
        
        # Return a list of column names for reference
        return {
            'hourly_columns': list(self.hourly_data.columns),
            'fear_greed_columns': list(self.fear_greed_data.columns),
            'merged_columns': list(merged_data.columns)
        }


class FinancialDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        
        self.y_direction = torch.tensor(y[:, 0], dtype=torch.float32).unsqueeze(1)
        self.y_volatility = torch.tensor(y[:, 1], dtype=torch.float32).unsqueeze(1)
        self.y_price_change = torch.tensor(y[:, 2], dtype=torch.float32).unsqueeze(1)
        self.y_spread = torch.tensor(y[:, 3], dtype=torch.float32).unsqueeze(1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (
            self.X[idx],
            {
                'direction': self.y_direction[idx],
                'volatility': self.y_volatility[idx],
                'price_change': self.y_price_change[idx],
                'spread': self.y_spread[idx]
            }
        )
