import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union
import talib
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CryptoDataPreprocessor:
    """
    Preprocessor for cryptocurrency data.
    Handles feature engineering, technical indicators, and data preparation for model input.
    """
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        self.volume_scaler = StandardScaler()
        self.feature_scalers = {}
        self.feature_columns = []
        self.target_columns = []
        self.processed_data = None
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators from OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data (must have Open, High, Low, Close, Volume columns)
            
        Returns:
            DataFrame with additional technical indicator columns
        """
        # Create a copy to avoid modifying the original dataframe
        result = df.copy()
        
        # Basic sanity check
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"Missing required columns for technical indicators: {missing}")
        
        try:
            # RSI (Relative Strength Index)
            result['RSI'] = talib.RSI(df['Close'].values, timeperiod=14)
            
            # Moving Averages
            result['SMA_7'] = talib.SMA(df['Close'].values, timeperiod=7)
            result['SMA_25'] = talib.SMA(df['Close'].values, timeperiod=25)
            result['SMA_99'] = talib.SMA(df['Close'].values, timeperiod=99)
            
            # Exponential Moving Averages
            result['EMA_9'] = talib.EMA(df['Close'].values, timeperiod=9)
            result['EMA_21'] = talib.EMA(df['Close'].values, timeperiod=21)
            
            # MACD (Moving Average Convergence Divergence)
            macd, macd_signal, macd_hist = talib.MACD(
                df['Close'].values, 
                fastperiod=12, 
                slowperiod=26, 
                signalperiod=9
            )
            result['MACD'] = macd
            result['MACD_Signal'] = macd_signal
            result['MACD_Hist'] = macd_hist
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                df['Close'].values,
                timeperiod=20,
                nbdevup=2,
                nbdevdn=2,
                matype=0
            )
            result['BB_Upper'] = bb_upper
            result['BB_Middle'] = bb_middle
            result['BB_Lower'] = bb_lower
            
            # Stochastic Oscillator
            slowk, slowd = talib.STOCH(
                df['High'].values,
                df['Low'].values,
                df['Close'].values,
                fastk_period=14,
                slowk_period=3,
                slowk_matype=0,
                slowd_period=3,
                slowd_matype=0
            )
            result['Stoch_K'] = slowk
            result['Stoch_D'] = slowd
            
            # ATR (Average True Range) - volatility indicator
            result['ATR'] = talib.ATR(
                df['High'].values,
                df['Low'].values,
                df['Close'].values,
                timeperiod=14
            )
            
            # CCI (Commodity Channel Index)
            result['CCI'] = talib.CCI(
                df['High'].values,
                df['Low'].values,
                df['Close'].values,
                timeperiod=14
            )
            
            # OBV (On Balance Volume)
            result['OBV'] = talib.OBV(df['Close'].values, df['Volume'].values)
            
            # Custom indicators - price changes
            result['Price_Change'] = df['Close'].pct_change()
            result['Price_Change_2d'] = df['Close'].pct_change(2)
            result['Price_Change_5d'] = df['Close'].pct_change(5)
            
            # Custom indicators - volume changes
            result['Volume_Change'] = df['Volume'].pct_change()
            result['Volume_Change_2d'] = df['Volume'].pct_change(2)
            result['Volume_Change_5d'] = df['Volume'].pct_change(5)
            
            # Custom indicators - MA crossover signals
            result['SMA_7_25_Ratio'] = result['SMA_7'] / result['SMA_25']
            result['EMA_9_21_Ratio'] = result['EMA_9'] / result['EMA_21']
            
            # Relative position within Bollinger Bands (0 = at lower band, 1 = at upper band)
            bb_range = result['BB_Upper'] - result['BB_Lower']
            result['BB_Position'] = (df['Close'] - result['BB_Lower']) / bb_range
            
            # Log returns (better statistical properties than percentage changes)
            result['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
            
            logger.info(f"Added {len(result.columns) - len(df.columns)} technical indicators")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            raise
    
    def add_orderbook_features(self, df: pd.DataFrame, orderbook_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add orderbook features to the main dataframe.
        
        Args:
            df: Main dataframe with OHLCV and technical indicators
            orderbook_df: Dataframe with orderbook data
            
        Returns:
            DataFrame with additional orderbook feature columns
        """
        result = df.copy()
        
        # Verify orderbook dataframe has required columns
        required_cols = ["Open Interest", "Funding Rate"]
        if not all(col in orderbook_df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in orderbook_df.columns]
            raise ValueError(f"Missing required orderbook columns: {missing}")
        
        try:
            # Merge based on timestamp index
            result = pd.merge(
                result, 
                orderbook_df[required_cols], 
                left_index=True, 
                right_index=True,
                how='left'
            )
            
            # Add derived features
            # Funding rate changes
            result['Funding_Rate_Change'] = result['Funding Rate'].pct_change()
            result['Funding_Rate_MA_24'] = result['Funding Rate'].rolling(window=24).mean()
            
            # Open Interest changes
            result['OI_Change'] = result['Open Interest'].pct_change()
            result['OI_Change_24h'] = result['Open Interest'].pct_change(24)
            
            # OI to volume ratio (liquidity indicator)
            result['OI_to_Volume_Ratio'] = result['Open Interest'] / result['Volume']
            
            # OI trend indicators (longer-term trends)
            result['OI_SMA_24'] = result['Open Interest'].rolling(window=24).mean()
            result['OI_SMA_72'] = result['Open Interest'].rolling(window=72).mean()
            result['OI_Trend'] = result['OI_SMA_24'] / result['OI_SMA_72']
            
            logger.info(f"Added {len(result.columns) - len(df.columns)} orderbook features")
            
            return result
            
        except Exception as e:
            logger.error(f"Error adding orderbook features: {e}")
            raise
    
    def add_sentiment_features(self, df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add sentiment features to the main dataframe.
        Daily sentiment data is forward-filled to match hourly data.
        
        Args:
            df: Main dataframe with OHLCV and other features
            sentiment_df: Dataframe with daily sentiment data
            
        Returns:
            DataFrame with additional sentiment feature columns
        """
        result = df.copy()
        
        # Verify sentiment dataframe has required columns
        required_cols = ["Fear and Greed Value", "Fear and Greed Classification"]
        if not all(col in sentiment_df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in sentiment_df.columns]
            raise ValueError(f"Missing required sentiment columns: {missing}")
        
        try:
            # Resample daily to hourly with forward fill
            hourly_sentiment = sentiment_df.resample('H').ffill()
            
            # Merge based on timestamp index
            result = pd.merge(
                result, 
                hourly_sentiment[required_cols], 
                left_index=True, 
                right_index=True,
                how='left'
            )
            
            # Forward fill any remaining NaN values 
            # (this happens when we have hourly data but not daily sentiment data)
            result["Fear and Greed Value"] = result["Fear and Greed Value"].ffill()
            result["Fear and Greed Classification"] = result["Fear and Greed Classification"].ffill()
            
            # Add derived features
            # Convert classification to numeric (one-hot encoding)
            classifications = result["Fear and Greed Classification"].unique()
            for cls in classifications:
                col_name = f"Sentiment_{cls.replace(' ', '_')}"
                result[col_name] = (result["Fear and Greed Classification"] == cls).astype(int)
            
            # Sentiment changes
            result["FG_Value_Change"] = result["Fear and Greed Value"].diff()
            result["FG_Value_Change_3d"] = result["Fear and Greed Value"].diff(3*24)  # 3 days (at hourly intervals)
            
            # Moving averages of sentiment
            result["FG_Value_MA_3d"] = result["Fear and Greed Value"].rolling(window=3*24).mean()
            result["FG_Value_MA_7d"] = result["Fear and Greed Value"].rolling(window=7*24).mean()
            
            logger.info(f"Added {len(result.columns) - len(df.columns)} sentiment features")
            
            return result
            
        except Exception as e:
            logger.error(f"Error adding sentiment features: {e}")
            raise
    
    def create_target_variables(self, df: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
        """
        Create target variables for price prediction.
        
        Args:
            df: Main dataframe with all features
            horizon: Prediction horizon in hours (default: 24 hours)
            
        Returns:
            DataFrame with added target columns
        """
        result = df.copy()
        
        try:
            # Future price
            result[f'Future_Close_{horizon}h'] = result['Close'].shift(-horizon)
            
            # Price direction (binary classification target)
            result[f'Direction_{horizon}h'] = (result[f'Future_Close_{horizon}h'] > result['Close']).astype(int)
            
            # Price change percentage
            result[f'PriceChange_Pct_{horizon}h'] = (
                (result[f'Future_Close_{horizon}h'] - result['Close']) / result['Close'] * 100
            )
            
            # Categorize price change
            def categorize_change(pct_change):
                if pct_change <= -5:
                    return 'Large_Down'
                elif -5 < pct_change <= -1:
                    return 'Medium_Down'
                elif -1 < pct_change < 1:
                    return 'Small'
                elif 1 <= pct_change < 5:
                    return 'Medium_Up'
                else:  # >= 5
                    return 'Large_Up'
            
            result[f'PriceChange_Cat_{horizon}h'] = result[f'PriceChange_Pct_{horizon}h'].apply(categorize_change)
            
            # One-hot encode the categories
            categories = ['Large_Down', 'Medium_Down', 'Small', 'Medium_Up', 'Large_Up']
            for cat in categories:
                col_name = f'Target_{cat}_{horizon}h'
                result[col_name] = (result[f'PriceChange_Cat_{horizon}h'] == cat).astype(int)
            
            # Store target column names
            self.target_columns = [
                f'Direction_{horizon}h',
                f'PriceChange_Pct_{horizon}h',
                f'PriceChange_Cat_{horizon}h'
            ] + [f'Target_{cat}_{horizon}h' for cat in categories]
            
            logger.info(f"Created {len(self.target_columns)} target variables for {horizon}h horizon")
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating target variables: {e}")
            raise
    
    def clean_and_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset and normalize features.
        
        Args:
            df: Main dataframe with all features and targets
            
        Returns:
            Cleaned and normalized DataFrame
        """
        result = df.copy()
        
        try:
            # Drop rows with NaN values in essential columns
            essential_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI']
            result = result.dropna(subset=essential_cols)
            
            # Store all feature columns (will be needed for the model)
            self.feature_columns = [col for col in result.columns 
                                   if col not in self.target_columns 
                                   and not col.startswith('Future_')]
            
            # Apply different scaling to different feature groups
            # Price-related features
            price_cols = ['Open', 'High', 'Low', 'Close', 'SMA_7', 'SMA_25', 'SMA_99', 
                          'EMA_9', 'EMA_21', 'BB_Upper', 'BB_Middle', 'BB_Lower']
            
            # Log-scaled features
            log_cols = ['Volume', 'Open Interest', 'OBV']
            
            # Percentage features (already normalized)
            pct_cols = [col for col in self.feature_columns 
                       if 'Change' in col or 'Ratio' in col or 'Position' in col]
            
            # Other features to standardize
            std_cols = [col for col in self.feature_columns 
                       if col not in price_cols + log_cols + pct_cols]
            
            # Scale price-related features with MinMaxScaler
            if price_cols:
                price_data = result[price_cols].values
                self.price_scaler.fit(price_data)
                result[price_cols] = self.price_scaler.transform(price_data)
            
            # Apply log transformation to high-range features
            for col in log_cols:
                if col in result.columns:
                    # Add small epsilon to avoid log(0)
                    result[col] = np.log1p(result[col])
            
            # Standardize remaining features 
            for col in std_cols:
                if col in result.columns:
                    scaler = StandardScaler()
                    result[col] = scaler.fit_transform(result[[col]])
                    self.feature_scalers[col] = scaler
            
            logger.info(f"Cleaned and normalized data. {len(result)} rows remaining")
            
            return result
            
        except Exception as e:
            logger.error(f"Error cleaning and normalizing data: {e}")
            raise
    
    def prepare_sequence_data(self, df: pd.DataFrame, seq_length: int = 24) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequence data for the transformer model.
        
        Args:
            df: Cleaned and normalized dataframe
            seq_length: Number of time steps in each sequence
            
        Returns:
            Tuple of (X, y) where X is sequence data and y is target data
        """
        try:
            # Get feature and target data
            feature_data = df[self.feature_columns].values
            
            # For classification targets
            direction_target = df[f'Direction_{seq_length}h'].values
            category_targets = df[[col for col in df.columns if col.startswith('Target_')]].values
            
            # For regression target (percentage change)
            regression_target = df[f'PriceChange_Pct_{seq_length}h'].values
            
            # Create sequences
            X = []
            y_direction = []
            y_category = []
            y_regression = []
            
            for i in range(len(df) - seq_length):
                X.append(feature_data[i:i + seq_length])
                y_direction.append(direction_target[i + seq_length])
                y_category.append(category_targets[i + seq_length])
                y_regression.append(regression_target[i + seq_length])
            
            # Convert to numpy arrays
            X = np.array(X)
            y_direction = np.array(y_direction)
            y_category = np.array(y_category)
            y_regression = np.array(y_regression)
            
            # Combine targets into a dictionary
            y = {
                'direction': y_direction,
                'category': y_category,
                'regression': y_regression
            }
            
            logger.info(f"Created sequence data with shape {X.shape}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing sequence data: {e}")
            raise
    
    def process_data(self, 
                    candles_df: pd.DataFrame, 
                    orderbook_df: pd.DataFrame,
                    sentiment_df: pd.DataFrame,
                    horizon: int = 24,
                    seq_length: int = 24,
                    return_sequences: bool = True) -> Union[pd.DataFrame, Tuple[np.ndarray, Dict]]:
        """
        Full data processing pipeline.
        
        Args:
            candles_df: DataFrame with OHLCV data
            orderbook_df: DataFrame with orderbook data
            sentiment_df: DataFrame with sentiment data
            horizon: Prediction horizon in hours
            seq_length: Sequence length for transformer input
            return_sequences: If True, return X and y for sequences, else return processed DataFrame
            
        Returns:
            Either processed DataFrame or (X, y) tuple for model training
        """
        try:
            # 1. Calculate technical indicators
            df = self.calculate_technical_indicators(candles_df)
            
            # 2. Add orderbook features
            df = self.add_orderbook_features(df, orderbook_df)
            
            # 3. Add sentiment features
            df = self.add_sentiment_features(df, sentiment_df)
            
            # 4. Create target variables
            df = self.create_target_variables(df, horizon=horizon)
            
            # 5. Clean and normalize
            df = self.clean_and_normalize(df)
            
            # Store the processed data
            self.processed_data = df
            
            if return_sequences:
                # 6. Prepare sequence data for transformer
                X, y = self.prepare_sequence_data(df, seq_length=seq_length)
                return X, y
            else:
                return df
                
        except Exception as e:
            logger.error(f"Error in data processing pipeline: {e}")
            raise


# Example usage
if __name__ == "__main__":
    from crypto_data_interpreter import CryptoDataInterpreter
    
    try:
        # Load the data
        interpreter = CryptoDataInterpreter()
        candles, orderbook, sentiment = interpreter.load_all_data()
        
        # Process the data
        preprocessor = CryptoDataPreprocessor()
        X, y = preprocessor.process_data(
            candles_df=candles,
            orderbook_df=orderbook,
            sentiment_df=sentiment,
            horizon=24,  # 24-hour prediction
            seq_length=24  # Use 24 hours of data
        )
        
        logger.info(f"Input shape: {X.shape}")
        logger.info(f"Direction target shape: {y['direction
