import pandas as pd
import numpy as np
from pathlib import Path
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CryptoDataInterpreter:
    """
    Data interpreter for cryptocurrency data from multiple sources.
    Handles loading and basic formatting of data files.
    """
    
    def __init__(self, data_dir=None):
        """
        Initialize the data interpreter.
        
        Args:
            data_dir (str, optional): Directory containing data files.
        """
        self.data_dir = Path(data_dir) if data_dir else Path("./data")
        self.candles = None
        self.orderbook = None
        self.sentiment = None
    
    def load_candle_data(self, filename=None, from_csv=True):
        """
        Load hourly candle data with OHLCV.
        
        Args:
            filename (str, optional): Name of the candle data file
            from_csv (bool): If True, load from CSV, else from API (not implemented)
        
        Returns:
            pd.DataFrame: Loaded candle data
        """
        if from_csv:
            filepath = self.data_dir / (filename or "candles.csv")
            logger.info(f"Loading candle data from {filepath}")
            
            try:
                df = pd.read_csv(filepath)
                
                # Check required columns
                required_cols = ["Timestamp", "Open", "Close", "High", "Low", "Volume"]
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    raise ValueError(f"Candle data missing columns: {missing_cols}")
                
                # Convert timestamp to datetime
                df["Timestamp"] = pd.to_datetime(df["Timestamp"])
                df.set_index("Timestamp", inplace=True)
                
                # Ensure all price and volume data is numeric
                for col in ["Open", "Close", "High", "Low", "Volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                
                self.candles = df
                logger.info(f"Loaded {len(df)} candle records")
                return df
                
            except Exception as e:
                logger.error(f"Failed to load candle data: {e}")
                raise
        else:
            # TODO: Implement API loading
            logger.error("API loading not yet implemented")
            raise NotImplementedError("API loading not yet implemented")
    
    def load_orderbook_data(self, filename=None, from_csv=True):
        """
        Load hourly orderbook data with Open Interest and Funding Rate.
        
        Args:
            filename (str, optional): Name of the orderbook data file
            from_csv (bool): If True, load from CSV, else from API (not implemented)
        
        Returns:
            pd.DataFrame: Loaded orderbook data
        """
        if from_csv:
            filepath = self.data_dir / (filename or "orderbook.csv")
            logger.info(f"Loading orderbook data from {filepath}")
            
            try:
                df = pd.read_csv(filepath)
                
                # Check required columns
                required_cols = ["Timestamp", "Open Interest", "Funding Rate"]
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    raise ValueError(f"Orderbook data missing columns: {missing_cols}")
                
                # Convert timestamp to datetime
                df["Timestamp"] = pd.to_datetime(df["Timestamp"])
                df.set_index("Timestamp", inplace=True)
                
                # Ensure all data is numeric
                for col in ["Open Interest", "Funding Rate"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                
                self.orderbook = df
                logger.info(f"Loaded {len(df)} orderbook records")
                return df
                
            except Exception as e:
                logger.error(f"Failed to load orderbook data: {e}")
                raise
        else:
            # TODO: Implement API loading
            logger.error("API loading not yet implemented")
            raise NotImplementedError("API loading not yet implemented")
    
    def load_sentiment_data(self, filename=None, from_csv=True):
        """
        Load daily sentiment data with Fear and Greed index.
        
        Args:
            filename (str, optional): Name of the sentiment data file
            from_csv (bool): If True, load from CSV, else from API (not implemented)
        
        Returns:
            pd.DataFrame: Loaded sentiment data
        """
        if from_csv:
            filepath = self.data_dir / (filename or "sentiment.csv")
            logger.info(f"Loading sentiment data from {filepath}")
            
            try:
                df = pd.read_csv(filepath)
                
                # Check required columns
                required_cols = ["Timestamp", "Fear and Greed Value", "Fear and Greed Classification"]
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    raise ValueError(f"Sentiment data missing columns: {missing_cols}")
                
                # Convert timestamp to datetime
                df["Timestamp"] = pd.to_datetime(df["Timestamp"])
                df.set_index("Timestamp", inplace=True)
                
                # Ensure value is numeric
                df["Fear and Greed Value"] = pd.to_numeric(df["Fear and Greed Value"], errors="coerce")
                
                self.sentiment = df
                logger.info(f"Loaded {len(df)} sentiment records")
                return df
                
            except Exception as e:
                logger.error(f"Failed to load sentiment data: {e}")
                raise
        else:
            # TODO: Implement API loading
            logger.error("API loading not yet implemented")
            raise NotImplementedError("API loading not yet implemented")
    
    def load_all_data(self, candle_file=None, orderbook_file=None, sentiment_file=None):
        """
        Load all data sources at once.
        
        Args:
            candle_file (str, optional): Name of the candle data file
            orderbook_file (str, optional): Name of the orderbook data file
            sentiment_file (str, optional): Name of the sentiment data file
        
        Returns:
            tuple: (candles_df, orderbook_df, sentiment_df)
        """
        candles = self.load_candle_data(filename=candle_file)
        orderbook = self.load_orderbook_data(filename=orderbook_file)
        sentiment = self.load_sentiment_data(filename=sentiment_file)
        
        return candles, orderbook, sentiment
    
    def check_data_alignment(self):
        """
        Check if the loaded datasets have aligned timestamps.
        
        Returns:
            bool: True if data is properly aligned
        """
        if self.candles is None or self.orderbook is None:
            logger.error("Candle and orderbook data must be loaded before checking alignment")
            return False
        
        # Check hourly data alignment
        hourly_indices = sorted(set(self.candles.index).intersection(set(self.orderbook.index)))
        
        # Check sentiment data if loaded (it's daily, so we'll be more flexible)
        if self.sentiment is not None:
            sentiment_dates = set(self.sentiment.index.date)
            hourly_dates = set(pd.DatetimeIndex(hourly_indices).date)
            
            missing_dates = hourly_dates.difference(sentiment_dates)
            if missing_dates:
                logger.warning(f"Missing sentiment data for {len(missing_dates)} days")
        
        missing_candles = set(self.orderbook.index).difference(set(self.candles.index))
        missing_orderbook = set(self.candles.index).difference(set(self.orderbook.index))
        
        if missing_candles or missing_orderbook:
            logger.warning(f"Data misalignment: {len(missing_candles)} missing candles, {len(missing_orderbook)} missing orderbook entries")
            return False
        
        logger.info(f"Data alignment check passed with {len(hourly_indices)} aligned hourly records")
        return True
    
    def get_date_range(self):
        """
        Get the date range covered by the loaded data.
        
        Returns:
            tuple: (start_date, end_date) as datetime objects
        """
        dates = []
        
        if self.candles is not None:
            dates.extend([self.candles.index.min(), self.candles.index.max()])
        
        if self.orderbook is not None:
            dates.extend([self.orderbook.index.min(), self.orderbook.index.max()])
            
        if self.sentiment is not None:
            dates.extend([self.sentiment.index.min(), self.sentiment.index.max()])
        
        if not dates:
            return None, None
            
        return min(dates), max(dates)


# Example usage
if __name__ == "__main__":
    interpreter = CryptoDataInterpreter()
    
    try:
        candles, orderbook, sentiment = interpreter.load_all_data()
        
        if interpreter.check_data_alignment():
            start_date, end_date = interpreter.get_date_range()
            logger.info(f"Data covers period from {start_date} to {end_date}")
            
            # Print sample data
            logger.info("\nCandle data sample:")
            logger.info(candles.head())
            
            logger.info("\nOrderbook data sample:")
            logger.info(orderbook.head())
            
            logger.info("\nSentiment data sample:")
            logger.info(sentiment.head())
    except Exception as e:
        logger.error(f"Error in data interpretation: {e}")
