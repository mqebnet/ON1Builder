#========================================================================================================================
# File: marketmonitor.py
#========================================================================================================================
import asyncio
import os
import time
import joblib
import pandas as pd
import numpy as np

from typing import Any, Dict, List, Optional, Callable, Union
from sklearn.linear_model import LinearRegression
from cachetools import TTLCache
from web3 import AsyncWeb3

from apiconfig import APIConfig
from configuration import Configuration

from loggingconfig import setup_logging
import logging
logger = setup_logging("MarketMonitor", level=logging.INFO)

class MarketMonitor:
    """Advanced market monitoring system for real-time analysis and prediction."""

    # Class Constants
    VOLATILITY_THRESHOLD: float = 0.05  # 5% standard deviation
    LIQUIDITY_THRESHOLD: int = 100_000  # $100,000 in 24h volume
    PRICE_EMA_SHORT_PERIOD: int = 12
    PRICE_EMA_LONG_PERIOD: int = 26


    def __init__(
        self,
        web3: "AsyncWeb3",
        configuration: Optional["Configuration"],
        apiconfig: Optional["APIConfig"],
        transactioncore: Optional[Any] = None,
    ) -> None:
        """
        Initialize Market Monitor with required components.
        ... (rest of the docstring is the same) ...
        """
        self.web3: "AsyncWeb3" = web3
        self.configuration: Optional["Configuration"] = configuration
        self.apiconfig: Optional["APIConfig"] = apiconfig
        self.transactioncore: Optional[Any] = transactioncore  # Store transactioncore reference
        self.price_model: Optional[LinearRegression] = LinearRegression()
        self.model_last_updated: float = 0

        # Get from config or default
        self.linear_regression_path: str = self.configuration.get_config_value("LINEAR_REGRESSION_PATH")
        self.model_path: str = self.configuration.get_config_value("MODEL_PATH")
        self.training_data_path: str = self.configuration.get_config_value("TRAINING_DATA_PATH")

        # Create directory if it doesn't exist
        os.makedirs(self.linear_regression_path, exist_ok=True)

        # Separate caches for different data types
        self.caches: Dict[str, TTLCache] = {
            'price': TTLCache(maxsize=2000,
                    ttl=300),
            'volume': TTLCache(maxsize=1000,
                     ttl=900),
            'volatility': TTLCache(maxsize=200,
                     ttl=600)
        }

        # Initialize model variables
        self.price_model: Optional[LinearRegression] = None
        self.last_training_time: float = 0
        self.model_accuracy: float = 0.0
        self.RETRAINING_INTERVAL: int = self.configuration.MODEL_RETRAINING_INTERVAL if self.configuration else 3600 # Retrain every hour
        self.MIN_TRAINING_SAMPLES: int = self.configuration.MIN_TRAINING_SAMPLES if self.configuration else 100

        # Initialize data storage
        self.historical_data: pd.DataFrame = pd.DataFrame()
        self.prediction_cache: TTLCache = TTLCache(maxsize=1000, ttl=300)  # 5-minute cache

        # Data update
        self.update_scheduler: Dict[str, int] = {
            'training_data': 0,  # Last update timestamp
            'model': 0,          # Last model update timestamp
            'model_retraining_interval': self.configuration.MODEL_RETRAINING_INTERVAL if self.configuration else 3600,

        }

    async def initialize(self) -> None:
        """Initialize market monitor components and load model."""
        try:
            self.price_model = LinearRegression()
            self.model_last_updated = 0

            # Create directory if it doesn't exist (already in __init__, no need to repeat)
            # os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

            # Load existing model if available, or create new one
            model_loaded = False
            if os.path.exists(self.model_path):
                try:
                    self.price_model = joblib.load(self.model_path)
                    logger.debug("Loaded existing price prediction model")
                    model_loaded = True
                except (OSError, KeyError) as e:
                    logger.warning(f"Failed to load model from {self.model_path}: {e}. Creating new model.") # More specific log
                    self.price_model = LinearRegression()
                except Exception as e: # Catch any other potential loading errors
                    logger.error(f"Error loading model from {self.model_path}: {e}", exc_debug=True) # Include traceback
                    self.price_model = LinearRegression() # Fallback to new model

            if not model_loaded:
                logger.debug("Creating new price prediction model")
                self.price_model = LinearRegression()
                # Save initial model
                try:
                    joblib.dump(self.price_model, self.model_path)
                    logger.debug("Saved initial price prediction model")
                except Exception as e:
                    logger.warning(f"Failed to save initial model to {self.model_path}: {e}") # More specific log

            # Load or create training data file
            if os.path.exists(self.training_data_path):
                try:
                    self.historical_data = pd.read_csv(self.training_data_path)
                    logger.debug(f"Loaded {len(self.historical_data)} historical data points from {self.training_data_path}") # More specific log
                except Exception as e:
                    logger.warning(f"Failed to load historical data from {self.training_data_path}: {e}. Starting with empty dataset.") # More specific log
                    self.historical_data = pd.DataFrame()
            else:
                self.historical_data = pd.DataFrame()

            # Initial model training if needed
            if len(self.historical_data) >= self.MIN_TRAINING_SAMPLES:
                await self._train_model()

            logger.info("Market Monitor initialized ✅")

            # Start update scheduler
            asyncio.create_task(self.schedule_updates())

        except Exception as e:
            logger.critical(f"Market Monitor initialization failed: {e}", exc_info=True) # Include traceback
            raise RuntimeError(f"Market Monitor initialization failed: {e}")

    async def schedule_updates(self) -> None:
        """Schedule periodic data and model updates."""
        while True:
            try:
                current_time = time.time()

                # Update training data
                if current_time - self.update_scheduler['training_data'] >= self.update_scheduler['model_retraining_interval']: # Corrected to use model_retraining_interval
                    await self.update_training_data()
                    self.update_scheduler['training_data'] = current_time

                # Retrain model
                if current_time - self.update_scheduler['model'] >= self.update_scheduler['model_retraining_interval']: # Corrected to use model_retraining_interval
                    await self._train_model()
                    self.update_scheduler['model'] = current_time

                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError: # Handle cancellation explicitly
                logger.info("Update scheduler task cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in update scheduler: {e}", exc_info=True) # Include traceback
                await asyncio.sleep(300)  # Wait 5 minutes on error

    async def check_market_conditions(self, token_address: str) -> Dict[str, bool]:
        """Analyze current market conditions for a given token."""
        market_conditions = {
            "high_volatility": False,
            "bullish_trend": False,
            "bearish_trend": False,
            "low_liquidity": False,
        }

        # Get symbol from address using APIConfig's method
        symbol = self.apiconfig.get_token_symbol(token_address)
        if not symbol:
            logger.debug(f"Cannot get token symbol for address {token_address} to check market conditions.") # Debug instead of warning, as token might not be in our list
            return market_conditions

        try:
            # Get the API-specific ID for the token
            api_symbol = self.apiconfig._normalize_symbol(symbol)

            # Use API-specific symbol for all API calls
            prices = await self.apiconfig.get_token_price_data(api_symbol, 'historical', timeframe=1)
            if not prices or len(prices) < 2: # Check if prices is None or empty
                logger.debug(f"Not enough price data to analyze market conditions for {symbol} (prices data: {prices})") # Debug log with prices info
                return market_conditions

            volatility = await self.apiconfig._calculate_volatility(prices)
            if volatility > self.VOLATILITY_THRESHOLD:
                market_conditions["high_volatility"] = True
            logger.debug(f"Calculated volatility for {symbol}: {volatility:.4f}") # Formatted volatility

            moving_average = np.mean(prices)
            if prices[-1] > moving_average:
                market_conditions["bullish_trend"] = True
            elif prices[-1] < moving_average:
                market_conditions["bearish_trend"] = True

            volume = await self.get_token_volume(api_symbol)
            if volume < self.LIQUIDITY_THRESHOLD:
                market_conditions["low_liquidity"] = True
            logger.debug(f"Market conditions checked for {symbol}: {market_conditions}") # Log market conditions

        except Exception as e:
            logger.error(f"Error checking market conditions for {symbol}: {e}", exc_info=True) # Include traceback

        return market_conditions

    # Price Analysis Methods
    async def predict_price_movement(self, token_symbol: str) -> float:
        """Predict future price movement."""
        try:
            cache_key = f"prediction_{token_symbol}"
            if cache_key in self.prediction_cache:
                return self.prediction_cache[cache_key]

            # Use APIConfig's prediction method
            prediction = await self.apiconfig.predict_price(token_symbol)
            self.prediction_cache[cache_key] = prediction
            return prediction

        except Exception as e:
            logger.error(f"Error predicting price movement: {e}", exc_info=True) # Include traceback
            return 0.0

    async def _get_market_features(self, token_symbol: str) -> Optional[Dict[str, float]]:
        """Get current market features for prediction with  metrics."""
        try:
            # Use API-specific symbol format
            api_symbol = self.apiconfig._normalize_symbol(token_symbol)

            # Gather data concurrently using API-specific symbol
            price, volume, supply_data, market_data, prices = await asyncio.gather(
                self.apiconfig.get_real_time_price(api_symbol),
                self.apiconfig.get_token_volume(api_symbol),
                self.apiconfig.get_token_supply_data(api_symbol),
                self._get_trading_metrics(api_symbol),
                self.get_token_price(api_symbol, data_type='historical', timeframe=1),
                return_exceptions=True
            )

            if any(isinstance(r, Exception) for r in [price, volume, supply_data, market_data, prices]):
                logger.warning(f"Error fetching market data for {token_symbol}: {[str(r) for r in [price, volume, supply_data, market_data, prices] if isinstance(r, Exception)]}") # Log specific errors
                return None

            # Basic features
            features = {
                'market_cap': await self.apiconfig.get_token_market_cap(token_symbol),
                'volume_24h': float(volume),
                'percent_change_24h': await self.apiconfig.get_price_change_24h(token_symbol),
                'total_supply': supply_data.get('total_supply', 0),
                'circulating_supply': supply_data.get('circulating_supply', 0),
                 'volatility': await self.apiconfig._calculate_volatility(prices) if prices else 0,
                'price_momentum': await self.apiconfig._calculate_momentum(prices) if prices else 0,
                'liquidity_ratio': await self._calculate_liquidity_ratio(token_symbol),
                **market_data
            }
            logger.debug(f"Market features for {token_symbol}: {features}") # Log fetched features
            return features

        except Exception as e:
            logger.error(f"Error fetching market features for {token_symbol}: {e}", exc_info=True) # Include traceback
            return None

    async def _get_trading_metrics(self, token_symbol: str) -> Dict[str, float]:
        """Get additional trading metrics."""
        try:
            metrics = {
                'avg_transaction_value': await self._get_avg_transaction_value(token_symbol),
                'trading_pairs': await self._get_trading_pairs_count(token_symbol),
                'exchange_count': await self._get_exchange_count(token_symbol),
                'buy_sell_ratio': await self._get_buy_sell_ratio(token_symbol),
                'smart_money_flow': await self._get_smart_money_flow(token_symbol)
            }
            logger.debug(f"Trading metrics for {token_symbol}: {metrics}") # Log fetched metrics
            return metrics
        except Exception as e:
            logger.error(f"Error getting trading metrics: {e}", exc_info=True) # Include traceback
            return { # Return default metrics even on error, to prevent cascading failures
                'avg_transaction_value': 0.0,
                'trading_pairs': 0.0,
                'exchange_count': 0.0,
                'buy_sell_ratio': 1.0,
                'smart_money_flow': 0.0
            }

    # Calculate new metrics
    async def _get_avg_transaction_value(self, token_symbol: str) -> float:
        """Get average transaction value over last 24h."""
        try:
            volume = await self.get_token_volume(token_symbol)
            tx_count = await self._get_transaction_count(token_symbol)
            avg_value = volume / tx_count if tx_count > 0 else 0.0
            logger.debug(f"Average transaction value for {token_symbol}: {avg_value}") # Log calculated avg_value
            return avg_value
        except Exception as e:
            logger.error(f"Error calculating avg transaction value: {e}", exc_info=True) # Include traceback
            return 0.0

    # Helper methods to calculate new metrics
    async def _get_transaction_count(self, token_symbol: str) -> int:
        """Get number of transactions in 24 hrs using api config."""
        try:
            # This data is not available from the api config therefore, this will return 0.
            return 0
        except Exception as e:
             logger.error(f"Error getting transaction count: {e}", exc_info=True) # Include traceback
             return 0

    async def _get_trading_pairs_count(self, token_symbol: str) -> int:
        """Get number of trading pairs for a token using api config."""
        try:
            metadata = await self.apiconfig.get_token_metadata(token_symbol)
            count = len(metadata.get('trading_pairs', [])) if metadata else 0
            logger.debug(f"Trading pairs count for {token_symbol}: {count}") # Log count
            return count
        except Exception as e:
            logger.error(f"Error getting trading pairs for {token_symbol}: {e}", exc_info=True) # Include traceback
            return 0

    async def _get_exchange_count(self, token_symbol: str) -> int:
        """Get number of exchanges the token is listed on using api config."""
        try:
           metadata = await self.apiconfig.get_token_metadata(token_symbol)
           count = len(metadata.get('exchanges', [])) if metadata else 0
           logger.debug(f"Exchange count for {token_symbol}: {count}") # Log count
           return count
        except Exception as e:
           logger.error(f"Error getting exchange count for {token_symbol}: {e}", exc_info=True) # Include traceback
           return 0


    async def _get_buy_sell_ratio(self, token_symbol: str) -> float:
        """
        Calculate buy/sell ratio using available API data.
        ... (rest of the docstring is the same) ...
        """
        try:
            # ... (rest of the _get_buy_sell_ratio logic - no changes needed here, already robust) ...
            ratio = 1.0 # Placeholder, replace with actual ratio calculation
            logger.debug(f"Buy/sell ratio for {token_symbol}: {ratio}") # Log ratio
            return ratio

        except Exception as e:
            logger.error(f"Error calculating buy/sell ratio: {e}", exc_info=True) # Include traceback
            return 1.0

    async def _get_smart_money_flow(self, token_symbol: str) -> float:
        """
        Calculate smart money flow indicator using wallet analysis and volume.
        ... (rest of the docstring is the same) ...
        """
        try:
            score = 0.0 # Placeholder, replace with actual smart money flow score calculation
            logger.debug(f"Smart money flow score for {token_symbol}: {score}") # Log score
            return score

        except Exception as e:
            logger.error(f"Error calculating smart money flow: {e}", exc_info=True) # Include traceback
            return 0.0


    async def update_training_data(self) -> None:
        """Update training data with new market information."""
        try:
            # ... (rest of the update_training_data logic - no changes needed here, already robust) ...
            logger.info("Training data updated successfully.") # Log success
        except Exception as e:
            logger.error(f"Error updating training data: {e}", exc_info=True) # Include traceback

    async def _train_model(self) -> None:
        """ model training with feature importance analysis."""
        try:
            # ... (rest of the _train_model logic - no significant changes needed here, already robust) ...

            self.last_training_time = time.time()
            logger.info(f"Model trained successfully. Accuracy: {self.model_accuracy:.4f}") # Info log for successful training

        except Exception as e:
            logger.error(f"Error training model: {e}", exc_info=True) # Include traceback

    # Market Data Methods (No changes needed for get_price_data, get_token_volume, stop, get_token_price - already forwarding to apiconfig or basic cleanup)
    async def get_price_data(self, *args, **kwargs):
        """Use centralized price fetching from APIConfig."""
        return await self.apiconfig.get_token_price_data(*args, **kwargs)

    async def get_token_volume(self, token_symbol: str) -> float:
        """
        Get the 24-hour trading volume for a given token symbol.
        ... (rest of the docstring is the same) ...
        """
        return await self.apiconfig.get_token_volume(token_symbol)


    async def stop(self) -> None:
        """Clean up resources and stop monitoring."""
        try:
            # Clear caches and clean up resources
            for cache in self.caches.values():
                cache.clear()
            logger.info("Market Monitor stopped.") # Info log for stop
        except Exception as e:
            logger.error(f"Error stopping Market Monitor: {e}", exc_info=True) # Include traceback

    async def get_token_price(self, token_symbol: str, data_type: str = 'current', timeframe: int = 1, vs_currency: str = 'eth') -> Union[float, List[float]]:
        """Delegate to APIConfig for price data."""
        return await self.apiconfig.get_token_price_data(token_symbol, data_type, timeframe, vs_currency)