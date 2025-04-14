import asyncio
import os
import time
import joblib
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union
from sklearn.linear_model import LinearRegression
from cachetools import TTLCache
from web3 import AsyncWeb3

from apiconfig import APIConfig
from configuration import Configuration
from loggingconfig import setup_logging
import logging

logger = setup_logging("MarketMonitor", level=logging.INFO)

class MarketMonitor:
    """
    Monitors market data in real-time and predicts price movement using a linear regression model.
    It periodically updates training data and retrains the model as needed.
    """
    VOLATILITY_THRESHOLD: float = 0.05  # 5% volatility threshold
    LIQUIDITY_THRESHOLD: float = 100_000  # Minimum volume threshold
    PRICE_EMA_SHORT_PERIOD: int = 12
    PRICE_EMA_LONG_PERIOD: int = 26

    def __init__(
        self,
        web3: AsyncWeb3,
        configuration: Configuration,
        apiconfig: APIConfig,
        transactioncore: Optional[Any] = None,
    ) -> None:
        self.web3 = web3
        self.configuration = configuration
        self.apiconfig = apiconfig
        self.transactioncore = transactioncore

        self.price_model: Optional[LinearRegression] = None
        self.last_training_time: float = 0.0
        self.model_accuracy: float = 0.0
        self.RETRAINING_INTERVAL: int = self.configuration.MODEL_RETRAINING_INTERVAL
        self.MIN_TRAINING_SAMPLES: int = self.configuration.MIN_TRAINING_SAMPLES

        # Cache for recent market data
        self.price_cache: TTLCache = TTLCache(maxsize=2000, ttl=300)
        self.update_scheduler: Dict[str, int] = {
            "training_data": 0,
            "model": 0,
            "model_retraining_interval": self.configuration.MODEL_RETRAINING_INTERVAL,
        }

        # Paths for the ML model and training data
        self.linear_regression_path: str = self.configuration.LINEAR_REGRESSION_PATH
        self.model_path: str = self.configuration.MODEL_PATH
        self.training_data_path: str = self.configuration.TRAINING_DATA_PATH

        # Ensure the training directory exists (using a blocking call offloaded to a thread)
        os.makedirs(self.linear_regression_path, exist_ok=True)

    async def initialize(self) -> None:
        """
        Initialize MarketMonitor by loading or training a price model and historical data.
        Schedules updates for training data and model retraining.
        """
        try:
            if os.path.exists(self.model_path):
                try:
                    # Loading a model is blocking so run in a thread.
                    self.price_model = await asyncio.to_thread(joblib.load, self.model_path)
                    logger.debug("Loaded existing price model.")
                except Exception as e:
                    logger.warning(f"Loading model failed: {e}; creating a new model.")
                    self.price_model = LinearRegression()
                    await asyncio.to_thread(joblib.dump, self.price_model, self.model_path)
            else:
                self.price_model = LinearRegression()
                await asyncio.to_thread(joblib.dump, self.price_model, self.model_path)

            if os.path.exists(self.training_data_path):
                try:
                    self.historical_data = await asyncio.to_thread(pd.read_csv, self.training_data_path)
                    logger.debug(f"Loaded {len(self.historical_data)} training data points.")
                except Exception as e:
                    logger.warning(f"Failed to load training data: {e}")
                    self.historical_data = pd.DataFrame()
            else:
                self.historical_data = pd.DataFrame()

            # If sufficient training data exists, train the model.
            if len(self.historical_data) >= self.MIN_TRAINING_SAMPLES:
                try:
                    await self.train_price_model()
                    logger.debug("Trained price model with available historical data.")
                except Exception as e:
                    logger.warning(f"Model training failed: {e}")
                    self.price_model = LinearRegression()
                    await asyncio.to_thread(joblib.dump, self.price_model, self.model_path)
                    self.historical_data = pd.DataFrame()
            else:
                logger.debug("Insufficient historical data for training; model remains unchanged.")

            logger.info("MarketMonitor initialized successfully.")
            asyncio.create_task(self.schedule_updates())
        except Exception as e:
            logger.critical(f"MarketMonitor initialization failed: {e}", exc_info=True)
            raise

    async def schedule_updates(self) -> None:
        """
        Periodically update training data and retrain the price model.
        """
        while True:
            try:
                current_time = time.time()
                if current_time - self.update_scheduler["training_data"] >= self.update_scheduler["model_retraining_interval"]:
                    await self.apiconfig.update_training_data()
                    self.update_scheduler["training_data"] = current_time
                if current_time - self.update_scheduler["model"] >= self.update_scheduler["model_retraining_interval"]:
                    await self.train_price_model()
                    self.update_scheduler["model"] = current_time
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                logger.info("MarketMonitor update scheduler cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in update scheduler: {e}", exc_info=True)
                await asyncio.sleep(300)

    async def check_market_conditions(self, token_address: str) -> Dict[str, bool]:
        """
        Evaluate market conditions for the given token based on historical price data and volume.
        """
        market_conditions = {
            "high_volatility": False,
            "bullish_trend": False,
            "bearish_trend": False,
            "low_liquidity": False,
        }
        # Get token symbol from APIConfig.
        symbol = self.apiconfig.get_token_symbol(token_address)
        if not symbol:
            logger.debug(f"Unable to determine token symbol for {token_address} in market conditions check.")
            return market_conditions
        try:
            api_symbol = symbol.upper()
            prices = await self.apiconfig.get_token_price_data(api_symbol, "historical", timeframe=1)
            if not prices or len(prices) < 2:
                logger.debug(f"Not enough price data for {symbol}.")
                return market_conditions

            volatility = np.std(prices) / np.mean(prices)
            if volatility > self.VOLATILITY_THRESHOLD:
                market_conditions["high_volatility"] = True

            avg_price = np.mean(prices)
            if prices[-1] > avg_price:
                market_conditions["bullish_trend"] = True
            elif prices[-1] < avg_price:
                market_conditions["bearish_trend"] = True

            volume = await self.apiconfig.get_token_volume(api_symbol)
            if volume < self.LIQUIDITY_THRESHOLD:
                market_conditions["low_liquidity"] = True

            logger.debug(f"Market conditions for {symbol}: {market_conditions}")
        except Exception as e:
            logger.error(f"Error checking market conditions for {symbol}: {e}", exc_info=True)
        return market_conditions

    async def predict_price_movement(self, token_symbol: str) -> float:
        """
        Predict future price movement for the given token using the trained model.
        Returns the predicted price.
        """
        try:
            cache_key = f"prediction_{token_symbol}"
            # Use a local cache if available; else, predict.
            if cache_key in self.apiconfig.prediction_cache:
                return self.apiconfig.prediction_cache[cache_key]
            prediction = await self.apiconfig.predict_price(token_symbol)
            self.apiconfig.prediction_cache[cache_key] = prediction
            return prediction
        except Exception as e:
            logger.error(f"Error predicting price for {token_symbol}: {e}")
            return 0.0

    async def get_token_price_data(
        self,
        token_symbol: str,
        data_type: str = "current",
        timeframe: int = 1,
        vs_currency: str = "eth"
    ) -> Union[float, List[float]]:
        """
        Retrieve token price data either as a current price or historical data.
        """
        return await self.apiconfig.get_token_price_data(token_symbol, data_type, timeframe, vs_currency)

    async def train_price_model(self) -> None:
        """
        Train a linear regression model for price prediction using available training data.
        """
        try:
            training_data_path = self.configuration.TRAINING_DATA_PATH
            model_path = self.configuration.MODEL_PATH
            if not os.path.exists(training_data_path):
                logger.warning("Training data file not found; skipping training.")
                return
            df = await asyncio.to_thread(pd.read_csv, training_data_path)
            if len(df) < self.MIN_TRAINING_SAMPLES:
                logger.warning(f"Insufficient training samples: {len(df)} (required: {self.MIN_TRAINING_SAMPLES}).")
                return
            features = ['price_usd', 'volume_24h', 'market_cap', 'volatility', 'liquidity_ratio', 'price_momentum']
            X = df[features].fillna(0)
            y = df['price_usd'].fillna(0)
            model = LinearRegression()
            model.fit(X, y)
            await asyncio.to_thread(joblib.dump, model, model_path)
            self.price_model = model
            logger.info(f"Price model trained and saved to {model_path}.")
        except Exception as e:
            logger.error(f"Error training price model: {e}", exc_info=True)

    async def stop(self) -> None:
        """
        Stop MarketMonitor operations by clearing caches.
        """
        try:
            for cache in [self.price_cache]:
                cache.clear()
            logger.info("MarketMonitor stopped.")
        except Exception as e:
            logger.error(f"Error stopping MarketMonitor: {e}", exc_info=True)
