# -*- coding: utf-8 -*-
"""
MarketMonitor Module

Provides advanced market monitoring for real-time analysis, prediction, and risk assessment.
It gathers market data (prices, volumes, volatility, etc.), supports model training and prediction,
and exposes various metrics for strategy evaluation.
"""

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
    Advanced market monitoring system for real-time analysis and prediction.

    Attributes:
        VOLATILITY_THRESHOLD (float): Threshold to flag high volatility.
        LIQUIDITY_THRESHOLD (int): Minimum volume threshold to avoid low-liquidity warnings.
        PRICE_EMA_SHORT_PERIOD (int): Short-term EMA period.
        PRICE_EMA_LONG_PERIOD (int): Long-term EMA period.
    """

    VOLATILITY_THRESHOLD: float = 0.05
    LIQUIDITY_THRESHOLD: int = 100_000
    PRICE_EMA_SHORT_PERIOD: int = 12
    PRICE_EMA_LONG_PERIOD: int = 26

    def __init__(
        self,
        web3: AsyncWeb3,
        configuration: Optional[Configuration],
        apiconfig: Optional[APIConfig],
        transactioncore: Optional[Any] = None,
    ) -> None:
        """
        Initialize the MarketMonitor.

        Args:
            web3 (AsyncWeb3): An asynchronous Web3 instance.
            configuration (Optional[Configuration]): The configuration instance.
            apiconfig (Optional[APIConfig]): The API configuration instance.
            transactioncore (Optional[Any]): The TransactionCore instance (if needed).
        """
        self.web3 = web3
        self.configuration = configuration
        self.apiconfig = apiconfig
        self.transactioncore = transactioncore

        # Model and training related attributes (model training handled by APIConfig)
        self.price_model: Optional[LinearRegression] = None
        self.last_training_time = 0
        self.model_accuracy = 0.0
        self.RETRAINING_INTERVAL = self.configuration.MODEL_RETRAINING_INTERVAL
        self.MIN_TRAINING_SAMPLES = self.configuration.MIN_TRAINING_SAMPLES
        self.historical_data = pd.DataFrame()
        self.prediction_cache = TTLCache(maxsize=1000, ttl=300)
        self.update_scheduler = {
            'training_data': 0,
            'model': 0,
            'model_retraining_interval': self.configuration.MODEL_RETRAINING_INTERVAL
        }
        # Create caches for frequently used data
        self.caches = {
            'price': TTLCache(maxsize=2000, ttl=300),
            'volume': TTLCache(maxsize=1000, ttl=900),
            'volatility': TTLCache(maxsize=200, ttl=600)
        }
        # Ensure the directory for model training exists
        self.linear_regression_path = self.configuration.get_config_value("LINEAR_REGRESSION_PATH")
        self.model_path = self.configuration.get_config_value("MODEL_PATH")
        self.training_data_path = self.configuration.get_config_value("TRAINING_DATA_PATH")
        os.makedirs(self.linear_regression_path, exist_ok=True)

    async def initialize(self) -> None:
        """
        Initialize the MarketMonitor by loading a price model and historical data.
        Schedules periodic updates for training data and model retraining.
        """
        try:
            # Load or initialize the price model
            if os.path.exists(self.model_path):
                try:
                    self.price_model = joblib.load(self.model_path)
                    logger.debug("Loaded existing price model.")
                except Exception as e:
                    logger.warning(f"Loading model failed: {e}; creating a new model.")
                    self.price_model = LinearRegression()
                    joblib.dump(self.price_model, self.model_path)
            else:
                self.price_model = LinearRegression()
                joblib.dump(self.price_model, self.model_path)

            # Load historical training data if available
            if os.path.exists(self.training_data_path):
                try:
                    self.historical_data = pd.read_csv(self.training_data_path)
                    logger.debug(f"Loaded {len(self.historical_data)} historical data points.")
                except Exception as e:
                    logger.warning(f"Failed to load training data: {e}")
                    self.historical_data = pd.DataFrame()
            else:
                self.historical_data = pd.DataFrame()

            # Optionally, trigger model training if sufficient data exists (handled via APIConfig updates)
            if len(self.historical_data) >= self.MIN_TRAINING_SAMPLES:
                try:
                    await self.apiconfig.train_price_model()
                    logger.debug("Trained price model with historical data.")
                except Exception as e:
                    logger.warning(f"Model training failed: {e}")
                    self.price_model = LinearRegression()
                    joblib.dump(self.price_model, self.model_path)
                    self.historical_data = pd.DataFrame()
            else:
                logger.debug("Not enough historical data for model training.")
                self.price_model = LinearRegression()
                joblib.dump(self.price_model, self.model_path)
                self.historical_data = pd.DataFrame()
                pass

            logger.info("MarketMonitor initialized.")
            asyncio.create_task(self.schedule_updates())
        except Exception as e:
            logger.critical(f"MarketMonitor initialization failed: {e}", exc_info=True)
            raise

    async def schedule_updates(self) -> None:
        """
        Periodically update training data and retrain the price model based on the configured interval.
        """
        while True:
            try:
                current_time = time.time()
                if current_time - self.update_scheduler['training_data'] >= self.update_scheduler['model_retraining_interval']:
                    await self.apiconfig.update_training_data()
                    self.update_scheduler['training_data'] = current_time
                if current_time - self.update_scheduler['model'] >= self.update_scheduler['model_retraining_interval']:
                    await self.apiconfig.train_price_model()
                    self.update_scheduler['model'] = current_time
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                logger.info("Update scheduler task cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in update scheduler: {e}", exc_info=True)
                await asyncio.sleep(300)

    async def check_market_conditions(self, token_address: str) -> Dict[str, bool]:
        """
        Evaluate market conditions for a given token based on historical price data and volume.

        Args:
            token_address (str): The token contract address.

        Returns:
            Dict[str, bool]: A dictionary indicating various market condition flags.
        """
        market_conditions = {
            "high_volatility": False,
            "bullish_trend": False,
            "bearish_trend": False,
            "low_liquidity": False,
        }
        symbol = self.apiconfig.get_token_symbol(token_address)
        if not symbol:
            logger.debug(f"Cannot get token symbol for address {token_address} to check market conditions.")
            return market_conditions
        try:
            api_symbol = self.apiconfig._normalize_symbol(symbol)
            prices = await self.apiconfig.get_token_price_data(api_symbol, 'historical', timeframe=1)
            if not prices or len(prices) < 2:
                logger.debug(f"Not enough price data for {symbol} (prices: {prices}).")
                return market_conditions

            volatility = self.apiconfig._calculate_volatility(prices)
            if volatility > self.VOLATILITY_THRESHOLD:
                market_conditions["high_volatility"] = True

            moving_average = np.mean(prices)
            if prices[-1] > moving_average:
                market_conditions["bullish_trend"] = True
            elif prices[-1] < moving_average:
                market_conditions["bearish_trend"] = True

            volume = await self.get_token_volume(api_symbol)
            if volume < self.LIQUIDITY_THRESHOLD:
                market_conditions["low_liquidity"] = True

            logger.debug(f"Market conditions for {symbol}: {market_conditions}")
        except Exception as e:
            logger.error(f"Error checking market conditions for {symbol}: {e}", exc_info=True)
        return market_conditions

    async def predict_price_movement(self, token_symbol: str) -> float:
        """
        Predict future price movement for a token using a trained model.

        Args:
            token_symbol (str): The token symbol.

        Returns:
            float: The predicted price.
        """
        try:
            cache_key = f"prediction_{token_symbol}"
            if cache_key in self.prediction_cache:
                return self.prediction_cache[cache_key]
            prediction = await self.apiconfig.predict_price(token_symbol)
            self.prediction_cache[cache_key] = prediction
            return prediction
        except Exception as e:
            logger.error(f"Error predicting price movement: {e}", exc_info=True)
            return 0.0

    async def _get_market_features(self, token_symbol: str) -> Optional[Dict[str, float]]:
        """
        Gather market features required for prediction and risk assessment.

        Args:
            token_symbol (str): The token symbol.

        Returns:
            Optional[Dict[str, float]]: Dictionary of market features, or None if data is incomplete.
        """
        try:
            api_symbol = self.apiconfig._normalize_symbol(token_symbol)
            # Gather data concurrently
            results = await asyncio.gather(
                self.apiconfig.get_real_time_price(api_symbol),
                self.apiconfig.get_token_volume(api_symbol),
                self.apiconfig.get_token_supply_data(api_symbol),
                self._get_trading_metrics(api_symbol),
                self.get_token_price(api_symbol, data_type='historical', timeframe=1),
                return_exceptions=True
            )
            price, volume, supply_data, market_data, prices = results
            if any(isinstance(r, Exception) for r in results):
                logger.warning(f"Error gathering market data for {token_symbol}: {results}")
                return None

            features = {
                'market_cap': await self.apiconfig.get_token_market_cap(token_symbol),
                'volume_24h': float(volume),
                'percent_change_24h': await self.apiconfig.get_price_change_24h(token_symbol),
                'total_supply': supply_data.get('total_supply', 0),
                'circulating_supply': supply_data.get('circulating_supply', 0),
                'volatility': self.apiconfig._calculate_volatility(prices) if prices else 0,
                'price_momentum': self.apiconfig._calculate_momentum(prices) if prices else 0,
                'liquidity_ratio': await self._calculate_liquidity_ratio(token_symbol),
                **market_data
            }
            logger.debug(f"Market features for {token_symbol}: {features}")
            return features
        except Exception as e:
            logger.error(f"Error fetching market features for {token_symbol}: {e}", exc_info=True)
            return None

    async def _get_trading_metrics(self, token_symbol: str) -> Dict[str, float]:
        """
        Retrieve trading metrics for a token.

        Args:
            token_symbol (str): The token symbol.

        Returns:
            Dict[str, float]: Trading metrics including average transaction value, trading pairs count,
            exchange count, buy/sell ratio, and smart money flow.
        """
        try:
            metrics = {
                'avg_transaction_value': await self._get_avg_transaction_value(token_symbol),
                'trading_pairs': await self._get_trading_pairs_count(token_symbol),
                'exchange_count': await self._get_exchange_count(token_symbol),
                'buy_sell_ratio': await self._get_buy_sell_ratio(token_symbol),
                'smart_money_flow': await self._get_smart_money_flow(token_symbol)
            }
            logger.debug(f"Trading metrics for {token_symbol}: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Error getting trading metrics: {e}", exc_info=True)
            return {
                'avg_transaction_value': 0.0,
                'trading_pairs': 0.0,
                'exchange_count': 0.0,
                'buy_sell_ratio': 1.0,
                'smart_money_flow': 0.0
            }

    async def _get_avg_transaction_value(self, token_symbol: str) -> float:
        """
        Calculate the average transaction value for a token.

        Args:
            token_symbol (str): The token symbol.

        Returns:
            float: The average transaction value.
        """
        try:
            volume = await self.get_token_volume(token_symbol)
            tx_count = await self._get_transaction_count(token_symbol)
            avg_value = volume / tx_count if tx_count > 0 else 0.0
            logger.debug(f"Average transaction value for {token_symbol}: {avg_value}")
            return avg_value
        except Exception as e:
            logger.error(f"Error calculating average transaction value: {e}", exc_info=True)
            return 0.0

    async def _get_transaction_count(self, token_symbol: str) -> int:
        """
        Get the total number of transactions for a token.
        (Placeholder: returns 0 as a default.)

        Args:
            token_symbol (str): The token symbol.

        Returns:
            int: The number of transactions.
        """
        try:
            return 0
        except Exception as e:
            logger.error(f"Error getting transaction count: {e}", exc_info=True)
            return 0

    async def _get_trading_pairs_count(self, token_symbol: str) -> int:
        """
        Get the count of trading pairs for a token.

        Args:
            token_symbol (str): The token symbol.

        Returns:
            int: Number of trading pairs.
        """
        try:
            metadata = await self.apiconfig.get_token_metadata(token_symbol)
            count = len(metadata.get('trading_pairs', [])) if metadata else 0
            logger.debug(f"Trading pairs count for {token_symbol}: {count}")
            return count
        except Exception as e:
            logger.error(f"Error getting trading pairs for {token_symbol}: {e}", exc_info=True)
            return 0

    async def _get_exchange_count(self, token_symbol: str) -> int:
        """
        Get the number of exchanges where a token is traded.

        Args:
            token_symbol (str): The token symbol.

        Returns:
            int: The exchange count.
        """
        try:
            metadata = await self.apiconfig.get_token_metadata(token_symbol)
            count = len(metadata.get('exchanges', [])) if metadata else 0
            logger.debug(f"Exchange count for {token_symbol}: {count}")
            return count
        except Exception as e:
            logger.error(f"Error getting exchange count for {token_symbol}: {e}", exc_info=True)
            return 0

    async def _get_buy_sell_ratio(self, token_symbol: str) -> float:
        """
        Get the buy/sell ratio for a token.
        (Placeholder: returns 1.0)

        Args:
            token_symbol (str): The token symbol.

        Returns:
            float: The buy/sell ratio.
        """
        try:
            ratio = 1.0
            logger.debug(f"Buy/sell ratio for {token_symbol}: {ratio}")
            return ratio
        except Exception as e:
            logger.error(f"Error calculating buy/sell ratio: {e}", exc_info=True)
            return 1.0

    async def _get_smart_money_flow(self, token_symbol: str) -> float:
        """
        Get the smart money flow score for a token.
        (Placeholder: returns 0.0)

        Args:
            token_symbol (str): The token symbol.

        Returns:
            float: The smart money flow score.
        """
        try:
            score = 0.0
            logger.debug(f"Smart money flow score for {token_symbol}: {score}")
            return score
        except Exception as e:
            logger.error(f"Error calculating smart money flow: {e}", exc_info=True)
            return 0.0

    async def get_token_volume(self, token_symbol: str) -> float:
        """
        Retrieve the 24-hour trading volume for a token.

        Args:
            token_symbol (str): The token symbol.

        Returns:
            float: The trading volume.
        """
        return await self.apiconfig.get_token_volume(token_symbol)

    async def get_token_price(self, token_symbol: str, data_type: str = 'current', timeframe: int = 1, vs_currency: str = 'eth') -> Union[float, List[float]]:
        """
        Retrieve token price data.

        Args:
            token_symbol (str): The token symbol.
            data_type (str): 'current' for latest price, 'historical' for historical data.
            timeframe (int): Number of days for historical data.
            vs_currency (str): Target currency (default "eth").

        Returns:
            Union[float, List[float]]: The price data.
        """
        return await self.apiconfig.get_token_price_data(token_symbol, data_type, timeframe, vs_currency)

    async def _calculate_liquidity_ratio(self, token: str) -> float:
        """
        Calculate the liquidity ratio as volume divided by market cap.

        Args:
            token (str): The token symbol.

        Returns:
            float: The liquidity ratio.
        """
        try:
            volume = await self.get_token_volume(token)
            metadata = await self.apiconfig.get_token_metadata(token)
            market_cap = metadata.get('market_cap', 0) if metadata else 0
            return volume / market_cap if market_cap > 0 else 0.0
        except Exception as e:
            logger.error(f"Error calculating liquidity ratio: {e}")
            return 0.0

    async def stop(self) -> None:
        """
        Stop the MarketMonitor by clearing caches.
        """
        try:
            for cache in self.caches.values():
                cache.clear()
            logger.info("MarketMonitor stopped.")
        except Exception as e:
            logger.error(f"Error stopping MarketMonitor: {e}", exc_info=True)

    async def _is_arbitrage_opportunity(self, token_symbol: str) -> bool:
        """
        Determine if there is an arbitrage opportunity for a token.

        Args:
            token_symbol (str): The token symbol.

        Returns:
            bool: True if arbitrage opportunity exists, else False.
        """
        try:
            price = await self.apiconfig.get_real_time_price(token_symbol)
            if not price:
                return False
            price_data = await self.get_token_price(token_symbol, 'historical', 1)
            if not price_data or len(price_data) < 2:
                return False
            price_diff = price - price_data[0]
            return price_diff > 0
        except Exception as e:
            logger.error(f"Error checking arbitrage opportunity: {e}", exc_info=True)
            return False

    async def _get_trading_metrics(self, token_symbol: str) -> Dict[str, float]:
        """
        Retrieve additional trading metrics for a token.

        Args:
            token_symbol (str): The token symbol.

        Returns:
            Dict[str, float]: A dictionary of trading metrics.
        """
        try:
            metrics = {
                'avg_transaction_value': await self._get_avg_transaction_value(token_symbol),
                'trading_pairs': await self._get_trading_pairs_count(token_symbol),
                'exchange_count': await self._get_exchange_count(token_symbol),
                'buy_sell_ratio': await self._get_buy_sell_ratio(token_symbol),
                'smart_money_flow': await self._get_smart_money_flow(token_symbol)
            }
            logger.debug(f"Trading metrics for {token_symbol}: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Error getting trading metrics: {e}", exc_info=True)
            return {
                'avg_transaction_value': 0.0,
                'trading_pairs': 0.0,
                'exchange_count': 0.0,
                'buy_sell_ratio': 1.0,
                'smart_money_flow': 0.0
            }

    # The following methods for calculating trading metrics are placeholders.
    async def _get_avg_transaction_value(self, token_symbol: str) -> float:
        try:
            volume = await self.get_token_volume(token_symbol)
            tx_count = await self._get_transaction_count(token_symbol)
            avg_value = volume / tx_count if tx_count > 0 else 0.0
            logger.debug(f"Average transaction value for {token_symbol}: {avg_value}")
            return avg_value
        except Exception as e:
            logger.error(f"Error calculating average transaction value: {e}", exc_info=True)
            return 0.0

    async def _get_transaction_count(self, token_symbol: str) -> int:
        try:
            # Placeholder for actual transaction count
            return 0
        except Exception as e:
            logger.error(f"Error getting transaction count: {e}", exc_info=True)
            return 0

    async def _get_trading_pairs_count(self, token_symbol: str) -> int:
        try:
            metadata = await self.apiconfig.get_token_metadata(token_symbol)
            count = len(metadata.get('trading_pairs', [])) if metadata else 0
            logger.debug(f"Trading pairs count for {token_symbol}: {count}")
            return count
        except Exception as e:
            logger.error(f"Error getting trading pairs for {token_symbol}: {e}", exc_info=True)
            return 0

    async def _get_exchange_count(self, token_symbol: str) -> int:
        try:
            metadata = await self.apiconfig.get_token_metadata(token_symbol)
            count = len(metadata.get('exchanges', [])) if metadata else 0
            logger.debug(f"Exchange count for {token_symbol}: {count}")
            return count
        except Exception as e:
            logger.error(f"Error getting exchange count for {token_symbol}: {e}", exc_info=True)
            return 0

    async def _get_buy_sell_ratio(self, token_symbol: str) -> float:
        try:
            ratio = 1.0
            logger.debug(f"Buy/sell ratio for {token_symbol}: {ratio}")
            return ratio
        except Exception as e:
            logger.error(f"Error calculating buy/sell ratio: {e}", exc_info=True)
            return 1.0

    async def _get_smart_money_flow(self, token_symbol: str) -> float:
        try:
            score = 0.0
            logger.debug(f"Smart money flow for {token_symbol}: {score}")
            return score
        except Exception as e:
            logger.error(f"Error calculating smart money flow: {e}", exc_info=True)
            return 0.0
