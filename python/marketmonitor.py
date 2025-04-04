# LICENSE: MIT // github.com/John0n1/ON1Builder

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
        """
        self.web3 = web3
        self.configuration = configuration
        self.apiconfig = apiconfig
        self.transactioncore = transactioncore

        # Model and training related attributes.
        self.price_model: Optional[LinearRegression] = None
        self.last_training_time = 0
        self.model_accuracy = 0.0
        self.RETRAINING_INTERVAL = self.configuration.MODEL_RETRAINING_INTERVAL
        self.MIN_TRAINING_SAMPLES = self.configuration.MIN_TRAINING_SAMPLES
        self.historical_data = pd.DataFrame()
        self.prediction_cache = TTLCache(maxsize=1000, ttl=300)

        # Scheduler for periodic updates.
        self.update_scheduler = {
            'training_data': 0,
            'model': 0,
            'model_retraining_interval': self.configuration.MODEL_RETRAINING_INTERVAL
        }

        # Caches for frequently used data.
        self.caches = {
            'price': TTLCache(maxsize=2000, ttl=300),
            'volume': TTLCache(maxsize=1000, ttl=900),
            'volatility': TTLCache(maxsize=200, ttl=600)
        }

        # Ensure the directory for model training exists.
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
            # Load or initialize the price model.
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

            # Load historical training data if available.
            if os.path.exists(self.training_data_path):
                try:
                    self.historical_data = pd.read_csv(self.training_data_path)
                    logger.debug(f"Loaded {len(self.historical_data)} historical data points.")
                except Exception as e:
                    logger.warning(f"Failed to load training data: {e}")
                    self.historical_data = pd.DataFrame()
            else:
                self.historical_data = pd.DataFrame()

            # Train model if sufficient data exists.
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
        """
        market_conditions = {
            "high_volatility": False,
            "bullish_trend": False,
            "bearish_trend": False,
            "low_liquidity": False,
        }
        # Get token symbol using APIConfig.
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
        (Placeholder: This method should aggregate multiple metrics.)
        """
        try:
            api_symbol = self.apiconfig._normalize_symbol(token_symbol)
            results = await asyncio.gather(
                self.apiconfig.get_real_time_price(api_symbol),
                self.apiconfig.get_token_volume(api_symbol),
                self.apiconfig.get_token_supply_data(api_symbol),
                self._get_trading_metrics(api_symbol),
                self.get_token_price(api_symbol, data_type='historical', timeframe=1),
                return_exceptions=True
            )
            if any(isinstance(r, Exception) for r in results):
                logger.warning(f"Error gathering market data for {token_symbol}: {results}")
                return None
            return {
                'market_cap': await self.apiconfig.get_token_market_cap(token_symbol),
                'volume_24h': float(results[1]),
                'percent_change_24h': await self.apiconfig.get_price_change_24h(token_symbol),
                'total_supply': results[2].get('total_supply', 0) if isinstance(results[2], dict) else 0,
                'circulating_supply': results[2].get('circulating_supply', 0) if isinstance(results[2], dict) else 0,
                'volatility': self.apiconfig._calculate_volatility(results[4]) if results[4] else 0,
                'price_momentum': self.apiconfig._calculate_momentum(results[4]) if results[4] else 0,
                'liquidity_ratio': await self._calculate_liquidity_ratio(token_symbol),
                **(results[3] if isinstance(results[3], dict) else {})
            }
        except Exception as e:
            logger.error(f"Error fetching market features for {token_symbol}: {e}", exc_info=True)
            return None

    async def _get_trading_metrics(self, token_symbol: str) -> Dict[str, float]:
        """
        Retrieve trading metrics for a token.
        (Placeholder: These methods need proper implementation.)
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
        (Placeholder: Implementation required.)
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
        (Placeholder: returns 0 by default.)
        """
        try:
            return 0
        except Exception as e:
            logger.error(f"Error getting transaction count: {e}", exc_info=True)
            return 0

    async def _get_trading_pairs_count(self, token_symbol: str) -> int:
        """
        Get the count of trading pairs for a token.
        (Placeholder: Implementation required.)
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
        (Placeholder: Implementation required.)
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
        """
        return await self.apiconfig.get_token_volume(token_symbol)

    async def get_token_price(self, token_symbol: str, data_type: str = 'current', timeframe: int = 1, vs_currency: str = 'eth') -> Union[float, List[float]]:
        """
        Retrieve token price data.
        """
        return await self.apiconfig.get_token_price_data(token_symbol, data_type, timeframe, vs_currency)

    async def _calculate_liquidity_ratio(self, token: str) -> float:
        """
        Calculate the liquidity ratio as volume divided by market cap.
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
        (Simplistic placeholder implementation.)
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
        (Placeholder implementation.)
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

    async def update_training_data(self) -> None:
        """
        Update training data for model training by fetching data for all tokens.
        """
        try:
            token_addresses = await self.configuration.get_token_addresses()
            update_tasks = [self._gather_training_data(token) for token in token_addresses]
            updates = await asyncio.gather(*update_tasks, return_exceptions=True)
            valid_updates = [update for update in updates if isinstance(update, dict)]
            if valid_updates:
                await self._write_training_data(valid_updates)
                await self.train_price_model()
        except Exception as e:
            logger.error(f"Error updating training data: {e}")

    async def _gather_training_data(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Gather training data for a token for model training.
        (Placeholder: Data aggregation logic required.)
        """
        try:
            data = await asyncio.gather(
                self.get_real_time_price(token),
                self.get_token_volume(token),
                self._fetch_market_data(token),
                return_exceptions=True
            )
            if any(isinstance(r, Exception) for r in data):
                logger.warning(f"Error gathering data for {token}")
                return None
            return {
                'timestamp': int(time.time()),
                'symbol': token,
                'price_usd': float(data[0]),
                'volume_24h': float(data[1]),
                **(data[2] if isinstance(data[2], dict) else {})
            }
        except Exception as e:
            logger.error(f"Error gathering training data: {e}")
            return None

    async def _fetch_market_data(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Fetch comprehensive market data for a token.
        (Placeholder: Simplistic aggregation.)
        """
        try:
            cache_key = f"market_data_{token}"
            if cache_key in self.caches['price']:
                return self.caches['price'][cache_key]
            metadata = await self.apiconfig.get_token_metadata(token)
            volume = await self.get_token_volume(token)
            price_history = await self.get_token_price(token, 'historical', timeframe=7)
            price_volatility = self.apiconfig._calculate_volatility(price_history) if price_history else 0
            market_data = {
                'market_cap': metadata.get('market_cap', 0) if metadata else 0,
                'volume_24h': float(volume) if volume else 0,
                'percent_change_24h': metadata.get('percent_change_24h', 0) if metadata else 0,
                'total_supply': metadata.get('total_supply', 0) if metadata else 0,
                'circulating_supply': metadata.get('circulating_supply', 0) if metadata else 0,
                'volatility': price_volatility,
                'price_momentum': self.apiconfig._calculate_momentum(price_history) if price_history else 0,
                'liquidity_ratio': await self._calculate_liquidity_ratio(token),
                'trading_pairs': len(metadata.get('trading_pairs', [])) if metadata else 0,
                'exchange_count': len(metadata.get('exchanges', [])) if metadata else 0
            }
            self.caches['price'][cache_key] = market_data
            return market_data
        except Exception as e:
            logger.error(f"Error fetching market data for {token}: {e}", exc_info=True)
            return None

    async def train_price_model(self) -> None:
        """
        Train a linear regression model for price prediction and save the model.
        """
        try:
            training_data_path = self.configuration.TRAINING_DATA_PATH
            model_path = self.configuration.MODEL_PATH
            if not os.path.exists(training_data_path):
                logger.warning("No training data found")
                return
            df = pd.read_csv(training_data_path)
            if len(df) < self.configuration.MIN_TRAINING_SAMPLES:
                logger.warning(f"Insufficient training data: {len(df)} samples")
                return
            features = ['price_usd', 'volume_24h', 'market_cap', 'volatility',
                        'liquidity_ratio', 'price_momentum']
            X = df[features].fillna(0)
            y = df['price_usd'].fillna(0)
            model = LinearRegression()
            model.fit(X, y)
            joblib.dump(model, model_path)
            logger.debug(f"Price model trained and saved to {model_path}")
        except Exception as e:
            logger.error(f"Error training price model: {e}")

    async def predict_price(self, token: str) -> float:
        """
        Predict the price of a token using the trained model.
        """
        try:
            model_path = self.configuration.MODEL_PATH
            if not os.path.exists(model_path):
                await self.train_price_model()
                if not os.path.exists(model_path):
                    return 0.0
            model = joblib.load(model_path)
            market_data = await self._fetch_market_data(token)
            if not market_data:
                return 0.0
            features = pd.DataFrame([market_data])
            prediction = model.predict(features)[0]
            return float(prediction)
        except Exception as e:
            logger.error(f"Error predicting price: {e}")
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

# --- End file: marketmonitor.py ---
