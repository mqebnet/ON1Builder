import asyncio
from decimal import Decimal
import os
import time
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, AsyncGenerator, Set
from sklearn.linear_model import LinearRegression
from cachetools import TTLCache
from web3 import AsyncWeb3
from pathlib import Path
import aiofiles

# Assuming configuration facade and APIConfig
from configuration import Configuration
from apiconfig import APIConfig
 # A5: Needed for injection
from loggingconfig import setup_logging
import logging
from transactioncore import TransactionCore # A5: Import TransactionCore

logger = setup_logging("MarketMonitor", level=logging.DEBUG)


class MarketMonitor:
    """
    Monitors real-time market data, manages historical data persistence,
    trains a price prediction model, and provides market condition analysis.
    """
    # Class constants for thresholds (can be overridden by config)
    DEFAULT_VOLATILITY_THRESHOLD: float = 0.05
    DEFAULT_LIQUIDITY_THRESHOLD: float = 100_000 # In USD volume

    # Model/Data related constants
    DEFAULT_RETRAINING_INTERVAL: int = 3600  # seconds
    DEFAULT_MIN_TRAINING_SAMPLES: int = 100
    DEFAULT_TRAINING_CHUNK_SIZE: int = 10000 # For stream reading if needed

    def __init__(
        self,
        web3: AsyncWeb3, # Keep web3 if needed for on-chain data access
        configuration: Configuration,
        apiconfig: APIConfig,
        transactioncore: TransactionCore, # A5: Inject TransactionCore
    ) -> None:

        self.web3 = web3
        self.configuration = configuration
        self.apiconfig = apiconfig
        self.transactioncore = transactioncore # A5: Store injected instance

        self.price_model: Optional[LinearRegression] = None
        self.last_training_time: float = 0.0
        self.model_accuracy: float = 0.0 # Consider implementing accuracy calculation

        # Load settings from configuration, using defaults if not present
        self.retraining_interval: int = self.configuration.get_config_value(
            "MODEL_RETRAINING_INTERVAL", self.DEFAULT_RETRAINING_INTERVAL
        )
        self.min_training_samples: int = self.configuration.get_config_value(
            "MIN_TRAINING_SAMPLES", self.DEFAULT_MIN_TRAINING_SAMPLES
        )
        self.volatility_threshold: float = self.configuration.get_config_value(
            "VOLATILITY_THRESHOLD", self.DEFAULT_VOLATILITY_THRESHOLD
        )
        self.liquidity_threshold: float = self.configuration.get_config_value(
            "LIQUIDITY_THRESHOLD", self.DEFAULT_LIQUIDITY_THRESHOLD
        )

        # Cache for market data (prices, volumes etc.)
        self.market_data_cache: TTLCache = TTLCache(maxsize=5000, ttl=300) # Increased size

        # Paths (consider resolving via config_paths module if A15 is implemented)
        base_path = self.configuration.BASE_PATH
        self.model_dir = Path(self.configuration.get_config_value("MODEL_DIR", base_path / "runtime" / "models"))
        self.data_dir = Path(self.configuration.get_config_value("DATA_DIR", base_path / "runtime" / "data"))
        self.model_path = self.model_dir / "price_model.joblib"
        self.training_data_path = self.data_dir / "training_data.csv"

        # Ensure directories exist
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # A10: Store task handles for graceful shutdown
        self._scheduled_tasks: Set[asyncio.Task] = set()
        self._is_running: bool = False

        # A2: Parameterized logging
        logger.info(
            "MarketMonitor initialized. Model path: %s, Data path: %s",
            self.model_path, self.training_data_path
        )

    # --------------------------------------------------------------------- #
    # INITIALISATION & SCHEDULING                                           #
    # --------------------------------------------------------------------- #
    async def initialize(self) -> None:
        """
        Loads the price prediction model and historical data, then starts
        the periodic update tasks.
        """
        if self._is_running:
            logger.warning("MarketMonitor already initialized and running.")
            return

        # A2: Parameterized logging
        logger.info("Initializing MarketMonitor...")
        try:
            # Load the price model
            await self._load_model()

            # Check if sufficient data exists and train initially if needed
            # No need to load all data into memory here
            if await self._has_sufficient_training_data():
                 # A2: Parameterized logging
                 logger.info("Sufficient training data found. Training initial model...")
                 await self.train_price_model()
            else:
                # A2: Parameterized logging
                logger.info("Insufficient training data. Model will be trained after data collection.")

            self._is_running = True
            # A10: Start and store the scheduler task handle
            scheduler_task = asyncio.create_task(self._schedule_updates())
            self._scheduled_tasks.add(scheduler_task)
            # Ensure the task is tracked even if it finishes/errors immediately
            scheduler_task.add_done_callback(self._scheduled_tasks.discard)

            # A2: Parameterized logging
            logger.info("MarketMonitor initialized successfully. Update scheduler started.")

        except Exception as e:
            # A2: Parameterized logging
            logger.critical("MarketMonitor initialization failed: %s", e, exc_info=True)
            self._is_running = False # Ensure state reflects failure
            raise # Re-raise to signal failure to MainCore

    async def _load_model(self) -> None:
        """Loads the price prediction model from disk."""
        if await asyncio.to_thread(self.model_path.exists):
            try:
                # Use asyncio.to_thread for blocking joblib call
                self.price_model = await asyncio.to_thread(joblib.load, self.model_path)
                # A2: Parameterized logging
                logger.info("Loaded existing price model from %s.", self.model_path)
            except Exception as e:
                # A2: Parameterized logging
                logger.warning("Failed to load model from %s: %s. Will create a new one.", self.model_path, e)
                self.price_model = LinearRegression()
                # Optionally save the new empty model immediately
                # await asyncio.to_thread(joblib.dump, self.price_model, self.model_path)
        else:
            # A2: Parameterized logging
            logger.info("No existing model found at %s. Creating a new model.", self.model_path)
            self.price_model = LinearRegression()

    async def _has_sufficient_training_data(self) -> bool:
        """Checks if the training data file exists and has enough rows without loading it all."""
        if not await asyncio.to_thread(self.training_data_path.exists):
            return False
        try:
            # Quickly check row count (approximation for large files)
            # This is simpler than full streaming just for a count check
            # For very large files, a more optimized row count might be needed
            async with aiofiles.open(self.training_data_path, mode='r', encoding='utf-8') as f:
                 # Read a small chunk to get header, then estimate based on size / first few lines
                 # Simpler: just count lines for moderate files
                 line_count = 0
                 async for _ in f:
                     line_count += 1
                     if line_count > self.min_training_samples: # Stop early if enough found
                          return True
            return line_count > self.min_training_samples # Check exact count if loop finished
        except Exception as e:
            logger.warning("Error checking training data size: %s", e)
            return False


    async def _schedule_updates(self) -> None:
        """Periodically updates training data and retrains the price model."""
        # A2: Parameterized logging
        logger.info("Starting MarketMonitor update loop (Interval: %d s).", self.retraining_interval)
        last_update_time = time.monotonic()

        while self._is_running:
            try:
                # Calculate time until next update
                now = time.monotonic()
                time_since_last = now - last_update_time
                sleep_duration = max(0, self.retraining_interval - time_since_last)

                # A2: Parameterized logging
                logger.debug("Scheduler sleeping for %.2f seconds.", sleep_duration)
                await asyncio.sleep(sleep_duration)

                if not self._is_running: break # Check again after sleep

                last_update_time = time.monotonic() # Reset timer before starting work

                # A2: Parameterized logging
                logger.info("Running scheduled market update...")
                # Use TaskGroup for concurrent updates if needed, otherwise sequential is fine
                await self.update_training_data()
                await self.train_price_model()
                # A2: Parameterized logging
                logger.info("Scheduled market update completed.")

            except asyncio.CancelledError:
                # A2: Parameterized logging
                logger.info("MarketMonitor update scheduler task cancelled.")
                break # Exit the loop cleanly on cancellation
            except Exception as e:
                # A2: Parameterized logging
                logger.error("Error in MarketMonitor update scheduler: %s", e, exc_info=True)
                # Avoid busy-looping on persistent errors
                await asyncio.sleep(60)


    # --------------------------------------------------------------------- #
    # MARKET CONDITION ANALYSIS                                             #
    # --------------------------------------------------------------------- #
    async def check_market_conditions(self, token_address_or_symbol: str) -> Dict[str, Union[bool, float]]:
        """
        Evaluates market conditions (volatility, trend, liquidity) for a token.

        Args:
            token_address_or_symbol: The token's address or symbol.

        Returns:
            Dictionary with condition flags and values.
        """
        log_extra = {"component": "MarketMonitor", "token": token_address_or_symbol} # A16
        conditions = {
            "high_volatility": False,
            "bullish_trend": False, # Trend based on recent price movement
            "low_liquidity": False,
            "volatility_value": 0.0,
            "liquidity_usd": 0.0,
            "price_change_pct": 0.0 # e.g., 24h change
        }

        if not token_address_or_symbol:
            # A2: Parameterized logging
            logger.debug("Token address/symbol is empty, cannot check market conditions.", extra=log_extra)
            return conditions

        # Use APIConfig to get symbol/address and data
        token_symbol = self.apiconfig.get_token_symbol(token_address_or_symbol) if token_address_or_symbol.startswith("0x") else token_address_or_symbol.upper()
        if not token_symbol:
            # A2: Parameterized logging
            logger.warning("Could not determine symbol for %s.", token_address_or_symbol, extra=log_extra)
            return conditions
        log_extra["token_symbol"] = token_symbol # Add symbol to context

        try:
            # Use cached data if available, otherwise fetch
            cache_key = f"market_conditions:{token_symbol}"
            if cache_key in self.market_data_cache:
                 cached_conditions = self.market_data_cache[cache_key]
                 logger.debug("Using cached market conditions for %s", token_symbol, extra=log_extra)
                 return cached_conditions

            # Fetch required data concurrently if possible
            # Example: Get 7 days of prices for volatility/trend, 24h volume for liquidity
            price_task = asyncio.create_task(self.apiconfig.get_token_price_data(token_symbol, data_type="historical", timeframe=7, vs="usd"))
            volume_task = asyncio.create_task(self.apiconfig.get_token_volume(token_symbol)) # Assuming volume is 24h USD

            prices_usd = await price_task
            volume_24h_usd = await volume_task

            if isinstance(prices_usd, list) and len(prices_usd) >= 2:
                prices_array = np.array([float(p) for p in prices_usd]) # Ensure float
                mean_price = np.mean(prices_array)
                if mean_price > 0:
                    std_dev = np.std(prices_array)
                    volatility = std_dev / mean_price
                    conditions["volatility_value"] = volatility
                    if volatility > self.volatility_threshold:
                        conditions["high_volatility"] = True

                # Trend check: Compare last price to average or first price
                price_change = prices_array[-1] - prices_array[0]
                if prices_array[0] > 0:
                     conditions["price_change_pct"] = (price_change / prices_array[0]) * 100
                     if price_change > 0:
                         conditions["bullish_trend"] = True # Simple trend check
                else:
                     conditions["price_change_pct"] = float('inf') if price_change > 0 else float('-inf')
                     conditions["bullish_trend"] = price_change > 0


            else:
                 # A2: Parameterized logging
                 logger.debug("Insufficient historical price data for %s to calculate volatility/trend.", token_symbol, extra=log_extra)

            if volume_24h_usd is not None:
                conditions["liquidity_usd"] = float(volume_24h_usd)
                if volume_24h_usd < self.liquidity_threshold:
                    conditions["low_liquidity"] = True
            else:
                # A2: Parameterized logging
                logger.debug("Volume data unavailable for %s.", token_symbol, extra=log_extra)

            # Cache the computed conditions
            self.market_data_cache[cache_key] = conditions
            # A2: Parameterized logging
            logger.debug("Market conditions for %s: %s", token_symbol, conditions, extra=log_extra)

        except Exception as e:
            # A2: Parameterized logging
            logger.error("Error checking market conditions for %s: %s", token_symbol, e, exc_info=True, extra=log_extra)

        return conditions

    # --------------------------------------------------------------------- #
    # PRICE PREDICTION                                                      #
    # --------------------------------------------------------------------- #
    async def predict_price_movement(self, token_symbol: str) -> Optional[float]:
        """
        Predicts the future price (e.g., next hour/day in USD) for a token symbol
        using the trained model.

        Args:
            token_symbol: The token symbol (e.g., "ETH", "WBTC").

        Returns:
            Predicted price as float, or None if prediction fails.
        """
        log_extra = {"component": "MarketMonitor", "token_symbol": token_symbol} # A16
        if self.price_model is None:
            # A2: Parameterized logging
            logger.warning("Price model not loaded/trained. Cannot predict.", extra=log_extra)
            return None

        try:
            # Check cache first
            cache_key = f"prediction_{token_symbol}"
            if cache_key in self.market_data_cache:
                return self.market_data_cache[cache_key]

            # Prepare features for prediction (needs current market data)
            # This requires fetching similar features used in training
            features = await self._prepare_prediction_features(token_symbol, log_extra)
            if features is None:
                 return None # Error logged in helper

            # Use asyncio.to_thread for the potentially blocking predict call
            predicted_value = await asyncio.to_thread(self.price_model.predict, features)
            prediction = float(predicted_value[0]) # Model predicts a single value

            # Cache the prediction
            self.market_data_cache[cache_key] = prediction
            # A2: Parameterized logging
            logger.debug("Predicted price for %s: %.4f USD", token_symbol, prediction, extra=log_extra)
            return prediction

        except Exception as e:
            # A2: Parameterized logging
            logger.error("Error predicting price for %s: %s", token_symbol, e, exc_info=True, extra=log_extra)
            return None

    async def _prepare_prediction_features(self, token_symbol: str, log_extra: Dict) -> Optional[pd.DataFrame]:
        """Fetches current data and prepares the feature DataFrame for prediction."""
        try:
            # Fetch necessary data points (concurrently?)
            # Example: Need current price, volume, maybe recent volatility/momentum
            price_task = asyncio.create_task(self.apiconfig.get_token_price_data(token_symbol, "current", vs="usd"))
            volume_task = asyncio.create_task(self.apiconfig.get_token_volume(token_symbol))
            # Could also fetch recent historical for volatility/momentum if needed by model
            # hist_prices_task = asyncio.create_task(self.apiconfig.get_token_price_data(token_symbol, "historical", timeframe=7, vs="usd"))

            current_price_usd = await price_task
            volume_24h_usd = await volume_task
            # hist_prices_usd = await hist_prices_task

            if current_price_usd is None or volume_24h_usd is None:
                logger.warning("Missing required data (price/volume) for %s prediction.", token_symbol, extra=log_extra)
                return None

            # --- Construct Feature Row ---
            # This MUST match the features used during training *exactly*
            # Need to calculate: market_cap, volatility, liquidity_ratio, price_momentum
            # These calculations might require more API calls (e.g., for market cap, historical prices)
            # Placeholder: using simplified features for now
            # TODO: Implement full feature calculation matching training data
            feature_data = {
                # Features must match training order and names
                "price_usd": float(current_price_usd),
                "volume_24h": float(volume_24h_usd),
                "market_cap": 0.0, # Placeholder - Fetch market cap if model needs it
                "volatility": 0.0, # Placeholder - Calculate recent volatility if needed
                "liquidity_ratio": 0.0, # Placeholder - Calculate if needed (Volume / MarketCap)
                "price_momentum": 0.0, # Placeholder - Calculate recent price change if needed
                # Add other features if the model was trained with them
            }

            # Ensure order matches training
            feature_columns = [
                "price_usd", "volume_24h", "market_cap", "volatility",
                "liquidity_ratio", "price_momentum"
                # Add others if needed
            ]
            features_df = pd.DataFrame([feature_data], columns=feature_columns)
            features_df = features_df.fillna(0) # Handle potential NaNs

            return features_df

        except Exception as e:
            logger.error("Failed to prepare prediction features for %s: %s", token_symbol, e, exc_info=True, extra=log_extra)
            return None


    # --------------------------------------------------------------------- #
    # WRAPPER AROUND APIConfig (if needed, or use apiconfig directly)       #
    # --------------------------------------------------------------------- #
    async def get_token_price_data(
        self,
        token_symbol_or_address: str,
        data_type: str = "current", # "current" or "historical"
        timeframe: int = 1, # For historical: number of data points or days
        vs: str = "usd", # vs currency
    ) -> Union[Optional[Decimal], List[float]]: # Return type depends on data_type
        """Convenience wrapper around APIConfig for price data."""
        # Consider caching results here if APIConfig doesn't cache appropriately
        cache_key = f"price_data:{token_symbol_or_address}:{data_type}:{timeframe}:{vs}"
        if cache_key in self.market_data_cache:
             return self.market_data_cache[cache_key]

        result = await self.apiconfig.get_token_price_data(
            token_symbol_or_address, data_type, timeframe, vs
        )
        if result is not None:
             self.market_data_cache[cache_key] = result
        return result


    # --------------------------------------------------------------------- #
    # TRAINING DATA PIPELINE                                                #
    # --------------------------------------------------------------------- #
    # A13: Append directly to CSV
    async def update_training_data(self) -> None:
        """
        Fetches fresh market data for configured tokens and appends it
        directly to the training data CSV file.
        """
        log_extra = {"component": "MarketMonitor"} # A16
        # A2: Parameterized logging
        logger.info("Updating training data at %s...", self.training_data_path)

        # Get list of tokens to fetch data for (e.g., from configuration or APIConfig)
        # Ensure we handle both symbols and addresses if necessary
        token_symbols = list(self.apiconfig.token_symbol_to_address.keys()) # Assuming APIConfig holds the primary list

        new_rows_data = []
        fetch_tasks = []

        # Create tasks to fetch data for each token concurrently
        for symbol in token_symbols:
            if symbol and not symbol.startswith("_"): # Skip internal/comment keys
                fetch_tasks.append(asyncio.create_task(self._fetch_single_token_data(symbol)))

        # Gather results from all fetch tasks
        results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        for result, symbol in zip(results, token_symbols):
             if isinstance(result, dict):
                 new_rows_data.append(result)
             elif isinstance(result, Exception):
                 # A2: Parameterized logging
                  logger.error("Failed to fetch training data for token %s: %s", symbol, result, extra=log_extra)
             # If result is None (logged in helper), skip


        if not new_rows_data:
            # A2: Parameterized logging
            logger.info("No new training data fetched.")
            return

        # Convert list of dicts to DataFrame
        new_df = pd.DataFrame(new_rows_data)

        # Define columns explicitly to ensure order and handle missing ones
        # This order MUST match the order expected by the training process
        columns = [
            "timestamp", "symbol", "price_usd", "market_cap", "volume_24h",
            "percent_change_24h", "total_supply", "circulating_supply",
            "volatility", "liquidity_ratio", "avg_transaction_value", # Requires on-chain data?
            "trading_pairs", "exchange_count", "price_momentum",
            "buy_sell_ratio", # Requires sentiment/order book data?
            "smart_money_flow" # Requires on-chain analysis?
        ]
        # Reindex DF to match columns, filling missing with NaN or 0
        new_df = new_df.reindex(columns=columns)

        # A13: Append to CSV directly
        file_exists = await asyncio.to_thread(self.training_data_path.exists)
        mode = "a" if file_exists else "w"
        write_header = not file_exists

        try:
            # Use aiofiles for async write, but pandas to_csv is sync
            # Option 1: Convert to CSV string and write async
            csv_string = new_df.to_csv(index=False, header=write_header, lineterminator='\n')
            async with aiofiles.open(self.training_data_path, mode=mode, encoding='utf-8') as f:
                await f.write(csv_string)

            # Option 2: Run sync pandas write in thread (simpler)
            # await asyncio.to_thread(
            #     new_df.to_csv, self.training_data_path, mode=mode, header=write_header, index=False
            # )

            # A2: Parameterized logging
            logger.info(
                "Appended %d new rows to training data %s. File exists: %s, Header written: %s",
                len(new_df), self.training_data_path, file_exists, write_header, extra=log_extra
            )
        except Exception as e:
             # A2: Parameterized logging
             logger.error(
                "Failed to write training data to %s: %s",
                self.training_data_path, e, exc_info=True, extra=log_extra
             )


    async def _fetch_single_token_data(self, token_symbol: str) -> Optional[Dict[str, Any]]:
        """Fetches and calculates features for a single token for the training data."""
        log_extra = {"component": "MarketMonitor", "token_symbol": token_symbol} # A16
        try:
            # Fetch required data points using APIConfig (concurrently if possible within APIConfig)
            price_hist_task = asyncio.create_task(self.apiconfig.get_token_price_data(token_symbol, "historical", timeframe=1, vs="usd")) # 24h prices
            volume_task = asyncio.create_task(self.apiconfig.get_token_volume(token_symbol))
            metadata_task = asyncio.create_task(self.apiconfig.get_token_metadata(token_symbol)) # Assuming this exists in APIConfig

            prices_usd_24h = await price_hist_task
            volume_24h_usd = await volume_task
            metadata = await metadata_task

            # --- Validate data ---
            if not isinstance(prices_usd_24h, list) or len(prices_usd_24h) < 2:
                logger.debug("Insufficient 24h price data for %s.", token_symbol, extra=log_extra)
                return None
            if volume_24h_usd is None:
                 logger.debug("Volume data unavailable for %s.", token_symbol, extra=log_extra)
                 volume_24h_usd = 0.0 # Use 0 if unavailable
            if not metadata:
                 logger.debug("Metadata unavailable for %s.", token_symbol, extra=log_extra)
                 metadata = {} # Use empty dict if unavailable

            # --- Calculate features ---
            prices_array = np.array([float(p) for p in prices_usd_24h])
            current_price = prices_array[-1]
            avg_price = np.mean(prices_array)
            volatility = (np.std(prices_array) / avg_price) if avg_price > 0 else 0.0
            price_start_24h = prices_array[0]
            percent_change_24h = ((current_price - price_start_24h) / price_start_24h * 100) if price_start_24h > 0 else 0.0
            price_momentum = (current_price - price_start_24h) / price_start_24h if price_start_24h > 0 else 0.0 # Simple momentum

            market_cap = float(metadata.get("market_cap", 0.0))
            total_supply = float(metadata.get("total_supply", 0.0))
            circulating_supply = float(metadata.get("circulating_supply", 0.0))
            trading_pairs = int(metadata.get("trading_pairs", 0)) # Example metadata fields
            exchange_count = len(metadata.get("exchanges", [])) # Example metadata fields

            liquidity_ratio = float(volume_24h_usd / market_cap) if market_cap > 0 else 0.0

            # Placeholder features (require more complex data sources)
            avg_transaction_value = 0.0
            buy_sell_ratio = 1.0
            smart_money_flow = 0.0

            # Construct the row dictionary
            row = {
                "timestamp": int(datetime.utcnow().timestamp()),
                "symbol": token_symbol,
                "price_usd": current_price,
                "market_cap": market_cap,
                "volume_24h": float(volume_24h_usd),
                "percent_change_24h": percent_change_24h,
                "total_supply": total_supply,
                "circulating_supply": circulating_supply,
                "volatility": volatility,
                "liquidity_ratio": liquidity_ratio,
                "avg_transaction_value": avg_transaction_value,
                "trading_pairs": trading_pairs,
                "exchange_count": exchange_count,
                "price_momentum": price_momentum,
                "buy_sell_ratio": buy_sell_ratio,
                "smart_money_flow": smart_money_flow,
            }
            return row

        except Exception as e:
            logger.error("Error fetching data for token %s: %s", token_symbol, e, exc_info=True, extra=log_extra)
            return None # Return None on error


    # --------------------------------------------------------------------- #
    # MODEL TRAINING                                                        #
    # --------------------------------------------------------------------- #
    # A13: Potentially stream-read data for training
    async def train_price_model(self) -> None:
        """
        Trains the linear regression price model using data from the CSV file.
        Uses stream reading if data is potentially large.
        """
        log_extra = {"component": "MarketMonitor"} # A16
        if not await asyncio.to_thread(self.training_data_path.exists):
            # A2: Parameterized logging
            logger.warning("Training data file %s not found. Skipping training.", self.training_data_path, extra=log_extra)
            return

        # A2: Parameterized logging
        logger.info("Starting price model training using data from %s...", self.training_data_path, extra=log_extra)

        try:
            # Define features and target variable
            # Ensure these EXACTLY match the columns prepared in update_training_data and _prepare_prediction_features
            features_cols = [
                "price_usd", "volume_24h", "market_cap", "volatility",
                "liquidity_ratio", "price_momentum"
                # Add other feature column names here if used
            ]
            target_col = "price_usd" 
            model = LinearRegression() # Re-initialize model for fresh training
            chunks = []
            required_cols = features_cols + [target_col]

            # Use asyncio.to_thread for the blocking pandas read_csv call
            df_iterator = await asyncio.to_thread(
                pd.read_csv,
                self.training_data_path,
                iterator=True,
                chunksize=self.DEFAULT_TRAINING_CHUNK_SIZE,
                usecols=lambda c: c in required_cols # Only load necessary columns
            )

            processed_rows = 0
            for chunk in df_iterator:
                # Basic preprocessing within the chunk
                chunk = chunk.dropna(subset=[target_col]) # Drop rows where target is missing
                chunk[features_cols] = chunk[features_cols].fillna(0) # Fill missing features with 0
                if not chunk.empty:
                    chunks.append(chunk)
                    processed_rows += len(chunk)
                    logger.debug("Processed training chunk, total rows so far: %d", processed_rows)

            if not chunks:
                 logger.warning("No valid data chunks found after processing. Skipping training.", extra=log_extra)
                 return

            # Concatenate chunks into a single DataFrame
            full_df = pd.concat(chunks, ignore_index=True)

            if len(full_df) < self.min_training_samples:
                 logger.warning(
                    "Insufficient training samples after cleaning: %d (required: %d). Skipping training.",
                    len(full_df), self.min_training_samples, extra=log_extra
                 )
                 return

            # Prepare features (X) and target (y)
            X_train = full_df[features_cols]
            y_train = full_df[target_col]

            # Fit the model (run in thread)
            await asyncio.to_thread(model.fit, X_train, y_train)

            # Save the trained model (run in thread)
            await asyncio.to_thread(joblib.dump, model, self.model_path)

            self.price_model = model # Update the instance's model
            self.last_training_time = time.monotonic()

            # Optionally calculate and log model accuracy (e.g., R^2 score)
            # score = await asyncio.to_thread(model.score, X_train, y_train)
            # self.model_accuracy = score
            # logger.info(f"Model training completed. Accuracy (R^2): {score:.4f}. Model saved to {self.model_path}", extra=log_extra)
            # A2: Parameterized logging
            logger.info(
                "Price model training completed with %d samples. Model saved to %s",
                len(full_df), self.model_path, extra=log_extra
            )

        except FileNotFoundError:
             logger.error("Training data file %s not found during training.", self.training_data_path, extra=log_extra)
        except KeyError as e:
             logger.error("Missing expected column in training data: %s. Check CSV header and feature list.", e, extra=log_extra)
        except Exception as e:
            # A2: Parameterized logging
            logger.error("Error during price model training: %s", e, exc_info=True, extra=log_extra)
            # Do not update self.price_model if training failed


    async def _count_csv_rows(self) -> int:
        """Efficiently counts rows in a CSV, skipping header."""
        try:
             count = 0
             async with aiofiles.open(self.training_data_path, 'r', encoding='utf-8') as f:
                  await f.readline() # Skip header
                  async for _ in f:
                       count += 1
             return count
        except Exception:
             return 0
        
    async def calculate_optimal_sandwich_amount(
        self,
        token_symbol: str,
        target_price: float,
        slippage: float = 0.01, # 1% slippage
        gas_price: float = 100, # Example gas price in Gwei
    ) -> Optional[float]:
        """
        Calculates the optimal sandwich amount for a token based on market conditions.

        Args:
            token_symbol: The token symbol (e.g., "ETH").
            target_price: The target price for the sandwich.
            slippage: The acceptable slippage percentage.
            gas_price: The gas price in Gwei.

        Returns:
            Optimal sandwich amount in USD or None if calculation fails.
        """
        log_extra = {"component": "MarketMonitor", "token_symbol": token_symbol}
        try:
            # Fetch current price and volume
            current_price = await self.get_token_price_data(token_symbol, "current", vs="usd")
            volume_24h = await self.get_token_volume(token_symbol)

            if current_price is None or volume_24h is None:
                logger.warning("Failed to fetch current price or volume for %s.", token_symbol, extra=log_extra)
                return None

            # Calculate optimal sandwich amount based on market conditions
            # Placeholder formula: Adjust as needed based on actual market analysis
            optimal_amount = (target_price - current_price) * (1 + slippage) * (volume_24h / gas_price)
            return optimal_amount
        except Exception as e:
            logger.error("Error calculating optimal sandwich amount for %s: %s", token_symbol, e, exc_info=True, extra=log_extra)
            return None



    # --------------------------------------------------------------------- #
    # SHUTDOWN                                                              #
    # --------------------------------------------------------------------- #
    # A10: Implement stop method to cancel scheduled tasks
    async def stop(self) -> None:
        """
        Stops the MarketMonitor, cancels background tasks, and clears caches.
        """
        log_extra = {"component": "MarketMonitor"} # A16
        # A2: Parameterized logging
        logger.info("Stopping MarketMonitor...", extra=log_extra)
        self._is_running = False # Signal loops to stop

        # Cancel all stored tasks
        cancelled_tasks = []
        if self._scheduled_tasks:
            logger.debug("Cancelling %d scheduled tasks...", len(self._scheduled_tasks), extra=log_extra)
            for task in list(self._scheduled_tasks): # Iterate over a copy
                task.cancel()
                cancelled_tasks.append(task)

        # Wait for tasks to finish cancellation
        if cancelled_tasks:
            await asyncio.gather(*cancelled_tasks, return_exceptions=True)
            logger.debug("Scheduled tasks cancellation complete.", extra=log_extra)
        self._scheduled_tasks.clear()

        # Clear caches
        self.market_data_cache.clear()
        logger.debug("Market data cache cleared.", extra=log_extra)

        # A2: Parameterized logging
        logger.info("MarketMonitor stopped.")