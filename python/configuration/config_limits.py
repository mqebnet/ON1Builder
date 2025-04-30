from typing import Callable, Optional, Dict, Any
from decimal import Decimal
import logging

logger = logging.getLogger("ConfigLimits") # Use child logger

class ConfigLimitLoader:
    """Handles loading and validation of numeric limits and thresholds."""

    # Define default values for limits
    DEFAULT_LIMITS = {
        # Gas related
        "MAX_GAS_PRICE_GWEI": 500,
        "DEFAULT_GAS_LIMIT": 1_000_000,
        "BASE_GAS_LIMIT": 21000, # For simple transfers
        "DEFAULT_CANCEL_GAS_PRICE_GWEI": 60,
        "ETH_TX_GAS_PRICE_MULTIPLIER": 1.2,

        # Slippage (% as float, e.g., 0.01 for 1%)
        "SLIPPAGE_DEFAULT": 0.005, # 0.5%
        "MIN_SLIPPAGE": 0.001, # 0.1%
        "MAX_SLIPPAGE": 0.05, # 5%
        "SLIPPAGE_HIGH_CONGESTION": 0.01, # 1%
        "SLIPPAGE_LOW_CONGESTION": 0.003, # 0.3%

        # Profitability & Balance
        "MIN_PROFIT_ETH": "0.0005", # In ETH (string for Decimal)
        "MIN_BALANCE_ETH": "0.01", # In ETH (string for Decimal)
        "MIN_PROFIT_MULTIPLIER": 1.5, # Multiplier vs gas cost

        # Strategy Thresholds
        "AGGRESSIVE_FRONT_RUN_MIN_VALUE_ETH": "0.1",
        "AGGRESSIVE_FRONT_RUN_RISK_SCORE_THRESHOLD": 0.7,
        "FRONT_RUN_OPPORTUNITY_SCORE_THRESHOLD": 75,
        "VOLATILITY_FRONT_RUN_SCORE_THRESHOLD": 75,
        "ADVANCED_FRONT_RUN_RISK_SCORE_THRESHOLD": 75,
        "PRICE_DIP_BACK_RUN_THRESHOLD": 0.995, # e.g., execute if predicted price < current * 0.995
        "FLASHLOAN_BACK_RUN_PROFIT_PERCENTAGE": 0.002, # 0.2% required profit
        "HIGH_VOLUME_BACK_RUN_DEFAULT_THRESHOLD_USD": 100000.0,
        "SANDWICH_ATTACK_GAS_PRICE_THRESHOLD_GWEI": 150,
        "PRICE_BOOST_SANDWICH_MOMENTUM_THRESHOLD": 0.01, # 1% momentum

        # Market Monitor
        "MODEL_RETRAINING_INTERVAL": 3600, # seconds
        "MIN_TRAINING_SAMPLES": 100,
        "MODEL_ACCURACY_THRESHOLD": 0.6, # Example R^2 threshold
        "VOLATILITY_THRESHOLD": 0.05, # 5% stddev/mean
        "LIQUIDITY_THRESHOLD": 50000.0, # Min USD volume

        # Mempool Monitor
        "MEMPOOL_MAX_RETRIES": 3,
        "MEMPOOL_RETRY_DELAY": 1, # seconds
        "MEMPOOL_BATCH_SIZE": 50,
        "MEMPOOL_MAX_PARALLEL_FETCH": 10,
        "MEMPOOL_MAX_PARALLEL_ANALYSIS": 10,
        "MEMPOOL_HASH_QUEUE_SIZE": 2000,
        "MEMPOOL_ANALYSIS_QUEUE_SIZE": 1000,
        "MEMPOOL_PROFIT_QUEUE_SIZE": 500,
        "MEMPOOL_USE_FILTER": True, # Boolean handled by _get_env_bool
        "MEMPOOL_POLL_INTERVAL": 1.0, # seconds


        # Nonce Core
        "NONCE_CACHE_TTL": 60, # seconds
        "NONCE_RETRY_DELAY": 0.5, # seconds
        "NONCE_MAX_RETRIES": 5,
        "NONCE_TRANSACTION_TIMEOUT": 180, # seconds

        # SafetyNet
        "SAFETYNET_CACHE_TTL": 120, # seconds
        "SAFETYNET_GAS_PRICE_TTL": 15, # seconds

        # API Config
        "API_PRICE_CACHE_TTL": 60,
        "API_VOLUME_CACHE_TTL": 300,
        "API_METADATA_CACHE_TTL": 3600,
        "API_HISTORICAL_CACHE_TTL": 1800,
        "PREDICTION_CACHE_TTL": 300, # Matches MarketMonitor

        # Other Timeouts
        "PROFITABLE_TX_PROCESS_TIMEOUT": 1.5, # Max time StrategyNet waits for queue

        # A Misc Threshold (from original config) - naming could be improved
        "HIGH_VALUE_THRESHOLD_WEI": 1_000_000_000_000_000_000, # 1 ETH in Wei
    }


    def __init__(self, get_env_int_func: Callable, get_env_float_func: Callable):
        """
        Initializes the loader.

        Args:
            get_env_int_func: Function to retrieve integer env vars.
            get_env_float_func: Function to retrieve float env vars.
        """
        self._get_env_int = get_env_int_func
        self._get_env_float = get_env_float_func
        logger.debug("ConfigLimitLoader initialized.")

    def load_limits(self) -> None:
        """Loads all numeric limits using the provided environment functions."""
        logger.debug("Loading configuration limits and thresholds...")
        for key, default_value in self.DEFAULT_LIMITS.items():
            # Determine type and use appropriate getter
            value: Any
            if isinstance(default_value, bool):
                 # Booleans handled in ConfigCore, skip here or add _get_env_bool if needed
                 logger.debug("Skipping boolean limit '%s' in ConfigLimitLoader.", key)
                 continue # Assume handled by ConfigCore._get_env_bool
            elif isinstance(default_value, int) or (isinstance(default_value, str) and default_value.isdigit()):
                 # Handle integers, including large ones like WEI threshold
                 value = self._get_env_int(key, int(default_value), f"{key} Limit/Threshold")
            elif isinstance(default_value, float) or isinstance(default_value, str):
                 # Handle floats and strings intended as Decimals (like MIN_PROFIT_ETH)
                 # Let the float loader handle parsing for now
                 value = self._get_env_float(key, float(default_value), f"{key} Limit/Threshold")
                 # Post-process specific keys needing Decimal? Or handle in usage.
                 # Example: Convert MIN_PROFIT_ETH to Decimal after loading as float
                 # if key == "MIN_PROFIT_ETH" and value is not None:
                 #    try: value = Decimal(str(value))
                 #    except InvalidOperation: logger.error(...)
            else:
                 logger.warning("Unsupported default type for limit '%s': %s. Skipping.", key, type(default_value))
                 continue

            # Set the loaded value as an attribute
            setattr(self, key, value)

        # Perform basic validation on loaded limits (e.g., ensure positive where required)
        self._validate_loaded_limits()
        logger.debug("Configuration limits and thresholds loaded.")

    def _validate_loaded_limits(self) -> None:
        """Performs basic validation checks on loaded numeric limits."""
        logger.debug("Validating loaded limits...")
        # Example Validations:
        if self.MAX_GAS_PRICE_GWEI <= 0:
             logger.warning("MAX_GAS_PRICE_GWEI (%s) should be positive.", self.MAX_GAS_PRICE_GWEI)
        if self.MIN_SLIPPAGE < 0 or self.MAX_SLIPPAGE < self.MIN_SLIPPAGE:
             logger.warning("Invalid slippage settings: MIN=%s, MAX=%s", self.MIN_SLIPPAGE, self.MAX_SLIPPAGE)
        if self.MIN_PROFIT_ETH < 0: # Check the float value before potential Decimal conversion
             logger.warning("MIN_PROFIT_ETH (%s) should not be negative.", self.MIN_PROFIT_ETH)

        # Add more checks as needed...
        logger.debug("Limit validation finished.")
