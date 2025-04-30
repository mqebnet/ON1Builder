import os
import json
import dotenv
import aiofiles
import aiohttp
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set
import asyncio
from decimal import Decimal, InvalidOperation

from eth_utils import is_checksum_address, to_checksum_address

# Import helpers from sibling modules
from .config_paths import ConfigPathResolver
from .config_limits import ConfigLimitLoader

# Assume logging setup is available
from loggingconfig import setup_logging
import logging

logger = setup_logging("ConfigCore", level=logging.DEBUG)

class Configuration:
    """
    Loads, validates, and provides access to configuration settings
    from environment variables and JSON files, utilizing helper modules
    for path resolution and limit loading.
    """
    # Define default values directly here or load from a defaults file/dict
    DEFAULT_ENV_PATH = ".env"
    DEFAULT_BASE_PATH = Path(__file__).parent.parent.parent # Adjust if structure differs (e.g., python/ is root)

    # --- Core Settings Defaults ---
    DEFAULT_LOG_LEVEL = "INFO"
    DEFAULT_WEB3_MAX_RETRIES = 3
    DEFAULT_WEB3_RETRY_DELAY = 5
    DEFAULT_ENABLE_MEMORY_PROFILING = True
    DEFAULT_RUNTIME_DIR = "./runtime" # For weights, logs, etc.
    DEFAULT_MODEL_DIR_SUFFIX = "models"
    DEFAULT_DATA_DIR_SUFFIX = "data"

    def __init__(self, env_path: Optional[str] = None) -> None:
        """
        Initializes the Configuration instance.

        Args:
            env_path: Optional path to the .env file.
        """
        self.env_path = Path(env_path or self.DEFAULT_ENV_PATH)
        self.BASE_PATH = self.DEFAULT_BASE_PATH # Base path of the project

        self._load_env() # Load environment variables first

        # Initialize helper modules
        self._path_resolver = ConfigPathResolver(self.BASE_PATH, self._get_env_str)
        self._limit_loader = ConfigLimitLoader(self._get_env_int, self._get_env_float)

        # Load configuration values (core, paths, limits)
        self._load_core_settings()
        self._path_resolver.load_paths() # Load paths using the resolver
        self._limit_loader.load_limits() # Load limits using the loader

        self._validate_critical_settings()
        logger.info("Configuration initialization complete. Base path: %s", self.BASE_PATH)

    def _load_env(self) -> None:
        """Loads environment variables from the specified .env file."""
        try:
            if self.env_path.exists():
                 dotenv.load_dotenv(dotenv_path=self.env_path, override=True)
                 logger.debug("Loaded environment variables from: %s", self.env_path)
            else:
                 logger.warning(".env file not found at %s. Relying on system environment variables.", self.env_path)
        except Exception as e:
            logger.error("Error loading .env file from %s: %s", self.env_path, e)
            # Continue, relying on system environment or defaults


    def _load_core_settings(self) -> None:
        """Loads core operational settings from the environment."""
        logger.debug("Loading core configuration settings...")
        # Network/Web3
        self.HTTP_ENDPOINT = self._get_env_str("HTTP_ENDPOINT", None, "HTTP Provider URL", required=True)
        self.WEBSOCKET_ENDPOINT = self._get_env_str("WEBSOCKET_ENDPOINT", None, "WebSocket Provider URL", required=False)
        self.IPC_ENDPOINT = self._get_env_str("IPC_ENDPOINT", None, "IPC Provider Path", required=False)
        self.WEB3_MAX_RETRIES = self._get_env_int("WEB3_MAX_RETRIES", self.DEFAULT_WEB3_MAX_RETRIES, "Web3 connection retries")
        self.WEB3_RETRY_DELAY = self._get_env_int("WEB3_RETRY_DELAY", self.DEFAULT_WEB3_RETRY_DELAY, "Web3 retry delay (s)")

        # Wallet
        self.WALLET_ADDRESS = self._get_env_str("WALLET_ADDRESS", None, "Bot's Wallet Address", required=True)
        self.WALLET_KEY = self._get_env_str("WALLET_KEY", None, "Bot's Wallet Private Key", required=True, sensitive=True)

        # API Keys (Marked as sensitive)
        self.ETHERSCAN_API_KEY = self._get_env_str("ETHERSCAN_API_KEY", None, "Etherscan API Key", required=False, sensitive=True)
        self.INFURA_PROJECT_ID = self._get_env_str("INFURA_PROJECT_ID", None, "Infura Project ID", required=False, sensitive=True) # Often needed
        self.COINGECKO_API_KEY = self._get_env_str("COINGECKO_API_KEY", None, "CoinGecko API Key (Pro)", required=False, sensitive=True)
        self.COINMARKETCAP_API_KEY = self._get_env_str("COINMARKETCAP_API_KEY", None, "CoinMarketCap API Key", required=False, sensitive=True)
        self.CRYPTOCOMPARE_API_KEY = self._get_env_str("CRYPTOCOMPARE_API_KEY", None, "CryptoCompare API Key", required=False, sensitive=True)
        self.ONEINCH_DEV_PORTAL_TOKEN = self._get_env_str("ONEINCH_DEV_PORTAL_TOKEN", None, "1inch Dev Portal Token", required=False, sensitive=True)
        self.GRAPH_API_KEY = self._get_env_str("GRAPH_API_KEY", None, "The Graph API Key", required=False, sensitive=True)

        # General Operation
        self.LOG_LEVEL = self._get_env_str("LOG_LEVEL", self.DEFAULT_LOG_LEVEL, "Logging level (DEBUG, INFO, WARNING, ERROR)")
        self.ENABLE_MEMORY_PROFILING = self._get_env_bool("ENABLE_MEMORY_PROFILING", self.DEFAULT_ENABLE_MEMORY_PROFILING, "Enable tracemalloc")
        self.COMPONENT_HEALTH_CHECK_INTERVAL = self._get_env_int("COMPONENT_HEALTH_CHECK_INTERVAL", 60, "Component health check interval (s)")

        # Runtime Directories (relative to BASE_PATH or absolute)
        runtime_base = self._get_env_str("RUNTIME_DIR", self.DEFAULT_RUNTIME_DIR, "Base directory for runtime files (weights, data)")
        self.RUNTIME_DIR = self._resolve_runtime_path(runtime_base)
        self.MODEL_DIR = self.RUNTIME_DIR / self._get_env_str("MODEL_DIR_SUFFIX", self.DEFAULT_MODEL_DIR_SUFFIX, "Subdirectory for models")
        self.DATA_DIR = self.RUNTIME_DIR / self._get_env_str("DATA_DIR_SUFFIX", self.DEFAULT_DATA_DIR_SUFFIX, "Subdirectory for data")

        # Ensure runtime directories exist
        self.RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)

        logger.debug("Core settings loaded.")
        # Note: Specific addresses (WETH, USDC, Routers) are now loaded via ConfigPathResolver


    def _validate_critical_settings(self) -> None:
        """Performs essential validation checks after loading."""
        logger.debug("Validating critical configuration settings...")
        if not any([self.HTTP_ENDPOINT, self.WEBSOCKET_ENDPOINT, self.IPC_ENDPOINT]):
             raise ValueError("No Web3 provider endpoint configured (HTTP_ENDPOINT, WEBSOCKET_ENDPOINT, or IPC_ENDPOINT required).")

        # Validate wallet address format
        try:
             self.WALLET_ADDRESS = self._validate_ethereum_address(self.WALLET_ADDRESS, "WALLET_ADDRESS")
        except ValueError as e:
             logger.critical("Invalid WALLET_ADDRESS: %s", e)
             raise

        # Validate private key format (basic check)
        if not (self.WALLET_KEY and self.WALLET_KEY.startswith("0x") and len(self.WALLET_KEY) == 66):
             if not (self.WALLET_KEY and not self.WALLET_KEY.startswith("0x") and len(self.WALLET_KEY) == 64):
                 logger.critical("Invalid WALLET_KEY format. Must be a 64-character hex string, optionally prefixed with '0x'.")
                 raise ValueError("Invalid WALLET_KEY format.")

        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.LOG_LEVEL.upper() not in valid_log_levels:
             logger.warning("Invalid LOG_LEVEL '%s'. Defaulting to INFO.", self.LOG_LEVEL)
             self.LOG_LEVEL = "INFO"

        logger.debug("Critical settings validation passed.")
        # Add more validations as needed (e.g., check API key formats)


    # --- Env Var Helpers (moved from original class) ---

    def _get_env_str(self, var_name: str, default: Optional[str], description: str, required: bool = False, sensitive: bool = False) -> Optional[str]:
        """Retrieves a string environment variable."""
        value = os.getenv(var_name, default)
        log_value = "********" if sensitive and value else value # Mask sensitive values in logs

        if value is None and required:
            logger.error("Missing required environment variable: %s (%s)", var_name, description)
            raise ValueError(f"Missing required environment variable: {var_name}")

        # Log retrieval (even if None, unless required and missing)
        if value is not None:
             logger.debug("Loaded %s: %s", var_name, log_value)
        elif default is not None:
             logger.debug("Using default for %s: %s", var_name, default)
        # No log if optional and not set

        return value

    def _get_env_int(self, var_name: str, default: Optional[int], description: str) -> Optional[int]:
        """Retrieves an integer environment variable."""
        value_str = os.getenv(var_name)
        if value_str is None or value_str == "":
            logger.debug("Using default for %s: %s", var_name, default)
            return default
        try:
            parsed_value = int(self._parse_numeric_string(value_str))
            logger.debug("Loaded %s: %d", var_name, parsed_value)
            return parsed_value
        except (ValueError, TypeError) as e:
            logger.error("Invalid integer format for %s: '%s'. Error: %s. Using default: %s", var_name, value_str, e, default)
            return default

    def _get_env_float(self, var_name: str, default: Optional[float], description: str) -> Optional[float]:
        """Retrieves a float environment variable."""
        value_str = os.getenv(var_name)
        if value_str is None or value_str == "":
            logger.debug("Using default for %s: %s", var_name, default)
            return default
        try:
            parsed_value = float(self._parse_numeric_string(value_str))
            logger.debug("Loaded %s: %f", var_name, parsed_value)
            return parsed_value
        except (ValueError, TypeError) as e:
            logger.error("Invalid float format for %s: '%s'. Error: %s. Using default: %s", var_name, value_str, e, default)
            return default

    def _get_env_bool(self, var_name: str, default: bool, description: str) -> bool:
        """Retrieves a boolean environment variable (treats 'true', '1', 'yes' as True)."""
        value_str = os.getenv(var_name)
        if value_str is None:
            logger.debug("Using default for %s: %s", var_name, default)
            return default
        parsed_value = value_str.lower() in ['true', '1', 't', 'y', 'yes']
        logger.debug("Loaded %s: %s (parsed from '%s')", var_name, parsed_value, value_str)
        return parsed_value

    def _parse_numeric_string(self, value: str) -> str:
        """Removes comments and underscores from numeric strings."""
        return value.split("#")[0].strip().replace("_", "")

    def _validate_ethereum_address(self, address: Optional[str], var_name: str) -> Optional[str]:
        """Validates and checksums an Ethereum address string."""
        if address is None:
            return None # Allow optional addresses
        if not isinstance(address, str):
            raise ValueError(f"{var_name} must be a string, got {type(address)}")
        address = address.strip()
        if not address.startswith("0x") or len(address) != 42:
            raise ValueError(f"Invalid {var_name}: '{address}' must be a 42-character hex string starting with '0x'.")
        try:
            # Return checksummed version
            return to_checksum_address(address)
        except ValueError as e:
            # Catches invalid hex characters or non-direct checksum errors
            raise ValueError(f"Invalid {var_name} address format '{address}': {e}") from e


    def _resolve_runtime_path(self, path_str: str) -> Path:
         """Resolves a path relative to BASE_PATH if not absolute."""
         path = Path(path_str)
         if path.is_absolute():
              return path
         else:
              return (self.BASE_PATH / path).resolve()


    # --- Public Accessors ---

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a configuration value by key from this instance,
        falling back to helpers if necessary (paths, limits).
        """
        # Check direct attributes first (core settings)
        if hasattr(self, key):
            return getattr(self, key)
        # Check path resolver
        if hasattr(self._path_resolver, key):
            return getattr(self._path_resolver, key)
        # Check limit loader
        if hasattr(self._limit_loader, key):
             return getattr(self._limit_loader, key)

        # Key not found
        logger.warning("Configuration key '%s' not found. Returning default: %s", key, default)
        return default

    def get_all_config_values(self) -> Dict[str, Any]:
        """Returns a dictionary of all loaded configuration values."""
        all_values = {}
        all_values.update(vars(self)) # Core settings
        all_values.update(vars(self._path_resolver)) # Paths
        all_values.update(vars(self._limit_loader)) # Limits

        # Clean internal attributes (starting with _)
        all_values = {k: v for k, v in all_values.items() if not k.startswith('_')}

        # Mask sensitive values like WALLET_KEY, API keys
        for key in list(all_values.keys()):
             if "KEY" in key.upper() or "SECRET" in key.upper() or "TOKEN" in key.upper() or "PASSWORD" in key.upper():
                  if isinstance(all_values[key], str) and all_values[key]:
                       all_values[key] = "********"

        return all_values

    # --- Async Loading Helpers (for JSON files) ---

    async def _load_json_safe(self, file_path: Path, description: str) -> Any:
        """Safely loads and parses JSON data from a file asynchronously."""
        if not file_path or not isinstance(file_path, Path):
             raise ValueError(f"Invalid file path provided for {description}.")
        if not await asyncio.to_thread(file_path.exists):
            raise FileNotFoundError(f"Missing {description} file: {file_path}")

        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()
            data = json.loads(content)
            logger.debug("Loaded %s from %s (%d entries/items)", description, file_path.name, len(data) if hasattr(data, '__len__') else 1)
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {description} file ({file_path.name}): {e}")
        except Exception as e:
            raise RuntimeError(f"Error reading {description} file {file_path.name}: {e}") from e

    # --- Specific Data Loaders (using _load_json_safe) ---

    async def get_active_tokens(self) -> List[str]:
        """
        Loads the list of actively monitored token symbols or addresses.
        Example: Reads from token_addresses.json keys.
        """
        try:
             # Use the path resolved by ConfigPathResolver
             address_map = await self._load_json_safe(self.TOKEN_ADDRESSES, "token addresses")
             if not isinstance(address_map, dict):
                  logger.error("Token addresses file format error: Expected a dictionary.")
                  return []
             # Return symbols (keys) or addresses (values) based on need
             active_list = [k for k in address_map.keys() if not k.startswith("_")] # Use symbols
             # active_list = [v for k, v in address_map.items() if not k.startswith("_")] # Use addresses
             logger.info("Loaded %d active tokens for monitoring.", len(active_list))
             return active_list
        except (FileNotFoundError, ValueError, RuntimeError) as e:
             logger.error("Failed to load active tokens: %s. Returning empty list.", e)
             return []