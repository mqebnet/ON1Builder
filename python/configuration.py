# LICENSE: MIT // github.com/John0n1/ON1Builder

import os
import json
import aiofiles
import dotenv
import aiohttp
from pathlib import Path
from typing import Any, Dict, List, Optional

from eth_utils import is_checksum_address
from loggingconfig import setup_logging
import logging

logger = setup_logging("Configuration", level=logging.INFO)


class Configuration:
    """
    Loads and validates configuration from environment variables and JSON files.
    Provides methods to access configuration data with error handling and validation.
    """

    def __init__(self, env_path: Optional[str] = None) -> None:
        """
        Initialize Configuration by loading environment variables and setting default values.
        """
        self.env_path = env_path if env_path else ".env"
        # BASE_PATH is set to the parent of the parent directory of this file.
        self.BASE_PATH = Path(__file__).parent.parent
        self._load_env()
        self._initialize_defaults()

    def _initialize_defaults(self) -> None:
        """
        Initialize configuration parameters with default values.
        These values are overridden by environment variables if set.
        """
        # --------------------- General Settings ---------------------
        self.MAX_GAS_PRICE = self._get_env_int("MAX_GAS_PRICE", 100_000_000_000, "Maximum Gas Price (Wei)")
        self.GAS_LIMIT = self._get_env_int("GAS_LIMIT", 1_000_000, "Default Gas Limit")
        self.MAX_SLIPPAGE = self._get_env_float("MAX_SLIPPAGE", 0.01, "Maximum Slippage Tolerance (Fraction)")
        self.MIN_PROFIT = self._get_env_float("MIN_PROFIT", 0.001, "Minimum Profit Threshold (ETH)")
        self.MIN_BALANCE = self._get_env_float("MIN_BALANCE", 0.000001, "Minimum Account Balance (ETH) Warning Threshold")
        self.MEMORY_CHECK_INTERVAL = self._get_env_int("MEMORY_CHECK_INTERVAL", 300, "Memory Check Interval (seconds)")
        self.COMPONENT_HEALTH_CHECK_INTERVAL = self._get_env_int("COMPONENT_HEALTH_CHECK_INTERVAL", 60, "Component Health Check Interval (seconds)")
        self.PROFITABLE_TX_PROCESS_TIMEOUT = self._get_env_float("PROFITABLE_TX_PROCESS_TIMEOUT", 1.0, "Profitable Transaction Processing Timeout (seconds)")

        # --------------------- Standard Addresses ---------------------
        self.WETH_ADDRESS = self._get_env_str("WETH_ADDRESS", None, "Wrapped ETH Address")
        self.USDC_ADDRESS = self._get_env_str("USDC_ADDRESS", None, "USDC Address")
        self.USDT_ADDRESS = self._get_env_str("USDT_ADDRESS", None, "USDT Address")

        # --------------------- API Keys and Endpoints ---------------------
        self.ETHERSCAN_API_KEY = self._get_env_str("ETHERSCAN_API_KEY", None, "Etherscan API Key")
        self.INFURA_PROJECT_ID = self._get_env_str("INFURA_PROJECT_ID", None, "Infura Project ID")
        self.INFURA_API_KEY = self._get_env_str("INFURA_API_KEY", None, "Infura API Key (Alternative)")
        self.COINGECKO_API_KEY = self._get_env_str("COINGECKO_API_KEY", None, "CoinGecko API Key")
        self.COINMARKETCAP_API_KEY = self._get_env_str("COINMARKETCAP_API_KEY", None, "CoinMarketCap API Key")
        self.CRYPTOCOMPARE_API_KEY = self._get_env_str("CRYPTOCOMPARE_API_KEY", None, "CryptoCompare API Key")
        self.HTTP_ENDPOINT = self._get_env_str("HTTP_ENDPOINT", None, "HTTP Provider Endpoint URL")
        self.WEBSOCKET_ENDPOINT = self._get_env_str("WEBSOCKET_ENDPOINT", None, "WebSocket Provider Endpoint URL (Optional)")
        self.IPC_ENDPOINT = self._get_env_str("IPC_ENDPOINT", None, "IPC Provider Endpoint Path (Optional)")

        # --------------------- Account Configuration ---------------------
        self.WALLET_ADDRESS = self._get_env_str("WALLET_ADDRESS", None, "Wallet Address (Public)")
        self.WALLET_KEY = self._get_env_str("WALLET_KEY", None, "Wallet Private Key")

        # --------------------- File Paths ---------------------
        self.ERC20_ABI = self._resolve_path("ERC20_ABI", "Path to ERC20 ABI JSON file")
        self.AAVE_FLASHLOAN_ABI = self._resolve_path("AAVE_FLASHLOAN_ABI", "Path to Aave Flashloan ABI JSON file")
        self.AAVE_POOL_ABI = self._resolve_path("AAVE_POOL_ABI", "Path to Aave Pool ABI JSON file")
        self.UNISWAP_ABI = self._resolve_path("UNISWAP_ABI", "Path to Uniswap Router ABI JSON file")
        self.SUSHISWAP_ABI = self._resolve_path("SUSHISWAP_ABI", "Path to Sushiswap Router ABI JSON file")
        self.ERC20_SIGNATURES = self._resolve_path("ERC20_SIGNATURES", "Path to ERC20 Signatures JSON file")
        self.TOKEN_ADDRESSES = self._resolve_path("TOKEN_ADDRESSES", "Path to Token Addresses JSON file")
        self.TOKEN_SYMBOLS = self._resolve_path("TOKEN_SYMBOLS", "Path to Token Symbols JSON file")
        self.GAS_PRICE_ORACLE_ABI = self._resolve_path("GAS_PRICE_ORACLE_ABI", "Path to Gas Price Oracle ABI JSON file")

        # --------------------- Router Addresses ---------------------
        self.UNISWAP_ADDRESS = self._get_env_str("UNISWAP_ADDRESS", None, "Uniswap Router Contract Address")
        self.SUSHISWAP_ADDRESS = self._get_env_str("SUSHISWAP_ADDRESS", None, "Sushiswap Router Contract Address")
        self.AAVE_POOL_ADDRESS = self._get_env_str("AAVE_POOL_ADDRESS", None, "Aave Lending Pool Contract Address")
        self.AAVE_FLASHLOAN_ADDRESS = self._get_env_str("AAVE_FLASHLOAN_ADDRESS", None, "Aave Flashloan Contract Address")
        self.GAS_PRICE_ORACLE_ADDRESS = self._get_env_str("GAS_PRICE_ORACLE_ADDRESS", None, "Gas Price Oracle Contract Address")

        # --------------------- Slippage and Gas Configuration ---------------------
        self.SLIPPAGE_DEFAULT = self._get_env_float("SLIPPAGE_DEFAULT", 0.1, "Default Slippage Tolerance (%)")
        self.MIN_SLIPPAGE = self._get_env_float("MIN_SLIPPAGE", 0.01, "Minimum Slippage Tolerance (%)")
        self.MAX_SLIPPAGE = self._get_env_float("MAX_SLIPPAGE", 0.5, "Maximum Slippage Tolerance (%)")
        self.SLIPPAGE_HIGH_CONGESTION = self._get_env_float("SLIPPAGE_HIGH_CONGESTION", 0.05, "Slippage for High Network Congestion (%)")
        self.SLIPPAGE_LOW_CONGESTION = self._get_env_float("SLIPPAGE_LOW_CONGESTION", 0.2, "Slippage for Low Network Congestion (%)")
        self.MAX_GAS_PRICE_GWEI = self._get_env_int("MAX_GAS_PRICE_GWEI", 500, "Maximum Gas Price (Gwei)")
        self.MIN_PROFIT_MULTIPLIER = self._get_env_float("MIN_PROFIT_MULTIPLIER", 2.0, "Minimum Profit Multiplier for Transactions")
        self.BASE_GAS_LIMIT = self._get_env_int("BASE_GAS_LIMIT", 21000, "Base Gas Limit for Simple Transactions")
        self.DEFAULT_CANCEL_GAS_PRICE_GWEI = self._get_env_int("DEFAULT_CANCEL_GAS_PRICE_GWEI", 60, "Default Gas Price for Cancellation Transactions (Gwei)")
        self.ETH_TX_GAS_PRICE_MULTIPLIER = self._get_env_float("ETH_TX_GAS_PRICE_MULTIPLIER", 1.2, "Gas Price Multiplier for ETH Transfer Transactions")

        # --------------------- ML Model Configuration ---------------------
        self.MODEL_RETRAINING_INTERVAL = self._get_env_int("MODEL_RETRAINING_INTERVAL", 3600, "Model Retraining Interval (seconds)")
        self.MIN_TRAINING_SAMPLES = self._get_env_int("MIN_TRAINING_SAMPLES", 100, "Minimum Training Samples for Model Training")
        self.MODEL_ACCURACY_THRESHOLD = self._get_env_float("MODEL_ACCURACY_THRESHOLD", 0.7, "Minimum Model Accuracy Threshold")
        self.PREDICTION_CACHE_TTL = self._get_env_int("PREDICTION_CACHE_TTL", 300, "Price Prediction Cache TTL (seconds)")
        self.LINEAR_REGRESSION_PATH = str(self.BASE_PATH / "linear_regression")
        self.MODEL_PATH = str(self.BASE_PATH / "linear_regression" / "price_model.joblib")
        self.TRAINING_DATA_PATH = str(self.BASE_PATH / "linear_regression" / "training_data.csv")

        # --------------------- Mempool Monitor Configuration ---------------------
        self.MEMPOOL_MAX_RETRIES = self._get_env_int("MEMPOOL_MAX_RETRIES", 3, "Maximum Retries for Mempool Transaction Fetching")
        self.MEMPOOL_RETRY_DELAY = self._get_env_int("MEMPOOL_RETRY_DELAY", 2, "Retry Delay for Mempool Transaction Fetching (seconds)")
        self.MEMPOOL_BATCH_SIZE = self._get_env_int("MEMPOOL_BATCH_SIZE", 10, "Batch Size for Processing Mempool Transactions")
        self.MEMPOOL_MAX_PARALLEL_TASKS = self._get_env_int("MEMPOOL_MAX_PARALLEL_TASKS", 5, "Maximum Parallel Tasks for Mempool Processing")

        # --------------------- Nonce Core Configuration ---------------------
        self.NONCE_CACHE_TTL = self._get_env_int("NONCE_CACHE_TTL", 60, "Nonce Cache TTL (seconds)")
        self.NONCE_RETRY_DELAY = self._get_env_int("NONCE_RETRY_DELAY", 1, "Retry Delay for Nonce Fetching (seconds)")
        self.NONCE_MAX_RETRIES = self._get_env_int("NONCE_MAX_RETRIES", 5, "Maximum Retries for Nonce Fetching")
        self.NONCE_TRANSACTION_TIMEOUT = self._get_env_int("NONCE_TRANSACTION_TIMEOUT", 120, "Transaction Confirmation Timeout (seconds)")

        # --------------------- Safety Net Configuration ---------------------
        self.SAFETYNET_CACHE_TTL = self._get_env_int("SAFETYNET_CACHE_TTL", 300, "Safety Net Cache TTL (seconds)")
        self.SAFETYNET_GAS_PRICE_TTL = self._get_env_int("SAFETYNET_GAS_PRICE_TTL", 30, "Safety Net Gas Price Cache TTL (seconds)")

        # --------------------- Strategy Net Configuration ---------------------
        self.AGGRESSIVE_FRONT_RUN_MIN_VALUE_ETH = self._get_env_float("AGGRESSIVE_FRONT_RUN_MIN_VALUE_ETH", 0.1, "Minimum ETH Value for Aggressive Front-Run Strategy")
        self.AGGRESSIVE_FRONT_RUN_RISK_SCORE_THRESHOLD = self._get_env_float("AGGRESSIVE_FRONT_RUN_RISK_SCORE_THRESHOLD", 0.7, "Risk Score Threshold for Aggressive Front-Run Strategy")
        self.FRONT_RUN_OPPORTUNITY_SCORE_THRESHOLD = self._get_env_int("FRONT_RUN_OPPORTUNITY_SCORE_THRESHOLD", 75, "Opportunity Score Threshold for Predictive Front-Run Strategy")
        self.VOLATILITY_FRONT_RUN_SCORE_THRESHOLD = self._get_env_int("VOLATILITY_FRONT_RUN_SCORE_THRESHOLD", 75, "Volatility Score Threshold for Volatility Front-Run Strategy")
        self.ADVANCED_FRONT_RUN_RISK_SCORE_THRESHOLD = self._get_env_int("ADVANCED_FRONT_RUN_RISK_SCORE_THRESHOLD", 75, "Risk Score Threshold for Advanced Front-Run Strategy")
        self.PRICE_DIP_BACK_RUN_THRESHOLD = self._get_env_float("PRICE_DIP_BACK_RUN_THRESHOLD", 0.99, "Price Dip Threshold for Price Dip Back-Run Strategy (Fraction of Current Price)")
        self.FLASHLOAN_BACK_RUN_PROFIT_PERCENTAGE = self._get_env_float("FLASHLOAN_BACK_RUN_PROFIT_PERCENTAGE", 0.02, "Profit Percentage for Flashloan Back-Run Strategy (%)")
        self.HIGH_VOLUME_BACK_RUN_DEFAULT_THRESHOLD_USD = self._get_env_float("HIGH_VOLUME_BACK_RUN_DEFAULT_THRESHOLD_USD", 100000, "Default Volume Threshold for High Volume Back-Run Strategy (USD)")
        self.SANDWICH_ATTACK_GAS_PRICE_THRESHOLD_GWEI = self._get_env_int("SANDWICH_ATTACK_GAS_PRICE_THRESHOLD_GWEI", 200, "Maximum Gas Price for Sandwich Attack Strategy (Gwei)")
        self.PRICE_BOOST_SANDWICH_MOMENTUM_THRESHOLD = self._get_env_float("PRICE_BOOST_SANDWICH_MOMENTUM_THRESHOLD", 0.02, "Price Momentum Threshold for Price Boost Sandwich Strategy (%)")

        # --------------------- Mempool High Value Transaction Monitoring ---------------------
        self.HIGH_VALUE_THRESHOLD = self._get_env_int("HIGH_VALUE_THRESHOLD", 1_000_000_000_000_000_000, "Value Threshold for High-Value Transaction Monitoring (Wei)")

    def _load_env(self) -> None:
        """Load environment variables from the .env file."""
        dotenv.load_dotenv(dotenv_path=self.env_path)
        logger.debug(f"Environment variables loaded from: {self.env_path}")

    def _validate_ethereum_address(self, address: str, var_name: str) -> str:
        """
        Validate Ethereum address format and convert it to checksum format.
        """
        if not isinstance(address, str):
            raise ValueError(f"{var_name} must be a string, got {type(address)}")
        address = address.strip()
        if not address.startswith('0x') or len(address) != 42:
            raise ValueError(f"Invalid {var_name}: Must be a 42-character hex string starting with '0x'.")
        try:
            if not is_checksum_address(address):
                from eth_utils import to_checksum_address
                address = to_checksum_address(address)
            return address
        except Exception as e:
            raise ValueError(f"Invalid {var_name} address format: {str(e)}")

    def _get_env_str(self, var_name: str, default: Optional[str], description: str) -> str:
        """
        Get an environment variable as a string.
        """
        value = os.getenv(var_name, default)
        if value is None:
            logger.error(f"Missing required environment variable: {var_name} ({description})")
            raise ValueError(f"Missing environment variable: {var_name} ({description})")
        if 'ADDRESS' in var_name and not any(x in var_name for x in ['ABI', 'PATH', 'ADDRESSES', 'SIGNATURES', 'SYMBOLS']) and value != default:
            try:
                return self._validate_ethereum_address(value, var_name)
            except ValueError as e:
                logger.error(f"Invalid Ethereum address format for {var_name} ({description}): {e}")
                raise
        logger.debug(f"Loaded {var_name} from environment: {value[:5]}... ({description})")
        return value

    def _parse_numeric_string(self, value: str) -> str:
        """Clean numeric strings by removing underscores and comments."""
        value = value.split('#')[0].strip()
        return value.replace('_', '')

    def _get_env_int(self, var_name: str, default: Optional[int], description: str) -> int:
        """
        Get an environment variable as an integer.
        """
        try:
            value = os.getenv(var_name)
            if value is None:
                if default is None:
                    logger.error(f"Missing required integer environment variable: {var_name} ({description})")
                    raise ValueError(f"Missing environment variable: {var_name} ({description})")
                logger.warning(f"Environment variable {var_name} ({description}) not set, using default: {default}")
                return default
            cleaned_value = self._parse_numeric_string(value)
            int_value = int(cleaned_value)
            logger.debug(f"Loaded {var_name} from environment: {int_value} ({description})")
            return int_value
        except ValueError as e:
            logger.error(f"Invalid or missing integer environment variable: {var_name} ({description}) - {e}")
            raise

    def _get_env_float(self, var_name: str, default: Optional[float], description: str) -> float:
        """
        Get an environment variable as a float.
        """
        try:
            value = os.getenv(var_name)
            if value is None:
                if default is None:
                    logger.error(f"Missing required float environment variable: {var_name} ({description})")
                    raise ValueError(f"Missing environment variable: {var_name} ({description})")
                logger.warning(f"Environment variable {var_name} ({description}) not set, using default: {default}")
                return default
            cleaned_value = self._parse_numeric_string(value)
            float_value = float(cleaned_value)
            logger.debug(f"Loaded {var_name} from environment: {float_value} ({description})")
            return float_value
        except ValueError as e:
            logger.error(f"Invalid or missing float environment variable: {var_name} ({description}) - {e}")
            raise

    def _resolve_path(self, path_env_var: str, description: str) -> Path:
        """
        Resolve a file path from an environment variable and ensure the file exists.
        """
        path_str = self._get_env_str(path_env_var, None, description)
        full_path = self.BASE_PATH / path_str
        if not full_path.exists():
            logger.error(f"File not found: {full_path} ({description})")
            raise FileNotFoundError(f"File not found: {full_path} ({description})")
        logger.debug(f"Resolved path: {full_path} ({description})")
        return full_path

    async def _load_json_safe(self, file_path: Path, description: str) -> Any:
        """
        Load JSON data from a file with proper error handling.
        """
        try:
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
                try:
                    data = json.loads(content)
                    logger.debug(f"Loaded {description} from {file_path}")
                    return data
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in {description} file: {e} in {file_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing {description} file: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading {description}: {e} from {file_path}")

    async def get_token_addresses(self) -> List[str]:
        """
        Get the list of monitored token addresses from the token addresses JSON file.
        """
        data = await self._load_json_safe(self.TOKEN_ADDRESSES, "monitored tokens")
        if not isinstance(data, dict):
            logger.error("Invalid format for token addresses: must be a dictionary")
            raise ValueError("Invalid format for token addresses: must be a dictionary")
        addresses = list(data.keys())
        logger.info(f"Loaded {len(addresses)} token addresses")
        return addresses

    async def get_token_symbols(self) -> Dict[str, str]:
        """
        Get the mapping of token addresses to symbols from the token symbols JSON file.
        """
        data = await self._load_json_safe(self.TOKEN_SYMBOLS, "token symbols")
        if not isinstance(data, dict):
            logger.error("Invalid format for token symbols: must be a dictionary")
            raise ValueError("Invalid format for token symbols: must be a dictionary")
        return data

    async def get_erc20_signatures(self) -> Dict[str, str]:
        """
        Load ERC20 function signatures from a JSON file.
        """
        data = await self._load_json_safe(self.ERC20_SIGNATURES, "ERC20 function signatures")
        if not isinstance(data, dict):
            logger.error("Invalid format for ERC20 signatures: must be a dictionary")
            raise ValueError("Invalid format for ERC20 signatures: must be a dictionary")
        return data

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a configuration value safely.
        """
        try:
            return getattr(self, key, default)
        except AttributeError:
            logger.warning(f"Configuration key '{key}' not found, using default: {default}")
            return default

    def get_all_config_values(self) -> Dict[str, Any]:
        """
        Return all configuration values as a dictionary.
        """
        return vars(self)

    async def load(self, web3=None) -> None:
        """
        Load and validate all configuration data including directories, critical ABI paths,
        API key validation, and Ethereum address checks.
        """
        try:
            self._create_required_directories()
            await self._load_critical_abis()
            await self._validate_api_keys()
            self._validate_addresses()
            logger.debug("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Configuration load failed: {str(e)}")
            raise

    def _create_required_directories(self) -> None:
        """Create necessary directories if they do not exist."""
        required_dirs = [
            self.LINEAR_REGRESSION_PATH,
            self.BASE_PATH / "abi",
            self.BASE_PATH / "utils"
        ]
        for directory in required_dirs:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")

    async def _load_critical_abis(self) -> None:
        """Load and validate critical ABI file paths."""
        try:
            self.AAVE_FLASHLOAN_ABI_PATH = self._resolve_path("AAVE_FLASHLOAN_ABI", "Path to AAVE Flashloan ABI file")
            self.AAVE_POOL_ABI_PATH = self._resolve_path("AAVE_POOL_ABI", "Path to AAVE Pool ABI file")
            self.ERC20_ABI_PATH = self._resolve_path("ERC20_ABI", "Path to ERC20 ABI file")
            self.UNISWAP_ABI_PATH = self._resolve_path("UNISWAP_ABI", "Path to Uniswap ABI file")
            self.SUSHISWAP_ABI_PATH = self._resolve_path("SUSHISWAP_ABI", "Path to Sushiswap ABI file")
            self.GAS_PRICE_ORACLE_ABI_PATH = self._resolve_path("GAS_PRICE_ORACLE_ABI", "Path to Gas Price Oracle ABI file")
        except Exception as e:
            raise RuntimeError(f"Failed to load critical ABI paths: {e}")

    async def _validate_api_keys(self) -> None:
        """
        Validate that required API keys are set and functioning by making test requests.
        """
        required_keys = [
            ('ETHERSCAN_API_KEY', "https://api.etherscan.io/api?module=stats&action=ethprice&apikey={key}"),
            ('INFURA_API_KEY', None),
            ('COINGECKO_API_KEY', None),
            ('COINMARKETCAP_API_KEY', "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest?CMC_PRO_API_KEY={key}&limit=1"),
            ('CRYPTOCOMPARE_API_KEY', "https://min-api.cryptocompare.com/data/price?fsym=BTC&tsyms=USD,JPY,EUR&api_key={key}")
        ]
        missing_keys = []
        invalid_keys = []

        async with aiohttp.ClientSession() as session:
            for key_name, test_url_template in required_keys:
                api_key = getattr(self, key_name, None)
                if not api_key:
                    missing_keys.append(key_name)
                    continue

                if key_name == "INFURA_API_KEY":
                    test_url = f"https://mainnet.infura.io/v3/{api_key}"
                    payload = {"jsonrpc": "2.0", "id": 1, "method": "web3_clientVersion", "params": []}
                    try:
                        async with session.post(test_url, json=payload, timeout=10) as response:
                            data = await response.json()
                            if "result" not in data:
                                invalid_keys.append(f"{key_name} (Invalid response)")
                            else:
                                logger.debug(f"API key {key_name} validated successfully.")
                    except Exception as e:
                        invalid_keys.append(f"{key_name} (Error: {e})")
                    continue

                if key_name == "COINGECKO_API_KEY":
                    test_url = "https://api.coingecko.com/api/v3/ping"
                    try:
                        async with session.get(test_url, timeout=10) as response:
                            if response.status != 200:
                                invalid_keys.append(f"{key_name} (Status: {response.status})")
                            else:
                                logger.debug(f"API key {key_name} validated successfully (free endpoint).")
                    except Exception as e:
                        invalid_keys.append(f"{key_name} (Error: {e})")
                    continue

                if key_name == "CRYPTOCOMPARE_API_KEY":
                    test_url = test_url_template.format(key=api_key)
                    try:
                        async with session.get(test_url, timeout=10) as response:
                            if response.status != 200:
                                invalid_keys.append(f"{key_name} (Status: {response.status})")
                            else:
                                logger.debug(f"API key {key_name} validated successfully.")
                    except Exception as e:
                        invalid_keys.append(f"{key_name} (Error: {e})")
                    continue

                test_url = test_url_template.format(key=api_key)
                try:
                    async with session.get(test_url, timeout=10) as response:
                        if response.status != 200:
                            invalid_keys.append(f"{key_name} (Status: {response.status})")
                        else:
                            logger.debug(f"API key {key_name} validated successfully.")
                except Exception as e:
                    invalid_keys.append(f"{key_name} (Error: {e})")

        if missing_keys:
            logger.warning(f"Missing API keys: {', '.join(missing_keys)}")
        if invalid_keys:
            logger.error(f"Invalid API keys: {', '.join(invalid_keys)}")
            raise ValueError(f"Invalid API keys detected: {', '.join(invalid_keys)}")

    def _validate_addresses(self) -> None:
        """
        Validate all Ethereum addresses in configuration.
        """
        address_fields = [
            'WALLET_ADDRESS',
            'UNISWAP_ADDRESS',
            'SUSHISWAP_ADDRESS',
            'AAVE_POOL_ADDRESS',
            'AAVE_FLASHLOAN_ADDRESS',
            'GAS_PRICE_ORACLE_ADDRESS'
        ]
        for field in address_fields:
            value = getattr(self, field, None)
            if value:
                try:
                    setattr(self, field, self._validate_ethereum_address(value, field))
                except ValueError as e:
                    logger.error(f"Invalid address for {field}: {e}")
                    raise

    async def load(self, web3=None) -> None:
        """
        Load and validate all configuration data including directories, critical ABI paths,
        API key validation, and Ethereum address checks.
        """
        try:
            self._create_required_directories()
            await self._load_critical_abis()
            await self._validate_api_keys()
            self._validate_addresses()
            logger.debug("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Configuration load failed: {str(e)}")
            raise

    def _create_required_directories(self) -> None:
        """Create necessary directories if they do not exist."""
        required_dirs = [
            self.LINEAR_REGRESSION_PATH,
            self.BASE_PATH / "abi",
            self.BASE_PATH / "utils"
        ]
        for directory in required_dirs:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")

    async def _load_critical_abis(self) -> None:
        """Load and validate critical ABI file paths."""
        try:
            self.AAVE_FLASHLOAN_ABI_PATH = self._resolve_path("AAVE_FLASHLOAN_ABI", "Path to AAVE Flashloan ABI file")
            self.AAVE_POOL_ABI_PATH = self._resolve_path("AAVE_POOL_ABI", "Path to AAVE Pool ABI file")
            self.ERC20_ABI_PATH = self._resolve_path("ERC20_ABI", "Path to ERC20 ABI file")
            self.UNISWAP_ABI_PATH = self._resolve_path("UNISWAP_ABI", "Path to Uniswap ABI file")
            self.SUSHISWAP_ABI_PATH = self._resolve_path("SUSHISWAP_ABI", "Path to Sushiswap ABI file")
            self.GAS_PRICE_ORACLE_ABI_PATH = self._resolve_path("GAS_PRICE_ORACLE_ABI", "Path to Gas Price Oracle ABI file")
        except Exception as e:
            raise RuntimeError(f"Failed to load critical ABI paths: {e}")

    async def get_token_addresses(self) -> List[str]:
        """
        Get the list of monitored token addresses from the token addresses JSON file.
        """
        data = await self._load_json_safe(self.TOKEN_ADDRESSES, "monitored tokens")
        if not isinstance(data, dict):
            logger.error("Invalid format for token addresses: must be a dictionary")
            raise ValueError("Invalid format for token addresses: must be a dictionary")
        addresses = list(data.keys())
        logger.info(f"Loaded {len(addresses)} token addresses")
        return addresses

    async def get_token_symbols(self) -> Dict[str, str]:
        """
        Get the mapping of token addresses to symbols from the token symbols JSON file.
        """
        data = await self._load_json_safe(self.TOKEN_SYMBOLS, "token symbols")
        if not isinstance(data, dict):
            logger.error("Invalid format for token symbols: must be a dictionary")
            raise ValueError("Invalid format for token symbols: must be a dictionary")
        return data

    async def get_erc20_signatures(self) -> Dict[str, str]:
        """
        Load ERC20 function signatures from a JSON file.
        """
        data = await self._load_json_safe(self.ERC20_SIGNATURES, "ERC20 function signatures")
        if not isinstance(data, dict):
            logger.error("Invalid format for ERC20 signatures: must be a dictionary")
            raise ValueError("Invalid format for ERC20 signatures: must be a dictionary")
        return data

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a configuration value safely.
        """
        try:
            return getattr(self, key, default)
        except AttributeError:
            logger.warning(f"Configuration key '{key}' not found, using default: {default}")
            return default

    def get_all_config_values(self) -> Dict[str, Any]:
        """
        Return all configuration values as a dictionary.
        """
        return vars(self)

# --- End file: configuration.py ---
