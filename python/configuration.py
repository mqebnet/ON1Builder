import os
import json
import dotenv
import aiofiles
import aiohttp
from pathlib import Path
from typing import Any, Dict, List, Optional
import asyncio

from eth_utils import is_checksum_address, to_checksum_address
from loggingconfig import setup_logging
import logging

logger = setup_logging("Configuration", level=logging.DEBUG)


class Configuration:
    """
    Loads and validates configuration from environment variables and JSON files.
    Provides methods to access configuration data with error handling and validation.
    """

    def __init__(self, env_path: Optional[str] = None) -> None:
        """
        Initialize the Configuration by loading environment variables and setting defaults.

        Args:
            env_path (Optional[str]): Path to the .env file. Defaults to ".env".
        """
        self.env_path: str = env_path if env_path else ".env"
        # Set BASE_PATH to the root directory of the project.
        self.BASE_PATH: Path = Path(__file__).parent.parent

        # Load environment variables and initialize default settings.
        self._load_env()
        self._initialize_defaults()

    def _load_env(self) -> None:
        """
        Load environment variables from the specified .env file.
        """
        dotenv.load_dotenv(dotenv_path=self.env_path)
        logger.debug(f"Environment variables loaded from: {self.env_path}")

    def _initialize_defaults(self) -> None:
        """
        Initialize all configuration parameters. Environment values override these defaults.
        """
        # --------------------- General Settings ---------------------
        self.MAX_GAS_PRICE: int = self._get_env_int("MAX_GAS_PRICE", 100_000_000_000, "Maximum gas price in Wei")
        self.GAS_LIMIT: int = self._get_env_int("GAS_LIMIT", 1_000_000, "Default gas limit")
        self.MAX_SLIPPAGE: float = self._get_env_float("MAX_SLIPPAGE", 0.01, "Maximum slippage (fraction)")
        self.MIN_PROFIT: float = self._get_env_float("MIN_PROFIT", 0.001, "Minimum profit threshold in ETH")
        self.MIN_BALANCE: float = self._get_env_float("MIN_BALANCE", 0.000001, "Minimum balance threshold in ETH")
        self.MEMORY_CHECK_INTERVAL: int = self._get_env_int("MEMORY_CHECK_INTERVAL", 300, "Memory check interval (seconds)")
        self.COMPONENT_HEALTH_CHECK_INTERVAL: int = self._get_env_int("COMPONENT_HEALTH_CHECK_INTERVAL", 60, "Component health check interval (seconds)")
        self.PROFITABLE_TX_PROCESS_TIMEOUT: float = self._get_env_float("PROFITABLE_TX_PROCESS_TIMEOUT", 1.0, "Profitable transaction processing timeout (seconds)")

        # --------------------- Standard Addresses ---------------------
        # Use known mainnet addresses where possible.
        self.WETH_ADDRESS: str = self._get_env_str(
            "WETH_ADDRESS",
            "0xC02aaa39b223FE8D0a0e5C4F27eAD9083C756Cc2",
            "WETH Address"
        )
        self.USDC_ADDRESS: str = self._get_env_str(
            "USDC_ADDRESS",
            "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
            "USDC Address"
        )
        self.USDT_ADDRESS: str = self._get_env_str(
            "USDT_ADDRESS",
            "0xdAC17F958D2ee523a2206206994597C13D831ec7",
            "USDT Address"
        )

        # --------------------- API Keys and Endpoints ---------------------
        self.ETHERSCAN_API_KEY: str = self._get_env_str(
            "ETHERSCAN_API_KEY", "", "Etherscan API Key", required=True
        )
        self.INFURA_PROJECT_ID: str = self._get_env_str(
            "INFURA_PROJECT_ID", "", "Infura Project ID", required=True
        )
        self.INFURA_API_KEY: str = self._get_env_str(
            "INFURA_API_KEY", "", "Infura API Key", required=True
        )
        self.COINGECKO_API_KEY: str = self._get_env_str(
            "COINGECKO_API_KEY", "", "CoinGecko API Key", required=True
        )
        self.COINMARKETCAP_API_KEY: str = self._get_env_str(
            "COINMARKETCAP_API_KEY", "", "CoinMarketCap API Key", required=True
        )
        self.CRYPTOCOMPARE_API_KEY: str = self._get_env_str(
            "CRYPTOCOMPARE_API_KEY", "", "CryptoCompare API Key", required=True
        )
        self.HTTP_ENDPOINT: str = self._get_env_str(
            "HTTP_ENDPOINT", "http://localhost:8545", "HTTP Provider Endpoint URL", required=True
        )
        self.WEBSOCKET_ENDPOINT: str = self._get_env_str(
            "WEBSOCKET_ENDPOINT", "ws://localhost:8546", "WebSocket Provider Endpoint URL", required=False
        )
        self.IPC_ENDPOINT: str = self._get_env_str(
            "IPC_ENDPOINT", "", "IPC Provider Endpoint Path", required=False
        )

        # --------------------- Account Configuration ---------------------
        self.WALLET_ADDRESS: str = self._get_env_str(
            "WALLET_ADDRESS", "", "Wallet Address (Public)", required=True
        )
        self.WALLET_KEY: str = self._get_env_str(
            "WALLET_KEY", "", "Wallet Private Key", required=True
        )

        # --------------------- File Paths ---------------------
        self.ERC20_ABI: Path = self._resolve_path("ERC20_ABI", "Path to ERC20 ABI JSON file")
        self.AAVE_FLASHLOAN_ABI: Path = self._resolve_path("AAVE_FLASHLOAN_ABI", "Path to Aave Flashloan ABI JSON file")
        self.AAVE_POOL_ABI: Path = self._resolve_path("AAVE_POOL_ABI", "Path to Aave Pool ABI JSON file")
        self.UNISWAP_ABI: Path = self._resolve_path("UNISWAP_ABI", "Path to Uniswap Router ABI JSON file")
        self.SUSHISWAP_ABI: Path = self._resolve_path("SUSHISWAP_ABI", "Path to Sushiswap Router ABI JSON file")
        self.ERC20_SIGNATURES: Path = self._resolve_path("ERC20_SIGNATURES", "Path to ERC20 Signatures JSON file")
        self.TOKEN_ADDRESSES: Path = self._resolve_path("TOKEN_ADDRESSES", "Path to Token Addresses JSON file")
        self.TOKEN_SYMBOLS: Path = self._resolve_path("TOKEN_SYMBOLS", "Path to Token Symbols JSON file")
        self.GAS_PRICE_ORACLE_ABI: Path = self._resolve_path("GAS_PRICE_ORACLE_ABI", "Path to Gas Price Oracle ABI JSON file")

        # --------------------- Router Addresses ---------------------
        self.UNISWAP_ADDRESS: str = self._get_env_str(
            "UNISWAP_ADDRESS",
            "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
            "Uniswap Router Address"
        )
        self.SUSHISWAP_ADDRESS: str = self._get_env_str(
            "SUSHISWAP_ADDRESS",
            "0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F",
            "Sushiswap Router Address"
        )
        # Use a known Aave Pool Address on Mainnet; for flashloan and gas oracle addresses, require environment.
        self.AAVE_POOL_ADDRESS: str = self._get_env_str(
            "AAVE_POOL_ADDRESS",
            "0xb53c1a33016b2dc2ff3653530bff1848a515c8c5",
            "Aave Lending Pool Address",
            required=True
        )
        self.AAVE_FLASHLOAN_ADDRESS: str = self._get_env_str(
            "AAVE_FLASHLOAN_ADDRESS", "", "Aave Flashloan Contract Address", required=True
        )
        self.GAS_PRICE_ORACLE_ADDRESS: str = self._get_env_str(
            "GAS_PRICE_ORACLE_ADDRESS", "", "Gas Price Oracle Contract Address", required=True
        )

        # --------------------- Slippage and Gas Configuration ---------------------
        self.SLIPPAGE_DEFAULT: float = self._get_env_float("SLIPPAGE_DEFAULT", 0.1, "Default slippage tolerance")
        self.MIN_SLIPPAGE: float = self._get_env_float("MIN_SLIPPAGE", 0.01, "Minimum slippage tolerance")
        self.MAX_SLIPPAGE: float = self._get_env_float("MAX_SLIPPAGE", 0.5, "Maximum slippage tolerance")
        self.SLIPPAGE_HIGH_CONGESTION: float = self._get_env_float("SLIPPAGE_HIGH_CONGESTION", 0.05, "Slippage during high network congestion")
        self.SLIPPAGE_LOW_CONGESTION: float = self._get_env_float("SLIPPAGE_LOW_CONGESTION", 0.2, "Slippage during low network congestion")
        self.MAX_GAS_PRICE_GWEI: int = self._get_env_int("MAX_GAS_PRICE_GWEI", 500, "Maximum gas price (Gwei)")
        self.MIN_PROFIT_MULTIPLIER: float = self._get_env_float("MIN_PROFIT_MULTIPLIER", 2.0, "Minimum profit multiplier")
        self.BASE_GAS_LIMIT: int = self._get_env_int("BASE_GAS_LIMIT", 21000, "Base gas limit for simple transactions")
        self.DEFAULT_CANCEL_GAS_PRICE_GWEI: int = self._get_env_int("DEFAULT_CANCEL_GAS_PRICE_GWEI", 60, "Default cancellation gas price (Gwei)")
        self.ETH_TX_GAS_PRICE_MULTIPLIER: float = self._get_env_float("ETH_TX_GAS_PRICE_MULTIPLIER", 1.2, "ETH transfer gas price multiplier")

        # --------------------- ML Model Configuration ---------------------
        self.MODEL_RETRAINING_INTERVAL: int = self._get_env_int("MODEL_RETRAINING_INTERVAL", 3600, "Model retraining interval (seconds)")
        self.MIN_TRAINING_SAMPLES: int = self._get_env_int("MIN_TRAINING_SAMPLES", 100, "Minimum training samples required")
        self.MODEL_ACCURACY_THRESHOLD: float = self._get_env_float("MODEL_ACCURACY_THRESHOLD", 0.7, "Minimum model accuracy threshold")
        self.PREDICTION_CACHE_TTL: int = self._get_env_int("PREDICTION_CACHE_TTL", 300, "Prediction cache TTL (seconds)")
        self.LINEAR_REGRESSION_PATH: str = str(self.BASE_PATH / "linear_regression")
        self.MODEL_PATH: str = str(self.BASE_PATH / "linear_regression" / "price_model.joblib")
        self.TRAINING_DATA_PATH: str = str(self.BASE_PATH / "linear_regression" / "training_data.csv")

        # --------------------- Mempool Monitor Configuration ---------------------
        self.MEMPOOL_MAX_RETRIES: int = self._get_env_int("MEMPOOL_MAX_RETRIES", 3, "Mempool maximum retry count")
        self.MEMPOOL_RETRY_DELAY: int = self._get_env_int("MEMPOOL_RETRY_DELAY", 2, "Mempool retry delay (seconds)")
        self.MEMPOOL_BATCH_SIZE: int = self._get_env_int("MEMPOOL_BATCH_SIZE", 10, "Mempool batch size")
        self.MEMPOOL_MAX_PARALLEL_TASKS: int = self._get_env_int("MEMPOOL_MAX_PARALLEL_TASKS", 5, "Mempool max parallel tasks")

        # --------------------- Nonce Core Configuration ---------------------
        self.NONCE_CACHE_TTL: int = self._get_env_int("NONCE_CACHE_TTL", 60, "Nonce cache TTL (seconds)")
        self.NONCE_RETRY_DELAY: int = self._get_env_int("NONCE_RETRY_DELAY", 1, "Nonce retry delay (seconds)")
        self.NONCE_MAX_RETRIES: int = self._get_env_int("NONCE_MAX_RETRIES", 5, "Nonce maximum retry count")
        self.NONCE_TRANSACTION_TIMEOUT: int = self._get_env_int("NONCE_TRANSACTION_TIMEOUT", 120, "Transaction timeout for nonce management (seconds)")

        # --------------------- Safety Net Configuration ---------------------
        self.SAFETYNET_CACHE_TTL: int = self._get_env_int("SAFETYNET_CACHE_TTL", 300, "SafetyNet cache TTL (seconds)")
        self.SAFETYNET_GAS_PRICE_TTL: int = self._get_env_int("SAFETYNET_GAS_PRICE_TTL", 30, "SafetyNet gas price cache TTL (seconds)")

        # --------------------- Strategy Net Configuration ---------------------
        self.AGGRESSIVE_FRONT_RUN_MIN_VALUE_ETH: float = self._get_env_float("AGGRESSIVE_FRONT_RUN_MIN_VALUE_ETH", 0.1, "Minimum ETH value for aggressive front-run")
        self.AGGRESSIVE_FRONT_RUN_RISK_SCORE_THRESHOLD: float = self._get_env_float("AGGRESSIVE_FRONT_RUN_RISK_SCORE_THRESHOLD", 0.7, "Risk score threshold for aggressive front-run")
        self.FRONT_RUN_OPPORTUNITY_SCORE_THRESHOLD: int = self._get_env_int("FRONT_RUN_OPPORTUNITY_SCORE_THRESHOLD", 75, "Opportunity score threshold for front-run")
        self.VOLATILITY_FRONT_RUN_SCORE_THRESHOLD: int = self._get_env_int("VOLATILITY_FRONT_RUN_SCORE_THRESHOLD", 75, "Volatility score threshold for front-run")
        self.ADVANCED_FRONT_RUN_RISK_SCORE_THRESHOLD: int = self._get_env_int("ADVANCED_FRONT_RUN_RISK_SCORE_THRESHOLD", 75, "Risk score threshold for advanced front-run")
        self.PRICE_DIP_BACK_RUN_THRESHOLD: float = self._get_env_float("PRICE_DIP_BACK_RUN_THRESHOLD", 0.99, "Price dip threshold for back-run strategy")
        self.FLASHLOAN_BACK_RUN_PROFIT_PERCENTAGE: float = self._get_env_float("FLASHLOAN_BACK_RUN_PROFIT_PERCENTAGE", 0.02, "Flashloan back-run profit percentage")
        self.HIGH_VOLUME_BACK_RUN_DEFAULT_THRESHOLD_USD: float = self._get_env_float("HIGH_VOLUME_BACK_RUN_DEFAULT_THRESHOLD_USD", 100000, "High volume back-run threshold in USD")
        self.SANDWICH_ATTACK_GAS_PRICE_THRESHOLD_GWEI: int = self._get_env_int("SANDWICH_ATTACK_GAS_PRICE_THRESHOLD_GWEI", 200, "Gas price threshold for sandwich attack (Gwei)")
        self.PRICE_BOOST_SANDWICH_MOMENTUM_THRESHOLD: float = self._get_env_float("PRICE_BOOST_SANDWICH_MOMENTUM_THRESHOLD", 0.02, "Price boost sandwich momentum threshold")

        # --------------------- Mempool High Value Transaction Monitoring ---------------------
        self.HIGH_VALUE_THRESHOLD: int = self._get_env_int("HIGH_VALUE_THRESHOLD", 1_000_000_000_000_000_000, "High value transaction threshold in Wei")

    def _validate_ethereum_address(self, address: str, var_name: str) -> str:
        """
        Validate the Ethereum address format and convert it to checksum format.

        Args:
            address (str): The address to validate.
            var_name (str): The name of the environment variable.

        Returns:
            str: The validated checksum address.

        Raises:
            ValueError: If the address format is invalid.
        """
        if not isinstance(address, str):
            raise ValueError(f"{var_name} must be a string, got {type(address)}")
        address = address.strip()
        if not address.startswith("0x") or len(address) != 42:
            raise ValueError(f"Invalid {var_name}: Must be a 42-character hex string starting with '0x'.")
        try:
            if not is_checksum_address(address):
                address = to_checksum_address(address)
            return address
        except Exception as e:
            raise ValueError(f"Invalid {var_name} address format: {e}")

    def _parse_numeric_string(self, value: str) -> str:
        """
        Clean numeric strings by removing underscores and inline comments.

        Args:
            value (str): The numeric string.

        Returns:
            str: The cleaned string.
        """
        value = value.split("#")[0].strip()
        return value.replace("_", "")

    def _get_env_str(self, var_name: str, default: Optional[str], description: str, required: bool = False) -> str:
        """
        Retrieve an environment variable as a string.

        Args:
            var_name (str): The environment variable name.
            default (Optional[str]): The default value if not set.
            description (str): Description of the variable.
            required (bool): Whether this variable is required.

        Returns:
            str: The variable's value.

        Raises:
            ValueError: If a required variable is missing.
        """
        value = os.getenv(var_name, default)
        if (value is None or value == "") and required:
            logger.error(f"Missing required environment variable: {var_name} ({description})")
            raise ValueError(f"Missing required environment variable: {var_name} ({description})")
        # If the variable is meant to be an Ethereum address, perform validation.
        if "ADDRESS" in var_name and not any(x in var_name.upper() for x in ["ABI", "PATH", "ADDRESSES", "SIGNATURES", "SYMBOLS"]) and value not in [default, ""]:
            try:
                value = self._validate_ethereum_address(value, var_name)
            except ValueError as e:
                logger.error(f"Invalid Ethereum address for {var_name} ({description}): {e}")
                raise
        display_val = value if len(value) < 10 else value[:10] + "..."
        logger.debug(f"Loaded {var_name} from environment: {display_val} ({description})")
        return value

    def _get_env_int(self, var_name: str, default: Optional[int], description: str) -> int:
        """
        Retrieve an environment variable as an integer.

        Args:
            var_name (str): The variable name.
            default (Optional[int]): The default value if missing.
            description (str): Description of the variable.

        Returns:
            int: The integer value.

        Raises:
            ValueError: If conversion fails.
        """
        try:
            value = os.getenv(var_name)
            if value is None or value == "":
                if default is None:
                    logger.error(f"Missing required integer: {var_name} ({description})")
                    raise ValueError(f"Missing environment variable: {var_name} ({description})")
                logger.warning(f"{var_name} not set; using default: {default}")
                return default
            cleaned_value = self._parse_numeric_string(value)
            int_value = int(cleaned_value)
            logger.debug(f"Loaded {var_name}: {int_value} ({description})")
            return int_value
        except ValueError as e:
            logger.error(f"Invalid integer for {var_name} ({description}) - {e}")
            raise

    def _get_env_float(self, var_name: str, default: Optional[float], description: str) -> float:
        """
        Retrieve an environment variable as a float.

        Args:
            var_name (str): The variable name.
            default (Optional[float]): The default value.
            description (str): Description of the variable.

        Returns:
            float: The float value.

        Raises:
            ValueError: If conversion fails.
        """
        try:
            value = os.getenv(var_name)
            if value is None or value == "":
                if default is None:
                    logger.error(f"Missing required float: {var_name} ({description})")
                    raise ValueError(f"Missing environment variable: {var_name} ({description})")
                logger.warning(f"{var_name} not set; using default: {default}")
                return default
            cleaned_value = self._parse_numeric_string(value)
            float_value = float(cleaned_value)
            logger.debug(f"Loaded {var_name}: {float_value} ({description})")
            return float_value
        except ValueError as e:
            logger.error(f"Invalid float for {var_name} ({description}) - {e}")
            raise

    def _resolve_path(self, path_env_var: str, description: str) -> Path:
        """
        Resolve a file path from an environment variable and verify its existence.

        Args:
            path_env_var (str): The environment variable containing the path.
            description (str): Description of the file.

        Returns:
            Path: The absolute path to the file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path_str = self._get_env_str(path_env_var, None, description, required=True)
        full_path = self.BASE_PATH / path_str
        if not full_path.exists():
            logger.error(f"File not found: {full_path} ({description})")
            raise FileNotFoundError(f"File not found: {full_path} ({description})")
        logger.debug(f"Resolved path: {full_path} ({description})")
        return full_path

    async def _load_json_safe(self, file_path: Path, description: str) -> Any:
        """
        Asynchronously load JSON data from a file with error handling.

        Args:
            file_path (Path): The path to the JSON file.
            description (str): Description for logging.

        Returns:
            Any: Parsed JSON content.

        Raises:
            FileNotFoundError: If the file is missing.
            ValueError: If the JSON is invalid.
            RuntimeError: For other errors.
        """
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()
            data = json.loads(content)
            logger.debug(f"Loaded {description} from {file_path}")
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing {description} file: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {description} file: {e} in {file_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading {description}: {e} from {file_path}")

    async def get_token_addresses(self) -> List[str]:
        """
        Retrieve the list of monitored token addresses from a JSON file.

        Returns:
            List[str]: A list of token addresses.

        Raises:
            ValueError: If the JSON format is invalid.
        """
        data = await self._load_json_safe(self.TOKEN_ADDRESSES, "monitored tokens")
        if not isinstance(data, dict):
            logger.error("Invalid token addresses format; must be a dictionary.")
            raise ValueError("Invalid format for token addresses; expected a dictionary.")
        addresses = list(data.keys())
        logger.info(f"Loaded {len(addresses)} token addresses.")
        return addresses

    async def get_token_symbols(self) -> Dict[str, str]:
        """
        Retrieve the mapping of token addresses to symbols from a JSON file.

        Returns:
            Dict[str, str]: A mapping of addresses to symbols.

        Raises:
            ValueError: If the format is invalid.
        """
        data = await self._load_json_safe(self.TOKEN_SYMBOLS, "token symbols")
        if not isinstance(data, dict):
            logger.error("Invalid token symbols format; must be a dictionary.")
            raise ValueError("Invalid format for token symbols; expected a dictionary.")
        return data

    async def get_erc20_signatures(self) -> Dict[str, str]:
        """
        Retrieve the ERC20 function signatures from a JSON file.

        Returns:
            Dict[str, str]: A mapping of function names to signatures.

        Raises:
            ValueError: If the format is invalid.
        """
        data = await self._load_json_safe(self.ERC20_SIGNATURES, "ERC20 function signatures")
        if not isinstance(data, dict):
            logger.error("Invalid ERC20 signatures format; must be a dictionary.")
            raise ValueError("Invalid format for ERC20 signatures; expected a dictionary.")
        return data

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a configuration value safely.

        Args:
            key (str): The configuration key.
            default (Any): Default value if not found.

        Returns:
            Any: The configuration value.
        """
        try:
            return getattr(self, key, default)
        except AttributeError:
            logger.warning(f"Configuration key '{key}' not found; using default: {default}")
            return default

    def get_all_config_values(self) -> Dict[str, Any]:
        """
        Return a dictionary of all configuration values.

        Returns:
            Dict[str, Any]: All configuration attributes.
        """
        return vars(self)

    async def load(self, web3: Optional[Any] = None) -> None:
        """
        Load and validate all configuration data, including creation of required directories,
        loading critical ABI files, validating API keys, and verifying Ethereum addresses.

        Args:
            web3 (Optional[Any]): A Web3 instance if available (for context).

        Raises:
            RuntimeError: If any step of the load process fails.
        """
        try:
            await self._create_required_directories()
            await self._load_critical_abis()
            await self._validate_api_keys()
            self._validate_addresses()
            logger.debug("Configuration loaded successfully.")
        except Exception as e:
            logger.error(f"Configuration load failed: {e}")
            raise

    async def _create_required_directories(self) -> None:
        """
        Ensure that necessary directories exist. If not, create them asynchronously.
        """
        required_dirs = [
            Path(self.LINEAR_REGRESSION_PATH),
            self.BASE_PATH / "abi",
            self.BASE_PATH / "utils"
        ]
        for directory in required_dirs:
            await asyncio.to_thread(os.makedirs, directory, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")

    async def _load_critical_abis(self) -> None:
        """
        Verify that critical ABI files exist by resolving their paths.
        
        Raises:
            RuntimeError: If any critical ABI file is missing.
        """
        try:
            self.AAVE_FLASHLOAN_ABI_PATH = self._resolve_path("AAVE_FLASHLOAN_ABI", "AAVE Flashloan ABI file path")
            self.AAVE_POOL_ABI_PATH = self._resolve_path("AAVE_POOL_ABI", "AAVE Pool ABI file path")
            self.ERC20_ABI_PATH = self._resolve_path("ERC20_ABI", "ERC20 ABI file path")
            self.UNISWAP_ABI_PATH = self._resolve_path("UNISWAP_ABI", "Uniswap ABI file path")
            self.SUSHISWAP_ABI_PATH = self._resolve_path("SUSHISWAP_ABI", "Sushiswap ABI file path")
            self.GAS_PRICE_ORACLE_ABI_PATH = self._resolve_path("GAS_PRICE_ORACLE_ABI", "Gas Price Oracle ABI file path")
        except Exception as e:
            raise RuntimeError(f"Failed to load critical ABI paths: {e}")

    async def _validate_api_keys(self) -> None:
        """
        Validate that essential API keys are set and respond appropriately by making test HTTP requests.

        Raises:
            ValueError: If any API key is found invalid.
        """
        required_keys = [
            ("ETHERSCAN_API_KEY", "https://api.etherscan.io/api?module=stats&action=ethprice&apikey={key}"),
            ("INFURA_API_KEY", None),
            ("COINGECKO_API_KEY", None),
            ("COINMARKETCAP_API_KEY", "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest?CMC_PRO_API_KEY={key}&limit=1"),
            ("CRYPTOCOMPARE_API_KEY", "https://min-api.cryptocompare.com/data/price?fsym=BTC&tsyms=USD,JPY,EUR&api_key={key}")
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
                    payload = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "web3_clientVersion",
                        "params": []
                    }
                    try:
                        async with session.post(test_url, json=payload, timeout=10) as response:
                            data = await response.json()
                            if "result" not in data:
                                invalid_keys.append(f"{key_name} (Invalid response)")
                            else:
                                logger.debug(f"{key_name} validated successfully.")
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
                                logger.debug(f"{key_name} validated successfully (free endpoint).")
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
                                logger.debug(f"{key_name} validated successfully.")
                    except Exception as e:
                        invalid_keys.append(f"{key_name} (Error: {e})")
                    continue

                # Default validation using the provided test URL
                test_url = test_url_template.format(key=api_key)
                try:
                    async with session.get(test_url, timeout=10) as response:
                        if response.status != 200:
                            invalid_keys.append(f"{key_name} (Status: {response.status})")
                        else:
                            logger.debug(f"{key_name} validated successfully.")
                except Exception as e:
                    invalid_keys.append(f"{key_name} (Error: {e})")

        if missing_keys:
            logger.warning(f"Missing API keys: {', '.join(missing_keys)}")
        if invalid_keys:
            logger.error(f"Invalid API keys: {', '.join(invalid_keys)}")
            raise ValueError(f"Invalid API keys detected: {', '.join(invalid_keys)}")

    def _validate_addresses(self) -> None:
        """
        Validate all Ethereum addresses defined in the configuration.
        """
        address_fields = [
            "WALLET_ADDRESS",
            "UNISWAP_ADDRESS",
            "SUSHISWAP_ADDRESS",
            "AAVE_POOL_ADDRESS",
            "AAVE_FLASHLOAN_ADDRESS",
            "GAS_PRICE_ORACLE_ADDRESS"
        ]
        for field in address_fields:
            value = getattr(self, field, None)
            if value:
                try:
                    validated = self._validate_ethereum_address(value, field)
                    setattr(self, field, validated)
                except ValueError as e:
                    logger.error(f"Invalid address for {field}: {e}")
                    raise

# --- End of configuration.py ---
