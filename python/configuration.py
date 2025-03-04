#========================================================================================================================
# https://github.com/John0n1/0xBuilder

import os
import json
import aiofiles
import dotenv
import aiohttp

from eth_utils import is_checksum_address
from pathlib import Path
from typing import Any, Dict, List, Optional


from loggingconfig import setup_logging
import logging

logger = setup_logging("Configuration", level=logging.INFO)



class Configuration:
    """
    Loads configuration from environment variables and JSON files.
    Provides methods to access and validate configuration data.
    """

    def __init__(self, env_path: Optional[str] = None) -> None:
        """
        Initialize Configuration with default values.
        """
        self.env_path = env_path if env_path else ".env"
        self.BASE_PATH = Path(__file__).parent.parent
        self._load_env()
        self._initialize_defaults()
        self.signatures: Dict[str, Dict[str, str]] = {}
        self.method_selectors: Dict[str, Dict[str, str]] = {}

    def _initialize_defaults(self) -> None:
        """Initialize configuration with default values."""
        # General settings        
        DEFAULT_MAX_GAS_PRICE = 100_000_000_000 
        DEFAULT_GAS_LIMIT = 1_000_000
        DEFAULT_MAX_SLIPPAGE = 0.01
        DEFAULT_MIN_PROFIT = 0.001
        DEFAULT_MIN_BALANCE = 0.000001
        DEFAULT_MEMORY_CHECK_INTERVAL = 300
        DEFAULT_COMPONENT_HEALTH_CHECK_INTERVAL = 60
        DEFAULT_PROFITABLE_TX_PROCESS_TIMEOUT = 1.0

        self.MAX_GAS_PRICE = self._get_env_int("MAX_GAS_PRICE", DEFAULT_MAX_GAS_PRICE)
        self.GAS_LIMIT = self._get_env_int("GAS_LIMIT", DEFAULT_GAS_LIMIT)
        self.MAX_SLIPPAGE = self._get_env_float("MAX_SLIPPAGE", DEFAULT_MAX_SLIPPAGE)
        self.MIN_PROFIT = self._get_env_float("MIN_PROFIT", DEFAULT_MIN_PROFIT)
        self.MIN_BALANCE = self._get_env_float("MIN_BALANCE", DEFAULT_MIN_BALANCE)
        self.MEMORY_CHECK_INTERVAL = self._get_env_int("MEMORY_CHECK_INTERVAL", DEFAULT_MEMORY_CHECK_INTERVAL)
        self.COMPONENT_HEALTH_CHECK_INTERVAL = self._get_env_int("COMPONENT_HEALTH_CHECK_INTERVAL", DEFAULT_COMPONENT_HEALTH_CHECK_INTERVAL)
        self.PROFITABLE_TX_PROCESS_TIMEOUT = self._get_env_float("PROFITABLE_TX_PROCESS_TIMEOUT", DEFAULT_PROFITABLE_TX_PROCESS_TIMEOUT)


        # Standard addresses
        self.WETH_ADDRESS = self._get_env_str("WETH_ADDRESS")
        self.USDC_ADDRESS = self._get_env_str("USDC_ADDRESS")
        self.USDT_ADDRESS = self._get_env_str("USDT_ADDRESS")

        # API Keys and Endpoints
        self.ETHERSCAN_API_KEY: str = self._get_env_str("ETHERSCAN_API_KEY")
        self.INFURA_PROJECT_ID: str = self._get_env_str("INFURA_PROJECT_ID")
        self.INFURA_API_KEY: str = self._get_env_str("INFURA_API_KEY")
        self.COINGECKO_API_KEY: str = self._get_env_str("COINGECKO_API_KEY")
        self.COINMARKETCAP_API_KEY: str = self._get_env_str("COINMARKETCAP_API_KEY")
        self.CRYPTOCOMPARE_API_KEY: str = self._get_env_str("CRYPTOCOMPARE_API_KEY")
        self.HTTP_ENDPOINT: Optional[str] = self._get_env_str("HTTP_ENDPOINT")
        self.WEBSOCKET_ENDPOINT: Optional[str] = self._get_env_str("WEBSOCKET_ENDPOINT", None)
        self.IPC_ENDPOINT: Optional[str] = self._get_env_str("IPC_ENDPOINT", None)

        # Account
        self.WALLET_ADDRESS: str = self._get_env_str("WALLET_ADDRESS")
        self.WALLET_KEY: str = self._get_env_str("WALLET_KEY")
        self.API_KEY: str = self._get_env_str("API_KEY")

        # Paths
        self.ERC20_ABI: Path = self._resolve_path("ERC20_ABI")
        self.AAVE_FLASHLOAN_ABI: Path = self._resolve_path("AAVE_FLASHLOAN_ABI")
        self.AAVE_POOL_ABI: Path = self._resolve_path("AAVE_POOL_ABI")
        self.UNISWAP_ABI: Path = self._resolve_path("UNISWAP_ABI")
        self.SUSHISWAP_ABI: Path = self._resolve_path("SUSHISWAP_ABI")
        self.ERC20_SIGNATURES: Path = self._resolve_path("ERC20_SIGNATURES")
        self.TOKEN_ADDRESSES: Path = self._resolve_path("TOKEN_ADDRESSES")
        self.TOKEN_SYMBOLS: Path = self._resolve_path("TOKEN_SYMBOLS")
        self.GAS_PRICE_ORACLE_ABI: Path = self._resolve_path("GAS_PRICE_ORACLE_ABI")

        # Router Addresses
        self.UNISWAP_ADDRESS: str = self._get_env_str("UNISWAP_ADDRESS")
        self.SUSHISWAP_ADDRESS: str = self._get_env_str("SUSHISWAP_ADDRESS")
        self.AAVE_POOL_ADDRESS: str = self._get_env_str("AAVE_POOL_ADDRESS")
        self.AAVE_FLASHLOAN_ADDRESS: str = self._get_env_str("AAVE_FLASHLOAN_ADDRESS")
        self.GAS_PRICE_ORACLE_ADDRESS: str = self._get_env_str("GAS_PRICE_ORACLE_ADDRESS")

        self.SLIPPAGE_DEFAULT: float = self._get_env_float("SLIPPAGE_DEFAULT", 0.1)
        self.MIN_SLIPPAGE: float = self._get_env_float("MIN_SLIPPAGE", 0.01)
        self.MAX_SLIPPGAGE: float = self._get_env_float("MAX_SLIPPGAGE", 0.5)
        self.SLIPPAGE_HIGH_CONGESTION: float = self._get_env_float("SLIPPAGE_HIGH_CONGESTION", 0.05)
        self.SLIPPAGE_LOW_CONGESTION: float = self._get_env_float("SLIPPAGE_LOW_CONGESTION", 0.2)
        self.MAX_GAS_PRICE_GWEI: int = self._get_env_int("MAX_GAS_PRICE_GWEI", 500)
        self.MIN_PROFIT_MULTIPLIER: float = self._get_env_float("MIN_PROFIT_MULTIPLIER", 2.0)
        self.BASE_GAS_LIMIT: int = self._get_env_int("BASE_GAS_LIMIT", 21000)
        self.DEFAULT_CANCEL_GAS_PRICE_GWEI: int = self._get_env_int("DEFAULT_CANCEL_GAS_PRICE_GWEI", 60)  
        self.ETH_TX_GAS_PRICE_MULTIPLIER: float = self._get_env_float("ETH_TX_GAS_PRICE_MULTIPLIER", 1.2)

        # ML model configuration
        self.MODEL_RETRAINING_INTERVAL: int = self._get_env_int("MODEL_RETRAINING_INTERVAL", 3600) 
        self.MIN_TRAINING_SAMPLES: int = self._get_env_int("MIN_TRAINING_SAMPLES", 100)
        self.MODEL_ACCURACY_THRESHOLD: float = self._get_env_float("MODEL_ACCURACY_THRESHOLD", 0.7)
        self.PREDICTION_CACHE_TTL: int = self._get_env_int("PREDICTION_CACHE_TTL", 300) 
        self.LINEAR_REGRESSION_PATH: str = str(self.BASE_PATH / "linear_regression")
        self.MODEL_PATH: str = str(self.BASE_PATH / "linear_regression" / "price_model.joblib")
        self.TRAINING_DATA_PATH: str = str(self.BASE_PATH / "linear_regression" / "training_data.csv")

        self.LINEAR_REGRESSION_PATH = str(self.BASE_PATH / "linear_regression")
        self.MODEL_PATH = str(self.BASE_PATH / "linear_regression" / "price_model.joblib")
        self.TRAINING_DATA_PATH = str(self.BASE_PATH / "linear_regression" / "training_data.csv")
        os.makedirs(self.LINEAR_REGRESSION_PATH, exist_ok=True)


        # Mempool Monitor Configs
        self.MEMPOOL_MAX_RETRIES: int = self._get_env_int("MEMPOOL_MAX_RETRIES", 3)
        self.MEMPOOL_RETRY_DELAY: int = self._get_env_int("MEMPOOL_RETRY_DELAY", 2)
        self.MEMPOOL_BATCH_SIZE: int = self._get_env_int("MEMPOOL_BATCH_SIZE", 10)
        self.MEMPOOL_MAX_PARALLEL_TASKS: int = self._get_env_int("MEMPOOL_MAX_PARALLEL_TASKS", 5)

        # Nonce Core Configs
        self.NONCE_CACHE_TTL: int = self._get_env_int("NONCE_CACHE_TTL", 60)
        self.NONCE_RETRY_DELAY: int = self._get_env_int("NONCE_RETRY_DELAY", 1)
        self.NONCE_MAX_RETRIES: int = self._get_env_int("NONCE_MAX_RETRIES", 5)
        self.NONCE_TRANSACTION_TIMEOUT: int = self._get_env_int("NONCE_TRANSACTION_TIMEOUT", 120)

        # Safety Net Configs
        self.SAFETYNET_CACHE_TTL: int = self._get_env_int("SAFETYNET_CACHE_TTL", 300)
        self.SAFETYNET_GAS_PRICE_TTL: int = self._get_env_int("SAFETYNET_GAS_PRICE_TTL", 30)

        # Strategy Net Configs
        self.AGGRESSIVE_FRONT_RUN_MIN_VALUE_ETH: float = self._get_env_float("AGGRESSIVE_FRONT_RUN_MIN_VALUE_ETH", 0.1)
        self.AGGRESSIVE_FRONT_RUN_RISK_SCORE_THRESHOLD: float = self._get_env_float("AGGRESSIVE_FRONT_RUN_RISK_SCORE_THRESHOLD", 0.7)
        self.FRONT_RUN_OPPORTUNITY_SCORE_THRESHOLD: int = self._get_env_int("FRONT_RUN_OPPORTUNITY_SCORE_THRESHOLD", 75)
        self.VOLATILITY_FRONT_RUN_SCORE_THRESHOLD: int = self._get_env_int("VOLATILITY_FRONT_RUN_SCORE_THRESHOLD", 75)
        self.ADVANCED_FRONT_RUN_RISK_SCORE_THRESHOLD: int = self._get_env_int("ADVANCED_FRONT_RUN_RISK_SCORE_THRESHOLD", 75)
        self.PRICE_DIP_BACK_RUN_THRESHOLD: float = self._get_env_float("PRICE_DIP_BACK_RUN_THRESHOLD", 0.99)
        self.FLASHLOAN_BACK_RUN_PROFIT_PERCENTAGE: float = self._get_env_float("FLASHLOAN_BACK_RUN_PROFIT_PERCENTAGE", 0.02)
        self.HIGH_VOLUME_BACK_RUN_DEFAULT_THRESHOLD_USD: float = self._get_env_float("HIGH_VOLUME_BACK_RUN_DEFAULT_THRESHOLD_USD", 100000)
        self.SANDWICH_ATTACK_GAS_PRICE_THRESHOLD_GWEI: int = self._get_env_int("SANDWICH_ATTACK_GAS_PRICE_THRESHOLD_GWEI", 200)
        self.PRICE_BOOST_SANDWICH_MOMENTUM_THRESHOLD: float = self._get_env_float("PRICE_BOOST_SANDWICH_MOMENTUM_THRESHOLD", 0.02)

        self.HIGH_VALUE_THRESHOLD: int = self._get_env_int("HIGH_VALUE_THRESHOLD", 1000000000000000000) 


    def _load_env(self) -> None:
        """Load environment variables fromenv file."""
        dotenv.load_dotenv(dotenv_path=self.env_path)
        logger.debug(f"Environment variables loaded from: {self.env_path}")

    def _validate_ethereum_address(self, address: str, var_name: str) -> str:
        """
        Validate Ethereum address format.
        Returns: Normalized address string
        Raises: ValueError if invalid
        """
        if not isinstance(address, str):
            raise ValueError(f"{var_name} must be a string, got {type(address)}")

        address = address.strip()
        try:
            if not is_checksum_address(address):
                # Convert lowercase to checksum without using web3
                if address.lower() == address and len(address) == 42 and address.startswith('0x'): 
                    from eth_utils import to_checksum_address
                    address = to_checksum_address(address)
                else:
                    raise ValueError(f"Invalid {var_name}: Not a checksummed address.")
            return address
        except Exception as e:
            raise ValueError(f"Invalid {var_name} address format: {str(e)}")

    def _get_env_str(self, var_name: str, default: Optional[str] = None) -> str:
        """Get an environment variable as string, raising error if missing."""
        value = os.getenv(var_name, default)
        if value is None:
            if default is None: 
                logger.error(f"Missing environment variable: {var_name}")
                raise ValueError(f"Missing environment variable: {var_name}")
            return default 


        if 'ADDRESS' in var_name and not any(path_suffix in var_name for path_suffix in ['ABI', 'PATH', 'ADDRESSES', 'SIGNATURES', 'SYMBOLS']) and value != default:
            try:
                return self._validate_ethereum_address(value, var_name)
            except ValueError as e:
                logger.error(f"Invalid Ethereum address format for {var_name}: {e}")
                raise

        return value

    def _parse_numeric_string(self, value: str) -> str:
        """Remove underscores and comments from numeric strings."""
        value = value.split('#')[0].strip()
        return value.replace('_', '')

    def _get_env_int(self, var_name: str, default: Optional[int] = None) -> int:
        """Get an environment variable as int, raising error if missing or invalid."""
        try:
            value = os.getenv(var_name)
            if value is None:
                if default is None:
                    raise ValueError(f"Missing environment variable: {var_name}")
                return default
            cleaned_value = self._parse_numeric_string(value)
            return int(cleaned_value)
        except ValueError as e:
            logger.error(f"Invalid or missing integer environment variable: {var_name} - {e}")
            raise

    def _get_env_float(self, var_name: str, default: Optional[float] = None) -> float:
        """Get an environment variable as float, raising error if missing or invalid."""
        try:
            value = os.getenv(var_name)
            if value is None:
                if default is None:
                    raise ValueError(f"Missing environment variable: {var_name}")
                return default
            cleaned_value = self._parse_numeric_string(value)
            return float(cleaned_value)
        except ValueError as e:
            logger.error(f"Invalid or missing float environment variable: {var_name} - {e}")
            raise

    def _resolve_path(self, path_env_var: str) -> Path:
        """Resolve a path from environment variable and ensure the file exists."""
        path_str = self._get_env_str(path_env_var)
        full_path = self.BASE_PATH / path_str
        if not full_path.exists():
            logger.error(f"File not found: {full_path}")
            raise FileNotFoundError(f"File not found: {full_path}")
        logger.debug(f"Resolved path: {full_path}")
        return full_path

    async def _load_json_safe(self, file_path: Path, description: str) -> Any:
        """Load JSON with better error handling and validation."""
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
         """Get the list of monitored token addresses from the config file."""
         data = await self._load_json_safe(self.TOKEN_ADDRESSES, "monitored tokens")
         if not isinstance(data, dict):
             logger.error("Invalid format for token addresses: must be a dictionary")
             raise ValueError("Invalid format for token addresses: must be a dictionary in token addresses file")

         addresses = list(data.keys())
         logger.info(f"Loaded {len(addresses)} token addresses")
         return addresses

    async def get_token_symbols(self) -> Dict[str, str]:
        """Get the mapping of token addresses to symbols from the config file."""
        data = await self._load_json_safe(self.TOKEN_SYMBOLS, "token symbols")
        if not isinstance(data, dict):
            logger.error("Invalid format for token symbols: must be a dict")
            raise ValueError("Invalid format for token symbols: must be a dict in token symbols file")
        return data

    async def get_erc20_signatures(self) -> Dict[str, str]:
        """Load ERC20 function signatures from JSON."""
        data = await self._load_json_safe(self.ERC20_SIGNATURES, "ERC20 function signatures")
        if not isinstance(data, dict):
            logger.error("Invalid format for ERC20 signatures: must be a dict")
            raise ValueError("Invalid format for ERC20 signatures: must be a dict in ERC20 signatures file")
        return data

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Safe configuration value access with default.
        """
        try:
            return getattr(self, key, default)
        except AttributeError:
            logger.warning(f"Configuration key '{key}' not found, using default: {default}")
            return default

    def get_all_config_values(self) -> Dict[str, Any]:
         """Returns all configuration values as a dictionary."""
         return vars(self)

    async def load_abi_from_path(self, path: Path) -> List[Dict[str, Any]]:
        """Load and return ABI content from a file path."""
        try:
            async with aiofiles.open(path, 'r') as f:
                content = await f.read()
                abi = json.loads(content)
                if not isinstance(abi, list):
                    raise ValueError(f"Invalid ABI format in {path}: not a list")
                return abi
        except Exception as e:
            logger.error(f"Failed to load ABI from {path}: {e}")
            raise

    async def load(self, web3=None) -> None: 
        """Load and validate all configuration data."""
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
        """Create necessary directories if they don't exist."""
        required_dirs = [
            self.LINEAR_REGRESSION_PATH,
            self.BASE_PATH / "abi",
            self.BASE_PATH / "utils"
        ]

        for directory in required_dirs:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")

    async def _load_critical_abis(self) -> None:
        """Load and validate critical ABIs (Now just loads paths)."""
        try:
            self.AAVE_FLASHLOAN_ABI_PATH = self._resolve_path("AAVE_FLASHLOAN_ABI") 
            self.AAVE_POOL_ABI_PATH = self._resolve_path("AAVE_POOL_ABI")
            self.ERC20_ABI_PATH = self._resolve_path("ERC20_ABI") 
            self.UNISWAP_ABI_PATH = self._resolve_path("UNISWAP_ABI") 
            self.SUSHISWAP_ABI_PATH = self._resolve_path("SUSHISWAP_ABI") 
            self.GAS_PRICE_ORACLE_ABI_PATH = self._resolve_path("GAS_PRICE_ORACLE_ABI") 

        except Exception as e:
            raise RuntimeError(f"Failed to load critical ABIs paths: {e}")

    async def _validate_api_keys(self) -> None:
        """Validate required API keys are set and functional."""
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
        """Validate all Ethereum addresses in configuration."""
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
                    setattr(self, field,
                           self._validate_ethereum_address(value, field))
                except ValueError as e:
                    logger.error(f"Invalid address for {field}: {e}")
                    raise