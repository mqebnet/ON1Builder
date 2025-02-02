import os
import json
import aiofiles
import dotenv

from eth_utils import function_signature_to_4byte_selector
from pathlib import Path
from typing import Any, Dict, List, Optional

import logging as logger
from main_core import setup_logging

setup_logging()

logger = logger.getLogger(__name__)

# Constants
DEFAULT_MAX_GAS_PRICE = 100_000_000_000  # 100 Gwei in wei
DEFAULT_GAS_LIMIT = 1_000_000
DEFAULT_MAX_SLIPPAGE = 0.01
DEFAULT_MIN_PROFIT = 0.001
DEFAULT_MIN_BALANCE = 0.000001

# Standard Ethereum addresses
MAINNET_ADDRESSES = {
    'WETH': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
    'USDC': '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
    'USDT': '0xdAC17F958D2ee523a2206206994597C13D831ec7'
}

class Configuration:
    """
    Loads configuration from environment variables and JSON files.
    Provides methods to access and validate configuration data.
    """

    def __init__(self, env_path: Optional[str] = None) -> None:
        """
        Initialize Configuration with default values.
        Args:
            env_path: Optional path to the .env file. Defaults to the current directory.
        """
        self.env_path = env_path if env_path else ".env"
        self._load_env()
        self._initialize_defaults()
        self.signatures: Dict[str, Dict[str, str]] = {}
        self.method_selectors: Dict[str, Dict[str, str]] = {}
        self.BASE_PATH = Path(__file__).parent.parent

    def _initialize_defaults(self) -> None:
        """Initialize configuration with default values."""
        # General settings
        self.MAX_GAS_PRICE = self._get_env_int("MAX_GAS_PRICE", DEFAULT_MAX_GAS_PRICE)
        self.GAS_LIMIT = self._get_env_int("GAS_LIMIT", DEFAULT_GAS_LIMIT)
        self.MAX_SLIPPAGE = self._get_env_float("MAX_SLIPPAGE", DEFAULT_MAX_SLIPPAGE)
        self.MIN_PROFIT = self._get_env_float("MIN_PROFIT", DEFAULT_MIN_PROFIT)
        self.MIN_BALANCE = self._get_env_float("MIN_BALANCE", DEFAULT_MIN_BALANCE)

        # Standard addresses
        self.WETH_ADDRESS = MAINNET_ADDRESSES['WETH']
        self.USDC_ADDRESS = MAINNET_ADDRESSES['USDC']
        self.USDT_ADDRESS = MAINNET_ADDRESSES['USDT']

        # API Keys and Endpoints
        self.ETHERSCAN_API_KEY: str = self._get_env_str("ETHERSCAN_API_KEY")
        self.INFURA_PROJECT_ID: str = self._get_env_str("INFURA_PROJECT_ID")
        self.INFURA_API_KEY: str = self._get_env_str("INFURA_API_KEY")
        self.COINGECKO_API_KEY: str = self._get_env_str("COINGECKO_API_KEY")
        self.COINMARKETCAP_API_KEY: str = self._get_env_str("COINMARKETCAP_API_KEY")
        self.CRYPTOCOMPARE_API_KEY: str = self._get_env_str("CRYPTOCOMPARE_API_KEY")
        self.HTTP_ENDPOINT: Optional[str] = self._get_env_str("HTTP_ENDPOINT", None)
        self.WEBSOCKET_ENDPOINT: Optional[str] = self._get_env_str("WEBSOCKET_ENDPOINT", None)
        self.IPC_ENDPOINT: Optional[str] = self._get_env_str("IPC_ENDPOINT", None)
        
        # Account
        self.WALLET_ADDRESS: str = self._get_env_str("WALLET_ADDRESS")
        self.WALLET_KEY: str = self._get_env_str("WALLET_KEY")

        # Paths
        self.ERC20_ABI: Path = self._resolve_path("ERC20_ABI")
        self.AAVE_FLASHLOAN_ABI: Path = self._resolve_path("AAVE_FLASHLOAN_ABI")
        self.AAVE_POOL_ABI: Path = self._resolve_path("AAVE_POOL_ABI")
        self.UNISWAP_ABI: Path = self._resolve_path("UNISWAP_ABI")
        self.SUSHISWAP_ABI: Path = self._resolve_path("SUSHISWAP_ABI")
        self.ERC20_SIGNATURES: Path = self._resolve_path("ERC20_SIGNATURES")
        self.TOKEN_ADDRESSES: Path = self._resolve_path("TOKEN_ADDRESSES")
        self.TOKEN_SYMBOLS: Path = self._resolve_path("TOKEN_SYMBOLS")

        # Router Addresses
        self.UNISWAP_ADDRESS: str = self._get_env_str("UNISWAP_ADDRESS")
        self.SUSHISWAP_ADDRESS: str = self._get_env_str("SUSHISWAP_ADDRESS")
        self.AAVE_POOL_ADDRESS: str = self._get_env_str("AAVE_POOL_ADDRESS")
        self.AAVE_FLASHLOAN_ADDRESS: str = self._get_env_str("AAVE_FLASHLOAN_ADDRESS")
    
        # Set default values for strategies
        self.SLIPPAGE_DEFAULT: float = 0.1
        self.SLIPPAGE_MIN: float = 0.01
        self.SLIPPAGE_MAX: float = 0.5
        self.SLIPPAGE_HIGH_CONGESTION: float = 0.05
        self.SLIPPAGE_LOW_CONGESTION: float = 0.2
        self.MAX_GAS_PRICE_GWEI: int = 500
        self.MIN_PROFIT_MULTIPLIER: float = 2.0
        self.BASE_GAS_LIMIT: int = 21000
        
        # ML model configuration
        self.MODEL_RETRAINING_INTERVAL: int = 3600  # 1 hour
        self.MIN_TRAINING_SAMPLES: int = 100
        self.MODEL_ACCURACY_THRESHOLD: float = 0.7
        self.PREDICTION_CACHE_TTL: int = 300  # 5 minutes
        self.LINEAR_REGRESSION_PATH: str = str(self.BASE_PATH / "linear_regression")
        self.MODEL_PATH: str = str(self.BASE_PATH / "linear_regression" / "price_model.joblib")
        self.TRAINING_DATA_PATH: str = str(self.BASE_PATH / "linear_regression" / "training_data.csv")
        
        # Ensure ML paths are absolute
        self.LINEAR_REGRESSION_PATH = str(self.BASE_PATH / "linear_regression")
        self.MODEL_PATH = str(self.BASE_PATH / "linear_regression" / "price_model.joblib")
        self.TRAINING_DATA_PATH = str(self.BASE_PATH / "linear_regression" / "training_data.csv")
        
        # Create directories if they don't exist
        os.makedirs(self.LINEAR_REGRESSION_PATH, exist_ok=True)
        
    def _load_env(self) -> None:
        """Load environment variables fromenv file."""
        dotenv.load_dotenv(dotenv_path=self.env_path)
        logger.debug(f"Environment variables loaded from: {self.env_path}")

    def _validate_ethereum_address(self, address: str, var_name: str) -> str:
        """
        Validate Ethereum address format with improved checks.
        Returns: Normalized address string
        Raises: ValueError if invalid
        """
        if not isinstance(address, str):
            raise ValueError(f"{var_name} must be a string, got {type(address)}")
        
        clean_address = address.lower().strip()
        if not clean_address.startswith('0x'):
            clean_address = '0x' + clean_address
            
        if len(clean_address) != 42:
            raise ValueError(f"Invalid {var_name} length: {len(clean_address)}")
            
        try:
            int(clean_address[2:], 16)
            return clean_address
        except ValueError:
            raise ValueError(f"Invalid hex format for {var_name}")

    def _get_env_str(self, var_name: str, default: Optional[str] = None) -> str:
        """Get an environment variable as string, raising error if missing."""
        value = os.getenv(var_name, default)
        if value is None:
            logger.error(f"Missing environment variable: {var_name}")
            raise ValueError(f"Missing environment variable: {var_name}")
            
        # Only validate Ethereum addresses for contract addresses
        if 'ADDRESS' in var_name and not any(path_suffix in var_name for path_suffix in ['ABI', 'PATH', 'ADDRESSES', 'SIGNATURES', 'SYMBOLS']):
            return self._validate_ethereum_address(value, var_name)
            
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
            # Clean the value before parsing
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
            # Clean the value before parsing
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
                    raise ValueError(f"Invalid JSON in {description}: {e}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing {description} file: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading {description}: {e}")

    async def get_token_addresses(self) -> List[str]:
         """Get the list of monitored token addresses from the config file."""
         data = await self._load_json_safe(self.TOKEN_ADDRESSES, "monitored tokens")
         if not isinstance(data, dict):
             logger.error("Invalid format for token addresses: must be a dictionary")
             raise ValueError("Invalid format for token addresses: must be a dictionary")
             
         # Extract addresses from dictionary keys
         addresses = list(data.keys())
         logger.info(f"Loaded {len(addresses)} token addresses")
         return addresses

    async def get_token_symbols(self) -> Dict[str, str]:
        """Get the mapping of token addresses to symbols from the config file."""
        data = await self._load_json_safe(self.TOKEN_SYMBOLS, "token symbols")
        if not isinstance(data, dict):
            logger.error("Invalid format for token symbols: must be a dict")
            raise ValueError("Invalid format for token symbols: must be a dict")
        return data

    async def get_erc20_signatures(self) -> Dict[str, str]:
        """Load ERC20 function signatures from JSON."""
        data = await self._load_json_safe(self.ERC20_SIGNATURES, "ERC20 function signatures")
        if not isinstance(data, dict):
            logger.error("Invalid format for ERC20 signatures: must be a dict")
            raise ValueError("Invalid format for ERC20 signatures: must be a dict")
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

    async def load(self) -> None:
        """Load and validate all configuration data."""
        try:
            # Create required directories if they don't exist
            self._create_required_directories()
            
            # Load and validate critical ABIs
            await self._load_critical_abis()
            
            # Validate API keys
            self._validate_api_keys()
            
            # Validate addresses
            self._validate_addresses()
            
            logger.info("Configuration loaded successfully")
            
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
        """Load and validate critical ABIs."""
        try:
            self.AAVE_FLASHLOAN_ABI = await self.load_abi_from_path(
                self._resolve_path("AAVE_FLASHLOAN_ABI")
            )
            self.AAVE_POOL_ABI = await self.load_abi_from_path(
                self._resolve_path("AAVE_POOL_ABI")
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load critical ABIs: {e}")

    def _validate_api_keys(self) -> None:
        """Validate required API keys are set."""
        required_keys = [
            'ETHERSCAN_API_KEY',
            'INFURA_API_KEY',
            'COINGECKO_API_KEY',
            'COINMARKETCAP_API_KEY',
            'CRYPTOCOMPARE_API_KEY'
        ]
        
        missing_keys = [
            key for key in required_keys 
            if not getattr(self, key, None)
        ]
        
        if missing_keys:
            logger.warning(f"Missing API keys: {', '.join(missing_keys)}")

    def _validate_addresses(self) -> None:
        """Validate all Ethereum addresses in configuration."""
        address_fields = [
            'WALLET_ADDRESS',
            'UNISWAP_ADDRESS',
            'SUSHISWAP_ADDRESS',
            'AAVE_POOL_ADDRESS',
            'AAVE_FLASHLOAN_ADDRESS'
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

    def _validate_abi(self, abi: List[Dict], abi_type: str) -> bool:
        """Validate the structure and required methods of an ABI."""
        if not isinstance(abi, list):
            logger.error(f"Invalid ABI format for {abi_type}")
            return False

        found_methods = {
            item.get('name') for item in abi
            if item.get('type') == 'function' and 'name' in item
        }

        required_methods = {
            'erc20': {'transfer', 'approve', 'transferFrom', 'balanceOf'},
            'uniswap': {'swapExactTokensForTokens', 'swapTokensForExactTokens', 'addLiquidity', 'getAmountsOut'},
            'sushiswap': {'swapExactTokensForTokens', 'swapTokensForExactTokens', 'addLiquidity', 'getAmountsOut'},
            'aave_flashloan': {'fn_RequestFlashLoan', 'executeOperation', 'ADDRESSES_PROVIDER', 'POOL'},
            'aave': {'admin', 'implementation', 'upgradeToAndCall'}
        }

        required = required_methods.get(abi_type, set())
        if not required.issubset(found_methods):
            missing = required - found_methods
            logger.error(f"Missing required methods in {abi_type} ABI: {missing}")
            return False

        return True

    def _extract_signatures(self, abi: List[Dict], abi_type: str) -> None:
        """Extract function signatures and method selectors from an ABI."""
        signatures = {}
        selectors = {}

        for item in abi:
            if item.get('type') == 'function':
                name = item.get('name')
                if name:
                    inputs = ','.join(inp.get('type', '') for inp in item.get('inputs', []))
                    signature = f"{name}({inputs})"
                    selector = function_signature_to_4byte_selector(signature)
                    hex_selector = selector.hex()

                    signatures[name] = signature
                    selectors[hex_selector] = name

        self.signatures[abi_type] = signatures
        self.method_selectors[abi_type] = selectors

    async def _load_abi_from_path(self, abi_path: Path, abi_type: str) -> List[Dict]:
        """Load and validate ABI content from the specified path."""
        try:
            if not abi_path.exists():
                logger.error(f"ABI file not found: {abi_path}")
                raise FileNotFoundError(f"ABI file not found: {abi_path}")

            async with aiofiles.open(abi_path, 'r', encoding='utf-8') as f:
                abi_content = await f.read()
                abi = json.loads(abi_content)
                logger.debug(f"ABI content loaded from {abi_path}")

            if not self._validate_abi(abi, abi_type):
                raise ValueError(f"Validation failed for {abi_type} ABI from file {abi_path}")

            return abi
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for {abi_type} in file {abi_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading ABI {abi_type}: {e}")
            raise
