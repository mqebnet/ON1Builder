import os
import json
from typing import Any, Dict, Optional
import dotenv
import aiofiles
import aiohttp
import asyncio
from pathlib import Path
from eth_utils import is_checksum_address, to_checksum_address
from loggingconfig import setup_logging
import yaml

logger = setup_logging("Configuration", level="DEBUG")  

class Configuration:
    def __init__(self, env_path: str = ".env", config_file: str = "config.yaml", environment: str = "development") -> None:
        dotenv.load_dotenv(dotenv_path=env_path)
        self.BASE_PATH = Path(__file__).parent.parent
        self.config_file = config_file
        self.environment = environment
        self.config_data = self._load_config_file()

        self.MAX_GAS_PRICE = self.get_config_value("MAX_GAS_PRICE", 100000000000)
        self.GAS_LIMIT = self.get_config_value("GAS_LIMIT", 1000000)
        self.MAX_SLIPPAGE = self.get_config_value("MAX_SLIPPAGE", 0.01)
        self.MIN_PROFIT = self.get_config_value("MIN_PROFIT", 0.001)
        self.MIN_BALANCE = self.get_config_value("MIN_BALANCE", 0.000001)
        self.MEMORY_CHECK_INTERVAL = self.get_config_value("MEMORY_CHECK_INTERVAL", 300)
        self.COMPONENT_HEALTH_CHECK_INTERVAL = self.get_config_value("COMPONENT_HEALTH_CHECK_INTERVAL", 60)
        self.PROFITABLE_TX_PROCESS_TIMEOUT = self.get_config_value("PROFITABLE_TX_PROCESS_TIMEOUT", 1.0)
        self.WETH_ADDRESS = self._validate_address(self.get_config_value("WETH_ADDRESS", ""), "WETH_ADDRESS")
        self.USDC_ADDRESS = self._validate_address(self.get_config_value("USDC_ADDRESS", ""), "USDC_ADDRESS")
        self.USDT_ADDRESS = self._validate_address(self.get_config_value("USDT_ADDRESS", ""), "USDT_ADDRESS")
        self.ETHERSCAN_API_KEY = self.get_config_value("ETHERSCAN_API_KEY", "")
        self.INFURA_PROJECT_ID = self.get_config_value("INFURA_PROJECT_ID", "")
        self.INFURA_API_KEY = self.get_config_value("INFURA_API_KEY", "")
        self.COINGECKO_API_KEY = self.get_config_value("COINGECKO_API_KEY", "")
        self.COINMARKETCAP_API_KEY = self.get_config_value("COINMARKETCAP_API_KEY", "")
        self.CRYPTOCOMPARE_API_KEY = self.get_config_value("CRYPTOCOMPARE_API_KEY", "")
        self.HTTP_ENDPOINT = self.get_config_value("HTTP_ENDPOINT", "")
        self.WEBSOCKET_ENDPOINT = self.get_config_value("WEBSOCKET_ENDPOINT", "")
        self.IPC_ENDPOINT = self.get_config_value("IPC_ENDPOINT", "")
        self.WALLET_ADDRESS = self._validate_address(self.get_config_value("WALLET_ADDRESS", ""), "WALLET_ADDRESS")
        self.WALLET_KEY = self.get_config_value("WALLET_KEY", "")
        self.ERC20_ABI = self._resolve_path(self.get_config_value("ERC20_ABI", ""), "ERC20_ABI")
        self.AAVE_FLASHLOAN_ABI = self._resolve_path(self.get_config_value("AAVE_FLASHLOAN_ABI", ""), "AAVE_FLASHLOAN_ABI")
        self.AAVE_POOL_ABI = self._resolve_path(self.get_config_value("AAVE_POOL_ABI", ""), "AAVE_POOL_ABI")
        self.UNISWAP_ABI = self._resolve_path(self.get_config_value("UNISWAP_ABI", ""), "UNISWAP_ABI")
        self.SUSHISWAP_ABI = self._resolve_path(self.get_config_value("SUSHISWAP_ABI", ""), "SUSHISWAP_ABI")
        self.ERC20_SIGNATURES = self._resolve_path(self.get_config_value("ERC20_SIGNATURES", ""), "ERC20_SIGNATURES")
        self.TOKEN_ADDRESSES = self._resolve_path(self.get_config_value("TOKEN_ADDRESSES", ""), "TOKEN_ADDRESSES")
        self.TOKEN_SYMBOLS = self._resolve_path(self.get_config_value("TOKEN_SYMBOLS", ""), "TOKEN_SYMBOLS")
        self.GAS_PRICE_ORACLE_ABI = self._resolve_path(self.get_config_value("GAS_PRICE_ORACLE_ABI", ""), "GAS_PRICE_ORACLE_ABI")
        self.UNISWAP_ADDRESS = self._validate_address(self.get_config_value("UNISWAP_ADDRESS", ""), "UNISWAP_ADDRESS")
        self.SUSHISWAP_ADDRESS = self._validate_address(self.get_config_value("SUSHISWAP_ADDRESS", ""), "SUSHISWAP_ADDRESS")
        self.AAVE_POOL_ADDRESS = self._validate_address(self.get_config_value("AAVE_POOL_ADDRESS", ""), "AAVE_POOL_ADDRESS")
        self.AAVE_FLASHLOAN_ADDRESS = self._validate_address(self.get_config_value("AAVE_FLASHLOAN_ADDRESS", ""), "AAVE_FLASHLOAN_ADDRESS")
        self.GAS_PRICE_ORACLE_ADDRESS = self._validate_address(self.get_config_value("GAS_PRICE_ORACLE_ADDRESS", ""), "GAS_PRICE_ORACLE_ADDRESS")
        self.SLIPPAGE_DEFAULT = self.get_config_value("SLIPPAGE_DEFAULT", 0.1)
        self.MIN_SLIPPAGE = self.get_config_value("MIN_SLIPPAGE", 0.01)
        self.MAX_SLIPPAGE = self.get_config_value("MAX_SLIPPAGE", 0.5)
        self.SLIPPAGE_HIGH_CONGESTION = self.get_config_value("SLIPPAGE_HIGH_CONGESTION", 0.05)
        self.SLIPPAGE_LOW_CONGESTION = self.get_config_value("SLIPPAGE_LOW_CONGESTION", 0.2)
        self.MAX_GAS_PRICE_GWEI = self.get_config_value("MAX_GAS_PRICE_GWEI", 500)
        self.MIN_PROFIT_MULTIPLIER = self.get_config_value("MIN_PROFIT_MULTIPLIER", 2.0)
        self.BASE_GAS_LIMIT = self.get_config_value("BASE_GAS_LIMIT", 21000)
        self.DEFAULT_CANCEL_GAS_PRICE_GWEI = self.get_config_value("DEFAULT_CANCEL_GAS_PRICE_GWEI", 60)
        self.ETH_TX_GAS_PRICE_MULTIPLIER = self.get_config_value("ETH_TX_GAS_PRICE_MULTIPLIER", 1.2)
        self.MODEL_RETRAINING_INTERVAL = self.get_config_value("MODEL_RETRAINING_INTERVAL", 3600)
        self.MIN_TRAINING_SAMPLES = self.get_config_value("MIN_TRAINING_SAMPLES", 100)
        self.MODEL_ACCURACY_THRESHOLD = self.get_config_value("MODEL_ACCURACY_THRESHOLD", 0.7)
        self.PREDICTION_CACHE_TTL = self.get_config_value("PREDICTION_CACHE_TTL", 300)
        self.LINEAR_REGRESSION_PATH = str(self.BASE_PATH / "linear_regression")
        self.MODEL_PATH = str(self.BASE_PATH / "linear_regression" / "price_model.joblib")
        self.TRAINING_DATA_PATH = str(self.BASE_PATH / "linear_regression" / "training_data.csv")
        self.VOLATILITY_THRESHOLD = self.get_config_value("VOLATILITY_THRESHOLD", 0.05)
        self.LIQUIDITY_THRESHOLD = self.get_config_value("LIQUIDITY_THRESHOLD", 100000)
        self.MEMPOOL_MAX_RETRIES = self.get_config_value("MEMPOOL_MAX_RETRIES", 3)
        self.MEMPOOL_RETRY_DELAY = self.get_config_value("MEMPOOL_RETRY_DELAY", 2)
        self.MEMPOOL_BATCH_SIZE = self.get_config_value("MEMPOOL_BATCH_SIZE", 10)
        self.MEMPOOL_MAX_PARALLEL_TASKS = self.get_config_value("MEMPOOL_MAX_PARALLEL_TASKS", 5)
        self.NONCE_CACHE_TTL = self.get_config_value("NONCE_CACHE_TTL", 60)
        self.NONCE_RETRY_DELAY = self.get_config_value("NONCE_RETRY_DELAY", 1)
        self.NONCE_MAX_RETRIES = self.get_config_value("NONCE_MAX_RETRIES", 5)
        self.NONCE_TRANSACTION_TIMEOUT = self.get_config_value("NONCE_TRANSACTION_TIMEOUT", 120)
        self.SAFETYNET_CACHE_TTL = self.get_config_value("SAFETYNET_CACHE_TTL", 300)
        self.SAFETYNET_GAS_PRICE_TTL = self.get_config_value("SAFETYNET_GAS_PRICE_TTL", 30)
        self.AGGRESSIVE_FRONT_RUN_MIN_VALUE_ETH = self.get_config_value("AGGRESSIVE_FRONT_RUN_MIN_VALUE_ETH", 0.1)
        self.AGGRESSIVE_FRONT_RUN_RISK_SCORE_THRESHOLD = self.get_config_value("AGGRESSIVE_FRONT_RUN_RISK_SCORE_THRESHOLD", 0.7)
        self.FRONT_RUN_OPPORTUNITY_SCORE_THRESHOLD = self.get_config_value("FRONT_RUN_OPPORTUNITY_SCORE_THRESHOLD", 75)
        self.VOLATILITY_FRONT_RUN_SCORE_THRESHOLD = self.get_config_value("VOLATILITY_FRONT_RUN_SCORE_THRESHOLD", 75)
        self.ADVANCED_FRONT_RUN_RISK_SCORE_THRESHOLD = self.get_config_value("ADVANCED_FRONT_RUN_RISK_SCORE_THRESHOLD", 75)
        self.PRICE_DIP_BACK_RUN_THRESHOLD = self.get_config_value("PRICE_DIP_BACK_RUN_THRESHOLD", 0.99)
        self.FLASHLOAN_BACK_RUN_PROFIT_PERCENTAGE = self.get_config_value("FLASHLOAN_BACK_RUN_PROFIT_PERCENTAGE", 0.02)
        self.HIGH_VOLUME_BACK_RUN_DEFAULT_THRESHOLD_USD = self.get_config_value("HIGH_VOLUME_BACK_RUN_DEFAULT_THRESHOLD_USD", 100000)
        self.SANDWICH_ATTACK_GAS_PRICE_THRESHOLD_GWEI = self.get_config_value("SANDWICH_ATTACK_GAS_PRICE_THRESHOLD_GWEI", 200)
        self.PRICE_BOOST_SANDWICH_MOMENTUM_THRESHOLD = self.get_config_value("PRICE_BOOST_SANDWICH_MOMENTUM_THRESHOLD", 0.02)
        self.HIGH_VALUE_THRESHOLD = self.get_config_value("HIGH_VALUE_THRESHOLD", 1000000000000000000)
        self.WEB3_MAX_RETRIES = self.get_config_value("WEB3_MAX_RETRIES", 3)
        self.WEB3_RETRY_DELAY = self.get_config_value("WEB3_RETRY_DELAY", 2)

    def _validate_address(self, addr: str, name: str) -> str:
        if not addr:
            return ""
        if not addr.startswith("0x") or len(addr) != 42:
            raise ValueError(f"{name} invalid")
        return addr if is_checksum_address(addr) else to_checksum_address(addr)

    def _resolve_path(self, path_str: str, name: str) -> str:
        if not path_str:
            return ""
        full = self.BASE_PATH / path_str
        if not full.exists():
            raise FileNotFoundError(f"{name} not found: {full}")
        return str(full)

    async def _load_json(self, path: str) -> Any:
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            return json.loads(await f.read())
    
    async def get_token_addresses(self) -> Dict[str, str]:
        if not hasattr(self, "_token_addresses"):
            self._token_addresses = await self._load_json(self.TOKEN_ADDRESSES)
        return self._token_addresses
    
    async def get_token_symbols(self) -> Dict[str, str]:
        if not hasattr(self, "_token_symbols"):
            self._token_symbols = await self._load_json(self.TOKEN_SYMBOLS)
        return self._token_symbols
    
    async def get_erc20_abi(self) -> Dict[str, Any]:
        if not hasattr(self, "_erc20_abi"):
            self._erc20_abi = await self._load_json(self.ERC20_ABI)
        return self._erc20_abi
    
    async def get_erc20_signatures(self) -> Dict[str, str]:
        if not hasattr(self, "_erc20_signatures"):
            self._erc20_signatures = await self._load_json(self.ERC20_SIGNATURES)
        return self._erc20_signatures
    
    async def _load_json_safe(self, path: str, name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Safely load a JSON file. Logs errors if the file is missing or invalid.
        """
        try:
            return await self._load_json(path)
        except FileNotFoundError:
            logger.error(f"File not found: {path} ({name})" if name else f"File not found: {path}")
            return None
        except json.JSONDecodeError:
            logger.error(f"JSON decode error for file: {path} ({name})" if name else f"JSON decode error for file: {path}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading JSON from {path} ({name}): {e}" if name else f"Unexpected error loading JSON from {path}: {e}")
            return None
        
    async def load(self) -> None:
        await asyncio.to_thread(lambda: os.makedirs(self.BASE_PATH / "linear_regression", exist_ok=True))
        await self._load_json(self.ERC20_ABI)
        await self._load_json(self.TOKEN_ADDRESSES)
        await self._load_json(self.TOKEN_SYMBOLS)
        await self._load_json(self.ERC20_SIGNATURES)

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a configuration value by its key.
        If the key is not found, return the provided default value.
        """
        return self.config_data.get(self.environment, {}).get(key, default)

    def _load_config_file(self) -> Dict[str, Any]:
        """
        Load configuration settings from a YAML file.
        """
        try:
            with open(self.config_file, "r") as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_file}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {self.config_file} - {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error loading configuration file: {self.config_file} - {e}")
            return {}
