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

logger = setup_logging("Configuration", level="DEBUG")  

class Configuration:
    def __init__(self, env_path: str = ".env") -> None:
        dotenv.load_dotenv(dotenv_path=env_path)
        self.BASE_PATH = Path(__file__).parent.parent
        self.MAX_GAS_PRICE = int(os.getenv("MAX_GAS_PRICE", "100000000000"))
        self.GAS_LIMIT = int(os.getenv("GAS_LIMIT", "1000000"))
        self.MAX_SLIPPAGE = float(os.getenv("MAX_SLIPPAGE", "0.01"))
        self.MIN_PROFIT = float(os.getenv("MIN_PROFIT", "0.001"))
        self.MIN_BALANCE = float(os.getenv("MIN_BALANCE", "0.000001"))
        self.MEMORY_CHECK_INTERVAL = int(os.getenv("MEMORY_CHECK_INTERVAL", "300"))
        self.COMPONENT_HEALTH_CHECK_INTERVAL = int(os.getenv("COMPONENT_HEALTH_CHECK_INTERVAL", "60"))
        self.PROFITABLE_TX_PROCESS_TIMEOUT = float(os.getenv("PROFITABLE_TX_PROCESS_TIMEOUT", "1.0"))
        self.WETH_ADDRESS = self._validate_address(os.getenv("WETH_ADDRESS", ""), "WETH_ADDRESS")
        self.USDC_ADDRESS = self._validate_address(os.getenv("USDC_ADDRESS", ""), "USDC_ADDRESS")
        self.USDT_ADDRESS = self._validate_address(os.getenv("USDT_ADDRESS", ""), "USDT_ADDRESS")
        self.ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY", "")
        self.INFURA_PROJECT_ID = os.getenv("INFURA_PROJECT_ID", "")
        self.INFURA_API_KEY = os.getenv("INFURA_API_KEY", "")
        self.COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", "")
        self.COINMARKETCAP_API_KEY = os.getenv("COINMARKETCAP_API_KEY", "")
        self.CRYPTOCOMPARE_API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY", "")
        self.HTTP_ENDPOINT = os.getenv("HTTP_ENDPOINT", "")
        self.WEBSOCKET_ENDPOINT = os.getenv("WEBSOCKET_ENDPOINT", "")
        self.IPC_ENDPOINT = os.getenv("IPC_ENDPOINT", "")
        self.WALLET_ADDRESS = self._validate_address(os.getenv("WALLET_ADDRESS", ""), "WALLET_ADDRESS")
        self.WALLET_KEY = os.getenv("WALLET_KEY", "")
        self.ERC20_ABI = self._resolve_path(os.getenv("ERC20_ABI", ""), "ERC20_ABI")
        self.AAVE_FLASHLOAN_ABI = self._resolve_path(os.getenv("AAVE_FLASHLOAN_ABI", ""), "AAVE_FLASHLOAN_ABI")
        self.AAVE_POOL_ABI = self._resolve_path(os.getenv("AAVE_POOL_ABI", ""), "AAVE_POOL_ABI")
        self.UNISWAP_ABI = self._resolve_path(os.getenv("UNISWAP_ABI", ""), "UNISWAP_ABI")
        self.SUSHISWAP_ABI = self._resolve_path(os.getenv("SUSHISWAP_ABI", ""), "SUSHISWAP_ABI")
        self.ERC20_SIGNATURES = self._resolve_path(os.getenv("ERC20_SIGNATURES", ""), "ERC20_SIGNATURES")
        self.TOKEN_ADDRESSES = self._resolve_path(os.getenv("TOKEN_ADDRESSES", ""), "TOKEN_ADDRESSES")
        self.TOKEN_SYMBOLS = self._resolve_path(os.getenv("TOKEN_SYMBOLS", ""), "TOKEN_SYMBOLS")
        self.GAS_PRICE_ORACLE_ABI = self._resolve_path(os.getenv("GAS_PRICE_ORACLE_ABI", ""), "GAS_PRICE_ORACLE_ABI")
        self.UNISWAP_ADDRESS = self._validate_address(os.getenv("UNISWAP_ADDRESS", ""), "UNISWAP_ADDRESS")
        self.SUSHISWAP_ADDRESS = self._validate_address(os.getenv("SUSHISWAP_ADDRESS", ""), "SUSHISWAP_ADDRESS")
        self.AAVE_POOL_ADDRESS = self._validate_address(os.getenv("AAVE_POOL_ADDRESS", ""), "AAVE_POOL_ADDRESS")
        self.AAVE_FLASHLOAN_ADDRESS = self._validate_address(os.getenv("AAVE_FLASHLOAN_ADDRESS", ""), "AAVE_FLASHLOAN_ADDRESS")
        self.GAS_PRICE_ORACLE_ADDRESS = self._validate_address(os.getenv("GAS_PRICE_ORACLE_ADDRESS", ""), "GAS_PRICE_ORACLE_ADDRESS")
        self.SLIPPAGE_DEFAULT = float(os.getenv("SLIPPAGE_DEFAULT", "0.1"))
        self.MIN_SLIPPAGE = float(os.getenv("MIN_SLIPPAGE", "0.01"))
        self.MAX_SLIPPAGE = float(os.getenv("MAX_SLIPPAGE", "0.5"))
        self.SLIPPAGE_HIGH_CONGESTION = float(os.getenv("SLIPPAGE_HIGH_CONGESTION", "0.05"))
        self.SLIPPAGE_LOW_CONGESTION = float(os.getenv("SLIPPAGE_LOW_CONGESTION", "0.2"))
        self.MAX_GAS_PRICE_GWEI = int(os.getenv("MAX_GAS_PRICE_GWEI", "500"))
        self.MIN_PROFIT_MULTIPLIER = float(os.getenv("MIN_PROFIT_MULTIPLIER", "2.0"))
        self.BASE_GAS_LIMIT = int(os.getenv("BASE_GAS_LIMIT", "21000"))
        self.DEFAULT_CANCEL_GAS_PRICE_GWEI = int(os.getenv("DEFAULT_CANCEL_GAS_PRICE_GWEI", "60"))
        self.ETH_TX_GAS_PRICE_MULTIPLIER = float(os.getenv("ETH_TX_GAS_PRICE_MULTIPLIER", "1.2"))
        self.MODEL_RETRAINING_INTERVAL = int(os.getenv("MODEL_RETRAINING_INTERVAL", "3600"))
        self.MIN_TRAINING_SAMPLES = int(os.getenv("MIN_TRAINING_SAMPLES", "100"))
        self.MODEL_ACCURACY_THRESHOLD = float(os.getenv("MODEL_ACCURACY_THRESHOLD", "0.7"))
        self.PREDICTION_CACHE_TTL = int(os.getenv("PREDICTION_CACHE_TTL", "300"))
        self.LINEAR_REGRESSION_PATH = str(self.BASE_PATH / "linear_regression")
        self.MODEL_PATH = str(self.BASE_PATH / "linear_regression" / "price_model.joblib")
        self.TRAINING_DATA_PATH = str(self.BASE_PATH / "linear_regression" / "training_data.csv")
        self.VOLATILITY_THRESHOLD = float(os.getenv("VOLATILITY_THRESHOLD", "0.05"))
        self.LIQUIDITY_THRESHOLD = float(os.getenv("LIQUIDITY_THRESHOLD", "100000"))
        self.MEMPOOL_MAX_RETRIES = int(os.getenv("MEMPOOL_MAX_RETRIES", "3"))
        self.MEMPOOL_RETRY_DELAY = int(os.getenv("MEMPOOL_RETRY_DELAY", "2"))
        self.MEMPOOL_BATCH_SIZE = int(os.getenv("MEMPOOL_BATCH_SIZE", "10"))
        self.MEMPOOL_MAX_PARALLEL_TASKS = int(os.getenv("MEMPOOL_MAX_PARALLEL_TASKS", "5"))
        self.NONCE_CACHE_TTL = int(os.getenv("NONCE_CACHE_TTL", "60"))
        self.NONCE_RETRY_DELAY = int(os.getenv("NONCE_RETRY_DELAY", "1"))
        self.NONCE_MAX_RETRIES = int(os.getenv("NONCE_MAX_RETRIES", "5"))
        self.NONCE_TRANSACTION_TIMEOUT = int(os.getenv("NONCE_TRANSACTION_TIMEOUT", "120"))
        self.SAFETYNET_CACHE_TTL = int(os.getenv("SAFETYNET_CACHE_TTL", "300"))
        self.SAFETYNET_GAS_PRICE_TTL = int(os.getenv("SAFETYNET_GAS_PRICE_TTL", "30"))
        self.AGGRESSIVE_FRONT_RUN_MIN_VALUE_ETH = float(os.getenv("AGGRESSIVE_FRONT_RUN_MIN_VALUE_ETH", "0.1"))
        self.AGGRESSIVE_FRONT_RUN_RISK_SCORE_THRESHOLD = float(os.getenv("AGGRESSIVE_FRONT_RUN_RISK_SCORE_THRESHOLD", "0.7"))
        self.FRONT_RUN_OPPORTUNITY_SCORE_THRESHOLD = int(os.getenv("FRONT_RUN_OPPORTUNITY_SCORE_THRESHOLD", "75"))
        self.VOLATILITY_FRONT_RUN_SCORE_THRESHOLD = int(os.getenv("VOLATILITY_FRONT_RUN_SCORE_THRESHOLD", "75"))
        self.ADVANCED_FRONT_RUN_RISK_SCORE_THRESHOLD = int(os.getenv("ADVANCED_FRONT_RUN_RISK_SCORE_THRESHOLD", "75"))
        self.PRICE_DIP_BACK_RUN_THRESHOLD = float(os.getenv("PRICE_DIP_BACK_RUN_THRESHOLD", "0.99"))
        self.FLASHLOAN_BACK_RUN_PROFIT_PERCENTAGE = float(os.getenv("FLASHLOAN_BACK_RUN_PROFIT_PERCENTAGE", "0.02"))
        self.HIGH_VOLUME_BACK_RUN_DEFAULT_THRESHOLD_USD = float(os.getenv("HIGH_VOLUME_BACK_RUN_DEFAULT_THRESHOLD_USD", "100000"))
        self.SANDWICH_ATTACK_GAS_PRICE_THRESHOLD_GWEI = int(os.getenv("SANDWICH_ATTACK_GAS_PRICE_THRESHOLD_GWEI", "200"))
        self.PRICE_BOOST_SANDWICH_MOMENTUM_THRESHOLD = float(os.getenv("PRICE_BOOST_SANDWICH_MOMENTUM_THRESHOLD", "0.02"))
        self.HIGH_VALUE_THRESHOLD = int(os.getenv("HIGH_VALUE_THRESHOLD", "1000000000000000000"))
        self.WEB3_MAX_RETRIES = int(os.getenv("WEB3_MAX_RETRIES", "3"))
        self.WEB3_RETRY_DELAY = int(os.getenv("WEB3_RETRY_DELAY", "2"))

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
        return getattr(self, key, default)
