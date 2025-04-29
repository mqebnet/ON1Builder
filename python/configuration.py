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
    Loads configuration from environment variables and JSON files.
    """
    def __init__(self, env_path: Optional[str] = None) -> None:
        self.env_path = env_path or ".env"
        self.BASE_PATH = Path(__file__).parent.parent
        self._load_env()
        self._initialize_defaults()

    def _load_env(self) -> None:
        dotenv.load_dotenv(dotenv_path=self.env_path)
        logger.debug(f"Loaded environment from: {self.env_path}")

    def _initialize_defaults(self) -> None:
        self.MAX_GAS_PRICE = self._get_env_int("MAX_GAS_PRICE", 100_000_000_000, "Maximum gas price")
        self.GAS_LIMIT = self._get_env_int("GAS_LIMIT", 1_000_000, "Default gas limit")
        self.MAX_SLIPPAGE = self._get_env_float("MAX_SLIPPAGE", 0.01, "Maximum slippage")
        self.MIN_PROFIT = self._get_env_float("MIN_PROFIT", 0.001, "Minimum profit in ETH")
        self.MIN_BALANCE = self._get_env_float("MIN_BALANCE", 0.000001, "Minimum balance in ETH")
        self.MEMORY_CHECK_INTERVAL = self._get_env_int("MEMORY_CHECK_INTERVAL", 300, "Memory check interval")
        self.COMPONENT_HEALTH_CHECK_INTERVAL = self._get_env_int("COMPONENT_HEALTH_CHECK_INTERVAL", 60, "Component health check")
        self.PROFITABLE_TX_PROCESS_TIMEOUT = self._get_env_float("PROFITABLE_TX_PROCESS_TIMEOUT", 1.0, "Transaction processing timeout")
        self.WETH_ADDRESS = self._get_env_str("WETH_ADDRESS", "0xC02aaa39b223FE8D0a0e5C4F27eAD9083C756Cc2", "WETH Address")
        self.USDC_ADDRESS = self._get_env_str("USDC_ADDRESS", "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", "USDC Address")
        self.USDT_ADDRESS = self._get_env_str("USDT_ADDRESS", "0xdAC17F958D2ee523a2206206994597C13D831ec7", "USDT Address")
        self.ETHERSCAN_API_KEY = self._get_env_str("ETHERSCAN_API_KEY", "", "Etherscan API Key", required=True)
        self.INFURA_PROJECT_ID = self._get_env_str("INFURA_PROJECT_ID", "", "Infura Project ID", required=True)
        self.INFURA_API_KEY = self._get_env_str("INFURA_API_KEY", "", "Infura API Key", required=True)
        self.COINGECKO_API_KEY = self._get_env_str("COINGECKO_API_KEY", "", "CoinGecko API Key", required=True)
        self.COINMARKETCAP_API_KEY = self._get_env_str("COINMARKETCAP_API_KEY", "", "CoinMarketCap API Key", required=True)
        self.CRYPTOCOMPARE_API_KEY = self._get_env_str("CRYPTOCOMPARE_API_KEY", "", "CryptoCompare API Key", required=True)
        self.HTTP_ENDPOINT = self._get_env_str("HTTP_ENDPOINT", "http://localhost:8545", "HTTP Provider URL", required=True)
        self.WEBSOCKET_ENDPOINT = self._get_env_str("WEBSOCKET_ENDPOINT", "ws://localhost:8546", "WebSocket Provider URL", required=False)
        self.IPC_ENDPOINT = self._get_env_str("IPC_ENDPOINT", "", "IPC Provider Path", required=False)
        self.WALLET_ADDRESS = self._get_env_str("WALLET_ADDRESS", "", "Wallet Address", required=True)
        self.WALLET_KEY = self._get_env_str("WALLET_KEY", "", "Wallet Private Key", required=True)
        self.ERC20_ABI = self._resolve_path("ERC20_ABI", "Path to ERC20 ABI")
        self.AAVE_FLASHLOAN_ABI = self._resolve_path("AAVE_FLASHLOAN_ABI", "Path to Aave Flashloan ABI")
        self.AAVE_POOL_ABI = self._resolve_path("AAVE_POOL_ABI", "Path to Aave Pool ABI")
        self.UNISWAP_ABI = self._resolve_path("UNISWAP_ABI", "Path to Uniswap ABI")
        self.SUSHISWAP_ABI = self._resolve_path("SUSHISWAP_ABI", "Path to Sushiswap ABI")
        self.ERC20_SIGNATURES = self._resolve_path("ERC20_SIGNATURES", "Path to ERC20 Signatures")
        self.TOKEN_ADDRESSES = self._resolve_path("TOKEN_ADDRESSES", "Path to Token Addresses")
        self.TOKEN_SYMBOLS = self._resolve_path("TOKEN_SYMBOLS", "Path to Token Symbols")
        self.GAS_PRICE_ORACLE_ABI = self._resolve_path("GAS_PRICE_ORACLE_ABI", "Path to Gas Price Oracle ABI")
        self.UNISWAP_ADDRESS = self._get_env_str("UNISWAP_ADDRESS", "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D", "Uniswap Router")
        self.SUSHISWAP_ADDRESS = self._get_env_str("SUSHISWAP_ADDRESS", "0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F", "Sushiswap Router")
        self.AAVE_POOL_ADDRESS = self._get_env_str("AAVE_POOL_ADDRESS", "0xb53c1a33016b2dc2ff3653530bff1848a515c8c5", "Aave Lending Pool", required=True)
        self.AAVE_FLASHLOAN_ADDRESS = self._get_env_str("AAVE_FLASHLOAN_ADDRESS", "", "Aave Flashloan Address", required=True)
        self.GAS_PRICE_ORACLE_ADDRESS = self._get_env_str("GAS_PRICE_ORACLE_ADDRESS", "", "Gas Price Oracle Address", required=True)
        self.SLIPPAGE_DEFAULT = self._get_env_float("SLIPPAGE_DEFAULT", 0.1, "Default slippage")
        self.MIN_SLIPPAGE = self._get_env_float("MIN_SLIPPAGE", 0.01, "Minimum slippage")
        self.MAX_SLIPPAGE = self._get_env_float("MAX_SLIPPAGE", 0.5, "Maximum slippage")
        self.SLIPPAGE_HIGH_CONGESTION = self._get_env_float("SLIPPAGE_HIGH_CONGESTION", 0.05, "High congestion slippage")
        self.SLIPPAGE_LOW_CONGESTION = self._get_env_float("SLIPPAGE_LOW_CONGESTION", 0.2, "Low congestion slippage")
        self.MAX_GAS_PRICE_GWEI = self._get_env_int("MAX_GAS_PRICE_GWEI", 500, "Max gas price (Gwei)")
        self.MIN_PROFIT_MULTIPLIER = self._get_env_float("MIN_PROFIT_MULTIPLIER", 2.0, "Min profit multiplier")
        self.BASE_GAS_LIMIT = self._get_env_int("BASE_GAS_LIMIT", 21000, "Base gas limit")
        self.DEFAULT_CANCEL_GAS_PRICE_GWEI = self._get_env_int("DEFAULT_CANCEL_GAS_PRICE_GWEI", 60, "Cancel gas price (Gwei)")
        self.ETH_TX_GAS_PRICE_MULTIPLIER = self._get_env_float("ETH_TX_GAS_PRICE_MULTIPLIER", 1.2, "ETH tx gas multiplier")
        self.MODEL_RETRAINING_INTERVAL = self._get_env_int("MODEL_RETRAINING_INTERVAL", 3600, "Model retraining interval")
        self.MIN_TRAINING_SAMPLES = self._get_env_int("MIN_TRAINING_SAMPLES", 100, "Minimum training samples")
        self.MODEL_ACCURACY_THRESHOLD = self._get_env_float("MODEL_ACCURACY_THRESHOLD", 0.7, "Model accuracy threshold")
        self.PREDICTION_CACHE_TTL = self._get_env_int("PREDICTION_CACHE_TTL", 300, "Prediction cache TTL")
        self.LINEAR_REGRESSION_PATH = str(self.BASE_PATH / "linear_regression")
        self.MODEL_PATH = str(self.BASE_PATH / "linear_regression" / "price_model.joblib")
        self.TRAINING_DATA_PATH = str(self.BASE_PATH / "linear_regression" / "training_data.csv")
        self.VOLATILITY_THRESHOLD = self._get_env_float("VOLATILITY_THRESHOLD", 0.05, "Volatility threshold")
        self.LIQUIDITY_THRESHOLD = self._get_env_float("LIQUIDITY_THRESHOLD", 100_000, "Liquidity threshold")
        self.MEMPOOL_MAX_RETRIES = self._get_env_int("MEMPOOL_MAX_RETRIES", 3, "Mempool max retries")
        self.MEMPOOL_RETRY_DELAY = self._get_env_int("MEMPOOL_RETRY_DELAY", 2, "Mempool retry delay")
        self.MEMPOOL_BATCH_SIZE = self._get_env_int("MEMPOOL_BATCH_SIZE", 10, "Mempool batch size")
        self.MEMPOOL_MAX_PARALLEL_TASKS = self._get_env_int("MEMPOOL_MAX_PARALLEL_TASKS", 5, "Mempool parallel tasks")
        self.NONCE_CACHE_TTL = self._get_env_int("NONCE_CACHE_TTL", 60, "Nonce cache TTL")
        self.NONCE_RETRY_DELAY = self._get_env_int("NONCE_RETRY_DELAY", 1, "Retry delay for nonce")
        self.NONCE_MAX_RETRIES = self._get_env_int("NONCE_MAX_RETRIES", 5, "Nonce max retries")
        self.NONCE_TRANSACTION_TIMEOUT = self._get_env_int("NONCE_TRANSACTION_TIMEOUT", 120, "Nonce transaction timeout")
        self.SAFETYNET_CACHE_TTL = self._get_env_int("SAFETYNET_CACHE_TTL", 300, "SafetyNet cache TTL")
        self.SAFETYNET_GAS_PRICE_TTL = self._get_env_int("SAFETYNET_GAS_PRICE_TTL", 30, "SafetyNet gas price TTL")
        self.AGGRESSIVE_FRONT_RUN_MIN_VALUE_ETH = self._get_env_float("AGGRESSIVE_FRONT_RUN_MIN_VALUE_ETH", 0.1, "Front-run min ETH value")
        self.AGGRESSIVE_FRONT_RUN_RISK_SCORE_THRESHOLD = self._get_env_float("AGGRESSIVE_FRONT_RUN_RISK_SCORE_THRESHOLD", 0.7, "Front-run risk threshold")
        self.FRONT_RUN_OPPORTUNITY_SCORE_THRESHOLD = self._get_env_int("FRONT_RUN_OPPORTUNITY_SCORE_THRESHOLD", 75, "Front-run opportunity score")
        self.VOLATILITY_FRONT_RUN_SCORE_THRESHOLD = self._get_env_int("VOLATILITY_FRONT_RUN_SCORE_THRESHOLD", 75, "Volatility front-run score")
        self.ADVANCED_FRONT_RUN_RISK_SCORE_THRESHOLD = self._get_env_int("ADVANCED_FRONT_RUN_RISK_SCORE_THRESHOLD", 75, "Advanced front-run risk score")
        self.PRICE_DIP_BACK_RUN_THRESHOLD = self._get_env_float("PRICE_DIP_BACK_RUN_THRESHOLD", 0.99, "Back-run price dip threshold")
        self.FLASHLOAN_BACK_RUN_PROFIT_PERCENTAGE = self._get_env_float("FLASHLOAN_BACK_RUN_PROFIT_PERCENTAGE", 0.02, "Flashloan back-run profit")
        self.HIGH_VOLUME_BACK_RUN_DEFAULT_THRESHOLD_USD = self._get_env_float("HIGH_VOLUME_BACK_RUN_DEFAULT_THRESHOLD_USD", 100000, "High volume back-run USD threshold")
        self.SANDWICH_ATTACK_GAS_PRICE_THRESHOLD_GWEI = self._get_env_int("SANDWICH_ATTACK_GAS_PRICE_THRESHOLD_GWEI", 200, "Sandwich attack gas threshold")
        self.PRICE_BOOST_SANDWICH_MOMENTUM_THRESHOLD = self._get_env_float("PRICE_BOOST_SANDWICH_MOMENTUM_THRESHOLD", 0.02, "Sandwich momentum threshold")
        self.HIGH_VALUE_THRESHOLD = self._get_env_int("HIGH_VALUE_THRESHOLD", 1_000_000_000_000_000_000, "High value tx threshold (Wei)")

    def _validate_ethereum_address(self, address: str, var_name: str) -> str:
        if not isinstance(address, str):
            raise ValueError(f"{var_name} must be a string.")
        address = address.strip()
        if not address.startswith("0x") or len(address) != 42:
            raise ValueError(f"Invalid {var_name}: must be a 42-character hex string starting with '0x'.")
        try:
            return address if is_checksum_address(address) else to_checksum_address(address)
        except Exception as e:
            raise ValueError(f"Invalid {var_name} address format: {e}")

    def _parse_numeric_string(self, value: str) -> str:
        return value.split("#")[0].strip().replace("_", "")

    def _get_env_str(self, var_name: str, default: Optional[str], description: str, required: bool = False) -> str:
        value = os.getenv(var_name, default)
        if (value is None or value == "") and required:
            logger.error(f"Missing required variable: {var_name}")
            raise ValueError(f"Missing {var_name}")
        if "ADDRESS" in var_name and not any(x in var_name.upper() for x in ["ABI", "PATH", "ADDRESSES", "SIGNATURES", "SYMBOLS"]) and value not in [default, ""]:
            try:
                value = self._validate_ethereum_address(value, var_name)
            except ValueError as e:
                logger.error(f"Invalid {var_name}: {e}")
                raise
        logger.debug(f"Loaded {var_name}: {value[:10] + '...' if len(value) > 10 else value}")
        return value

    def _get_env_int(self, var_name: str, default: Optional[int], description: str) -> int:
        try:
            value = os.getenv(var_name)
            if value is None or value == "":
                return default  # Use default if not set
            return int(self._parse_numeric_string(value))
        except ValueError as e:
            logger.error(f"Invalid integer for {var_name}: {e}")
            raise

    def _get_env_float(self, var_name: str, default: Optional[float], description: str) -> float:
        try:
            value = os.getenv(var_name)
            if value is None or value == "":
                return default
            return float(self._parse_numeric_string(value))
        except ValueError as e:
            logger.error(f"Invalid float for {var_name}: {e}")
            raise

    def _resolve_path(self, path_env_var: str, description: str) -> Path:
        path_str = self._get_env_str(path_env_var, None, description, required=True)
        full_path = self.BASE_PATH / path_str
        if not full_path.exists():
            logger.error(f"File not found: {full_path}")
            raise FileNotFoundError(f"File not found: {full_path}")
        logger.debug(f"Resolved path: {full_path}")
        return full_path

    async def _load_json_safe(self, file_path: Path, description: str) -> Any:
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()
            data = json.loads(content)
            logger.debug(f"Loaded {description} from {file_path}")
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing {description} file: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {description} file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading {description}: {e}")

    async def get_token_addresses(self) -> List[str]:
        data = await self._load_json_safe(self.TOKEN_ADDRESSES, "token addresses")
        if not isinstance(data, dict):
            logger.error("Token addresses must be a dictionary.")
            raise ValueError("Invalid format for token addresses.")
        addresses = list(data.keys())
        logger.info(f"Loaded {len(addresses)} token addresses.")
        return addresses

    async def get_token_symbols(self) -> Dict[str, str]:
        data = await self._load_json_safe(self.TOKEN_SYMBOLS, "token symbols")
        if not isinstance(data, dict):
            logger.error("Token symbols must be a dictionary.")
            raise ValueError("Invalid format for token symbols.")
        return data

    async def get_erc20_signatures(self) -> Dict[str, str]:
        data = await self._load_json_safe(self.ERC20_SIGNATURES, "ERC20 signatures")
        if not isinstance(data, dict):
            logger.error("ERC20 signatures must be a dictionary.")
            raise ValueError("Invalid format for ERC20 signatures.")
        return data

    def get_config_value(self, key: str, default: Any = None) -> Any:
        try:
            return getattr(self, key, default)
        except AttributeError:
            logger.warning(f"Configuration key '{key}' not found; using default.")
            return default

    def get_all_config_values(self) -> Dict[str, Any]:
        return vars(self)

    async def load(self, web3: Optional[Any] = None) -> None:
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
        required_dirs = [
            Path(self.LINEAR_REGRESSION_PATH),
            self.BASE_PATH / "abi",
            self.BASE_PATH / "utils"
        ]
        for directory in required_dirs:
            await asyncio.to_thread(os.makedirs, directory, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")

    async def _load_critical_abis(self) -> None:
        try:
            self.AAVE_FLASHLOAN_ABI_PATH = self._resolve_path("AAVE_FLASHLOAN_ABI", "AAVE Flashloan ABI")
            self.AAVE_POOL_ABI_PATH = self._resolve_path("AAVE_POOL_ABI", "AAVE Pool ABI")
            self.ERC20_ABI_PATH = self._resolve_path("ERC20_ABI", "ERC20 ABI")
            self.UNISWAP_ABI_PATH = self._resolve_path("UNISWAP_ABI", "Uniswap ABI")
            self.SUSHISWAP_ABI_PATH = self._resolve_path("SUSHISWAP_ABI", "Sushiswap ABI")
            self.GAS_PRICE_ORACLE_ABI_PATH = self._resolve_path("GAS_PRICE_ORACLE_ABI", "Gas Price Oracle ABI")
        except Exception as e:
            raise RuntimeError(f"Failed to load critical ABI paths: {e}")

    async def _validate_api_keys(self) -> None:
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
                    payload = {"jsonrpc": "2.0", "id": 1, "method": "web3_clientVersion", "params": []}
                    try:
                        async with session.post(test_url, json=payload, timeout=10) as response:
                            data = await response.json()
                            if "result" not in data:
                                invalid_keys.append(f"{key_name} (Invalid response)")
                            else:
                                logger.debug(f"{key_name} validated.")
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
                                logger.debug(f"{key_name} validated.")
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
                                logger.debug(f"{key_name} validated.")
                    except Exception as e:
                        invalid_keys.append(f"{key_name} (Error: {e})")
                    continue
                test_url = test_url_template.format(key=api_key)
                try:
                    async with session.get(test_url, timeout=10) as response:
                        if response.status != 200:
                            invalid_keys.append(f"{key_name} (Status: {response.status})")
                        else:
                            logger.debug(f"{key_name} validated.")
                except Exception as e:
                    invalid_keys.append(f"{key_name} (Error: {e})")
        if missing_keys:
            logger.warning(f"Missing API keys: {', '.join(missing_keys)}")
        if invalid_keys:
            logger.error(f"Invalid API keys: {', '.join(invalid_keys)}")

    def _validate_addresses(self) -> None:
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
