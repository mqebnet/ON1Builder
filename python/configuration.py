# File: python/configuration.py
"""
ON1Builder · Configuration
==========================

Loads all environment variables (.env), resolves paths, validates Ethereum
addresses and exposes **async helpers** for reading token-lists & signatures.

Key improvements
----------------
* `get_token_addresses()` now returns the **addresses**, not the dict keys.
* Added `ConfigError` for clean upstream handling.
* All path helpers accept **absolute** OR project-relative paths.
* One-time path resolution – no duplicate work in `_load_critical_abis()`.
* Async-safe JSON loader with descriptive errors.
* Centralised numeric parsing incl. support for hex (`0x…`) & underscores.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiofiles
import dotenv
from eth_utils import is_checksum_address, to_checksum_address

from loggingconfig import setup_logging

logger = setup_logging("Configuration", level=20)


# --------------------------------------------------------------------------- #
#                                EXCEPTIONS                                   #
# --------------------------------------------------------------------------- #

class ConfigError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
#                                CLASS                                         #
# --------------------------------------------------------------------------- #

class Configuration:
    """Load .env (or ENV) → strongly-typed attributes + utility coroutines."""

    # regex to strip `# comment` and `_` in numbers
    _NUM_RE = re.compile(r"[_,]")

    def __init__(self, env_path: str | Path | None = None) -> None:
        self.BASE_PATH = Path(__file__).resolve().parent.parent
        self.env_path = Path(env_path or ".env")

        self._load_env()
        self._bootstrap_constants()
        # defer heavy stuff (ABI path validation etc.) until `.load()` async

    # ====================================================================== #
    #                          INITIALISATION HELPERS                         #
    # ====================================================================== #

    def _load_env(self) -> None:
        if self.env_path.exists():
            dotenv.load_dotenv(self.env_path)
            logger.debug("dotenv loaded from %s", self.env_path)
        else:
            logger.debug("dotenv file %s not found – relying on OS env", self.env_path)

    # ------------------------------------------------------------------ #
    #                     primitive env-parsers                           #
    # ------------------------------------------------------------------ #

    def _num(self, raw: str | None, default: str | int | float) -> str | int | float:
        if raw in (None, ""):
            return default
        # allow hex (`0x…`) for gas limits
        raw = self._NUM_RE.sub("", raw)
        return int(raw, 0) if raw.startswith(("0x", "0X")) else float(raw) if "." in raw else int(raw)

    def _env(self, key: str, default: Any = "", required: bool = False) -> str:
        val = os.getenv(key, default)
        if required and val in ("", None):
            raise ConfigError(f"Missing required env var {key}")
        return str(val)

    # ------------------------------------------------------------------ #
    #                       bootstrap constants                           #
    # ------------------------------------------------------------------ #

    def _bootstrap_constants(self) -> None:
        # --- Gas / profit thresholds ---------------------------------- #
        self.MAX_GAS_PRICE: int = int(self._num(self._env("MAX_GAS_PRICE", "100000000000"), 100_000_000_000))
        self.GAS_LIMIT: int = int(self._num(self._env("GAS_LIMIT", "1000000"), 1_000_000))
        self.MIN_PROFIT: float = float(self._num(self._env("MIN_PROFIT", "0.001"), 0.001))
        self.MIN_BALANCE: float = float(self._num(self._env("MIN_BALANCE", "0.000001"), 1e-6))

        # --- Timings ---------------------------------------------------- #
        self.MEMORY_CHECK_INTERVAL = int(self._num(self._env("MEMORY_CHECK_INTERVAL", "300"), 300))
        self.COMPONENT_HEALTH_CHECK_INTERVAL = int(
            self._num(self._env("COMPONENT_HEALTH_CHECK_INTERVAL", "60"), 60)
        )
        self.PROFITABLE_TX_PROCESS_TIMEOUT = float(
            self._num(self._env("PROFITABLE_TX_PROCESS_TIMEOUT", "1.0"), 1.0)
        )

        # --- Addresses --------------------------------------------------- #
        self.WALLET_ADDRESS = self._eth_addr("WALLET_ADDRESS", required=True)
        self.UNISWAP_ADDRESS = self._eth_addr("UNISWAP_ADDRESS", default="0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D")
        self.SUSHISWAP_ADDRESS = self._eth_addr("SUSHISWAP_ADDRESS", default="0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F")
        self.AAVE_POOL_ADDRESS = self._eth_addr("AAVE_POOL_ADDRESS", required=True)
        self.AAVE_FLASHLOAN_ADDRESS = self._eth_addr("AAVE_FLASHLOAN_ADDRESS", required=True)
        self.GAS_PRICE_ORACLE_ADDRESS = self._eth_addr("GAS_PRICE_ORACLE_ADDRESS", required=True)

        # --- API Keys ---------------------------------------------------- #
        self.ETHERSCAN_API_KEY = self._env("ETHERSCAN_API_KEY", required=True)
        self.INFURA_PROJECT_ID = self._env("INFURA_PROJECT_ID", required=True)
        self.INFURA_API_KEY = self._env("INFURA_API_KEY", required=True)
        self.COINGECKO_API_KEY = self._env("COINGECKO_API_KEY", "")
        self.COINMARKETCAP_API_KEY = self._env("COINMARKETCAP_API_KEY", "")
        self.CRYPTOCOMPARE_API_KEY = self._env("CRYPTOCOMPARE_API_KEY", "")

        # --- RPC Endpoints ---------------------------------------------- #
        self.HTTP_ENDPOINT = self._env("HTTP_ENDPOINT", required=True)
        self.WEBSOCKET_ENDPOINT = self._env("WEBSOCKET_ENDPOINT", "")
        self.IPC_ENDPOINT = self._env("IPC_ENDPOINT", "")

        # --- Static JSON / ABI paths ------------------------------------ #
        self.ERC20_ABI = self._path("ERC20_ABI", "abi/erc20_abi.json")
        self.AAVE_FLASHLOAN_ABI = self._path("AAVE_FLASHLOAN_ABI", "abi/aave_flashloan_abi.json")
        self.AAVE_POOL_ABI = self._path("AAVE_POOL_ABI", "abi/aave_pool_abi.json")
        self.UNISWAP_ABI = self._path("UNISWAP_ABI", "abi/uniswap_abi.json")
        self.SUSHISWAP_ABI = self._path("SUSHISWAP_ABI", "abi/sushiswap_abi.json")
        self.ERC20_SIGNATURES = self._path("ERC20_SIGNATURES", "utils/erc20_signatures.json")
        self.TOKEN_ADDRESSES = self._path("TOKEN_ADDRESSES", "utils/token_addresses.json")
        self.TOKEN_SYMBOLS = self._path("TOKEN_SYMBOLS", "utils/token_symbols.json")
        self.GAS_PRICE_ORACLE_ABI = self._path("GAS_PRICE_ORACLE_ABI", "abi/gas_price_oracle_abi.json")

        # --- ML ---------------------------------------------------------- #
        self.LINEAR_REGRESSION_PATH = str(self.BASE_PATH / "linear_regression")
        self.MODEL_PATH = str(Path(self.LINEAR_REGRESSION_PATH) / "price_model.joblib")
        self.TRAINING_DATA_PATH = str(Path(self.LINEAR_REGRESSION_PATH) / "training_data.csv")

        # --- Gas & Core Limits ---
        self.MAX_GAS_PRICE_WEI = int(self._num(self._env("MAX_GAS_PRICE_WEI", "100000000000"), 100_000_000_000))
        self.MAX_GAS_PRICE_GWEI = int(self._num(self._env("MAX_GAS_PRICE_GWEI", "500"), 500))
        self.GAS_LIMIT = int(self._num(self._env("GAS_LIMIT", "1000000"), 1_000_000))
        
        # --- Slippage Settings ---
        self.MAX_SLIPPAGE = float(self._num(self._env("MAX_SLIPPAGE", "0.05"), 0.05))
        self.MIN_SLIPPAGE = float(self._num(self._env("MIN_SLIPPAGE", "0.01"), 0.01))
        self.SLIPPAGE_DEFAULT = float(self._num(self._env("SLIPPAGE_DEFAULT", "0.05"), 0.05))
        self.SLIPPAGE_HIGH_CONGESTION = float(self._num(self._env("SLIPPAGE_HIGH_CONGESTION", "0.05"), 0.05))
        self.SLIPPAGE_LOW_CONGESTION = float(self._num(self._env("SLIPPAGE_LOW_CONGESTION", "0.02"), 0.02))
        
        # --- Gas and Profit Settings ---
        self.MIN_PROFIT_MULTIPLIER = float(self._num(self._env("MIN_PROFIT_MULTIPLIER", "2.0"), 2.0))
        self.BASE_GAS_LIMIT = int(self._num(self._env("BASE_GAS_LIMIT", "21000"), 21_000))
        self.DEFAULT_CANCEL_GAS_PRICE_GWEI = int(self._num(self._env("DEFAULT_CANCEL_GAS_PRICE_GWEI", "60"), 60))
        self.ETH_TX_GAS_PRICE_MULTIPLIER = float(self._num(self._env("ETH_TX_GAS_PRICE_MULTIPLIER", "1.2"), 1.2))
        
        # --- Model Settings ---
        self.MODEL_RETRAINING_INTERVAL = int(self._num(self._env("MODEL_RETRAINING_INTERVAL", "3600"), 3600))
        self.MIN_TRAINING_SAMPLES = int(self._num(self._env("MIN_TRAINING_SAMPLES", "100"), 100))
        self.MODEL_ACCURACY_THRESHOLD = float(self._num(self._env("MODEL_ACCURACY_THRESHOLD", "0.7"), 0.7))
        self.PREDICTION_CACHE_TTL = int(self._num(self._env("PREDICTION_CACHE_TTL", "300"), 300))
        
        # --- Strategy Settings ---
        self.AGGRESSIVE_FRONT_RUN_MIN_VALUE_ETH = float(self._num(self._env("AGGRESSIVE_FRONT_RUN_MIN_VALUE_ETH", "0.1"), 0.1))
        self.AGGRESSIVE_FRONT_RUN_RISK_SCORE_THRESHOLD = float(self._num(self._env("AGGRESSIVE_FRONT_RUN_RISK_SCORE_THRESHOLD", "0.7"), 0.7))
        self.FRONT_RUN_OPPORTUNITY_SCORE_THRESHOLD = int(self._num(self._env("FRONT_RUN_OPPORTUNITY_SCORE_THRESHOLD", "75"), 75))
        self.VOLATILITY_FRONT_RUN_SCORE_THRESHOLD = int(self._num(self._env("VOLATILITY_FRONT_RUN_SCORE_THRESHOLD", "75"), 75))
        self.ADVANCED_FRONT_RUN_RISK_SCORE_THRESHOLD = int(self._num(self._env("ADVANCED_FRONT_RUN_RISK_SCORE_THRESHOLD", "75"), 75))
        self.PRICE_DIP_BACK_RUN_THRESHOLD = float(self._num(self._env("PRICE_DIP_BACK_RUN_THRESHOLD", "0.99"), 0.99))
        self.FLASHLOAN_BACK_RUN_PROFIT_PERCENTAGE = float(self._num(self._env("FLASHLOAN_BACK_RUN_PROFIT_PERCENTAGE", "0.02"), 0.02))
        self.HIGH_VOLUME_BACK_RUN_DEFAULT_THRESHOLD_USD = int(self._num(self._env("HIGH_VOLUME_BACK_RUN_DEFAULT_THRESHOLD_USD", "100000"), 100000))
        self.SANDWICH_ATTACK_GAS_PRICE_THRESHOLD_GWEI = int(self._num(self._env("SANDWICH_ATTACK_GAS_PRICE_THRESHOLD_GWEI", "200"), 200))
        self.PRICE_BOOST_SANDWICH_MOMENTUM_THRESHOLD = float(self._num(self._env("PRICE_BOOST_SANDWICH_MOMENTUM_THRESHOLD", "0.02"), 0.02))
        
        # --- Mempool Settings ---
        self.MEMPOOL_MAX_RETRIES = int(self._num(self._env("MEMPOOL_MAX_RETRIES", "3"), 3))
        self.MEMPOOL_RETRY_DELAY = int(self._num(self._env("MEMPOOL_RETRY_DELAY", "2"), 2))
        self.MEMPOOL_BATCH_SIZE = int(self._num(self._env("MEMPOOL_BATCH_SIZE", "10"), 10))
        self.MEMPOOL_MAX_PARALLEL_TASKS = int(self._num(self._env("MEMPOOL_MAX_PARALLEL_TASKS", "5"), 5))
        
        # --- Safety Net Settings ---
        self.SAFETYNET_CACHE_TTL = int(self._num(self._env("SAFETYNET_CACHE_TTL", "300"), 300))
        self.SAFETYNET_GAS_PRICE_TTL = int(self._num(self._env("SAFETYNET_GAS_PRICE_TTL", "30"), 30))
        
        # --- Nonce Settings ---
        self.NONCE_CACHE_TTL = int(self._num(self._env("NONCE_CACHE_TTL", "60"), 60))
        self.NONCE_RETRY_DELAY = int(self._num(self._env("NONCE_RETRY_DELAY", "1"), 1))
        self.NONCE_MAX_RETRIES = int(self._num(self._env("NONCE_MAX_RETRIES", "5"), 5))
        self.NONCE_TRANSACTION_TIMEOUT = int(self._num(self._env("NONCE_TRANSACTION_TIMEOUT", "120"), 120))
        
        # --- High Value Threshold ---
        self.HIGH_VALUE_THRESHOLD = int(self._num(self._env("HIGH_VALUE_THRESHOLD", "1000000000000000000"), 1000000000000000000))

    # ------------------------------------------------------------------ #
    #                       VALIDATION HELPERS                            #
    # ------------------------------------------------------------------ #

    def _eth_addr(self, key: str, *, default: str = "", required: bool = False) -> str:
        raw = self._env(key, default, required)
        if raw == "":
            return ""  # optional address left blank
        if not is_checksum_address(raw):
            try:
                raw = to_checksum_address(raw)
            except ValueError as exc:
                raise ConfigError(f"{key} invalid address") from exc
        return raw

    def _path(self, key: str, fallback: str) -> Path:
        raw = self._env(key, fallback)
        p = Path(raw) if Path(raw).is_absolute() else (self.BASE_PATH / raw)
        if not p.exists():
            raise ConfigError(f"{key} path does not exist: {p}")
        return p

    # ====================================================================== #
    #                             ASYNC HELPERS                               #
    # ====================================================================== #

    async def _read_json(self, pth: Path, descr: str) -> Any:
        try:
            async with aiofiles.open(pth, "r", encoding="utf-8") as f:
                txt = await f.read()
            return json.loads(txt)
        except FileNotFoundError as exc:
            raise ConfigError(f"{descr} file not found: {pth}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{descr} invalid JSON") from exc

    async def get_token_addresses(self) -> List[str]:
        data = await self._read_json(self.TOKEN_ADDRESSES, "token_addresses")
        if not isinstance(data, dict):
            raise ConfigError("token_addresses JSON must be object")
        # return **addresses** in lower-case
        return [addr.lower() for addr in data.values()]

    async def get_token_symbols(self) -> Dict[str, str]:
        data = await self._read_json(self.TOKEN_SYMBOLS, "token_symbols")
        if not isinstance(data, dict):
            raise ConfigError("token_symbols JSON must be object")
        return data

    async def get_erc20_signatures(self) -> Dict[str, str]:
        data = await self._read_json(self.ERC20_SIGNATURES, "erc20_signatures")
        if not isinstance(data, dict):
            raise ConfigError("erc20_signatures JSON must be object")
        return data

    # ====================================================================== #
    #                         ASYNC VALIDATION ROUTINES                       #
    # ====================================================================== #

    async def load(self) -> None:
        """Extra validation that requires I/O (e.g. ABI files present)."""
        # ensure training directory exists
        await asyncio.to_thread(Path(self.LINEAR_REGRESSION_PATH).mkdir, parents=True, exist_ok=True)

        abi_paths = [
            self.ERC20_ABI,
            self.UNISWAP_ABI,
            self.SUSHISWAP_ABI,
            self.AAVE_FLASHLOAN_ABI,
            self.AAVE_POOL_ABI,
        ]
        missing = [p for p in abi_paths if not p.exists()]
        if missing:
            raise ConfigError(f"Missing ABI files: {', '.join(str(m) for m in missing)}")

        logger.debug("Configuration fully validated.")

    # ------------------------------------------------------------------ #
    #                    convenience (non-async)                          #
    # ------------------------------------------------------------------ #

    def get_config_value(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __repr__(self) -> str:  # noqa: D401
        return f"<Configuration wallet={self.WALLET_ADDRESS[:10]}…>"
