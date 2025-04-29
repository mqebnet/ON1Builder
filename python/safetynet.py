import asyncio
from decimal import Decimal
from typing import Any, Dict, Tuple, Optional

from cachetools import TTLCache
from web3 import AsyncWeb3
from eth_account import Account

from apiconfig import APIConfig
from configuration import Configuration
from marketmonitor import MarketMonitor

from loggingconfig import setup_logging
import logging

logger = setup_logging("SafetyNet", level=logging.DEBUG)


class SafetyNet:
    """
    Provides risk management and transaction validation for MEV operations.
    It handles balance caching, profit verification, gas estimation, slippage adjustment,
    network congestion monitoring, and risk assessment.
    """
    def __init__(
        self,
        web3: AsyncWeb3,
        configuration: Configuration,
        address: Optional[str] = None,
        account: Optional[Account] = None,
        apiconfig: Optional[APIConfig] = None,
        marketmonitor: Optional[MarketMonitor] = None,
    ) -> None:
        self.web3: AsyncWeb3 = web3
        self.configuration: Configuration = configuration
        self.address: Optional[str] = address
        self.account: Optional[Account] = account
        self.apiconfig: Optional[APIConfig] = apiconfig
        self.marketmonitor: Optional[MarketMonitor] = marketmonitor

        self.price_cache: TTLCache = TTLCache(maxsize=2000, ttl=self.configuration.SAFETYNET_CACHE_TTL)
        self.gas_price_cache: TTLCache = TTLCache(maxsize=1, ttl=self.configuration.SAFETYNET_GAS_PRICE_TTL)

        self.price_lock: asyncio.Lock = asyncio.Lock()

        logger.info("SafetyNet initialized and ready!")
        self.SLIPPAGE_CONFIG: Dict[str, float] = {
            "default": self.configuration.SLIPPAGE_DEFAULT,
            "min": self.configuration.MIN_SLIPPAGE,
            "max": self.configuration.MAX_SLIPPAGE,
            "high_congestion": self.configuration.SLIPPAGE_HIGH_CONGESTION,
            "low_congestion": self.configuration.SLIPPAGE_LOW_CONGESTION,
        }
        self.GAS_CONFIG: Dict[str, float] = {
            "max_gas_price_gwei": float(self.configuration.MAX_GAS_PRICE_GWEI),
            "min_profit_multiplier": self.configuration.MIN_PROFIT_MULTIPLIER,
            "base_gas_limit": self.configuration.BASE_GAS_LIMIT
        }

    async def initialize(self) -> None:
        """
        Verify web3 connectivity and initialize SafetyNet.
        Raises:
            RuntimeError: If the web3 instance is not properly connected.
        """
        try:
            if not self.web3:
                raise RuntimeError("Web3 instance is not provided in SafetyNet.")
            if not await self.web3.is_connected():
                raise RuntimeError("Web3 connection failed in SafetyNet.")
            logger.debug("SafetyNet successfully connected to the blockchain.")
        except Exception as e:
            logger.critical(f"SafetyNet initialization failed: {e}", exc_info=True)
            raise

    async def get_balance(self, account: Account) -> Decimal:
        """
        Retrieve the account balance in ETH with caching.
        """
        cache_key = f"balance_{account.address}"
        if cache_key in self.price_cache:
            logger.debug("Balance retrieved from cache.")
            return self.price_cache[cache_key]

        for attempt in range(3):
            try:
                balance_wei = await self.web3.eth.get_balance(account.address)
                balance_eth = Decimal(self.web3.from_wei(balance_wei, "ether"))
                self.price_cache[cache_key] = balance_eth
                logger.debug(f"Fetched balance: {balance_eth} ETH")
                return balance_eth
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed to fetch balance: {e}")
                await asyncio.sleep(2 ** attempt)
        logger.error("Failed to fetch balance after multiple retries.")
        return Decimal("0")

    async def ensure_profit(
        self,
        transaction_data: Dict[str, Any],
        minimum_profit_eth: Optional[float] = None,
    ) -> bool:
        """
        Confirm a transaction yields sufficient profit after gas and slippage adjustments.

        Args:
            transaction_data (Dict[str, Any]): Details including output token, amounts, gas price, gas used.
            minimum_profit_eth (Optional[float]): Minimum required profit in ETH; defaults to config.MIN_PROFIT.

        Returns:
            bool: True if the estimated profit exceeds the threshold, else False.
        """
        try:
            rt_price = await self.apiconfig.get_real_time_price(transaction_data["output_token"])
            if rt_price is None:
                logger.warning("Real-time price unavailable; cannot ensure profit.")
                return False

            gas_price = Decimal(transaction_data["gas_price"])
            gas_used = Decimal(transaction_data["gas_used"])
            gas_cost = (gas_price * gas_used * Decimal("1e-9")).quantize(Decimal("0.000000001"))

            slippage = await self.adjust_slippage_tolerance()
            profit = await self._calculate_profit(transaction_data, rt_price, slippage, gas_cost)
            self._log_profit_calculation(transaction_data, rt_price, gas_cost, profit, minimum_profit_eth or self.configuration.MIN_PROFIT)
            return profit > Decimal(minimum_profit_eth or self.configuration.MIN_PROFIT)
        except KeyError as e:
            logger.error(f"Missing required key in transaction data: {e}")
            return False
        except Exception as e:
            logger.error(f"Error in ensure_profit: {e}", exc_info=True)
            return False

    def _validate_gas_parameters(self, gas_price_gwei: Decimal, gas_used: Decimal) -> bool:
        """
        Validate that gas parameters are within safe thresholds.
        """
        if gas_used <= 0:
            logger.error("Gas used must be greater than zero.")
            return False
        if gas_price_gwei > Decimal(self.GAS_CONFIG["max_gas_price_gwei"]):
            logger.warning(f"Gas price {gas_price_gwei} Gwei exceeds allowed maximum.")
            return False
        return True

    def _calculate_gas_cost(self, gas_price_gwei: Decimal, gas_used: Decimal) -> Decimal:
        """
        Calculate total gas cost in ETH using a fixed conversion.
        """
        return (gas_price_gwei * gas_used * Decimal("1e-9")).quantize(Decimal("0.000000001"))

    async def _calculate_profit(
        self,
        transaction_data: Dict[str, Any],
        real_time_price: Decimal,
        slippage: float,
        gas_cost: Decimal,
    ) -> Decimal:
        """
        Calculate expected profit: Adjust expected output by slippage, then subtract input amount and gas cost.
        """
        try:
            expected_output = real_time_price * Decimal(transaction_data["amountOut"])
            input_amount = Decimal(transaction_data["amountIn"])
            adjusted_output = expected_output * (1 - Decimal(slippage))
            profit = adjusted_output - input_amount - gas_cost
            return profit.quantize(Decimal("0.000000001"))
        except Exception as e:
            logger.error(f"Error calculating profit: {e}", exc_info=True)
            return Decimal("0")

    def _log_profit_calculation(
        self,
        tx_data: Dict[str, Any],
        real_time_price: Decimal,
        gas_cost: Decimal,
        profit: Decimal,
        min_profit: float
    ) -> None:
        """
        Log details of profit calculation.
        """
        profitable = "Yes" if profit > Decimal(min_profit) else "No"
        logger.debug(
            f"Profit Calculation Summary:\n"
            f" Token: {tx_data.get('output_token')}\n"
            f" Real-time Price: {real_time_price:.6f} ETH\n"
            f" Input Amount: {Decimal(tx_data.get('amountIn')):.6f} ETH\n"
            f" Expected Output: {Decimal(tx_data.get('amountOut')):.6f}\n"
            f" Gas Cost: {gas_cost:.6f} ETH\n"
            f" Calculated Profit: {profit:.6f} ETH\n"
            f" Minimum Required Profit: {min_profit} ETH\n"
            f" Profitable: {profitable}"
        )

    async def estimate_gas(self, transaction_data: Dict[str, Any]) -> int:
        """
        Estimate gas for a transaction.
        """
        try:
            gas_estimate = await self.web3.eth.estimate_gas(transaction_data)
            return gas_estimate
        except Exception as e:
            logger.error(f"Gas estimation failed: {e}", exc_info=True)
            return self.GAS_CONFIG["base_gas_limit"]

    async def adjust_slippage_tolerance(self) -> float:
        """
        Adjust slippage based on network congestion.
        """
        try:
            congestion = await self.get_network_congestion()
            if congestion > 0.8:
                slippage = self.SLIPPAGE_CONFIG["high_congestion"]
            elif congestion < 0.2:
                slippage = self.SLIPPAGE_CONFIG["low_congestion"]
            else:
                slippage = self.SLIPPAGE_CONFIG["default"]
            # Ensure slippage falls between min and max.
            slippage = max(self.SLIPPAGE_CONFIG["min"], min(slippage, self.SLIPPAGE_CONFIG["max"]))
            logger.debug(f"Adjusted slippage to {slippage * 100:.2f}% based on network congestion {congestion*100:.2f}%")
            return slippage
        except Exception as e:
            logger.error(f"Error adjusting slippage tolerance: {e}", exc_info=True)
            return self.SLIPPAGE_CONFIG["default"]

    async def get_network_congestion(self) -> float:
        """
        Estimate network congestion based on the latest block's gas usage.
        """
        try:
            latest_block = await self.web3.eth.get_block("latest")
            gas_used = Decimal(latest_block["gasUsed"])
            gas_limit = Decimal(latest_block["gasLimit"])
            congestion = float((gas_used / gas_limit).quantize(Decimal("0.0001")))
            logger.debug(f"Network congestion: {congestion*100:.2f}%")
            return congestion
        except Exception as e:
            logger.error(f"Error fetching network congestion: {e}", exc_info=True)
            return 0.5

    async def check_transaction_safety(self, tx_data: Dict[str, Any], check_type: str = "all") -> Tuple[bool, Dict[str, Any]]:
        """
        Check transaction safety by validating gas price, estimated profit, and network congestion.
        Returns a tuple (is_safe, details).
        """
        try:
            messages = []
            gas_ok = True
            profit_ok = True
            congestion_ok = True

            if check_type in ("all", "gas"):
                current_gas = await self.get_dynamic_gas_price()
                if current_gas > Decimal(self.configuration.MAX_GAS_PRICE_GWEI):
                    gas_ok = False
                    messages.append(f"Gas price {current_gas} Gwei exceeds limit {self.configuration.MAX_GAS_PRICE_GWEI} Gwei.")

            if check_type in ("all", "profit"):
                current_price = await self.apiconfig.get_real_time_price(tx_data["output_token"])
                slippage = await self.adjust_slippage_tolerance()
                gas_cost = self._calculate_gas_cost(Decimal(tx_data["gas_price"]), Decimal(tx_data["gas_used"]))
                profit = await self._calculate_profit(tx_data, current_price, slippage, gas_cost)
                if profit < Decimal(self.configuration.MIN_PROFIT):
                    profit_ok = False
                    messages.append(f"Calculated profit {profit:.6f} ETH below threshold {self.configuration.MIN_PROFIT} ETH.")

            if check_type in ("all", "network"):
                congestion = await self.get_network_congestion()
                if congestion > 0.8:
                    congestion_ok = False
                    messages.append(f"Network congestion too high: {congestion*100:.2f}%.")

            is_safe = gas_ok and profit_ok and congestion_ok
            return is_safe, {
                "is_safe": is_safe,
                "gas_ok": gas_ok,
                "profit_ok": profit_ok,
                "congestion_ok": congestion_ok,
                "messages": messages
            }
        except Exception as e:
            logger.error(f"Error in transaction safety check: {e}", exc_info=True)
            return False, {"is_safe": False, "messages": [str(e)]}

    async def get_dynamic_gas_price(self) -> Decimal:
        """
        Fetch dynamic gas price in Gwei, using caching to minimize network calls.
        """
        if "gas_price" in self.gas_price_cache:
            return self.gas_price_cache["gas_price"]
        try:
            latest_block = await self.web3.eth.get_block("latest")
            base_fee = latest_block.get("baseFeePerGas")
            if base_fee:
                gas_price_wei = base_fee * 2
                gas_price_gwei = Decimal(str(self.web3.from_wei(gas_price_wei, "gwei")))
            else:
                raw_gas_price = await self.web3.eth.gas_price
                gas_price_gwei = Decimal(str(self.web3.from_wei(raw_gas_price, "gwei")))
            self.gas_price_cache["gas_price"] = gas_price_gwei
            logger.debug(f"Dynamic gas price fetched: {gas_price_gwei} Gwei")
            return gas_price_gwei
        except Exception as e:
            logger.error(f"Error fetching dynamic gas price: {e}", exc_info=True)
            return Decimal(self.configuration.MAX_GAS_PRICE_GWEI)

    async def validate_transaction(self, tx_data: Dict[str, Any]) -> bool:
        """
        Validate that tx_data contains required fields and safe gas parameters.
        """
        try:
            required_fields = ["output_token", "amountOut", "amountIn", "gas_price", "gas_used"]
            for field in required_fields:
                if field not in tx_data:
                    logger.error(f"Missing field in tx_data: {field}")
                    return False

            gas_price = Decimal(tx_data["gas_price"])
            gas_used = Decimal(tx_data["gas_used"])
            if not self._validate_gas_parameters(gas_price, gas_used):
                return False
            return True
        except Exception as e:
            logger.error(f"Error during transaction validation: {e}", exc_info=True)
            return False

    async def stop(self) -> None:
        """
        Gracefully stop SafetyNet operations.
        """
        try:
            if self.apiconfig:
                await self.apiconfig.close()
            logger.info("SafetyNet stopped successfully.")
        except Exception as e:
            logger.error(f"Error stopping SafetyNet: {e}", exc_info=True)
            raise
