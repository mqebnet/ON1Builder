import asyncio
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from typing import Any, Dict, Tuple, Optional, Union

from cachetools import TTLCache
from web3 import AsyncWeb3
from web3.types import Wei
from eth_account.signers.local import LocalAccount # Use LocalAccount for typing

# Assuming configuration facade
from configuration import Configuration
from apiconfig import APIConfig
# MarketMonitor import deferred to prevent circular dependency issues if needed later
# from marketmonitor import MarketMonitor

from loggingconfig import setup_logging
import logging

logger = setup_logging("SafetyNet", level=logging.DEBUG)

# Define a constant for ETH decimals to avoid magic numbers
ETH_DECIMALS = Decimal("18")
WEI_PER_ETH = Decimal("1e18")


class SafetyNet:
    """
    Provides risk management and transaction validation for MEV operations.
    Handles balance checks, profit verification, gas estimation, slippage,
    network congestion, and general transaction safety checks.
    """
    def __init__(
        self,
        web3: AsyncWeb3,
        configuration: Configuration,
        account: LocalAccount, # Expect a LocalAccount object
        apiconfig: APIConfig,
        # MarketMonitor is optional, can be added later if needed for complex checks
        # marketmonitor: Optional[MarketMonitor] = None,
    ) -> None:
        self.web3: AsyncWeb3 = web3
        self.configuration: Configuration = configuration
        self.account: LocalAccount = account # Store the full account object
        self.address: str = account.address # Keep address for convenience
        self.apiconfig: APIConfig = apiconfig
        # self.marketmonitor: Optional[MarketMonitor] = marketmonitor

        # Cache settings from configuration
        self._balance_cache: TTLCache = TTLCache(maxsize=10, ttl=self.configuration.SAFETYNET_CACHE_TTL) # Cache balance
        self.price_cache: TTLCache = TTLCache(maxsize=2000, ttl=self.configuration.SAFETYNET_CACHE_TTL) # Cache external prices
        self.gas_price_cache: TTLCache = TTLCache(maxsize=5, ttl=self.configuration.SAFETYNET_GAS_PRICE_TTL) # Cache gas prices

        # Locks for thread safety (though primarily async, good practice if shared)
        self.price_lock: asyncio.Lock = asyncio.Lock()
        self._balance_lock: asyncio.Lock = asyncio.Lock() # Lock for balance cache access
        self._gas_lock: asyncio.Lock = asyncio.Lock() # Lock for gas price cache access

        # Configuration constants (loaded once for performance)
        self._max_gas_price_gwei = Decimal(self.configuration.MAX_GAS_PRICE_GWEI)
        self._min_profit_eth = Decimal(str(self.configuration.MIN_PROFIT)) # Load MIN_PROFIT from config as Decimal
        self._base_gas_limit = self.configuration.BASE_GAS_LIMIT

        self.SLIPPAGE_CONFIG: Dict[str, Decimal] = {
            "default": Decimal(str(self.configuration.SLIPPAGE_DEFAULT)),
            "min": Decimal(str(self.configuration.MIN_SLIPPAGE)),
            "max": Decimal(str(self.configuration.MAX_SLIPPAGE)),
            "high_congestion": Decimal(str(self.configuration.SLIPPAGE_HIGH_CONGESTION)),
            "low_congestion": Decimal(str(self.configuration.SLIPPAGE_LOW_CONGESTION)),
        }

        # A2: Parameterized logging
        logger.info("SafetyNet initialized for address %s.", self.address)

    async def initialize(self) -> None:
        """Verify web3 connectivity and perform initial checks."""
        try:
            if not self.web3 or not await self.web3.is_connected():
                raise RuntimeError("Web3 connection failed in SafetyNet.")
            # Perform an initial balance fetch to populate cache and check account
            await self.get_balance(force_refresh=True)
            # A2: Parameterized logging
            logger.debug("SafetyNet connection verified and initial balance fetched.")
        except Exception as e:
            # A2: Parameterized logging
            logger.critical("SafetyNet initialization failed: %s", e, exc_info=True)
            raise

    async def get_balance(self, force_refresh: bool = False) -> Decimal:
        """
        Retrieve the account balance in ETH with caching and locking.

        Args:
            force_refresh: If True, bypass cache and fetch fresh balance.

        Returns:
            Account balance in ETH as Decimal.
        """
        cache_key = self.address
        async with self._balance_lock:
            if not force_refresh and cache_key in self._balance_cache:
                # A2: Parameterized logging
                logger.debug("Balance retrieved from cache for %s.", self.address)
                return self._balance_cache[cache_key]

            # A2: Parameterized logging
            logger.debug("Fetching balance from chain for %s...", self.address)
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    balance_wei: Wei = await self.web3.eth.get_balance(self.account.address)
                    balance_eth = Decimal(balance_wei) / WEI_PER_ETH
                    self._balance_cache[cache_key] = balance_eth
                    # A2: Parameterized logging
                    logger.debug("Fetched balance for %s: %.8f ETH", self.address, balance_eth)
                    return balance_eth
                except Exception as e:
                    # A2: Parameterized logging
                    logger.warning(
                        "Attempt %d/%d failed to fetch balance for %s: %s",
                        attempt + 1, max_retries, self.address, e
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1 * (attempt + 1)) # Exponential backoff
            # A2: Parameterized logging
            logger.error("Failed to fetch balance for %s after %d retries.", self.address, max_retries)
            # Return cached value if available, otherwise 0
            return self._balance_cache.get(cache_key, Decimal("0"))

    # A8: Decimal Consistency
    async def ensure_profit(
        self,
        # Expect amounts and gas figures potentially as str, int, float, or Decimal
        expected_output_amount: Union[str, int, float, Decimal],
        output_token_symbol_or_addr: str,
        input_amount_eth: Union[str, int, float, Decimal], # Assuming input is in ETH for simplicity
        gas_price_gwei: Union[str, int, float, Decimal],
        gas_used: Union[str, int, float, Decimal],
        minimum_profit_eth_override: Optional[Union[str, float, Decimal]] = None,
        tx_hash: Optional[str] = None # A16: For logging context
    ) -> bool:
        """
        Checks if a potential transaction is likely to yield sufficient profit
        after estimated gas costs and slippage. Uses Decimal for all calculations.

        Args:
            expected_output_amount: The expected amount of the output token (in token units).
            output_token_symbol_or_addr: Symbol or address of the output token.
            input_amount_eth: The amount of ETH being spent (or equivalent value).
            gas_price_gwei: Estimated gas price in Gwei.
            gas_used: Estimated gas units to be used.
            minimum_profit_eth_override: Optional override for the minimum profit threshold (in ETH).
            tx_hash: Optional transaction hash for logging context.

        Returns:
            bool: True if the estimated profit exceeds the threshold, False otherwise.
        """
        log_extra = {"component": "SafetyNet", "tx_hash": tx_hash or "N/A"} # A16

        try:
            # A8: Convert all numeric inputs to Decimal at the start
            dec_expected_output = Decimal(str(expected_output_amount))
            dec_input_amount_eth = Decimal(str(input_amount_eth))
            dec_gas_price_gwei = Decimal(str(gas_price_gwei))
            dec_gas_used = Decimal(str(gas_used))
            dec_min_profit_eth = Decimal(str(minimum_profit_eth_override)) if minimum_profit_eth_override is not None else self._min_profit_eth

            # A2: Parameterized logging
            logger.debug(
                "Ensure Profit Check: Expected Output: %s, Input ETH: %s, Gas Price Gwei: %s, Gas Used: %s, Min Profit ETH: %s",
                dec_expected_output, dec_input_amount_eth, dec_gas_price_gwei, dec_gas_used, dec_min_profit_eth,
                extra=log_extra
            )

            # Validate gas parameters (using Decimal)
            if not self._validate_gas_parameters(dec_gas_price_gwei, dec_gas_used, log_extra):
                return False # Validation logs errors

            # Get real-time price of the output token in ETH
            # Assuming get_real_time_price returns Decimal or None
            output_price_eth: Optional[Decimal] = await self.apiconfig.get_real_time_price(output_token_symbol_or_addr, vs="eth")
            if output_price_eth is None:
                # A2: Parameterized logging
                logger.warning(
                    "Real-time price for '%s' unavailable; cannot ensure profit.",
                    output_token_symbol_or_addr, extra=log_extra
                )
                return False
            if output_price_eth <= Decimal("0"):
                 logger.warning(
                    "Real-time price for '%s' is zero or negative (%.18f); cannot calculate profit.",
                    output_token_symbol_or_addr, output_price_eth, extra=log_extra
                 )
                 return False


            # Calculate gas cost in ETH (using Decimal)
            gas_cost_eth = self._calculate_gas_cost_eth(dec_gas_price_gwei, dec_gas_used)

            # Get dynamic slippage tolerance
            slippage_factor = await self.adjust_slippage_tolerance() # Returns Decimal

            # Calculate profit (using Decimal)
            # Profit = (Expected Output * Price * (1 - Slippage)) - Input Cost - Gas Cost
            expected_output_value_eth = dec_expected_output * output_price_eth
            adjusted_output_value_eth = expected_output_value_eth * (Decimal("1") - slippage_factor)
            profit_eth = adjusted_output_value_eth - dec_input_amount_eth - gas_cost_eth

            # Log the calculation details
            self._log_profit_calculation(
                output_token=output_token_symbol_or_addr,
                output_price_eth=output_price_eth,
                input_amount_eth=dec_input_amount_eth,
                expected_output=dec_expected_output,
                adjusted_output_eth=adjusted_output_value_eth,
                gas_cost_eth=gas_cost_eth,
                calculated_profit_eth=profit_eth,
                min_profit_eth=dec_min_profit_eth,
                log_extra=log_extra
            )

            # A8: Compare Profit (Decimal) with Minimum Profit (Decimal)
            is_profitable = profit_eth > dec_min_profit_eth

            if is_profitable:
                # A2: Parameterized logging
                logger.debug("Transaction deemed profitable (%.8f ETH > %.8f ETH).", profit_eth, dec_min_profit_eth, extra=log_extra)
            else:
                # A2: Parameterized logging
                logger.debug("Transaction deemed not profitable (%.8f ETH <= %.8f ETH).", profit_eth, dec_min_profit_eth, extra=log_extra)

            return is_profitable

        except (InvalidOperation, TypeError) as e:
             # A2: Parameterized logging
             logger.error("Invalid numeric value encountered in ensure_profit: %s", e, extra=log_extra)
             return False
        except Exception as e:
            # A2: Parameterized logging
            logger.error("Error in ensure_profit calculation: %s", e, exc_info=True, extra=log_extra)
            return False


    def _validate_gas_parameters(self, gas_price_gwei: Decimal, gas_used: Decimal, log_extra: Dict) -> bool:
        """Validates gas price and units using Decimal."""
        if gas_used <= Decimal("0"):
            # A2: Parameterized logging
            logger.error("Gas used must be positive, but got %s.", gas_used, extra=log_extra)
            return False
        # Using the cached Decimal version of max gas price
        if gas_price_gwei > self._max_gas_price_gwei:
            # A2: Parameterized logging
            logger.warning(
                "Gas price %s Gwei exceeds configured maximum of %s Gwei.",
                gas_price_gwei, self._max_gas_price_gwei, extra=log_extra
            )
            # Decide if this should be a hard failure or just a warning
            # return False # Uncomment to make it a hard failure
        return True

    def _calculate_gas_cost_eth(self, gas_price_gwei: Decimal, gas_used: Decimal) -> Decimal:
        """Calculates the gas cost in ETH using Decimal arithmetic."""
        # Gas Cost (ETH) = Gas Used * Gas Price (Gwei) / 1e9
        gas_cost_eth = (gas_used * gas_price_gwei) / Decimal("1e9")
        # Quantize to reasonable precision (e.g., 18 decimal places like ETH)
        return gas_cost_eth.quantize(Decimal("1e-18"), rounding=ROUND_DOWN)

    def _log_profit_calculation(
        self,
        output_token: str,
        output_price_eth: Decimal,
        input_amount_eth: Decimal,
        expected_output: Decimal,
        adjusted_output_eth: Decimal,
        gas_cost_eth: Decimal,
        calculated_profit_eth: Decimal,
        min_profit_eth: Decimal,
        log_extra: Dict
    ) -> None:
        """Logs the details of the profit calculation using Decimal."""
        profitable_str = "Yes" if calculated_profit_eth > min_profit_eth else "No"
        # A2: Parameterized logging with detailed Decimal values
        logger.debug(
            "--- Profit Calculation Summary ---\n"
            " Output Token: %s\n"
            " Output Price: %.18f ETH/Token\n"
            " Input Cost: %.18f ETH\n"
            " Expected Output: %s Tokens\n"
            " Adjusted Output Value: %.18f ETH (after slippage)\n"
            " Gas Cost: %.18f ETH\n"
            " Calculated Net Profit: %.18f ETH\n"
            " Minimum Required Profit: %.18f ETH\n"
            " Profitable: %s\n"
            "--- End Summary ---",
            output_token, output_price_eth, input_amount_eth, expected_output,
            adjusted_output_eth, gas_cost_eth, calculated_profit_eth,
            min_profit_eth, profitable_str,
            extra=log_extra
        )

    async def estimate_gas(self, transaction_data: Dict[str, Any], tx_hash: Optional[str] = None) -> int:
        """
        Estimate gas for a transaction, returning a default on failure.

        Args:
            transaction_data: The transaction dictionary.
            tx_hash: Optional transaction hash for logging.

        Returns:
            Estimated gas units as int, or base limit on failure.
        """
        log_extra = {"component": "SafetyNet", "tx_hash": tx_hash or "N/A"} # A16
        try:
            # Ensure 'from' field is present and correct checksum
            if 'from' not in transaction_data or not self.web3.is_address(transaction_data['from']):
                 transaction_data['from'] = self.address # Use safety net's account address

            gas_estimate: Wei = await self.web3.eth.estimate_gas(transaction_data)
            # A2: Parameterized logging
            logger.debug("Gas estimation successful: %d units.", gas_estimate, extra=log_extra)
            # Add a small buffer (e.g., 10-20%)? Or rely on caller to do this.
            # return int(gas_estimate * Decimal("1.1"))
            return int(gas_estimate)
        except ValueError as e:
             # Often indicates revert without reason string or bad input
             # A2: Parameterized logging
             logger.warning("Gas estimation failed (ValueError): %s. TX Data: %s", e, transaction_data, extra=log_extra)
             return self._base_gas_limit # Return default
        except Exception as e:
            # Catch other potential errors (network issues, etc.)
            # A2: Parameterized logging
            logger.error("Gas estimation failed unexpectedly: %s", e, exc_info=True, extra=log_extra)
            return self._base_gas_limit # Return default

    async def adjust_slippage_tolerance(self) -> Decimal:
        """
        Adjusts slippage tolerance based on network congestion. Returns Decimal.
        """
        log_extra = {"component": "SafetyNet"} # A16
        try:
            congestion = await self.get_network_congestion() # Returns float

            # Use Decimal for comparisons and configuration values
            dec_congestion = Decimal(str(congestion))
            high_congestion_threshold = Decimal("0.8")
            low_congestion_threshold = Decimal("0.2")

            if dec_congestion > high_congestion_threshold:
                slippage = self.SLIPPAGE_CONFIG["high_congestion"]
            elif dec_congestion < low_congestion_threshold:
                slippage = self.SLIPPAGE_CONFIG["low_congestion"]
            else:
                slippage = self.SLIPPAGE_CONFIG["default"]

            # Clamp slippage between min and max configured values
            slippage = max(self.SLIPPAGE_CONFIG["min"], min(slippage, self.SLIPPAGE_CONFIG["max"]))

            # A2: Parameterized logging
            logger.debug(
                "Adjusted slippage to %.4f (%.2f%%) based on network congestion %.4f (%.2f%%)",
                slippage, slippage * 100, dec_congestion, dec_congestion * 100,
                extra=log_extra
            )
            return slippage # Return Decimal
        except Exception as e:
            # A2: Parameterized logging
            logger.error("Error adjusting slippage tolerance: %s. Falling back to default.", e, exc_info=True, extra=log_extra)
            return self.SLIPPAGE_CONFIG["default"] # Return default Decimal

    async def get_network_congestion(self) -> float:
        """
        Estimates network congestion based on the latest block's gas usage.
        Returns congestion as a float between 0.0 and 1.0.
        """
        log_extra = {"component": "SafetyNet"} # A16
        try:
            latest_block = await self.web3.eth.get_block("latest")
            gas_used = Decimal(latest_block.get("gasUsed", 0))
            gas_limit = Decimal(latest_block.get("gasLimit", 1)) # Avoid division by zero

            if gas_limit == 0:
                 logger.warning("Latest block gas limit is zero, cannot calculate congestion.", extra=log_extra)
                 return 0.5 # Return neutral value

            congestion = float((gas_used / gas_limit).quantize(Decimal("0.0001")))
            # Clamp between 0 and 1
            congestion = max(0.0, min(congestion, 1.0))

            # A2: Parameterized logging
            logger.debug("Network congestion calculated: %.4f (%.2f%%)", congestion, congestion * 100, extra=log_extra)
            return congestion
        except Exception as e:
            # A2: Parameterized logging
            logger.error("Error fetching network congestion: %s. Returning default 0.5", e, exc_info=True, extra=log_extra)
            return 0.5 # Return a neutral default value on error

    async def check_transaction_safety(
        self,
        tx_data: Dict[str, Any], # Transaction data potentially used for checks
        check_type: str = "all",
        tx_hash: Optional[str] = None # A16
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Performs multiple safety checks on a potential transaction.

        Args:
            tx_data: Dictionary containing transaction details (may need 'output_token', 'gas_price', etc.).
            check_type: Which checks to perform ('all', 'gas', 'profit', 'network').
            tx_hash: Optional transaction hash for logging.

        Returns:
            Tuple (is_safe, details_dict). details_dict contains individual check results.
        """
        log_extra = {"component": "SafetyNet", "tx_hash": tx_hash or "N/A"} # A16
        results = {
            "is_safe": True,
            "gas_ok": True,
            "profit_ok": True,
            "network_ok": True,
            "messages": [],
        }

        try:
            # --- Gas Price Check ---
            if check_type in ("all", "gas"):
                # Use dynamic gas price for the check, compare against MAX (Decimal)
                current_gas_gwei = await self.get_dynamic_gas_price() # Returns Decimal
                if current_gas_gwei > self._max_gas_price_gwei:
                    results["gas_ok"] = False
                    msg = f"Current gas price {current_gas_gwei} Gwei exceeds limit {self._max_gas_price_gwei} Gwei."
                    results["messages"].append(msg)
                    logger.warning(msg, extra=log_extra)

            # --- Profit Check ---
            # Requires more context from tx_data (amounts, tokens etc.)
            # Placeholder: assumes ensure_profit can be called if enough data exists
            if check_type in ("all", "profit"):
                # Example: Check if enough data exists to call ensure_profit
                required_keys = {"expected_output_amount", "output_token_symbol_or_addr", "input_amount_eth", "gas_price_gwei", "gas_used"}
                if all(key in tx_data for key in required_keys):
                    is_profitable = await self.ensure_profit(
                        expected_output_amount=tx_data["expected_output_amount"],
                        output_token_symbol_or_addr=tx_data["output_token_symbol_or_addr"],
                        input_amount_eth=tx_data["input_amount_eth"],
                        gas_price_gwei=tx_data["gas_price_gwei"],
                        gas_used=tx_data["gas_used"],
                        tx_hash=tx_hash # Pass hash context
                    )
                    if not is_profitable:
                        results["profit_ok"] = False
                        msg = "Profit check failed (see previous logs for details)."
                        results["messages"].append(msg)
                        # No need for extra logger call here, ensure_profit logs details
                else:
                     logger.debug("Skipping profit check in check_transaction_safety: Insufficient data.", extra=log_extra)
                     # Optionally mark as not OK if profit check is essential but can't run
                     # results["profit_ok"] = False
                     # results["messages"].append("Insufficient data for profit check.")


            # --- Network Congestion Check ---
            if check_type in ("all", "network"):
                congestion = await self.get_network_congestion() # Returns float
                high_congestion_threshold = 0.8 # Example threshold
                if congestion > high_congestion_threshold:
                    results["network_ok"] = False
                    msg = f"Network congestion high: {congestion*100:.2f}% (>{high_congestion_threshold*100:.0f}%)."
                    results["messages"].append(msg)
                    logger.warning(msg, extra=log_extra)

            # Determine overall safety
            results["is_safe"] = all([results["gas_ok"], results["profit_ok"], results["network_ok"]])

            return results["is_safe"], results

        except Exception as e:
            # A2: Parameterized logging
            logger.error("Error during transaction safety check: %s", e, exc_info=True, extra=log_extra)
            results["is_safe"] = False
            results["messages"].append(f"Unexpected error during safety check: {e}")
            return False, results

    async def get_dynamic_gas_price(self) -> Decimal:
        """
        Fetches the current recommended gas price (legacy or EIP-1559 based) in Gwei.
        Uses caching and locking. Returns Decimal.
        """
        cache_key = "dynamic_gas_price"
        async with self._gas_lock:
            if cache_key in self.gas_price_cache:
                return self.gas_price_cache[cache_key]

            try:
                latest_block = await self.web3.eth.get_block("latest")
                gas_price_gwei: Decimal

                if "baseFeePerGas" in latest_block and latest_block["baseFeePerGas"] is not None:
                    # EIP-1559 logic: Suggest maxFeePerGas = 2 * base + priority
                    base_fee_wei = latest_block["baseFeePerGas"]
                    # Use eth_maxPriorityFeePerGas if available, otherwise estimate
                    try:
                         priority_fee_wei = await self.web3.eth.max_priority_fee
                    except Exception:
                         # Fallback simple priority fee (e.g., 1 Gwei) if RPC fails
                         priority_fee_wei = self.web3.to_wei(1, "gwei")

                    # Suggest a maxFeePerGas based on base fee and priority fee
                    # Example: maxFee = 2 * base + priority (adjust multiplier as needed)
                    suggested_max_fee_wei = (base_fee_wei * 2) + priority_fee_wei
                    gas_price_gwei = Decimal(suggested_max_fee_wei) / Decimal("1e9")
                    # A2: Parameterized logging
                    logger.debug(
                        "EIP-1559 Gas Price: Base=%.4f Gwei, Priority=%.4f Gwei -> Suggested Max Fee=%.4f Gwei",
                        Decimal(base_fee_wei) / Decimal("1e9"),
                        Decimal(priority_fee_wei) / Decimal("1e9"),
                        gas_price_gwei
                    )
                else:
                    # Legacy gas price logic
                    gas_price_wei = await self.web3.eth.gas_price
                    gas_price_gwei = Decimal(gas_price_wei) / Decimal("1e9")
                    # A2: Parameterized logging
                    logger.debug("Legacy Gas Price: %.4f Gwei", gas_price_gwei)

                # Clamp to configured max gas price before caching
                gas_price_gwei = min(gas_price_gwei, self._max_gas_price_gwei)

                self.gas_price_cache[cache_key] = gas_price_gwei
                # A2: Parameterized logging
                logger.debug("Dynamic gas price updated: %.4f Gwei", gas_price_gwei)
                return gas_price_gwei

            except Exception as e:
                # A2: Parameterized logging
                logger.error("Error fetching dynamic gas price: %s. Returning max allowed.", e, exc_info=True)
                # Return the configured max on error, ensuring it's cached briefly
                self.gas_price_cache[cache_key] = self._max_gas_price_gwei
                return self._max_gas_price_gwei

    async def validate_transaction_data(self, tx_data: Dict[str, Any], tx_hash: Optional[str] = None) -> bool:
        """
        Validates that transaction data contains required fields and safe numeric values.

        Args:
            tx_data: The transaction dictionary.
            tx_hash: Optional transaction hash for logging.

        Returns:
            True if basic validation passes, False otherwise.
        """
        log_extra = {"component": "SafetyNet", "tx_hash": tx_hash or "N/A"} # A16
        required_fields = ["to", "value", "gasPrice", "gas"] # Adjust based on tx type (legacy vs EIP-1559)
        # EIP-1559 fields: "maxFeePerGas", "maxPriorityFeePerGas"

        missing_fields = [field for field in required_fields if field not in tx_data]
        if missing_fields:
            # A2: Parameterized logging
            logger.error("Missing required fields in tx_data: %s", missing_fields, extra=log_extra)
            return False

        try:
            # Validate numeric fields can be converted and are sensible
            dec_gas_price = Decimal(str(tx_data["gasPrice"])) / Decimal("1e9") # Convert Wei to Gwei for validation
            dec_gas_limit = Decimal(str(tx_data["gas"]))
            dec_value = Decimal(str(tx_data["value"]))

            if dec_gas_limit <= 0:
                 logger.error("Invalid gas limit: %s must be positive.", dec_gas_limit, extra=log_extra)
                 return False
            if dec_gas_price <= 0:
                 logger.error("Invalid gas price: %s must be positive.", dec_gas_price, extra=log_extra)
                 return False
            if dec_value < 0:
                 logger.error("Invalid value: %s cannot be negative.", dec_value, extra=log_extra)
                 return False

            # Validate 'to' address
            if not self.web3.is_address(tx_data["to"]):
                 logger.error("Invalid 'to' address: %s", tx_data["to"], extra=log_extra)
                 return False
            tx_data["to"] = self.web3.to_checksum_address(tx_data["to"]) # Ensure checksum

            # Add EIP-1559 field validation if present
            if "maxFeePerGas" in tx_data or "maxPriorityFeePerGas" in tx_data:
                if "maxFeePerGas" not in tx_data or "maxPriorityFeePerGas" not in tx_data:
                    logger.error("Both maxFeePerGas and maxPriorityFeePerGas required for EIP-1559.", extra=log_extra)
                    return False
                dec_max_fee = Decimal(str(tx_data["maxFeePerGas"]))
                dec_priority_fee = Decimal(str(tx_data["maxPriorityFeePerGas"]))
                if dec_max_fee <= 0 or dec_priority_fee < 0: # Priority fee can be 0
                     logger.error("Invalid EIP-1559 gas fees: MaxFee=%s, PriorityFee=%s", dec_max_fee, dec_priority_fee, extra=log_extra)
                     return False
                if dec_priority_fee > dec_max_fee:
                     logger.error("Invalid EIP-1559 fees: PriorityFee (%s) > MaxFee (%s)", dec_priority_fee, dec_max_fee, extra=log_extra)
                     return False


            # A2: Parameterized logging
            logger.debug("Transaction data basic validation passed.", extra=log_extra)
            return True

        except (InvalidOperation, TypeError, KeyError) as e:
            # A2: Parameterized logging
            logger.error("Error validating transaction data fields: %s. Data: %s", e, tx_data, extra=log_extra)
            return False
        except Exception as e:
             logger.error("Unexpected error during transaction validation: %s", e, exc_info=True, extra=log_extra)
             return False


    async def stop(self) -> None:
        """Gracefully stop SafetyNet operations (e.g., clear caches)."""
        try:
            # Clear caches on stop
            async with self._balance_lock:
                self._balance_cache.clear()
            async with self._gas_lock:
                self.gas_price_cache.clear()
            async with self.price_lock:
                self.price_cache.clear()

            # Close API config session if managed here (or ensure it's closed elsewhere)
            # await self.apiconfig.close() # If SafetyNet owns the session

            # A2: Parameterized logging
            logger.info("SafetyNet stopped successfully.")
        except Exception as e:
            # A2: Parameterized logging
            logger.error("Error stopping SafetyNet: %s", e, exc_info=True)
            # Don't re-raise in stop(), just log the error