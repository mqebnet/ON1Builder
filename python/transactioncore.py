import asyncio
import time
from eth_typing import HexStr
import hexbytes
from web3 import AsyncWeb3
from web3.exceptions import TransactionNotFound, ContractLogicError, Web3Exception
from web3.types import TxParams, Wei, HexBytes, Nonce, TxReceipt # Import specific types
from eth_account import Account
from eth_account.signers.local import LocalAccount # Specific type
from typing import Any, Dict, List, Optional, Tuple, Union, cast

# Assuming configuration facade and other components
from configuration import Configuration
from apiconfig import APIConfig
# Defer MarketMonitor import if circular dependency is an issue
#rom marketmonitor import MarketMonitor
from noncecore import NonceCore
from safetynet import WEI_PER_ETH, SafetyNet
from abiregistry import ABIRegistry # Import if needed for local ABI handling

from loggingconfig import setup_logging
import logging
from decimal import Decimal, getcontext

# Set Decimal precision if needed globally for profit calculations
# getcontext().prec = 30 # Example: Set precision to 30 places

logger = setup_logging("TransactionCore", level=logging.DEBUG)

# Type alias for strategy function result
StrategyResult = Tuple[bool, Decimal] # (success: bool, profit_eth: Decimal)

class TransactionCore:
    """
    Handles the construction, signing, simulation, execution, and strategy
    implementation of Ethereum transactions for MEV operations.
    """
    # Default gas limit, adjust as needed based on typical strategy complexity
    DEFAULT_GAS_LIMIT: int = 500_000 # Increased default
    # Default gas price multiplier if not using EIP-1559
    DEFAULT_GAS_PRICE_MULTIPLIER: float = 1.1
    # Default gas price for cancellation transactions (Gwei)
    DEFAULT_CANCEL_GAS_GWEI: int = 60 # Loaded from config now

    def __init__(
        self,
        web3: AsyncWeb3,
        account: LocalAccount, # Expect LocalAccount type
        configuration: Configuration,
        noncecore: NonceCore, # Required dependency
        safetynet: SafetyNet, # Required dependency
        apiconfig: APIConfig, # Required dependency
        # MarketMonitor is optional, can be injected later or passed if needed now
        marketmonitor: Optional['MarketMonitor'] = None, # Forward reference if needed
        abiregistry: Optional[ABIRegistry] = None, # Pass pre-initialized registry
    ):
        self.web3: AsyncWeb3 = web3
        self.account: LocalAccount = account
        self.configuration: Configuration = configuration
        self.noncecore: NonceCore = noncecore
        self.safetynet: SafetyNet = safetynet
        self.apiconfig: APIConfig = apiconfig
        self.marketmonitor: Optional['MarketMonitor'] = marketmonitor
        # Use passed ABIRegistry or create one if needed locally (less ideal)
        self.abiregistry: ABIRegistry = abiregistry if abiregistry else ABIRegistry()

        # Load settings from config
        self.gas_price_multiplier: float = self.configuration.get_config_value(
            "ETH_TX_GAS_PRICE_MULTIPLIER", self.DEFAULT_GAS_PRICE_MULTIPLIER
        )
        self.default_gas_limit: int = self.configuration.get_config_value(
            "DEFAULT_GAS_LIMIT", self.DEFAULT_GAS_LIMIT
        )
        self.cancel_gas_gwei: int = self.configuration.get_config_value(
            "DEFAULT_CANCEL_GAS_PRICE_GWEI", self.DEFAULT_CANCEL_GAS_GWEI
        )
        self._base_gas_limit: int = self.configuration.get_config_value("BASE_GAS_LIMIT", 21000) # Base for simple ETH transfer


        # Contract instances (will be initialized in initialize)
        # Use generic 'Any' or specific ContractFunction type if available
        self.uniswap_router_contract: Optional[Any] = None
        self.sushiswap_router_contract: Optional[Any] = None
        self.aave_pool_contract: Optional[Any] = None
        # Add other required contracts (e.g., flashloan provider/receiver)
        self.aave_flashloan_contract: Optional[Any] = None


    async def initialize(self) -> None:
        """
        Initializes the TransactionCore by loading ABIs (if not passed),
        creating contract instances, and validating them.
        """
        log_extra = {"component": "TransactionCore"} # A16
        logger.info("Initializing TransactionCore...", extra=log_extra)
        try:
            # Initialize ABIRegistry if it wasn't passed pre-initialized
            if not self.abiregistry._initialized:
                 logger.info("Initializing ABIRegistry within TransactionCore...", extra=log_extra)
                 # Use base path from configuration
                 await self.abiregistry.initialize(self.configuration.BASE_PATH)

            # --- Load Contract Instances ---
            # Example: Uniswap V2 Router
            uniswap_addr = self.configuration.get_config_value("UNISWAP_ADDRESS")
            uniswap_abi = self.abiregistry.get_abi('uniswap')
            if uniswap_addr and uniswap_abi:
                 self.uniswap_router_contract = self.web3.eth.contract(address=uniswap_addr, abi=uniswap_abi)
                 await self._validate_contract(self.uniswap_router_contract, "Uniswap V2 Router", log_extra)
            else:
                 logger.warning("Uniswap V2 Router not configured or ABI missing.", extra=log_extra)

            # Example: Sushiswap Router
            sushiswap_addr = self.configuration.get_config_value("SUSHISWAP_ADDRESS")
            sushiswap_abi = self.abiregistry.get_abi('sushiswap')
            if sushiswap_addr and sushiswap_abi:
                 self.sushiswap_router_contract = self.web3.eth.contract(address=sushiswap_addr, abi=sushiswap_abi)
                 await self._validate_contract(self.sushiswap_router_contract, "Sushiswap Router", log_extra)
            else:
                 logger.warning("Sushiswap Router not configured or ABI missing.", extra=log_extra)

            # Example: Aave V3 Pool
            aave_pool_addr = self.configuration.get_config_value("AAVE_POOL_ADDRESS")
            aave_pool_abi = self.abiregistry.get_abi('aave') # Assuming 'aave' key holds pool ABI
            if aave_pool_addr and aave_pool_abi:
                 self.aave_pool_contract = self.web3.eth.contract(address=aave_pool_addr, abi=aave_pool_abi)
                 await self._validate_contract(self.aave_pool_contract, "Aave V3 Pool", log_extra)
            else:
                 logger.warning("Aave Pool not configured or ABI missing.", extra=log_extra)

            # Example: Aave Flashloan Contract (Provider or Receiver?)
            # This highly depends on the flashloan implementation
            aave_flashloan_addr = self.configuration.get_config_value("AAVE_FLASHLOAN_ADDRESS")
            aave_flashloan_abi = self.abiregistry.get_abi('aave_flashloan')
            if aave_flashloan_addr and aave_flashloan_abi:
                 self.aave_flashloan_contract = self.web3.eth.contract(address=aave_flashloan_addr, abi=aave_flashloan_abi)
                 await self._validate_contract(self.aave_flashloan_contract, "Aave Flashloan Contract", log_extra)
            else:
                 logger.warning("Aave Flashloan contract not configured or ABI missing. Flashloan strategies may fail.", extra=log_extra)

            # Initialize ERC20 contract factory (useful for dynamic token interactions)
            self.erc20_abi = self.abiregistry.get_abi('erc20')
            if not self.erc20_abi:
                 logger.error("ERC20 ABI not loaded. Token interactions will fail.", extra=log_extra)
                 raise RuntimeError("Missing critical ERC20 ABI")

            logger.info("TransactionCore initialized successfully.", extra=log_extra)

        except Exception as e:
            logger.critical("Error initializing TransactionCore: %s", e, exc_info=True, extra=log_extra)
            raise # Propagate initialization failure

    def get_erc20_contract(self, token_address: str) -> Optional[Any]:
         """Creates an ERC20 contract instance for a given address."""
         if not self.erc20_abi:
              logger.error("ERC20 ABI missing, cannot create contract instance.")
              return None
         try:
              checksum_address = self.web3.to_checksum_address(token_address)
              return self.web3.eth.contract(address=checksum_address, abi=self.erc20_abi)
         except ValueError:
              logger.error("Invalid token address format: %s", token_address)
              return None


    async def _validate_contract(self, contract: Any, name: str, log_extra: Dict) -> None:
        """Validates contract deployment by checking for bytecode."""
        if not contract or not contract.address:
             logger.warning("Contract object for %s is invalid.", name, extra=log_extra)
             raise ValueError(f"Invalid contract object for {name}")
        address = contract.address
        logger.debug("Validating contract %s at %s...", name, address, extra=log_extra)
        try:
            code = await self.web3.eth.get_code(address)
            if code == HexBytes("0x"):
                logger.error("Validation failed for %s: No bytecode found at address %s.", name, address, extra=log_extra)
                raise ValueError(f"{name} contract not found at {address}")
            else:
                logger.debug("Contract %s validated successfully (bytecode found).", name, extra=log_extra)
        except Exception as e:
            logger.error("Error validating contract %s at %s: %s", name, address, e, exc_info=True, extra=log_extra)
            raise # Re-raise validation errors


    async def build_transaction(
        self,
        to: str,
        value: Wei = Wei(0),
        data: str = "0x",
        gas_limit_override: Optional[int] = None,
        nonce_override: Optional[Nonce] = None,
        gas_params_override: Optional[Dict[str, Wei]] = None # For 'gasPrice' or 'maxFeePerGas'/'maxPriorityFeePerGas'
    ) -> TxParams:
        """
        Builds a transaction dictionary with appropriate nonce, gas parameters,
        and other necessary fields.

        Args:
            to: The recipient address.
            value: Amount of Ether (in Wei) to send.
            data: Transaction data (e.g., function call bytecode).
            gas_limit_override: Manually specify gas limit (ignores estimation).
            nonce_override: Manually specify nonce.
            gas_params_override: Manually specify gas price or EIP-1559 fees.

        Returns:
            A transaction parameter dictionary (TxParams).
        """
        log_extra = {"component": "TransactionCore"} # A16
        try:
            chain_id = await self.web3.eth.chain_id
            nonce = nonce_override if nonce_override is not None else await self.noncecore.get_next_nonce()
            # A2: Parameterized logging
            logger.debug("Building transaction for Nonce %d on Chain ID %d", nonce, chain_id, extra=log_extra)

            tx_params: TxParams = {
                "from": self.account.address,
                "to": self.web3.to_checksum_address(to), # Ensure checksum address
                "value": value,
                "nonce": nonce,
                "chainId": chain_id,
                "data": data,
                # Gas will be estimated or set later
            }

            # --- Determine Gas Parameters ---
            if gas_params_override:
                 if "gasPrice" in gas_params_override:
                      tx_params["gasPrice"] = gas_params_override["gasPrice"]
                 elif "maxFeePerGas" in gas_params_override and "maxPriorityFeePerGas" in gas_params_override:
                      tx_params["maxFeePerGas"] = gas_params_override["maxFeePerGas"]
                      tx_params["maxPriorityFeePerGas"] = gas_params_override["maxPriorityFeePerGas"]
                 else:
                      logger.warning("Invalid gas_params_override provided. Falling back to dynamic.", extra=log_extra)
                      await self._set_dynamic_gas_params(tx_params, log_extra)
            else:
                 await self._set_dynamic_gas_params(tx_params, log_extra)


            # --- Determine Gas Limit ---
            if gas_limit_override:
                tx_params["gas"] = Wei(gas_limit_override)
                logger.debug("Using provided gas limit override: %d", gas_limit_override, extra=log_extra)
            else:
                estimated_gas = await self.estimate_gas_smart(tx_params, log_extra)
                # Add buffer to estimated gas (e.g., 20%)
                buffered_gas = int(estimated_gas * 1.2)
                # Ensure it doesn't exceed a reasonable max (or default if estimate failed)
                tx_params["gas"] = Wei(min(buffered_gas, self.default_gas_limit if estimated_gas != self.default_gas_limit else self.default_gas_limit )) # Handle case where estimation failed
                logger.debug("Estimated gas: %d, Used gas limit: %d", estimated_gas, tx_params["gas"], extra=log_extra)

            # A2: Parameterized logging
            logger.debug("Transaction built: %s", tx_params, extra=log_extra)
            return tx_params

        except Exception as e:
            logger.error("Error building transaction: %s", e, exc_info=True, extra=log_extra)
            raise # Propagate build errors


    async def _set_dynamic_gas_params(self, tx_params: TxParams, log_extra: Dict) -> None:
         """Fetches dynamic gas prices/fees and adds them to the transaction."""
         try:
              latest_block = await self.web3.eth.get_block("latest")
              supports_eip1559 = "baseFeePerGas" in latest_block and latest_block["baseFeePerGas"] is not None

              if supports_eip1559:
                   base_fee = latest_block["baseFeePerGas"]
                   # Get priority fee suggestion (can fail, provide fallback)
                   try:
                        priority_fee = await self.web3.eth.max_priority_fee
                   except Exception:
                        priority_fee = self.web3.to_wei(2, 'gwei') # Fallback priority fee (e.g., 2 Gwei)
                        logger.warning("Failed to get max_priority_fee, using fallback: %s Wei", priority_fee, extra=log_extra)

                   # Calculate maxFeePerGas (e.g., base * 2 + priority)
                   max_fee = (base_fee * 2) + priority_fee

                   # Apply safety checks from SafetyNet (clamp to max allowed)
                   # Assuming get_dynamic_gas_price handles clamping internally now based on EIP1559
                   clamped_max_fee_gwei = await self.safetynet.get_dynamic_gas_price() # Returns clamped max fee in Gwei as Decimal
                   clamped_max_fee_wei = self.web3.to_wei(clamped_max_fee_gwei, 'gwei')

                   # Ensure priority fee isn't higher than clamped max fee
                   final_priority_fee = min(priority_fee, clamped_max_fee_wei)

                   tx_params["maxFeePerGas"] = Wei(clamped_max_fee_wei)
                   tx_params["maxPriorityFeePerGas"] = Wei(final_priority_fee)
                   logger.debug(
                       "Using EIP-1559 gas: MaxFee=%s Wei, PriorityFee=%s Wei (Base Fee: %s Wei)",
                       tx_params["maxFeePerGas"], tx_params["maxPriorityFeePerGas"], base_fee, extra=log_extra
                   )
              else:
                   # Legacy transaction: Get dynamic gas price and apply multiplier
                   gas_price_gwei = await self.safetynet.get_dynamic_gas_price() # Returns Decimal Gwei (already clamped)
                   multiplied_gas_price_gwei = gas_price_gwei * Decimal(str(self.gas_price_multiplier))
                   # Ensure multiplied price doesn't exceed absolute max again (belt-and-suspenders)
                   final_gas_price_gwei = min(multiplied_gas_price_gwei, self.safetynet._max_gas_price_gwei)
                   tx_params["gasPrice"] = self.web3.to_wei(final_gas_price_gwei, 'gwei')
                   logger.debug("Using legacy gas: gasPrice=%s Wei (%.2f Gwei)", tx_params["gasPrice"], final_gas_price_gwei, extra=log_extra)

         except Exception as e:
              logger.error("Failed to set dynamic gas parameters: %s. Transaction might fail.", e, exc_info=True, extra=log_extra)
              # Optionally set a default fallback gas price
              tx_params["gasPrice"] = self.web3.to_wei(self.safetynet._max_gas_price_gwei / Decimal(2), 'gwei') # Example: half max


    async def estimate_gas_smart(self, tx: TxParams, log_extra: Dict) -> int:
        """Estimates gas, handling potential reverts and providing fallbacks."""
        # Remove fields not allowed by estimate_gas if present
        tx_estimate = tx.copy()
        tx_estimate.pop("gas", None) # Remove 'gas' itself if present
        # Ensure 'from' is present
        if 'from' not in tx_estimate:
             tx_estimate['from'] = self.account.address

        try:
            gas_estimate: Wei = await self.web3.eth.estimate_gas(tx_estimate)
            # A2: Parameterized logging
            logger.debug("Gas estimated successfully: %d", gas_estimate, extra=log_extra)
            return int(gas_estimate)
        except ContractLogicError as e:
            # Transaction would revert
            logger.warning("Gas estimation failed: Transaction would revert. Reason: %s. TX: %s", e, tx_estimate, extra=log_extra)
            # Cannot execute, return default/high value to prevent submission? Or raise?
            # Returning default limit might lead to failed tx submission. Maybe return 0 or raise?
            # Let's return default for now, but caller should be aware.
            return self.default_gas_limit
        except ValueError as e:
             # Often indicates gas estimation failed without specific revert reason
             logger.warning("Gas estimation failed (ValueError): %s. TX: %s", e, tx_estimate, extra=log_extra)
             return self.default_gas_limit
        except Exception as e:
            # Network errors, etc.
            logger.error("Unexpected error during gas estimation: %s", e, exc_info=True, extra=log_extra)
            return self.default_gas_limit


    # A6: Use Account.sign_transaction for AsyncWeb3 compatibility
    def sign_transaction(self, transaction: TxParams) -> Tuple[HexBytes, HexStr]:
        """Signs the transaction using the loaded private key."""
        log_extra = {"component": "TransactionCore", "nonce": transaction.get("nonce")} # A16
        logger.debug("Signing transaction with nonce %s...", transaction.get("nonce"), extra=log_extra)
        try:
            # Clean transaction dict for sign_transaction (expects specific fields)
            # TxParams type hint usually handles this, but double-check if issues arise
            signed_tx = Account.sign_transaction(transaction, self.account.key)
            tx_hash = signed_tx.hash.hex()
            log_extra_signed = {**log_extra, "tx_hash": tx_hash} # A16
            # A2: Parameterized logging
            logger.debug("Transaction signed successfully. Hash: %s", tx_hash, extra=log_extra_signed)
            return signed_tx.rawTransaction, tx_hash

        except Exception as e:
            logger.error("Transaction signing failed: %s", e, exc_info=True, extra=log_extra)
            raise # Propagate signing errors


    async def send_raw_transaction(self, raw_tx: HexBytes, tx_hash: HexStr) -> HexStr:
        """Sends a signed raw transaction and returns the transaction hash."""
        log_extra = {"component": "TransactionCore", "tx_hash": tx_hash} # A16
        try:
            # A2: Parameterized logging
            logger.debug("Sending raw transaction %s...", tx_hash, extra=log_extra)
            sent_tx_hash_bytes: HexBytes = await self.web3.eth.send_raw_transaction(raw_tx)
            sent_tx_hash = sent_tx_hash_bytes.hex()
            if sent_tx_hash != tx_hash:
                 # This should ideally not happen if signing and hashing are correct
                 logger.warning("Sent TX hash %s differs from calculated hash %s!", sent_tx_hash, tx_hash, extra=log_extra)
            logger.info("Transaction %s sent successfully to node.", sent_tx_hash, extra=log_extra)
            return sent_tx_hash
        except ValueError as e:
             # Common errors: 'nonce too low', 'insufficient funds', 'gas price too low', 'already known'
             logger.error("Failed to send transaction %s (ValueError): %s", tx_hash, e, extra=log_extra)
             # Handle specific errors if needed (e.g., nonce too low might require nonce refresh)
             if "nonce too low" in str(e):
                  logger.warning("Nonce too low detected for tx %s. Consider refreshing nonce.", tx_hash, extra=log_extra)
                  # Optionally trigger nonce refresh: await self.noncecore.refresh_nonce()
             raise # Re-raise to indicate failure
        except Exception as e:
             # Network errors, etc.
             logger.error("Unexpected error sending transaction %s: %s", tx_hash, e, exc_info=True, extra=log_extra)
             raise


    async def simulate_transaction(self, transaction: TxParams) -> bool:
        """
        Simulates the transaction using eth_call to check for reverts.

        Args:
            transaction: The transaction parameters dictionary (unsigned).

        Returns:
            True if the simulation succeeds (doesn't revert), False otherwise.
        """
        # Prepare transaction for eth_call (needs 'from', 'to', 'value', 'data')
        sim_tx = transaction.copy()
        sim_tx['from'] = self.account.address # Ensure 'from' is set
        # Remove fields not used by eth_call
        sim_tx.pop("nonce", None)
        sim_tx.pop("gas", None)
        sim_tx.pop("gasPrice", None)
        sim_tx.pop("maxFeePerGas", None)
        sim_tx.pop("maxPriorityFeePerGas", None)
        sim_tx.pop("chainId", None)

        tx_info = f"To: {sim_tx.get('to')}, Value: {sim_tx.get('value', 0)}, Nonce: {transaction.get('nonce', 'N/A')}"
        log_extra = {"component": "TransactionCore", "tx_info": tx_info} # A16
        logger.debug("Simulating transaction (%s)...", tx_info, extra=log_extra)

        try:
            # Use 'pending' block identifier for mempool simulation if supported, else 'latest'
            # Some nodes might not fully support 'pending' simulation state.
            await self.web3.eth.call(sim_tx, block_identifier="pending")
            logger.debug("Transaction simulation successful (did not revert).", extra=log_extra)
            return True
        except ContractLogicError as e:
            # Simulation failed - transaction would revert
            logger.warning("Transaction simulation failed: Revert detected. Reason: %s", e, extra=log_extra)
            return False
        except ValueError as e:
             # Often indicates node error or invalid input for eth_call
             logger.warning("Transaction simulation failed (ValueError): %s", e, extra=log_extra)
             return False # Treat as failure
        except Exception as e:
            # Network errors, etc.
            logger.error("Unexpected error during transaction simulation: %s", e, exc_info=True, extra=log_extra)
            return False # Treat as failure


    async def execute_transaction(
        self,
        tx_params: TxParams,
        simulate: bool = True # Control whether to simulate before sending
        ) -> Optional[str]:
        """
        Builds, signs, optionally simulates, and sends a transaction with retries.

        Args:
            tx_params: Transaction parameters (may lack nonce, gas).
            simulate: Whether to simulate the transaction before sending.

        Returns:
            The transaction hash (hex string) if successfully sent, otherwise None.
        """
        tx_hash: Optional[str] = None
        log_extra = {"component": "TransactionCore"} # A16 Base extra
        max_retries = self.configuration.get_config_value("TX_EXECUTION_RETRIES", 1) # Configurable retries for execution itself
        retry_delay = self.configuration.get_config_value("TX_EXECUTION_RETRY_DELAY", 1) # Delay between retries

        for attempt in range(max_retries):
            log_extra_attempt = {**log_extra, "attempt": attempt + 1}
            try:
                full_tx_params = tx_params
                nonce = full_tx_params.get("nonce", "N/A")
                log_extra_attempt["nonce"] = nonce


                # 2. Simulate (Optional)
                if simulate:
                    sim_success = await self.simulate_transaction(full_tx_params)
                    if not sim_success:
                         logger.error("Transaction simulation failed for nonce %s. Aborting execution.", nonce, extra=log_extra_attempt)
                         # Don't retry simulation failures, the logic is likely flawed
                         return None # Abort

                # 3. Sign Transaction
                raw_tx, tx_hash = self.sign_transaction(full_tx_params)
                log_extra_attempt["tx_hash"] = tx_hash


                # 4. Send Transaction
                sent_hash = await self.send_raw_transaction(raw_tx, tx_hash)

                # 5. Track Nonce (Important!)
                # Schedule tracking, don't block execution waiting for receipt here
                asyncio.create_task(self.noncecore.track_transaction(sent_hash, cast(Nonce, nonce)))
                logger.debug("Scheduled nonce tracking for tx %s (Nonce %s).", sent_hash, nonce, extra=log_extra_attempt)

                return sent_hash # Success

            except (ValueError, Web3Exception) as send_error:
                 # Catch errors from send_raw_transaction specifically
                 logger.error("Attempt %d/%d: Failed to send transaction (Nonce %s, Hash %s): %s",
                             attempt + 1, max_retries, nonce, tx_hash or "N/A", send_error, extra=log_extra_attempt)
                 # Check for nonce errors and potentially retry or refresh
                 if "nonce too low" in str(send_error):
                      logger.warning("Nonce conflict detected. Refreshing nonce before retry...", extra=log_extra_attempt)
                      await self.noncecore.refresh_nonce() # Refresh aggressively
                      # Continue to next attempt after delay
                 elif "replacement transaction underpriced" in str(send_error):
                      logger.warning("Replacement transaction underpriced. Consider increasing gas for tx %s.", tx_hash, extra=log_extra_attempt)
                      # Decide whether to retry with higher gas or abort
                      return None # Abort for now

                 # Retry other ValueErrors after delay
                 if attempt < max_retries - 1:
                      await asyncio.sleep(retry_delay * (1.5**attempt))
                 else:
                      logger.error("Aborting execution for nonce %s after %d failed attempts.", nonce, max_retries, extra=log_extra_attempt)
                      return None # Failed after retries

            except Exception as e:
                # Catch errors from build, simulate, sign
                logger.error("Attempt %d/%d: Unexpected error during transaction execution (Nonce %s): %s",
                             attempt + 1, max_retries, nonce, e, exc_info=True, extra=log_extra_attempt)
                # Don't retry unexpected errors usually
                return None # Abort

        return None # Should be unreachable if loop logic is correct


    async def cancel_transaction(self, nonce: Nonce) -> bool:
        """
        Attempts to cancel a pending transaction by sending a zero-value tx
        with the same nonce and higher gas price.
        """
        log_extra = {"component": "TransactionCore", "nonce": nonce} # A16
        logger.warning("Attempting to cancel transaction with nonce %d...", nonce, extra=log_extra)
        try:
            # Determine gas price for cancellation (e.g., 10-20% higher than current dynamic price)
            current_gas_gwei = await self.safetynet.get_dynamic_gas_price()
            cancel_gas_price_gwei = max(current_gas_gwei * Decimal("1.2"), Decimal(self.cancel_gas_gwei)) # 20% boost or config default
            # Ensure it doesn't exceed absolute max
            cancel_gas_price_gwei = min(cancel_gas_price_gwei, self.safetynet._max_gas_price_gwei)
            cancel_gas_price_wei = self.web3.to_wei(cancel_gas_price_gwei, 'gwei')

            # Build zero-value transaction to self
            cancel_tx_params: TxParams = {
                "from": self.account.address,
                "to": self.account.address,
                "value": Wei(0),
                "nonce": nonce,
                "gas": Wei(self._base_gas_limit), # Use base gas limit (21000)
                "chainId": await self.web3.eth.chain_id,
                "gasPrice": Wei(cancel_gas_price_wei),
            }

            # Sign and send
            raw_tx, tx_hash = self.sign_transaction(cancel_tx_params)
            log_extra["tx_hash"] = tx_hash # Add hash for logging
            sent_hash = await self.send_raw_transaction(raw_tx, tx_hash)

            logger.info("Cancellation transaction %s sent for nonce %d.", sent_hash, nonce, extra=log_extra)
            # Track the cancellation tx itself (optional, depends on nonce logic)
            # asyncio.create_task(self.noncecore.track_transaction(sent_hash, nonce))
            return True

        except Exception as e:
            logger.error("Failed to send cancellation transaction for nonce %d: %s", nonce, e, exc_info=True, extra=log_extra)
            return False
    async def _calculate_profit_from_receipt(
         self,
         receipt: TxReceipt,
         # Add parameters needed to determine profit, e.g.:
         input_token_addr: Optional[str],
         output_token_addr: Optional[str],
         amount_in: Decimal, # Amount spent (e.g., in ETH or token units)
         expected_amount_out: Decimal # Expected amount received
         # Or pass value changes if simpler
         # value_change_eth: Decimal
         ) -> Decimal:
         """Calculates profit based on tx receipt and strategy details."""
         log_extra = {"component": "TransactionCore", "tx_hash": receipt.transactionHash.hex()}
         logger.debug("Calculating profit for transaction %s...", receipt.transactionHash.hex(), extra=log_extra)

         try:
              gas_used = Decimal(receipt.gasUsed)
              # Handle legacy vs EIP-1559 gas price
              effective_gas_price_wei = Decimal(receipt.get("effectiveGasPrice", 0)) # Wei (EIP-1559+)
              if effective_gas_price_wei == 0 and "gasPrice" in receipt: # Fallback for older receipts?
                    effective_gas_price_wei = Decimal(receipt.get("gasPrice",0))

              if effective_gas_price_wei == 0:
                   logger.warning("Could not determine effective gas price from receipt %s", receipt.transactionHash.hex(), extra=log_extra)
                   gas_cost_eth = Decimal("0")
              else:
                   gas_cost_eth = (gas_used * effective_gas_price_wei) / WEI_PER_ETH
                   logger.debug("Gas cost for tx %s: %.18f ETH", receipt.transactionHash.hex(), gas_cost_eth, extra=log_extra)
              profit_eth = -gas_cost_eth

              if input_token_addr and output_token_addr and amount_in and expected_amount_out:
                   input_price = await self.apiconfig.get_real_time_price(input_token_addr)
                   output_price = await self.apiconfig.get_real_time_price(output_token_addr)

                   if input_price and output_price:
                        value_in = amount_in * Decimal(str(input_price))
                        value_out = expected_amount_out * Decimal(str(output_price))
                        profit_eth = value_out - value_in - gas_cost_eth
                        logger.debug("Calculated profit for tx %s: ValueIn=%.18f, ValueOut=%.18f, GasCost=%.18f, Profit=%.18f ETH",
                                     receipt.transactionHash.hex(), value_in, value_out, gas_cost_eth, profit_eth, extra=log_extra)
                   else:
                        logger.warning("Could not fetch prices for input/output tokens to calculate profit.", extra=log_extra)

              logger.debug("Estimated profit for tx %s: %.18f ETH", receipt.transactionHash.hex(), profit_eth, extra=log_extra)
              return profit_eth

         except Exception as e:
              logger.error("Error calculating profit for receipt %s: %s", receipt.transactionHash.hex(), e, exc_info=True, extra=log_extra)
              return Decimal("0")

    # --- Helper Methods (e.g., Profit Transfer) ---

    # A14: Use Wei-native math where possible
    async def transfer_profit_to_account(self, amount_wei: Wei, target_account: str) -> bool:
        """
        Transfers specified amount (in Wei) from the bot's account to another account.

        Args:
            amount_wei: The amount to transfer in Wei.
            target_account: The recipient Ethereum address.

        Returns:
            True if the transfer transaction was sent successfully, False otherwise.
        """
        log_extra = {"component": "TransactionCore", "target_account": target_account, "amount_wei": amount_wei} # A16
        if amount_wei <= 0:
             logger.warning("Transfer amount must be positive, got %d Wei.", amount_wei, extra=log_extra)
             return False

        logger.info("Attempting to transfer %d Wei (%.8f ETH) to %s...",
                    amount_wei, Decimal(amount_wei)/WEI_PER_ETH, target_account, extra=log_extra)
        try:
            # Build simple ETH transfer transaction
            transfer_tx = await self.build_transaction(
                to=target_account,
                value=amount_wei,
                gas_limit_override=self._base_gas_limit # Use base gas for ETH transfer
            )

            # Execute the transaction
            tx_hash = await self.execute_transaction(transfer_tx, simulate=False) # No need to simulate simple transfer usually

            if tx_hash:
                log_extra["tx_hash"] = tx_hash
                logger.info("Profit transfer transaction sent successfully: %s", tx_hash, extra=log_extra)
                return True
            else:
                logger.error("Profit transfer transaction failed to send.", extra=log_extra)
                return False
        except Exception as e:
            logger.error("Error during profit transfer to %s: %s", target_account, e, exc_info=True, extra=log_extra)
            return False

    async def get_transaction_receipt(self, tx_hash: str, timeout: int = 120) -> Optional[TxReceipt]:
        """Waits for and returns the transaction receipt."""
        log_extra = {"component": "TransactionCore", "tx_hash": tx_hash} # A16
        try:
            receipt = await self.web3.eth.wait_for_transaction_receipt(HexBytes(tx_hash), timeout=timeout)
            if receipt.status == 1:
                logger.debug("Transaction %s confirmed successfully (Block: %d).", tx_hash, receipt.blockNumber, extra=log_extra)
            else:
                logger.warning("Transaction %s confirmed but failed (reverted) (Block: %d).", tx_hash, receipt.blockNumber, extra=log_extra)
            return receipt
        except asyncio.TimeoutError:
            logger.warning("Timeout (%ds) waiting for transaction receipt for %s.", timeout, tx_hash, extra=log_extra)
            return None
        except TransactionNotFound:
            logger.error("Transaction %s not found while waiting for receipt (orphaned?).", tx_hash, extra=log_extra)
            return None
        except Exception as e:
             logger.error("Error waiting for transaction receipt %s: %s", tx_hash, e, exc_info=True, extra=log_extra)
             return None


    async def _execute_strategy_transaction(
        self,
        strategy_name: str,
        tx_params: TxParams,
        target_tx_hash: Optional[str] = "N/A" # Hash of the tx being targeted by MEV
    ) -> Tuple[bool, Optional[str], Optional[TxReceipt]]:
        """Helper to execute a single transaction for a strategy and get receipt."""
        nonce = tx_params.get("nonce", "N/A")
        log_extra = {"component": "TransactionCore", "strategy": strategy_name, "target_tx_hash": target_tx_hash, "nonce": nonce}

        tx_hash = await self.execute_transaction(tx_params, simulate=True) # Simulate strategy txs

        if not tx_hash:
            logger.warning("%s strategy failed: Transaction execution failed for nonce %s.", strategy_name, nonce, extra=log_extra)
            return False, None, None

        log_extra["tx_hash"] = tx_hash # Add own hash
        logger.info("%s strategy transaction sent: %s (Nonce: %s)", strategy_name, tx_hash, nonce, extra=log_extra)

        # Wait for receipt to confirm success/failure and get gas used
        receipt = await self.get_transaction_receipt(tx_hash)

        if receipt is None:
            logger.warning("%s strategy uncertain: No receipt received for %s.", strategy_name, tx_hash, extra=log_extra)
            # Consider if this should be True or False. Assume False if outcome unknown.
            return False, tx_hash, None
        elif receipt.status == 0:
            logger.warning("%s strategy failed: Transaction %s reverted.", strategy_name, tx_hash, extra=log_extra)
            return False, tx_hash, receipt
        else:
            logger.info("%s strategy transaction %s confirmed.", strategy_name, tx_hash, extra=log_extra)
            return True, tx_hash, receipt


    async def _calculate_profit_from_receipt(
         self,
         receipt: TxReceipt,
         # Add parameters needed to determine profit, e.g.:
         input_token_addr: Optional[str],
         output_token_addr: Optional[str],
         amount_in: Decimal, # Amount spent (e.g., in ETH or token units)
         expected_amount_out: Decimal # Expected amount received
         # Or pass value changes if simpler
         # value_change_eth: Decimal
         ) -> Decimal:
         """Calculates profit based on tx receipt and strategy details."""
         log_extra = {"component": "TransactionCore", "tx_hash": receipt.transactionHash.hex()}
         logger.debug("Calculating profit for transaction %s...", receipt.transactionHash.hex(), extra=log_extra)

         try:
              gas_used = Decimal(receipt.gasUsed)
              # Handle legacy vs EIP-1559 gas price
              effective_gas_price_wei = Decimal(receipt.get("effectiveGasPrice", 0)) # Wei (EIP-1559+)
              if effective_gas_price_wei == 0 and "gasPrice" in receipt: # Fallback for older receipts?
                    effective_gas_price_wei = Decimal(receipt.get("gasPrice",0))

              if effective_gas_price_wei == 0:
                   logger.warning("Could not determine effective gas price from receipt %s", receipt.transactionHash.hex(), extra=log_extra)
                   gas_cost_eth = Decimal("0")
              else:
                   gas_cost_eth = (gas_used * effective_gas_price_wei) / WEI_PER_ETH

              # --- Implement actual profit calculation based on logs/token transfers ---
              # This is highly strategy-dependent.  Example:
              # 1. If it's a simple ETH transfer, profit is -gas_cost.
              # 2. If it's a token swap, analyze Transfer events to determine input/output amounts.
              # 3. Get prices of input/output tokens around the execution time (using APIConfig).
              # 4. Calculate profit = (Value Out) - (Value In) - (Gas Cost).

              # Placeholder: Simulate profit = -(gas_cost) if no other info
              profit_eth = -gas_cost_eth

              # Example: If input and output tokens are defined, attempt to calculate profit
              if input_token_addr and output_token_addr and amount_in and expected_amount_out:
                   # Get current prices for input and output tokens
                   input_price = await self.apiconfig.get_real_time_price(input_token_addr)
                   output_price = await self.apiconfig.get_real_time_price(output_token_addr)

                   if input_price and output_price:
                        value_in = amount_in * Decimal(str(input_price))
                        value_out = expected_amount_out * Decimal(str(output_price))
                        profit_eth = value_out - value_in - gas_cost_eth
                        logger.debug("Calculated profit for tx %s: ValueIn=%.18f, ValueOut=%.18f, GasCost=%.18f, Profit=%.18f ETH",
                                     receipt.transactionHash.hex(), value_in, value_out, gas_cost_eth, profit_eth, extra=log_extra)
                   else:
                        logger.warning("Could not fetch prices for input/output tokens to calculate profit.", extra=log_extra)

              logger.debug("Estimated profit for tx %s: %.18f ETH", receipt.transactionHash.hex(), profit_eth, extra=log_extra)
              return profit_eth

         except Exception as e:
              logger.error("Error calculating profit for receipt %s: %s", receipt.transactionHash.hex(), e, exc_info=True, extra=log_extra)
              return Decimal("0")


    async def handle_eth_transaction(self, target_tx: Dict[str, Any]) -> StrategyResult:
        """
        Example Strategy: Handles a simple ETH transfer.
        For MEV, this would likely involve front-running or other interaction.
        This implementation just sends a similar transaction.
        """
        strategy_name = "handle_eth_transaction"
        target_tx_hash = target_tx.get("tx_hash", "N/A")
        log_extra = {"component": "TransactionCore", "strategy": strategy_name, "target_tx_hash": target_tx_hash} # A16

        try:
            to_addr = target_tx.get("to")
            value_wei = target_tx.get("value", 0) # Expect Wei

            if not to_addr or value_wei <= 0:
                logger.debug("Invalid target tx data for %s. Skipping.", strategy_name, extra=log_extra)
                return False, Decimal("0") # Failed

            # Build a simple ETH transfer tx
            tx_params = await self.build_transaction(
                to=to_addr,
                value=Wei(value_wei),
                gas_limit_override=self._base_gas_limit # Use base gas for simple transfer
            )
            log_extra["nonce"] = tx_params.get("nonce")

            # Execute and get receipt
            success, tx_hash, receipt = await self._execute_strategy_transaction(strategy_name, tx_params, target_tx_hash)

            if success and receipt:
                # Calculate profit (for simple transfer, profit is likely negative gas cost)
                profit = await self._calculate_profit_from_receipt(receipt)
                return True, profit
            else:
                # Execution or confirmation failed
                return False, Decimal("0")

        except Exception as e:
            logger.error("Error in %s strategy: %s", strategy_name, e, exc_info=True, extra=log_extra)
            return False, Decimal("0") # Failed


    async def execute_sandwich_attack(self, target_tx: Dict[str, Any]) -> StrategyResult:
        """Execute a sandwich attack on a DEX trade by front-running and back-running."""
        strategy_name = "execute_sandwich_attack"
        target_tx_hash = target_tx.get("tx_hash", "N/A")
        log_extra = {"component": "TransactionCore", "strategy": strategy_name, "target_tx_hash": target_tx_hash}

        try:
            # 1. Decode target transaction (assume Uniswap V2 swap)
            if not self.uniswap_router_contract:
                logger.error("Uniswap router contract not initialized", extra=log_extra)
                return False, Decimal("0")

            decoded_input = self.uniswap_router_contract.decode_function_input(target_tx.get("input", "0x"))
            if not decoded_input or "function" not in decoded_input:
                logger.error("Could not decode target transaction input", extra=log_extra)
                return False, Decimal("0")

            # Get token addresses and amounts from decoded input
            path = decoded_input[1].get("path", [])
            if len(path) < 2:
                return False, Decimal("0")

            token_in, token_out = path[0], path[-1]
            amount_in = decoded_input[1].get("amountIn", 0)

            # 2. Calculate optimal amounts for sandwich
            # Use MarketMonitor to check liquidity and price impact
            if self.marketmonitor:
                optimal_amount = await self.marketmonitor.calculate_optimal_sandwich_amount(
                    token_in, token_out, amount_in
                )
            else:
                logger.error("MarketMonitor required for sandwich attacks", extra=log_extra)
                return False, Decimal("0")

            if not optimal_amount or optimal_amount <= 0:
                logger.debug("No profitable sandwich opportunity found", extra=log_extra)
                return False, Decimal("0")

            # 3. Build front-run transaction (buy tokens)
            front_run_data = self.uniswap_router_contract.encode_abi(
                fn_name="swapExactETHForTokens",
                args=[
                    optimal_amount,  # amountOutMin
                    path,
                    self.account.address,
                    int(time.time()) + 60  # deadline
                ]
            )

            front_tx = await self.build_transaction(
                to=self.uniswap_router_contract.address,
                value=Wei(optimal_amount),
                data=front_run_data,
                gas_params_override={"maxFeePerGas": Wei(int(target_tx.get("maxFeePerGas", 0) * 1.2))}
            )

            # 4. Execute front-run
            success_front, tx_hash_front, receipt_front = await self._execute_strategy_transaction(
                f"{strategy_name}_front", front_tx, target_tx_hash
            )

            if not success_front:
                return False, Decimal("0")

            # 5. Wait for target transaction confirmation
            await self.web3.eth.wait_for_transaction_receipt(target_tx_hash)

            # 6. Build and execute back-run (sell tokens)
            # Get balance of acquired tokens
            token_contract = self.get_erc20_contract(token_out)
            if not token_contract:
                return False, Decimal("0")

            balance = await token_contract.functions.balanceOf(self.account.address).call()
            
            back_run_data = self.uniswap_router_contract.encodeABI(
                fn_name="swapExactTokensForETH",
                args=[
                    balance,  # amountIn
                    0,  # amountOutMin 
                    path[::-1],  # Reverse path
                    self.account.address,
                    int(time.time()) + 60  # deadline
                ]
            )

            back_tx = await self.build_transaction(
                to=self.uniswap_router_contract.address,
                data=back_run_data
            )

            success_back, tx_hash_back, receipt_back = await self._execute_strategy_transaction(
                f"{strategy_name}_back", back_tx, target_tx_hash
            )

            if not success_back:
                return False, Decimal("0")

            # 7. Calculate total profit
            if receipt_front and receipt_back:
                profit_front = await self._calculate_profit_from_receipt(receipt_front,
                    token_in, token_out, Decimal(str(optimal_amount)), Decimal("0"))
                profit_back = await self._calculate_profit_from_receipt(receipt_back,
                    token_out, token_in, Decimal(str(balance)), Decimal("0"))
                total_profit = profit_front + profit_back

                logger.info("Sandwich attack completed. Profit: %.6f ETH", total_profit, extra=log_extra)
                return True, total_profit

            return False, Decimal("0")

        except Exception as e:
            logger.error("Sandwich attack failed: %s", str(e), exc_info=True, extra=log_extra)
            return False, Decimal("0")

    async def front_run(self, target_tx: Dict[str, Any]) -> StrategyResult:
        """Execute a front-running strategy by placing a similar transaction with higher gas."""
        strategy_name = "front_run" 
        target_tx_hash = target_tx.get("tx_hash", "N/A")
        log_extra = {"component": "TransactionCore", "strategy": strategy_name, "target_tx_hash": target_tx_hash}

        try:
            # 1. Analyze target transaction
            target_value = Wei(target_tx.get("value", 0))
            target_gas_price = Wei(target_tx.get("gasPrice", 0))
            target_max_fee = Wei(target_tx.get("maxFeePerGas", 0))

            # Skip if transaction value is too low
            min_value = Wei(self.web3.to_wei(0.1, 'ether'))  # Example threshold
            if target_value < min_value:
                logger.debug("Target value too low for front-running", extra=log_extra)
                return False, Decimal("0")

            # 2. Build front-run transaction
            gas_params_override = {}
            if target_max_fee > 0:
                gas_params_override["maxFeePerGas"] = Wei(int(target_max_fee * 1.2))
                gas_params_override["maxPriorityFeePerGas"] = Wei(int(target_tx.get("maxPriorityFeePerGas", 0) * 1.2))
            elif target_gas_price > 0:
                gas_params_override["gasPrice"] = Wei(int(target_gas_price * 1.2))

            tx_params = await self.build_transaction(
                to=target_tx.get("to"),
                value=target_value,
                data=target_tx.get("input", "0x"),
                gas_params_override=gas_params_override
            )

            # 3. Execute transaction
            success, tx_hash, receipt = await self._execute_strategy_transaction(
                strategy_name, tx_params, target_tx_hash
            )

            if success and receipt:
                profit = await self._calculate_profit_from_receipt(receipt)
                logger.info("Front-run executed. Profit: %.6f ETH", profit, extra=log_extra)
                return True, profit

            return False, Decimal("0")

        except Exception as e:
            logger.error("Front-run failed: %s", str(e), exc_info=True, extra=log_extra)
            return False, Decimal("0")

    # Additional strategy implementations removed to fit character limit

    async def stop(self) -> None:
        """Gracefully stop TransactionCore (if needed, e.g., close resources)."""
        # Currently, no specific resources to close here, but placeholder for future use.
        log_extra = {"component": "TransactionCore"} # A16
        logger.info("TransactionCore stopping.", extra=log_extra)
        # No explicit stop actions needed currently
        logger.info("TransactionCore stopped.", extra=log_extra)