import asyncio
import hexbytes
from web3 import AsyncWeb3
from web3.exceptions import TransactionNotFound, ContractLogicError
from eth_account import Account
from typing import Any, Dict, List, Optional
from decimal import Decimal

from abiregistry import ABIRegistry
from apiconfig import APIConfig
from configuration import Configuration
from marketmonitor import MarketMonitor
from mempoolmonitor import MempoolMonitor  
from noncecore import NonceCore
from safetynet import SafetyNet
from loggingconfig import setup_logging
import logging

logger = setup_logging("TransactionCore", level=logging.DEBUG)


class TransactionCore:
    """
    Main transaction engine responsible for building, signing, simulating,
    and executing transactions for MEV strategies.

    In addition to the basic execution and signing methods, this class now
    implements several MEV strategy functions:
      - front_run
      - aggressive_front_run
      - predictive_front_run
      - volatility_front_run
      - back_run
      - price_dip_back_run
      - flashloan_back_run
      - high_volume_back_run

    Each strategy uses a combination of market data, dynamic gas pricing,
    and transaction parameters to decide whether or not to execute a trade.
    """
    DEFAULT_GAS_LIMIT: int = 100_000
    DEFAULT_PROFIT_TRANSFER_MULTIPLIER: Decimal = Decimal("1e18")
    DEFAULT_GAS_PRICE_GWEI: int = 50

    def __init__(
        self,
        web3: AsyncWeb3,
        account: Account,
        configuration: Configuration,
        apiconfig: Optional[APIConfig] = None,
        marketmonitor: Optional[MarketMonitor] = None,
        mempoolmonitor: Optional[MempoolMonitor] = None,
        noncecore: Optional[NonceCore] = None,
        safetynet: Optional[SafetyNet] = None,
        gas_price_multiplier: float = 1.1,
        erc20_abi: Optional[List[Dict[str, Any]]] = None,
        uniswap_address: Optional[str] = None,
        uniswap_abi: Optional[List[Dict[str, Any]]] = None
    ):
        self.web3: AsyncWeb3 = web3
        self.account: Account = account
        self.configuration: Configuration = configuration
        self.apiconfig: Optional[APIConfig] = apiconfig
        self.marketmonitor: Optional[MarketMonitor] = marketmonitor
        self.mempoolmonitor: Optional[MempoolMonitor] = mempoolmonitor
        self.noncecore: Optional[NonceCore] = noncecore
        self.safetynet: Optional[SafetyNet] = safetynet
        self.gas_price_multiplier = gas_price_multiplier
        self.erc20_abi: List[Dict[str, Any]] = erc20_abi or []
        self.current_profit: Decimal = Decimal("0")
        self.abiregistry: ABIRegistry = ABIRegistry()
        self.uniswap_address: Optional[str] = uniswap_address
        self.uniswap_abi: List[Dict[str, Any]] = uniswap_abi or []
        self.uniswap_router_contract = None 
        self.sushiswap_router_contract = None

    async def initialize(self) -> None:
        """
        Initialize the transaction engine by loading ABIs,
        deploying key contracts, and validating them.
        """
        try:
            await self.abiregistry.initialize(self.configuration.BASE_PATH)
            aave_flashloan_abi = self.abiregistry.get_abi('aave_flashloan')
            aave_pool_abi = self.abiregistry.get_abi('aave')
            uniswap_abi = self.abiregistry.get_abi('uniswap')
            sushiswap_abi = self.abiregistry.get_abi('sushiswap')

            # Initialize Aave flashloan contract
            self.aave_flashloan = self.web3.eth.contract(
                address=self.web3.to_checksum_address(self.configuration.AAVE_FLASHLOAN_ADDRESS),
                abi=aave_flashloan_abi
            )
            await self._validate_contract(self.aave_flashloan, "Aave Flashloan")

            # Initialize Aave pool contract
            self.aave_pool = self.web3.eth.contract(
                address=self.web3.to_checksum_address(self.configuration.AAVE_POOL_ADDRESS),
                abi=aave_pool_abi
            )
            await self._validate_contract(self.aave_pool, "Aave Pool")

            # Initialize Uniswap router contract if configured
            if uniswap_abi and self.configuration.UNISWAP_ADDRESS:
                self.uniswap_router_contract = self.web3.eth.contract(
                    address=self.web3.to_checksum_address(self.configuration.UNISWAP_ADDRESS),
                    abi=uniswap_abi
                )
                await self._validate_contract(self.uniswap_router_contract, "Uniswap Router")
            else:
                logger.warning("Uniswap settings not fully configured.")

            # Initialize Sushiswap router contract if configured
            if sushiswap_abi and self.configuration.SUSHISWAP_ADDRESS:
                self.sushiswap_router_contract = self.web3.eth.contract(
                    address=self.web3.to_checksum_address(self.configuration.SUSHISWAP_ADDRESS),
                    abi=sushiswap_abi
                )
                await self._validate_contract(self.sushiswap_router_contract, "Sushiswap Router")
            else:
                logger.warning("Sushiswap settings not fully configured.")

            logger.info("TransactionCore initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing TransactionCore: {e}", exc_info=True)
            raise

    async def _validate_contract(self, contract: Any, name: str) -> None:
        """
        Validate that a contract is properly deployed by checking code existence.
        """
        try:
            code = await self.web3.eth.get_code(contract.address)
            if code and code != "0x":
                logger.debug(f"{name} validated (code exists).")
            else:
                raise ValueError(f"{name} code not found at address {contract.address}")
        except Exception as e:
            logger.error(f"Error validating {name}: {e}")
            raise

    async def build_transaction(self, function_call: Any, additional_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Build a transaction with dynamic gas parameters and correct nonce.
        """
        additional_params = additional_params or {}
        try:
            chain_id = await self.web3.eth.chain_id
            latest_block = await self.web3.eth.get_block("latest")
            supports_eip1559 = "baseFeePerGas" in latest_block
            nonce = await self.noncecore.get_nonce()

            tx_params = {
                "chainId": chain_id,
                "nonce": nonce,
                "from": self.account.address,
                "gas": self.DEFAULT_GAS_LIMIT  # Add default gas
            }
            if supports_eip1559:
                base_fee = latest_block["baseFeePerGas"]
                priority_fee = await self.web3.eth.max_priority_fee
                tx_params.update({
                    "maxFeePerGas": int(base_fee * 2),
                    "maxPriorityFeePerGas": int(priority_fee)
                })
            else:
                gas_price = await self.safetynet.get_dynamic_gas_price()
                tx_params["gasPrice"] = int(self.web3.to_wei(gas_price * self.gas_price_multiplier, "gwei"))

            tx_details = function_call.buildTransaction(tx_params)
            tx_details.update(additional_params)
            estimated_gas = await self.estimate_gas_smart(tx_details)
            tx_details["gas"] = int(estimated_gas * 1.1)
            return tx_details
        except Exception as e:
            logger.error(f"Error in build_transaction: {e}", exc_info=True)
            raise

    async def estimate_gas_smart(self, tx: Dict[str, Any]) -> int:
        """
        Estimate gas required for a transaction with fallback to a default limit.
        """
        try:
            gas_estimate = await self.web3.eth.estimate_gas(tx)
            logger.debug(f"Estimated gas: {gas_estimate}")
            return gas_estimate
        except (ContractLogicError, TransactionNotFound) as e:
            logger.error(f"Gas estimation error: {e}")
            return self.DEFAULT_GAS_LIMIT
        except Exception as e:
            logger.error(f"Unexpected gas estimation error: {e}", exc_info=True)
            return self.DEFAULT_GAS_LIMIT

    def _clean_transaction_data(self, tx: Dict[str, Any]) -> Dict[str, Any]:
        """Remove non-transaction fields from the transaction dict."""
        valid_fields = {
            'nonce', 'gas', 'gasPrice', 'maxFeePerGas', 'maxPriorityFeePerGas',
            'to', 'value', 'data', 'chainId', 'from', 'type'
        }
        cleaned_tx = {k: v for k, v in tx.items() if k in valid_fields}
        # Chain ID is required, fetch it if missing
        return cleaned_tx

    async def sign_transaction(self, transaction: Dict[str, Any]) -> bytes:
        """Sign the given transaction using the account's private key."""
        try:
            # Clean the transaction data and ensure chainId is set
            clean_tx = self._clean_transaction_data(transaction)
            if 'chainId' not in clean_tx:
                clean_tx['chainId'] = await self.web3.eth.chain_id
            
            signed_tx = self.web3.eth.account.sign_transaction(
                clean_tx,
                private_key=self.account.key,
            )
            logger.debug(f"Transaction signed (Nonce: {clean_tx['nonce']}, ChainId: {clean_tx['chainId']}).")
            return signed_tx.raw_transaction
        except Exception as e:
            logger.error(f"Transaction signing error: {e}", exc_info=True)
            raise

    async def call_contract_function(self, signed_tx: bytes) -> hexbytes.HexBytes:
        """
        Send a signed transaction and return the transaction hash.
        """
        try:
            tx_hash = await self.web3.eth.send_raw_transaction(signed_tx)
            logger.debug("Transaction sent successfully.")
            return tx_hash
        except Exception as e:
            logger.error(f"Error sending transaction: {e}", exc_info=True)
            raise

    async def simulate_transaction(self, transaction: Dict[str, Any]) -> bool:
        """
        Simulate the transaction to verify whether it would execute successfully.
        """
        logger.debug(f"Simulating transaction (Nonce: {transaction.get('nonce', 'unknown')}).")
        try:
            await self.web3.eth.call(transaction, block_identifier="pending")
            logger.debug("Transaction simulation succeeded.")
            return True
        except ContractLogicError as e:
            logger.error(f"Contract logic error during simulation: {e}")
            return False
        except Exception as e:
            logger.error(f"Error simulating transaction: {e}", exc_info=True)
            return False

    async def execute_transaction(self, tx: Dict[str, Any]) -> Optional[str]:
        """
        Execute a transaction with retries and gas price checks.
        """
        max_retries = self.configuration.MEMPOOL_MAX_RETRIES
        retry_delay = self.configuration.MEMPOOL_RETRY_DELAY
        
        # Ensure required fields are present
        if "nonce" not in tx:
            tx["nonce"] = await self.noncecore.get_nonce()
        if "gas" not in tx:
            tx["gas"] = self.DEFAULT_GAS_LIMIT

        for attempt in range(1, max_retries + 1):
            try:
                signed_tx = await self.sign_transaction(tx)
                tx_hash = await self.call_contract_function(signed_tx)
                logger.info(f"Transaction sent: {tx_hash.hex()}")
                await self.noncecore.refresh_nonce()
                return tx_hash.hex()
            except Exception as e:
                logger.error(f"Transaction execution attempt {attempt} failed: {e}", exc_info=True)
                await asyncio.sleep(retry_delay * (attempt + 1))
            gas_price_gwei = Decimal(tx.get("gasPrice", self.DEFAULT_GAS_PRICE_GWEI))
            if gas_price_gwei > Decimal(self.configuration.MAX_GAS_PRICE_GWEI):
                logger.warning(f"Gas price {gas_price_gwei} exceeds maximum threshold.")
                return None
        logger.error("Failed to execute transaction after retries.")
        return None

    async def cancel_transaction(self, nonce: int) -> bool:
        """
        Cancel a pending transaction by sending a zero-value transaction with the same nonce.
        """
        cancel_tx = {
            "to": self.account.address,
            "value": 0,
            "gas": 21000,
            "gasPrice": self.web3.to_wei(self.configuration.DEFAULT_CANCEL_GAS_PRICE_GWEI, "gwei"),
            "nonce": nonce,
            "chainId": await self.web3.eth.chain_id,
            "from": self.account.address,
        }
        try:
            signed_cancel_tx = await self.sign_transaction(cancel_tx)
            tx_hash = await self.web3.eth.send_raw_transaction(signed_cancel_tx)
            logger.debug(f"Cancellation tx sent: {tx_hash.hex()}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling tx (nonce {nonce}): {e}", exc_info=True)
            return False

    async def withdraw_eth(self) -> bool:
        """
        Withdraw ETH from the flashloan contract.
        """
        try:
            withdraw_function = self.aave_flashloan.functions.withdrawETH()
            tx = await self.build_transaction(withdraw_function)
            tx_hash = await self.execute_transaction(tx)
            if tx_hash:
                logger.info(f"ETH withdrawn successfully: {tx_hash}")
                return True
            else:
                logger.warning("ETH withdrawal failed.")
                return False
        except Exception as e:
            logger.error(f"Error withdrawing ETH: {e}", exc_info=True)
            return False

    async def transfer_profit_to_account(self, amount: Decimal, target_account: str) -> bool:
        """
        Transfer profit to a specified account.
        """
        try:
            transfer_function = self.aave_flashloan.functions.transfer(
                self.web3.to_checksum_address(target_account),
                int(amount * self.DEFAULT_PROFIT_TRANSFER_MULTIPLIER)
            )
            tx = await self.build_transaction(transfer_function)
            tx_hash = await self.execute_transaction(tx)
            if tx_hash:
                logger.info(f"Profit transferred: {tx_hash}")
                return True
            else:
                logger.warning("Profit transfer failed.")
                return False
        except Exception as e:
            logger.error(f"Error transferring profit to {target_account}: {e}", exc_info=True)
            return False

    #
    # Strategy Methods
    #
    async def handle_eth_transaction(self, target_tx: Dict[str, Any]) -> bool:
        """
        Process an ETH transfer transaction by building and executing a front-run style transaction.
        """
        try:
            tx_hash = target_tx.get("tx_hash", "unknown")
            # Validate that the transaction value is positive (in Wei)
            eth_value_wei = int(target_tx.get("value", 0))
            if eth_value_wei <= 0:
                logger.debug("ETH transaction value is zero or negative; skipping execution.")
                return False

            # Construct the transaction details.
            nonce = await self.noncecore.get_nonce()
            chain_id = await self.web3.eth.chain_id

            tx_details = {
                "to": target_tx.get("to", ""),  # Recipient should match the target transaction's 'to'
                "value": eth_value_wei,
                "nonce": nonce,
                "chainId": chain_id,
                "from": self.account.address,
            }

            # Set gas limit for standard ETH transfer.
            tx_details["gas"] = 21000

            # Determine gas price:
            # If the target transaction provides a gas price, use it multiplied by a safety factor.
            raw_gas_price = target_tx.get("gasPrice")
            if raw_gas_price:
                tx_details["gasPrice"] = int(int(raw_gas_price) * self.configuration.ETH_TX_GAS_PRICE_MULTIPLIER)
            else:
                # Otherwise, fetch the dynamic gas price.
                gas_price = await self.safetynet.get_dynamic_gas_price()
                tx_details["gasPrice"] = int(self.web3.to_wei(gas_price * self.gas_price_multiplier, "gwei"))

            # Log the transaction details.
            logger.debug(
                f"Handling ETH transaction:\n"
                f"  Original tx hash: {tx_hash}\n"
                f"  Value: {self.web3.from_wei(eth_value_wei, 'ether')} ETH\n"
                f"  To: {tx_details['to']}\n"
                f"  Nonce: {tx_details['nonce']}\n"
                f"  Gas Price: {tx_details.get('gasPrice')} (in Wei)"
            )

            # Execute the transaction.
            executed_tx_hash = await self.execute_transaction(tx_details)
            if executed_tx_hash:
                logger.info(f"Successfully executed ETH transaction with hash: {executed_tx_hash}")
                return True
            else:
                logger.warning("ETH transaction execution failed.")
                return False
        except Exception as e:
            logger.error(f"Error handling ETH transaction: {e}", exc_info=True)
            return False


    async def execute_sandwich_attack(self, target_tx: Dict[str, Any]) -> bool:
        """
        sandwich attack strategy 
        

        """
        try:
            logger.info(f"Starting sandwich attack for tx: {target_tx.get('tx_hash', 'unknown')}")
            
            # Step 1: Prepare flashloan transaction
            flashloan_asset = target_tx.get("asset")
            flashloan_amount = target_tx.get("flashloan_amount", 0)
            flashloan_tx = await self.prepare_flashloan_transaction(flashloan_asset, flashloan_amount)
            if flashloan_tx is None:
                logger.error("Failed to prepare flashloan transaction.")
                return False

            # Step 2: Simulate flashloan transaction
            flashloan_simulated = await self.simulate_transaction(flashloan_tx)
            if not flashloan_simulated:
                logger.error("Flashloan transaction simulation failed.")
                return False

            # Step 3: Execute flashloan transaction
            flashloan_result = await self.execute_transaction(flashloan_tx)
            if (flashloan_result is None):
                logger.error("Flashloan transaction execution failed.")
                return False

            logger.debug(f"Flashloan executed successfully: {flashloan_result}")

            # Step 4: Prepare & simulate front-run transaction
            front_run_simulated = await self.simulate_transaction(target_tx)
            if not front_run_simulated:
                logger.error("Front-run transaction simulation failed.")
                return False

            # Step 5: Execute front-run transaction
            front_run_result = await self.front_run(target_tx)
            if not front_run_result:
                logger.error("Front-run transaction execution failed.")
                return False

            logger.debug("Front-run executed successfully.")

            # Step 6: Prepare & simulate back-run transaction
            back_run_simulated = await self.simulate_transaction(target_tx)
            if not back_run_simulated:
                logger.error("Back-run transaction simulation failed.")
                return False

            # Step 7: Execute back-run transaction
            back_run_result = await self.back_run(target_tx)
            if not back_run_result:
                logger.error("Back-run transaction execution failed.")
                return False

            logger.info("Sandwich attack executed successfully: flashloan, front-run, and back-run all completed.")
            return True

        except Exception as e:
            logger.error(f"Sandwich attack encountered an error: {e}", exc_info=True)
            return False   


    async def front_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute a basic front-run strategy on the target transaction.
        """
        try:
            logger.debug(f"Executing front-run for tx: {target_tx.get('tx_hash', 'unknown')}")
            success = await self.execute_transaction(target_tx)
            return success is not None
        except Exception as e:
            logger.error(f"Front-run execution error: {e}", exc_info=True)
            return False

    async def aggressive_front_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute an aggressive front-run strategy by increasing the gas price multiplier.
        This strategy is only triggered if the transaction value exceeds a specified aggressive threshold.
        """
        try:
            aggressive_threshold = Decimal(self.configuration.AGGRESSIVE_FRONT_RUN_MIN_VALUE_ETH)
            tx_value_eth = Decimal(self.web3.from_wei(target_tx.get("value", 0), "ether"))
            if tx_value_eth < aggressive_threshold:
                logger.debug(f"Tx value {tx_value_eth} ETH below aggressive threshold {aggressive_threshold} ETH; skipping aggressive front-run.")
                return False

            original_multiplier = self.gas_price_multiplier
            aggressive_multiplier = original_multiplier * 1.5
            self.gas_price_multiplier = aggressive_multiplier
            logger.debug(f"Aggressive front-run: gas multiplier increased from {original_multiplier} to {aggressive_multiplier}.")
            result = await self.front_run(target_tx)
            self.gas_price_multiplier = original_multiplier
            return result
        except Exception as e:
            logger.error(f"Aggressive front-run failed: {e}", exc_info=True)
            return False

    async def predictive_front_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute a predictive front-run strategy based on price forecasts.
        Uses predicted price from APIConfig and compares it with the current price.
        If the expected increase exceeds a defined threshold (e.g. 5%), it executes a front-run.
        """
        try:
            output_token = target_tx.get("output_token")
            if not output_token:
                logger.debug("Missing output token; skipping predictive front-run.")
                return False
            
            current_price = await self.apiconfig.get_real_time_price(output_token)
            predicted_price = await self.apiconfig.predict_price(output_token)
            if current_price is None or predicted_price is None:
                logger.debug("Missing price data; skipping predictive front-run.")
                return False
            percentage_increase = (Decimal(predicted_price) - Decimal(current_price)) / Decimal(current_price)
            threshold = Decimal("0.05")  # 5% threshold
            logger.debug(f"Predictive front-run: expected increase {(percentage_increase*100):.2f}% (threshold {threshold*100:.2f}%).")
            if percentage_increase >= threshold:
                return await self.front_run(target_tx)
            else:
                logger.debug("Predicted increase insufficient; skipping predictive front-run.")
                return False
        except Exception as e:
            logger.error(f"Predictive front-run failed: {e}", exc_info=True)
            return False

    async def volatility_front_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute a volatility-based front-run strategy.
        If market conditions (as determined by MarketMonitor) indicate high volatility,
        the strategy will attempt to front-run.
        """
        try:
            if not self.marketmonitor:
                logger.error("MarketMonitor not initialized for volatility check")
                return False

            # Add null check for target address
            target_address = target_tx.get("to")
            if not target_address:
                logger.debug("Missing target address; skipping volatility front-run.")
                return False

            market_conditions = await self.marketmonitor.check_market_conditions(target_address)
            if market_conditions.get("high_volatility", False):
                logger.debug("High volatility detected; executing volatility front-run.")
                return await self.front_run(target_tx)
            else:
                logger.debug("Volatility not high; skipping volatility front-run.")
                return False
        except Exception as e:
            logger.error(f"Volatility front-run failed: {e}", exc_info=True)
            return False

    async def back_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute a basic back-run strategy on the target transaction.
        """
        try:
            logger.debug(f"Executing back-run for tx: {target_tx.get('tx_hash', 'unknown')}")
            success = await self.execute_transaction(target_tx)
            return success is not None
        except Exception as e:
            logger.error(f"Back-run execution error: {e}", exc_info=True)
            return False

    async def price_dip_back_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute a back-run strategy targeting a price dip.
        If the predicted price is lower than the current price multiplied by a threshold factor,
        the back-run is executed.
        """
        try:
            output_token = target_tx.get("output_token")
            current_price = await self.apiconfig.get_real_time_price(output_token)
            predicted_price = await self.apiconfig.predict_price(output_token)
            if current_price is None or predicted_price is None:
                logger.debug("Missing price data; skipping price dip back-run.")
                return False
            threshold = Decimal(str(self.configuration.PRICE_DIP_BACK_RUN_THRESHOLD))
            if Decimal(predicted_price) < Decimal(current_price) * threshold:
                logger.debug(f"Price dip condition met: predicted {predicted_price} < current {current_price} * {threshold}. Executing back-run.")
                return await self.back_run(target_tx)
            else:
                logger.debug("Price dip condition not met; skipping back-run.")
                return False
        except Exception as e:
            logger.error(f"Price dip back-run failed: {e}", exc_info=True)
            return False

    async def flashloan_back_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute a flashloan-enabled back-run strategy.
        If the estimated profit (possibly calculated prior and stored in target_tx)
        exceeds a defined threshold, attempt to back-run using flashloan funding.
        """
        try:
            threshold = Decimal(str(self.configuration.FLASHLOAN_BACK_RUN_PROFIT_PERCENTAGE))
            estimated_profit = Decimal(target_tx.get("estimated_profit", "0"))
            if estimated_profit > threshold:
                logger.debug("Flashloan profit threshold met; executing flashloan back-run.")
                return await self.back_run(target_tx)
            else:
                logger.debug("Flashloan profit insufficient; skipping flashloan back-run.")
                return False
        except Exception as e:
            logger.error(f"Flashloan back-run failed: {e}", exc_info=True)
            return False

    async def high_volume_back_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute a back-run strategy when high trading volume is detected.
        If the transaction volume (in USD) meets or exceeds a configured threshold,
        the back-run strategy is triggered.
        """
        try:
            threshold = Decimal(str(self.configuration.HIGH_VOLUME_BACK_RUN_DEFAULT_THRESHOLD_USD))
            tx_volume = Decimal(target_tx.get("volume_usd", "0"))
            if tx_volume >= threshold:
                logger.debug(f"High volume detected: {tx_volume} USD >= threshold {threshold} USD. Executing back-run.")
                return await self.back_run(target_tx)
            else:
                logger.debug(f"Transaction volume {tx_volume} USD below threshold {threshold} USD; skipping high volume back-run.")
                return False
        except Exception as e:
            logger.error(f"High volume back-run failed: {e}", exc_info=True)
            return False

    async def stop(self) -> None:
        """
        Gracefully stop TransactionCore and related components.
        """
        try:
            if self.safetynet:
                await self.safetynet.stop()
            if self.noncecore:
                await self.noncecore.stop()
            logger.debug("TransactionCore stopped successfully.")
        except Exception as e:
            logger.error(f"Error stopping TransactionCore: {e}", exc_info=True)

