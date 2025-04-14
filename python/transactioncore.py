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

logger = setup_logging("TransactionCore", level=logging.INFO)

class TransactionCore:
    """
    Main transaction engine responsible for building, signing, simulating,
    and executing transactions for MEV strategies.
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

    async def sign_transaction(self, transaction: Dict[str, Any]) -> bytes:
        """
        Sign the given transaction using the account's private key.
        """
        try:
            signed_tx = self.web3.eth.account.sign_transaction(
                transaction,
                private_key=self.account.key,
            )
            logger.debug(f"Transaction signed (Nonce: {transaction['nonce']}).")
            return signed_tx.rawTransaction
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
            if gas_price_gwei > self.configuration.MAX_GAS_PRICE_GWEI:
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

    async def front_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute a front-run strategy on a target transaction.
        """
        try:
            logger.debug(f"Executing front-run strategy for tx: {target_tx.get('tx_hash', 'unknown')}")
            success = await self.execute_transaction(target_tx)
            return success is not None
        except Exception as e:
            logger.error(f"Front-run execution error: {e}", exc_info=True)
            return False

    async def back_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute a back-run strategy on a target transaction.
        """
        try:
            logger.debug(f"Executing back-run strategy for tx: {target_tx.get('tx_hash', 'unknown')}")
            success = await self.execute_transaction(target_tx)
            return success is not None
        except Exception as e:
            logger.error(f"Back-run execution error: {e}", exc_info=True)
            return False

    async def execute_sandwich_attack(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute a sandwich attack strategy by combining flashloan, front-run, and back-run.
        """
        try:
            logger.debug(f"Executing sandwich attack for tx: {target_tx.get('tx_hash', 'unknown')}")
            flashloan_tx = await self.prepare_flashloan_transaction(target_tx.get("asset"), target_tx.get("flashloan_amount"))
            front_tx_success = await self.front_run(target_tx)
            back_tx_success = await self.back_run(target_tx)
            if flashloan_tx and front_tx_success and back_tx_success:
                logger.info("Sandwich attack executed successfully.")
                return True
            else:
                logger.warning("Sandwich attack failed in one or more components.")
                return False
        except Exception as e:
            logger.error(f"Sandwich attack error: {e}", exc_info=True)
            return False

    async def prepare_flashloan_transaction(self, flashloan_asset: str, flashloan_amount: int) -> Optional[Dict[str, Any]]:
        """
        Prepare a flashloan transaction.
        """
        if flashloan_amount <= 0:
            logger.debug("Flashloan amount is not positive; skipping flashloan tx.")
            return None
        try:
            function_call = self.aave_flashloan.functions.fn_RequestFlashLoan(flashloan_asset, flashloan_amount)
            tx = await self.build_transaction(function_call)
            return tx
        except Exception as e:
            logger.error(f"Error preparing flashloan tx: {e}", exc_info=True)
            return None

    async def stop(self) -> None:
        """
        Gracefully stop TransactionCore and its related components.
        """
        try:
            if self.safetynet:
                await self.safetynet.stop()
            if self.noncecore:
                await self.noncecore.stop()
            logger.debug("TransactionCore stopped.")
        except Exception as e:
            logger.error(f"Error stopping TransactionCore: {e}", exc_info=True)
