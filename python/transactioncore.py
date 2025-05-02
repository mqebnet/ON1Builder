from typing import Any, Dict, List, Optional
from decimal import Decimal

import asyncio
from web3 import AsyncWeb3
from web3.exceptions import ContractLogicError, TransactionNotFound
from eth_account import Account

from abiregistry import ABIRegistry
from apiconfig import APIConfig
from configuration import Configuration
from marketmonitor import MarketMonitor
from mempoolmonitor import MempoolMonitor
from noncecore import NonceCore
from safetynet import SafetyNet
from loggingconfig import setup_logging

logger = setup_logging("TransactionCore", level="DEBUG")  


class TransactionCore:
    """
    Transaction engine for building, signing, simulating, and executing MEV transactions.
    """

    DEFAULT_GAS_LIMIT: int = 100_000
    ETH_TRANSFER_GAS: int = 21_000

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
    ):
        self.web3 = web3
        self.account = account
        self.configuration = configuration
        self.apiconfig = apiconfig
        self.marketmonitor = marketmonitor
        self.mempoolmonitor = mempoolmonitor
        self.noncecore = noncecore
        self.safetynet = safetynet
        self.gas_price_multiplier = gas_price_multiplier
        self.abiregistry = ABIRegistry()
        self.aave_flashloan = None
        self.aave_pool = None
        self.uniswap_router = None
        self.sushiswap_router = None
        self.erc20_abi: List[Dict[str, Any]] = []
        self.flashloan_abi: List[Dict[str, Any]] = []

    async def initialize(self) -> None:
        """
        Initialize the transaction engine by loading ABIs and validating contracts.
        """
        await self.abiregistry.initialize(self.configuration.BASE_PATH)
        self.erc20_abi = self.abiregistry.get_abi("erc20") or []
        self.flashloan_abi = self.abiregistry.get_abi("aave_flashloan") or []
        aave_pool_abi = self.abiregistry.get_abi("aave") or []
        uniswap_abi = self.abiregistry.get_abi("uniswap") or []
        sushiswap_abi = self.abiregistry.get_abi("sushiswap") or []

        self.aave_flashloan = self.web3.eth.contract(
            address=self.web3.to_checksum_address(self.configuration.AAVE_FLASHLOAN_ADDRESS),
            abi=self.flashloan_abi
        )
        await self._validate_contract(self.aave_flashloan)

        self.aave_pool = self.web3.eth.contract(
            address=self.web3.to_checksum_address(self.configuration.AAVE_POOL_ADDRESS),
            abi=aave_pool_abi
        )
        await self._validate_contract(self.aave_pool)

        if uniswap_abi and self.configuration.UNISWAP_ADDRESS:
            self.uniswap_router = self.web3.eth.contract(
                address=self.web3.to_checksum_address(self.configuration.UNISWAP_ADDRESS),
                abi=uniswap_abi
            )
            await self._validate_contract(self.uniswap_router)

        if sushiswap_abi and self.configuration.SUSHISWAP_ADDRESS:
            self.sushiswap_router = self.web3.eth.contract(
                address=self.web3.to_checksum_address(self.configuration.SUSHISWAP_ADDRESS),
                abi=sushiswap_abi
            )
            await self._validate_contract(self.sushiswap_router)

        logger.info("TransactionCore initialized successfully.")

    async def _validate_contract(self, contract: Any) -> None:
        """
        Validate that a contract is deployed by checking on-chain code.
        """
        code = await self.web3.eth.get_code(contract.address)
        if not code or code == b"":
            raise ValueError(f"No code at address {contract.address}")

    async def build_transaction(
        self,
        function_call: Any,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build a transaction with dynamic gas parameters and correct nonce.
        """
        additional_params = additional_params or {}
        chain_id = await self.web3.eth.chain_id
        latest = await self.web3.eth.get_block("latest")
        supports_1559 = "baseFeePerGas" in latest
        nonce = await self.noncecore.get_nonce()

        params: Dict[str, Any] = {
            "chainId": chain_id,
            "nonce": nonce,
            "from": self.account.address,
            "gas": self.DEFAULT_GAS_LIMIT
        }

        if supports_1559:
            base = latest["baseFeePerGas"]
            priority = await self.web3.eth.max_priority_fee
            params.update({
                "maxFeePerGas": int(base * 2),
                "maxPriorityFeePerGas": int(priority)
            })
        else:
            dynamic = await self.safetynet.get_dynamic_gas_price()
            params["gasPrice"] = int(self.web3.to_wei(dynamic * self.gas_price_multiplier, "gwei"))

        tx = function_call.buildTransaction(params)
        tx.update(additional_params)
        estimated = await self._estimate_gas(tx)
        tx["gas"] = int(estimated * 1.1)
        return tx

    async def _estimate_gas(self, tx: Dict[str, Any]) -> int:
        """
        Estimate gas with fallback.
        """
        try:
            return await self.web3.eth.estimate_gas(tx)
        except (ContractLogicError, TransactionNotFound):
            return self.DEFAULT_GAS_LIMIT

    def _clean_tx(self, tx: Dict[str, Any]) -> Dict[str, Any]:
        """
        Keep only valid transaction fields.
        """
        valid = {
            "nonce", "gas", "gasPrice", "maxFeePerGas",
            "maxPriorityFeePerGas", "to", "value", "data",
            "chainId", "from", "type"
        }
        return {k: v for k, v in tx.items() if k in valid}

    async def sign_transaction(self, tx: Dict[str, Any]) -> bytes:
        """
        Sign a transaction dict.
        """
        clean = self._clean_tx(tx)
        if "chainId" not in clean:
            clean["chainId"] = await self.web3.eth.chain_id
        signed = self.web3.eth.account.sign_transaction(clean, private_key=self.account.key)
        return signed.rawTransaction

    async def send_signed(self, raw: bytes) -> str:
        """
        Send a signed transaction and return its hash.
        """
        tx_hash = await self.web3.eth.send_raw_transaction(raw)
        await self.noncecore.refresh_nonce()
        return tx_hash.hex()

    async def simulate_transaction(self, tx: Dict[str, Any]) -> bool:
        """
        Simulate a transaction by eth_call.
        """
        try:
            await self.web3.eth.call(tx, block_identifier="pending")
            return True
        except ContractLogicError:
            return False

    async def execute_transaction(self, tx: Dict[str, Any]) -> Optional[str]:
        """
        Execute with retries, honoring gas price cap.
        """
        max_retries = self.configuration.MEMPOOL_MAX_RETRIES
        delay = self.configuration.MEMPOOL_RETRY_DELAY

        if "nonce" not in tx:
            tx["nonce"] = await self.noncecore.get_nonce()
        if "gas" not in tx:
            tx["gas"] = self.DEFAULT_GAS_LIMIT

        for attempt in range(1, max_retries + 1):
            try:
                raw = await self.sign_transaction(tx)
                return await self.send_signed(raw)
            except Exception:
                await asyncio.sleep(delay * attempt)

            gp = tx.get("gasPrice") or tx.get("maxFeePerGas", 0)
            if gp and gp > self.web3.to_wei(self.configuration.MAX_GAS_PRICE_GWEI, "gwei"):
                return None
        return None

    async def withdraw_eth(self) -> bool:
        """
        Withdraw ETH from the flashloan contract to owner.
        """
        fn = self.aave_flashloan.functions.withdrawETH()
        tx = await self.build_transaction(fn)
        sent = await self.execute_transaction(tx)
        return bool(sent)

    async def transfer_profit_to_account(self, token_address: str, amount: Decimal, target: str) -> bool:
        """
        Transfer ERC20 profit to a specified account.
        """
        token = self.web3.eth.contract(
            address=self.web3.to_checksum_address(token_address),
            abi=self.erc20_abi
        )
        amt = int(amount * Decimal(10) ** await token.functions.decimals().call())
        fn = token.functions.transfer(self.web3.to_checksum_address(target), amt)
        tx = await self.build_transaction(fn)
        sent = await self.execute_transaction(tx)
        return bool(sent)

    async def handle_eth_transaction(self, target_tx: Dict[str, Any]) -> bool:
        """
        Process an ETH transfer transaction via front-run style execution.
        """
        value = int(target_tx.get("value", 0))
        if value <= 0:
            return False

        nonce = await self.noncecore.get_nonce()
        chain_id = await self.web3.eth.chain_id
        tx = {
            "to": target_tx.get("to", ""),
            "value": value,
            "nonce": nonce,
            "chainId": chain_id,
            "from": self.account.address,
            "gas": self.ETH_TRANSFER_GAS
        }

        gp = target_tx.get("gasPrice")
        if gp:
            tx["gasPrice"] = int(int(gp) * self.configuration.ETH_TX_GAS_PRICE_MULTIPLIER)
        else:
            dynamic = await self.safetynet.get_dynamic_gas_price()
            tx["gasPrice"] = int(self.web3.to_wei(dynamic * self.gas_price_multiplier, "gwei"))

        sent = await self.execute_transaction(tx)
        return bool(sent)

    async def front_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute a basic front-run strategy on the target transaction.
        """
        sent = await self.execute_transaction(target_tx)
        return bool(sent)

    async def back_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute a basic back-run strategy on the target transaction.
        """
        sent = await self.execute_transaction(target_tx)
        return bool(sent)
    
    async def execute_sandwich_attack(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute a sandwich attack by sending a front-run tx and a back-run tx around the target.
        """
        front_tx = target_tx.copy()

        front_tx["gasPrice"] = int(front_tx.get("gasPrice", 0) * 1.15) if front_tx.get("gasPrice") else int(self.web3.to_wei((await self.safetynet.get_dynamic_gas_price()) * self.gas_price_multiplier * 1.15, "gwei"))
        back_tx = target_tx.copy()

        back_tx["gasPrice"] = int(back_tx.get("gasPrice", 0) * 0.9) if back_tx.get("gasPrice") else int(self.web3.to_wei((await self.safetynet.get_dynamic_gas_price()) * self.gas_price_multiplier * 0.9, "gwei"))
        res_front = await self.execute_transaction(front_tx)

        await asyncio.sleep(1)
        res_back = await self.execute_transaction(back_tx)
        return bool(res_front) and bool(res_back)

    async def aggressive_front_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Aggressively increase gas price by 30%.
        """
        if "gasPrice" in target_tx:
            target_tx["gasPrice"] = int(target_tx["gasPrice"] * 1.3)
        else:
            dynamic = await self.safetynet.get_dynamic_gas_price()
            target_tx["gasPrice"] = int(self.web3.to_wei(dynamic * self.gas_price_multiplier * 1.3, "gwei"))
        return await self.execute_transaction(target_tx)

    async def predictive_front_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute front-run only if simulation indicates success.
        """
        simulation_success = await self.simulate_transaction(target_tx)
        if simulation_success:
            return await self.front_run(target_tx)
        return False

    async def volatility_front_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Increase gas price by a volatility multiplier (e.g., 1.5).
        """
        volatile_multiplier = 1.5
        if "gasPrice" in target_tx:
            target_tx["gasPrice"] = int(target_tx["gasPrice"] * volatile_multiplier)
        else:
            dynamic = await self.safetynet.get_dynamic_gas_price()
            target_tx["gasPrice"] = int(self.web3.to_wei(dynamic * self.gas_price_multiplier * volatile_multiplier, "gwei"))
        return await self.execute_transaction(target_tx)

    async def price_dip_back_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute back-run with reduced gas price due to a price dip (reduce by 20%).
        """
        if "gasPrice" in target_tx:
            target_tx["gasPrice"] = int(target_tx["gasPrice"] * 0.8)
        else:
            dynamic = await self.safetynet.get_dynamic_gas_price()
            target_tx["gasPrice"] = int(self.web3.to_wei(dynamic * self.gas_price_multiplier * 0.8, "gwei"))
        return await self.execute_transaction(target_tx)

    async def flashloan_back_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Use flashloan logic before executing back-run.
        """
        flashloan_success = await self.withdraw_eth()
        if flashloan_success:
            return await self.back_run(target_tx)
        return False

    async def high_volume_back_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        In high volume, adjust gas price down slightly (reduce by 15%) to capture back-run opportunities.
        """
        if "gasPrice" in target_tx:
            target_tx["gasPrice"] = int(target_tx["gasPrice"] * 0.85)
        else:
            dynamic = await self.safetynet.get_dynamic_gas_price()
            target_tx["gasPrice"] = int(self.web3.to_wei(dynamic * self.gas_price_multiplier * 0.85, "gwei"))
        return await self.execute_transaction(target_tx)
    async def bundle_transactions(self, transactions: List[Dict[str, Any]]) -> List[str]:
        """
        Bundle multiple transactions into a single transaction.
        """
        bundle_tx = []
        for tx in transactions:
            clean_tx = self._clean_tx(tx)
            bundle_tx.append(clean_tx)
        main_tx = bundle_tx[0]
        main_tx["data"] = self.web3.to_hex(self.web3.to_bytes(text="Bundle Transaction"))
        signed_bundle = await self.sign_transaction(main_tx)
        tx_hashes = []
        for raw in signed_bundle:
            tx_hash = await self.send_signed(raw)
            tx_hashes.append(tx_hash)
        return tx_hashes
        
    async def execute_bundle(self, transactions: List[Dict[str, Any]]) -> List[str]:
        """
        Execute a bundle of transactions.
        """
        tx_hashes = await self.bundle_transactions(transactions)
        for tx_hash in tx_hashes:
            try:
                receipt = await self.web3.eth.wait_for_transaction_receipt(tx_hash)
                if receipt.status == 1:
                    logger.info(f"Transaction {tx_hash} executed successfully.")
                else:
                    logger.error(f"Transaction {tx_hash} failed.")
            except Exception as e:
                logger.error(f"Error waiting for transaction {tx_hash}: {e}")
        return tx_hashes
    
    async def flashloan_front_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Use flashloan logic before executing front-run.
        """
        flashloan_success = await self.withdraw_eth()
        if flashloan_success:
            return await self.front_run(target_tx)
        return False
    async def flashloan_sandwich_attack(self, target_tx: Dict[str, Any]) -> bool:
        """
        Use flashloan logic before executing sandwich attack.
        """
        flashloan_success = await self.withdraw_eth()
        if flashloan_success:
            return await self.execute_sandwich_attack(target_tx)
        return False

    async def stop(self) -> None:
        """
        Gracefully stop TransactionCore.
        """
        if self.safetynet:
            await self.safetynet.stop()
        if self.noncecore:
            await self.noncecore.stop()
