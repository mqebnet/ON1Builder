from typing import Any, Dict, List, Optional
from decimal import Decimal

import asyncio
from web3 import AsyncWeb3
from web3.exceptions import ContractLogicError, TransactionNotFound, Web3ValueError
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

    async def stop(self) -> None:
        """
        Gracefully stop TransactionCore.
        """
        if self.safetynet:
            await self.safetynet.stop()
        if self.noncecore:
            await self.noncecore.stop()
