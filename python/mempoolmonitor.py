import asyncio
from typing import List, Dict, Any, Optional
from cachetools import TTLCache
from web3 import AsyncWeb3
from web3.exceptions import TransactionNotFound
from configuration import Configuration
from safetynet import SafetyNet
from noncecore import NonceCore
from apiconfig import APIConfig
from marketmonitor import MarketMonitor
from loggingconfig import setup_logging

logger = setup_logging("Mempool_Monitor", level="DEBUG")  


class MempoolMonitor:
    def __init__(
        self,
        web3: AsyncWeb3,
        safetynet: SafetyNet,
        noncecore: NonceCore,
        apiconfig: APIConfig,
        monitored_tokens: List[str],
        configuration: Configuration,
        marketmonitor: MarketMonitor,
    ):
        self.web3 = web3
        self.safetynet = safetynet
        self.noncecore = noncecore
        self.apiconfig = apiconfig
        self.marketmonitor = marketmonitor
        self.monitored_tokens = {
            apiconfig.get_token_address(t).lower() if not t.startswith("0x") else t.lower()
            for t in monitored_tokens
        }
        self.configuration = configuration
        self.pending = asyncio.Queue()
        self.profitable = asyncio.Queue()
        self.task_queue = asyncio.PriorityQueue()
        self.processed = set()
        self.cache = {}
        self.running = False
        self.semaphore = asyncio.Semaphore(self.configuration.MEMPOOL_MAX_PARALLEL_TASKS)

    async def initialize(self) -> None:
        self.pending = asyncio.Queue()
        self.profitable = asyncio.Queue()
        self.task_queue = asyncio.PriorityQueue()
        self.processed.clear()
        self.cache.clear()
        self.running = False

    async def start_monitoring(self) -> None:
        if self.running:
            return
        self.running = True
        await asyncio.gather(self._run_monitoring(), self._process_task_queue())

    async def _run_monitoring(self) -> None:
        filter_obj = None
        try:
            filter_obj = await self.web3.eth.filter("pending")
            await filter_obj.get_new_entries()
        except Exception:
            filter_obj = None
        if filter_obj:
            while self.running:
                try:
                    hashes = await filter_obj.get_new_entries()
                    if hashes:
                        await self._handle_hashes([h.hex() for h in hashes])
                    await asyncio.sleep(1)
                except Exception:
                    await asyncio.sleep(1)
        else:
            last_block = await self.web3.eth.block_number
            while self.running:
                current = await self.web3.eth.block_number
                for n in range(last_block + 1, current + 1):
                    block = await self.web3.eth.get_block(n, full_transactions=True)
                    for tx in block.transactions:
                        h = tx.hash.hex() if hasattr(tx, "hash") else tx["hash"].hex()
                        await self._handle_hashes([h])
                last_block = current
                await asyncio.sleep(1)

    async def _handle_hashes(self, hashes: List[str]) -> None:
        for h in hashes:
            if h not in self.processed:
                self.processed.add(h)
                priority = await self._calculate_priority(h)
                await self.task_queue.put((priority, h))

    async def _calculate_priority(self, tx_hash: str) -> int:
        tx = await self._get_transaction(tx_hash)
        if not tx:
            return float("inf")
        return -tx.get("gasPrice", 0)

    async def _get_transaction(self, tx_hash: str) -> Optional[Dict[str, Any]]:
        if tx_hash in self.cache:
            return self.cache[tx_hash]
        backoff = self.configuration.MEMPOOL_RETRY_DELAY
        for _ in range(self.configuration.MEMPOOL_MAX_RETRIES):
            try:
                tx = await self.web3.eth.get_transaction(tx_hash)
                self.cache[tx_hash] = tx
                return tx
            except TransactionNotFound:
                await asyncio.sleep(backoff)
                backoff *= 1.5
            except Exception:
                return None
        return None

    async def _process_task_queue(self) -> None:
        while self.running:
            priority, tx_hash = await self.task_queue.get()
            await self.semaphore.acquire()
            asyncio.create_task(self._process_transaction(tx_hash))

    async def _process_transaction(self, tx_hash: str) -> None:
        try:
            tx = await self._get_transaction(tx_hash)
            if not tx:
                return
            analysis = await self._analyze_transaction(tx, tx_hash)
            if analysis.get("is_profitable"):
                await self.profitable.put(analysis)
        finally:
            self.semaphore.release()

    async def _analyze_transaction(self, tx: Dict[str, Any], tx_hash: str) -> Dict[str, Any]:
        to_addr = tx.get("to", "")
        value = tx.get("value", 0)
        gas_price = tx.get("gasPrice", 0)
        if value <= 0 or gas_price <= 0:
            return {"is_profitable": False}
        addr = to_addr.lower()
        if addr not in self.monitored_tokens:
            return {"is_profitable": False}
        gas_used = await self.safetynet.estimate_gas(tx)
        tx_data = {
            "output_token": addr,
            "amountIn": float(self.web3.from_wei(value, "ether")),
            "amountOut": float(self.web3.from_wei(value, "ether")),
            "gas_price": float(self.web3.from_wei(gas_price, "gwei")),
            "gas_used": float(gas_used),
        }
        safe, details = await self.safetynet.check_transaction_safety(tx_data, check_type="profit")
        if safe and details.get("profit_ok"):
            return {
                "is_profitable": True,
                "tx_hash": tx_hash,
                "tx": tx,
                "analysis": details,
                "strategy_type": "front_run"
            }
        return {"is_profitable": False}

    async def stop(self) -> None:
        self.running = False
        await asyncio.sleep(0)
