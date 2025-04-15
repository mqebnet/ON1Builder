# python/mempoolmonitor.py

import asyncio

from typing import List, Any, Optional
from web3 import AsyncWeb3
from web3.exceptions import TransactionNotFound
from configuration import Configuration
from safetynet import SafetyNet
from noncecore import NonceCore
from apiconfig import APIConfig
from marketmonitor import MarketMonitor
from loggingconfig import setup_logging
import logging

logger = setup_logging("MempoolMonitor", level=logging.DEBUG)

class MempoolMonitor:
    """
    Monitors the Ethereum mempool for pending transactions, processes them, and queues
    profitable transactions for further strategic evaluation.
    """
    def __init__(self,
                 web3: AsyncWeb3,
                 safetynet: SafetyNet,
                 noncecore: NonceCore,
                 apiconfig: APIConfig,
                 monitored_tokens: List[str],
                 configuration: Configuration,
                 erc20_abi_path: str,  # now a full path string
                 marketmonitor: MarketMonitor):
        self.web3 = web3
        self.configuration = configuration
        self.safetynet = safetynet
        self.noncecore = noncecore
        self.apiconfig = apiconfig
        self.marketmonitor = marketmonitor
        self.monitored_tokens = set(monitored_tokens)
        self.pending_transactions = asyncio.Queue()
        self.profitable_transactions = asyncio.Queue()
        self.task_queue = asyncio.PriorityQueue()
        self.processed_transactions = set()
        self.cache = {}
        self.backoff_factor = 1.5
        self.running = False

    async def initialize(self) -> None:
        """Reinitialize mempool monitoring queues and caches."""
        self.running = False
        self.pending_transactions = asyncio.Queue()
        self.profitable_transactions = asyncio.Queue()
        self.task_queue = asyncio.PriorityQueue()
        self.processed_transactions.clear()
        self.cache.clear()
        logger.debug("MempoolMonitor initialized.")

    async def start_monitoring(self) -> None:
        """Start the mempool monitoring process."""
        if self.running:
            logger.debug("MempoolMonitor is already running.")
            return
        self.running = True
        monitor_task = asyncio.create_task(self._run_monitoring())
        processor_task = asyncio.create_task(self._process_task_queue())
        logger.info("Mempool monitoring started.")
        await asyncio.gather(monitor_task, processor_task)

    async def _run_monitoring(self) -> None:
        """Attempt filter-based monitoring; if it fails, fallback to polling."""
        while self.running:
            pending_filter = await self._setup_pending_filter()
            if pending_filter:
                try:
                    while self.running:
                        tx_hashes = await pending_filter.get_new_entries()
                        if tx_hashes:
                            await self._handle_new_transactions(tx_hashes)
                        await asyncio.sleep(1)
                except Exception as e:
                    logger.error(f"Error using filter-based monitoring: {e}")
                    await asyncio.sleep(2)
            else:
                await self._poll_pending_transactions()

    async def _setup_pending_filter(self) -> Optional[Any]:
        """Set up a pending transaction filter if possible."""
        try:
            pending_filter = await self.web3.eth.filter("pending")
            # Test the filter with a short timeout.
            await asyncio.wait_for(pending_filter.get_new_entries(), timeout=5)
            logger.debug("Using filter-based pending transactions.")
            return pending_filter
        except Exception as e:
            logger.warning(f"Pending filter unavailable: {e}. Falling back to polling.")
            return None

    async def _poll_pending_transactions(self) -> None:
        """Poll new blocks and process transactions."""
        last_block = await self.web3.eth.block_number
        while self.running:
            try:
                current_block = await self.web3.eth.block_number
                if current_block <= last_block:
                    await asyncio.sleep(1)
                    continue
                for block_num in range(last_block + 1, current_block + 1):
                    try:
                        block = await self.web3.eth.get_block(block_num, full_transactions=True)
                        if block and block.transactions:
                            tx_hashes = []
                            for tx in block.transactions:
                                # Support both dict and object types.
                                if hasattr(tx, "hash"):
                                    tx_hashes.append(tx.hash.hex())
                                elif isinstance(tx, dict) and "hash" in tx:
                                    tx_hashes.append(tx["hash"].hex())
                            if tx_hashes:
                                await self._handle_new_transactions(tx_hashes)
                    except Exception as e:
                        logger.error(f"Error processing block {block_num}: {e}")
                        continue
                last_block = current_block
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Polling error: {e}")
                await asyncio.sleep(2)

    async def _handle_new_transactions(self, tx_hashes: List[str]) -> None:
        """Process new transaction hashes in a batch."""
        for tx_hash in tx_hashes:
            await self._queue_transaction(tx_hash)

    async def _queue_transaction(self, tx_hash: str) -> None:
        """Queue a transaction if not already processed."""
        if not tx_hash or tx_hash in self.processed_transactions:
            return
        self.processed_transactions.add(tx_hash)
        priority = await self._calculate_priority(tx_hash)
        await self.task_queue.put((priority, tx_hash))

    async def _calculate_priority(self, tx_hash: str) -> int:
        """
        Calculate a priority for the transaction based on its gas price.
        Higher gas price means higher priority (i.e. more negative).
        """
        try:
            tx = await self._get_transaction_with_retry(tx_hash)
            if not tx:
                return float("inf")
            gas_price = tx.get("gasPrice", 0)
            return -gas_price
        except Exception as e:
            logger.error(f"Priority calculation error for {tx_hash}: {e}")
            return float("inf")

    async def _get_transaction_with_retry(self, tx_hash: str) -> dict:
        backoff = self.configuration.MEMPOOL_RETRY_DELAY
        retries = self.configuration.MEMPOOL_MAX_RETRIES
        for _ in range(retries):
            try:
                if tx_hash in self.cache:
                    return self.cache[tx_hash]
                tx = await self.web3.eth.get_transaction(tx_hash)
                self.cache[tx_hash] = tx
                return tx
            except TransactionNotFound:
                await asyncio.sleep(backoff)
                backoff *= self.backoff_factor
            except Exception as e:
                logger.error(f"Error fetching transaction {tx_hash}: {e}")
                return {}
        return {}

    async def _process_task_queue(self) -> None:
        """Continuously process transactions from the priority queue."""
        semaphore = asyncio.Semaphore(self.configuration.MEMPOOL_MAX_PARALLEL_TASKS)
        while self.running:
            try:
                priority, tx_hash = await self.task_queue.get()
                async with semaphore:
                    asyncio.create_task(self.process_transaction(tx_hash))
            except Exception as e:
                logger.error(f"Error in task queue processing: {e}")
                await asyncio.sleep(1)

    async def process_transaction(self, tx_hash: str) -> None:
        """Analyze a transaction and, if deemed profitable, add it to the profitable queue."""
        try:
            tx = await self._get_transaction_with_retry(tx_hash)
            if not tx:
                return
            analysis = await self.analyze_transaction(tx)
            if analysis.get("is_profitable"):
                await self.profitable_transactions.put(analysis)
                logger.debug(f"Queued profitable transaction: {analysis.get('tx_hash')}")
        except Exception as e:
            logger.error(f"Error processing transaction {tx_hash}: {e}")

    async def analyze_transaction(self, tx: dict) -> dict:
        """
        Analyze a transaction for profitability.
        This is a stub that you should extend with your actual analysis logic.
        """
        result = {"is_profitable": False}
        try:
            gas_price = tx.get("gasPrice", 0)
            value = tx.get("value", 0)
            if gas_price > 0 and value > 0:
                result = {
                    "is_profitable": True,
                    "tx_hash": tx.get("hash").hex() if "hash" in tx else "unknown",
                    "gasPrice": gas_price,
                    "value": value,
                    "strategy_type": "front_run"  # Placeholder strategy assignment.
                }
        except Exception as e:
            logger.error(f"Analysis error: {e}")
        return result

    async def stop(self) -> None:
        """Stop the mempool monitor."""
        self.running = False
        logger.info("MempoolMonitor stopped.")
