import asyncio
from decimal import Decimal
from typing import List, Any, Optional, Dict, Set, Tuple
from web3 import AsyncWeb3
from web3.types import TxData, HexStr
from web3.exceptions import TransactionNotFound
from web3.eth import AsyncEth
from web3.datastructures import AttributeDict
from cachetools import TTLCache

from configuration import Configuration
from safetynet import SafetyNet
from noncecore import NonceCore
from apiconfig import APIConfig
from marketmonitor import MarketMonitor
from loggingconfig import setup_logging
import logging
import time
import random
from collections import deque

logger = setup_logging("MempoolMonitor", level=logging.DEBUG)

PROCESSED_TX_MAX_SIZE = 50_000
PROCESSED_TX_TTL = 900

class MempoolMonitor:
    """
    Monitors the Ethereum mempool (or pending block transactions) for transactions
    relevant to configured tokens/strategies, analyzes them, and queues potentially
    profitable opportunities for the StrategyNet.
    """
    DEFAULT_MAX_QUEUE_SIZE = 1000
    DEFAULT_POLL_INTERVAL = 1.0
    DEFAULT_BATCH_SIZE = 50
    DEFAULT_MAX_PARALLEL_FETCH = 10
    DEFAULT_RETRY_DELAY = 0.5
    DEFAULT_MAX_RETRIES = 3

    def __init__(self,
                 web3: AsyncWeb3,
                 safetynet: SafetyNet,
                 noncecore: NonceCore,
                 apiconfig: APIConfig,
                 monitored_tokens: List[str],
                 configuration: Configuration,
                 marketmonitor: MarketMonitor):
        self.web3: AsyncWeb3 = web3
        self.configuration: Configuration = configuration
        self.safetynet: SafetyNet = safetynet
        self.noncecore: NonceCore = noncecore
        self.apiconfig: APIConfig = apiconfig
        self.marketmonitor: MarketMonitor = marketmonitor

        self.monitored_tokens_addr: Set[str] = self._normalize_token_list(monitored_tokens)

        self._tx_hash_queue: asyncio.Queue[HexStr] = asyncio.Queue(maxsize=self.configuration.get_config_value("MEMPOOL_HASH_QUEUE_SIZE", self.DEFAULT_MAX_QUEUE_SIZE * 2))
        self._tx_analysis_queue: asyncio.Queue[AttributeDict[str, Any]] = asyncio.Queue(maxsize=self.configuration.get_config_value("MEMPOOL_ANALYSIS_QUEUE_SIZE", self.DEFAULT_MAX_QUEUE_SIZE))
        self.profitable_transactions: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=self.configuration.get_config_value("MEMPOOL_PROFIT_QUEUE_SIZE", self.DEFAULT_MAX_QUEUE_SIZE))

        self._processed_tx_lock = asyncio.Lock()
        self.processed_transactions: TTLCache = TTLCache(maxsize=PROCESSED_TX_MAX_SIZE, ttl=PROCESSED_TX_TTL)

        self._tx_data_cache: TTLCache = TTLCache(maxsize=1000, ttl=60)
        self._tx_cache_lock = asyncio.Lock()

        self.running: bool = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._fetch_tasks: Set[asyncio.Task] = set()
        self._analysis_tasks: Set[asyncio.Task] = set()
        self._monitor_method: str = "unknown"

        self.poll_interval: float = self.configuration.get_config_value("MEMPOOL_POLL_INTERVAL", self.DEFAULT_POLL_INTERVAL)
        self.batch_size: int = self.configuration.get_config_value("MEMPOOL_BATCH_SIZE", self.DEFAULT_BATCH_SIZE)
        self.max_parallel_fetch: int = self.configuration.get_config_value("MEMPOOL_MAX_PARALLEL_FETCH", self.DEFAULT_MAX_PARALLEL_FETCH)
        self.max_parallel_analysis: int = self.configuration.get_config_value("MEMPOOL_MAX_PARALLEL_ANALYSIS", self.DEFAULT_MAX_PARALLEL_FETCH)
        self.retry_delay: float = self.configuration.get_config_value("MEMPOOL_RETRY_DELAY", self.DEFAULT_RETRY_DELAY)
        self.max_retries: int = self.configuration.get_config_value("MEMPOOL_MAX_RETRIES", self.DEFAULT_MAX_RETRIES)


    def _normalize_token_list(self, tokens: List[str]) -> Set[str]:
        addresses: Set[str] = set()
        logger.debug("Normalizing monitored token list: %s", tokens)
        for token in tokens:
            if not token: continue
            if token.startswith("0x"):
                try:
                    addr = self.web3.to_checksum_address(token)
                    addresses.add(addr.lower())
                except ValueError:
                    logger.warning("Invalid address format in monitored list: %s", token)
            else:
                addr = self.apiconfig.get_token_address(token.upper())
                if addr:
                    addresses.add(addr.lower())
                else:
                    logger.warning("Could not find address for monitored symbol: %s", token.upper())
        logger.info("Monitoring %d unique token addresses: %s", len(addresses), list(addresses) if len(addresses) < 5 else f"{list(addresses)[:5]}...")
        return addresses

    async def initialize(self) -> None:
        while not self._tx_hash_queue.empty(): await self._tx_hash_queue.get()
        while not self._tx_analysis_queue.empty(): await self._tx_analysis_queue.get()
        while not self.profitable_transactions.empty(): await self.profitable_transactions.get()

        async with self._processed_tx_lock:
            self.processed_transactions.clear()
        async with self._tx_cache_lock:
            self._tx_data_cache.clear()

        self.running = False
        self._monitor_task = None
        self._fetch_tasks.clear()
        self._analysis_tasks.clear()
        logger.debug("MempoolMonitor initialized/reinitialized.")

    async def start_monitoring(self) -> None:
        if self.running:
            logger.warning("MempoolMonitor is already running.")
            return

        self.running = True
        logger.info("Starting Mempool Monitoring Pipeline...")

        self._monitor_task = asyncio.create_task(self._run_monitoring(), name="MempoolListener")

        for i in range(self.max_parallel_fetch):
            task = asyncio.create_task(self._fetch_transaction_worker(), name=f"TxFetchWorker-{i+1}")
            self._fetch_tasks.add(task)
            task.add_done_callback(self._fetch_tasks.discard)

        for i in range(self.max_parallel_analysis):
            task = asyncio.create_task(self._analyze_transaction_worker(), name=f"TxAnalysisWorker-{i+1}")
            self._analysis_tasks.add(task)
            task.add_done_callback(self._analysis_tasks.discard)

        logger.info(
            "MempoolMonitor started with %d fetch workers and %d analysis workers.",
            len(self._fetch_tasks), len(self._analysis_tasks)
        )
        if self._monitor_task:
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                logger.info("Mempool listener task was cancelled.")
            except Exception as e:
                 logger.error("Mempool listener task failed: %s", e, exc_info=True)


    async def _run_monitoring(self) -> None:
        use_filter = self.configuration.get_config_value("MEMPOOL_USE_FILTER", True)
        pending_filter: Optional[Any] = None

        if use_filter:
            pending_filter = await self._setup_pending_filter()

        if pending_filter:
            self._monitor_method = "filter"
            logger.info("Using filter-based mempool monitoring.")
            await self._monitor_with_filter(pending_filter)
        else:
            self._monitor_method = "polling"
            logger.info("Falling back to polling-based mempool monitoring (Interval: %.2f s).", self.poll_interval)
            await self._poll_pending_transactions()


    async def _setup_pending_filter(self) -> Optional[Any]:
        if not hasattr(self.web3.eth, "filter"):
             logger.warning("Web3 provider does not support eth_filter method.")
             return None
        try:
            logger.debug("Attempting to create 'pending' transaction filter...")
            pending_filter = await self.web3.eth.filter("pending")
            async with asyncio.timeout(10):
                await pending_filter.get_new_entries()
            logger.debug("Pending transaction filter created successfully.")
            return pending_filter
        except asyncio.TimeoutError:
             logger.warning("Pending filter test timed out. Filter might work but slow? Assuming functional.")
             return pending_filter
        except NotImplementedError:
             logger.warning("Provider does not implement 'pending' filter type.")
             return None
        except Exception as e:
            logger.warning("Failed to create or test pending filter: %s. Falling back to polling.", e, exc_info=True)
            return None

    async def _monitor_with_filter(self, pending_filter: Any) -> None:
        while self.running:
            try:
                tx_hashes: List[HexStr] = await pending_filter.get_new_entries()
                if tx_hashes:
                    logger.debug("Filter received %d new transaction hashes.", len(tx_hashes))
                    await self._enqueue_tx_hashes(tx_hashes)

                await asyncio.sleep(self.poll_interval / 2)

            except asyncio.CancelledError:
                logger.info("Filter monitoring task cancelled.")
                break
            except Exception as e:
                logger.error("Error reading from pending filter: %s. Re-trying setup.", e, exc_info=True)
                await asyncio.sleep(self.retry_delay * 5)
                new_filter = await self._setup_pending_filter()
                if new_filter:
                    pending_filter = new_filter
                    logger.info("Recreated pending transaction filter.")
                else:
                    logger.error("Failed to recreate filter. Switching to polling mode.")
                    self._monitor_method = "polling"
                    await self._poll_pending_transactions()
                    break

    async def _poll_pending_transactions(self) -> None:
        last_block_number = -1
        while self.running:
            try:
                current_block_number = await self.web3.eth.block_number
                if last_block_number == -1:
                    last_block_number = current_block_number - 1

                if current_block_number > last_block_number:
                    logger.debug("Polling new blocks from %d to %d", last_block_number + 1, current_block_number)
                    block_tasks = []
                    for block_num in range(last_block_number + 1, current_block_number + 1):
                         block_tasks.append(asyncio.create_task(self.web3.eth.get_block(block_num, full_transactions=False)))

                    block_results = await asyncio.gather(*block_tasks, return_exceptions=True)

                    new_hashes: List[HexStr] = []
                    for result in block_results:
                        if isinstance(result, Exception):
                             logger.warning("Error fetching block during polling: %s", result)
                        elif result and result.transactions:
                             new_hashes.extend([tx_hash.hex() for tx_hash in result.transactions])

                    if new_hashes:
                         logger.debug("Polling found %d transactions in blocks %d-%d.", len(new_hashes), last_block_number + 1, current_block_number)
                         await self._enqueue_tx_hashes(new_hashes)

                    last_block_number = current_block_number

                await asyncio.sleep(self.poll_interval)

            except asyncio.CancelledError:
                logger.info("Polling monitoring task cancelled.")
                break
            except Exception as e:
                logger.error("Error during block polling: %s", e, exc_info=True)
                await asyncio.sleep(self.poll_interval * 3)


    async def _enqueue_tx_hashes(self, tx_hashes: List[HexStr]) -> None:
        queued_count = 0
        skipped_count = 0
        async with self._processed_tx_lock:
            for tx_hash in tx_hashes:
                if tx_hash and tx_hash not in self.processed_transactions:
                    self.processed_transactions[tx_hash] = True
                    try:
                        self._tx_hash_queue.put_nowait(tx_hash)
                        queued_count += 1
                    except asyncio.QueueFull:
                        logger.warning("Transaction hash queue is full. Discarding hash: %s", tx_hash)
                        skipped_count += 1
                else:
                    skipped_count += 1

        if queued_count > 0:
            logger.debug("Enqueued %d new tx hashes. Skipped %d (duplicates/full queue). Queue size: %d",
                         queued_count, skipped_count, self._tx_hash_queue.qsize())

    async def _fetch_transaction_worker(self) -> None:
        logger.info("Transaction Fetch Worker started.")
        while self.running:
            try:
                tx_hash = await self._tx_hash_queue.get()
                log_extra = {"component": "MempoolMonitor", "tx_hash": tx_hash}

                async with self._tx_cache_lock:
                    if tx_hash in self._tx_data_cache:
                        tx_data = self._tx_data_cache[tx_hash]
                        logger.debug("Transaction data cache hit for %s", tx_hash, extra=log_extra)
                    else:
                        tx_data = None

                if tx_data is None:
                    tx_data = await self._get_transaction_with_retry(tx_hash, log_extra)

                if tx_data:
                     async with self._tx_cache_lock:
                           self._tx_data_cache[tx_hash] = tx_data
                     if self._is_transaction_relevant(tx_data):
                          try:
                              self._tx_analysis_queue.put_nowait(tx_data)
                              logger.debug("Transaction %s fetched and queued for analysis. Analysis queue size: %d", tx_hash, self._tx_analysis_queue.qsize(), extra=log_extra)
                          except asyncio.QueueFull:
                              logger.warning("Analysis queue full. Discarding fetched tx: %s", tx_hash, extra=log_extra)
                     else:
                          logger.debug("Transaction %s fetched but deemed irrelevant. Discarding.", tx_hash, extra=log_extra)

                self._tx_hash_queue.task_done()

            except asyncio.CancelledError:
                 logger.info("Transaction Fetch Worker cancelled.")
                 break
            except Exception as e:
                 logger.error("Error in Transaction Fetch Worker: %s", e, exc_info=True)
                 await asyncio.sleep(1)


    async def _get_transaction_with_retry(self, tx_hash: HexStr, log_extra: Dict) -> Optional[AttributeDict[str, Any]]:
        for attempt in range(self.max_retries):
            try:
                tx: Optional[TxData] = await self.web3.eth.get_transaction(tx_hash)
                if tx:
                    return tx
                else:
                     logger.warning("get_transaction returned None for known hash %s (attempt %d)", tx_hash, attempt+1, extra=log_extra)
                     raise TransactionNotFound(f"Tx {tx_hash} returned None")

            except TransactionNotFound:
                logger.debug("Transaction %s not found (attempt %d/%d). It might have been orphaned or mined quickly.",
                             tx_hash, attempt + 1, self.max_retries, extra=log_extra)
                return None
            except asyncio.TimeoutError:
                 logger.warning("Timeout fetching transaction %s (attempt %d/%d). Retrying...",
                              tx_hash, attempt + 1, self.max_retries, extra=log_extra)
            except Exception as e:
                logger.warning(
                    "Error fetching transaction %s (attempt %d/%d): %s. Retrying...",
                    tx_hash, attempt + 1, self.max_retries, e, extra=log_extra
                )

            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay * (1.5 ** attempt))

        logger.error("Failed to fetch transaction %s after %d attempts.", tx_hash, self.max_retries, extra=log_extra)
        return None

    def _is_transaction_relevant(self, tx: AttributeDict[str, Any]) -> bool:
        to_address = tx.get('to')
        if to_address and to_address.lower() in self.monitored_tokens_addr:
            return True

        return False



    async def analyze_transaction(self, tx: AttributeDict[str, Any]) -> Optional[Dict[str, Any]]:
        tx_hash = tx.get('hash', b'').hex()
        log_extra = {"component": "MempoolMonitor", "tx_hash": tx_hash}

        try:
            to_addr = tx.get('to')
            from_addr = tx.get('from')
            value_wei = tx.get('value', 0)
            gas_price_wei = tx.get('gasPrice')
            max_fee_wei = tx.get('maxFeePerGas')
            priority_fee_wei = tx.get('maxPriorityFeePerGas')
            input_data = tx.get('input', '0x')

            effective_gas_price = gas_price_wei or max_fee_wei or 0

            is_profitable_opportunity = False
            strategy_type = "unknown"
            decoded_data = None

            if to_addr:
                dex_routers = {
                    self.configuration.UNISWAP_ADDRESS.lower(),
                    self.configuration.SUSHISWAP_ADDRESS.lower()
                }
                if to_addr.lower() in dex_routers:
                    try:
                        router_abi = None
                        if to_addr.lower() == self.configuration.UNISWAP_ADDRESS.lower():
                            router_abi = self.apiconfig.get_abi('uniswap')
                        elif to_addr.lower() == self.configuration.SUSHISWAP_ADDRESS.lower():
                            router_abi = self.apiconfig.get_abi('sushiswap')

                        if router_abi:
                            contract = self.web3.eth.contract(address=to_addr, abi=router_abi)
                            decoded_data = contract.decode_function_input(input_data)

                            function_name = decoded_data[0].fn_name
                            if "swap" in function_name.lower():
                                is_profitable_opportunity = True
                                strategy_type = "sandwich_attack"

                                logger.debug(f"Decoded DEX swap: Function={function_name}, Parameters={decoded_data[1]}", extra=log_extra)
                        else:
                            logger.warning("Could not load ABI for DEX router %s. Skipping detailed analysis.", to_addr, extra=log_extra)

                    except Exception as decode_err:
                        logger.warning("Failed to decode DEX input data for %s: %s", tx_hash, decode_err, extra=log_extra)

            if to_addr and to_addr.lower() in self.monitored_tokens_addr and value_wei > self.web3.to_wei(0.1, 'ether'):
                is_profitable_opportunity = True
                strategy_type = "eth_transfer_monitor"
                logger.debug("Detected high-value ETH transfer to monitored address %s", to_addr, extra=log_extra)

            if effective_gas_price > self.web3.to_wei(100, 'gwei') and value_wei > self.web3.to_wei(0.05, 'ether'):
                is_profitable_opportunity = True
                strategy_type = "front_run"
                logger.debug("Detected high gas price tx (%.2f Gwei) - potential front-run", effective_gas_price / 1e9, extra=log_extra)

            if is_profitable_opportunity:
                result = {
                    "is_profitable": True,
                    "tx_hash": tx_hash,
                    "strategy_type": strategy_type,
                    "gasPrice": effective_gas_price,
                    "value": value_wei,
                    "to": to_addr,
                    "from": from_addr,
                    "input": input_data,
                    "nonce": tx.get('nonce'),
                    "gas": tx.get('gas'),
                    "token_symbol": self.apiconfig.get_token_symbol(to_addr) if to_addr else "ETH",
                    "token_address": to_addr,
                    "estimated_profit_eth": Decimal("0.0"),
                    "confidence_score": 0.5,
                    "decoded_data": decoded_data
                }
                return result
            else:
                return None

        except Exception as e:
            logger.error("Unexpected error during transaction analysis for %s: %s", tx_hash, e, exc_info=True, extra=log_extra)
            return None


    async def _analyze_transaction_worker(self) -> None:
        logger.info("Transaction Analysis Worker started.")
        while self.running:
            try:
                tx_data = await self._tx_analysis_queue.get()
                tx_hash = tx_data.get('hash', b'').hex()
                log_extra = {"component": "MempoolMonitor", "tx_hash": tx_hash}

                try:
                    analysis_result = await self.analyze_transaction(tx_data)

                    if analysis_result and analysis_result.get("is_profitable"):
                        try:
                             self.profitable_transactions.put_nowait(analysis_result)
                             logger.info(
                                 "Transaction %s analyzed as profitable. Queued for strategy execution. Strategy queue size: %d",
                                 tx_hash, self.profitable_transactions.qsize(), extra=log_extra
                             )
                        except asyncio.QueueFull:
                            logger.warning("Profitable transaction queue full. Discarding analysis result for: %s", tx_hash, extra=log_extra)
                    else:
                        logger.debug("Transaction %s analyzed as not profitable.", tx_hash, extra=log_extra)

                except Exception as analysis_error:
                     logger.error("Error during analysis of tx %s: %s", tx_hash, analysis_error, exc_info=True, extra=log_extra)

                self._tx_analysis_queue.task_done()

            except asyncio.CancelledError:
                 logger.info("Transaction Analysis Worker cancelled.")
                 break
            except Exception as e:
                 logger.error("Error in Transaction Analysis Worker: %s", e, exc_info=True)
                 await asyncio.sleep(1)


    async def stop(self) -> None:
        if not self.running:
            return
        logger.info("Stopping MempoolMonitor...")
        self.running = False

        if self._monitor_task:
            self._monitor_task.cancel()

        all_tasks = list(self._fetch_tasks) + list(self._analysis_tasks)
        if self._monitor_task:
             all_tasks.append(self._monitor_task)

        logger.debug("Cancelling %d MempoolMonitor tasks...", len(all_tasks))
        for task in all_tasks:
             if not task.done():
                  task.cancel()

        await asyncio.gather(*all_tasks, return_exceptions=True)
        logger.debug("MempoolMonitor tasks cancellation complete.")

        self._fetch_tasks.clear()
        self._analysis_tasks.clear()
        self._monitor_task = None

        async with self._processed_tx_lock:
            self.processed_transactions.clear()
        async with self._tx_cache_lock:
            self._tx_data_cache.clear()

        logger.info("MempoolMonitor stopped.")
