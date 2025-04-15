import asyncio
import time
from typing import Any

from cachetools import TTLCache
from web3 import AsyncWeb3
from web3.exceptions import Web3ValueError, TransactionNotFound
from configuration import Configuration
from loggingconfig import setup_logging
import logging

logger = setup_logging("NonceCore", level=logging.DEBUG)

class NonceCore:
    """
    Advanced nonce management system for Ethereum transactions with caching,
    auto-recovery, and comprehensive error handling.

    This module is critical to ensure that nonce assignment remains consistent
    and avoids transaction collisions.
    """

    def __init__(self, web3: AsyncWeb3, address: str, configuration: Configuration) -> None:
        """
        Initialize the NonceCore instance.

        Args:
            web3 (AsyncWeb3): An asynchronous Web3 instance.
            address (str): The Ethereum address for nonce management.
            configuration (Configuration): Configuration data.
        """
        self.web3: AsyncWeb3 = web3
        self.configuration: Configuration = configuration
        self.address: str = address
        self.lock: asyncio.Lock = asyncio.Lock()
        self.nonce_cache: TTLCache = TTLCache(maxsize=1, ttl=self.configuration.NONCE_CACHE_TTL)
        self.last_sync: float = time.monotonic()
        self.pending_transactions: set[int] = set()
        self._initialized: bool = False

    async def initialize(self) -> None:
        """
        Initialize nonce management with fallback mechanisms and error recovery.
        """
        if self._initialized:
            logger.debug("NonceCore already initialized.")
            return
        try:
            await self._init_nonce()
            self._initialized = True
            logger.debug("NonceCore initialized successfully.")
        except Web3ValueError as e:
            logger.error(f"Web3ValueError during nonce initialization: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize NonceCore: {e}", exc_info=True)
            raise

    async def _init_nonce(self) -> None:
        """
        Fetch the current nonce from the chain and any pending nonce,
        then store the higher value in the cache.
        """
        try:
            current_nonce = await self._fetch_current_nonce_with_retries()
            pending_nonce = await self._get_pending_nonce()
            self.nonce_cache[self.address] = max(current_nonce, pending_nonce)
            self.last_sync = time.monotonic()
            logger.debug(f"Initial nonce set to {self.nonce_cache[self.address]}")
        except Exception as e:
            logger.error(f"Error during nonce initialization: {e}", exc_info=True)
            raise

    async def get_nonce(self, force_refresh: bool = False) -> int:
        """
        Retrieve the next available nonce, optionally forcing a refresh from chain.
        """
        if not self._initialized:
            await self.initialize()
        if force_refresh or self._should_refresh_cache():
            await self.refresh_nonce()
        return self.nonce_cache.get(self.address, 0)

    async def refresh_nonce(self) -> None:
        """
        Refresh the nonce from the blockchain while using a lock to ensure consistency.
        """
        async with self.lock:
            try:
                current_nonce = await self.web3.eth.get_transaction_count(self.address)
                self.nonce_cache[self.address] = current_nonce
                self.last_sync = time.monotonic()
                logger.debug(f"Nonce refreshed to {current_nonce}.")
            except Exception as e:
                logger.error(f"Error refreshing nonce: {e}", exc_info=True)

    async def _fetch_current_nonce_with_retries(self) -> int:
        """
        Try to fetch the current nonce from the blockchain with exponential backoff.
        """
        backoff = self.configuration.NONCE_RETRY_DELAY
        for attempt in range(self.configuration.NONCE_MAX_RETRIES):
            try:
                nonce = await self.web3.eth.get_transaction_count(self.address)
                logger.debug(f"Fetched current nonce: {nonce}")
                return nonce
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed to fetch nonce: {e}. Retrying in {backoff} seconds...")
                await asyncio.sleep(backoff)
                backoff *= 2
        raise Web3ValueError("Failed to fetch current nonce after multiple retries")

    async def _get_pending_nonce(self) -> int:
        """
        Retrieve the nonce from pending transactions.
        """
        try:
            pending = await self.web3.eth.get_transaction_count(self.address, 'pending')
            logger.debug(f"Fetched pending nonce: {pending}")
            return pending
        except Exception as e:
            logger.error(f"Error fetching pending nonce: {e}", exc_info=True)
            raise Web3ValueError(f"Failed to fetch pending nonce: {e}")

    async def track_transaction(self, tx_hash: str, nonce: int) -> None:
        """
        Track a transaction by adding its nonce to the pending set and later remove it.
        """
        self.pending_transactions.add(nonce)
        try:
            receipt = await self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=self.configuration.NONCE_TRANSACTION_TIMEOUT)
            if receipt.status == 1:
                logger.info(f"Transaction {tx_hash} (Nonce: {nonce}) succeeded.")
            else:
                logger.error(f"Transaction {tx_hash} (Nonce: {nonce}) failed with status {receipt.status}.")
        except TransactionNotFound:
            logger.warning(f"Transaction {tx_hash} (Nonce: {nonce}) not found.")
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for transaction receipt of {tx_hash} (Nonce: {nonce}).")
        except Exception as e:
            logger.error(f"Error tracking transaction {tx_hash} (Nonce: {nonce}): {e}", exc_info=True)
        finally:
            self.pending_transactions.discard(nonce)

    async def sync_nonce_with_chain(self) -> None:
        """
        Force a full synchronization with the blockchain’s nonce state.
        """
        async with self.lock:
            await self.refresh_nonce()

    async def reset(self) -> None:
        """
        Reset the nonce cache and pending transaction set.
        """
        async with self.lock:
            self.nonce_cache.clear()
            self.pending_transactions.clear()
            await self.refresh_nonce()
            logger.debug("NonceCore reset successfully.")

    async def stop(self) -> None:
        """
        Perform necessary shutdown actions for the nonce manager.
        """
        if not self._initialized:
            return
        try:
            await self.reset()
            logger.info("NonceCore stopped successfully.")
        except Exception as e:
            logger.error(f"Error stopping NonceCore: {e}", exc_info=True)

    def _should_refresh_cache(self) -> bool:
        """
        Determine if the nonce cache should be refreshed based on elapsed time.
        """
        elapsed = time.monotonic() - self.last_sync
        return elapsed > self.configuration.NONCE_CACHE_TTL

    async def get_next_nonce(self) -> int:
        """
        Retrieve the next nonce and increment the cached value.
        """
        async with self.lock:
            current_nonce = await self.get_nonce()
            next_nonce = current_nonce + 1
            self.nonce_cache[self.address] = next_nonce
            return next_nonce
