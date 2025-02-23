#========================================================================================================================
# https://github.com/John0n1/0xBuilder

# This file contains the NonceCore class, which is responsible for managing nonces for Ethereum transactions.
# It ensures that we always have the correct nonce for the next transaction, even in the presence of pending transactions.
# The class uses a TTLCache to store the current nonce, and it refreshes the nonce from the chain when necessary.
#========================================================================================================================
import asyncio
import time

from cachetools import TTLCache
from web3 import AsyncWeb3
from web3.exceptions import Web3ValueError,  TransactionNotFound

from configuration import Configuration

from loggingconfig import setup_logging
import logging
logger = setup_logging("NonceCore", level=logging.INFO)

class NonceCore:
    """
    Advanced nonce management system for Ethereum transactions with caching,
    auto-recovery, and comprehensive error handling.
    """
    def __init__(
        self,
        web3: AsyncWeb3,
        address: str,
        configuration: "Configuration",
    ):
        """
        Initialize the Nonce Core.

        Args:
            web3: An AsyncWeb3 instance.
            address: The Ethereum address for which to manage nonces.
            configuration: Configuration object containing settings.
        """
        self.pending_transactions: set[int] = set()
        self.web3: AsyncWeb3 = web3
        self.configuration: "Configuration" = configuration
        self.address: str = address
        self.lock: asyncio.Lock = asyncio.Lock()
        self.nonce_cache: TTLCache = TTLCache(maxsize=1, ttl=self.configuration.NONCE_CACHE_TTL) # Use configurable CACHE_TTL
        self.last_sync: float = time.monotonic()
        self._initialized: bool = False

    async def initialize(self) -> None:
        """Initialize the nonce manager with error recovery."""
        if self._initialized:
            logger.debug("NonceCore already initialized.")
            return
        try:
            await self._init_nonce()
            self._initialized = True
            logger.debug("NonceCore initialized ✅")
        except Web3ValueError as e:
            logger.error(f"Web3ValueError during NonceCore initialization: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize NonceCore: {e}", exc_debug=True) # Include traceback
            raise

    async def _init_nonce(self) -> None:
        """Initialize nonce with fallback mechanisms."""
        try:
            current_nonce = await self._fetch_current_nonce_with_retries()
            pending_nonce = await self._get_pending_nonce()
            # Use the higher of current or pending nonce
            self.nonce_cache[self.address] = max(current_nonce, pending_nonce)
            self.last_sync = time.monotonic()
            logger.debug(f"Initial nonce set to {self.nonce_cache[self.address]}")
        except Exception as e:
            logger.error(f"Error initializing nonce: {e}", exc_info=True) # Include traceback
            raise


    async def get_nonce(self, force_refresh: bool = False) -> int:
        """Get next available nonce with optional force refresh."""
        if not self._initialized:
            await self.initialize()

        if force_refresh or self._should_refresh_cache():
            await self.refresh_nonce()

        return self.nonce_cache.get(self.address, 0)

    async def refresh_nonce(self) -> None:
        """Refresh nonce from chain with conflict resolution."""
        async with self.lock:
            try:
                current_nonce = await self.web3.eth.get_transaction_count(self.address)
                self.nonce_cache[self.address] = current_nonce
                self.last_sync = time.monotonic()
                logger.debug(f"Nonce refreshed to {current_nonce}.")
            except Exception as e:
                logger.error(f"Error refreshing nonce: {e}", exc_info=True) # Include traceback


    async def _fetch_current_nonce_with_retries(self) -> int:
        """Fetch current nonce with exponential backoff."""
        backoff = self.configuration.NONCE_RETRY_DELAY # Use configurable RETRY_DELAY
        for attempt in range(self.configuration.NONCE_MAX_RETRIES): # Use configurable MAX_RETRIES
            try:
                nonce = await self.web3.eth.get_transaction_count(self.address)
                logger.debug(f"Fetched current nonce: {nonce}") # More specific log message
                return nonce
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed to fetch current nonce: {e}. Retrying in {backoff} seconds...") # More specific warning
                await asyncio.sleep(backoff)
                backoff *= 2
        raise Web3ValueError("Failed to fetch current nonce after multiple retries") # More descriptive error message

    async def _get_pending_nonce(self) -> int:
        """Get highest nonce from pending transactions."""
        try:
            pending = await self.web3.eth.get_transaction_count(self.address, 'pending')
            logger.debug(f"Fetched pending nonce: {pending}") # More specific log message
            return pending
        except Exception as e:
            logger.error(f"Error fetching pending nonce: {e}", exc_info=True) # Include traceback
            # Instead of retrying, raise exception for upper layer to manage it.
            raise Web3ValueError(f"Failed to fetch pending nonce from provider: {e}") # More descriptive error message


    async def track_transaction(self, tx_hash: str, nonce: int) -> None:
        """Track pending transaction for nonce management."""
        self.pending_transactions.add(nonce)
        try:
            receipt = await self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=self.configuration.NONCE_TRANSACTION_TIMEOUT) # Use configurable TRANSACTION_TIMEOUT
            if receipt.status == 1:
                logger.info(f"Transaction {tx_hash} (Nonce: {nonce}) succeeded.") # Added nonce to log
            else:
                logger.error(f"Transaction {tx_hash} (Nonce: {nonce}) failed with status {receipt.status}.") # Added nonce and status to log
        except TransactionNotFound: # Catch specific Web3 exception
            logger.warning(f"Transaction {tx_hash} (Nonce: {nonce}) not found during receipt fetching.") # More specific warning
        except asyncio.TimeoutError: # Catch specific asyncio exception
            logger.warning(f"Timeout waiting for transaction receipt of {tx_hash} (Nonce: {nonce}).") # More specific warning
        except Exception as e:
            logger.error(f"Error tracking transaction {tx_hash} (Nonce: {nonce}): {e}", exc_info=True) # Include traceback and nonce
        finally:
            self.pending_transactions.discard(nonce)

    async def _handle_nonce_error(self) -> None:
        """Handle nonce-related errors with recovery attempt."""
        logger.warning("Handling nonce-related error. Refreshing nonce.")
        await self.refresh_nonce()

    async def sync_nonce_with_chain(self) -> None:
        """Force synchronization with chain state."""
        async with self.lock:
            await self.refresh_nonce()

    async def reset(self) -> None:
        """Complete reset of nonce manager state."""
        async with self.lock:
            self.nonce_cache.clear()
            self.pending_transactions.clear()
            await self.refresh_nonce()
            logger.debug("NonceCore reset. OK ✅")

    async def stop(self) -> None:
        """Stop nonce manager operations."""
        if not self._initialized:
            return
        try:
            await self.reset()
            logger.info("Nonce Core stopped successfully.")
        except Exception as e:
            logger.error(f"Error stopping Nonce Core: {e}", exc_info=True) # Include traceback

    def _should_refresh_cache(self) -> bool:
        """Check if the nonce cache should be refreshed based on time."""
        return time.monotonic() - self.last_sync > self.configuration.NONCE_CACHE_TTL # Use configurable CACHE_TTL

    async def get_next_nonce(self) -> int:
        """Fetch the next available nonce."""
        async with self.lock:
            current_nonce = await self.get_nonce()
            next_nonce = current_nonce + 1
            self.nonce_cache[self.address] = next_nonce
            return next_nonce