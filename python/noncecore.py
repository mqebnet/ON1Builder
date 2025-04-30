import asyncio
import time
from typing import Dict, Optional, Tuple
from cachetools import TTLCache
from hexbytes import HexBytes
from web3 import AsyncWeb3
from web3.types import Nonce, Address, HexStr, TxReceipt
from web3.exceptions import Web3ValueError, TransactionNotFound
from configuration import Configuration # Facade import
from loggingconfig import setup_logging
import logging

logger = setup_logging("NonceCore", level=logging.DEBUG)

class NonceCore:
    """
    Manages nonces for an Ethereum account, handling caching, synchronization,
    retries, and tracking of pending transactions to prevent reuse conflicts.
    Ensures thread-safe access to the nonce cache.
    """
    DEFAULT_CACHE_TTL = 60 # seconds
    DEFAULT_RETRY_DELAY = 0.5 # seconds
    DEFAULT_MAX_RETRIES = 5
    DEFAULT_TX_TIMEOUT = 180 # seconds

    def __init__(self, web3: AsyncWeb3, address: Address, configuration: Configuration) -> None:
        """Initializes NonceCore."""
        self.web3: AsyncWeb3 = web3
        self.configuration: Configuration = configuration
        self.address: Address = web3.to_checksum_address(address) # Ensure checksum

        # A7: Lock for protecting nonce_cache and related state (last_sync)
        self.lock: asyncio.Lock = asyncio.Lock()

        # Load configuration values
        self.cache_ttl: int = self.configuration.get_config_value("NONCE_CACHE_TTL", self.DEFAULT_CACHE_TTL)
        self.retry_delay: float = self.configuration.get_config_value("NONCE_RETRY_DELAY", self.DEFAULT_RETRY_DELAY)
        self.max_retries: int = self.configuration.get_config_value("NONCE_MAX_RETRIES", self.DEFAULT_MAX_RETRIES)
        self.tx_timeout: int = self.configuration.get_config_value("NONCE_TRANSACTION_TIMEOUT", self.DEFAULT_TX_TIMEOUT)

        # A7: Nonce cache protected by self.lock
        # Store tuple: (nonce, timestamp)
        self._nonce_cache: Dict[Address, Tuple[Nonce, float]] = {}
        # self.last_sync: float = 0.0 # Replaced by timestamp in cache

        # Track nonces currently used in transactions sent but not yet confirmed/failed
        self.pending_nonces: set[Nonce] = set() # Also potentially needs locking if modified outside main async tasks
        self._pending_lock: asyncio.Lock = asyncio.Lock() # Separate lock for pending set

        self._initialized: bool = False
        self._init_lock: asyncio.Lock = asyncio.Lock() # Lock for initialization itself

        logger.info("NonceCore created for address %s.", self.address)

    async def initialize(self) -> None:
        """Initializes the nonce manager by fetching the current nonce."""
        async with self._init_lock:
            if self._initialized:
                logger.debug("NonceCore already initialized.")
                return
            logger.info("Initializing NonceCore for %s...", self.address)
            try:
                # Perform initial fetch and cache population under lock
                await self.refresh_nonce(force=True)
                self._initialized = True
                logger.info("NonceCore initialized successfully.")
            except Exception as e:
                logger.critical("Failed to initialize NonceCore: %s", e, exc_info=True)
                raise # Propagate initialization failure


    async def _fetch_nonce_from_chain(self, block_identifier="pending") -> Nonce:
        """Fetches nonce from the blockchain with retries."""
        current_retry_delay = self.retry_delay
        for attempt in range(self.max_retries):
            try:
                # A2: Parameterized logging
                logger.debug("Fetching nonce for %s (block: %s, attempt %d/%d)...",
                             self.address, block_identifier, attempt + 1, self.max_retries)
                nonce: Nonce = await self.web3.eth.get_transaction_count(self.address, block_identifier=block_identifier)
                logger.debug("Fetched nonce %d for %s (block: %s).", nonce, self.address, block_identifier)
                return nonce
            except asyncio.TimeoutError:
                 logger.warning("Timeout fetching nonce (attempt %d/%d). Retrying...", attempt + 1, self.max_retries)
            except Exception as e:
                # A2: Parameterized logging
                logger.warning(
                    "Failed to fetch nonce (attempt %d/%d): %s. Retrying in %.2f seconds...",
                    attempt + 1, self.max_retries, e, current_retry_delay
                )
            if attempt < self.max_retries - 1:
                await asyncio.sleep(current_retry_delay)
                current_retry_delay *= 1.5 # Exponential backoff

        logger.error("Failed to fetch nonce for %s after %d attempts.", self.address, self.max_retries)
        raise Web3ValueError(f"Could not fetch nonce for {self.address} after {self.max_retries} retries.")


    async def refresh_nonce(self, force: bool = False) -> Nonce:
        """
        Refreshes the nonce from the blockchain ('pending' state).
        Protected by lock.

        Args:
            force: If True, forces refresh even if cache is valid.

        Returns:
            The refreshed nonce.
        """
        async with self.lock: # A7: Acquire lock for cache write
            now = time.monotonic()
            cached_nonce, cache_time = self._nonce_cache.get(self.address, (Nonce(-1), 0.0))

            # Check if refresh is needed
            if not force and cached_nonce != -1 and (now - cache_time) < self.cache_ttl:
                 logger.debug("Nonce refresh skipped, cache is still valid (age %.2fs < TTL %ds).", now - cache_time, self.cache_ttl)
                 return cached_nonce

            # Fetch nonce from chain ('pending' is usually best for next nonce)
            try:
                 refreshed_nonce = await self._fetch_nonce_from_chain(block_identifier="pending")
                 self._nonce_cache[self.address] = (refreshed_nonce, now)
                 logger.info("Nonce cache refreshed for %s to %d.", self.address, refreshed_nonce)
                 return refreshed_nonce
            except Web3ValueError as e:
                 logger.error("Failed to refresh nonce cache due to fetch error: %s", e)
                 # Keep stale cache value if fetch fails? Or clear it?
                 # Keep stale for now, get_next_nonce logic might handle it.
                 # If cache existed, return the stale value, otherwise raise
                 if cached_nonce != -1:
                      logger.warning("Returning stale nonce %d due to refresh failure.", cached_nonce)
                      return cached_nonce
                 else:
                      raise # Reraise if no previous value exists


    async def get_nonce(self) -> Nonce:
        """
        Gets the current expected nonce, refreshing from chain if cache is stale.
        Protected by lock.

        Returns:
            The current nonce to be used.
        """
        if not self._initialized: await self.initialize()

        async with self.lock: # A7: Acquire lock for cache read/potential write
            now = time.monotonic()
            cached_nonce, cache_time = self._nonce_cache.get(self.address, (Nonce(-1), 0.0))

            if cached_nonce == -1 or (now - cache_time) >= self.cache_ttl:
                 # Cache miss or expired, need to refresh (refresh handles logging)
                 # Call refresh within the current lock context
                 try:
                      return await self.refresh_nonce(force=True) # Force refresh
                 except Web3ValueError:
                      # If refresh fails and cache was initially empty, we have a problem
                      if cached_nonce == -1:
                           logger.critical("Cannot determine initial nonce for %s!", self.address)
                           raise RuntimeError(f"Failed to determine initial nonce for {self.address}")
                      else:
                           # Should not happen if refresh_nonce handles stale return on error
                           logger.error("Unexpected state: Refresh failed but stale cache existed.")
                           return cached_nonce # Return stale as last resort
            else:
                 # Cache hit and valid
                 logger.debug("Using cached nonce %d for %s.", cached_nonce, self.address)
                 return cached_nonce


    async def get_next_nonce(self) -> Nonce:
        """
        Atomically retrieves the current nonce and increments the cached value
        in anticipation of its use. Protected by lock.

        Returns:
            The nonce value that should be used for the next transaction.
        """
        if not self._initialized: await self.initialize()

        async with self.lock: # A7: Acquire lock for atomic read-increment-write
            # Get current nonce (potentially triggers refresh if needed)
            current_nonce = await self.get_nonce() # This call handles cache validity check

            # Check if this nonce is already marked as pending (should ideally not happen with get_next_nonce logic)
            async with self._pending_lock:
                 if current_nonce in self.pending_nonces:
                      logger.warning("Attempting to use nonce %d which is already pending! Trying next one.", current_nonce)
                      # This indicates a potential logic issue elsewhere or rapid tx submission.
                      # We should probably increment past the pending ones.
                      # Find the lowest nonce NOT in pending_nonces >= current_nonce
                      next_available = current_nonce
                      while next_available in self.pending_nonces:
                           next_available = Nonce(next_available + 1)
                      logger.warning("Adjusted next nonce to %d to skip pending ones.", next_available)
                      current_nonce = next_available


            # Assume 'current_nonce' is the one to *use* for the transaction.
            # Increment the cache to reflect the *next* expected nonce after this one is used.
            next_expected_nonce = Nonce(current_nonce + 1)
            self._nonce_cache[self.address] = (next_expected_nonce, time.monotonic())

            # A2: Parameterized logging
            logger.info("Providing nonce %d for use. Cache incremented to %d.", current_nonce, next_expected_nonce)

            # Mark the provided nonce as pending *before* returning
            async with self._pending_lock:
                 self.pending_nonces.add(current_nonce)
                 logger.debug("Nonce %d marked as pending. Pending set size: %d", current_nonce, len(self.pending_nonces))

            return current_nonce # Return the nonce that was just reserved


    async def track_transaction(self, tx_hash: HexStr, nonce: Nonce) -> None:
        """
        Monitors a sent transaction and removes its nonce from the pending set
        once it's confirmed (succeeded or failed) or timed out.
        """
        log_extra = {"component": "NonceCore", "tx_hash": tx_hash, "nonce": nonce}
        logger.debug("Tracking transaction %s (Nonce %d)...", tx_hash, nonce, extra=log_extra)
        receipt: Optional[TxReceipt] = None
        try:
            receipt = await self.web3.eth.wait_for_transaction_receipt(HexBytes(tx_hash), timeout=self.tx_timeout)
            if receipt.status == 1:
                logger.info("Transaction %s (Nonce %d) confirmed successfully.", tx_hash, nonce, extra=log_extra)
            else:
                logger.warning("Transaction %s (Nonce %d) confirmed but FAILED (reverted).", tx_hash, nonce, extra=log_extra)
                # Consider if failed nonce should trigger immediate refresh? Maybe not.

        except TransactionNotFound:
            # A2: Parameterized logging
            logger.warning("Transaction %s (Nonce %d) not found after sending (orphaned or dropped?).", tx_hash, nonce, extra=log_extra)
        except asyncio.TimeoutError:
            # A2: Parameterized logging
            logger.warning("Timeout (%ds) waiting for receipt of tx %s (Nonce %d). Assuming stuck or dropped.",
                         self.tx_timeout, tx_hash, nonce, extra=log_extra)
            # If a tx times out, the nonce might be stuck. Consider trying to cancel it or forcing a refresh.
            # await self.force_nonce_refresh_and_clear_pending(nonce) # Example recovery
        except Exception as e:
            # A2: Parameterized logging
            logger.error("Error tracking transaction %s (Nonce %d): %s", tx_hash, nonce, e, exc_info=True, extra=log_extra)
        finally:
            # Always remove the nonce from the pending set once tracking finishes
            async with self._pending_lock:
                if nonce in self.pending_nonces:
                     self.pending_nonces.discard(nonce)
                     logger.debug("Nonce %d removed from pending set. Pending size: %d", nonce, len(self.pending_nonces), extra=log_extra)
                else:
                     # This might happen if reset() was called during tracking
                     logger.debug("Nonce %d was not in pending set upon tracking completion.", nonce, extra=log_extra)

            # Optional: Refresh nonce if the transaction failed or timed out,
            # as the chain's nonce might not have advanced.
            if receipt is None or receipt.status == 0:
                 logger.debug("Triggering nonce refresh due to failed/timed-out tx %s (Nonce %d).", tx_hash, nonce, extra=log_extra)
                 try:
                      await self.refresh_nonce(force=True)
                 except Exception as refresh_err:
                      logger.error("Failed to refresh nonce after failed/timed-out tx %s: %s", tx_hash, refresh_err, extra=log_extra)


    async def reset(self) -> None:
        """Resets the nonce cache and pending transactions, then re-initializes."""
        logger.warning("Resetting NonceCore state...")
        async with self.lock: # Lock cache access during reset
             self._nonce_cache.clear()
        async with self._pending_lock: # Lock pending set access
             self.pending_nonces.clear()
        self._initialized = False # Force re-initialization
        logger.info("NonceCore state cleared. Re-initializing...")
        # Re-initialize by fetching current nonce
        await self.initialize()
        logger.info("NonceCore reset complete.")


    async def stop(self) -> None:
        """Performs cleanup actions for NonceCore."""
        logger.info("Stopping NonceCore.")
        # No explicit network resources to close here, just clear state
        async with self.lock:
             self._nonce_cache.clear()
        async with self._pending_lock:
             self.pending_nonces.clear()
        self._initialized = False
        logger.info("NonceCore stopped.")
