# noncecore.py
import asyncio
import time
import random
from cachetools import TTLCache
from web3 import AsyncWeb3
from web3.exceptions import Web3ValueError
from configuration import Configuration
from loggingconfig import setup_logging

logger = setup_logging("Nonce_Core", level="DEBUG")  

class NonceCore:
    """
    Manages Ethereum nonces with caching and retry logic.
    """
    def __init__(self, web3: AsyncWeb3, address: str, configuration: Configuration) -> None:
        self.web3 = web3
        self.configuration = configuration
        self.address = address
        self.lock = asyncio.Lock()
        self.nonce_cache = TTLCache(maxsize=1, ttl=self.configuration.NONCE_CACHE_TTL)
        self.last_sync = time.monotonic()
        self.pending_transactions = set()
        self._initialized = False

    async def initialize(self) -> None:
        """Initializes the nonce cache from chain and pending state."""
        if self._initialized:
            return
        await self._init_nonce()
        self._initialized = True
        asyncio.create_task(self._periodic_sync())

    async def _init_nonce(self) -> None:
        """Fetches the highest of on-chain and pending nonces."""
        current = await self._fetch_current_nonce_with_retries()
        pending = await self._get_pending_nonce()
        self.nonce_cache[self.address] = max(current, pending)
        self.last_sync = time.monotonic()

    async def get_nonce(self, force_refresh: bool = False) -> int:
        """Retrieves the next available nonce, refreshing if needed."""
        if not self._initialized:
            await self.initialize()
        if force_refresh or (time.monotonic() - self.last_sync) > self.configuration.NONCE_CACHE_TTL:
            await self.refresh_nonce()
        return self.nonce_cache.get(self.address, 0)

    async def refresh_nonce(self) -> None:
        """Refreshes nonce from the blockchain."""
        async with self.lock:
            try:
                nonce = await self.web3.eth.get_transaction_count(self.address)
                self.nonce_cache[self.address] = nonce
                self.last_sync = time.monotonic()
            except Exception as e:
                logger.error(f"Error refreshing nonce: {e}", exc_info=True)

    async def _fetch_current_nonce_with_retries(self) -> int:
        """Fetches on-chain nonce with exponential backoff and jitter."""
        delay = self.configuration.NONCE_RETRY_DELAY
        for _ in range(self.configuration.NONCE_MAX_RETRIES):
            try:
                return await self.web3.eth.get_transaction_count(self.address)
            except Exception:
                await asyncio.sleep(delay + random.uniform(0, delay))
                delay *= 2
        raise Web3ValueError("Failed to fetch current nonce")

    async def _get_pending_nonce(self) -> int:
        """Fetches the pending nonce from the node."""
        try:
            return await self.web3.eth.get_transaction_count(self.address, "pending")
        except Exception as e:
            raise Web3ValueError(f"Failed to fetch pending nonce: {e}")

    async def track_transaction(self, tx_hash: str, nonce: int) -> None:
        """Tracks a transaction until receipt and removes its nonce from pending set."""
        self.pending_transactions.add(nonce)
        try:
            await self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=self.configuration.NONCE_TRANSACTION_TIMEOUT)
        except Exception:
            pass
        finally:
            self.pending_transactions.discard(nonce)
            await self.refresh_nonce()

    async def reset(self) -> None:
        """Clears cache and synchronizes nonce."""
        async with self.lock:
            self.nonce_cache.clear()
            self.pending_transactions.clear()
            await self.refresh_nonce()

    async def stop(self) -> None:
        """Stops the nonce manager."""
        await self.reset()

    async def _periodic_sync(self) -> None:
        """Periodically refreshes the nonce cache."""
        while self._initialized:
            await asyncio.sleep(self.configuration.NONCE_CACHE_TTL)
            await self.refresh_nonce()
