import asyncio
from decimal import Decimal
import os
import tracemalloc
import time
import sys
import signal
from typing import Any, Dict, List, Optional, Set, Type, Union

from web3 import AsyncWeb3
from web3.types import Wei
from web3.eth import AsyncEth
from web3.middleware import ExtraDataToPOAMiddleware
from web3.providers.async_base import AsyncBaseProvider
from web3 import AsyncIPCProvider, AsyncHTTPProvider
from web3 import WebSocketProvider
from eth_account import Account
from eth_account.signers.local import LocalAccount # Specific type

# Assuming configuration facade
from configuration import Configuration
# Import components
from apiconfig import APIConfig
from marketmonitor import MarketMonitor
from mempoolmonitor import MempoolMonitor
from noncecore import NonceCore
from safetynet import WEI_PER_ETH, SafetyNet
from strategynet import StrategyNet
from transactioncore import TransactionCore

from loggingconfig import setup_logging
import logging

logger = setup_logging("MainCore", level=logging.DEBUG)

ComponentType = Union[APIConfig, MarketMonitor, MempoolMonitor, NonceCore, SafetyNet, StrategyNet, TransactionCore]


class MainCore:
    """
    Orchestrates all ON1Builder components, manages the main application lifecycle,
    monitors memory usage, handles signals, and ensures graceful shutdown.
    """
    # Default intervals (can be overridden by configuration)
    DEFAULT_MEMORY_CHECK_INTERVAL = 300 # seconds
    DEFAULT_HEALTH_CHECK_INTERVAL = 60 # seconds
    DEFAULT_PROFITABLE_TX_TIMEOUT = 2.0 # seconds
    DEFAULT_WEB3_RETRIES = 3
    DEFAULT_WEB3_RETRY_DELAY = 5 # seconds

    def __init__(self, configuration: Configuration) -> None:
        self.configuration = configuration
        self.web3: Optional[AsyncWeb3] = None
        self.account: Optional[LocalAccount] = None
        self.running: bool = False
        self._shutdown_event: asyncio.Event = asyncio.Event()
        self._tasks: Set[asyncio.Task] = set() # Store main loop tasks

        # Initialize components dictionary with expected types (or None)
        self.components: Dict[str, Optional[ComponentType]] = {
            "apiconfig": None,
            "noncecore": None,
            "safetynet": None,
            "transactioncore": None, # A5: Initialize early
            "marketmonitor": None,
            "mempoolmonitor": None,
            "strategynet": None,
        }
        # Track health status (True = healthy, False = unhealthy/unknown)
        self._component_health: Dict[str, bool] = {name: False for name in self.components}

        # Load intervals and settings from config
        self.memory_check_interval = self.configuration.get_config_value("MEMORY_CHECK_INTERVAL", self.DEFAULT_MEMORY_CHECK_INTERVAL)
        self.health_check_interval = self.configuration.get_config_value("COMPONENT_HEALTH_CHECK_INTERVAL", self.DEFAULT_HEALTH_CHECK_INTERVAL)
        self.profitable_tx_timeout = self.configuration.get_config_value("PROFITABLE_TX_PROCESS_TIMEOUT", self.DEFAULT_PROFITABLE_TX_TIMEOUT)
        self.web3_max_retries = self.configuration.get_config_value("WEB3_MAX_RETRIES", self.DEFAULT_WEB3_RETRIES)
        self.web3_retry_delay = self.configuration.get_config_value("WEB3_RETRY_DELAY", self.DEFAULT_WEB3_RETRY_DELAY)

        self.loop = asyncio.get_running_loop() # Get current event loop

        # Tracemalloc setup
        self.memory_snapshot: Optional[tracemalloc.Snapshot] = None
        if self.configuration.get_config_value("ENABLE_MEMORY_PROFILING", True): # Check config flag
            tracemalloc.start(25) # Increase stack depth for better context
            self.memory_snapshot = tracemalloc.take_snapshot()
            # A2: Parameterized logging
            logger.info("Tracemalloc memory tracing started.")
        else:
            # A2: Parameterized logging
             logger.info("Memory tracing disabled by configuration.")

        # A2: Parameterized logging
        logger.info("Initializing ON1Builder MainCore...")

    async def _load_configuration(self) -> None:
        """Loads configuration using the Configuration class."""
        try:

            await self.configuration.load() # If load method is async
            logger.info("Configuration loaded/validated successfully.")
        except Exception as e:
            # A2: Parameterized logging
            logger.critical("Configuration load/validation failed: %s", e, exc_info=True)
            raise # Propagate critical error

    async def _initialize_web3(self) -> AsyncWeb3:
        """Initializes and validates the AsyncWeb3 connection."""
        provider = await self._get_provider()
        if not provider:
             raise RuntimeError("No valid Web3 provider endpoint configured or available.")

        web3 = AsyncWeb3(provider, modules={"eth": (AsyncEth,)})

        # Test connection with timeout and retries
        connected = False
        for attempt in range(self.web3_max_retries):
            try:
                # A2: Parameterized logging
                logger.debug("Attempting Web3 connection (attempt %d/%d)...", attempt + 1, self.web3_max_retries)
                # Use request_timeout for provider if supported, or asyncio.wait_for
                # Some providers might handle timeout internally
                async with asyncio.timeout(15): # 15 second timeout for connection check
                    if await web3.is_connected():
                        chain_id = await web3.eth.chain_id
                        # A2: Parameterized logging
                        logger.info("Web3 connected successfully to chain ID %d via %s.", chain_id, type(provider).__name__)
                        connected = True
                        break
                    else:
                         logger.warning("web3.is_connected() returned False on attempt %d.", attempt + 1)

            except asyncio.TimeoutError:
                 # A2: Parameterized logging
                 logger.warning("Web3 connection attempt %d timed out.", attempt + 1)
            except Exception as e:
                # A2: Parameterized logging
                logger.warning("Web3 connection attempt %d failed: %s", attempt + 1, e)

            if not connected and attempt < self.web3_max_retries - 1:
                await asyncio.sleep(self.web3_retry_delay * (attempt + 1)) # Exponential backoff

        if not connected:
            raise RuntimeError(f"Failed to connect to Web3 provider after {self.web3_max_retries} attempts.")

        await self._add_middleware(web3)
        return web3

    async def _get_provider(self) -> Optional[AsyncBaseProvider]:
        """Selects the best available asynchronous provider based on configuration."""
        # Try WebSocket first (usually preferred for monitoring)
        if self.configuration.WEBSOCKET_ENDPOINT:
            try:
                # A2: Parameterized logging
                logger.debug("Attempting WebSocket connection to %s...", self.configuration.WEBSOCKET_ENDPOINT)
                # Increased timeout for initial connection
                provider =WebSocketProvider(self.configuration.WEBSOCKET_ENDPOINT, websocket_timeout=60)
                # Test the connection immediately (though AsyncWeb3 does this too)
                # await provider.connect() # connect is often implicit or handled by web3.py
                # Perform a simple request to verify
                async with asyncio.timeout(15):
                     await provider.make_request("web3_clientVersion", [])
                # A2: Parameterized logging
                logger.info("Using WebSocket Provider: %s", self.configuration.WEBSOCKET_ENDPOINT)
                return provider
            except Exception as e:
                # A2: Parameterized logging
                logger.warning(
                    "WebSocket provider failed (%s): %s. Trying HTTP.",
                     self.configuration.WEBSOCKET_ENDPOINT, e
                )

        # Fallback to HTTP
        if self.configuration.HTTP_ENDPOINT:
            try:
                # A2: Parameterized logging
                logger.debug("Attempting HTTP connection to %s...", self.configuration.HTTP_ENDPOINT)
                # Set request timeout if needed (e.g., request_kwargs={'timeout': 10})
                provider = AsyncHTTPProvider(self.configuration.HTTP_ENDPOINT)
                 # Perform a simple request to verify
                async with asyncio.timeout(10):
                     await provider.make_request("eth_blockNumber", [])
                # A2: Parameterized logging
                logger.info("Using HTTP Provider: %s", self.configuration.HTTP_ENDPOINT)
                return provider
            except Exception as e:
                # A2: Parameterized logging
                logger.warning(
                    "HTTP provider failed (%s): %s. Trying IPC.",
                    self.configuration.HTTP_ENDPOINT, e
                )

        # Fallback to IPC
        if self.configuration.IPC_ENDPOINT:
            try:
                # A2: Parameterized logging
                logger.debug("Attempting IPC connection to %s...", self.configuration.IPC_ENDPOINT)
                provider = AsyncIPCProvider(self.configuration.IPC_ENDPOINT, timeout=15)
                 # Perform a simple request to verify
                async with asyncio.timeout(10):
                     await provider.make_request("eth_blockNumber", [])
                # A2: Parameterized logging
                logger.info("Using IPC Provider: %s", self.configuration.IPC_ENDPOINT)
                return provider
            except Exception as e:
                # A2: Parameterized logging
                logger.warning("IPC provider failed (%s): %s.", self.configuration.IPC_ENDPOINT, e)

        # A2: Parameterized logging
        logger.error("No functional Web3 provider found in configuration.")
        return None

    async def _add_middleware(self, web3: AsyncWeb3) -> None:
        """Adds necessary middleware (e.g., PoA) based on chain ID."""
        try:
            chain_id = await web3.eth.chain_id
            # Common PoA Chain IDs (add others if needed)
            # Mainnets: Binance Smart Chain (56), Polygon (137)
            # Testnets: Goerli (5), Sepolia (11155111), Rinkeby (4 - deprecated), Ropsten (3 - deprecated)
            # Others: xDai/Gnosis (100), Energy Web (246), POA Core (99)
            poa_chains = {56, 137, 100, 99} # Example set
            # Testnets might also need it
            testnet_poa_chains = {5, 11155111} # Goerli, Sepolia

            if chain_id in poa_chains or chain_id in testnet_poa_chains:
                # Inject PoA middleware for PoA chains
                web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
                # A2: Parameterized logging
                logger.info("Injected PoA middleware for chain ID %d.", chain_id)
            else:
                 # A2: Parameterized logging
                 logger.debug("No specific middleware needed for chain ID %d.", chain_id)
        except Exception as e:
            # A2: Parameterized logging
            logger.error("Failed to check chain ID or inject middleware: %s", e, exc_info=True)
            # Decide if this is critical - potentially raise error
            # raise RuntimeError(f"Middleware injection failed: {e}")


    async def _check_account_balance(self) -> None:
        """Checks if the configured account has a sufficient minimum balance."""
        if not self.account or not self.web3:
            raise ValueError("Account or Web3 not initialized for balance check.")

        try:
            # Load MIN_BALANCE from config, default to a small value
            min_balance_eth_str = self.configuration.get_config_value("MIN_BALANCE", "0.001")
            min_balance_eth = Decimal(min_balance_eth_str)

            balance_wei: Wei = await self.web3.eth.get_balance(self.account.address)
            balance_eth = Decimal(balance_wei) / WEI_PER_ETH

            # A2: Parameterized logging
            logger.info(
                "Account %s Balance: %.8f ETH",
                self.account.address, balance_eth
            )

            if balance_eth < min_balance_eth:
                # A2: Parameterized logging
                logger.warning(
                    "Account balance %.8f ETH is below the configured minimum of %s ETH!",
                    balance_eth, min_balance_eth_str
                )
                # Decide if this should halt operation (e.g., raise Exception)
                # raise RuntimeError("Insufficient account balance to operate.")
            else:
                 logger.debug("Account balance check passed.")

        except ValueError: # Handle case where MIN_BALANCE in config is invalid
            logger.error("Invalid MIN_BALANCE format in configuration: '%s'. Using default 0.001 ETH.", min_balance_eth_str)
            if await self.web3.eth.get_balance(self.account.address) < self.web3.to_wei(Decimal("0.001"), "ether"):
                 logger.warning("Account balance is below the default minimum of 0.001 ETH.")
        except Exception as e:
            # A2: Parameterized logging
            logger.error("Failed to check account balance: %s", e, exc_info=True)
            raise # Propagate error if balance check fails critically


    async def initialize_components(self) -> None:
        """
        Initializes all ON1Builder components in the correct order, handling dependencies.
        """
        # A2: Parameterized logging
        logger.info("Initializing ON1Builder components...")
        try:
            # 1. Load Configuration (should be done in __init__ or early sync)
            # await self._load_configuration() # Assuming async load exists

            # 2. Initialize Web3 Connection & Account
            self.web3 = await self._initialize_web3()
            try:
                self.account = Account.from_key(self.configuration.WALLET_KEY)
                logger.info("Wallet account loaded for address: %s", self.account.address)
                 # Validate checksum if needed, though from_key handles format
                if not self.web3.is_address(self.configuration.WALLET_ADDRESS) or \
                   self.web3.to_checksum_address(self.configuration.WALLET_ADDRESS) != self.account.address:
                     logger.warning("Configured WALLET_ADDRESS does not match the private key's address!")
                     # Potentially raise error here if consistency is critical
            except Exception as e:
                logger.critical("Failed to load wallet account from WALLET_KEY: %s", e)
                raise ValueError("Invalid WALLET_KEY provided.") from e

            await self._check_account_balance() # Check balance early

            # 3. Initialize Core Utilities (API Config, Nonce, SafetyNet)
            logger.debug("Initializing core utilities...")
            self.components["apiconfig"] = APIConfig(self.configuration)
            await self.components["apiconfig"].initialize()
            self._component_health["apiconfig"] = True

            self.components["noncecore"] = NonceCore(self.web3, self.account.address, self.configuration)
            await self.components["noncecore"].initialize()
            self._component_health["noncecore"] = True

            # SafetyNet needs web3, config, account, apiconfig
            self.components["safetynet"] = SafetyNet(
                web3=self.web3,
                configuration=self.configuration,
                account=self.account,
                apiconfig=self.components["apiconfig"]
            )
            await self.components["safetynet"].initialize()
            self._component_health["safetynet"] = True

            # A5: Instantiate TransactionCore *before* MarketMonitor
            logger.debug("Initializing TransactionCore...")
            self.components["transactioncore"] = TransactionCore(
                web3=self.web3,
                account=self.account,
                configuration=self.configuration,
                noncecore=self.components["noncecore"],
                safetynet=self.components["safetynet"],
                apiconfig=self.components["apiconfig"],
                # marketmonitor will be injected later if needed by TxCore
            )
            # TransactionCore init is potentially heavy (ABI loading), await it
            await self.components["transactioncore"].initialize()
            self._component_health["transactioncore"] = True


            # 5. Initialize Monitoring Components (MarketMonitor, MempoolMonitor)
            logger.debug("Initializing monitoring components...")
            # A5: Inject TransactionCore into MarketMonitor constructor
            self.components["marketmonitor"] = MarketMonitor(
                web3=self.web3, # Pass web3 if MarketMonitor needs it
                configuration=self.configuration,
                apiconfig=self.components["apiconfig"],
                transactioncore=self.components["transactioncore"] # Inject TxCore
            )
            await self.components["marketmonitor"].initialize()
            self._component_health["marketmonitor"] = True

            # Inject MarketMonitor back into TransactionCore if TxCore needs it post-init
            # (Dependency injection or post-init setup)
            # Example: self.components["transactioncore"].set_market_monitor(self.components["marketmonitor"])

            # MempoolMonitor needs web3, safetynet, noncecore, apiconfig, config, marketmonitor
            token_symbols_or_addresses = await self.configuration.get_active_tokens() # Assumes a method in config
            self.components["mempoolmonitor"] = MempoolMonitor(
                web3=self.web3,
                safetynet=self.components["safetynet"],
                noncecore=self.components["noncecore"],
                apiconfig=self.components["apiconfig"],
                monitored_tokens=token_symbols_or_addresses,
                configuration=self.configuration,
                marketmonitor=self.components["marketmonitor"]
            )
            await self.components["mempoolmonitor"].initialize()
            self._component_health["mempoolmonitor"] = True


            # 6. Initialize Strategy Network
            logger.debug("Initializing StrategyNet...")
            self.components["strategynet"] = StrategyNet(
                transactioncore=self.components["transactioncore"],
                marketmonitor=self.components["marketmonitor"],
                safetynet=self.components["safetynet"],
                config=self.configuration # Pass main config
                # apiconfig is accessible via other components if needed
            )
            await self.components["strategynet"].initialize()
            self._component_health["strategynet"] = True

            # A2: Parameterized logging
            logger.info("All ON1Builder components initialized successfully.")

        except Exception as e:
            # A2: Parameterized logging
            logger.critical("Component initialization failed: %s", e, exc_info=True)
            # Perform partial cleanup of already initialized components before raising
            await self.stop_components() # Call stop on successfully initialized ones
            raise # Re-raise the critical error


    async def _check_component_health(self) -> None:
        """Periodically checks the health of registered components."""
        # A2: Parameterized logging
        logger.info("Starting component health check loop (Interval: %d s).", self.health_check_interval)
        while self.running:
            await asyncio.sleep(self.health_check_interval)
            if not self.running: break # Check after sleep

            # A2: Parameterized logging
            logger.debug("Running component health checks...")
            all_healthy = True
            temp_health = {}

            for name, component in self.components.items():
                if component is None:
                    # Component failed initialization or was never created
                    temp_health[name] = False
                    all_healthy = False
                    continue

                # Check if component has a specific health check method
                if hasattr(component, "is_healthy") and callable(component.is_healthy):
                    try:
                        is_healthy = await component.is_healthy()
                        temp_health[name] = is_healthy
                        if not is_healthy:
                            all_healthy = False
                            # A2: Parameterized logging
                            logger.warning("Component '%s' reported unhealthy.", name)
                    except Exception as e:
                        temp_health[name] = False
                        all_healthy = False
                        # A2: Parameterized logging
                        logger.error("Error during health check for component '%s': %s", name, e, exc_info=True)
                else:
                    # Assume healthy if initialized and no specific check exists
                    temp_health[name] = True

            self._component_health = temp_health # Update health status atomically

            if not all_healthy:
                unhealthy_list = [name for name, healthy in self._component_health.items() if not healthy]
                # A2: Parameterized logging
                logger.warning("Unhealthy components detected: %s", unhealthy_list)
                # TODO: Implement recovery logic or trigger emergency shutdown if needed


    async def _monitor_memory(self) -> None:
        """Periodically monitors memory usage using tracemalloc."""
        if not tracemalloc.is_tracing():
            logger.info("Memory monitoring disabled (tracemalloc not active).")
            return

        # A2: Parameterized logging
        logger.info("Starting memory monitoring loop (Interval: %d s).", self.memory_check_interval)
        last_snapshot = self.memory_snapshot or tracemalloc.take_snapshot()

        while self.running:
            await asyncio.sleep(self.memory_check_interval)
            if not self.running: break # Check after sleep

            try:
                current_snapshot = tracemalloc.take_snapshot()
                if last_snapshot:
                    stats = current_snapshot.compare_to(last_snapshot, "lineno")
                    total_diff = sum(s.size_diff for s in stats)
                    total_current = sum(s.size for s in current_snapshot.statistics("lineno"))

                    # A2: Parameterized logging
                    logger.debug(
                        "Memory Usage: Current=%.2f MiB, Diff since last check=%.2f MiB",
                        total_current / (1024 * 1024),
                        total_diff / (1024 * 1024)
                    )

                    # Log significant changes
                    if abs(total_diff) > (5 * 1024 * 1024): # Log if diff > 5 MiB
                         # A2: Parameterized logging
                         logger.warning("Significant memory change detected: %.2f MiB", total_diff / (1024*1024))
                         logger.warning("Top 5 memory differences:")
                         for i, stat in enumerate(stats[:5]):
                             logger.warning(f"  {i+1}: {stat}")

                last_snapshot = current_snapshot

            except Exception as e:
                 # A2: Parameterized logging
                 logger.error("Error during memory monitoring: %s", e, exc_info=True)
                 # Avoid excessive logging on repeated errors
                 await asyncio.sleep(self.memory_check_interval * 2)


    async def _process_profitable_transactions(self) -> None:
        """Continuously processes transactions deemed profitable by MempoolMonitor."""
        if not self.components.get("mempoolmonitor") or not self.components.get("strategynet"):
             logger.error("MempoolMonitor or StrategyNet not available for processing transactions.")
             return

        mempool_monitor: MempoolMonitor = self.components["mempoolmonitor"]
        strategy_net: StrategyNet = self.components["strategynet"]

        # A2: Parameterized logging
        logger.info("Starting profitable transaction processing loop...")

        while self.running:
            try:
                # Wait for a transaction from the queue with a timeout
                target_tx = await asyncio.wait_for(
                    mempool_monitor.profitable_transactions.get(),
                    timeout=self.profitable_tx_timeout
                )

                tx_hash = target_tx.get('tx_hash', 'N/A') # A16
                strategy_type = target_tx.get("strategy_type", "unknown")
                log_extra = {"component": "MainCore", "tx_hash": tx_hash, "strategy_type": strategy_type} # A16

                # A2: Parameterized logging
                logger.debug("Dequeued profitable tx %s for strategy type '%s'.", tx_hash, strategy_type, extra=log_extra)

                # Execute the best strategy for this transaction
                success = await strategy_net.execute_best_strategy(target_tx, strategy_type)

                # A2 + A16: Parameterized logging + context
                logger.info(
                    "Processing complete for tx %s (Strategy: %s). Success: %s",
                    tx_hash, strategy_type, success, extra=log_extra
                )

                mempool_monitor.profitable_transactions.task_done()

            except asyncio.TimeoutError:
                # No transaction received within the timeout, continue loop
                continue
            except asyncio.CancelledError:
                logger.info("Profitable transaction processing task cancelled.")
                break # Exit loop cleanly
            except Exception as e:
                # A2: Parameterized logging
                logger.error("Error processing profitable transaction: %s", e, exc_info=True, extra={"component": "MainCore"})
                # Avoid tight loop on continuous errors
                await asyncio.sleep(1)


    async def run(self) -> None:
        """Starts all background tasks and waits for shutdown signal."""
        if self.running:
            logger.warning("MainCore run loop is already active.")
            return

        # A2: Parameterized logging
        logger.info("Starting MainCore run loop...")
        self.running = True
        self.setup_signal_handlers()

        # Create main tasks
        try:
             # Get component instances (assume they are initialized)
             mempool_monitor = self.components.get("mempoolmonitor")

             # Start background tasks using asyncio.create_task and store handles
             if mempool_monitor and hasattr(mempool_monitor, "start_monitoring"):
                  self._add_task(asyncio.create_task(mempool_monitor.start_monitoring(), name="MempoolMonitorTask"))

             self._add_task(asyncio.create_task(self._process_profitable_transactions(), name="TxProcessorTask"))

             if tracemalloc.is_tracing():
                  self._add_task(asyncio.create_task(self._monitor_memory(), name="MemoryMonitorTask"))

             self._add_task(asyncio.create_task(self._check_component_health(), name="HealthCheckTask"))

             logger.info("All %d main tasks created. Waiting for completion or shutdown...", len(self._tasks))

             # Wait for shutdown event or tasks finishing/failing
             # await self._shutdown_event.wait() # Wait for explicit stop signal

             # Alternative: Wait for any task to complete (potentially due to error)
             if self._tasks:
                  done, pending = await asyncio.wait(self._tasks, return_when=asyncio.FIRST_COMPLETED)
                  for task in done:
                      try:
                          await task # Raise exception if task failed
                      except Exception as e:
                           logger.error("Task %s completed with error: %s", task.get_name(), e, exc_info=True)
                           # Trigger shutdown on critical task failure?
                           # self.initiate_shutdown()

             else:
                  logger.warning("No main tasks were started.")
                  await self._shutdown_event.wait()


        except asyncio.CancelledError:
            logger.info("Main run loop cancelled.")
        except Exception as e:
            # A2: Parameterized logging
            logger.critical("Fatal error in MainCore run loop: %s", e, exc_info=True)
        finally:
             logger.info("Main run loop concluding. Initiating final shutdown sequence.")
             await self.stop() # Ensure stop is called


    def setup_signal_handlers(self) -> None:
        """Sets up handlers for SIGINT and SIGTERM."""
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                self.loop.add_signal_handler(sig, self.initiate_shutdown, sig)
            logger.debug("Signal handlers for SIGINT and SIGTERM installed.")
        except NotImplementedError:
             # Signal handlers may not be available on all platforms (e.g., Windows without ProactorEventLoop)
             logger.warning("Signal handlers not fully supported on this platform. Use /stop endpoint or Ctrl+C (may be less graceful).")


    def initiate_shutdown(self, sig: Optional[signal.Signals] = None) -> None:
        """Initiates the graceful shutdown sequence, callable from signal handler."""
        if not self.running:
             return # Already shutting down

        if sig:
             # A2: Parameterized logging
             logger.warning("Received signal %s. Initiating graceful shutdown...", sig.name)
        else:
             logger.warning("Shutdown initiated programmatically.")

        # Prevent double execution
        if self._shutdown_event.is_set():
             logger.debug("Shutdown already in progress.")
             return

        self.running = False # Stop main loops
        self._shutdown_event.set() # Signal run() loop and other waiting tasks

        asyncio.run_coroutine_threadsafe(self.stop(), self.loop)


    async def stop(self) -> None:
        """Gracefully stops all components and background tasks."""
        if not self._shutdown_event.is_set(): # Ensure shutdown is signaled
             self.running = False
             self._shutdown_event.set()

        logger.warning("MainCore stopping sequence initiated...")

        # 1. Stop components in reverse order of initialization (or logical order)
        await self.stop_components()

        # 2. Cancel remaining background tasks managed by MainCore
        logger.debug("Cancelling MainCore tasks...")
        for task in list(self._tasks): # Iterate copy
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True) # Wait for cancellation
            logger.debug("MainCore tasks cancelled.")
        self._tasks.clear()


        # 3. Disconnect Web3 provider
        if self.web3 and hasattr(self.web3.provider, 'disconnect'):
            try:
                await self.web3.provider.disconnect()
                logger.info("Web3 provider disconnected.")
            except Exception as e:
                logger.error("Error disconnecting Web3 provider: %s", e, exc_info=True)
        self.web3 = None # Clear reference

        # 4. Log final memory stats if enabled
        if tracemalloc.is_tracing():
            self._log_final_memory_stats()
            tracemalloc.stop()
            logger.info("Tracemalloc stopped.")

        logger.info("MainCore shutdown complete.")


    async def stop_components(self) -> None:
         """Stops initialized components in a safe order."""
         logger.info("Stopping ON1Builder components...")
         # Define stop order (reverse of typical init, dependencies considered)
         stop_order = ["strategynet", "mempoolmonitor", "marketmonitor", "transactioncore", "safetynet", "noncecore", "apiconfig"]
         stopped_successfully = set()

         for name in stop_order:
             component = self.components.get(name)
             if component and hasattr(component, "stop") and callable(component.stop):
                  logger.debug("Stopping component: %s...", name)
                  try:
                       await component.stop()
                       stopped_successfully.add(name)
                       logger.debug("Component %s stopped.", name)
                  except Exception as e:
                       logger.error("Error stopping component '%s': %s", name, e, exc_info=True)
             # Clear component reference after stopping attempt
             self.components[name] = None

         logger.info("Component stopping finished. Successfully stopped: %s", sorted(list(stopped_successfully)))


    def _log_final_memory_stats(self) -> None:
        """Logs final memory statistics compared to the initial snapshot."""
        if not self.memory_snapshot:
            logger.debug("Initial memory snapshot not available, cannot log final stats.")
            return

        logger.debug("Logging final memory allocation statistics...")
        try:
            current_snapshot = tracemalloc.take_snapshot()
            stats = current_snapshot.compare_to(self.memory_snapshot, "lineno")
            total_diff = sum(s.size_diff for s in stats)
            total_current = sum(s.size for s in current_snapshot.statistics("lineno"))

            logger.info(
                "Final Memory Usage: Current=%.2f MiB, Total Diff since start=%.2f MiB",
                total_current / (1024 * 1024),
                total_diff / (1024 * 1024)
            )
            if stats:
                 logger.debug("Top 10 memory differences at shutdown:")
                 for i, stat in enumerate(stats[:10]):
                      logger.debug(f"  {i+1}: {stat}")
            else:
                 logger.debug("No significant memory differences detected at shutdown.")

        except Exception as e:
            # A2: Parameterized logging
            logger.error("Failed to generate final memory stats: %s", e, exc_info=True)

    def _add_task(self, task: asyncio.Task) -> None:
        """Adds a task to the internal set and adds a callback to remove it when done."""
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)


async def main_entry_point() -> None:
    """Asynchronous entry point for running the bot."""
    core: Optional[MainCore] = None
    try:
        # Setup logging early
        setup_logging("ON1Builder", level=logging.INFO) # Set root logger level if needed

        configuration = Configuration()
        # Perform configuration loading/validation (assuming sync or handled in Configuration init)
        # Example: configuration.validate()

        core = MainCore(configuration)
        await core.initialize_components()
        await core.run() # This blocks until shutdown

    except (ValueError, RuntimeError, ConnectionError) as e: # Catch specific init or connection errors
         logger.critical("Critical error during initialization or runtime: %s", e, exc_info=True)
         # Exit code indicates error
         sys.exit(1)
    except asyncio.CancelledError:
        logger.info("Main execution cancelled.")
    except Exception as e:
        logger.critical("Unhandled exception in main_entry_point: %s", e, exc_info=True)
        sys.exit(1) # Generic error exit code
    finally:
        if core and core.running:
             logger.warning("Main entry point finished unexpectedly. Initiating emergency stop.")
             # This path shouldn't normally be reached if run() and stop() work correctly
             await core.stop() # Attempt graceful stop anyway

        logger.info("ON1Builder application finished.")
        # Ensure tracemalloc is stopped if it was started
        if tracemalloc.is_tracing():
            tracemalloc.stop()


if __name__ == "__main__":
    # Configure PYTHONASYNCIODEBUG=1 for development/testing event loop issues (A3)
    # Example: export PYTHONASYNCIODEBUG=1
    if os.getenv("PYTHONASYNCIODEBUG") == "1":
        logging.basicConfig(level=logging.DEBUG) # Show asyncio debug logs
        logger.warning("Asyncio debug mode enabled.")


    # Standard way to run the async main function
    try:
        asyncio.run(main_entry_point())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down...")
        # The signal handler in MainCore should handle this, but this is a fallback.
    except Exception as e:
         # Final catch-all for errors during asyncio.run itself
         logger.critical("Fatal error during asyncio.run: %s", e, exc_info=True)
         sys.exit(1)