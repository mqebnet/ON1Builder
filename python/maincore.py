#========================================================================================================================
# https://github.com/John0n1/0xBuilder

import asyncio
import tracemalloc
import async_timeout
import time

from typing import Any, Dict, List, Optional, Tuple, Union
from web3 import AsyncWeb3
from web3.eth import AsyncEth
from web3.middleware import ExtraDataToPOAMiddleware
from web3 import AsyncIPCProvider, AsyncHTTPProvider, WebSocketProvider
from eth_account import Account
from web3.exceptions import Web3Exception

from abiregistry import ABIRegistry
from apiconfig import APIConfig
from configuration import Configuration
from marketmonitor import MarketMonitor
from mempoolmonitor import MempoolMonitor
from noncecore import NonceCore
from safetynet import SafetyNet
from strategynet import StrategyNet
from transactioncore import TransactionCore

from loggingconfig import setup_logging
import logging
import sys
logger = setup_logging("MainCore", level=logging.INFO)

# Constants

# Custom types
ComponentType = Union[
    APIConfig,
    NonceCore,
    SafetyNet,
    MarketMonitor,
    MempoolMonitor,
    TransactionCore,
    StrategyNet
]

class MainCore:
    """
    Builds and manages the entire MEV bot, initializing all components,
    managing connections, and orchestrating the main execution loop.
    """

    def __init__(self, configuration: "Configuration") -> None:
        """
        Initialize the main application core.

        Args:
            configuration: Configuration object containing settings.
        """
        tracemalloc.start()
        self.memory_snapshot = tracemalloc.take_snapshot()
        self.configuration = configuration
        self.web3: Optional[AsyncWeb3] = None
        self.account: Optional[Account] = None
        self.running: bool = False
        self._shutdown_event: asyncio.Event = asyncio.Event()
        self.components: Dict[str, Optional[ComponentType]] = {
            'apiconfig': None,
            'noncecore': None,
            'safetynet': None,
            'marketmonitor': None,
            'mempoolmonitor': None,
            'transactioncore': None,
            'strategynet': None,
        }
        self._component_health: Dict[str, bool] = {name: False for name in self.components}
        self.WEB3_MAX_RETRIES = 3  
        logger.info("Initializing 0xBuilder...")
        time.sleep(2)  
    async def _initialize_components(self) -> None:
        """Initialize all components in the correct dependency order."""
        try:
            # 1. First initialize configuration and load ABIs
            logger.debug("Loading Configuration...")
            await self._load_configuration()
            logger.info("Configuration initialized ✅")
            await asyncio.sleep(1)

            logger.debug("Initializing Web3...")
            self.web3 = await self._initialize_web3()
            logger.info("Web3 initialized ✅")
            await asyncio.sleep(1)
            if not self.web3:
                raise RuntimeError("Failed to initialize Web3 connection")
            self.configuration.web3 = self.web3 # Set web3 instance in configuration

            self.account = Account.from_key(self.configuration.WALLET_KEY)
            await self._check_account_balance()
            logger.info(f"Account {self.account.address} initialized ✅")
            await asyncio.sleep(1)

            # Initialize ABI Registry and load ABIs (needs to be after web3 for chain-aware ABIs if needed)
            logger.debug("Initializing ABI Registry...")
            abiregistry = ABIRegistry()
            await abiregistry.initialize(self.configuration.BASE_PATH)
            logger.info("ABI Registry initialized ✅")
            await asyncio.sleep(1) 
            # Load and validate ERC20 ABI (needs ABI Registry initialized)
            erc20_abi = abiregistry.get_abi('erc20') # Load directly from registry
            if not erc20_abi:
                raise ValueError("Failed to load ERC20 ABI from Registry")

            # 2. Initialize API config (depends on Configuration)
            logger.debug("Initializing API Config...")
            self.components['apiconfig'] = APIConfig(self.configuration)
            await self.components['apiconfig'].initialize()
            logger.info("API Config initialized ✅")
            await asyncio.sleep(1) 

            # 3. Initialize nonce core (depends on Web3, Configuration)
            logger.debug("Initializing Nonce Core...")
            self.components['noncecore'] = NonceCore(
                self.web3,
                self.account.address,
                self.configuration
            )
            await self.components['noncecore'].initialize()
            logger.info(f"NonceCore initialized with nonce {await self.components['noncecore'].get_nonce()} ✅") # Get nonce after init
            await asyncio.sleep(1) 
            # 4. Initialize safety net (depends on Web3, Configuration, APIConfig, MarketMonitor - Market Monitor is optional here and could be None initially)
            logger.debug("Initializing SafetyNet...")
            self.components['safetynet'] = SafetyNet(
                self.web3,
                self.configuration,
                self.account.address,
                self.account,
                self.components['apiconfig'],
                marketmonitor = self.components.get('marketmonitor')
            )
            await self.components['safetynet'].initialize()
            logger.info("SafetyNet initialized ✅ ")
            await asyncio.sleep(1) 
            # 5. Initialize transaction core with corrected parameters:
            logger.debug("Initializing Transaction Core...")
            self.components['transactioncore'] = TransactionCore(
                self.web3,
                self.account,
                self.configuration.AAVE_FLASHLOAN_ADDRESS, 
                self.configuration.AAVE_POOL_ADDRESS,      
                noncecore=self.components['noncecore'],
                safetynet=self.components['safetynet'],
                configuration=self.configuration
            )
            await self.components['transactioncore'].initialize()
            logger.info("Transaction Core initialized ✅")
            await asyncio.sleep(1) 

            logger.debug("Initializing Market Monitor...")
            self.components['marketmonitor'] = MarketMonitor(
                web3=self.web3,
                configuration=self.configuration,
                apiconfig=self.components['apiconfig'],
                transactioncore=self.components['transactioncore']
            )
            await self.components['marketmonitor'].initialize()
            logger.info("Market Monitor initialized ✅ ")
            await asyncio.sleep(1)


            logger.debug("Initializing Mempool Monitor...")
            self.components['mempoolmonitor'] = MempoolMonitor(
                web3=self.web3,
                safetynet=self.components['safetynet'],
                noncecore=self.components['noncecore'],
                apiconfig=self.components['apiconfig'],
                monitored_tokens=await self.configuration.get_token_addresses(),
                configuration=self.configuration,
                erc20_abi=erc20_abi,
                marketmonitor=self.components['marketmonitor']
            )
            await self.components['mempoolmonitor'].initialize()
            logger.info("Mempool Monitor initialized ✅ ")
            await asyncio.sleep(1) 

            logger.debug("Initializing StrategyNet...")
            self.components['strategynet'] = StrategyNet(
                self.components['transactioncore'],
                self.components['marketmonitor'],
                self.components['safetynet'],
                 self.components['apiconfig']
            )
            await self.components['strategynet'].initialize()
            logger.info("StrategyNet initialized ✅")
            await asyncio.sleep(1) 
            logger.info("All vital components initialized successfully ✅")
            await asyncio.sleep(1) 
        except Exception as e:
            logger.critical(f"Component initialization failed: {e}", exc_info=True)
            raise

    async def initialize(self) -> None:
        """Initialize all components with proper error handling."""
        try:
            before_snapshot = tracemalloc.take_snapshot()
            await self._initialize_components()
            after_snapshot = tracemalloc.take_snapshot()

            # Log memory usage
            top_stats = after_snapshot.compare_to(before_snapshot, 'lineno')
            logger.debug("Memory allocation during initialization:")
            for stat in top_stats[:3]:
                logger.debug(str(stat))

            logger.info("Main Core initialization complete ✅")

        except Exception as e:
            logger.critical(f"Main Core initialization failed: {e}", exc_info=True)
            raise

    async def _load_configuration(self) -> None:
        """Load all configuration elements in the correct order."""
        try:
            # First load the configuration itself
            await self.configuration.load(self.web3) 

        except FileNotFoundError as fnfe:
            logger.critical(f"Configuration file not found: {fnfe}")
            raise
        except ValueError as ve: 
            logger.critical(f"Configuration validation error: {ve}")
            raise
        except Exception as e: 
            logger.critical(f"Failed to load configuration: {e}", exc_info=True)
            raise


    async def _initialize_web3(self) -> Optional[AsyncWeb3]:
        """Initialize Web3 connection with error handling and retries."""
        providers = await self._get_providers()
        if not providers:
            logger.error("No valid endpoints provided!")
            return None

        max_retries = self.configuration.get_config_value("WEB3_MAX_RETRIES", 3)
        retry_delay = self.configuration.get_config_value("WEB3_RETRY_DELAY", 2)

        for provider_name, provider in providers:
             for attempt in range(max_retries):
                try:
                    logger.debug(f"Attempting connection with {provider_name} (attempt {attempt + 1})...")
                    web3 = AsyncWeb3(provider, modules={"eth": (AsyncEth,)})
                    async with async_timeout.timeout(10):
                        if await web3.is_connected():
                            chain_id = await web3.eth.chain_id
                            logger.info(f"Connected to network via {provider_name} (Chain ID: {chain_id})")
                            await self._add_middleware(web3)
                            return web3
                except asyncio.TimeoutError:
                    logger.warning(f"Connection timeout with {provider_name}")
                    continue

                except Web3Exception as w3e:
                    logger.warning(f"{provider_name} connection attempt {attempt + 1} failed (Web3Exception): {w3e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (attempt + 1))
                    continue
                except Exception as e:
                    logger.warning(f"{provider_name} connection attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (attempt + 1))
                    continue
             logger.error(f"All attempts failed for {provider_name}")
        logger.error("Failed to initialize Web3 with any provider")
        return None

    async def _get_providers(self) -> List[Tuple[str, Union[AsyncIPCProvider, AsyncHTTPProvider, WebSocketProvider]]]:
        """Get list of available providers with validation."""
        providers = []

        if self.configuration.HTTP_ENDPOINT:
             try:
                http_provider = AsyncHTTPProvider(self.configuration.HTTP_ENDPOINT)
                await http_provider.make_request('eth_blockNumber', [])
                providers.append(("HTTP Provider", http_provider))
                logger.info("Linked to Ethereum network via HTTP Provider. ✅")
                return providers
             except Exception as e:
                 logger.warning(f"HTTP Provider failed. {e} ❌ - Attempting WebSocket... ")

        if self.configuration.WEBSOCKET_ENDPOINT:
            try:
                ws_provider = WebSocketProvider(self.configuration.WEBSOCKET_ENDPOINT)
                await ws_provider.connect()
                providers.append(("WebSocket Provider", ws_provider))
                logger.info("Linked to Ethereum network via WebSocket Provider. ✅")
                return providers
            except Exception as e:
                logger.warning(f"WebSocket Provider failed. {e} ❌ - Attempting IPC... ")

        if self.configuration.IPC_ENDPOINT:
            try:
                ipc_provider = AsyncIPCProvider(self.configuration.IPC_ENDPOINT)
                await ipc_provider.make_request('eth_blockNumber', [])
                providers.append(("IPC Provider", ipc_provider))
                logger.info("Linked to Ethereum network via IPC Provider. ✅")
                return providers
            except Exception as e:
                 logger.warning(f"IPC Provider failed. {e} ❌ - All providers failed.")

        logger.critical("No more providers are available! ❌")
        return providers

    async def _add_middleware(self, web3: AsyncWeb3) -> None:
        """Middleware based on network."""
        try:
            chain_id = await web3.eth.chain_id
            if (chain_id in {99, 100, 77, 7766, 56}):  # POA networks
                web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
                logger.info(f"Injected POA middleware.")
            elif chain_id in {1, 3, 4, 5, 42, 420}:  # ETH networks
                logger.info(f"ETH network detected (no middleware injected).")
                pass
            else:
                logger.warning(f"Unknown network (Chain ID: {chain_id}); no middleware injected.")
        except Exception as e:
            logger.error(f"Middleware configuration failed: {e}", exc_info=True)
            raise

    async def _check_account_balance(self) -> None:
        """Check account balance """
        if not self.account or not self.web3:
            raise ValueError("Account or Web3 not initialized before balance check")

        try:
            min_balance = float(self.configuration.get_config_value("MIN_ETH_BALANCE", 0.000001))
            balance = await self.web3.eth.get_balance(self.account.address)
            balance_eth = float(self.web3.from_wei(balance, 'ether'))

            logger.info(f"Account {self.account.address[:8]}...{self.account.address[-6:]}")
            logger.info(f"Balance: {balance_eth:.4f} ETH")

            if balance_eth < min_balance:
                logger.warning(f"Critical: Low account balance ({balance_eth:.4f} ETH), below threshold of {min_balance} ETH")


        except Exception as e:
            logger.error(f"Balance check failed: {e}", exc_info=True)
            raise


    async def _load_abi(self, abi_path: str, abiregistry: "ABIRegistry") -> List[Dict[str, Any]]:
        """Load contract ABI from a file with better path handling."""
        try:
            abi = abiregistry.get_abi('erc20')
            if not abi:
                raise ValueError("Failed to load ERC20 ABI using ABI Registry")
            return abi
        except Exception as e:
            logger.error(f"Error loading ERC20 ABI from ABI Registry: {e}", exc_info=True)
            raise

    async def run(self) -> None:
        """ main execution loop with component health monitoring."""
        logger.info("Starting 0xBuilder main loop...")
        self.running = True
        initial_snapshot = tracemalloc.take_snapshot()

        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self.components['mempoolmonitor'].start_monitoring())
                tg.create_task(self._process_profitable_transactions())
                tg.create_task(self._monitor_memory(initial_snapshot))
                tg.create_task(self._check_component_health())
                logger.info("All tasks started in TaskGroup. Monitoring has initiated.")

        except* asyncio.CancelledError:
            logger.info("Graceful shutdown initiated - tasks cancelled.")
        except* Exception as e:
            logger.error(f"Fatal error in run loop: {e}", exc_info=True)
        finally:
            await self.stop() 

    async def _monitor_memory(self, initial_snapshot: tracemalloc.Snapshot) -> None:
        """ memory monitoring with leak detection."""
        last_snapshot = initial_snapshot
        while self.running:
            try:
                current_snapshot = tracemalloc.take_snapshot()
                diff_stats = current_snapshot.compare_to(last_snapshot, 'lineno')

                significant_changes = [stat for stat in diff_stats if abs(stat.size_diff) > 1024 * 1024]  # > 1MB
                if significant_changes:
                    logger.warning("Significant memory changes detected:")
                    for stat in significant_changes[:3]:
                        logger.warning(str(stat))

                last_snapshot = current_snapshot
                await asyncio.sleep(self.configuration.MEMORY_CHECK_INTERVAL) 
            except asyncio.CancelledError:
                logger.info("Memory monitoring task cancelled.")
                break
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}", exc_info=True)

    async def _check_component_health(self) -> None:
        """Periodic health check of all components."""
        while self.running:
            try:
                for name, component in self.components.items():
                    if component and hasattr(component, 'is_healthy'):
                        self._component_health[name] = await component.is_healthy()
                    else:
                        self._component_health[name] = component is not None 

                if not all(self._component_health.values()):
                    unhealthy = [name for name, healthy in self._component_health.items() if not healthy]
                    logger.warning(f"Unhealthy components detected: {unhealthy}")

                await asyncio.sleep(self.configuration.COMPONENT_HEALTH_CHECK_INTERVAL) 
            except asyncio.CancelledError:
                logger.info("Component health check task cancelled.")
                break
            except Exception as e:
                logger.error(f"Health check error: {e}", exc_info=True)
                await asyncio.sleep(5) 

    async def stop(self) -> None:
        """graceful shutdown."""
        if not self.running:
            return

        logger.warning("Initiating graceful shutdown...")
        self.running = False
        self._shutdown_event.set()

        try:
            stop_tasks = []
            for name, component in self.components.items():
                if component and hasattr(component, 'stop'):
                    task = asyncio.create_task(self._stop_component(name, component))
                    stop_tasks.append(task)

            if stop_tasks:
                done, pending = await asyncio.wait(stop_tasks, timeout=10)
                if pending:
                    logger.warning(f"Forcing {len(pending)} component(s) to stop due to timeout.")
                    for task in pending:
                        task.cancel()

            if self.web3 and hasattr(self.web3.provider, 'disconnect'):
                await self.web3.provider.disconnect()
                logger.info("Web3 provider disconnected.")
            self._log_final_memory_stats()

        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)
        finally:
            tracemalloc.stop()
            logger.info("Shutdown complete")

    def _log_final_memory_stats(self) -> None:
        """Log final memory statistics."""
        try:
            final_snapshot = tracemalloc.take_snapshot()
            top_stats = final_snapshot.compare_to(self.memory_snapshot, 'lineno')

            logger.debug("Final memory allocation changes:")
            for stat in top_stats[:5]:
                logger.debug(str(stat))
        except Exception as e:
            logger.error(f"Error logging final memory stats: {e}", exc_info=True)

    async def _stop_component(self, name: str, component: Any) -> None:
        """Stop a single component with error handling."""
        try:
            await component.stop()
            logger.debug(f"Stopped component: {name}")
        except Exception as e:
            logger.error(f"Error stopping component {name}: {e}", exc_info=True)

    async def _process_profitable_transactions(self) -> None:
        """Process profitable transactions from the queue."""
        strategy = self.components['strategynet']
        monitor = self.components['mempoolmonitor']

        while self.running:
            try:
                try:
                    tx = await asyncio.wait_for(monitor.profitable_transactions.get(), timeout=self.configuration.PROFITABLE_TX_PROCESS_TIMEOUT) 
                    tx_hash = tx.get('tx_hash', 'Unknown')
                    strategy_type = tx.get('strategy_type', 'Unknown')
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    logger.info("Profitable transaction processing task cancelled.")
                    break

                logger.debug(f"Processing transaction {tx_hash[:8]}... with strategy type {strategy_type}")
                success = await strategy.execute_best_strategy(tx, strategy_type)

                if success:
                    logger.debug(f"Strategy execution successful for tx: {tx_hash[:8]}...")
                else:
                    logger.warning(f"Strategy execution failed for tx: {tx_hash[:8]}...")


                monitor.profitable_transactions.task_done()

            except Exception as e:
                logger.error(f"Error processing transaction: {e}", exc_info=True)

    
    async def emergency_shutdown(self) -> None:
        """Emergency shutdown with forced cancellation of tasks."""
        logger.warning("Emergency shutdown initiated...")
        self.running = False
        self._shutdown_event.set()
        try:

            for task in asyncio.all_tasks():
                task.cancel()

            await asyncio.gather(*asyncio.all_tasks())

            if self.web3 and hasattr(self.web3.provider, 'disconnect'):
                await self.web3.provider.disconnect()
                logger.info("Web3 provider disconnected.")
                
            self._log_final_memory_stats()

        except Exception as e:
            try:
                sys.exit(1)
            except Exception as se:
                logger.error(f"Error during emergency shutdown: {se}", exc_info=True)
        finally:
            tracemalloc.stop()
            logger.info("Emergency shutdown complete")
