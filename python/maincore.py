#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MainCore Module

This is the heart of the 0xBuilder project. It builds and manages the entire MEV bot,
initializing all components (configuration, Web3, account, ABI registry, APIConfig, NonceCore,
SafetyNet, TransactionCore, MarketMonitor, MempoolMonitor, and StrategyNet), orchestrating the
main execution loop, monitoring component health and memory usage, and handling graceful shutdown.
"""

import asyncio
import tracemalloc
import async_timeout
import time
import sys
import signal

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

logger = setup_logging("MainCore", level=logging.INFO)


# Custom type for components (for type hints)
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
    MainCore orchestrates the initialization and operation of all components in the MEV bot.
    It initializes configuration, Web3, account, and all supporting modules, and then enters
    a main execution loop that monitors component health, memory usage, and processes profitable transactions.
    """

    def __init__(self, configuration: Configuration) -> None:
        """
        Initialize the main application core.

        Args:
            configuration (Configuration): The configuration object containing settings.
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
        self.WEB3_MAX_RETRIES = self.configuration.get_config_value("WEB3_MAX_RETRIES", 3)
        logger.info("Initializing 0xBuilder...")
        time.sleep(2)  # Allow some time for the environment to stabilize

    async def _initialize_components(self) -> None:
        """
        Initialize all vital components in the correct order.
        """
        try:
            # 1. Load configuration
            logger.debug("Loading Configuration...")
            await self._load_configuration()
            logger.info("Configuration initialized ✅")
            await asyncio.sleep(1)

            # 2. Initialize Web3 connection
            logger.debug("Initializing Web3...")
            self.web3 = await self._initialize_web3()
            if not self.web3:
                raise RuntimeError("Failed to initialize Web3 connection")
            logger.info("Web3 initialized ✅")
            await asyncio.sleep(1)

            # 3. Initialize account and check balance
            self.account = Account.from_key(self.configuration.WALLET_KEY)
            await self._check_account_balance()
            logger.info(f"Account {self.account.address} initialized ✅")
            await asyncio.sleep(1)

            # 4. Initialize ABI Registry and load ABIs
            logger.debug("Initializing ABI Registry...")
            abiregistry = ABIRegistry()
            await abiregistry.initialize(self.configuration.BASE_PATH)
            logger.info("ABI Registry initialized ✅")
            await asyncio.sleep(1)
            erc20_abi = abiregistry.get_abi('erc20')
            if not erc20_abi:
                raise ValueError("Failed to load ERC20 ABI from Registry")

            # 5. Initialize APIConfig
            logger.debug("Initializing API Config...")
            self.components['apiconfig'] = APIConfig(self.configuration)
            await self.components['apiconfig'].initialize()
            logger.info("API Config initialized ✅")
            await asyncio.sleep(1)

            # 6. Initialize NonceCore
            logger.debug("Initializing Nonce Core...")
            self.components['noncecore'] = NonceCore(
                self.web3,
                self.account.address,
                self.configuration
            )
            await self.components['noncecore'].initialize()
            nonce = await self.components['noncecore'].get_nonce()
            logger.info(f"NonceCore initialized with nonce {nonce} ✅")
            await asyncio.sleep(1)

            # 7. Initialize SafetyNet
            logger.debug("Initializing SafetyNet...")
            self.components['safetynet'] = SafetyNet(
                self.web3,
                self.configuration,
                self.account.address,
                self.account,
                self.components['apiconfig'],
                marketmonitor=self.components.get('marketmonitor')
            )
            await self.components['safetynet'].initialize()
            logger.info("SafetyNet initialized ✅")
            await asyncio.sleep(1)

            # 8. Initialize TransactionCore
            logger.debug("Initializing Transaction Core...")
            self.components['transactioncore'] = TransactionCore(
                self.web3,
                self.account,
                self.configuration,
                noncecore=self.components['noncecore'],
                safetynet=self.components['safetynet'],
            )
            await self.components['transactioncore'].initialize()
            logger.info("Transaction Core initialized ✅")
            await asyncio.sleep(1)

            # 9. Initialize MarketMonitor
            logger.debug("Initializing Market Monitor...")
            self.components['marketmonitor'] = MarketMonitor(
                web3=self.web3,
                configuration=self.configuration,
                apiconfig=self.components['apiconfig'],
                transactioncore=self.components['transactioncore']
            )
            await self.components['marketmonitor'].initialize()
            logger.info("Market Monitor initialized ✅")
            await asyncio.sleep(1)

            # 10. Initialize MempoolMonitor
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
            logger.info("Mempool Monitor initialized ✅")
            await asyncio.sleep(1)

            # 11. Initialize StrategyNet
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
            logger.error(f"Error initializing components: {e}", exc_info=True)
            raise

    async def _load_configuration(self) -> None:
        """
        Load the configuration by invoking its load() method.
        """
        try:
            await self.configuration.load(self.web3)
        except Exception as e:
            logger.critical(f"Failed to load configuration: {e}", exc_info=True)
            raise

    async def _initialize_web3(self) -> Optional[AsyncWeb3]:
        """
        Initialize a Web3 connection using available providers with retries.
        """
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
                except async_timeout.TimeoutError:
                    logger.warning(f"Connection timeout with {provider_name}")
                    await self._cleanup_provider(provider)
                except Exception as e:
                    logger.warning(f"{provider_name} connection attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (attempt + 1))
                    await self._cleanup_provider(provider)
            logger.error(f"All attempts failed for {provider_name}")
        logger.error("Failed to initialize Web3 with any provider")
        return None

    async def _get_providers(self) -> List[Tuple[str, Union[AsyncIPCProvider, AsyncHTTPProvider, WebSocketProvider]]]:
        """
        Retrieve a list of available Web3 providers based on the configured endpoints.
        """
        providers = []

        if self.configuration.HTTP_ENDPOINT:
            http_provider = AsyncHTTPProvider(self.configuration.HTTP_ENDPOINT)
            try:
                await http_provider.make_request('eth_blockNumber', [])
                providers.append(("HTTP Provider", http_provider))
                logger.info("Linked to Ethereum network via HTTP Provider. ✅")
                return providers
            except Exception as e:
                logger.warning(f"HTTP Provider error: {e}")

        if self.configuration.WEBSOCKET_ENDPOINT:
            try:
                ws_provider = WebSocketProvider(self.configuration.WEBSOCKET_ENDPOINT)
                await ws_provider.connect()
                providers.append(("WebSocket Provider", ws_provider))
                logger.info("Linked to Ethereum network via WebSocket Provider. ✅")
                return providers
            except Exception as e:
                logger.warning(f"WebSocket Provider failed: {e} - Attempting IPC...")

        if self.configuration.IPC_ENDPOINT:
            try:
                ipc_provider = AsyncIPCProvider(self.configuration.IPC_ENDPOINT)
                await ipc_provider.make_request('eth_blockNumber', [])
                providers.append(("IPC Provider", ipc_provider))
                logger.info("Linked to Ethereum network via IPC Provider. ✅")
                return providers
            except Exception as e:
                logger.warning(f"IPC Provider failed: {e} - All providers failed.")

        logger.critical("No more providers are available! ❌")
        return providers

    async def _add_middleware(self, web3: AsyncWeb3) -> None:
        """
        Add network-specific middleware to the Web3 instance.
        """
        try:
            chain_id = await web3.eth.chain_id
            if chain_id in {99, 100, 77, 7766, 56}:  # POA networks
                web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
                logger.info("Injected POA middleware.")
            elif chain_id in {1, 3, 4, 5, 42, 420}:  # Ethereum networks
                logger.info("Ethereum network detected (no middleware injected).")
            else:
                logger.warning(f"Unknown network (Chain ID: {chain_id}); no middleware injected.")
        except Exception as e:
            logger.error(f"Middleware configuration failed: {e}", exc_info=True)
            raise

    async def _check_account_balance(self) -> None:
        """
        Check the account balance and log a warning if it is below the configured minimum.
        """
        if not self.account or not self.web3:
            raise ValueError("Account or Web3 not initialized before balance check")
        try:
            min_balance = float(self.configuration.get_config_value("MIN_BALANCE", 0.000001))
            balance = await self.web3.eth.get_balance(self.account.address)
            balance_eth = float(self.web3.from_wei(balance, 'ether'))
            logger.info(f"Account {self.account.address[:8]}...{self.account.address[-6:]}")
            logger.info(f"Balance: {balance_eth:.4f} ETH")
            if balance_eth < min_balance:
                logger.warning(f"Critical: Low account balance ({balance_eth:.4f} ETH), below threshold of {min_balance} ETH")
        except Exception as e:
            logger.error(f"Balance check failed: {e}", exc_info=True)
            raise

    async def run(self) -> None:
        """
        Main execution loop that starts vital background tasks such as mempool monitoring,
        processing profitable transactions, memory monitoring, and component health checks.
        """
        logger.info("Starting 0xBuilder main loop...")
        self.running = True
        initial_snapshot = tracemalloc.take_snapshot()

        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self.components['mempoolmonitor'].start_monitoring())
                tg.create_task(self._process_profitable_transactions())
                tg.create_task(self._monitor_memory(initial_snapshot))
                tg.create_task(self._check_component_health())
                logger.info("All tasks started. Monitoring has initiated.")
        except* asyncio.CancelledError:
            logger.info("Graceful shutdown initiated - tasks cancelled.")
        except* Exception as e:
            logger.error(f"Fatal error in run loop: {e}", exc_info=True)
        finally:
            await self.stop()

    async def _monitor_memory(self, initial_snapshot: tracemalloc.Snapshot) -> None:
        """
        Monitor memory usage periodically and log significant changes.
        """
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
        """
        Periodically check the health of all components.
        """
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
        """
        Perform a graceful shutdown by stopping all components and logging final memory statistics.
        """
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
        """
        Log final memory allocation statistics compared to the initial snapshot.
        """
        try:
            final_snapshot = tracemalloc.take_snapshot()
            top_stats = final_snapshot.compare_to(self.memory_snapshot, 'lineno')
            logger.debug("Final memory allocation changes:")
            for stat in top_stats[:5]:
                logger.debug(str(stat))
        except Exception as e:
            logger.error(f"Error logging final memory stats: {e}", exc_info=True)

    async def _stop_component(self, name: str, component: Any) -> None:
        """
        Stop a single component and log its shutdown status.
        """
        try:
            await component.stop()
            logger.debug(f"Stopped component: {name}")
        except Exception as e:
            logger.error(f"Error stopping component {name}: {e}", exc_info=True)

    async def _process_profitable_transactions(self) -> None:
        """
        Continuously process profitable transactions from the mempool monitor queue by invoking the best strategy.
        """
        strategy = self.components['strategynet']
        monitor = self.components['mempoolmonitor']

        while self.running:
            try:
                try:
                    tx = await asyncio.wait_for(
                        monitor.profitable_transactions.get(),
                        timeout=self.configuration.PROFITABLE_TX_PROCESS_TIMEOUT
                    )
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
        """
        Perform an emergency shutdown by canceling all tasks and disconnecting the Web3 provider.
        """
        logger.warning("Emergency shutdown initiated...")
        self.running = False
        self._shutdown_event.set()
        try:
            for task in asyncio.all_tasks():
                task.cancel()
            await asyncio.gather(*asyncio.all_tasks(), return_exceptions=True)
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
            sys.exit(0)

async def main() -> None:
    """Main entry point."""
    await run_bot()

async def run_bot() -> None:
    """Run the bot with graceful shutdown handling."""
    loop = asyncio.get_running_loop()

    def shutdown_handler() -> None:
        logger.debug("Received shutdown signal. Stopping the bot...")
        for task in asyncio.all_tasks(loop):
            task.cancel()

    try:
        tracemalloc.start()
        await asyncio.sleep(3)
       
        configuration = Configuration()
        core = MainCore(configuration)
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, shutdown_handler)

        await core.initialize()
        await core.run()

    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        if tracemalloc.is_tracing():
            snapshot = tracemalloc.take_snapshot()
            logger.debug("Top 10 memory allocations:")
            for stat in snapshot.statistics('lineno')[:10]:
                logger.debug(str(stat))
    finally:
        tracemalloc.stop()
        if tracemalloc.is_tracing():
            snapshot = tracemalloc.take_snapshot()
            logger.debug("Final memory allocations at shutdown:")
            for stat in snapshot.statistics('lineno')[:10]:
                logger.debug(str(stat))
        logger.debug("0xBuilder shutdown complete")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        snapshot = tracemalloc.take_snapshot()
        logger.critical(f"Program terminated with an error: {e}")
        logger.debug("Top 10 memory allocations at error:")
        top_stats = snapshot.statistics('lineno')
        for stat in top_stats[:10]:
            logger.debug(str(stat))
