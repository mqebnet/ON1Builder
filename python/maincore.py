import asyncio
import tracemalloc
import async_timeout
import time
import os
import sys
import signal
from typing import Any, Dict, List, Optional

from web3 import AsyncWeb3
from web3.eth import AsyncEth
from web3.middleware import ExtraDataToPOAMiddleware
from web3 import AsyncIPCProvider, AsyncHTTPProvider, WebSocketProvider
from eth_account import Account

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

logger = setup_logging("MainCore", level=logging.DEBUG)

class MainCore:
    """
    Orchestrates all components (configuration, blockchain connection, account,
    noncecore, safetynet, transactioncore, marketmonitor, mempoolmonitor, strategynet)
    and manages main loop, memory monitoring, and graceful shutdown.
    """
    def __init__(self, configuration: Configuration) -> None:
        tracemalloc.start()
        self.memory_snapshot = tracemalloc.take_snapshot()
        self.configuration = configuration
        self.web3: Optional[AsyncWeb3] = None
        self.account: Optional[Account] = None
        self.running: bool = False
        self._shutdown_event: asyncio.Event = asyncio.Event()
        self.components: Dict[str, Optional[Any]] = {
            "apiconfig": None,
            "noncecore": None,
            "safetynet": None,
            "marketmonitor": None,
            "mempoolmonitor": None,
            "transactioncore": None,
            "strategynet": None,
        }
        self._component_health: Dict[str, bool] = {name: False for name in self.components}
        self.WEB3_MAX_RETRIES = self.configuration.get_config_value("WEB3_MAX_RETRIES", 3)
        logger.info("Initializing ON1Builder...")
        self.loop = asyncio.get_event_loop()

    async def _load_configuration(self) -> None:
        try:
            await self.configuration.load()
            logger.info("Configuration successfully loaded.")
        except Exception as e:
            logger.critical(f"Configuration load failed: {e}", exc_info=True)
            raise

    async def _initialize_web3(self) -> Optional[AsyncWeb3]:
        providers = await self._get_providers()
        if not providers:
            logger.error("No valid Web3 provider endpoints available!")
            return None
        max_retries = self.configuration.get_config_value("WEB3_MAX_RETRIES", 3)
        retry_delay = self.configuration.get_config_value("WEB3_RETRY_DELAY", 2)
        for provider_name, provider in providers:
            for attempt in range(max_retries):
                try:
                    logger.debug(f"Connecting via {provider_name}, attempt {attempt+1}.")
                    web3 = AsyncWeb3(provider, modules={"eth": (AsyncEth,)})
                    async with async_timeout.timeout(10):
                        if await web3.is_connected():
                            chain_id = await web3.eth.chain_id
                            logger.info(f"Connected via {provider_name} (Chain ID: {chain_id}).")
                            await self._add_middleware(web3)
                            return web3
                except Exception as e:
                    logger.warning(f"{provider_name} connection attempt {attempt+1} failed: {e}")
                    await asyncio.sleep(retry_delay * (attempt+1))
            logger.error(f"All attempts failed for {provider_name}.")
        return None

    async def _get_providers(self) -> List[tuple]:
        providers = []
        if self.configuration.HTTP_ENDPOINT:
            http_provider = AsyncHTTPProvider(self.configuration.HTTP_ENDPOINT)
            try:
                await http_provider.make_request("eth_blockNumber", [])
                providers.append(("HTTP Provider", http_provider))
                logger.info("Connected via HTTP Provider.")
                return providers
            except Exception as e:
                logger.warning(f"HTTP Provider error: {e}")
        if self.configuration.WEBSOCKET_ENDPOINT:
            try:
                ws_provider = WebSocketProvider(self.configuration.WEBSOCKET_ENDPOINT)
                await ws_provider.connect()
                providers.append(("WebSocket Provider", ws_provider))
                logger.info("Connected via WebSocket Provider.")
                return providers
            except Exception as e:
                logger.warning(f"WebSocket Provider error: {e}; trying IPC.")
        if self.configuration.IPC_ENDPOINT:
            try:
                ipc_provider = AsyncIPCProvider(self.configuration.IPC_ENDPOINT)
                await ipc_provider.make_request("eth_blockNumber", [])
                providers.append(("IPC Provider", ipc_provider))
                logger.info("Connected via IPC Provider.")
                return providers
            except Exception as e:
                logger.warning(f"IPC Provider error: {e}")
        logger.critical("No provider available.")
        return providers

    async def _add_middleware(self, web3: AsyncWeb3) -> None:
        try:
            chain_id = await web3.eth.chain_id
            if chain_id in {99, 100, 77, 7766, 56}:
                web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
                logger.info("Injected POA middleware.")
            else:
                logger.info("No middleware needed for this network.")
        except Exception as e:
            logger.error(f"Middleware error: {e}", exc_info=True)
            raise

    async def _check_account_balance(self) -> None:
        if not self.account or not self.web3:
            raise ValueError("Account or Web3 not initialized.")
        try:
            min_balance = float(self.configuration.get_config_value("MIN_BALANCE", 0.000001))
            balance = await self.web3.eth.get_balance(self.account.address)
            balance_eth = float(self.web3.from_wei(balance, "ether"))
            logger.info(f"Account {self.account.address[:8]}... Balance: {balance_eth:.4f} ETH")
            if balance_eth < min_balance:
                logger.warning(f"Low balance: {balance_eth:.4f} ETH below threshold {min_balance} ETH")
        except Exception as e:
            logger.error(f"Error checking account balance: {e}", exc_info=True)
            raise

    async def initialize_components(self) -> None:
        """
        Initialize all components in order: configuration, web3, account, ABI registry,
        API configuration, noncecore, safetynet, transactioncore, marketmonitor, mempoolmonitor, strategynet.
        """
        try:
            await self._load_configuration()
            self.web3 = await self._initialize_web3()
            if not self.web3:
                raise RuntimeError("Web3 initialization failed.")
            self.account = Account.from_key(self.configuration.WALLET_KEY)
            await self._check_account_balance()

            # Initialize API Configuration
            self.components["apiconfig"] = APIConfig(self.configuration)
            await self.components["apiconfig"].initialize()

            # Initialize NonceCore
            self.components["noncecore"] = NonceCore(self.web3, self.account.address, self.configuration)
            await self.components["noncecore"].initialize()

            # Initialize SafetyNet
            self.components["safetynet"] = SafetyNet(self.web3, self.configuration, self.account.address, self.account, self.components["apiconfig"], self.components.get("marketmonitor"))
            await self.components["safetynet"].initialize()

            # Initialize MarketMonitor
            self.components["marketmonitor"] = MarketMonitor(
                web3=self.web3,
                configuration=self.configuration,
                apiconfig=self.components["apiconfig"],
                transactioncore=self.components["transactioncore"]
            )
            await self.components["marketmonitor"].initialize()

            # Initialize TransactionCore
            self.components["transactioncore"] = TransactionCore(
                self.web3,
                self.account,
                self.configuration,
                noncecore=self.components["noncecore"],
                safetynet=self.components["safetynet"],
                apiconfig=self.components["apiconfig"],
                marketmonitor=self.components["marketmonitor"]
            )
            await self.components["transactioncore"].initialize()

            # Initialize MempoolMonitor
            token_addresses = await self.configuration.get_token_addresses()
            self.components["mempoolmonitor"] = MempoolMonitor(
                web3=self.web3,
                safetynet=self.components["safetynet"],
                noncecore=self.components["noncecore"],
                apiconfig=self.components["apiconfig"],
                monitored_tokens=token_addresses,
                configuration=self.configuration,
                marketmonitor=self.components["marketmonitor"]
            )
            await self.components["mempoolmonitor"].initialize()

            # Initialize StrategyNet
            self.components["strategynet"] = StrategyNet(
                self.components["transactioncore"],
                self.components["marketmonitor"],
                self.components["safetynet"],
                self.components["apiconfig"]
            )
            await self.components["strategynet"].initialize()

            logger.info("All components initialized successfully.")
        except Exception as e:
            logger.error(f"Component initialization failed: {e}", exc_info=True)
            raise

    async def _check_component_health(self) -> None:
        while self.running:
            try:
                for name, component in self.components.items():
                    if component and hasattr(component, "is_healthy"):
                        self._component_health[name] = await component.is_healthy()
                    else:
                        self._component_health[name] = component is not None
                if not all(self._component_health.values()):
                    unhealthy = [name for name, healthy in self._component_health.items() if not healthy]
                    logger.warning(f"Unhealthy components: {unhealthy}")
                await asyncio.sleep(self.configuration.COMPONENT_HEALTH_CHECK_INTERVAL)
            except asyncio.CancelledError:
                logger.info("Component health check cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in component health check: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def _monitor_memory(self, initial_snapshot: Any) -> None:
        last_snapshot = initial_snapshot
        while self.running:
            try:
                current_snapshot = tracemalloc.take_snapshot()
                diff_stats = current_snapshot.compare_to(last_snapshot, "lineno")
                significant = [stat for stat in diff_stats if abs(stat.size_diff) > 1024 * 1024]
                if significant:
                    logger.warning("Significant memory changes detected:")
                    for stat in significant[:3]:
                        logger.warning(str(stat))
                last_snapshot = current_snapshot
                await asyncio.sleep(self.configuration.MEMORY_CHECK_INTERVAL)
            except asyncio.CancelledError:
                logger.info("Memory monitoring cancelled.")
                break
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}", exc_info=True)

    async def _process_profitable_transactions(self) -> None:
        while self.running:
            try:
                tx = await asyncio.wait_for(
                    self.components["mempoolmonitor"].profitable_transactions.get(),
                    timeout=self.configuration.PROFITABLE_TX_PROCESS_TIMEOUT
                )
                strategy_type = tx.get("strategy_type", "unknown")
                success = await self.components["strategynet"].execute_best_strategy(tx, strategy_type)
                if success:
                    logger.debug(f"Processed profitable tx: {tx.get('tx_hash','unknown')[:8]}...")
                self.components["mempoolmonitor"].profitable_transactions.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing profitable transaction: {e}", exc_info=True)

    async def run(self) -> None:
        logger.info("Starting MainCore run loop...")
        self.running = True
        initial_snapshot = tracemalloc.take_snapshot()
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self.components["mempoolmonitor"].start_monitoring())
                tg.create_task(self._process_profitable_transactions())
                tg.create_task(self._monitor_memory(initial_snapshot))
                tg.create_task(self._check_component_health())
                logger.info("All main tasks started.")
        except asyncio.CancelledError:
            logger.info("Main loop cancelled.")
        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}", exc_info=True)
        finally:
            await self.stop()

    async def stop(self) -> None:
        if not self.running:
            return
        logger.warning("Initiating graceful shutdown...")
        self.running = False
        self._shutdown_event.set()
        stop_tasks = []
        for name, comp in self.components.items():
            if comp and hasattr(comp, "stop"):
                stop_tasks.append(asyncio.create_task(comp.stop()))
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
        if self.web3 and hasattr(self.web3.provider, "disconnect"):
            await self.web3.provider.disconnect()
            logger.info("Web3 provider disconnected.")
        self._log_final_memory_stats()
        tracemalloc.stop()
        logger.info("MainCore shutdown complete.")

    def _log_final_memory_stats(self) -> None:
        try:
            final_snapshot = tracemalloc.take_snapshot()
            stats = final_snapshot.compare_to(self.memory_snapshot, "lineno")
            logger.debug("Final memory allocation changes:")
            for stat in stats[:5]:
                logger.debug(str(stat))
        except Exception as e:
            logger.error(f"Error logging memory stats: {e}", exc_info=True)

    async def emergency_shutdown(self) -> None:
        logger.warning("Emergency shutdown initiated...")
        self.running = False
        self._shutdown_event.set()
        for task in asyncio.all_tasks():
            task.cancel()
        await asyncio.gather(*asyncio.all_tasks(), return_exceptions=True)
        if self.web3 and hasattr(self.web3.provider, "disconnect"):
            await self.web3.provider.disconnect()
            logger.info("Web3 provider disconnected.")
        self._log_final_memory_stats()
        tracemalloc.stop()
        logger.info("Emergency shutdown complete.")
        sys.exit(0)

async def main() -> None:
    await run_bot()

async def run_bot() -> None:
    loop = asyncio.get_running_loop()
    def shutdown_handler() -> None:
        logger.debug("Shutdown signal received; cancelling tasks...")
        for task in asyncio.all_tasks(loop):
            task.cancel()
    try:
        tracemalloc.start()
        await asyncio.sleep(3)
        configuration = Configuration()
        core = MainCore(configuration)
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, shutdown_handler)
        await core.initialize_components()
        await core.run()
    except Exception as e:
        logger.critical(f"Fatal error in run_bot: {e}")
    finally:
        tracemalloc.stop()
        logger.debug("MainCore shutdown complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        snapshot = tracemalloc.take_snapshot()
        logger.critical(f"Program terminated with error: {e}")
        stats = snapshot.statistics("lineno")
        for stat in stats[:10]:
            logger.debug(str(stat))
        logger.debug("Program terminated.")
