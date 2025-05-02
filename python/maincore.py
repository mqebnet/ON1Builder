# maincore.py
import asyncio
import tracemalloc
import async_timeout
import sys
import signal
from typing import Any, Dict, List, Optional

from web3 import AsyncWeb3
from web3.eth import AsyncEth
from web3.middleware import ExtraDataToPOAMiddleware
from web3 import AsyncIPCProvider, AsyncHTTPProvider, WebSocketProvider
from eth_account import Account

from apiconfig import APIConfig
from configuration import Configuration
from marketmonitor import MarketMonitor
from mempoolmonitor import MempoolMonitor
from noncecore import NonceCore
from safetynet import SafetyNet
from strategynet import StrategyNet
from transactioncore import TransactionCore
from loggingconfig import setup_logging

logger = setup_logging("Main_Core", level="DEBUG")  

class MainCore:
    def __init__(self, configuration: Configuration) -> None:
        tracemalloc.start()
        self.memory_snapshot = tracemalloc.take_snapshot()
        self.configuration = configuration
        self.web3: Optional[AsyncWeb3] = None
        self.account: Optional[Account] = None
        self.running = False
        self._shutdown_event = asyncio.Event()
        self.components: Dict[str, Any] = {}
        self.component_health: Dict[str, bool] = {}
        self.loop = asyncio.get_event_loop()

    async def _load_configuration(self) -> None:
        await self.configuration.load()

    async def _initialize_web3(self) -> Optional[AsyncWeb3]:
        providers = await self._get_providers()
        if not providers:
            logger.error("No Web3 providers available.")
            return None
        for name, provider in providers:
            for attempt in range(self.configuration.get_config_value("WEB3_MAX_RETRIES", 3)):
                try:
                    logger.info(f"Attempting to connect to Web3 provider: {name} (Attempt {attempt + 1})")
                    web3 = AsyncWeb3(provider, modules={"eth": (AsyncEth,)})
                    async with async_timeout.timeout(10):
                        if await web3.is_connected():
                            chain_id = await web3.eth.chain_id
                            logger.info(f"Connected to Web3 provider: {name} with chain ID {chain_id}")
                            await self._add_middleware(web3, chain_id)
                            return web3
                except Exception as e:
                    logger.error(f"Failed to connect to Web3 provider {name}: {e}")
                    await asyncio.sleep(self.configuration.get_config_value("WEB3_RETRY_DELAY", 2))
        logger.error("All Web3 connection attempts failed.")
        return None

    async def _get_providers(self) -> List[tuple]:
        providers = []
        http = self.configuration.HTTP_ENDPOINT
        if http:
            providers.append(("http", AsyncHTTPProvider(http)))
        ws = self.configuration.WEBSOCKET_ENDPOINT
        if ws:
            api_key = self.configuration.get_config_value("GOOGLE_BLOCKCHAIN_API_KEY")
            ws_with_key = f"{ws}?key={api_key}"
            providers.append(("ws", WebSocketProvider(ws_with_key)))
        ipc = self.configuration.IPC_ENDPOINT
        if ipc:
            providers.append(("ipc", AsyncIPCProvider(ipc)))
        return providers

    async def _add_middleware(self, web3: AsyncWeb3, chain_id: int) -> None:
        if chain_id in {99, 100, 77, 7766, 56, 11155111}:
            web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

    async def _check_account_balance(self) -> None:
        balance = await self.web3.eth.get_balance(self.account.address)
        eth = float(self.web3.from_wei(balance, "ether"))
        if eth < self.configuration.MIN_BALANCE:
            logger.warning(f"Low balance {eth}")

    async def initialize_components(self) -> None:
        await self._load_configuration()
        self.web3 = await self._initialize_web3()
        if not self.web3:
            raise RuntimeError("Web3 init failed")
        self.account = Account.from_key(self.configuration.WALLET_KEY)
        await self._check_account_balance()
        apiconfig = APIConfig(self.configuration)
        await apiconfig.initialize()
        noncecore = NonceCore(self.web3, self.account.address, self.configuration)
        await noncecore.initialize()
        safetynet = SafetyNet(self.web3, self.configuration, self.account.address, self.account, apiconfig, None)
        await safetynet.initialize()
        marketmonitor = MarketMonitor(self.web3, self.configuration, apiconfig, None)
        await marketmonitor.initialize()
        transactioncore = TransactionCore(self.web3, self.account, self.configuration, apiconfig, marketmonitor, None, noncecore, safetynet)
        await transactioncore.initialize()
        mempoolmonitor = MempoolMonitor(
            self.web3,
            safetynet,
            noncecore,
            apiconfig,
            await self.configuration._load_json_safe(self.configuration.TOKEN_ADDRESSES, "TOKEN_ADDRESSES"),
            self.configuration,
            marketmonitor
        )
        await mempoolmonitor.initialize()
        strategynet = StrategyNet(transactioncore, marketmonitor, safetynet, apiconfig)
        await strategynet.initialize()
        self.components = {
            "apiconfig": apiconfig,
            "noncecore": noncecore,
            "safetynet": safetynet,
            "marketmonitor": marketmonitor,
            "transactioncore": transactioncore,
            "mempoolmonitor": mempoolmonitor,
            "strategynet": strategynet
        }

    async def run(self) -> None:
        self.running = True
        tasks = []
        tasks.append(asyncio.create_task(self.components["mempoolmonitor"].start_monitoring()))
        tasks.append(asyncio.create_task(self._process_transactions()))
        tasks.append(asyncio.create_task(self._monitor_memory()))
        await asyncio.gather(*tasks)

    async def _process_transactions(self) -> None:
        while self.running:
            try:
                tx = await asyncio.wait_for(self.components["mempoolmonitor"].profitable_transactions.get(), timeout=self.configuration.PROFITABLE_TX_PROCESS_TIMEOUT)
                await self.components["strategynet"].execute_best_strategy(tx, tx.get("strategy_type", "front_run"))
                self.components["mempoolmonitor"].profitable_transactions.task_done()
            except asyncio.TimeoutError:
                continue

    async def _monitor_memory(self) -> None:
        while self.running:
            snapshot = tracemalloc.take_snapshot()
            stats = snapshot.compare_to(self.memory_snapshot, "lineno")
            self.memory_snapshot = snapshot
            await asyncio.sleep(self.configuration.MEMORY_CHECK_INTERVAL)

    async def stop(self) -> None:
        self.running = False
        stop_tasks = [comp.stop() for comp in self.components.values() if hasattr(comp, "stop")]
        await asyncio.gather(*stop_tasks, return_exceptions=True)
        if hasattr(self.web3.provider, "disconnect"):
            await self.web3.provider.disconnect()
        tracemalloc.stop()

    async def emergency_shutdown(self) -> None:
        self.running = False
        for task in asyncio.all_tasks():
            task.cancel()
        await self.stop()
        sys.exit(0)

async def run_bot() -> None:
    config = Configuration()
    core = MainCore(config)
    await core.initialize_components()
    await core.run()

if __name__ == "__main__":
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        pass
