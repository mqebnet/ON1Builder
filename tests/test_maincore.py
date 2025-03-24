import pytest
from unittest.mock import AsyncMock, patch
from maincore import MainCore
from configuration import Configuration

@pytest.fixture
def configuration():
    config = Configuration()
    config.WALLET_KEY = "test_wallet_key"
    config.BASE_PATH = "test_base_path"
    config.HTTP_ENDPOINT = "http://localhost:8545"
    config.WEBSOCKET_ENDPOINT = "ws://localhost:8546"
    config.IPC_ENDPOINT = "/path/to/geth.ipc"
    return config

@pytest.fixture
def maincore(configuration):
    return MainCore(configuration)

@pytest.mark.asyncio
async def test_initialize_components(maincore):
    with patch.object(maincore, '_load_configuration', new_callable=AsyncMock) as mock_load_config, \
         patch.object(maincore, '_initialize_web3', new_callable=AsyncMock) as mock_init_web3, \
         patch.object(maincore, '_check_account_balance', new_callable=AsyncMock) as mock_check_balance, \
         patch('abiregistry.ABIRegistry.initialize', new_callable=AsyncMock) as mock_abi_init, \
         patch('apiconfig.APIConfig.initialize', new_callable=AsyncMock) as mock_api_config_init, \
         patch('noncecore.NonceCore.initialize', new_callable=AsyncMock) as mock_nonce_init, \
         patch('safetynet.SafetyNet.initialize', new_callable=AsyncMock) as mock_safetynet_init, \
         patch('transactioncore.TransactionCore.initialize', new_callable=AsyncMock) as mock_txcore_init, \
         patch('marketmonitor.MarketMonitor.initialize', new_callable=AsyncMock) as mock_marketmonitor_init, \
         patch('mempoolmonitor.MempoolMonitor.initialize', new_callable=AsyncMock) as mock_mempoolmonitor_init, \
         patch('strategynet.StrategyNet.initialize', new_callable=AsyncMock) as mock_strategynet_init:
        
        await maincore._initialize_components()

        mock_load_config.assert_called_once()
        mock_init_web3.assert_called_once()
        mock_check_balance.assert_called_once()
        mock_abi_init.assert_called_once()
        mock_api_config_init.assert_called_once()
        mock_nonce_init.assert_called_once()
        mock_safetynet_init.assert_called_once()
        mock_txcore_init.assert_called_once()
        mock_marketmonitor_init.assert_called_once()
        mock_mempoolmonitor_init.assert_called_once()
        mock_strategynet_init.assert_called_once()

@pytest.mark.asyncio
async def test_initialize(maincore):
    with patch.object(maincore, '_initialize_components', new_callable=AsyncMock) as mock_init_components:
        await maincore.initialize()
        mock_init_components.assert_called_once()

@pytest.mark.asyncio
async def test_run(maincore):
    with patch.object(maincore.components['mempoolmonitor'], 'start_monitoring', new_callable=AsyncMock) as mock_start_monitoring, \
         patch.object(maincore, '_process_profitable_transactions', new_callable=AsyncMock) as mock_process_tx, \
         patch.object(maincore, '_monitor_memory', new_callable=AsyncMock) as mock_monitor_memory, \
         patch.object(maincore, '_check_component_health', new_callable=AsyncMock) as mock_check_health:
        
        await maincore.run()

        mock_start_monitoring.assert_called_once()
        mock_process_tx.assert_called_once()
        mock_monitor_memory.assert_called_once()
        mock_check_health.assert_called_once()

@pytest.mark.asyncio
async def test_stop(maincore):
    with patch.object(maincore, '_stop_component', new_callable=AsyncMock) as mock_stop_component, \
         patch.object(maincore.web3.provider, 'disconnect', new_callable=AsyncMock) as mock_disconnect:
        
        await maincore.stop()

        mock_stop_component.assert_called()
        mock_disconnect.assert_called_once()

@pytest.mark.asyncio
async def test_emergency_shutdown(maincore):
    with patch('asyncio.all_tasks', return_value=[]), \
         patch.object(maincore.web3.provider, 'disconnect', new_callable=AsyncMock) as mock_disconnect:
        
        await maincore.emergency_shutdown()

        mock_disconnect.assert_called_once()
