import pytest
from unittest.mock import AsyncMock, patch
from src.transactioncore import TransactionCore
from src.configuration import Configuration

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
def transactioncore(configuration):
    return TransactionCore(configuration)

@pytest.mark.asyncio
async def test_initialize(transactioncore):
    with patch.object(transactioncore, 'initialize', new_callable=AsyncMock) as mock_initialize:
        await transactioncore.initialize()
        mock_initialize.assert_called_once()

@pytest.mark.asyncio
async def test_build_transaction(transactioncore):
    function_call = AsyncMock()
    additional_params = {"param1": "value1"}
    with patch.object(transactioncore, 'build_transaction', new_callable=AsyncMock) as mock_build_transaction:
        await transactioncore.build_transaction(function_call, additional_params)
        mock_build_transaction.assert_called_once_with(function_call, additional_params)

@pytest.mark.asyncio
async def test_execute_transaction(transactioncore):
    tx = {"to": "0xAddress", "value": 1000}
    with patch.object(transactioncore, 'execute_transaction', new_callable=AsyncMock) as mock_execute_transaction:
        await transactioncore.execute_transaction(tx)
        mock_execute_transaction.assert_called_once_with(tx)

@pytest.mark.asyncio
async def test_handle_eth_transaction(transactioncore):
    target_tx = {"tx_hash": "0xHash", "value": 1000}
    with patch.object(transactioncore, 'handle_eth_transaction', new_callable=AsyncMock) as mock_handle_eth_transaction:
        await transactioncore.handle_eth_transaction(target_tx)
        mock_handle_eth_transaction.assert_called_once_with(target_tx)

@pytest.mark.asyncio
async def test_simulate_transaction(transactioncore):
    transaction = {"to": "0xAddress", "value": 1000}
    with patch.object(transactioncore, 'simulate_transaction', new_callable=AsyncMock) as mock_simulate_transaction:
        await transactioncore.simulate_transaction(transaction)
        mock_simulate_transaction.assert_called_once_with(transaction)

@pytest.mark.asyncio
async def test_prepare_flashloan_transaction(transactioncore):
    flashloan_asset = "0xAsset"
    flashloan_amount = 1000
    with patch.object(transactioncore, 'prepare_flashloan_transaction', new_callable=AsyncMock) as mock_prepare_flashloan_transaction:
        await transactioncore.prepare_flashloan_transaction(flashloan_asset, flashloan_amount)
        mock_prepare_flashloan_transaction.assert_called_once_with(flashloan_asset, flashloan_amount)

@pytest.mark.asyncio
async def test_send_bundle(transactioncore):
    transactions = [{"to": "0xAddress", "value": 1000}]
    with patch.object(transactioncore, 'send_bundle', new_callable=AsyncMock) as mock_send_bundle:
        await transactioncore.send_bundle(transactions)
        mock_send_bundle.assert_called_once_with(transactions)

@pytest.mark.asyncio
async def test_front_run(transactioncore):
    target_tx = {"tx_hash": "0xHash", "value": 1000}
    with patch.object(transactioncore, 'front_run', new_callable=AsyncMock) as mock_front_run:
        await transactioncore.front_run(target_tx)
        mock_front_run.assert_called_once_with(target_tx)

@pytest.mark.asyncio
async def test_back_run(transactioncore):
    target_tx = {"tx_hash": "0xHash", "value": 1000}
    with patch.object(transactioncore, 'back_run', new_callable=AsyncMock) as mock_back_run:
        await transactioncore.back_run(target_tx)
        mock_back_run.assert_called_once_with(target_tx)

@pytest.mark.asyncio
async def test_execute_sandwich_attack(transactioncore):
    target_tx = {"tx_hash": "0xHash", "value": 1000}
    strategy = "default"
    with patch.object(transactioncore, 'execute_sandwich_attack', new_callable=AsyncMock) as mock_execute_sandwich_attack:
        await transactioncore.execute_sandwich_attack(target_tx, strategy)
        mock_execute_sandwich_attack.assert_called_once_with(target_tx, strategy)

@pytest.mark.asyncio
async def test_cancel_transaction(transactioncore):
    nonce = 1
    with patch.object(transactioncore, 'cancel_transaction', new_callable=AsyncMock) as mock_cancel_transaction:
        await transactioncore.cancel_transaction(nonce)
        mock_cancel_transaction.assert_called_once_with(nonce)

@pytest.mark.asyncio
async def test_withdraw_eth(transactioncore):
    with patch.object(transactioncore, 'withdraw_eth', new_callable=AsyncMock) as mock_withdraw_eth:
        await transactioncore.withdraw_eth()
        mock_withdraw_eth.assert_called_once()

@pytest.mark.asyncio
async def test_transfer_profit_to_account(transactioncore):
    amount = 1000
    account = "0xAccount"
    with patch.object(transactioncore, 'transfer_profit_to_account', new_callable=AsyncMock) as mock_transfer_profit_to_account:
        await transactioncore.transfer_profit_to_account(amount, account)
        mock_transfer_profit_to_account.assert_called_once_with(amount, account)

@pytest.mark.asyncio
async def test_stop(transactioncore):
    with patch.object(transactioncore, 'stop', new_callable=AsyncMock) as mock_stop:
        await transactioncore.stop()
        mock_stop.assert_called_once()
