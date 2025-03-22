import pytest
from unittest.mock import AsyncMock, patch
from python.mempoolmonitor import MempoolMonitor

@pytest.fixture
def mempool_monitor():
    return MempoolMonitor()

@pytest.mark.asyncio
async def test_handle_new_transactions(mempool_monitor):
    transactions = ["tx1", "tx2", "tx3"]
    with patch.object(mempool_monitor, '_process_transaction', new_callable=AsyncMock) as mock_process_transaction:
        await mempool_monitor._handle_new_transactions(transactions)
        assert mock_process_transaction.call_count == len(transactions)

@pytest.mark.asyncio
async def test_queue_transaction(mempool_monitor):
    transaction = "tx1"
    with patch.object(mempool_monitor, '_process_transaction', new_callable=AsyncMock) as mock_process_transaction:
        await mempool_monitor._queue_transaction(transaction)
        mock_process_transaction.assert_called_once_with(transaction)

@pytest.mark.asyncio
async def test_monitor_memory(mempool_monitor):
    with patch('python.mempoolmonitor.psutil.virtual_memory') as mock_virtual_memory:
        mock_virtual_memory.return_value.percent = 50
        await mempool_monitor._monitor_memory()
        assert mock_virtual_memory.called

@pytest.mark.asyncio
async def test_get_dynamic_gas_price(mempool_monitor):
    with patch('python.mempoolmonitor.web3.eth.getBlock') as mock_get_block:
        mock_get_block.return_value = {'baseFeePerGas': 1000000000}
        gas_price = await mempool_monitor.get_dynamic_gas_price()
        assert gas_price == 1000000000
