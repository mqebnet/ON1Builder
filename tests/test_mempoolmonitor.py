# LICENSE: MIT // github.com/John0n1/ON1Builder

import pytest
from unittest.mock import AsyncMock, patch
from python.txpool_monitor import TxpoolMonitor

@pytest.fixture
def txpool_monitor():
    return TxpoolMonitor()

@pytest.mark.asyncio
async def test_handle_new_transactions(txpool_monitor):
    transactions = ["tx1", "tx2", "tx3"]
    with patch.object(txpool_monitor, '_process_transaction', new_callable=AsyncMock) as mock_process_transaction:
        await txpool_monitor._handle_new_transactions(transactions)
        assert mock_process_transaction.call_count == len(transactions)

@pytest.mark.asyncio
async def test_queue_transaction(txpool_monitor):
    transaction = "tx1"
    with patch.object(txpool_monitor, '_process_transaction', new_callable=AsyncMock) as mock_process_transaction:
        await txpool_monitor._queue_transaction(transaction)
        mock_process_transaction.assert_called_once_with(transaction)

@pytest.mark.asyncio
async def test_monitor_memory(txpool_monitor):
    with patch('python.txpool_monitor.psutil.virtual_memory') as mock_virtual_memory:
        mock_virtual_memory.return_value.percent = 50
        await txpool_monitor._monitor_memory()
        assert mock_virtual_memory.called

@pytest.mark.asyncio
async def test_get_dynamic_gas_price(txpool_monitor):
    with patch('python.txpool_monitor.web3.eth.getBlock') as mock_get_block:
        mock_get_block.return_value = {'baseFeePerGas': 1000000000}
        gas_price = await txpool_monitor.get_dynamic_gas_price()
        assert gas_price == 1000000000
