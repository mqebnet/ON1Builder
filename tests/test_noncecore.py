# LICENSE: MIT // github.com/John0n1/ON1Builder

import pytest
from unittest.mock import AsyncMock, patch
from python.noncecore import NonceCore
from python.configuration import Configuration
from web3 import AsyncWeb3

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
def noncecore(configuration):
    web3 = AsyncWeb3()
    address = "0xYourEthereumAddress"
    return NonceCore(web3, address, configuration)

@pytest.mark.asyncio
async def test_initialize(noncecore):
    with patch.object(noncecore, 'initialize', new_callable=AsyncMock) as mock_initialize:
        await noncecore.initialize()
        mock_initialize.assert_called_once()

@pytest.mark.asyncio
async def test_get_nonce(noncecore):
    with patch.object(noncecore, 'get_nonce', new_callable=AsyncMock) as mock_get_nonce:
        await noncecore.get_nonce()
        mock_get_nonce.assert_called_once()

@pytest.mark.asyncio
async def test_refresh_nonce(noncecore):
    with patch.object(noncecore, 'refresh_nonce', new_callable=AsyncMock) as mock_refresh_nonce:
        await noncecore.refresh_nonce()
        mock_refresh_nonce.assert_called_once()

@pytest.mark.asyncio
async def test_track_transaction(noncecore):
    tx_hash = "0xTransactionHash"
    nonce = 1
    with patch.object(noncecore, 'track_transaction', new_callable=AsyncMock) as mock_track_transaction:
        await noncecore.track_transaction(tx_hash, nonce)
        mock_track_transaction.assert_called_once_with(tx_hash, nonce)

@pytest.mark.asyncio
async def test_sync_nonce_with_chain(noncecore):
    with patch.object(noncecore, 'sync_nonce_with_chain', new_callable=AsyncMock) as mock_sync_nonce_with_chain:
        await noncecore.sync_nonce_with_chain()
        mock_sync_nonce_with_chain.assert_called_once()

@pytest.mark.asyncio
async def test_reset(noncecore):
    with patch.object(noncecore, 'reset', new_callable=AsyncMock) as mock_reset:
        await noncecore.reset()
        mock_reset.assert_called_once()

@pytest.mark.asyncio
async def test_stop(noncecore):
    with patch.object(noncecore, 'stop', new_callable=AsyncMock) as mock_stop:
        await noncecore.stop()
        mock_stop.assert_called_once()

@pytest.mark.asyncio
async def test_get_next_nonce(noncecore):
    with patch.object(noncecore, 'get_next_nonce', new_callable=AsyncMock) as mock_get_next_nonce:
        await noncecore.get_next_nonce()
        mock_get_next_nonce.assert_called_once()
