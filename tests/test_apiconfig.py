# LICENSE: MIT // github.com/John0n1/ON1Builder

import pytest
from unittest.mock import AsyncMock, patch
from python.apiconfig import APIConfig
from python.configuration import Configuration

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
def apiconfig(configuration):
    return APIConfig(configuration)

@pytest.mark.asyncio
async def test_initialize(apiconfig):
    with patch.object(apiconfig, 'initialize', new_callable=AsyncMock) as mock_initialize:
        await apiconfig.initialize()
        mock_initialize.assert_called_once()

@pytest.mark.asyncio
async def test_get_token_symbol(apiconfig):
    address = "0xTokenAddress"
    with patch.object(apiconfig, 'get_token_symbol', new_callable=AsyncMock) as mock_get_token_symbol:
        await apiconfig.get_token_symbol(address)
        mock_get_token_symbol.assert_called_once_with(address)

@pytest.mark.asyncio
async def test_get_token_address(apiconfig):
    symbol = "TEST"
    with patch.object(apiconfig, 'get_token_address', new_callable=AsyncMock) as mock_get_token_address:
        await apiconfig.get_token_address(symbol)
        mock_get_token_address.assert_called_once_with(symbol)

@pytest.mark.asyncio
async def test_get_token_metadata(apiconfig):
    token = "TEST"
    with patch.object(apiconfig, 'get_token_metadata', new_callable=AsyncMock) as mock_get_token_metadata:
        await apiconfig.get_token_metadata(token)
        mock_get_token_metadata.assert_called_once_with(token)

@pytest.mark.asyncio
async def test_get_real_time_price(apiconfig):
    token = "TEST"
    vs_currency = "eth"
    with patch.object(apiconfig, 'get_real_time_price', new_callable=AsyncMock) as mock_get_real_time_price:
        await apiconfig.get_real_time_price(token, vs_currency)
        mock_get_real_time_price.assert_called_once_with(token, vs_currency)

@pytest.mark.asyncio
async def test_make_request(apiconfig):
    provider_name = "coingecko"
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": "bitcoin", "vs_currencies": "usd"}
    headers = {"x-cg-pro-api-key": "test_api_key"}
    with patch.object(apiconfig, 'make_request', new_callable=AsyncMock) as mock_make_request:
        await apiconfig.make_request(provider_name, url, params, headers)
        mock_make_request.assert_called_once_with(provider_name, url, params, headers)

@pytest.mark.asyncio
async def test_fetch_historical_prices(apiconfig):
    token = "TEST"
    days = 30
    with patch.object(apiconfig, 'fetch_historical_prices', new_callable=AsyncMock) as mock_fetch_historical_prices:
        await apiconfig.fetch_historical_prices(token, days)
        mock_fetch_historical_prices.assert_called_once_with(token, days)

@pytest.mark.asyncio
async def test_get_token_volume(apiconfig):
    token = "TEST"
    with patch.object(apiconfig, 'get_token_volume', new_callable=AsyncMock) as mock_get_token_volume:
        await apiconfig.get_token_volume(token)
        mock_get_token_volume.assert_called_once_with(token)

@pytest.mark.asyncio
async def test_get_token_price_data(apiconfig):
    token_symbol = "TEST"
    data_type = "current"
    timeframe = 1
    vs_currency = "eth"
    with patch.object(apiconfig, 'get_token_price_data', new_callable=AsyncMock) as mock_get_token_price_data:
        await apiconfig.get_token_price_data(token_symbol, data_type, timeframe, vs_currency)
        mock_get_token_price_data.assert_called_once_with(token_symbol, data_type, timeframe, vs_currency)

@pytest.mark.asyncio
async def test_update_training_data(apiconfig):
    with patch.object(apiconfig, 'update_training_data', new_callable=AsyncMock) as mock_update_training_data:
        await apiconfig.update_training_data()
        mock_update_training_data.assert_called_once()

@pytest.mark.asyncio
async def test_train_price_model(apiconfig):
    with patch.object(apiconfig, 'train_price_model', new_callable=AsyncMock) as mock_train_price_model:
        await apiconfig.train_price_model()
        mock_train_price_model.assert_called_once()

@pytest.mark.asyncio
async def test_predict_price(apiconfig):
    token = "TEST"
    with patch.object(apiconfig, 'predict_price', new_callable=AsyncMock) as mock_predict_price:
        await apiconfig.predict_price(token)
        mock_predict_price.assert_called_once_with(token)

@pytest.mark.asyncio
async def test_get_dynamic_gas_price(apiconfig):
    with patch.object(apiconfig, 'get_dynamic_gas_price', new_callable=AsyncMock) as mock_get_dynamic_gas_price:
        await apiconfig.get_dynamic_gas_price()
        mock_get_dynamic_gas_price.assert_called_once()
