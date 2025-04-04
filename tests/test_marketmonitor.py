# LICENSE: MIT // github.com/John0n1/ON1Builder

import pytest
from unittest.mock import AsyncMock, patch
from python.marketmonitor import MarketMonitor
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
def marketmonitor(configuration):
    return MarketMonitor(configuration)

@pytest.mark.asyncio
async def test_initialize(marketmonitor):
    with patch.object(marketmonitor, 'initialize', new_callable=AsyncMock) as mock_initialize:
        await marketmonitor.initialize()
        mock_initialize.assert_called_once()

@pytest.mark.asyncio
async def test_schedule_updates(marketmonitor):
    with patch.object(marketmonitor, 'schedule_updates', new_callable=AsyncMock) as mock_schedule_updates:
        await marketmonitor.schedule_updates()
        mock_schedule_updates.assert_called_once()

@pytest.mark.asyncio
async def test_check_market_conditions(marketmonitor):
    token_address = "0xTokenAddress"
    with patch.object(marketmonitor, 'check_market_conditions', new_callable=AsyncMock) as mock_check_market_conditions:
        await marketmonitor.check_market_conditions(token_address)
        mock_check_market_conditions.assert_called_once_with(token_address)

@pytest.mark.asyncio
async def test_predict_price_movement(marketmonitor):
    token_symbol = "TEST"
    with patch.object(marketmonitor, 'predict_price_movement', new_callable=AsyncMock) as mock_predict_price_movement:
        await marketmonitor.predict_price_movement(token_symbol)
        mock_predict_price_movement.assert_called_once_with(token_symbol)

@pytest.mark.asyncio
async def test_get_market_features(marketmonitor):
    token_symbol = "TEST"
    with patch.object(marketmonitor, '_get_market_features', new_callable=AsyncMock) as mock_get_market_features:
        await marketmonitor._get_market_features(token_symbol)
        mock_get_market_features.assert_called_once_with(token_symbol)

@pytest.mark.asyncio
async def test_get_trading_metrics(marketmonitor):
    token_symbol = "TEST"
    with patch.object(marketmonitor, '_get_trading_metrics', new_callable=AsyncMock) as mock_get_trading_metrics:
        await marketmonitor._get_trading_metrics(token_symbol)
        mock_get_trading_metrics.assert_called_once_with(token_symbol)

@pytest.mark.asyncio
async def test_get_avg_transaction_value(marketmonitor):
    token_symbol = "TEST"
    with patch.object(marketmonitor, '_get_avg_transaction_value', new_callable=AsyncMock) as mock_get_avg_transaction_value:
        await marketmonitor._get_avg_transaction_value(token_symbol)
        mock_get_avg_transaction_value.assert_called_once_with(token_symbol)

@pytest.mark.asyncio
async def test_get_transaction_count(marketmonitor):
    token_symbol = "TEST"
    with patch.object(marketmonitor, '_get_transaction_count', new_callable=AsyncMock) as mock_get_transaction_count:
        await marketmonitor._get_transaction_count(token_symbol)
        mock_get_transaction_count.assert_called_once_with(token_symbol)

@pytest.mark.asyncio
async def test_get_trading_pairs_count(marketmonitor):
    token_symbol = "TEST"
    with patch.object(marketmonitor, '_get_trading_pairs_count', new_callable=AsyncMock) as mock_get_trading_pairs_count:
        await marketmonitor._get_trading_pairs_count(token_symbol)
        mock_get_trading_pairs_count.assert_called_once_with(token_symbol)

@pytest.mark.asyncio
async def test_get_exchange_count(marketmonitor):
    token_symbol = "TEST"
    with patch.object(marketmonitor, '_get_exchange_count', new_callable=AsyncMock) as mock_get_exchange_count:
        await marketmonitor._get_exchange_count(token_symbol)
        mock_get_exchange_count.assert_called_once_with(token_symbol)

@pytest.mark.asyncio
async def test_get_buy_sell_ratio(marketmonitor):
    token_symbol = "TEST"
    with patch.object(marketmonitor, '_get_buy_sell_ratio', new_callable=AsyncMock) as mock_get_buy_sell_ratio:
        await marketmonitor._get_buy_sell_ratio(token_symbol)
        mock_get_buy_sell_ratio.assert_called_once_with(token_symbol)

@pytest.mark.asyncio
async def test_get_smart_money_flow(marketmonitor):
    token_symbol = "TEST"
    with patch.object(marketmonitor, '_get_smart_money_flow', new_callable=AsyncMock) as mock_get_smart_money_flow:
        await marketmonitor._get_smart_money_flow(token_symbol)
        mock_get_smart_money_flow.assert_called_once_with(token_symbol)

@pytest.mark.asyncio
async def test_get_price_data(marketmonitor):
    args = ("arg1", "arg2")
    kwargs = {"kwarg1": "value1"}
    with patch.object(marketmonitor, 'get_price_data', new_callable=AsyncMock) as mock_get_price_data:
        await marketmonitor.get_price_data(*args, **kwargs)
        mock_get_price_data.assert_called_once_with(*args, **kwargs)

@pytest.mark.asyncio
async def test_get_token_volume(marketmonitor):
    token_symbol = "TEST"
    with patch.object(marketmonitor, 'get_token_volume', new_callable=AsyncMock) as mock_get_token_volume:
        await marketmonitor.get_token_volume(token_symbol)
        mock_get_token_volume.assert_called_once_with(token_symbol)

@pytest.mark.asyncio
async def test_stop(marketmonitor):
    with patch.object(marketmonitor, 'stop', new_callable=AsyncMock) as mock_stop:
        await marketmonitor.stop()
        mock_stop.assert_called_once()

@pytest.mark.asyncio
async def test_get_token_price(marketmonitor):
    token_symbol = "TEST"
    data_type = "current"
    timeframe = 1
    vs_currency = "eth"
    with patch.object(marketmonitor, 'get_token_price', new_callable=AsyncMock) as mock_get_token_price:
        await marketmonitor.get_token_price(token_symbol, data_type, timeframe, vs_currency)
        mock_get_token_price.assert_called_once_with(token_symbol, data_type, timeframe, vs_currency)

@pytest.mark.asyncio
async def test_is_arbitrage_opportunity(marketmonitor):
    token_symbol = "TEST"
    with patch.object(marketmonitor, '_is_arbitrage_opportunity', new_callable=AsyncMock) as mock_is_arbitrage_opportunity:
        await marketmonitor._is_arbitrage_opportunity(token_symbol)
        mock_is_arbitrage_opportunity.assert_called_once_with(token_symbol)
