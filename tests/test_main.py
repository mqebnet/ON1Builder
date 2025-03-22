import pytest
from unittest.mock import AsyncMock, patch
from python.main import main, run_bot

@pytest.mark.asyncio
async def test_main():
    with patch('python.main.run_bot', new_callable=AsyncMock) as mock_run_bot:
        await main()
        mock_run_bot.assert_called_once()

@pytest.mark.asyncio
async def test_run_bot():
    with patch('python.maincore.MainCore.initialize', new_callable=AsyncMock) as mock_initialize, \
         patch('python.maincore.MainCore.run', new_callable=AsyncMock) as mock_run:
        await run_bot()
        mock_initialize.assert_called_once()
        mock_run.assert_called_once()
