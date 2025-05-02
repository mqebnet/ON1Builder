import asyncio
from web3 import AsyncWeb3, WebSocketProvider

async def test_websocket():
    ws_url = "wss://f5go94oysie8gi80q9efxp7wa.blockchainnodeengine.com"
    web3 = AsyncWeb3(WebSocketProvider(ws_url))
    try:
        connected = await web3.is_connected()
        if connected:
            chain_id = await web3.eth.chain_id
            print(f"Connected to WebSocket. Chain ID: {chain_id}")
        else:
            print("Failed to connect to WebSocket.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())
