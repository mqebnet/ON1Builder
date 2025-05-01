import asyncio
import os
import sys
# Ensure the main package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#!/usr/bin/env python3
import asyncio
import logging
import signal
from configuration import Configuration
from maincore import MainCore
from loggingconfig import setup_logging

logger = setup_logging("Main", level=logging.INFO)

async def main():
    config = Configuration()
    core = MainCore(config)
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _signal_handler():
        stop_event.set()

    loop.add_signal_handler(signal.SIGINT, _signal_handler)
    loop.add_signal_handler(signal.SIGTERM, _signal_handler)

    await core.initialize_components()
    runner = asyncio.create_task(core.run())
    await stop_event.wait()
    await core.stop()
    await runner

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
