#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#========================================================================================================================
# https://github.com/John0n1/ON1Builder

import signal
import asyncio
import tracemalloc
from maincore import MainCore
from configuration import Configuration

from loggingconfig import setup_logging
import logging

logger = setup_logging("Main", level=logging.INFO)


async def run_bot() -> None:
    """Run the bot with graceful shutdown handling."""
    loop = asyncio.get_running_loop()

    tracemalloc.start()
    await asyncio.sleep(3)
   
    configuration = Configuration()
    core = MainCore(configuration)

    # Define a shutdown handler that triggers a graceful stop.
    def shutdown_handler() -> None:
        logger.debug("Received shutdown signal. Initiating graceful shutdown...")
        asyncio.create_task(core.stop())

    # Register signal handlers for graceful shutdown.
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_handler)

    try:
        await core.initialize()
        await core.run()
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        if tracemalloc.is_tracing():
            snapshot = tracemalloc.take_snapshot()
            logger.debug("Top 10 memory allocations:")
            for stat in snapshot.statistics('lineno')[:10]:
                logger.debug(str(stat))
    finally:
        tracemalloc.stop()
        if tracemalloc.is_tracing():
            snapshot = tracemalloc.take_snapshot()
            logger.debug("Final memory allocations at shutdown:")
            for stat in snapshot.statistics('lineno')[:10]:
                logger.debug(str(stat))
        logger.debug("ON1Builder shutdown complete")


async def main() -> None:
    """Main entry point."""
    await run_bot()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        snapshot = tracemalloc.take_snapshot()
        logger.critical(f"Program terminated with an error: {e}")
        logger.debug("Top 10 memory allocations at error:")
        for stat in snapshot.statistics('lineno')[:10]:
            logger.debug(str(stat))
        logger.debug("ON1Builder terminated")