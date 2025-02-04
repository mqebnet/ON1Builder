import signal
import asyncio
from typing import Optional
import tracemalloc
from main_core import Main_Core
from configuration import Configuration

import logging as logger


logger = logger.getLogger(__name__)

async def run_bot():
    """Run the bot with graceful shutdown handling."""
    loop = asyncio.get_running_loop()

    def shutdown_handler():
        """Handle shutdown signals."""
        logger.info("Received shutdown signal. Stopping the bot...")


    try:
        # Start memory tracking
        tracemalloc.start()
        logger.info("Starting 0xBuilder...")

        # Initialize configuration
        configuration = Configuration()

        # Create and initialize main core
        core = Main_Core(configuration)

        # Register shutdown signals
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, shutdown_handler)

        # Initialize and run
        await core.initialize()
        await core.run()

    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        if tracemalloc.is_tracing(): # Keep snapshot in except block for immediate error info
            snapshot = tracemalloc.take_snapshot()
            logger.debug("Top 10 memory allocations:")
            for stat in snapshot.statistics('lineno')[:10]:
                logger.debug(str(stat))
    finally:
        # Stop memory tracking
        tracemalloc.stop()
        if tracemalloc.is_tracing(): 
            snapshot = tracemalloc.take_snapshot()
            logger.debug("Final memory allocations at shutdown:")
            for stat in snapshot.statistics('lineno')[:10]:
                logger.debug(str(stat))
        logger.info("0xBuilder shutdown complete")

async def main():
    """Main entry point."""
    await run_bot()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        # Get current memory snapshot on error
        snapshot = tracemalloc.take_snapshot()
        logger.critical(f"Program terminated with an error: {e}")
        logger.debug("Top 10 memory allocations at error:")
        top_stats = snapshot.statistics('lineno')
        for stat in top_stats[:10]:
            logger.debug(str(stat))