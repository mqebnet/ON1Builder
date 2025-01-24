import asyncio
import tracemalloc
import logging as logger  # Add this import

from typing import Optional

from main_core import Main_Core
from configuration import Configuration

import logging as logger
from main_core import setup_logging

setup_logging()

logger = logger.getLogger(__name__)

async def run_bot():
    """Run the bot with graceful shutdown handling."""
    loop = asyncio.get_running_loop()
    shutdown_handler_task: Optional[asyncio.Task] = None

    try:
        # Start memory tracking
        tracemalloc.start()
        logger.info("Starting 0xBuilder...")

        # Initialize configuration
        configuration = Configuration()

        # Create and initialize main core
        core = Main_Core(configuration)

        # Initialize and run
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
        # Stop memory tracking
        tracemalloc.stop()
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
