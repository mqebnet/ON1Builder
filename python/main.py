import asyncio
from maincore import MainCore
from configuration import Configuration
from loggingconfig import setup_logging
import logging

logger = setup_logging("Main", level=logging.DEBUG)

async def run_bot() -> None:
    configuration = Configuration()
    core = MainCore(configuration)
    await core.initialize_components()
    await core.run()

async def main() -> None:
    await run_bot()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.critical(f"Fatal error in main: {e}")
