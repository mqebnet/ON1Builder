import asyncio
import os
import sys
# Ensure the main package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from maincore import MainCore, main_entry_point # Import the async entry point
from configuration import Configuration # Import facade
from loggingconfig import setup_logging
import logging

# Setup root logger early if needed, or rely on MainCore's setup
# logger = setup_logging("MainEntry", level=logging.DEBUG) # Example root logger setup

async def run_bot_legacy() -> None:
    """Legacy function if needed, prefer main_entry_point."""
    logger = setup_logging("RunBotLegacy", level=logging.INFO) # Use specific logger
    logger.warning("Using legacy run_bot function. Prefer main_entry_point.")
    configuration = Configuration()
    core = MainCore(configuration)
    try:
        await core.initialize_components()
        await core.run()
    except Exception as e:
         logger.critical("Fatal error in run_bot_legacy: %s", e, exc_info=True)
         # Attempt cleanup if possible
         if core.running:
             await core.stop()
    finally:
         logger.info("run_bot_legacy finished.")


if __name__ == "__main__":
    # Basic configuration check (optional)
    if not os.getenv("WALLET_KEY"):
        print("ERROR: WALLET_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)
    if not os.getenv("HTTP_ENDPOINT"):
         print("ERROR: HTTP_ENDPOINT environment variable not set.", file=sys.stderr)
         sys.exit(1)

    # Configure PYTHONASYNCIODEBUG=1 for development/testing event loop issues (A3)
    if os.getenv("PYTHONASYNCIODEBUG") == "1":
        # BasicConfig can interfere with colorlog, setup logging carefully
        # logging.basicConfig(level=logging.DEBUG)
        log = setup_logging("AsyncioDebug", level=logging.WARNING)
        log.warning("Asyncio debug mode enabled via PYTHONASYNCIODEBUG=1.")
        # Consider setting root logger level if needed for asyncio logs
        # logging.getLogger().setLevel(logging.DEBUG)

    # Run the main asynchronous entry point
    try:
        # Use the refactored entry point from maincore
        asyncio.run(main_entry_point())
    except KeyboardInterrupt:
        # Logger might not be fully initialized here, use print
        print("\nKeyboardInterrupt received. Shutting down gracefully...", file=sys.stderr)
        # Signal handlers in MainCore should manage the shutdown.
        # Add a small delay to allow handlers to process.
        # time.sleep(1) # Not ideal in async context
    except Exception as e:
        # Final catch-all for errors during asyncio.run itself
        print(f"\nFATAL ERROR during application startup or shutdown: {e}", file=sys.stderr)
        # Optionally print traceback
        import traceback
        traceback.print_exc()
        sys.exit(1) # Indicate failure
