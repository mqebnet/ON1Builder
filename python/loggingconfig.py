#========================================================================================================================
# https://github.com/John0n1/0xBuilder

import logging
import colorlog
import asyncio  # added for animate_dots support
import sys
import itertools

def setup_logging(name, level=logging.DEBUG):
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(name)s | %(levelname)s: %(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red'
        }
    ))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

async def animate_dots_in_place(base_msg, duration):
    start = asyncio.get_event_loop().time()
    try:
        for dots in itertools.cycle([".", "..", "..."]):
            if asyncio.get_event_loop().time() - start >= duration:
                break
            sys.stdout.write("\r" + base_msg + dots)
            sys.stdout.flush()
            await asyncio.sleep(0.5)
    finally:
        sys.stdout.write("\r" + base_msg + "...\n")
        sys.stdout.flush()

def patch_logger_for_animation(logger):
    """
    Overrides logger.info so that if a message ends with '...',
    the message is printed once with its base text and the trailing dots animate in place.
    """
    original_info = logger.info
    def new_info(msg, *args, **kwargs):
        if isinstance(msg, str) and msg.rstrip().endswith("«"):
            base_msg = msg.rstrip()[:-3]
            # Directly print the base message without a newline
            sys.stdout.write(base_msg)
            sys.stdout.flush()
            loop = asyncio.get_running_loop()
            loop.create_task(animate_dots_in_place(base_msg, 3))
        else:
            original_info(msg, *args, **kwargs)
    logger.info = new_info
    sys.stdout.flush()

async def animate_dots():
    import itertools, sys
    for dots in itertools.cycle([".", "..", "..."]):
        sys.stdout.write("\rLoading " + dots)
        sys.stdout.flush()
        await asyncio.sleep(0.5)

async def _animate_dots_for(duration):
    task = asyncio.create_task(animate_dots())
    await asyncio.sleep(duration)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
