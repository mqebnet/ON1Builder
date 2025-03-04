#========================================================================================================================
# https://github.com/John0n1/0xBuilder

import logging
import sys
import threading
import time
import colorlog

def setup_logging(name, level=logging.INFO, spinner=False, spinner_message="Loading"):
    # Nested spinner functions for smoother animation
    def spinner_task(message, stop_event):
        spinner_chars = ['|', '/', '-', '\\']
        idx = 0
        while not stop_event.is_set():
            sys.stdout.write(f"\r{message}... {spinner_chars[idx % len(spinner_chars)]}")
            sys.stdout.flush()
            time.sleep(0.1)
            idx += 1
        sys.stdout.write("\r" + " " * (len(message) + 10) + "\r")

    # Optionally start spinner
    if spinner:
        stop_event = threading.Event()
        thread = threading.Thread(target=spinner_task, args=(spinner_message, stop_event), daemon=True)
        thread.start()
    # Stream handler with color formatting for console output
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = colorlog.StreamHandler(sys.stdout)
    formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(name)s | %(levelname)-8s: %(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red'
        }
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # Stop spinner if it was started
    if spinner:
        stop_event.set()
        thread.join()
    return logger


