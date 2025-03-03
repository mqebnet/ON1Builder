#========================================================================================================================
# https://github.com/John0n1/0xBuilder

import logging
import sys
import threading
import time
import colorlog

def setup_logging(name, level=logging.INFO):
    # Get the logger and remove any duplicate handlers.
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.hasHandlers():
        logger.handlers.clear()
    # Stream handler with color formatting for console output
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
    return logger

class LoadingSpinner:
    """
    A simple spinner animation for showing progress during loading.
    Usage:
      spinner = LoadingSpinner("Loading components")
      spinner.start()
      ... load components ...
      spinner.stop()
    """
    spinner_chars = ['|', '/', '-', '\\']

    def __init__(self, message="Loading"):
        self.message = message
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._spinner_task, daemon=True)

    def _spinner_task(self):
        idx = 0
        while not self.stop_event.is_set():
            sys.stdout.write(f"\r{self.message}... {self.spinner_chars[idx % len(self.spinner_chars)]}")
            sys.stdout.flush()
            time.sleep(0.1)
            idx += 1
        sys.stdout.write("\r" + " " * (len(self.message) + 10) + "\r")

    def start(self):
        self.stop_event.clear()
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()


