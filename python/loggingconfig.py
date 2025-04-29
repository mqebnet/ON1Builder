# LICENSE: MIT // github.com/John0n1/ON1Builder

import logging
import sys
import threading
import time
import colorlog

def setup_logging(name: str, level: int = logging.INFO, spinner: bool = False, spinner_message: str = "Loading") -> logging.Logger:
    """
    Configure and return a logger.

    Args:
        name (str): Logger name.
        level (int): Logging level.
        spinner (bool): Whether to display a spinner animation.
        spinner_message (str): Spinner message.
    
    Returns:
        logging.Logger: Configured logger.
    """
    def spinner_task(message: str, stop_event: threading.Event) -> None:
        spinner_chars = ['|', '/', '-', '\\']
        idx = 0
        while not stop_event.is_set():
            sys.stdout.write(f"\r{message}... {spinner_chars[idx % len(spinner_chars)]}")
            sys.stdout.flush()
            time.sleep(0.1)
            idx += 1
        sys.stdout.write("\r" + " " * (len(message) + 10) + "\r")
        sys.stdout.flush()

    if spinner:
        stop_event = threading.Event()
        thread = threading.Thread(target=spinner_task, args=(spinner_message, stop_event), daemon=True)
        thread.start()

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

    if spinner:
        stop_event.set()
        thread.join()

    return logger
# --- End file: loggingconfig.py ---
