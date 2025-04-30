import logging
import sys
import threading
import time
import colorlog
import json
from typing import Dict, Any, Optional

# Flag to control JSON output (e.g., via environment variable)
# USE_JSON_LOGGING = os.getenv("LOG_FORMAT", "color").lower() == "json"
USE_JSON_LOGGING = False # Default to color logging for console

class JsonFormatter(logging.Formatter):
    """Formats log records as JSON strings."""
    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        # Add standard fields if they exist
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
             log_entry["stack_info"] = self.formatStack(record.stack_info)

        # A16: Include 'extra' dictionary fields automatically
        # Standard attributes provided by logging.LogRecord
        standard_attrs = logging.LogRecord('', '', '', '', '', '', '', '').__dict__.keys()
        extra_data = {k: v for k, v in record.__dict__.items() if k not in standard_attrs and not k.startswith('_')}
        if extra_data:
             log_entry.update(extra_data) # Merge extra dict into the main entry

        # Handle specific 'extra' keys we expect (component, tx_hash) for consistency
        log_entry["component"] = getattr(record, 'component', None) or extra_data.get('component', 'N/A')
        log_entry["tx_hash"] = getattr(record, 'tx_hash', None) or extra_data.get('tx_hash', None) # Keep None if not present

        # Clean up potential duplicates or None values if needed
        if log_entry["component"] == 'N/A' and 'component' in log_entry:
            del log_entry['component'] # Remove if component wasn't actually set
        if log_entry["tx_hash"] is None and 'tx_hash' in log_entry:
             del log_entry['tx_hash'] # Remove if tx_hash wasn't actually set

        return json.dumps(log_entry)

class TaskSpinner:
    """Manages a simple spinner animation in a separate thread."""
    def __init__(self, message: str = "Working"):
        self._message = message
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._spinner_chars = ['◢', '◣', '◤', '◥'] #'|', '/', '-', '\\'
        self._idx = 0

    def _spin(self) -> None:
        """The function run by the spinner thread."""
        while not self._stop_event.is_set():
            try:
                char = self._spinner_chars[self._idx % len(self._spinner_chars)]
                # Write to stderr to avoid interfering with stdout logs if needed
                sys.stderr.write(f"\r{self._message}... {char}")
                sys.stderr.flush()
                time.sleep(0.15)
                self._idx += 1
            except Exception: # Ignore errors during spinning (e.g., IO errors)
                 break
        # Clear the spinner line
        try:
            sys.stderr.write("\r" + " " * (len(self._message) + 5) + "\r")
            sys.stderr.flush()
        except Exception:
             pass # Ignore errors during cleanup


    def start(self) -> None:
        """Starts the spinner thread."""
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._spin, daemon=True, name="SpinnerThread")
            self._thread.start()

    def stop(self) -> None:
        """Stops the spinner thread."""
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join(timeout=0.5) # Wait briefly for thread to clear line
        self._thread = None

# Global spinner instance (optional)
_global_spinner: Optional[TaskSpinner] = None

def setup_logging(name: str, level: int = logging.INFO, use_spinner: bool = False, spinner_message: str = "Loading") -> logging.Logger:
    """
    Configures and returns a logger instance. Supports color or JSON format.

    Args:
        name (str): The name for the logger.
        level (int): The logging level (e.g., logging.DEBUG, logging.INFO).
        use_spinner (bool): Display a loading spinner initially (only affects first setup call).
        spinner_message (str): Message to display with the spinner.

    Returns:
        logging.Logger: The configured logger instance.
    """
    global _global_spinner

    if use_spinner and _global_spinner is None:
        _global_spinner = TaskSpinner(spinner_message)
        _global_spinner.start()

    logger = logging.getLogger(name)
    # Set level ONLY if the new level is lower than current, avoid raising level implicitly
    if not logger.hasHandlers() or level < logger.level:
        logger.setLevel(level)

    # Prevent adding duplicate handlers
    if not logger.handlers:
        # Choose handler based on global flag or config
        if USE_JSON_LOGGING:
            handler = logging.StreamHandler(sys.stdout) # Or stderr
            formatter = JsonFormatter(datefmt='%Y-%m-%dT%H:%M:%S%z')
        else:
            # Use Colorlog for console output
            handler = colorlog.StreamHandler(sys.stdout)
            # A16: Ensure formatter can handle extra fields (though colorlog doesn't display them by default)
            # Example format string including potential extra fields if needed:
            # '%(log_color)s%(asctime)s [%(levelname)-8s] %(name)s:%(lineno)d: %(message)s %(tx_hash)s%(reset)s'
            formatter = colorlog.ColoredFormatter(
                '%(log_color)s[%(levelname)-8s] %(name)s: %(message)s%(reset)s',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                },
                secondary_log_colors={},
                style='%' # Use %-style formatting
            )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        # Propagate logs to root logger ONLY if this is not the root logger itself
        # logger.propagate = logger.name != "" # Or check against logging.root

    # Stop spinner after initial setup potentially
    if use_spinner and _global_spinner:
        _global_spinner.stop()
        _global_spinner = None # Allow spinner again if called later

    return logger