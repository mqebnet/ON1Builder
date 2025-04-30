from collections import deque
from typing import Any, Dict, List, Optional
from cachetools import TTLCache
from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import threading
import asyncio
import time
import queue
import logging
import sys
import os
from decimal import Decimal
import json

# Adjust path if necessary based on project structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))) # Add current dir
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Add parent dir if needed

from maincore import MainCore
from configuration import Configuration # Facade import
from loggingconfig import setup_logging
from mempoolmonitor import MempoolMonitor
from strategynet import StrategyNet
from safetynet import SafetyNet
from noncecore import NonceCore
from eth_account.signers.local import LocalAccount # Import for type check

# --- Flask App Setup ---
app = Flask(__name__, static_folder=None) # Disable default static folder
CORS(app) # Allow all origins for simplicity in development
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading') # Use threading mode for compatibility with sync Flask routes
# Note: 'threading' async_mode might have limitations compared to 'asyncio' or 'gevent'.
# If performance issues arise, consider restructuring or using a different web framework.

# --- Logging Setup ---
# Use a specific logger for the Flask app
ui_logger = setup_logging("FlaskUI", level=logging.INFO) # Use INFO level for UI logs by default

# --- Global Bot State ---
# Encapsulate bot state for better management
class BotState:
    def __init__(self):
        self.is_running: bool = False
        self.thread: Optional[threading.Thread] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.main_core: Optional[MainCore] = None
        self.lock: threading.Lock = threading.Lock() # Protect access to state variables

bot_state = BotState()

# --- WebSocket Log Handler ---
# A16: Modify handler to capture and forward structured log data
class WebSocketLogHandler(logging.Handler):
    MAX_QUEUE_SIZE = 200 # Limit memory usage for buffered logs

    def __init__(self):
        super().__init__(level=logging.DEBUG) # Capture all levels, filter later if needed
        self.log_queue = deque(maxlen=self.MAX_QUEUE_SIZE) # Use deque for efficient fixed-size queue

    def emit(self, record: logging.LogRecord):
        try:
            # Basic log entry
            log_entry = {
                'level': record.levelname,
                'name': record.name,
                'message': record.getMessage(), # Get formatted message
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(record.created)),
                # A16: Add extra fields if they exist
                'component': getattr(record, 'component', None),
                'tx_hash': getattr(record, 'tx_hash', None),
                # Add other potential structured fields here
            }
            # Remove None fields for cleaner output
            log_entry = {k: v for k, v in log_entry.items() if v is not None}

            self.log_queue.append(log_entry) # Append to deque

            # Emit to WebSocket clients
            # Use socketio.emit for thread safety when called from logging thread
            socketio.emit('log_message', log_entry)

        except Exception as e:
            # Handle potential errors during formatting or emit
            # Avoid logging within emit to prevent recursion
            print(f"Error in WebSocketLogHandler: {e}", file=sys.stderr)
            self.handleError(record)

# Create and add the handler to the root logger
ws_handler = WebSocketLogHandler()
# Optional: Add a specific formatter if needed, but getMessage() is usually sufficient
# ws_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s') # Example
# ws_handler.setFormatter(ws_formatter)
logging.getLogger().addHandler(ws_handler)
# Set root logger level to capture DEBUG from components if ws_handler level is DEBUG
logging.getLogger().setLevel(logging.DEBUG)
ui_logger.info("WebSocketLogHandler attached to root logger.")


# --- Bot Control Thread ---
def run_bot_in_thread():
    """Target function for the bot's background thread."""
    # This function runs in a separate thread, needs its own event loop
    global bot_state
    ui_logger.info("Bot thread started.")

    # Create and set a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    with bot_state.lock:
        bot_state.loop = loop
        # Ensure configuration is loaded before creating MainCore
        try:
             configuration = Configuration()
             # Perform any necessary sync validation if `load` isn't called in init
             # configuration.validate_sync() # Example
             bot_state.main_core = MainCore(configuration)
        except Exception as config_err:
             ui_logger.critical("Failed to load configuration in bot thread: %s", config_err, exc_info=True)
             bot_state.is_running = False # Ensure state reflects failure
             return # Exit thread if config fails

    main_core_instance = bot_state.main_core # Local reference

    try:
        # Run the main core initialization and run loop
        loop.run_until_complete(main_core_instance.initialize_components())
        ui_logger.info("MainCore initialized successfully in bot thread.")
        # run() will block until stopped or an error occurs
        loop.run_until_complete(main_core_instance.run())
        ui_logger.info("MainCore run loop finished in bot thread.")

    except Exception as e:
        ui_logger.critical("An error occurred in the bot thread: %s", e, exc_info=True)
        # Ensure stop is called on error if core exists
        if main_core_instance and main_core_instance.running:
             ui_logger.info("Attempting to stop MainCore due to error...")
             try:
                  # Schedule stop from the correct loop
                  loop.call_soon_threadsafe(asyncio.create_task, main_core_instance.stop())
             except Exception as stop_err:
                   ui_logger.error("Error scheduling stop after bot thread error: %s", stop_err)

    finally:
        ui_logger.info("Bot thread finishing.")
        with bot_state.lock:
             # Clean up loop resources
             try:
                  loop.run_until_complete(loop.shutdown_asyncgens())
                  loop.close()
                  ui_logger.info("Bot thread event loop closed.")
             except Exception as loop_close_err:
                  ui_logger.error("Error closing event loop in bot thread: %s", loop_close_err)

             # Update global state
             bot_state.is_running = False
             bot_state.loop = None
             # Keep main_core instance for potential status checks after stop? Or clear it?
             # bot_state.main_core = None # Clear if no longer needed


# --- Flask Routes ---
@app.route('/')
def serve_index():
    # Serve the main UI file (assuming it's in ../ui relative to app.py)
    ui_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ui'))
    return send_from_directory(ui_dir, 'index.html')

@app.route('/<path:filename>')
def serve_static_files(filename):
    # Serve other static files (JS, CSS) from the UI directory
     ui_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ui'))
     return send_from_directory(ui_dir, filename)

@app.route('/start', methods=['POST'])
def start_bot():
    """Starts the bot in a background thread."""
    global bot_state
    with bot_state.lock:
        if not bot_state.is_running:
            bot_state.is_running = True
            bot_state.thread = threading.Thread(target=run_bot_in_thread, daemon=True, name="ON1BuilderBotThread")
            bot_state.thread.start()
            ui_logger.info("Bot start request received. Starting background thread.")
            return jsonify({"status": "Bot starting..."}), 202 # Accepted
        else:
            ui_logger.warning("Received start request, but bot is already running.")
            return jsonify({"status": "Bot is already running"}), 409 # Conflict

@app.route('/stop', methods=['POST'])
def stop_bot():
    """Requests the bot to stop gracefully."""
    global bot_state
    with bot_state.lock:
        if bot_state.is_running and bot_state.main_core and bot_state.loop:
             ui_logger.info("Bot stop request received. Signaling MainCore to stop.")
             try:
                  # Schedule the async stop() method in the bot's event loop
                  # Use call_soon_threadsafe as we're calling from Flask thread -> bot thread loop
                  future = asyncio.run_coroutine_threadsafe(bot_state.main_core.stop(), bot_state.loop)
                  # Optionally wait briefly for acknowledgment (not full stop)
                  future.result(timeout=2) # Wait max 2s for scheduling/initiation
                  return jsonify({"status": "Bot stopping..."}), 202 # Accepted
             except TimeoutError:
                   ui_logger.warning("Timeout waiting for stop() initiation acknowledgement.")
                   return jsonify({"status": "Stop signal sent, acknowledgement timeout."}), 202
             except Exception as e:
                   ui_logger.error("Error occurred while trying to stop the bot: %s", e, exc_info=True)
                   return jsonify({"status": "Error signaling stop"}), 500
        elif bot_state.is_running:
             # Bot is marked as running, but core/loop might be missing (error state?)
             ui_logger.warning("Stop requested, but bot state is inconsistent (running=True, but core/loop missing). Forcing state update.")
             bot_state.is_running = False # Try to reset state
             return jsonify({"status": "Bot was in inconsistent running state. State reset."}), 409 # Conflict
        else:
             ui_logger.info("Stop requested, but bot is not running.")
             return jsonify({"status": "Bot is not running"}), 400 # Bad Request


@app.route('/status', methods=['GET'])
def get_status():
    """Returns the current status of the bot and its components."""
    global bot_state
    with bot_state.lock:
        status = {"bot_running": bot_state.is_running}
        core = bot_state.main_core
        if core:
             status["components_initialized"] = {name: comp is not None for name, comp in core.components.items()}
             # Add health status if available
             if hasattr(core, "_component_health"):
                  status["components_health"] = core._component_health
             # Add other relevant status info from core if needed
             # Example: status["mempool_monitor_method"] = getattr(core.components.get("mempoolmonitor"), "_monitor_method", "N/A")
        else:
             status["components_initialized"] = {}
             status["components_health"] = {}

    return jsonify(status), 200


# --- Helper for Sync Calls to Async ---
def run_async_from_sync(coro):
    """Runs an async coroutine from a sync context (Flask route)."""
    global bot_state
    loop = None
    with bot_state.lock: # Access loop safely
         if bot_state.is_running and bot_state.loop and bot_state.loop.is_running():
             loop = bot_state.loop

    if loop:
        try:
            # Schedule coro in the bot's loop and wait for result
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            # Wait for the result with a timeout to avoid blocking Flask thread indefinitely
            return future.result(timeout=5) # 5 second timeout
        except asyncio.TimeoutError:
            ui_logger.error("Timeout waiting for async result in run_async_from_sync.")
            return None
        except Exception as e:
            ui_logger.error("Error running async task from sync context: %s", e, exc_info=True)
            return None
    else:
        # Bot loop not available
        ui_logger.warning("Cannot run async task: Bot is not running or loop is unavailable.")
        return None

# --- Metrics Endpoint ---
# Helper function to safely get attributes or call methods
def _safe_get(obj: Optional[object], attr: str, default: Any = None) -> Any:
    if obj is None: return default
    return getattr(obj, attr, default)

def _safe_get_nested(obj: Optional[object], attrs: List[str], default: Any = None) -> Any:
     current = obj
     for attr in attrs:
          if current is None: return default
          current = getattr(current, attr, None)
     return current if current is not None else default


def get_live_metrics() -> Dict[str, Any]:
    """Fetches live metrics from running bot components."""
    default_metrics = {
        "timestamp": time.time(),
        "strategy_performance": {}, # Per strategy type
        "overall_profit_eth": "0.0",
        "account_balance_eth": "N/A",
        "network_congestion_pct": "N/A",
        "avg_gas_price_gwei": "N/A",
        "mempool_queue_sizes": {},
        "cache_stats": {},
        # Add more metrics as needed
    }
    if not bot_state.is_running or not bot_state.main_core:
        return default_metrics

    core = bot_state.main_core
    metrics = default_metrics.copy()

    try:
        # Get components safely
        safetynet: Optional[SafetyNet] = core.components.get("safetynet")
        strategynet: Optional[StrategyNet] = core.components.get("strategynet")
        mempoolmon: Optional[MempoolMonitor] = core.components.get("mempoolmonitor")
        noncecore: Optional[NonceCore] = core.components.get("noncecore")

        # --- Account Balance (Async) ---
        if safetynet and isinstance(safetynet.account, LocalAccount):
             balance = run_async_from_sync(safetynet.get_balance())
             metrics["account_balance_eth"] = f"{balance:.8f}" if isinstance(balance, Decimal) else "Error"
        else:
             metrics["account_balance_eth"] = "N/A (SafetyNet Error)"


        # --- Strategy Performance ---
        if strategynet:
             overall_profit = Decimal("0")
             perf_data = {}
             for stype, perf_metrics in strategynet.strategy_performance.items():
                 # Use safe attribute access for dataclass
                  perf_data[stype] = {
                      "executions": _safe_get(perf_metrics, "total_executions", 0),
                      "success_rate": f"{_safe_get(perf_metrics, 'success_rate', 0.0):.2%}",
                      "avg_exec_time_ms": f"{_safe_get(perf_metrics, 'avg_execution_time', 0.0) * 1000:.2f}",
                      "total_profit_eth": f"{_safe_get(perf_metrics, 'total_profit', Decimal('0')):.8f}",
                  }
                  overall_profit += _safe_get(perf_metrics, "total_profit", Decimal("0"))
             metrics["strategy_performance"] = perf_data
             metrics["overall_profit_eth"] = f"{overall_profit:.8f}"

        # --- Network & Gas (Async) ---
        if safetynet:
             congestion = run_async_from_sync(safetynet.get_network_congestion()) # Returns float 0-1
             metrics["network_congestion_pct"] = f"{congestion * 100:.2f}%" if isinstance(congestion, float) else "Error"

             gas_price = run_async_from_sync(safetynet.get_dynamic_gas_price()) # Returns Decimal Gwei
             metrics["avg_gas_price_gwei"] = f"{gas_price:.2f}" if isinstance(gas_price, Decimal) else "Error"

        # --- Queue Sizes ---
        if mempoolmon:
             metrics["mempool_queue_sizes"] = {
                 "hash_queue": _safe_get_nested(mempoolmon, ["_tx_hash_queue", "qsize"], -1)(), # Call qsize()
                 "analysis_queue": _safe_get_nested(mempoolmon, ["_tx_analysis_queue", "qsize"], -1)(),
                 "profit_queue": _safe_get_nested(mempoolmon, ["profitable_transactions", "qsize"], -1)(),
             }

        # --- Cache Stats (Example for one cache) ---
        if mempoolmon:
             processed_cache: Optional[TTLCache] = _safe_get(mempoolmon, "processed_transactions")
             if processed_cache:
                  metrics["cache_stats"]["processed_tx"] = {
                      "size": len(processed_cache),
                      "max_size": processed_cache.maxsize,
                      "ttl": processed_cache.ttl,
                  }

        # --- Nonce ---
        if noncecore:
             current_nonce = run_async_from_sync(noncecore.get_nonce())
             pending_tx_count = len(_safe_get(noncecore, "pending_transactions", set()))
             metrics["nonce_info"] = {
                 "current_nonce": current_nonce if isinstance(current_nonce, int) else "Error",
                 "pending_tracked_tx": pending_tx_count,
                 "cache_ttl": _safe_get_nested(noncecore, ["configuration", "NONCE_CACHE_TTL"])
             }


    except Exception as e:
        ui_logger.error("Error gathering live metrics: %s", e, exc_info=True)
        metrics["error"] = f"Failed to gather some metrics: {e}"

    metrics["timestamp"] = time.time() # Update timestamp at the end
    return metrics


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Returns live operational metrics from the bot."""
    # Use a helper to run async metric gathering if needed, or structure metrics gathering to be sync-safe
    live_metrics = get_live_metrics()
    # Use custom JSON encoder if Decimal is used directly
    # return Response(json.dumps(live_metrics, cls=DecimalEncoder), mimetype='application/json')
    return jsonify(live_metrics), 200


@app.route('/components', methods=['GET'])
def get_components_status():
    """Returns the initialization status of components."""
    global bot_state
    with bot_state.lock:
        core = bot_state.main_core
        if core and hasattr(core, "components"):
            # Return True if component object exists, False otherwise
            status = {name: comp is not None for name, comp in core.components.items()}
            return jsonify(status), 200
        else:
            return jsonify({"error": "Bot not running or components not initialized"}), 404

@app.route('/logs', methods=['GET'])
def get_logs():
     """Returns recent buffered logs."""
     # Access deque safely (though append/popleft are thread-safe, iterating might not be)
     # Convert deque to list for JSON serialization
     log_list = list(ws_handler.log_queue)
     return jsonify(log_list)


# --- WebSocket Events ---
@socketio.on('connect')
def handle_connect():
    """Handles new client connections by sending initial logs."""
    ui_logger.info(f"Client connected: {request.sid}")
    # Send recent logs to the newly connected client
    initial_logs = list(ws_handler.log_queue)
    emit('initial_logs', initial_logs)

@socketio.on('disconnect')
def handle_disconnect():
    """Handles client disconnections."""
    ui_logger.info(f"Client disconnected: {request.sid}")

@socketio.on('request_metrics')
def handle_request_metrics():
    """Handles client requests for updated metrics."""
    ui_logger.debug(f"Metrics requested by client: {request.sid}")
    live_metrics = get_live_metrics()
    emit('update_metrics', live_metrics)


# --- Main Execution ---
if __name__ == '__main__':
    ui_logger.info("Starting Flask development server with SocketIO...")
    # Use debug=False for production or when testing multi-threading/processing thoroughly
    # Use allow_unsafe_werkzeug=True only if necessary and understand the risks
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    # Note: Flask's default reloader can cause issues with background threads/loops. Disable it.
