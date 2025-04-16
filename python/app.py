from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import threading
import asyncio
import time
import queue
import logging

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))
from maincore import MainCore
from configuration import Configuration
from loggingconfig import setup_logging

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

logger = setup_logging("FlaskUI", level=20)

bot_running = False
bot_thread = None
bot_loop = None
main_core = None
main_core_lock = threading.Lock()

class WebSocketLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_queue = queue.Queue()
        self.current_level = logging.INFO

    def emit(self, record):
        try:
            log_entry = {
                'level': record.levelname,
                'message': self.format(record),
                'timestamp': time.strftime('%H:%M:%S', time.localtime(record.created))
            }
            self.log_queue.put(log_entry)
            socketio.emit('log_message', log_entry)
        except Exception:
            self.handleError(record)

# Initialize WebSocket logging
ws_handler = WebSocketLogHandler()
ws_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(ws_handler)

def run_bot_in_thread():
    global bot_running, bot_loop, main_core
    asyncio.set_event_loop(asyncio.new_event_loop())
    bot_loop = asyncio.get_event_loop()
    configuration = Configuration()
    main_core = MainCore(configuration)
    try:
        bot_loop.run_until_complete(main_core.initialize_components())
        bot_loop.run_until_complete(main_core.run())
    except Exception as e:
        logger.error(f"Bot error: {e}")
    finally:
        bot_running = False

@app.route('/')
def index():
    # Serve the UI index.html file
    ui_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ui'))
    return send_from_directory(ui_dir, 'index.html')

@app.route('/start', methods=['POST'])
def start_bot():
    global bot_running, bot_thread
    if not bot_running:
        bot_running = True
        bot_thread = threading.Thread(target=run_bot_in_thread, daemon=True)
        bot_thread.start()
        return jsonify({"status": "Bot started"}), 200
    else:
        return jsonify({"status": "Bot is already running"}), 400

@app.route('/stop', methods=['POST'])
def stop_bot():
    global bot_running, main_core, bot_loop
    if bot_running:
        bot_running = False
        if main_core and bot_loop:
            try:
                bot_loop.call_soon_threadsafe(asyncio.create_task, main_core.stop())
            except Exception as e:
                logger.error(f"Error stopping bot: {e}")
        return jsonify({"status": "Bot stopped"}), 200
    else:
        return jsonify({"status": "Bot is not running"}), 400

@app.route('/status', methods=['GET'])
def get_status():
    global bot_running, main_core
    status = {"running": bot_running}
    if main_core:
        status["components"] = {k: v is not None for k, v in getattr(main_core, "components", {}).items()}
    return jsonify(status), 200

def get_metrics_sync(coro):
    global bot_loop
    if bot_loop and bot_loop.is_running():
        fut = asyncio.run_coroutine_threadsafe(coro, bot_loop)
        try:
            return fut.result(timeout=5)
        except Exception as e:
            logger.error(f"Error fetching metrics: {e}")
            return None
    return None

def get_real_metrics():
    global main_core
    default_metrics = {
        "transaction_success_rate": 0.0,
        "average_execution_time": 0.0,
        "profitability": 0.0,
        "gas_usage": 0,
        "network_congestion": 0.0,
        "slippage": 0.0,
        "balance": 0.0,
        "number_of_transactions": 0
    }
    if not main_core or not hasattr(main_core, "components"):
        return default_metrics
    try:
        comps = main_core.components
        tc = comps.get("transactioncore")
        sn = comps.get("safetynet")
        stn = comps.get("strategynet")
        # Fetch balance asynchronously
        balance = 0.0
        if sn and hasattr(sn, "account"):
            balance = float(get_metrics_sync(sn.get_balance(sn.account)) or 0.0)
        # Fetch network congestion asynchronously
        network_congestion = 0.0
        if sn:
            network_congestion = float(get_metrics_sync(sn.get_network_congestion()) or 0.0)
        # Strategy performance
        strategy_perf = None
        if stn and hasattr(stn, "strategy_performance"):
            perf = stn.strategy_performance.get("front_run")
            if perf:
                strategy_perf = perf
        metrics = {
            "transaction_success_rate": getattr(strategy_perf, "success_rate", 0.0) if strategy_perf else 0.0,
            "average_execution_time": getattr(strategy_perf, "avg_execution_time", 0.0) if strategy_perf else 0.0,
            "profitability": float(getattr(strategy_perf, "profit", 0.0)) if strategy_perf else 0.0,
            "gas_usage": getattr(tc, "DEFAULT_GAS_LIMIT", 0) if tc else 0,
            "network_congestion": network_congestion,
            "slippage": sn.SLIPPAGE_CONFIG["default"] if sn and hasattr(sn, "SLIPPAGE_CONFIG") else 0.0,
            "balance": balance,
            "number_of_transactions": getattr(strategy_perf, "total_executions", 0) if strategy_perf else 0
        }
        return metrics
    except Exception as e:
        logger.error(f"Error fetching real metrics: {e}")
        return default_metrics

@app.route('/metrics', methods=['GET'])
def get_metrics():
    metrics = get_real_metrics()
    return jsonify(metrics), 200

@app.route('/components', methods=['GET'])
def get_components():
    global main_core
    if not main_core or not hasattr(main_core, "components"):
        return jsonify({"error": "Bot not running"}), 400
    status = {k: v is not None for k, v in main_core.components.items()}
    return jsonify(status), 200

@app.route('/set_log_level', methods=['POST'])
def set_log_level():
    level = request.json.get('level', 'INFO')
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO
    }
    if level in level_map:
        ws_handler.current_level = level_map[level]
        logging.getLogger().setLevel(level_map[level])
        return jsonify({"status": "Log level updated", "level": level}), 200
    return jsonify({"error": "Invalid log level"}), 400

@socketio.on('connect')
def handle_connect():
    recent_logs = list(ws_handler.log_queue.queue)[-100:]  # Get last 100 logs
    emit('initial_logs', recent_logs)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
