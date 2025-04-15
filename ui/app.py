from flask import Flask, jsonify, request
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

bot_running = False

@app.route('/start', methods=['POST'])
def start_bot():
    global bot_running
    if not bot_running:
        # Logic to start the bot
        bot_running = True
        return jsonify({"status": "Bot started"}), 200
    else:
        return jsonify({"status": "Bot is already running"}), 400

@app.route('/stop', methods=['POST'])
def stop_bot():
    global bot_running
    if bot_running:
        # Logic to stop the bot
        bot_running = False
        return jsonify({"status": "Bot stopped"}), 200
    else:
        return jsonify({"status": "Bot is not running"}), 400

@app.route('/metrics', methods=['GET'])
def get_metrics():
    # Logic to fetch and return performance metrics
    metrics = {
        "transaction_success_rate": 95.0,
        "average_execution_time": 0.5,
        "profitability": 1000.0,
        "gas_usage": 21000,
        "network_congestion": 0.2,
        "slippage": 0.01,
        "balance": 10.0,
        "number_of_transactions": 100
    }
    return jsonify(metrics), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
