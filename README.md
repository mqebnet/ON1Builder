# ON1Builder MEV

[![Python Version](https://img.shields.io/badge/Python-3.12+-blue.svg?logo=python)](https://www.python.org/downloads/release/python-3120/)
![GitHub last commit](https://img.shields.io/github/last-commit/John0n1/ON1Builder?display_timestamp=committer&logo=Github&logoColor=%23181717&color=cyan)

[![License](https://img.shields.io/badge/License-MIT-neon.svg)](LICENSE)


![on1builder](https://github.com/user-attachments/assets/59e03abe-67ee-4195-9030-63f49c48e46f)

**ON1Builder** is a powerful, production‑ready framework designed for high‑frequency trading and Maximal Extractable Value (MEV) strategies on decentralized platforms.  
It leverages fully asynchronous Python code, robust logging, and modular design to offer rapid, resilient, and configurable transaction execution.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Architecture](#project-architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Disclaimer](#disclaimer)

---

## Overview

ON1Builder empowers you to exploit MEV opportunities using advanced strategies:
- **Front-Running:** Capitalize on pending transactions.
- **Back-Running:** Execute trades after significant events.
- **Sandwich Attacks:** Profit from transaction ordering.
- **Flash Loan Arbitrage:** Leverage zero‑capital flash loans for profitable trades.

The framework is built with asynchronous Python (asyncio) so that all components—from network communication to complex strategy selection—operate without blocking the event loop.

---

## Key Features

- **Asynchronous Core Modules:** Each component (Configuration, ABIRegistry, NonceCore, SafetyNet, TransactionCore, MarketMonitor, MempoolMonitor, StrategyNet, MainCore) is designed to run asynchronously.
- **Robust Configuration:** A fully centralized and validated configuration system using a production‑ready `.env` file.
- **Smart Contract Interaction:** Uses verified ABIs for ERC20 tokens and decentralized protocols (Uniswap, Sushiswap, Aave).
- **Nonce Management:** Advanced nonce management with caching and retries.
- **Safety Checks:** Dynamic gas pricing, profit estimation with Decimal precision, and adjustable slippage tolerance based on real‑time network congestion.
- **Market Data Integration:** Aggregates live prices from multiple APIs with rate‑limiting, TTL caching, and model‑based predictions.
- **MEV Strategies:** Implements a range of strategies (front‑run, back‑run, sandwich) with a reinforcement‑learning approach to select the best tactic.
- **Extensive Logging & Monitoring:** Detailed logs, memory monitoring, component health checks, and graceful shutdown handling.

[![](https://mermaid.ink/img/pako:eNptk01T4zAMhv-KxwdOhaFNS2gOO5M2LQT6wVBmD-tyMInaekjsjKvsEkr_-yofhXaX5JDIfl5JluQdj0wM3ONrK7MNewqWmtHji_msPchVEoNlC5QWn9n5-Q82EBMjYzY0eqXWuZWojH6uJYMa8IU_CE-XBmI6YVMKk2xPN4bCjyKTa2ShXhmbfuMuEP5DyO6hOJUOBaWgIUKGho1wAxbylM0oRoMNKywQoVaoZKLegTKgf2OVXjdMUDEj0WywKaSZMQk7Y1NpXwFZIFE27Khix7swBo1qVbAHa1YK5UsCbJ5lxmJeRoLtfqlrxbhUfHxhH-xGjH7LJJcIFGIBSZn-AqmIsC6ej0Uzg-xYGBx83lRZ3LbF2BqN54_5oVrNRkcMZPT6_7ojFlLHf1S0YT4iIafbXTFO5HbDqLea-fZFUVLrspI1dduusFA8WMikBfZkpd7K6Khbt50aaSznxOoerNoOK_NOzDNUadmYG7mlgsyMjqhJUlPolIrceL6r6Hs6wAqwmAF67CfY4w6oRGFBDh7V9nCu-88yljJGvf2nnF9QNco2hbhGz07IiRi9QZTjd0ee1BN2OFX55S26RirmHtocWjwFGunS5LuSWXKa0xSW3KPfmCZsyZd6T5pM6l_GpAeZNfl6w72VTLZk5VlMAxIoSRc0_VyVOZpFoaNPDWi6qcPyLnGv2658cm_H38hyri96bqd35bQ7br_b77R4wb0r58K9dJ32lUOv23ev9y3-XiVxeXHtdvv0dC-dfs_p9Tr7vw5iRI0?type=png)](https://mermaid.live/edit#pako:eNptk01T4zAMhv-KxwdOhaFNS2gOO5M2LQT6wVBmD-tyMInaekjsjKvsEkr_-yofhXaX5JDIfl5JluQdj0wM3ONrK7MNewqWmtHji_msPchVEoNlC5QWn9n5-Q82EBMjYzY0eqXWuZWojH6uJYMa8IU_CE-XBmI6YVMKk2xPN4bCjyKTa2ShXhmbfuMuEP5DyO6hOJUOBaWgIUKGho1wAxbylM0oRoMNKywQoVaoZKLegTKgf2OVXjdMUDEj0WywKaSZMQk7Y1NpXwFZIFE27Khix7swBo1qVbAHa1YK5UsCbJ5lxmJeRoLtfqlrxbhUfHxhH-xGjH7LJJcIFGIBSZn-AqmIsC6ej0Uzg-xYGBx83lRZ3LbF2BqN54_5oVrNRkcMZPT6_7ojFlLHf1S0YT4iIafbXTFO5HbDqLea-fZFUVLrspI1dduusFA8WMikBfZkpd7K6Khbt50aaSznxOoerNoOK_NOzDNUadmYG7mlgsyMjqhJUlPolIrceL6r6Hs6wAqwmAF67CfY4w6oRGFBDh7V9nCu-88yljJGvf2nnF9QNco2hbhGz07IiRi9QZTjd0ee1BN2OFX55S26RirmHtocWjwFGunS5LuSWXKa0xSW3KPfmCZsyZd6T5pM6l_GpAeZNfl6w72VTLZk5VlMAxIoSRc0_VyVOZpFoaNPDWi6qcPyLnGv2658cm_H38hyri96bqd35bQ7br_b77R4wb0r58K9dJ32lUOv23ev9y3-XiVxeXHtdvv0dC-dfs_p9Tr7vw5iRI0)


---

## Project Architecture

The project is organized into the following directories and files:

```
/ON1Builder/
├── abi/
│   ├── erc20_abi.json
│   ├── aave_flashloan_abi.json
│   ├── aave_pool_abi.json
│   ├── uniswap_abi.json
│   ├── sushiswap_abi.json
│   ├── erc20_signatures.json
│   └── gas_price_oracle_abi.json
├── contracts/
│   ├── SimpleFlashloan.sol
│   └── IERC20.sol
├── linear_regression/
│   ├── price_model.joblib
│   └── training_data.csv
├── python/
│   ├── __init__.py
│   ├── configuration.py
│   ├── abiregistry.py
│   ├── noncecore.py
│   ├── safetynet.py
│   ├── transactioncore.py
│   ├── apiconfig.py
│   ├── marketmonitor.py
│   ├── mempoolmonitor.py
│   ├── strategynet.py
│   ├── maincore.py
│   ├── main.py
│   └── loggingconfig.py
├── utils/
│   ├── token_addresses.json
│   ├── token_symbols.json
├── tests/
│   └── <multiple test files for modules>
├── .env
├── .gitignore
├── README.md
└── requirements.txt
```

**Core Modules Explanation:**
- **Configuration:** Loads and validates environment variables and file paths.
- **ABIRegistry:** Loads ABIs from JSON files and extracts function selectors.
- **NonceCore:** Manages nonces with caching and retries to prevent transaction clashes.
- **SafetyNet:** Assesses transaction risk, calculates gas costs and slippage, and ensures profit viability.
- **TransactionCore:** Builds, signs, simulates, and executes transactions with robust error handling.
- **APIConfig:** Integrates with multiple third‑party APIs for live market data and historical prices.
- **MarketMonitor:** Continuously monitors market conditions and supports model training for price prediction.
- **MempoolMonitor:** Scans the Ethereum mempool for pending transactions and queues profitable ones.
- **StrategyNet:** Implements a reinforcement‑learning-based system to select and execute the best MEV strategy.
- **MainCore:** Orchestrates all components, runs the event loop, monitors memory and component health, and handles graceful shutdown.
- **main.py:** Entry‑point for running the bot.

---

## Prerequisites

- **Operating System:** Linux (Ubuntu 20.04+ recommended), Windows 10/11, macOS 12+
- **Python:** Version 3.12 or higher is required.
- **Ethereum Client:** Geth (or alternative clients like Nethermind/Besu) is recommended.
- **Dependencies:** See `requirements.txt`—all necessary libraries are listed there.
- **Hardware:** At least 4 CPU cores, 16GB RAM, and NVMe SSD storage is recommended for node synchronization.

---

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/John0n1/ON1Builder.git
   cd ON1Builder
   ```

2. **Create and Activate a Virtual Environment:**

   ```bash
   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate

   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**

   - Copy the provided `.env.example` (or use the provided full example) to a file named `.env` in the project root and update each key with real values.

   ```bash
   cp .env.example .env
   # Edit .env with your favorite editor:
   nano .env
   ```

---

## Configuration

The project uses a central `.env` file (see [the example below](#example-env-file)) along with JSON configuration files in the `utils/` directory. These files define:

- Network and trading parameters.
- API keys and endpoints.
- Ethereum account information.
- Paths to ABI files and ML model resources.

> **Note:** Ensure all file paths in the `.env` file match your project structure.

---

## Usage

### Starting the Bot

Ensure that:
- Your Ethereum node is up and running.
- The `.env` file is correctly set.
- All JSON configuration files are present and formatted correctly.

Activate the virtual environment and start the bot:

```bash
source venv/bin/activate  # Linux/macOS
# or on Windows:
# .\venv\Scripts\activate

python python/main.py
```

### Bot Behavior

- **Mempool Monitoring:**  
  Continuously scans the mempool for transactions that meet your criteria.

- **Strategy Execution:**  
  Uses various MEV strategies (front‑run, back‑run, sandwich) to execute profitable trades.

- **Logging:**  
  Detailed logs are written to both the console and a log file (`python/ON1Builder.log`).

- **Graceful Shutdown:**  
  The bot listens for shutdown signals (SIGINT/SIGTERM) to safely cancel tasks and disconnect from the Ethereum node.

---

## Example .env File

Below is an example of a full, production‑ready `.env` file covering all required keys (update values as needed):

```ini
# --------------------- General Settings ---------------------
MAX_GAS_PRICE=100000000000
GAS_LIMIT=1000000
MAX_SLIPPAGE=0.01
MIN_PROFIT=0.001
MIN_BALANCE=0.000001
MEMORY_CHECK_INTERVAL=300
COMPONENT_HEALTH_CHECK_INTERVAL=60
PROFITABLE_TX_PROCESS_TIMEOUT=1.0

# --------------------- Standard Addresses ---------------------
WETH_ADDRESS=0xC02aaa39b223FE8D0a0e5C4F27eAD9083C756Cc2
USDC_ADDRESS=0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48
USDT_ADDRESS=0xdAC17F958D2ee523a2206206994597C13D831ec7

# --------------------- API Keys and Endpoints ---------------------
ETHERSCAN_API_KEY=ABCDEF1234567890ABCDEF1234567890
INFURA_PROJECT_ID=infura_project_1234567890
INFURA_API_KEY=infura_api_key_1234567890
COINGECKO_API_KEY=coingecko_api_key_123456
COINMARKETCAP_API_KEY=coinmarketcap_api_key_123456
CRYPTOCOMPARE_API_KEY=cryptocompare_api_key_123456
HTTP_ENDPOINT=http://127.0.0.1:8545
WEBSOCKET_ENDPOINT=ws://127.0.0.1:8546
IPC_ENDPOINT=/path/to/geth.ipc

# --------------------- Account Configuration ---------------------
WALLET_ADDRESS=0xYourEthereumAddress000000000000000000000000
WALLET_KEY=0xYourPrivateKeyABCDEF1234567890ABCDEF1234567890ABCDEF1234567890ABCDEF1234

# --------------------- File Paths ---------------------
ERC20_ABI=abi/erc20_abi.json
AAVE_FLASHLOAN_ABI=abi/aave_flashloan_abi.json
AAVE_POOL_ABI=abi/aave_pool_abi.json
UNISWAP_ABI=abi/uniswap_abi.json
SUSHISWAP_ABI=abi/sushiswap_abi.json
ERC20_SIGNATURES=abi/erc20_signatures.json
TOKEN_ADDRESSES=utils/token_addresses.json
TOKEN_SYMBOLS=utils/token_symbols.json
GAS_PRICE_ORACLE_ABI=abi/gas_price_oracle_abi.json

# --------------------- Router Addresses ---------------------
UNISWAP_ADDRESS=0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D
SUSHISWAP_ADDRESS=0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F
AAVE_POOL_ADDRESS=0xb53c1a33016b2dc2ff3653530bff1848a515c8c5
AAVE_FLASHLOAN_ADDRESS=0xYourAaveFlashloanAddress000000000000000000
GAS_PRICE_ORACLE_ADDRESS=0xYourGasPriceOracleAddress00000000000000000

# --------------------- Slippage and Gas Configuration ---------------------
SLIPPAGE_DEFAULT=0.1
MIN_SLIPPAGE=0.01
MAX_SLIPPAGE=0.5
SLIPPAGE_HIGH_CONGESTION=0.05
SLIPPAGE_LOW_CONGESTION=0.2
MAX_GAS_PRICE_GWEI=500
MIN_PROFIT_MULTIPLIER=2.0
BASE_GAS_LIMIT=21000
DEFAULT_CANCEL_GAS_PRICE_GWEI=60
ETH_TX_GAS_PRICE_MULTIPLIER=1.2

# --------------------- ML Model Configuration ---------------------
MODEL_RETRAINING_INTERVAL=3600
MIN_TRAINING_SAMPLES=100
MODEL_ACCURACY_THRESHOLD=0.7
PREDICTION_CACHE_TTL=300
LINEAR_REGRESSION_PATH=linear_regression
MODEL_PATH=linear_regression/price_model.joblib
TRAINING_DATA_PATH=linear_regression/training_data.csv

# --------------------- Mempool Monitor Configuration ---------------------
MEMPOOL_MAX_RETRIES=3
MEMPOOL_RETRY_DELAY=2
MEMPOOL_BATCH_SIZE=10
MEMPOOL_MAX_PARALLEL_TASKS=5

# --------------------- Nonce Core Configuration ---------------------
NONCE_CACHE_TTL=60
NONCE_RETRY_DELAY=1
NONCE_MAX_RETRIES=5
NONCE_TRANSACTION_TIMEOUT=120

# --------------------- Safety Net Configuration ---------------------
SAFETYNET_CACHE_TTL=300
SAFETYNET_GAS_PRICE_TTL=30

# --------------------- Strategy Net Configuration ---------------------
AGGRESSIVE_FRONT_RUN_MIN_VALUE_ETH=0.1
AGGRESSIVE_FRONT_RUN_RISK_SCORE_THRESHOLD=0.7
FRONT_RUN_OPPORTUNITY_SCORE_THRESHOLD=75
VOLATILITY_FRONT_RUN_SCORE_THRESHOLD=75
ADVANCED_FRONT_RUN_RISK_SCORE_THRESHOLD=75
PRICE_DIP_BACK_RUN_THRESHOLD=0.99
FLASHLOAN_BACK_RUN_PROFIT_PERCENTAGE=0.02
HIGH_VOLUME_BACK_RUN_DEFAULT_THRESHOLD_USD=100000
SANDWICH_ATTACK_GAS_PRICE_THRESHOLD_GWEI=200
PRICE_BOOST_SANDWICH_MOMENTUM_THRESHOLD=0.02

# --------------------- Mempool High Value Transaction Monitoring ---------------------
HIGH_VALUE_THRESHOLD=1000000000000000000
```

---

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Clone your fork and create a feature branch.
3. Follow PEP8 style guidelines and write tests for any new features or bug fixes.
4. Submit a pull request with a detailed description of your changes.

See our [CONTRIBUTING.md](CONTRIBUTING.md) for further instructions.

---

## License

This project is licensed under the [MIT License](LICENSE). See the LICENSE file for full details.

---

## Disclaimer

Trading and MEV strategies involve significant risks. Use ON1Builder at your own risk. The developers are not responsible for any financial losses incurred from using this software.

---

Happy Trading!  