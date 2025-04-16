![on1builder](https://github.com/user-attachments/assets/59e03abe-67ee-4195-9030-63f49c48e46f)

# ON1Builder Project Documentation & README

[![Python Version](https://img.shields.io/badge/Python-3.12+-blue.svg?logo=python)](https://www.python.org/downloads/release/python-3120/)
![GitHub last commit](https://img.shields.io/github/last-commit/John0n1/ON1Builder?display_timestamp=committer&logo=Github&logoColor=%23181717&color=cyan)
[![License](https://img.shields.io/badge/License-MIT-neon.svg)](LICENSE)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Project Overview](#project-overview)
3. [Architecture and Components](#architecture-and-components)  
   3.1. [Environment & Configuration](#environment--configuration)  
   3.2. [Smart Contracts](#smart-contracts)  
   3.3. [Python Modules](#python-modules)  
   &nbsp;&nbsp;&nbsp;3.3.1. [APIConfig](#apiconfig)  
   &nbsp;&nbsp;&nbsp;3.3.2. [MarketMonitor](#marketmonitor)  
   &nbsp;&nbsp;&nbsp;3.3.3. [TransactionCore](#transactioncore)  
   &nbsp;&nbsp;&nbsp;3.3.4. [StrategyNet](#strategynet)  
   &nbsp;&nbsp;&nbsp;3.3.5. [NonceCore](#noncecore)  
   &nbsp;&nbsp;&nbsp;3.3.6. [SafetyNet](#safetynet)  
   &nbsp;&nbsp;&nbsp;3.3.7. [MainCore](#maincore)  
   &nbsp;&nbsp;&nbsp;3.3.8. [ABI Registry & Logging](#abi-registry--logging)  
   3.4. [User Interface / Dashboard](#user-interface--dashboard)
4. [Key Features and Statistics](#key-features-and-statistics)  
   4.1. [Flashloan and Arbitrage Strategies](#flashloan-and-arbitrage-strategies)  
   4.2. [Market Monitoring & Machine Learning](#market-monitoring--machine-learning)  
   4.3. [Transaction Execution and Safety](#transaction-execution-and-safety)  
   4.4. [Testing and Reliability](#testing-and-reliability)
5. [Project Dependencies and Ecosystem](#project-dependencies-and-ecosystem)
6. [Future Enhancements and Considerations](#future-enhancements-and-considerations)
7. [README: Installation, Configuration & Usage](#readme-installation-configuration--usage)
   7.1. [Prerequisites](#prerequisites)
   7.2. [Installation Steps](#installation-steps)
   7.3. [Configuration](#configuration)
   7.4. [Usage](#usage)
   7.5. [Contributing](#contributing)
   7.6. [License & Disclaimer](#license--disclaimer)
8. [Conclusion](#conclusion)
9. [Appendix: Charts and Diagrams](#appendix-charts-and-diagrams)

---

## 1. Executive Summary

ON1Builder is an advanced MEV (miner extractable value) and flashloan arbitrage trading bot designed for both testnet and mainnet Ethereum environments. The system integrates smart contract‑based flashloan execution (via Aave), decentralized exchange interactions (using Uniswap and Sushiswap), real‑time market monitoring through multiple APIs, and a machine learning module for predicting price movements. An asynchronous architecture ensures that every module—from configuration to nonce management—is robust and efficient. Additionally, the project comes with extensive logging, thorough testing, and a web‑based dashboard for real‑time monitoring.

---

## 2. Project Overview

The project is structured into several key parts:

- **Smart Contracts:**  
  A Solidity contract (SimpleFlashloan.sol) that manages flashloan execution, request logging, and fallback safety mechanisms.

- **Configuration & Environment:**  
  A robust environment system using a production‑ready `.env` file combined with JSON files (for ABI definitions and token mappings) to provide a centralized and validated configuration.

- **Python Application Layer:**  
  Multiple asynchronous modules handle market data fetching, transaction creation and execution, risk management, and a reinforcement‑learning mechanism for selecting optimal MEV strategies.

- **User Interface:**  
  A responsive HTML/CSS/JavaScript dashboard displays real‑time system status, performance metrics, and provides control actions (start/stop).

- **Testing Suite:**  
  A comprehensive set of tests (unit, integration, and end‑to‑end) ensures each module and function operates correctly.

---

## 3. Architecture and Components

### Environment & Configuration

- **.env File & Dependencies:**  
  The project uses a `.env` file to set critical parameters such as gas limits, slippage tolerances, API keys, and network endpoints. The `requirements.txt` provides a list of all dependencies (e.g., Flask, Web3, scikit‑learn, Pandas).

- **Configuration Module (`configuration.py`):**  
  Responsible for loading and validating environment variables, resolving file paths (for ABI, tokens, etc.), and creating required directories for ML model storage and logs.

### Smart Contracts

- **SimpleFlashloan.sol:**  
  Implements Aave‑based flashloan functionality with:
  
  - **fn_RequestFlashLoan:** Iterates over a list of tokens to request a flashloan.
  - **executeOperation:** Executes the flashloan logic and handles exceptions.
  - **Withdrawal Functions:** Allows the owner to withdraw tokens or ETH.

- **ABI Files:**  
  Various JSON files supply contract ABIs for ERC20 tokens and protocols like Uniswap, Sushiswap, and Aave.

### Python Modules

#### APIConfig
- **`apiconfig.py`:**  
  Integrates with multiple market data APIs (Binance, CoinGecko, etc.) to fetch both real‑time and historical data. It manages rate‑limiting and caching to ensure data accuracy.

#### MarketMonitor
- **`marketmonitor.py`:**  
  Gathers market data continuously, uses a linear regression model to predict price movements, and automates model retraining based on new training data.

#### TransactionCore
- **`transactioncore.py`:**  
  Builds, signs, and executes transactions. It supports modern transaction formats (EIP‑1559), simulates transactions to ensure profitability, and implements MEV strategies such as front‑run, back‑run, and sandwich attacks.

#### StrategyNet
- **`strategynet.py`:**  
  Uses reinforcement learning to dynamically select between various MEV strategies based on historical performance metrics. It adjusts strategy weights using a reward function that factors in profit, execution time, and risk.

#### NonceCore
- **`noncecore.py`:**  
  Manages sequential nonce assignment with a caching mechanism to handle rapid transaction submissions.

#### SafetyNet
- **`safetynet.py`:**  
  Validates transaction safety by evaluating gas prices, expected profits (using Decimal calculations), and by adjusting slippage tolerance based on network congestion.

#### MainCore
- **`maincore.py`:**  
  Acts as the central orchestrator by initializing all components, maintaining the event loop, monitoring system health and memory, and handling graceful shutdowns.

#### ABIRegistry & LoggingConfig
- **`abiregistry.py`:**  
  Loads and validates smart contract ABIs, extracting function signatures for seamless contract interactions.
- **`loggingconfig.py`:**  
  Provides a robust logging configuration with colorized output and optional spinner animations for improved readability and debugging.

### User Interface / Dashboard

- **Dashboard (ui/index.html):**  
  A modern web dashboard provides live metrics (transaction success rate, gas usage, account balance, etc.) and real-time logs via Socket.IO, allowing users to monitor and control the bot interactively.

---

## 4. Key Features and Statistics

### 4.1 Flashloan and Arbitrage Strategies

| **Strategy**                 | **Description**                                                     | **Key Parameters**                                           |
|------------------------------|---------------------------------------------------------------------|--------------------------------------------------------------|
| Flashloan Execution          | Requests and executes flashloans using Aave’s protocols             | Flashloan amount, asset, referral codes                      |
| Front-run                    | Preempts high‑value pending transactions by accelerating gas usage  | Gas price multiplier, risk thresholds, predicted price       |
| Aggressive & Predictive FR   | Uses risk scoring and ML forecasts to preemptively execute trades     | Aggressive thresholds; predicted price increase (%)          |
| Back-run & Price Dip BR      | Executes trades following target transactions to profit on a price dip | Price dip threshold, volume metrics                          |
| Sandwich Attack              | Combines flashloan, front‑run, and back‑run to capture sandwich profits | Integrated multi‑step execution with strict profit margins   |

### 4.2 Market Monitoring & Machine Learning

- **Live Data Aggregation:**  
  Collects real‑time prices and volume metrics from multiple APIs with built‑in rate‑limiting and TTL caching.

- **ML Model for Price Prediction:**  
  A linear regression model uses historical price, market cap, and volume data (stored in CSV format) to forecast future price movements.  
  _Example training data features:_

  | Timestamp  | Symbol | Price (USD) | Market Cap      | Volume (24h)   | Volatility | Liquidity Ratio | Price Momentum |
  |------------|--------|-------------|-----------------|----------------|------------|-----------------|----------------|
  | 1737308400 | BTC    | 106124.00   | 2.10e+12        | 7.09e+10       | 0.045      | 0.88            | 0.60           |
  | 1737308400 | ETH    | 3386.52     | 4.08e+11        | 3.23e+10       | 0.038      | 0.82            | 0.65           |

- **Automatic Model Retraining:**  
  The model is retrained every hour (configurable) when sufficient new data becomes available.

### 4.3 Transaction Execution and Safety

- **Dynamic Transaction Building:**  
  Transactions are built for both legacy and EIP‑1559 types, with dynamic gas limit estimation and risk-adjusted gas pricing.

- **Risk Management:**  
  SafetyNet evaluates transactions in real‑time, checking for profitability (after gas costs and slippage) and rejecting transactions that do not meet specified thresholds.

- **Nonce Management:**  
  NonceCore ensures that nonces are accurately assigned and updated to prevent collisions in high-frequency transaction scenarios.

### 4.4 Testing and Reliability

- **Comprehensive Test Suite:**  
  More than 40 tests cover unit, integration, and end‑to‑end scenarios, ensuring high system reliability.

- **Real‑Time Logging & Health Monitoring:**  
  The system continuously monitors component health and memory usage, with logs streamed to the dashboard for live troubleshooting.

---

## 5. Project Dependencies and Ecosystem

- **Blockchain Connectivity:**  
  - [Web3.py](https://github.com/ethereum/web3.py) for Ethereum interactions  
  - Node connectivity via Infura, IPC, or WebSocket

- **Smart Contracts:**  
  - [Aave V3](https://github.com/aave/aave-v3-core) for flashloan functionalities  
  - [Uniswap & Sushiswap](https://github.com/Uniswap) for DEX routing

- **Data & Machine Learning:**  
  - [scikit‑learn](https://scikit-learn.org/) for linear regression  
  - [Pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/) for data processing

- **Web/API Frameworks:**  
  - Flask, Flask‑CORS, and Flask‑SocketIO for backend API and real‑time dashboard

- **Testing Frameworks:**  
  - Pytest and unittest for automated testing

---

## 6. Future Enhancements and Considerations

- **Advanced ML Models:**  
  Upgrade from linear regression to neural networks or time‑series models for more sophisticated price predictions.
  
- **Refined Strategy Selection:**  
  Enhance the reinforcement learning mechanism in StrategyNet with additional market signals and risk assessments.
  
- **Scalability Improvements:**  
  Increase parallel processing capabilities in MempoolMonitor and TransactionCore to handle higher transaction volumes.
  
- **Security Audits:**  
  Conduct thorough third‑party audits for smart contracts and critical code modules.
  
- **Dashboard Enhancements:**  
  Integrate interactive charting (e.g., using Plotly or Matplotlib) and deeper analytics into the web UI.
  
- **Multi‑Chain Support:**  
  Extend the platform to work with multiple blockchain networks beyond Ethereum.

---

## 7. README: Installation, Configuration & Usage

### 7.1 Prerequisites

- **Operating System:**  
  Linux (Ubuntu 20.04+ recommended), Windows 10/11, or macOS 12+.
  
- **Python:**  
  Version 3.12 or higher.
  
- **Ethereum Client:**  
  Geth (or other clients such as Nethermind/Besu) for node connectivity.
  
- **Hardware:**  
  Recommended at least 4 CPU cores, 16GB RAM, and an NVMe SSD.

### 7.2 Installation Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/John0n1/ON1Builder.git
   cd ON1Builder
   ```

2. **Create and Activate a Virtual Environment:**

   ```bash
   # For Linux/macOS:
   python3 -m venv venv
   source venv/bin/activate

   # For Windows:
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**

   - Copy the example file to create your own `.env` file and update values accordingly.

   ```bash
   cp .env.example .env
   # Edit the .env file to include your API keys, wallet, and network endpoints:
   nano .env
   ```

### 7.3 Configuration

The project uses a centralized configuration system:
- **.env File:**  
  Contains all critical settings—gas limits, slippage, API keys, account details, and endpoints.
  
- **JSON Configuration Files:**  
  In the `utils/` and `abi/` directories, these files map token addresses, symbols, and contract ABIs.

For a fully configured production‑ready environment, please refer to the example `.env` file below:

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

### 7.4 Usage

After setting up your environment and configuring the project:

1. **Start the Bot:**

   Activate your virtual environment and run the main script:

   ```bash
   source venv/bin/activate  # For Linux/macOS
   # or on Windows:
   # .\venv\Scripts\activate

   python python/main.py
   ```

   - **Behavior:**  
     The bot will initialize all components (market monitors, transaction engines, strategy selectors, etc.), then begin watching the mempool for arbitrage and MEV opportunities.
   - **Dashboard:**  
     Access the dashboard by visiting `http://<your_server_ip>:5000` in your browser. The dashboard displays system status and real‑time logs via Socket.IO.

2. **Stopping the Bot:**

   Send a SIGINT (Ctrl+C) in the terminal or use the dashboard’s “Stop Bot” button, which will initiate a graceful shutdown.

### 7.5 Contributing

Contributions are welcome. To contribute:

1. Fork the repository.
2. Create a feature branch.
3. Follow [PEP8](https://www.python.org/dev/peps/pep-0008/) guidelines and add tests where applicable.
4. Submit a pull request with a detailed description.

Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) document (if provided) for further guidelines.

### 7.6 License & Disclaimer

- **License:**  
  This project is licensed under the MIT License. Please see the [LICENSE](LICENSE) file for details.

- **Disclaimer:**  
  MEV strategies and flashloan arbitrage involve significant financial risks. Use ON1Builder at your own risk; the developers are not responsible for any losses incurred.

---

## 8. Conclusion

ON1Builder is a comprehensive and innovative solution for automated MEV and flashloan trading on Ethereum. The system integrates smart contract flashloan execution, asynchronous market monitoring, dynamic transaction management, and a reinforcement‑learning strategy selector. Its rigorous configuration management, extensive logging, thorough testing, and responsive dashboard make it a robust tool for navigating the complex world of blockchain trading.

The documentation provided here is designed to serve both as an in‑depth project status report and a user manual. Regular updates and continued enhancements will help maintain the cutting edge of automated trading strategies.

---

## 9. Appendix: Charts and Diagrams

**Table 1: Summary of Main Components**

| **Component**        | **Key Function**                                       | **Location/Module**            |
|----------------------|--------------------------------------------------------|--------------------------------|
| Environment Config   | Load & validate environment variables                | .env, configuration.py         |
| Smart Contracts      | Flashloan and exchange interactions                    | SimpleFlashloan.sol, ABI JSONs |
| Market Monitor       | Data collection and ML price forecasting               | marketmonitor.py               |
| Transaction Engine   | Building, signing, and executing transactions          | transactioncore.py             |
| Strategy Selector    | Reinforcement learning–based strategy selection        | strategynet.py                 |
| Nonce Manager        | Sequence and tracking of transaction nonces            | noncecore.py                   |
| Risk Manager         | Profitability and safety validation                    | safetynet.py                   |
| Orchestrator         | Component initialization and main event loop           | maincore.py                    |
| ABI Registry         | Loading and validating smart contract interfaces       | abiregistry.py                 |
| Dashboard UI         | Real‑time monitoring and control interface             | ui/index.html                  |

**Chart 1: Data Flow Diagram**

---

![flow](docs/mermaid1.svg)

---

[![](https://mermaid.ink/img/pako:eNptk01T4zAMhv-KxwdOhaFNS2gOO5M2LQT6wVBmD-tyMInaekjsjKvsEkr_-yofhXaX5JDIfl5JluQdj0wM3ONrK7MNewqWmtHji_msPchVEoNlC5QWn9n5-Q82EBMjYzY0eqXWuZWojH6uJYMa8IU_CE-XBmI6YVMKk2xPN4bCjyKTa2ShXhmbfuMuEP5DyO6hOJUOBaWgIUKGho1wAxbylM0oRoMNKywQoVaoZKLegTKgf2OVXjdMUDEj0WywKaSZMQk7Y1NpXwFZIFE27Khix7swBo1qVbAHa1YK5UsCbJ5lxmJeRoLtfqlrxbhUfHxhH-xGjH7LJJcIFGIBSZn-AqmIsC6ej0Uzg-xYGBx83lRZ3LbF2BqN54_5oVrNRkcMZPT6_7ojFlLHf1S0YT4iIafbXTFO5HbDqLea-fZFUVLrspI1dduusFA8WMikBfZkpd7K6Khbt50aaSznxOoerNoOK_NOzDNUadmYG7mlgsyMjqhJUlPolIrceL6r6Hs6wAqwmAF67CfY4w6oRGFBDh7V9nCu-88yljJGvf2nnF9QNco2hbhGz07IiRi9QZTjd0ee1BN2OFX55S26RirmHtocWjwFGunS5LuSWXKa0xSW3KPfmCZsyZd6T5pM6l_GpAeZNfl6w72VTLZk5VlMAxIoSRc0_VyVOZpFoaNPDWi6qcPyLnGv2658cm_H38hyri96bqd35bQ7br_b77R4wb0r58K9dJ32lUOv23ev9y3-XiVxeXHtdvv0dC-dfs_p9Tr7vw5iRI0?type=png)](https://mermaid.live/edit#pako:eNptk01T4zAMhv-KxwdOhaFNS2gOO5M2LQT6wVBmD-tyMInaekjsjKvsEkr_-yofhXaX5JDIfl5JluQdj0wM3ONrK7MNewqWmtHji_msPchVEoNlC5QWn9n5-Q82EBMjYzY0eqXWuZWojH6uJYMa8IU_CE-XBmI6YVMKk2xPN4bCjyKTa2ShXhmbfuMuEP5DyO6hOJUOBaWgIUKGho1wAxbylM0oRoMNKywQoVaoZKLegTKgf2OVXjdMUDEj0WywKaSZMQk7Y1NpXwFZIFE27Khix7swBo1qVbAHa1YK5UsCbJ5lxmJeRoLtfqlrxbhUfHxhH-xGjH7LJJcIFGIBSZn-AqmIsC6ej0Uzg-xYGBx83lRZ3LbF2BqN54_5oVrNRkcMZPT6_7ojFlLHf1S0YT4iIafbXTFO5HbDqLea-fZFUVLrspI1dduusFA8WMikBfZkpd7K6Khbt50aaSznxOoerNoOK_NOzDNUadmYG7mlgsyMjqhJUlPolIrceL6r6Hs6wAqwmAF67CfY4w6oRGFBDh7V9nCu-88yljJGvf2nnF9QNco2hbhGz07IiRi9QZTjd0ee1BN2OFX55S26RirmHtocWjwFGunS5LuSWXKa0xSW3KPfmCZsyZd6T5pM6l_GpAeZNfl6w72VTLZk5VlMAxIoSRc0_VyVOZpFoaNPDWi6qcPyLnGv2658cm_H38hyri96bqd35bQ7br_b77R4wb0r58K9dJ32lUOv23ev9y3-XiVxeXHtdvv0dC-dfs_p9Tr7vw5iRI0)

---

**Table 2: Example Performance Metrics (Dynamic Values)**

| **Metric**                   | **Value**             | **Unit**   |
|------------------------------|-----------------------|------------|
| Transaction Success Rate     | 95.6                  | %          |
| Average Execution Time       | 1.35                  | seconds    |
| Profitability                | 0.256                 | ETH        |
| Gas Usage                    | 21000                 | units      |
| Network Congestion           | 45.2                  | %          |
| Slippage                     | 0.1                   | fraction   |
| Account Balance              | 12.5                  | ETH        |
| Transactions Executed        | 185                   | count      |

---

![on1builder](https://github.com/user-attachments/assets/59e03abe-67ee-4195-9030-63f49c48e46f)

---