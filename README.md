![on1builder](https://github.com/user-attachments/assets/59e03abe-67ee-4195-9030-63f49c48e46f)

[![Built With Python](https://img.shields.io/badge/Built%20with-Python%203.12-blue?logo=python)](https://www.python.org/)
![GitHub last commit](https://img.shields.io/github/last-commit/John0n1/ON1Builder?display_timestamp=committer&logo=Github&logoColor=%23181717&color=cyan)
[![Ethereum Mainnet](https://img.shields.io/badge/Ethereum-Mainnet-black?logo=ethereum)](https://ethereum.org/)


## Table of Contents

1. [Executive Summary](#1-executive-summary)  
2. [Project Overview](#2-project-overview)  
3. [Architecture and Components](#3-architecture-and-components)  
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
   &nbsp;&nbsp;&nbsp;3.3.8. [ABI Registry & Logging](#abiregistry--loggingconfig)  
   3.4. [User Interface / Dashboard](#user-interface--dashboard)  
4. [Key Features and Statistics](#4-key-features-and-statistics)  
   4.1. [Flashloan and Arbitrage Strategies](#41-flashloan-and-arbitrage-strategies)  
   4.2. [Market Monitoring & Machine Learning](#42-market-monitoring--machine-learning)  
   4.3. [Transaction Execution and Safety](#43-transaction-execution-and-safety)  
   4.4. [Testing and Reliability](#44-testing-and-reliability)  
5. [Project Dependencies and Ecosystem](#5-project-dependencies-and-ecosystem)  
6. [Future Enhancements and Considerations](#6-future-enhancements-and-considerations)  
7. [Installation, Configuration & Usage](#7-installation-configuration--usage)  
   7.1. [Prerequisites](#71-prerequisites)  
   7.2. [Installation Steps](#72-installation-steps)  
   7.3. [Configuration](#73-configuration)  
   7.4. [Usage](#74-usage)  
   7.5. [Contributing](#75-contributing)  
   7.6. [License](#76-license)  
8. [Appendix: Charts and Diagrams](#8-appendix-charts-and-diagrams)
9. [Disclaimer](#9-disclaimer)

---

## 1. Executive Summary

ON1Builder is a versatile Maximal Extractable Value (MEV) flashloan arbitrage bot operating on Ethereum's mainnet and testnets. It integrates:

- **Smart‑contract flashloans** via Aave V3  
- **Automated DEX routing** on Uniswap and Sushiswap  
- **Real‑time market data aggregation** through multiple free‑tier APIs  
- **Machine learning price predictions**  
- **Asynchronous, modular Python core** with robust nonce & safety management  
- **Web‑based dashboard** for monitoring and control  

This end‑to‑end platform empowers traders to deploy atomic arbitrage and sandwich strategies with maximum reliability and minimum configuration overhead.

---

## 2. Project Overview

- **Smart Contracts:**    
  A Solidity contract ('SimpleFlashloan.sol') that manages flashloan execution, request logging, and fallback safety mechanisms.  
  
- **Configuration & Environment:**    
  A robust environment system using a extensive .env file combined with JSON files (for ABI definitions and token mappings) to provide a centralized and validated configuration.  
  
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
  The project uses a .env file to set critical parameters such as gas limits, slippage tolerances, API keys, and network endpoints. The requirements.txt provides a list of all dependencies (e.g., Flask, Web3, scikit‑learn, Pandas).  
  
- **Configuration Module (configuration.py):**    
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
- **apiconfig.py:**    
  Integrates with multiple market data APIs (Binance, CoinGecko, etc.) to fetch both real‑time and historical data. It manages rate‑limiting and caching to ensure data accuracy.  
  
#### MarketMonitor  
- **marketmonitor.py:**    
  Gathers market data continuously, uses a linear regression model to predict price movements, and automates model retraining based on new training data.  
  
#### TransactionCore  
- **transactioncore.py:**    
  Builds, signs, and executes transactions. It supports modern transaction formats (EIP‑1559), simulates transactions to ensure profitability, and implements MEV strategies such as front‑run, back‑run, and sandwich attacks.  
  
#### StrategyNet  
- **strategynet.py:**    
  Uses reinforcement learning to dynamically select between various MEV strategies based on historical performance metrics. It adjusts strategy weights using a reward function that factors in profit, execution time, and risk.  
  
#### NonceCore  
- **noncecore.py:**    
  Manages sequential nonce assignment with a caching mechanism to handle rapid transaction submissions.  
  
#### SafetyNet  
- **safetynet.py:**    
  Validates transaction safety by evaluating gas prices, expected profits (using Decimal calculations), and by adjusting slippage tolerance based on network congestion.  
  
#### MainCore  
- **maincore.py:**    
  Acts as the central orchestrator by initializing all components, maintaining the event loop, monitoring system health and memory, and handling graceful shutdowns.  
  
#### ABIRegistry & LoggingConfig  
- **abiregistry.py:**    
  Loads and validates smart contract ABIs, extracting function signatures for seamless contract interactions.  
- **loggingconfig.py:**    
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

- **Ethereum Connectivity**  
  - [Web3.py](https://github.com/ethereum/web3.py)  
  - [Geth](https://geth.ethereum.org/) for execution layer (IPC)  
  - Prysm for consensus layer (IPC)  

- **Smart Contracts**  
  - [Aave V3 Core](https://github.com/aave/aave-v3-core)  
  - [Uniswap & Sushiswap](https://github.com/Uniswap)  

- **Data & ML**  
  - `scikit-learn`, `pandas`, `numpy`  

- **Web/API**  
  - Flask, Flask‑SocketIO, Flask‑CORS  

- **Testing**  
  - Pytest, unittest  

---

## 6. Future Enhancements and Considerations

- **Advanced ML**: Upgrade to neural nets / time‑series models.  
- **Strategy Refinement**: Add new signals (on‑chain, sentiment).  
- **Scaling**: Parallelize MempoolMonitor & TransactionCore further.  
- **Security Audits**: Third‑party review of smart contracts & core logic.  
- **Dashboard Analytics**: Interactive charting with Plotly or Matplotlib.  
- **Multi‑Chain**: Extend support to other EVM & non‑EVM chains.

---

## 7. Installation, Configuration & Usage

> 🐱‍💻 **Security Warning**  
> Don’t even trust your cats. They're sneaky and *can’t* be trusted.  
>  
> If you paste your private key in plaintext, congrats — you just gave your wallet the same security as a sticky note on a park bench.  
>  
> Keep it encrypted, offline, and far away from prying eyes (or “oops” deployments).  
>  
> Remember: one leak, and your ETH becomes someone else’s exit liquidity.  
>  
> **Not your opsec, not your coins.** 🔐😼


### 7.1 Prerequisites

- **OS**: Linux (Ubuntu 20.04+), Windows 10/11, or macOS 12+  
- **Python**: ≥ 3.12  
- **Hardware**: ≥ 4 cores, 32 (min 16) GB RAM, NVMe SSD  
- **Clients**: Geth & Prysm (IPC mode)
  (NOTE: Syncing requires minimum 1.3TB Storage and can take up to 24hours) 

---

### 7.2 Installation Steps

```bash
git clone https://github.com/John0n1/ON1Builder.git
cd ON1Builder
python3 -m venv venv
source venv/bin/activate       # Linux/macOS
# .\venv\Scripts\activate      # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 7.3 Configuration

Copy and edit the example `.env`:

```bash
cp .env.example .env
nano .env
```

#### Example `.env`

```ini
# API Keys & Endpoints
ETHERSCAN_API_KEY=YOUR_ETHERSCAN_KEY 
INFURA_PROJECT_ID=YOUR_INFURA_PROJECT_ID # Optional
COINGECKO_API_KEY=YOUR_COINGECKO_API_KEY 
COINMARKETCAP_API_KEY=YOUR_CMC_API_KEY 
CRYPTOCOMPARE_API_KEY=YOUR_CC_API_KEY  

# Provider IPC/HTTP/WS
IPC_ENDPOINT=~/ON1Builder/geth.ipc
HTTP_ENDPOINT=http://127.0.0.1:8545
WEBSOCKET_ENDPOINT=wss://127.0.0.1:8545

# Wallet
WALLET_ADDRESS=0xYourEthereumAddress
WALLET_KEY=0xYourPrivateKey 
```

---

### 7.4 Usage

### Running the Bot
```bash
# Start clients
# geth and prysm should already be running as per Section 7.4.1/2

# In project root
source venv/bin/activate    # or venv\Scripts\activate on Windows
python python/main.py
```
- Monitors mempool for MEV opportunities.
- Logs streamed to dashboard at `http://localhost:5000`.

#### 7.4.1 Geth (Execution Client) Setup

Install and run Geth in IPC-only mode:

```bash
# Install Geth on Ubuntu
sudo apt update && sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:ethereum/ethereum
sudo apt update && sudo apt install -y geth              

# Run with snap sync
geth \
  --syncmode snap \                                 
  --mainnet \
  --http=false \
  --ipcpath /ON1Builder/geth.ipc \
  --cache 12000 \
  --maxpeers 80                                        
```

#### 7.4.2 Prysm (Beacon Node) Setup

Install Prysm and run in IPC mode (no JWT):

```bash
# Install Prysm
curl https://raw.githubusercontent.com/prysmaticlabs/prysm/master/prysm.sh \
  --output prysm.sh && chmod +x prysm.sh                 

# Run beacon node, linking to Geth IPC
./prysm.sh beacon-chain \
  --execution-endpoint=/ON1Builder/geth.ipc \
  --mainnet \
  --checkpoint-sync-url=https://beaconstate.info \
  --genesis-beacon-api-url=https://beaconstate.info
```

#### 7.4.3 Free‑Tier API Keys

| Provider         | Free Tier Highlights                                                 |
|------------------|----------------------------------------------------------------------|
| [Infura](https://infura.io)        | 3 million credits/day, 1 API key       |
| [Coingecko](https://coingecko.com)       | Free API key, (trial)            |
| [Etherscan](https://etherscan.io)     | 100 k calls/day, 5 calls/s          |
| [CoinMarketCap](https://coinmarketcap.com) | Free API key, (trial)          |
| [Cryptocompare](https://Cryptocompare.com) | Free API key, (trial)          | 

#### 7.4.4 Flashloan Deployment via Remix

1. **Open Remix** and paste `SimpleFlashloan.sol` (importing Aave V3’s base contract).  
2. **Compile** with Solidity 0.8.10.  
3. **Connect MetaMask** (inject Web3) using a free RPC from one of your providers.  
4. **Deploy** to mainnet by providing the correct `PoolAddressesProvider` address for AAVE V3.  

See QuickNode’s tutorial for full Remix walkthrough and contract code.

---

### 7.5 Contributing

1. Fork & branch  
2. Adhere to PEP8 & add tests  
3. Submit a detailed PR
- (see `CONTRIBUTING`)  

---

### 7.6 License

- **License**: MIT (see `LICENSE`) 

---

## 8. Appendix: Charts and Diagrams

**Component Summary**  

| Component             | Function                                          | Location/Module              |
|-----------------------|---------------------------------------------------|------------------------------|
| Environment Config    | Load & validate `.env`                            | `configuration.py`           |
| Smart Contracts       | Flashloan & DEX interactions                      | `SimpleFlashloan.sol`, `abi/`|
| Market Monitor        | Data ingestion & ML forecasting                   | `marketmonitor.py`           |
| Transaction Engine    | Tx building, signing & execution                  | `transactioncore.py`         |
| Strategy Selector     | RL‑based MEV strategy selection                   | `strategynet.py`             |
| Nonce Manager         | High‑throughput nonce tracking                    | `noncecore.py`               |
| Risk Manager          | Profit & safety checks                            | `safetynet.py`               |
| Orchestrator          | Main event loop & health checks                   | `maincore.py`                |
| Dashboard UI          | Live metrics & control                            | `ui/index.html`              |

**Data Flow Diagram**  

![Data Flow](docs/mermaid1.svg)

[![](https://mermaid.ink/img/pako:eNptk01T4zAMhv-KxwdOhaFNS2gOO5M2LQT6wVBmD-tyMInaekjsjKvsEkr_-yofhXaX5JDIfl5JluQdj0wM3ONrK7MNewqWmtHji_msPchVEoNlC5QWn9n5-Q82EBMjYzY0eqXWuZWojH6uJYMa8IU_CE-XBmI6YVMKk2xPN4bCjyKTa2ShXhmbfuMuEP5DyO6hOJUOBaWgIUKGho1wAxbylM0oRoMNKywQoVaoZKLegTKgf2OVXjdMUDEj0WywKaSZMQk7Y1NpXwFZIFE27Khix7swBo1qVbAHa1YK5UsCbJ5lxmJeRoLtfqlrxbhUfHxhH-xGjH7LJJcIFGIBSZn-AqmIsC6ej0Uzg-xYGBx83lRZ3LbF2BqN54_5oVrNRkcMZPT6_7ojFlLHf1S0YT4iIafbXTFO5HbDqLea-fZFUVLrspI1dduusFA8WMikBfZkpd7K6Khbt50aaSznxOoerNoOK_NOzDNUadmYG7mlgsyMjqhJUlPolIrceL6r6Hs6wAqwmAF67CfY4w6oRGFBDh7V9nCu-88yljJGvf2nnF9QNco2hbhGz07IiRi9QZTjd0ee1BN2OFX55S26RirmHtocWjwFGunS5LuSWXKa0xSW3KPfmCZsyZd6T5pM6l_GpAeZNfl6w72VTLZk5VlMAxIoSRc0_VyVOZpFoaNPDWi6qcPyLnGv2658cm_H38hyri96bqd35bQ7br_b77R4wb0r58K9dJ32lUOv23ev9y3-XiVxeXHtdvv0dC-dfs_p9Tr7vw5iRI0?type=png)](https://mermaid.live/edit#pako:eNptk01T4zAMhv-KxwdOhaFNS2gOO5M2LQT6wVBmD-tyMInaekjsjKvsEkr_-yofhXaX5JDIfl5JluQdj0wM3ONrK7MNewqWmtHji_msPchVEoNlC5QWn9n5-Q82EBMjYzY0eqXWuZWojH6uJYMa8IU_CE-XBmI6YVMKk2xPN4bCjyKTa2ShXhmbfuMuEP5DyO6hOJUOBaWgIUKGho1wAxbylM0oRoMNKywQoVaoZKLegTKgf2OVXjdMUDEj0WywKaSZMQk7Y1NpXwFZIFE27Khix7swBo1qVbAHa1YK5UsCbJ5lxmJeRoLtfqlrxbhUfHxhH-xGjH7LJJcIFGIBSZn-AqmIsC6ej0Uzg-xYGBx83lRZ3LbF2BqN54_5oVrNRkcMZPT6_7ojFlLHf1S0YT4iIafbXTFO5HbDqLea-fZFUVLrspI1dduusFA8WMikBfZkpd7K6Khbt50aaSznxOoerNoOK_NOzDNUadmYG7mlgsyMjqhJUlPolIrceL6r6Hs6wAqwmAF67CfY4w6oRGFBDh7V9nCu-88yljJGvf2nnF9QNco2hbhGz07IiRi9QZTjd0ee1BN2OFX55S26RirmHtocWjwFGunS5LuSWXKa0xSW3KPfmCZsyZd6T5pM6l_GpAeZNfl6w72VTLZk5VlMAxIoSRc0_VyVOZpFoaNPDWi6qcPyLnGv2658cm_H38hyri96bqd35bQ7br_b77R4wb0r58K9dJ32lUOv23ev9y3-XiVxeXHtdvv0dC-dfs_p9Tr7vw5iRI0)

---

**Table 2:Performance Metrics (Dynamic Values)**

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

![ON1Builder Logo](https://github.com/user-attachments/assets/59e03abe-67ee-4195-9030-63f49c48e46f)

## 9. Disclaimer

**Disclaimer**: Use at your own risk—MEV flashloans incur financial risk.
