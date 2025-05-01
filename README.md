# ON1Builder

![ON1Builder Logo](https://github.com/user-attachments/assets/59e03abe-67ee-4195-9030-63f49c48e46f)

[![Built with Python 3.12](https://img.shields.io/badge/Built%20with-Python%203.12-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![ON1Builder Wiki](https://img.shields.io/badge/ON1Builder-Wiki-blue?logo=GitHub&logoColor=white)](https://github.com/John0n1/ON1Builder/wiki)

[![Last Commit](https://img.shields.io/github/last-commit/John0n1/ON1Builder?display_timestamp=committer&logo=GitHub&color=white)](https://github.com/John0n1/ON1Builder/commits/main)


**ON1Builder** is an advanced Ethereum MEV (Maximal Extractable Value) orchestrator. It continuously monitors the mempool, evaluates pending transactions, and automatically executes profitable strategies including flashloan arbitrage, front-running, back-running, and sandwich attacks. Built with modularity, resilience, and performance in mind.

---

## Table of Contents

1. [Key Features](#key-features)  
2. [Architecture Overview](#architecture-overview)  
3. [Prerequisites](#prerequisites)  
4. [Geth & Prysm Setup](#geth--prysm-setup)  
5. [Installation](#installation)  
6. [Configuration](#configuration)  
7. [Running ON1Builder](#running-on1builder)  
8. [Monitoring & Metrics](#monitoring--metrics)  
9. [Component Reference](#component-reference)  
10. [Security Best Practices](#security-best-practices)  
11. [Troubleshooting](#troubleshooting)  
12. [Extending & Custom Strategies](#extending--custom-strategies)  
13. [Roadmap](#roadmap)  
14. [License](#license)  

---

## Key Features

- **Mempool Surveillance**  
  - Filter- or poll-based detection of pending transactions  
  - Priority queuing based on gas price and custom heuristics  
- **MEV Strategy Suite**  
  - **Flashloan Arbitrage** via Aave v3  
  - **Front-Running**: basic, aggressive, predictive, volatility-based  
  - **Back-Running**: price-dip, flashloan-enabled, high-volume  
  - **Sandwich Attacks**: atomic sandwich orchestration  
- **Reinforcement Learning**  
  - Softmax-based selection with exploration/exploitation  
  - Continuous Q-learning weight updates from live execution metrics  
- **Robust Nonce Management**  
  - Caching with TTL, lock-protected refresh, pending-tx tracking  
  - Automatic recovery from out-of-sync nonces  
- **Dynamic SafetyNet**  
  - Real-time gas/slippage tuning, profit verification, network congestion assessment  
- **MarketMonitor ML Pipeline**  
  - Historical price ingestion, EMA features, linear regression model for price prediction  
- **High-Availability Core**  
  - Async TaskGroup orchestration, memory leak detection, component health checks  

---

## Architecture Overview

```
+-----------------+     +-----------------+     +----------------------+     +----------------------+
|  MempoolMonitor | --> |  StrategyNet    | --> |  TransactionCore     | --> |      Ethereum        |
| (filter/poll)   |     | (RL selection)  |     | (build & execute tx) |     |    (nodes & DeXs)    |
+-----------------+     +-----------------+     +----------------------+     +----------------------+
       ↓                        ↑                          ↑                            ↑
+--------------+          +------------+             +-------------+             +-------------+
| SafetyNet    |          | MarketMon. |             | NonceCore   |             | APIConfig   |
| (gas/slip)   |          | (prices)   |             | (nonce mgmt)|             | (data fetch)|
+--------------+          +------------+             +-------------+             +-------------+
```

- **MainCore** bootstraps config, providers, components, and supervises tasks.  
- Components communicate via typed async queues and share configuration objects.  
- Graceful shutdown on SIGINT/SIGTERM; emergency exit support.

---

## Prerequisites

- **OS**: Linux (Ubuntu 20.04+) or macOS  
- **Python**: 3.10+ with `venv` support  
- **Ethereum Clients**: Geth (for execution) & Prysm (for consensus if running your own beacon chain)  
- **API Keys**: Etherscan, Infura, CoinGecko, CoinMarketCap, CryptoCompare  

---

## Geth & Prysm Setup

### Geth (Execution Client)

1. **Install**  
   ```bash
   sudo apt-get update && sudo apt-get install -y software-properties-common
   sudo add-apt-repository -y ppa:ethereum/ethereum
   sudo apt-get update && sudo apt-get install -y geth
   ```
2. **Configure**  
   - Create data directory: `mkdir -p ~/ethereum/mainnet`  
   - `geth --http --http.addr 0.0.0.0 --http.api eth,net,web3 --datadir ~/ethereum/mainnet`
3. **Sync**  
   ```bash
   geth --syncmode "snap" --http --http.corsdomain "*" --http.api eth,net,web3,txpool,debug --datadir ~/ethereum/mainnet
   ```

### Prysm (Consensus Client)

1. **Install**  
   ```bash
   curl https://raw.githubusercontent.com/prysmaticlabs/prysm/master/prysm.sh | bash -s -- --auto-install
   ```
2. **Run Beacon Node**  
   ```bash
   prysm/beacon-chain.sh --datadir ~/ethereum/beacon \
     --http-web3provider=http://127.0.0.1:8545 \
     --monitoring-host=0.0.0.0 --monitoring-port=8080
   ```
3. **Run Validator (optional)**  
   ```bash
   prysm/validator.sh --datadir ~/ethereum/validator --beacon-rpc-provider=127.0.0.1:4000
   ```

---

## Installation

1. **Clone & Enter Directory**  
   ```bash
   git clone https://github.com/John0n1/ON1Builder.git
   cd ON1Builder
   ```
2. **Python Environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Dependencies**  
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Configuration

1. **Copy Template**  
   ```bash
   cp template.env .env
   ```
2. **Edit `.env`**  
   - RPC endpoints (`HTTP_ENDPOINT`, `WEBSOCKET_ENDPOINT`, `IPC_ENDPOINT`)  
   - Wallet (`WALLET_ADDRESS`, `WALLET_KEY`)  
   - API keys (`ETHERSCAN_API_KEY`, `INFURA_API_KEY`, etc.)  
   - Paths to ABIs and JSON utils  

3. **Verify JSON Resources**  
   - `utils/token_addresses.json`  
   - `utils/token_symbols.json`  
   - `abi/*.json`  

---

## Running ON1Builder

- **Start**  
  ```bash
  python3 main.py
  ```
- **Stop**  
  - Ctrl+C or kill the process  
- **Flask UI & Metrics** (optional)  
  ```bash
  python3 app.py
  # Visit: http://localhost:5000
  ```  
  - `/metrics` JSON endpoint  
  - WebSocket logs on UI  

---

## Monitoring & Metrics

- **Built-in HTTP server** via `FlaskUI`  
  - **GET** `/status` – bot & component health  
  - **GET** `/metrics` – real-time performance:  
    - transaction success rate  
    - avg execution time  
    - profitability  
    - gas usage & slippage  
    - mempool congestion  
- **Console Logs**: colored, leveled via `loggingconfig.py`  
- **Memory Watchdog**: tracemalloc diffs every `MEMORY_CHECK_INTERVAL`  

---

## Component Reference

| Component         | Responsibility                                               |
|-------------------|--------------------------------------------------------------|
| **Configuration** | Load env vars, resolve paths, validate keys & addresses      |
| **APIConfig**     | Multi-API price/volume fetch with rate-limiting & caching    |
| **SafetyNet**     | Profit/gas/slippage checks, network congestion monitoring    |
| **MarketMonitor** | Historical data ingestion, ML model training & prediction    |
| **NonceCore**     | Nonce caching, locked refresh, pending tx tracking           |
| **MempoolMonitor**| Pending tx capture, priority queuing, basic profitability   |
| **StrategyNet**   | Reinforcement learning strategy selection & weight updates   |
| **TransactionCore**| Build, sign, simulate, send transactions with retries       |
| **MainCore**      | Orchestrates init, run loop, health & memory monitoring      |
| **FlaskUI**       | Optional HTTP & WebSocket interface for logs & metrics       |

---

## Security Best Practices

- **Never commit `.env`** or private keys to VCS  
- **Use secure key management** (vault, HSM) in production  
- **Run as non-root** user with minimal permissions  
- **Monitor gas price spikes** and adjust `MAX_GAS_PRICE_GWEI`  
- **Rate-limit flashloan usage** to avoid attacking yourself  
- **Audit smart contracts** and ABI versions carefully  

---

## Troubleshooting

- **Connection errors**  
  - Check RPC URLs, firewall rules, peer sync status  
- **Nonce mismatches**  
  - Inspect logs from NonceCore; manually reset with `--reset-nonce` flag  
- **Strategy starvation**  
  - Tune `exploration_rate`, weight decay, threshold values  
- **Model training failures**  
  - Ensure `training_data.csv` has ≥ `MIN_TRAINING_SAMPLES`; check CSV format  

---

## Extending & Custom Strategies

1. **Add new methods** in `transactioncore.py` following existing signatures  
2. **Register** them in `StrategyNet._strategy_registry`  
3. **Tune** hyperparameters in `StrategyConfiguration`  
4. **Rebuild & redeploy**  

---

## Roadmap

- **Cross-chain MEV**: support BSC, Polygon, Arbitrum  
- **Advanced ML**: switch to neural-network price predictors  
- **Dashboard Integration**: Grafana + Prometheus metrics  
- **Automated Liquidity Provision** & limit-order MEV  

---

## License

This project is released under the **MIT License**.  
See [LICENSE](LICENSE) for full text.

