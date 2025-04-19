![on1builder](https://github.com/user-attachments/assets/59e03abe-67ee-4195-9030-63f49c48e46f)

[![Python Version](https://img.shields.io/badge/Python-3.12+-blue.svg?logo=python)](https://www.python.org/downloads/release/python-3120/)
![GitHub last commit](https://img.shields.io/github/last-commit/John0n1/ON1Builder?display_timestamp=committer&logo=Github&logoColor=%23181717&color=cyan)
[![License](https://img.shields.io/badge/License-MIT-neon.svg)](LICENSE)

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Project Overview](#project-overview)
3. [Architecture & Components](#architecture--components)
   - [Environment & Configuration](#environment--configuration)
   - [Smart Contracts](#smart-contracts)
   - [Python Modules](#python-modules)
   - [Dashboard UI](#dashboard-ui)
4. [Key Features](#key-features)
5. [Dependencies & Ecosystem](#dependencies--ecosystem)
6. [Ethereum Client Setup](#ethereum-client-setup)
   - [Geth (Execution Client)](#geth-execution-client)
   - [Prysm (Consensus/Beacon)](#prysm-consensusbeacon)
7. [API Keys & Environment](#api-keys--environment)
8. [Usage & Deployment](#usage--deployment)
   - [Running the Bot](#running-the-bot)
   - [Flashloan via Remix](#flashloan-via-remix)
9. [Future Enhancements](#future-enhancements)
10. [License & Disclaimer](#license--disclaimer)
11. [Appendix](#appendix)

---

## Executive Summary
ON1Builder is a comprehensive MEV and flashloan arbitrage framework for Ethereum, combining fast client sync (Geth), beacon-chain consensus (Prysm), modular Python services, Aave V3 flashloans, and real-time ML-driven strategy selection. It offers end-to-end configuration, robust logging, and a live dashboard for monitoring and control.

---

## Project Overview
- **Smart Contracts**: A `SimpleFlashloan.sol` contract interfaces with Aave V3 to request and repay flashloans, with built-in safety fallbacks.
- **Python Core**: Asynchronous modules for market data, transaction crafting, nonce management, safety checks, and reinforcement-learning strategy selection.
- **Node Infrastructure**: Local Geth execution client (snap sync) and Prysm beacon node communicating exclusively via IPC.
- **Dashboard**: A Flask + Socket.IO web UI for real-time insights and control.

---

## Architecture & Components

### Environment & Configuration
- **.env File**: Centralized storage of API keys, endpoints, gas/slippage settings, and token mappings.
- **Configuration Module**: Validates and loads environment variables, resolves file paths, and sets up logging and ML folders.

### Smart Contracts
- **SimpleFlashloan.sol**:
  - `requestFlashLoan(...)` to initiate multi-asset flashloans.
  - `executeOperation(...)` to execute arbitrage logic and repay the loan.
  - Emergency withdrawal functions for ETH/tokens.

### Python Modules
- **APIConfig**: Manages external APIs (Infura, Alchemy, Etherscan, Coingecko, etc.).
- **MarketMonitor**: Streams price/volume data, runs hourly regression retraining.
- **TransactionCore**: Constructs, simulates, and broadcasts EIP-1559 transactions.
- **StrategyNet**: Uses RL to optimize front-run, back-run, and sandwich strategies.
- **NonceCore**: Caches and sequentially issues nonces to avoid conflicts.
- **SafetyNet**: Validates profit/gas/slippage thresholds before sending.
- **MainCore**: Orchestrates event loop, health checks, and graceful shutdown.
- **ABIRegistry & LoggingConfig**: Loads ABIs for contracts and configures structured, colorized logs.

### Dashboard UI
- **Web Interface**: Live metrics (success rate, gas usage, account balance), log streaming, start/stop controls, built with Flask and Socket.IO.

---

## Key Features
- **Aave V3 Flashloans**: Multi-token support, immediate arbitrage, fallback safety.
- **MEV Strategies**: Front-run, back-run, sandwich, and composite strategies with ML-driven parameters.
- **Real-time ML**: Regression model retrained hourly for price momentum predictions.
- **Async Design**: Non-blocking modules ensuring high throughput and minimal latency.
- **Secure Node Setup**: Snap sync Geth and Prysm connected via IPC only.
- **Comprehensive Testing**: 40+ unit, integration, and end-to-end tests guarantee reliability.

---

## Dependencies & Ecosystem
- **Blockchain**: Web3.py, Aave V3 Core, Uniswap/Sushiswap ABIs.
- **ML & Data**: scikit-learn, Pandas, NumPy.
- **Backend**: Flask, Flask-SocketIO.
- **Testing**: Pytest, Unittest.

---

## Ethereum Client Setup

### Geth (Execution Client)
Run locally with fast snap sync and IPC:
```bash
geth \
  --syncmode snap \
  --mainnet \
  --http --http.api eth,net,engine,admin,web3,txpool \
  --cache 12000 \
  --ipcpath ~/ON1Builder/geth.ipc \
  --maxpeers 80 \
  --datadir ~/.ethereum
```
- Snap sync for quick bootstrapping.
- HTTP APIs for local tooling; IPC for external clients.
- Increased cache and peers for performance.

### Prysm (Consensus/Beacon)
Point to Geth’s IPC socket:
```bash
cd ~/ON1Builder
./prysm.sh beacon-chain \
  --execution-endpoint=~/ON1Builder/geth.ipc \
  --mainnet \
  --checkpoint-sync-url=https://beaconstate.info \
  --genesis-beacon-api-url=https://beaconstate.info
```
- Exclusive IPC-based EL<->CL communication.
- Checkpoint sync accelerates chain validation.

---

## API Keys & Environment
Populate `.env` with your own credentials and endpoints:
```ini
INFURA_PROJECT_ID=YOUR_INFURA_PROJECT_ID
ALCHEMY_API_KEY=YOUR_ALCHEMY_API_KEY
ETHERSCAN_API_KEY=YOUR_ETHERSCAN_API_KEY
COINGECKO_API_KEY=YOUR_COINGECKO_API_KEY
COINMARKETCAP_API_KEY=YOUR_COINMARKETCAP_API_KEY
CRYPTOCOMPARE_API_KEY=YOUR_CRYPTOCOMPARE_API_KEY
HTTP_ENDPOINT=http://127.0.0.1:8545
WEBSOCKET_ENDPOINT=ws://127.0.0.1:8546
```

---

## Usage & Deployment

### Running the Bot
```bash
# Start clients
# geth and prysm should already be running as per Section 6

# In project root
source venv/bin/activate    # or venv\Scripts\activate on Windows
python python/main.py
```
- Monitors mempool for MEV opportunities.
- Logs streamed to dashboard at `http://localhost:5000`.

### Flashloan via Remix
1. Open [Remix IDE](https://remix.ethereum.org).
2. Create `SimpleFlashloan.sol` with Aave V3 code.
3. Compile with Solidity ≥0.8.x.
4. Deploy using `Injected Web3` (MetaMask) on Goerli or Mainnet.
5. Call `requestFlashLoan(...)` and inspect via console.

---

## Future Enhancements
- Replace regression with LSTM/Transformer models.
- Multi-chain support (Optimism, Arbitrum).
- Integrate on-chain order-book DEXes.
- Add governance for community strategy proposals.

---

## License & Disclaimer
Licensed under MIT. Flashloan arbitrage entails financial risk; use at your own risk.

---

## 10. Appendix – Diagrams

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
![on1builder](https://github.com/user-attachments/assets/59e03abe-67ee-4195-9030-63f49c48e46f)
---