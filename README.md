# ON1Builder

![ON1Builder Logo](https://github.com/user-attachments/assets/59e03abe-67ee-4195-9030-63f49c48e46f)

[![Built with Python 3.12](https://img.shields.io/badge/Built%20with-Python%203.12-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[![Last Commit](https://img.shields.io/github/last-commit/John0n1/ON1Builder?display_timestamp=committer&logo=GitHub&color=white)](https://github.com/John0n1/ON1Builder/commits/main)


**ON1Builder** is an advanced Ethereum MEV (Maximal Extractable Value) orchestrator. It continuously monitors the mempool, evaluates pending transactions, and automatically executes profitable strategies including flashloan arbitrage, front-running, back-running, and sandwich attacks. Built with modularity, resilience, and performance in mind.

---

## Table of Contents

1. [Key Features](#key-features)  
2. [Prerequisites](#prerequisites)  
3. [Geth & Prysm Setup](#geth--prysm-setup)  
4. [Installation](#installation)  
5. [Configuration](#configuration)  
6. [Running ON1Builder](#running-on1builder)  
7. [Monitoring & Metrics](#monitoring--metrics)  
8. [Component Reference](#component-reference)  
9. [Security Best Practices](#security-best-practices)  
10. [Troubleshooting](#troubleshooting)  
11. [Extending & Custom Strategies](#extending--custom-strategies)  
12. [Roadmap](#roadmap)  
13. [License](#license)  

---

[![](https://mermaid.ink/img/pako:eNqNVttu4zYQ_RWCwC52ldi5OIljtw2Q9WZbA_Y2SBws2roIaGkscy2RKkmlceMA-ZD2tR-WL-mQukSylbZ6sEXyzMzhmYv9QH0ZAO3TeSR_9xdMGTL5OBUke3Q6CxVLFmQ8uh2xFSjyy5SSKf21RJC9iAtg6lZBqEBrLgVCPG_M_AUekBGeCS5Cz_t2pvbO3rW9RHEfbmOMGrW_ylnEZ96OZxTjFnYbMMPavr7z3pdRQARTkb1mn2_ekO-Kh9wMyR4ZSGGUjF62M2BJ_2b4Gv1PEdPLm6Hj7N7fiplOvtlBtmfX0l-CaQ9_RHvPa-JTY4Ik5jxMFTMowqtUMtRrdGo-HKnaTqkiiLucKPnpfDx6_z_IKSBXqTA8hn_hpuC2AG1xG2OKLCLPb7bIGWmDlZMmZIdI5S9Am4yvs3_xgIwwll5pA7GuOIY4kTIaS8GNVJn72lYeJMHLYY20zD2ZAwQ1ctc2IoSrz2Ccg8o6t1bAxRzZQQzCkMjWJaiaj4liQjPfEi_vubGX-5qlPAqw7jQPhf2yqlc9fZbCh9JHucqthV0THzsE6ndgczAvNyhWBX-ulyTES20kRWGN1qSr7hTK2Z5DouNRzfj8w_AKQo7ZWjnTyjo3xB2i0U896PnlMCvLzKpY5TZxGhneukNNpLJQt7lDbD5aEY-50f9ZrRf3BpRgWw39BWadq8uBC3thFqAgjQlueF4-Gt55IZgFlmGiVjquDJFLqwCS0Wj7gQsmnB4DycX34C-lfR8PyPPT3y81P1hgjddD2aoXNiMZwvMthAfkwKtUev0mQQh68xr50CGtFka7urieYPgv19hvrdZZ2WaFu_pYabVbZ2sNxmAjaCJnX8E364rNKxYJMwuNsvhSYKsKo9fV5L9qhoKRJawsukhyQasImXGutWsDoNKODacbTdaAKFuoyXfRKE3Eqs3QcF651vZZVaAytSofj1xgiWac81FWF8G6WD8__ZkoOeeGzSIg5v756S_yWwoprLcVqWw4W7gHPzWIbFRnY9NZuLmy3tSqCelmCY4ff4mp3dCvCR87GYn9cV43adpkg_I1VVkDtGjrQuWSkPMTrASLuY-jT5O3xE2y9XbeKpSclcNp7Ks7GaXxtkW5dOgfJpNLHMhRhIzLSVHQKfWsc21Oef08X7gDN06mgu7SUPGA9ucs0rBLY1Axs2v6YG2mFGdNjOO2j68BXmtKp-IRjRImfpYypn2jUjRTMg0XpZM0wdTAR87wd_wFgtMV1EBiwdJ-p3Pac05o_4He0_5hp9M-Oe0e7Z8eHXUPT45PDnfpivZbB732wcl-t3fQ6XX3e3jUfdylf7jA-23E9_A52u-ddo6Pe51dCoG9-Dj7C-k7QenjP4JCRs4?type=png)](https://mermaid.live/edit#pako:eNqNVttu4zYQ_RWCwC52ldi5OIljtw2Q9WZbA_Y2SBws2roIaGkscy2RKkmlceMA-ZD2tR-WL-mQukSylbZ6sEXyzMzhmYv9QH0ZAO3TeSR_9xdMGTL5OBUke3Q6CxVLFmQ8uh2xFSjyy5SSKf21RJC9iAtg6lZBqEBrLgVCPG_M_AUekBGeCS5Cz_t2pvbO3rW9RHEfbmOMGrW_ylnEZ96OZxTjFnYbMMPavr7z3pdRQARTkb1mn2_ekO-Kh9wMyR4ZSGGUjF62M2BJ_2b4Gv1PEdPLm6Hj7N7fiplOvtlBtmfX0l-CaQ9_RHvPa-JTY4Ik5jxMFTMowqtUMtRrdGo-HKnaTqkiiLucKPnpfDx6_z_IKSBXqTA8hn_hpuC2AG1xG2OKLCLPb7bIGWmDlZMmZIdI5S9Am4yvs3_xgIwwll5pA7GuOIY4kTIaS8GNVJn72lYeJMHLYY20zD2ZAwQ1ctc2IoSrz2Ccg8o6t1bAxRzZQQzCkMjWJaiaj4liQjPfEi_vubGX-5qlPAqw7jQPhf2yqlc9fZbCh9JHucqthV0THzsE6ndgczAvNyhWBX-ulyTES20kRWGN1qSr7hTK2Z5DouNRzfj8w_AKQo7ZWjnTyjo3xB2i0U896PnlMCvLzKpY5TZxGhneukNNpLJQt7lDbD5aEY-50f9ZrRf3BpRgWw39BWadq8uBC3thFqAgjQlueF4-Gt55IZgFlmGiVjquDJFLqwCS0Wj7gQsmnB4DycX34C-lfR8PyPPT3y81P1hgjddD2aoXNiMZwvMthAfkwKtUev0mQQh68xr50CGtFka7urieYPgv19hvrdZZ2WaFu_pYabVbZ2sNxmAjaCJnX8E364rNKxYJMwuNsvhSYKsKo9fV5L9qhoKRJawsukhyQasImXGutWsDoNKODacbTdaAKFuoyXfRKE3Eqs3QcF651vZZVaAytSofj1xgiWac81FWF8G6WD8__ZkoOeeGzSIg5v756S_yWwoprLcVqWw4W7gHPzWIbFRnY9NZuLmy3tSqCelmCY4ff4mp3dCvCR87GYn9cV43adpkg_I1VVkDtGjrQuWSkPMTrASLuY-jT5O3xE2y9XbeKpSclcNp7Ks7GaXxtkW5dOgfJpNLHMhRhIzLSVHQKfWsc21Oef08X7gDN06mgu7SUPGA9ucs0rBLY1Axs2v6YG2mFGdNjOO2j68BXmtKp-IRjRImfpYypn2jUjRTMg0XpZM0wdTAR87wd_wFgtMV1EBiwdJ-p3Pac05o_4He0_5hp9M-Oe0e7Z8eHXUPT45PDnfpivZbB732wcl-t3fQ6XX3e3jUfdylf7jA-23E9_A52u-ddo6Pe51dCoG9-Dj7C-k7QenjP4JCRs4)


## Key Features

- **Mempool Monitoring**  
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
   geth --syncmode "snap" --http --http.corsdomain "*" --http.api engine, admin, web3,txpool --datadir ~/ethereum/mainnet --ipcpath on1builder/geth.ipc
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

