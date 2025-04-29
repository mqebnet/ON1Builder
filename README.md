# ON1Builder

![ON1Builder Logo](https://github.com/user-attachments/assets/59e03abe-67ee-4195-9030-63f49c48e46f)

[![Built with Python 3.12](https://img.shields.io/badge/Built%20with-Python%203.12-blue?logo=python)](https://www.python.org/)
[![ON1Builder Wiki](https://img.shields.io/badge/ON1Builder-Wiki-blue?logo=GitHub&logoColor=white)](https://github.com/John0n1/ON1Builder/wiki)
[![Last Commit](https://img.shields.io/github/last-commit/John0n1/ON1Builder?display_timestamp=committer&logo=GitHub&color=white)](https://github.com/John0n1/ON1Builder/commits/main)


## Overview

ON1Builder is a MEV flashloan and arbitrage framework for Ethereum Mainnet and testnets. It integrates on-chain smart contracts, an asynchronous Python backend, and a web-based dashboard to automate strategy deployment, monitoring, and performance analysis.

### Key Capabilities
- **Aave V3 Flashloans:** Atomic borrowing and repayment within a single transaction.
- **Automated DEX Routing:** Optimal trade execution via Uniswap and Sushiswap.
- **Real-Time Market Data & Predictive Models:** Continuous data aggregation and hourly linear regression forecasting.
- **Robust Safety Mechanisms:** Nonce management, slippage controls, gas adjustment, and fallback routines.
- **Web Dashboard:** Full visibility into performance metrics, logs, and control functions.

---

## Prerequisites

Before installation, ensure your environment meets the following requirements:

- **Operating System:** Ubuntu 20.04+ (preferred), Windows 10/11, or macOS 12+.
- **Hardware:** Minimum 4 CPU cores, 16 GB RAM (32 GB recommended), NVMe SSD with ≥ 2 TB free space for blockchain data.
- **Python:** Version ≥ 3.12.
- **Ethereum Clients:** Geth (Execution Layer) and Prysm (Consensus Layer) configured for IPC communication.
- **APIs & Keys:** Valid keys for Etherscan, CoinGecko, CoinMarketCap, and CryptoCompare (free-tier suffices).

| Provider         | Free Tier Highlights                                                 |
|------------------|----------------------------------------------------------------------|
| [Infura](https://infura.io)        | 3 million credits/day, 1 API key       |
| [Coingecko](https://coingecko.com)       | Free API key, (trial)            |
| [Etherscan](https://etherscan.io)     | 100 k calls/day, 5 calls/s          |
| [CoinMarketCap](https://coinmarketcap.com) | Free API key, (trial)          |
| [Cryptocompare](https://Cryptocompare.com) | Free API key, (trial)          | 

---

## Installation and Configuration

### 1. Repository Setup

```bash
# Clone the repository
git clone https://github.com/John0n1/ON1Builder.git
cd ON1Builder

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: .\venv\Scripts\activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
``` 

### 2. Environment Variables

Copy the example environment file and update with your parameters:

```bash
cp .env.example .env
nano .env
```

#### Mandatory Variables
```
# Wallet Settings
WALLET_ADDRESS=0xYourEthereumAddress
WALLET_KEY=<YOUR_PRIVATE_KEY>         # Keep this secure 

# Node Endpoints (IPC or HTTP/WS) Atleast one must be set
IPC_ENDPOINT=~/ON1Builder/geth.ipc
HTTP_ENDPOINT=http://127.0.0.1:8545
WEBSOCKET_ENDPOINT=wss://127.0.0.1:8545

# API Keys
IǸFURA_PROJECT_ID=<YOUR_INFURA_PROJECT_ID>
ETHERSCAN_API_KEY=<YOUR_ETHERSCAN_KEY>
COINGECKO_API_KEY=<YOUR_COINGECKO_KEY>
COINMARKETCAP_API_KEY=<YOUR_CMC_KEY>
CRYPTOCOMPARE_API_KEY=<YOUR_CC_KEY>

# AAVE V3
AAVE_POOL_ADDRESS=<YOUR_AAVE_POOL_ADDRESS>
AAVE_FLASHLOAN_ADDRESS=<YOUR_AAVE_FLASHLOAN_CONTRACT_ADDRESS> 
```

> **Security Note:** Never expose `WALLET_KEY` in public repositories or logs.

### 3. Node Synchronization

Ensure your Execution client (Geth) and Consensus client (Prysm) are fully synchronized before running the bot.

Geth installation guide: [Install Geth](https://geth.ethereum.org/docs/install-and-build/installing-geth)
Prysm installation guide: [Install Prysm](https://docs.prylabs.network/docs/install-prysm)

#### Geth (Execution Client)
```bash
geth \
  --syncmode "snap" \
  --mainnet \
  --ipcpath "$IPC_ENDPOINT" \
  --cache=12000 \
  --maxpeers=80
```

#### Prysm (Consensus Client)
```bash
./prysm.sh beacon-chain \
  --execution-endpoint="$IPC_ENDPOINT" \
  --mainnet \
  --checkpoint-sync-url=https://beaconstate.info \
  --genesis-beacon-api-url=https://beaconstate.info
```

Monitor logs to confirm `SYNCED` status before proceeding.

---

## Quick Start

Once installation and node sync are complete:

1. **Activate environment**
   ```bash
   source venv/bin/activate
   ```
2. **Launch the bot**
   ```bash
   python python/main.py
   ```
3. **Access Dashboard**
   Open your browser at:  
   `http://localhost:5000`

The dashboard displays live metrics such as transaction success rates, average execution times, profit yields, gas usage, and network congestion.

---

## Project Structure

```text
ON1Builder/
├── abi/                   # ABI files for contracts
├── linear_regression/       # ML model files
├── logs/                  # Log files
├── docs/                 # Documentation & diagrams
├── utils/                # Utility scripts
├── python/                # Core Python modules
│   ├── configuration.py   # Env loader & validation
│   ├── apiconfig.py       # Market data API wrappers
│   ├── marketmonitor.py   # Data ingestion & ML forecasting
│   ├── transactioncore.py # Tx builder, signer, simulator
│   ├── mempoolmonitor.py  # Mempool monitoring & strategy selection
│   ├── app.py            # Flask app for dashboard
│   ├── strategynet.py     # Strategy selection via RL
│   ├── noncecore.py       # Nonce assignment logic
│   ├── safetynet.py       # Risk & safety checks
│   ├── maincore.py        # Event loop & orchestration
│   ├── abiregistry.py     # ABI loader
│   └── loggingconfig.py   # Logger setup
├── ui/                    # Web dashboard assets
├── contracts/             # Solidity source & ABIs
├── tests/                 # Unit, integration, E2E tests
├── .env.example           # Environment template
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore file
├── LICENSE                # Project license
└── README.md              # This file
```

---

## Testing

Run the complete test suite to verify integrity:
```bash
pytest --maxfail=1 --disable-warnings -q
```

---

## Contributing

1. Fork the repository and create a feature branch.
2. Ensure adherence to PEP8 and include appropriate tests.
3. Submit a pull request with a clear summary of changes.

Refer to `CONTRIBUTING.md` for full guidelines.

---

**Data Flow Diagrams**  

![Data Flow](docs/mermaid1.svg)

[![](https://mermaid.ink/img/pako:eNptk01T4zAMhv-KxwdOhaFNS2gOO5M2LQT6wVBmD-tyMInaekjsjKvsEkr_-yofhXaX5JDIfl5JluQdj0wM3ONrK7MNewqWmtHji_msPchVEoNlC5QWn9n5-Q82EBMjYzY0eqXWuZWojH6uJYMa8IU_CE-XBmI6YVMKk2xPN4bCjyKTa2ShXhmbfuMuEP5DyO6hOJUOBaWgIUKGho1wAxbylM0oRoMNKywQoVaoZKLegTKgf2OVXjdMUDEj0WywKaSZMQk7Y1NpXwFZIFE27Khix7swBo1qVbAHa1YK5UsCbJ5lxmJeRoLtfqlrxbhUfHxhH-xGjH7LJJcIFGIBSZn-AqmIsC6ej0Uzg-xYGBx83lRZ3LbF2BqN54_5oVrNRkcMZPT6_7ojFlLHf1S0YT4iIafbXTFO5HbDqLea-fZFUVLrspI1dduusFA8WMikBfZkpd7K6Khbt50aaSznxOoerNoOK_NOzDNUadmYG7mlgsyMjqhJUlPolIrceL6r6Hs6wAqwmAF67CfY4w6oRGFBDh7V9nCu-88yljJGvf2nnF9QNco2hbhGz07IiRi9QZTjd0ee1BN2OFX55S26RirmHtocWjwFGunS5LuSWXKa0xSW3KPfmCZsyZd6T5pM6l_GpAeZNfl6w72VTLZk5VlMAxIoSRc0_VyVOZpFoaNPDWi6qcPyLnGv2658cm_H38hyri96bqd35bQ7br_b77R4wb0r58K9dJ32lUOv23ev9y3-XiVxeXHtdvv0dC-dfs_p9Tr7vw5iRI0?type=png)](https://mermaid.live/edit#pako:eNptk01T4zAMhv-KxwdOhaFNS2gOO5M2LQT6wVBmD-tyMInaekjsjKvsEkr_-yofhXaX5JDIfl5JluQdj0wM3ONrK7MNewqWmtHji_msPchVEoNlC5QWn9n5-Q82EBMjYzY0eqXWuZWojH6uJYMa8IU_CE-XBmI6YVMKk2xPN4bCjyKTa2ShXhmbfuMuEP5DyO6hOJUOBaWgIUKGho1wAxbylM0oRoMNKywQoVaoZKLegTKgf2OVXjdMUDEj0WywKaSZMQk7Y1NpXwFZIFE27Khix7swBo1qVbAHa1YK5UsCbJ5lxmJeRoLtfqlrxbhUfHxhH-xGjH7LJJcIFGIBSZn-AqmIsC6ej0Uzg-xYGBx83lRZ3LbF2BqN54_5oVrNRkcMZPT6_7ojFlLHf1S0YT4iIafbXTFO5HbDqLea-fZFUVLrspI1dduusFA8WMikBfZkpd7K6Khbt50aaSznxOoerNoOK_NOzDNUadmYG7mlgsyMjqhJUlPolIrceL6r6Hs6wAqwmAF67CfY4w6oRGFBDh7V9nCu-88yljJGvf2nnF9QNco2hbhGz07IiRi9QZTjd0ee1BN2OFX55S26RirmHtocWjwFGunS5LuSWXKa0xSW3KPfmCZsyZd6T5pM6l_GpAeZNfl6w72VTLZk5VlMAxIoSRc0_VyVOZpFoaNPDWi6qcPyLnGv2658cm_H38hyri96bqd35bQ7br_b77R4wb0r58K9dJ32lUOv23ev9y3-XiVxeXHtdvv0dC-dfs_p9Tr7vw5iRI0)

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Disclaimer

Operating MEV strategies involves significant financial risk. Test thoroughly, use small allocations, and maintain secure key management at all times.

