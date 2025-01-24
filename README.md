# 0xBuilder MEV Bot

[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-red.svg)](CONTRIBUTING.md)
##
[![Python Version](https://img.shields.io/badge/Python-3.12.*-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Geth Version](https://img.shields.io/badge/Geth-v1.14.*-blue.svg)](https://geth.ethereum.org/)

![Status](https://img.shields.io/badge/Status-Work%20in%20Progress-orange.svg)


## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
  - [System Requirements](#system-requirements)
  - [Software Dependencies](#software-dependencies)
  - [Ethereum Node Setup](#ethereum-node-setup)
- [Installation](#installation)
  - [Cloning the Repository](#cloning-the-repository)
  - [Setting up Virtual Environment](#setting-up-virtual-environment)
  - [Installing Dependencies](#installing-dependencies)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [Configuration Files](#configuration-files)
- [Deploying the Flashloan Contract](#deploying-the-flashloan-contract)
- [Obtaining API Keys](#obtaining-api-keys)
- [Running the Bot](#running-the-bot)
- [Strategies](#strategies)
- [Logging](#logging)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Disclaimer](#disclaimer)




## Introduction

***0xBuilder***  **is an advanced Ethereum trading bot for high-frequency trading and MEV opportunities. It implements strategies like front-running, back-running, sandwich attacks, and flashloan executions using various DeFi protocols.**
**The bot monitors the Ethereum mempool for arbitrage opportunities, executes transactions, and interacts with smart contracts.**

**It uses flashloans for capital efficiency and dynamic gas pricing optimal transaction fees. It is highly configurable, supports multiple wallets, tokens, and trading pairs, and provides detailed logging for analysis. It integrates with various APIs for blockchain and market data, enabling real-time market analysis and strategy execution.**
**0xBuilder is designed for traders, developers, and researchers looking to leverage MEV opportunities in the Ethereum ecosystem.**

***Note:*** ***0xBuilder is a work in progress.***

[![](https://mermaid.ink/img/pako:eNp1kttu2zAMhl-FELBdNUBPF2uArcg5aZe0WHLTysbA2YotRJYMWd7qJtmzj5GdwNkwX8j-yU808ZNbFplYsC5LLOYprIaBBnp6_PKtX0oVCwtLh9aF0Ol8gT6faekkKvkudQIDo9cyCesrfU8MOAW1iNwhPxEuBU31G2TgkSGfiyw3RsHcUDVjCW2AoQdGvKdRVe8CVhZ1gVTM6KJBRh4Zb5-tWUuHP5SApzw31pVUq7rf11R9jg_s7kUUO5jwAaqoVOgbQx3D6E1EpVdzUzho1Vs6i04kVdiusjBtZgdkVJ2e-IamV3xsjXadb6UOzxLXvI_R5t_4DV9SG79klELPOUJCOMvf8l78E3Uk4r8bqs_pledmfKywSJVBfd_kp9d1plE3Z-r2qGo98_KBP-VOZvVUF4Z-Ch9hgkfPHzz0SA2vhau-L4Sjycg8J68GqYg2RePM55UtBfyGu8vO3acPzeXH_9l3SsJXXg9DFO2J1yt38rn9ZhcsEzZDGdPmbg_xgLlUZCJgXfqM0W4CFug9cVg6s6x0xLqOmrtg1pRJyrprVAWpMo_J2aFEWv_sFM1Rvxpz1Ps_TBb5rA?type=png)](https://mermaid.live/edit#pako:eNp1kttu2zAMhl-FELBdNUBPF2uArcg5aZe0WHLTysbA2YotRJYMWd7qJtmzj5GdwNkwX8j-yU808ZNbFplYsC5LLOYprIaBBnp6_PKtX0oVCwtLh9aF0Ol8gT6faekkKvkudQIDo9cyCesrfU8MOAW1iNwhPxEuBU31G2TgkSGfiyw3RsHcUDVjCW2AoQdGvKdRVe8CVhZ1gVTM6KJBRh4Zb5-tWUuHP5SApzw31pVUq7rf11R9jg_s7kUUO5jwAaqoVOgbQx3D6E1EpVdzUzho1Vs6i04kVdiusjBtZgdkVJ2e-IamV3xsjXadb6UOzxLXvI_R5t_4DV9SG79klELPOUJCOMvf8l78E3Uk4r8bqs_pledmfKywSJVBfd_kp9d1plE3Z-r2qGo98_KBP-VOZvVUF4Z-Ch9hgkfPHzz0SA2vhau-L4Sjycg8J68GqYg2RePM55UtBfyGu8vO3acPzeXH_9l3SsJXXg9DFO2J1yt38rn9ZhcsEzZDGdPmbg_xgLlUZCJgXfqM0W4CFug9cVg6s6x0xLqOmrtg1pRJyrprVAWpMo_J2aFEWv_sFM1Rvxpz1Ps_TBb5rA)

## Features

### Trading Capabilities
- **Mempool Monitoring**: Tracks the Ethereum mempool for arbitrage and profit opportunities
- **Strategy Execution**: Implements front-running, back-running, sandwich attacks, and flashloan executions
- **Flashloan Integration**: Utilizes flashloans for capital efficiency without initial capital
- **Market Analysis**: Analyzes market conditions using multiple APIs

### Technical Features
- **Dynamic Gas Pricing**: Adjusts gas prices based on network conditions
- **Nonce Management**: Manages nonces to prevent transaction failures
- **Safety Mechanisms**: Includes safety checks to manage risks
- **Transaction Bundling**: Groups multiple transactions for efficiency

### Integration & Customization
- **Smart Contract Interactions**: Interacts with DeFi protocols like Uniswap, Aave, Sushiswap, PancakeSwap, and Balancer
- **API Integration**: Connects to various APIs for blockchain and market data
- **Configurable Parameters**: Allows adjustment of parameters, strategies, and risk levels
- **Detailed Logging**: Provides logs of activities, transactions, and strategies
- **Customizable**: Supports multiple wallets, tokens, and trading pairs

## Project Structure

```
/0xBuilder/
├── abi/
│   ├── uniswap_abi.json
│   ├── erc20_abi.json
│   └── aave_pool_abi.json
├── contracts/
│   ├── SimpleFlashloan.sol
│   └── IERC20.sol
├── linear_regression/
│   ├── training_data.csv
│   └── price_model.joblib
├── python/
│   ├── safety_net.py
│   ├── strategy_net.py
│   ├── mempool_monitor.py
│   ├── market_monitor.py
│   ├── main.py
│   ├── transaction_core.py
│   ├── main_core.py
│   ├── nonce_core.py
│   ├── api_config.py
│   ├── configuration.py
|   ├── abi_registry.py
│   ├── 0xBuilder.log
|   ├── __init__.py
│   └── pyutils/
│    ├── strategyexecutionerror.py
│    ├── strategyconfiguration.py
│    └── colorformatter.py
├── utils/
│   ├── token_addresses.json
│   ├── erc20_signatures.json
│   └── token_symbols.json
├── .env
└── requirements.txt
```

### Description of Key Directories and Files

- **abi/**: Contains JSON files for various smart contract ABIs.
- **contracts/**: Includes Solidity smart contracts.
- **linear_regression/**: Contains data and models for regression analysis.
- **python/**: Contains Python scripts forming the core functionality.
- **utils/**: Stores utility JSON files for token data.
- **Logs/**: Maintains log files for tracking activities.
- **requirements.txt**: Lists Python dependencies.

## Prerequisites

Ensure the following before running 0xBuilder:

### System Requirements

- **Operating System**:
  - Linux (Ubuntu 20.04+ recommended)
  - Windows 10/11
  - macOS 12+
- **Network**:
  - Internet: Minimum 50Mbps (100Mbps recommended)
- **Hardware**:
  - CPU: 4+ cores, 3.0GHz+ (Intel i7/Ryzen 7 or better recommended)
  - RAM: Minimum 16GB (32GB recommended)
  - Storage: Minimum 1.3TB NVMe SSD (2TB recommended)

### Software Dependencies

- **Execution Client**: execution client compatible with Ethereum, see [Ethereum Node Setup](#ethereum-node-setup)
- **Development Tools**:
    - web3.py
    - Python packages from `requirements.txt`

### Additional Optional Tools
- **Git**: Latest stable version
- **Docker**: Optional for containerization
- **Build Tools**: make, gcc, platform-specific compilers

### Ethereum Node Setup

Choose and set up an execution client compatible with Ethereum:

| Client | Language | OS Support | Networks | Sync Methods |
|--------|----------|------------|----------|--------------|
| [Geth](https://geth.ethereum.org/) | Go | Linux, Windows, macOS | Mainnet, Sepolia, Holesky | Snap, Full |
| [Nethermind](https://www.nethermind.io/) | C#/.NET | Linux, Windows, macOS | Mainnet, Sepolia, Holesky | Snap, Fast, Full |
| [Besu](https://besu.hyperledger.org/) | Java | Linux, Windows, macOS | Mainnet, Sepolia, Holesky | Snap, Fast, Full |
| [Erigon](https://github.com/ledgerwatch/erigon) | Go | Linux, Windows, macOS | Mainnet, Sepolia, Holesky | Full |
| [Reth](https://reth.rs/) | Rust | Linux, Windows, macOS | Mainnet, Sepolia, Holesky | Full |
| [EthereumJS](https://github.com/ethereumjs/ethereumjs-monorepo) | TypeScript | Linux, Windows, macOS | Sepolia, Holesky | Full |

#### Geth Configuration

1. **Installation**:
   Follow the official [Geth installation guide](https://geth.ethereum.org/docs/install-and-build/installing-geth).

2. **Launch Node**:
```bash
./geth --mainnet --syncmode snap --http --http.api eth,net,admin,web3,txpool --ws --ws.api eth,net,admin,web3,txpool --maxpeers 100 --cache 16000 --ipcpath ~/0xBuilder/geth.ipc --allow-insecure-unlock --http.corsdomain "*" 
```

3. **Monitor Sync**:
```bash
   # Connect to node
   geth attach ipc:/path/to/geth.ipc

   # Check sync status
   > eth.syncing
```

#### Beacon Node Setup

1. **Installation**:
   Follow the official [Prysm installation guide](https://docs.prylabs.network/docs/install/install-with-script).

```bash 
   curl https://raw.githubusercontent.com/prysmaticlabs/prysm/master/prysm.sh --output prysm.sh
   chmod +x prysm.sh
```
2. **Launch Node**:
```bash
./prysm.sh beacon-chain --accept-terms-of-use --execution-endpoint ~/0xBuilder/geth.ipc --mainnet --checkpoint-sync-url https://beaconstate.info --genesis-beacon-api-url https://beaconstate.info
``` 
Install either:

- [Prysm](https://docs.prylabs.network/docs/getting-started)
- [Lighthouse](https://lighthouse-book.sigmaprime.io/installation.html)

## Installation

### Cloning the Repository

```bash
git clone https://github.com/John0n1/0xBuilder.git
cd 0xBuilder
```

### Setting up Virtual Environment

Using a virtual environment manages dependencies:

For Linux/MacOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Installing Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

### Environment Variables

1. Create a `.env` file:

```bash
# Linux/MacOS
cp .env.example .env

# Windows
copy .env.example .env
```

2. Configure variables:
   - Add API keys
   - Set node endpoints
   - Configure wallet details

3. Validate configuration:

```bash
ls -la .env
chmod 600 .env
```

Example `.env`:

```ini
# API Configuration
ETHERSCAN_API_KEY=your_etherscan_api_key
INFURA_PROJECT_ID=your_infura_project_id
COINGECKO_API_KEY=your_coingecko_api_key
COINMARKETCAP_API_KEY=your_coinmarketcap_api_key
CRYPTOCOMPARE_API_KEY=your_cryptocompare_api_key

# Ethereum Node Configuration
HTTP_ENDPOINT=http://127.0.0.1:8545
WS_ENDPOINT=wss://127.0.0.1:8546
IPC_ENDPOINT=/path/to/geth.ipc

# Wallet Configuration
PRIVATE_KEY=your_private_key
WALLET_ADDRESS=0xYourWalletAddress

# Token Configuration
TOKEN_ADDRESSES=utils/token_addresses.json
TOKEN_SYMBOLS=utils/token_symbols.json

# DEX Router Configurations
UNISWAP_ADDRESS=0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D
SUSHISWAP_ADDRESS=0xd9e1cE17F2641f24aE83637ab66a2cca9C378B9F

# ABI Paths
UNISWAP_ABI=abi/uniswap_abi.json
SUSHISWAP_ABI=abi/sushiswap_abi.json
ERC20_ABI=abi/erc20_abi.json

# Flashloan Configuration
AAVE_ADDRESS=0xYourFlashloanContractAddress
AAVE_POOL_ADDRESS=0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2
```

### Configuration Files

Ensure JSON files in `utils` are properly formatted:

| File | Description | Format |
|------|-------------|--------|
| `token_addresses.json` | Monitored token contracts | `{"symbol": "address"}` |
| `token_symbols.json` | Address to symbol mapping | `{"address": "symbol"}` |
| `erc20_signatures.json` | ERC20 function signatures | `{"name": "signature"}` |

## Deploying the Flashloan Contract

Deploy a flashloan contract compatible with Aave V3.

### Deployment Options

#### Using Remix IDE (Recommended)

1. Launch [Remix IDE](https://remix.ethereum.org/)
2. Create `SimpleFlashloan.sol`
3. Implement flashloan logic per Aave's specifications
4. Compile with Solidity v0.8.19+
5. Deploy via MetaMask
6. Update `.env` with contract address

#
3. **Example Flashloan Contract AAVE V3**:
   ```solidity
   // SPDX-License-Identifier: MIT
   pragma solidity ^0.8.20;

   import "https://github.com/aave/aave-v3-core/blob/master/contracts/flashloan/base/FlashLoanSimpleReceiverBase.sol";
   import "https://github.com/aave/aave-v3-core/blob/master/contracts/interfaces/IPoolAddressesProvider.sol";
   import "https://github.com/aave/aave-v3-core/blob/master/contracts/dependencies/openzeppelin/contracts/IERC20.sol";

   contract SimpleFlashLoan is FlashLoanSimpleReceiverBase {
       address payable public owner;

       event FlashLoanRequested(address token, uint256 amount);
       event FlashLoanExecuted(address token, uint256 amount, uint256 premium, bool success);

       constructor(address _addressProvider) FlashLoanSimpleReceiverBase(IPoolAddressesProvider(_addressProvider)) {
           owner = payable(msg.sender);
       }

       modifier onlyOwner() {
           require(msg.sender == owner, "Not contract owner");
           _;
       }

       function fn_RequestFlashLoan(address _token, uint256 _amount) public onlyOwner {
           emit FlashLoanRequested(_token, _amount);
           POOL.flashLoanSimple(address(this), _token, _amount, "", 0);
       }

       function executeOperation(
           address asset,
           uint256 amount,
           uint256 premium,
           address initiator,
           bytes calldata params
       ) external override returns (bool) {
           require(IERC20(asset).approve(address(POOL), amount + premium), "Approval failed");
           emit FlashLoanExecuted(asset, amount, premium, true);
           return true;
       }

       function withdrawToken(address _tokenAddress) public onlyOwner {
           IERC20 token = IERC20(_tokenAddress);
           uint256 balance = token.balanceOf(address(this));
           require(balance > 0, "No tokens to withdraw");
           token.transfer(owner, balance);
       }

       function withdrawETH() public onlyOwner {
           uint256 balance = address(this).balance;
           require(balance > 0, "No ETH to withdraw");
           owner.transfer(balance);
       }

       receive() external payable {}
   }
   ```

6. **Update Configuration**:
   ```ini
   AAVE_ADDRESS=0xYourDeployedContract
   ```

## Obtaining API Keys

Register and obtain API keys from:

1. [Infura](https://infura.io/)
2. [Etherscan](https://etherscan.io/apis)
3. [CoinGecko](https://www.coingecko.com/en/api)
4. [CoinMarketCap](https://coinmarketcap.com/api/)
5. [CryptoCompare](https://min-api.cryptocompare.com/)
6. [Binance](https://www.binance.com/en/support/faq/) (optional)

Store all API keys securely.

## Running the Bot

[![](https://mermaid.ink/img/pako:eNqFkcFugzAMhl_FyqUSKn0AJk1qgUo77LJNPQx6iMCAtZAwk6Chqte9wN5wT7IAZTsup_j3Z1v-fRGFKVFEombZNfCS3OUa_NtnsTIa4Qk705M1PJ4hDO_hkMWM0iKciK2TClI9EBvdorbnW-1hJuNsX1ga_mHjmU2yB91bqRR8f34tiU0QdNQB3fSQgfHdEeNU3e_shw2CzdolmbukWWx0RbVjhB3qAY6kcEXSGTlmJ2SqRlhJacnoiVkosRUtciup9JZcJi0XtvEzcxH5byn5LRe5vnpOOmueR12IyLLDrWDj6kZElVS9j1xX-s0Tkt7X9lftpH415i_GcrL2cbnAfIjrD5YVg78?type=png)](https://mermaid.live/edit#pako:eNqFkcFugzAMhl_FyqUSKn0AJk1qgUo77LJNPQx6iMCAtZAwk6Chqte9wN5wT7IAZTsup_j3Z1v-fRGFKVFEombZNfCS3OUa_NtnsTIa4Qk705M1PJ4hDO_hkMWM0iKciK2TClI9EBvdorbnW-1hJuNsX1ga_mHjmU2yB91bqRR8f34tiU0QdNQB3fSQgfHdEeNU3e_shw2CzdolmbukWWx0RbVjhB3qAY6kcEXSGTlmJ2SqRlhJacnoiVkosRUtciup9JZcJi0XtvEzcxH5byn5LRe5vnpOOmueR12IyLLDrWDj6kZElVS9j1xX-s0Tkt7X9lftpH415i_GcrL2cbnAfIjrD5YVg78)

### Prerequisites

- Synchronized Ethereum node
- Active beacon node
- Configured environment variables
- Valid API keys
- Deployed flashloan contract
- Wallet with with balance sufficient for gas fees

### Launch Sequence

1. Activate environment:
   ```bash
   source venv/bin/activate
   ```

2. Start bot:
   ```bash
   python python/main.py
   ```

### Monitoring

- Check `python/0xBuilder.log` for logs
- Monitor console for real-time status
- Use `Ctrl+C` to shutdown gracefully

### Performance Optimization

- Keep node fully synced
- Monitor API rate limits
- Maintain sufficient ETH balance
- Regularly check logs
- Update dependencies as needed

## Strategies

0xBuilder implements several trading strategies to capitalize on Ethereum network opportunities:

### Core Strategies

- **Front-Running**: Executes transactions ahead of profitable ones.
- **Back-Running**: Executes transactions immediately after profitable ones.
- **Sandwich Attacks**: Combines front- and back-running around target transactions.
- **Flashloan Arbitrage**: Uses flashloans for arbitrage without initial capital.

### Technical Components

- **Nonce Management**: Ensures proper transaction ordering.
- **Dynamic Gas Optimization**: Adjusts gas prices based on network conditions.
- **Real-time Market Analysis**: Identifies profitable opportunities.
- **Safety Protocols**: Validates transactions and assesses risks.
- **Transaction Bundling**: Groups multiple transactions per block.

## Logging

Logs are maintained in `python/0xBuilder.log`, including:

- Transaction detections
- Strategy executions
- Errors and exceptions
- Transaction results

Configure logging in `python/main_core.py` via `setup_logging()`.

## Troubleshooting


### Common Issues

| Issue | Solution |
|-------|----------|
| Node Connection Failures | Verify node status and endpoints |
| API Rate Limit Exceeded | Implement throttling or upgrade API tier |
| Insufficient Gas Balance | Ensure adequate ETH for fees |
| Nonce Synchronization | Reset nonce manager or synchronize manually |
| Node Sync Status | Ensure full node synchronization |

### Debug Tips

1. Enable verbose logging
2. Keep dependencies updated
3. Verify contract deployment on explorers
4. Test on testnets before mainnet

## Contributing

Review [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
Contributions are most welcome! ❤️❤️

### Contribution Process

1. Fork the repository
2. Create a feature branch
3. Follow PEP 8
4. Include unit tests
5. Submit a pull request

## License

Licensed under the [MIT License](LICENSE). See LICENSE file for details.

## Disclaimer

**IMPORTANT**: Me the author, and the project contributors, are not responsible for any financial losses or damages incurred by using 0xBuilder. Use at your own risk. Not in any instance should any contributor, author, or the project be held liable for any financial losses or damages. Use 0xBuilder at your own risk and discretion. 

### Risk Factors

- Strategies may be aggressive or unethical
- Cryptocurrency trading carries financial risks
- Smart contracts may have vulnerabilities

## Acknowledgements

Thanks to the following projects creating the foundation of many projects including this one: 

- ❤️ [Aave](https://github.com/aave/aave-v3-core) for Flashloan integration
- ❤️ [Uniswap](https://github.com/uniswap/uniswap-v2-core) for DEX integration
- ❤️ [Geth](https://github.com/ethereum/go-ethereum) for Ethereum node
- ❤️ [Remix](https://remix.ethereum.org/) for Solidity IDE
- ❤️ [Web3.py](https://github.com/ethereum/web3.py) for Python integration

