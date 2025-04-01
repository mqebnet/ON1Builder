# ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ON1Builder: Your MEV Toolkit for Ethereum

‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎‎‎[![Python Version](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3120/)
‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎‎[![Latest Release](https://img.shields.io/badge/Release-0.22.03_beta-black.svg)](https://github.com/John0n1/ON1Builder/releases/tag/v0.22.3-beta)

![ON1BUILDER](https://github.com/user-attachments/assets/4f60cb4e-1722-4609-a5f1-86e96ce4f6c7)

**Maximize your Ethereum trading potential with ON1Builder, a powerful framework designed for high-frequency trading and Maximal Extractable Value (MEV) strategies.**

ON1Builder empowers you with tools for:

* **Front-running**: Capitalize on pending transactions for profit.
* **Back-running**: Strategically execute trades after specific transactions.
* **Sandwich Attacks**: Profit from manipulating transaction order in decentralized exchanges.
* **Flash Loan Arbitrage**: Leverage flash loans for zero-capital arbitrage opportunities.

[![](https://mermaid.ink/img/pako:eNptk01T4zAMhv-KxwdOhaFNS2gOO5M2LQT6wVBmD-tyMInaekjsjKvsEkr_-yofhXaX5JDIfl5JluQdj0wM3ONrK7MNewqWmtHji_msPchVEoNlC5QWn9n5-Q82EBMjYzY0eqXWuZWojH6uJYMa8IU_CE-XBmI6YVMKk2xPN4bCjyKTa2ShXhmbfuMuEP5DyO6hOJUOBaWgIUKGho1wAxbylM0oRoMNKywQoVaoZKLegTKgf2OVXjdMUDEj0WywKaSZMQk7Y1NpXwFZIFE27Khix7swBo1qVbAHa1YK5UsCbJ5lxmJeRoLtfqlrxbhUfHxhH-xGjH7LJJcIFGIBSZn-AqmIsC6ej0Uzg-xYGBx83lRZ3LbF2BqN54_5oVrNRkcMZPT6_7ojFlLHf1S0YT4iIafbXTFO5HbDqLea-fZFUVLrspI1dduusFA8WMikBfZkpd7K6Khbt50aaSznxOoerNoOK_NOzDNUadmYG7mlgsyMjqhJUlPolIrceL6r6Hs6wAqwmAF67CfY4w6oRGFBDh7V9nCu-88yljJGvf2nnF9QNco2hbhGz07IiRi9QZTjd0ee1BN2OFX55S26RirmHtocWjwFGunS5LuSWXKa0xSW3KPfmCZsyZd6T5pM6l_GpAeZNfl6w72VTLZk5VlMAxIoSRc0_VyVOZpFoaNPDWi6qcPyLnGv2658cm_H38hyri96bqd35bQ7br_b77R4wb0r58K9dJ32lUOv23ev9y3-XiVxeXHtdvv0dC-dfs_p9Tr7vw5iRI0?type=png)](https://mermaid.live/edit#pako:eNptk01T4zAMhv-KxwdOhaFNS2gOO5M2LQT6wVBmD-tyMInaekjsjKvsEkr_-yofhXaX5JDIfl5JluQdj0wM3ONrK7MNewqWmtHji_msPchVEoNlC5QWn9n5-Q82EBMjYzY0eqXWuZWojH6uJYMa8IU_CE-XBmI6YVMKk2xPN4bCjyKTa2ShXhmbfuMuEP5DyO6hOJUOBaWgIUKGho1wAxbylM0oRoMNKywQoVaoZKLegTKgf2OVXjdMUDEj0WywKaSZMQk7Y1NpXwFZIFE27Khix7swBo1qVbAHa1YK5UsCbJ5lxmJeRoLtfqlrxbhUfHxhH-xGjH7LJJcIFGIBSZn-AqmIsC6ej0Uzg-xYGBx83lRZ3LbF2BqN54_5oVrNRkcMZPT6_7ojFlLHf1S0YT4iIafbXTFO5HbDqLea-fZFUVLrspI1dduusFA8WMikBfZkpd7K6Khbt50aaSznxOoerNoOK_NOzDNUadmYG7mlgsyMjqhJUlPolIrceL6r6Hs6wAqwmAF67CfY4w6oRGFBDh7V9nCu-88yljJGvf2nnF9QNco2hbhGz07IiRi9QZTjd0ee1BN2OFX55S26RirmHtocWjwFGunS5LuSWXKa0xSW3KPfmCZsyZd6T5pM6l_GpAeZNfl6w72VTLZk5VlMAxIoSRc0_VyVOZpFoaNPDWi6qcPyLnGv2658cm_H38hyri96bqd35bQ7br_b77R4wb0r58K9dJ32lUOv23ev9y3-XiVxeXHtdvv0dC-dfs_p9Tr7vw5iRI0)


> **⚠️ Important: Under Active Development**  
> ON1Builder is currently in active development. Expect potential bugs and use with extreme caution. **Your use is at your own risk.**

## Key Features

ON1Builder provides a comprehensive suite of features for effective MEV exploitation:

* **Mempool Monitoring**: Real-time analysis of the Ethereum mempool to identify profitable opportunities.
* **Advanced Trading Strategies**: Implements sophisticated strategies like front-running, back-running, sandwich attacks, and flash loan arbitrage.
* **Flash Loan Integration (Aave V3)**: Utilizes Aave V3 flash loans for capital-efficient trading.
* **Data-Driven Decisions**: Integrates with cryptocurrency APIs for real-time market data analysis.
* **Gas Optimization**: Dynamically adjusts gas prices to minimize transaction costs.
* **Robust Transaction Management**: Includes nonce management and atomic transaction bundling for reliable execution.
* **Risk Mitigation**: Built-in safety checks and risk assessments to protect operations.
* **DeFi Protocol Support**: Seamlessly interacts with popular DeFi platforms like Uniswap, Sushiswap, and Aave.
* **Highly Configurable**: Adaptable to various wallets, tokens, trading pairs, and strategies.
* **Detailed Logging**: Comprehensive logs for performance analysis, debugging, and strategy refinement.

## Project Structure

```
/ON1Builder/
├── abi/                      # Smart Contract ABIs (JSON format)
├── contracts/                # Solidity Smart Contracts (Flashloan, Interfaces)
├── linear_regression/        # Machine Learning for Price Prediction (Models, Data)
├── python/                  # Core Python Bot Logic
│   ├── safetynet.py          # Risk Management & Safety Checks
│   ├── strategynet.py        # MEV Strategy Implementation
│   ├── mempoolmonitor.py     # Mempool Monitoring Engine
│   ├── marketmonitor.py      # Market Data Analysis
│   ├── main.py               # Main Bot Entry Point
│   ├── transactioncore.py    # Transaction Handling & Execution
│   ├── maincore.py           # Core Application Logic
│   ├── noncecore.py          # Nonce Management
│   ├── apiconfig.py          # API Integration & Data Handling
│   ├── configuration.py      # Configuration Loading
│   ├── abiregistry.py        # Centralized ABI Management
│   ├── ON1Builder.log        # Default Log File
│   ├── __init__.py
│   └── pyutils/              # Utility Modules
├── utils/                    # JSON Configuration Files (Tokens, Signatures)
├── .env                      # Environment Variable Configuration
└── requirements.txt          # Python Dependencies
```

## Prerequisites

### System Requirements

* **Operating System**: Linux (Ubuntu 20.04+ recommended), Windows 10/11, macOS 12+
* **Internet**: Stable connection (50Mbps+ recommended)
* **Hardware**:
  * CPU: 4+ cores, 3.0GHz+ (e.g., Intel i7 or equivalent)
  * RAM: 16GB minimum
  * Storage: 1.5TB NVMe SSD (for local Ethereum node synchronization)

### Software Requirements

* **Python**: Version 3.12 or higher
* **Ethereum Execution Client**: One of:
  * [Geth](https://geth.ethereum.org/) (Recommended)
  * [Nethermind](https://www.nethermind.io/)
  * [Besu](https://besu.hyperledger.org/)
  * [Erigon](https://github.com/ledgerwatch/erigon)
  * [Reth](https://reth.rs/) (Testnets only)
  * [EthereumJS](https://github.com/ethereumjs/ethereumjs-monorepo)

## Ethereum Node Setup

Run your own Ethereum Execution and Consensus (Beacon) client. **Geth and Prysm are highly recommended.**

### Example: Setting up Geth

1. **Install Geth**: Follow [Geth installation guide](https://geth.ethereum.org/docs/install-and-build/installing-geth).
2. **Start Geth Node**:
   ```bash
   ./geth --mainnet --syncmode snap --http --http.api eth,net,admin,web3,txpool --ws --ws.api eth,net,admin,web3,txpool --maxpeers 100 --cache 16000 --ipcpath ~/ON1Builder/geth.ipc --allow-insecure-unlock --http.corsdomain "*"
   ```
3. **Verify Sync**:
   ```bash
   geth attach ipc:/path/to/geth.ipc
   > eth.syncing
   ```
   Wait until synchronization is complete.

### Beacon Node Setup

Use [Prysm](https://docs.prylabs.network/docs/install/install-with-script) or [Lighthouse](https://lighthouse-book.sigmaprime.io/installation.html).

```bash
./prysm.sh beacon-chain --mainnet --checkpoint-sync-url=https://beaconcha.in/api/v1/genesis --genesis-beacon-api-url=https://beaconcha.in --http-web3provider=/path/to/geth.ipc
```

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/John0n1/ON1Builder.git
   cd ON1Builder
   ```

2. **Create a Virtual Environment (Recommended):**
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
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Configuration

ON1Builder uses environment variables and JSON configuration files for customization.

### Environment Variables (.env file)

1. **Copy Example .env:**
   ```bash
   cp .env.example .env
   ```
2. **Edit .env**: Configure values such as API keys, Ethereum node endpoints, wallet details, and file paths.

### JSON Configuration Files (utils/ directory)

Customize the following files:

* **token_addresses.json**: Map token symbols to contract addresses.
* **token_symbols.json**: Map token symbols to API identifiers.
* **erc20_signatures.json**: Define ERC20 function signatures.

## Deploy Your Flash Loan Contract

Deploy the `SimpleFlashloan.sol` contract using Remix:

1. Open [Remix IDE](https://remix.ethereum.org/).
2. Create a new file `SimpleFlashloan.sol` and paste the following code:

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

3. Compile and deploy using Remix.  
4. Update your `.env` file with the deployed contract address:
   ```ini
   AAVE_FLASHLOAN_ADDRESS=0xYourDeployedContractAddress
   ```

## Register for API Keys

Obtain API keys from:

* [Infura](https://infura.io/)
* [Etherscan](https://etherscan.io/apis)
* [CoinGecko](https://www.coingecko.com/en/api)
* [CoinMarketCap](https://coinmarketcap.com/api/)
* [CryptoCompare](https://min-api.cryptocompare.com/)

Add your keys to the `.env` file.

## Run the Bot

Ensure:

* Your Ethereum node is fully synchronized.
* The `.env` file and JSON configuration files are correctly set.
* Your wallet is funded with enough ETH to cover gas costs.

**Steps:**

1. Activate your virtual environment:
   ```bash
   # Linux/macOS
   source venv/bin/activate

   # Windows
   .\venv\Scripts\activate
   ```
2. Start ON1Builder:
   ```bash
   python python/main.py
   ```

## Strategies Implemented

ON1Builder includes a range of MEV strategies:

**Front-Running Strategies:**

* Aggressive Front-Running
* Predictive Front-Running
* Volatility Front-Running
* Advanced Front-Running

**Back-Running Strategies:**

* Price Dip Back-Running
* Flashloan Back-Running
* High Volume Back-Running
* Advanced Back-Running

**Sandwich Attack Strategies:**

* Flash Profit Sandwich Attack
* Price Boost Sandwich Attack
* Arbitrage Sandwich Attack
* Advanced Sandwich Attack

**Other Strategies:**

* High-Value ETH Transfer Monitoring

## Logging and Monitoring

Detailed logs are written to `python/ON1Builder.log` and also output to the console. These logs provide insights into:

* Mempool activity
* Strategy execution and performance
* Profit tracking
* Error and exception details

## Troubleshooting

**Common Issues:**

| Issue                              | Solution                                                              |
| ---------------------------------- | --------------------------------------------------------------------- |
| **Node Connection Errors**         | Verify your Ethereum node is running and that the endpoints in `.env` are correct. |
| **API Rate Limit Exceeded**        | Throttle requests or upgrade your API plan for higher limits.         |
| **Insufficient ETH Balance**       | Ensure your wallet has enough ETH to cover gas costs.                  |
| **Nonce Synchronization Problems** | Restart the bot or reset the nonce manually if necessary.              |
| **Ethereum Node Not Synced**       | Wait until your Ethereum client is fully synchronized before running the bot. |
| **Flash Loan Contract Issues**     | Verify your flash loan contract is deployed correctly and the address is correct in `.env`. |

**Debugging Tips:**

1. Enable verbose logging by setting the log level to `DEBUG` in `python/maincore.py`.
2. Ensure all dependencies are installed (`pip install -r requirements.txt`).
3. Test on a testnet (e.g., Sepolia) before deploying to mainnet.
4. Review logs in `python/ON1Builder.log` for detailed error messages.

## Contributing

Contributions are highly welcome! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Write your code following PEP 8 guidelines and add tests.
4. Submit a pull request with detailed descriptions of your changes.

See `CONTRIBUTING.md` for more information.

## License

This project is licensed under the [MIT License](LICENSE). See the LICENSE file for details.

**Disclaimer:**  
Trading and MEV strategies involve significant risk. Use ON1Builder at your own risk. The developers are not responsible for any losses incurred using this software.
