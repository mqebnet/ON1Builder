# 0xBuilder: Your MEV Toolkit for Ethereum

[![Python Version](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Latest Release](https://img.shields.io/badge/Release-0.22.03_beta-black.svg)](https://github.com/John0n1/0xBuilder/releases/tag/v0.22.3-beta)


**Maximize your Ethereum trading potential with 0xBuilder, a powerful framework designed for high-frequency trading and Maximal Extractable Value (MEV) strategies.**

0xBuilder empowers you with tools for:

*   **Front-running**: Capitalize on pending transactions for profit.
*   **Back-running**: Strategically execute trades after specific transactions.
*   **Sandwich Attacks**:  Profit from manipulating transaction order in decentralized exchanges.
*   **Flash Loan Arbitrage**: Leverage flash loans for zero-capital arbitrage opportunities.

<p align="center">
  <a href="https://mermaid.live/edit#pako:eNp1kttu2zAMhl-FELBdNUBPF2uArcg5aZe0WHLTysbA2YotRJYMWd7qJtmzj5GdwNkwX8j-yU808ZNbFplYsC5LLOYprIaBBnp6_PKtX0oVCwtLh9aF0Ol8gT6faekkKvkudQIDo9cyCesrfU8MOAW1iNwhPxEuBU31G2TgkSGfiyw3RsHcUDVjCW2AoQdGvKdRVe8CVhZ1gVTM6KJBRh4Zb5-tWUuHP5SApzw31pVUq7rf11R9jg_s7kUUO5jwAaqoVOgbQx3D6E1EpVdzUzho1Vs6i04kVdiusjBtZgdkVJ2e-IamV3xsjXadb6UOzxLXvI_R5t_4DV9SG79klELPOUJCOMvf8l78E3Uk4r8bqs_pledmfKywSJVBfd_kp9d1plE3Z-r2qGo98_KBP-VOZvVUF4Z-Ch9hgkfPHzz0SA2vhau-L4Sjycg8J68GqYg2RePM55UtBfyGu8vO3acPzeXH_9l3SsJXXg9DFO2J1yt38rn9ZhcsEzZDGdPmbg_xgLlUZCJgXfqM0W4CFug9cVg6s6x0xLqOmrtg1pRJyrprVAWpMo_J2aFEWv_sFM1Rvxpz1Ps_TBb5rA">
    <img src="https://mermaid.ink/img/pako:eNp1kttu2zAMhl-FELBdNUBPF2uArcg5aZe0WHLTysbA2YotRJYMWd7qJtmzj5GdwNkwX8j-yU808ZNbFplYsC5LLOYprIaBBnp6_PKtX0oVCwtLh9aF0Ol8gT6faekkKvkudQIDo9cyCesrfU8MOAW1iNwhPxEuBU31G2TgkSGfiyw3RsHcUDVjCW2AoQdGvKdRVe8CVhZ1gVTM6KJBRh4Zb5-tWUuHP5SApzw31pVUq7rf11R9jg_s7kUUO5jwAaqoVOgbQx3D6E1EpVdzUzho1Vs6i04kVdiusjBtZgdkVJ2e-IamV3xsjXadb6UOzxLXvI_R5t_4DV9SG79klELPOUJCOMvf8l78E3Uk4r8bqs_pledmfKywSJVBfd_kp9d1plE3Z-r2qGo98_KBP-VOZvVUF4Z-Ch9hgkfPHzz0SA2vhau-L4Sjycg8J68GqYg2RePM55UtBfyGu8vO3acPzeXH_9l3SsJXXg9DFO2J1yt38rn9ZhcsEzZDGdPmbg_xgLlUZCJgXfqM0W4CFug9cVg6s6x0xLqOmrtg1pRJyrprVAWpMo_J2aFEWv_sFM1Rvxpz1Ps_TBb5rA?type=png" alt="0xBuilder Architecture Diagram">
  </a>
</p>

> **⚠️ Important: Under Active Development**
> 0xBuilder is currently in active development. Expect potential bugs and use with extreme caution. **Your use is at your own risk.**

## Key Features

0xBuilder provides a comprehensive suite of features for effective MEV exploitation:

*   **Mempool Monitoring**: Real-time analysis of the Ethereum mempool to identify profitable opportunities.
*   **Advanced Trading Strategies**: Implements sophisticated strategies like front-running, back-running, sandwich attacks, and flash loan arbitrage.
*   **Flash Loan Integration (Aave V3)**:  Utilizes Aave V3 flash loans for capital-efficient trading.
*   **Data-Driven Decisions**: Integrates with cryptocurrency APIs for real-time market data analysis.
*   **Gas Optimization**: Dynamically adjusts gas prices to minimize transaction costs.
*   **Robust Transaction Management**: Includes nonce management and atomic transaction bundling for reliable execution.
*   **Risk Mitigation**: Built-in safety checks and risk assessments to protect operations.
*   **DeFi Protocol Support**: Seamlessly interacts with popular DeFi platforms like Uniswap, Sushiswap, and Aave.
*   **Highly Configurable**: Adaptable to various wallets, tokens, trading pairs, and strategies.
*   **Detailed Logging**: Comprehensive logs for performance analysis, debugging, and strategy refinement.

## Project Structure

The project is organized into the following key directories:

```
/0xBuilder/
├── abi/                      # Smart Contract ABIs (JSON format)
├── contracts/              # Solidity Smart Contracts (Flashloan, Interfaces)
├── linear_regression/       # Machine Learning for Price Prediction (Models, Data)
├── python/                 # Core Python Bot Logic
│   ├── safetynet.py         # Risk Management & Safety Checks
│   ├── strategynet.py       # MEV Strategy Implementation
│   ├── mempoolmonitor.py    # Mempool Monitoring Engine
│   ├── marketmonitor.py     # Market Data Analysis
│   ├── main.py              # Main Bot Entry Point
│   ├── transactioncore.py   # Transaction Handling & Execution
│   ├── maincore.py          # Core Application Logic
│   ├── noncecore.py         # Nonce Management
│   ├── apiconfig.py         # API Integration & Data Handling
│   ├── configuration.py      # Configuration Loading
│   ├── abiregistry.py       # Centralized ABI Management
│   ├── 0xBuilder.log        # Default Log File
│   ├── __init__.py
│   └── pyutils/              # Utility Modules
├── utils/                    # JSON Configuration Files (Tokens, Signatures)
├── .env                      # Environment Variable Configuration
└── requirements.txt          # Python Dependencies
```

## Prerequisites

Before you begin, ensure your system meets these requirements:

### System Requirements

*   **Operating System**: Linux (Ubuntu 20.04+ recommended), Windows 10/11, macOS 12+
*   **Internet**: Stable internet connection (50Mbps+ recommended)
*   **Hardware**:
    *   CPU: 4+ cores, 3.0GHz+ (e.g., Intel i7 or equivalent)
    *   RAM: 16GB minimum
    *   Storage: 1.5TB NVMe SSD (for local Ethereum node synchronization)

### Software Requirements

*   **Python**: Version 3.12 or higher (check with `python3 --version`)
*   **Ethereum Execution Client**: Choose one of the following:
    *   [Geth](https://geth.ethereum.org/) (Go, **Recommended** for stability and performance)
    *   [Nethermind](https://www.nethermind.io/) (C#/.NET)
    *   [Besu](https://besu.hyperledger.org/) (Java)
    *   [Erigon](https://github.com/ledgerwatch/erigon) (Go)
    *   [Reth](https://reth.rs/) (Rust)
    *   [EthereumJS](https://github.com/ethereumjs/ethereumjs-monorepo) (TypeScript, Sepolia/Holesky testnets only)

### Ethereum Node Setup

You'll need to run your own Ethereum Execution and Consensus(Beacon) client to interact with the blockchain. **Geth and Prysm is highly recommended for its stability and speed.**

**Example: Setting up Geth**

1.  **Install Geth**: Follow the [official Geth installation guide](https://geth.ethereum.org/docs/install-and-build/installing-geth).

2.  **Start Geth Node**:  Run the following command, adjusting paths as needed:

    ```bash
    ./geth --mainnet --syncmode snap --http --http.api eth,net,admin,web3,txpool --ws --ws.api eth,net,admin,web3,txpool --maxpeers 100 --cache 16000 --ipcpath ~/0xBuilder/geth.ipc --allow-insecure-unlock --http.corsdomain "*"
    ```
    *   **Explanation**:
        *   `--mainnet`: Connects to the Ethereum main network.
        *   `--syncmode snap`: Uses snap sync for faster synchronization.
        *   `--http` & `--ws`: Enables HTTP and WebSocket APIs.
        *   `--http.api ...` & `--ws.api ...`:  Specifies API methods to expose.
        *   `--ipcpath`: Sets the path for IPC communication (used by the bot).
        *   `--allow-insecure-unlock`: Allows unlocking accounts via HTTP (use with caution).
        *   `--http.corsdomain "*" `:  Allows CORS requests from any domain (for development).

3.  **Verify Sync**:  Attach to your running Geth instance to monitor synchronization:

    ```bash
    geth attach ipc:/path/to/geth.ipc  # Replace with your actual ipc path
    > eth.syncing
    ```
    Wait until `eth.syncing` returns `false` or `null` to confirm full synchronization.

**Beacon Node**
 [Prysm](https://docs.prylabs.network/docs/install/install-with-script) or [Lighthouse](https://lighthouse-book.sigmaprime.io/installation.html) for setup instructions.

```bash
./prysm.sh beacon-chain --mainnet --checkpoint-sync-url=https://beaconcha.in/api/v1/genesis --genesis-beacon-api-url=https://beaconcha.in --http-web3provider=/path/to/geth.ipc
```

## Installation

Follow these steps to install and set up 0xBuilder:

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/John0n1/0xBuilder.git
    cd 0xBuilder
    ```

2.  **Create a Virtual Environment (Recommended):**

    ```bash
    # Linux/macOS
    python3 -m venv venv
    source venv/bin/activate

    # Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```
    This isolates project dependencies and prevents conflicts.

3.  **Install Dependencies:**

    ```bash
    python -m pip install --upgrade pip  # Upgrade pip
    pip install -r requirements.txt      # Install project dependencies
    ```

## Configuration

0xBuilder uses environment variables and JSON configuration files for customization.

### Environment Variables (`.env` file)

1.  **Copy Example `.env`:**

    ```bash
    # Linux/macOS
    cp .env.example .env

    # Windows
    copy .env.example .env
    ```

2.  **Edit `.env`**: Open the `.env` file and configure the following:

    ```ini
    # API Keys (Register for free trial keys)
    ETHERSCAN_API_KEY=YOUR_ETHERSCAN_API_KEY
    INFURA_PROJECT_ID=YOUR_INFURA_PROJECT_ID
    COINGECKO_API_KEY=YOUR_COINGECKO_API_KEY
    COINMARKETCAP_API_KEY=YOUR_COINMARKETCAP_API_KEY
    CRYPTOCOMPARE_API_KEY=YOUR_CRYPTOCOMPARE_API_KEY

    # Ethereum Node Endpoints (Adjust to your Geth setup)
    HTTP_ENDPOINT=http://127.0.0.1:8545
    WS_ENDPOINT=wss://127.0.0.1:8546
    IPC_ENDPOINT=~/0xBuilder/geth.ipc  # Or your Geth IPC path

    # Wallet Configuration
    WALLET_ADDRESS=0xYourEthereumWalletAddress
    WALLET_KEY=YOUR_PRIVATE_KEY # **Keep your private key secure!**

    # Token Configuration Files (Paths are relative to the project root)
    TOKEN_ADDRESSES=utils/token_addresses.json
    TOKEN_SYMBOLS=utils/token_symbols.json
    ERC20_SIGNATURES=utils/erc20_signatures.json

    # DEX Router Addresses (Mainnet addresses)
    UNISWAP_ADDRESS=0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D # Uniswap V2 Router
    SUSHISWAP_ADDRESS=0xd9e1cE17F2641f24aE83637ab66a2cca9C378B9F # Sushiswap Router

    # ABI File Paths (Paths are relative to the project root)
    UNISWAP_ABI=abi/uniswap_abi.json
    SUSHISWAP_ABI=abi/sushiswap_abi.json
    ERC20_ABI=abi/erc20_abi.json

    # Flash Loan Contract Configuration
    AAVE_FLASHLOAN_ADDRESS=0xYourDeployedFlashloanContractAddress # Your deployed contract address
    AAVE_POOL_ADDRESS=0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2 # Aave V3 Pool Address
    ```

### JSON Configuration Files (`utils/` directory)

Customize these files to define tokens and function signatures:

*   **`token_addresses.json`**:  Map token symbols to their contract addresses.
    ```json
    {
      "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
      "DAI": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
      "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eb48"
      // ... more tokens
    }
    ```

*   **`token_symbols.json`**: Map token symbols to API identifiers (for CoinGecko, CoinMarketCap, etc.).
    ```json
    {
      "WETH": "ethereum",
      "DAI": "dai",
      "USDC": "usd-coin"
      // ... more tokens
    }
    ```

*   **`erc20_signatures.json`**: Define ERC20 function signatures (selectors). **Generally, you don't need to modify this.**
    ```json
    {
      "transfer": "0xa9059cbb",
      "approve": "0x095ea7b3",
      "transferFrom": "0x23b872dd",
      "balanceOf": "0x70a08231",
      "totalSupply": "0x18160ddd",
      "decimals": "0x313ce567",
      "name": "0x06fd41ca",
      "symbol": "0x95d89b41",
      "allowance": "0xdd62ed3e"
    }
    ```

## Deploy Your Flash Loan Contract

To utilize flash loan arbitrage strategies, you need to deploy the provided `SimpleFlashloan.sol` contract.

**Recommended Deployment Method: Remix IDE**

1.  **Open Remix Online IDE**: Go to [https://remix.ethereum.org/](https://remix.ethereum.org/).
2.  **Create `SimpleFlashloan.sol`**: Create a new file named `SimpleFlashloan.sol` in Remix.
3.  **Paste Contract Code**: Copy the Solidity code below and paste it into `SimpleFlashloan.sol`.

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

4.  **Compile**: In Remix, navigate to the Solidity compiler tab and compile `SimpleFlashloan.sol` using Solidity version `0.8.20` or later.
5.  **Deploy**:
    *   Go to the "Deploy & Run Transactions" tab.
    *   Select "Injected Provider - MetaMask" as the Environment (connect your wallet).
    *   Ensure the correct network (e.g., Mainnet, Sepolia for testing) is selected in MetaMask.
    *   In the "Contract" dropdown, choose `SimpleFlashLoan - contracts/SimpleFlashloan.sol`.
    *   **Important**: In the `addressProvider (address)` field, you need to provide the **Aave V3 Pool Addresses Provider** address for the network you are deploying to.  You can find these addresses in the [Aave documentation](https://docs.aave.com/developers/deployed-contracts/v3-mainnet/ethereum-v3). For Mainnet, it's `0x2f39d218133AFaB97F5bf874E1b0cDF5d987aa37`.
    *   Click "Deploy". Confirm the transaction in MetaMask.
6.  **Update `.env`**: Once deployed, copy the contract address from Remix (it will appear after successful deployment) and update your `.env` file:

    ```ini
    AAVE_FLASHLOAN_ADDRESS=0xYourDeployedContractAddress
    ```

## Register for API Keys

To access real-time market data and blockchain information, you'll need to register for API keys from the following services. Most offer free tiers for development and testing:

1.  **Infura**: [https://infura.io/](https://infura.io/) (Ethereum node infrastructure)
2.  **Etherscan**: [https://etherscan.io/apis](https://etherscan.io/apis) (On-chain data and transaction insights)
3.  **CoinGecko**: [https://www.coingecko.com/en/api](https://www.coingecko.com/en/api) (Cryptocurrency market data)
4.  **CoinMarketCap**: [https://coinmarketcap.com/api/](https://coinmarketcap.com/api/) (Cryptocurrency market data)
5.  **CryptoCompare**: [https://min-api.cryptocompare.com/](https://min-api.cryptocompare.com/) (Cryptocurrency market data)

After registering, add your API keys to the `.env` file as described in the [Configuration](#configuration) section.

## Run the Bot

Before running the bot, ensure:

*   **Ethereum Node**: Your Ethereum Execution client (Geth or another client) is fully synchronized and running.
*   **Configuration**: You have correctly configured the `.env` file and JSON configuration files.
*   **Wallet Funding**: Your configured wallet (`WALLET_ADDRESS`) has sufficient ETH to cover transaction gas costs.

**Steps to Run:**

1.  **Activate Virtual Environment**:

    ```bash
    # Linux/macOS
    source venv/bin/activate

    # Windows
    .\venv\Scripts\activate
    ```

2.  **Start 0xBuilder**:

    ```bash
    python python/main.py
    ```

**Monitoring and Control:**

*   **Logs**: Check `python/0xBuilder.log` for detailed bot activity, strategy execution, and potential errors.
*   **Console Output**: Monitor the terminal for real-time status updates and information.
*   **Stopping the Bot**: Press `Ctrl+C` in the terminal to gracefully shut down the bot.

**Performance Tips:**

*   **Node Synchronization**: A fully synced Ethereum node is crucial for optimal performance.
*   **API Rate Limits**: Be mindful of API rate limits. Consider upgrading to paid tiers if needed for higher usage.
*   **Gas Management**: Ensure your wallet has enough ETH for gas. Monitor gas prices and adjust bot parameters if necessary.
*   **Log Analysis**: Regularly review logs to identify issues, optimize strategies, and track performance.
*   **Dependency Updates**: Keep your Python dependencies updated for performance improvements and security patches.

## Strategies Implemented

0xBuilder includes a range of MEV strategies, categorized for clarity:

**Front-Running Strategies:**

*   **Aggressive Front-Running**:  Attempts to front-run any profitable transaction.
*   **Predictive Front-Running**: Uses market analysis to predict and front-run specific types of transactions.
*   **Volatility Front-Running**: Targets transactions during periods of high price volatility.
*   **Advanced Front-Running**: Combines multiple signals and conditions for sophisticated front-running.

**Back-Running Strategies:**

*   **Price Dip Back-Running**: Executes back-running trades after detecting price dips.
*   **Flashloan Back-Running**: Utilizes flash loans to back-run specific transaction patterns.
*   **High Volume Back-Running**: Targets back-running opportunities during high trading volume periods.
*   **Advanced Back-Running**:  Employs complex logic and market data for optimized back-running.

**Sandwich Attack Strategies:**

*   **Flash Profit Sandwich Attack**: Aims to profit from sandwich attacks using flash loans.
*   **Price Boost Sandwich Attack**: Targets sandwich attacks where price manipulation is likely.
*   **Arbitrage Sandwich Attack**: Combines sandwich attacks with arbitrage opportunities.
*   **Advanced Sandwich Attack**: Leverages sophisticated techniques for more effective sandwich attacks.

**Other Strategies:**

*   **High-Value ETH Transfer Monitoring**: Detects and potentially capitalizes on large ETH transfers.

## Logging and Monitoring

0xBuilder provides detailed logging to help you understand bot behavior and performance. Logs are written to `python/0xBuilder.log`.

**Log Information Includes:**

*   **Mempool Activity**: Detection of potential MEV opportunities in real-time.
*   **Strategy Execution**: Details on strategy execution attempts, successes, and failures.
*   **Profit Tracking**: Records of profitable transactions and overall performance metrics.
*   **Error and Exception Reporting**: Immediate logging of errors and exceptions for quick troubleshooting.
*   **Detailed Activity Logs**: Comprehensive records of bot actions, market analysis, and transactions.

You can adjust logging verbosity in `python/maincore.py` by modifying the `setup_logging()` function.  Consider increasing the log level to `DEBUG` for more detailed information during troubleshooting.

## Troubleshooting

**Common Issues and Solutions:**

| Issue                               | Solution                                                                     |
| ----------------------------------- | ---------------------------------------------------------------------------- |
| **Node Connection Errors**            | Verify your Ethereum node is running and that the endpoints in `.env` are correct. |
| **API Rate Limit Exceeded**         | Implement request throttling or upgrade your API plan for higher limits.       |
| **Insufficient ETH Balance**        | Fund your wallet (`WALLET_ADDRESS`) with enough ETH to cover gas costs.          |
| **Nonce Synchronization Problems**  | Restart the bot or manually reset the nonce if necessary.                     |
| **Ethereum Node Not Synced**       | Ensure your Ethereum Execution client is fully synchronized with the network.  |
| **Flash Loan Contract Issues**       | Verify your flash loan contract is deployed correctly and the address is correct in `.env`. |

[![Report Issue](https://img.shields.io/badge/Report_issue-red.svg)](https://github.com/John0n1/0xBuilder/issues)

**Debugging Tips:**

1.  **Enable Verbose Logging**: Set the logging level to `DEBUG` in `python/maincore.py` for detailed output.
2.  **Check Dependencies**: Ensure all Python dependencies in `requirements.txt` are correctly installed and up-to-date (`pip install -r requirements.txt`).
3.  **Verify Contract Deployment**: Use a block explorer (like Etherscan) to confirm your flash loan contract is deployed to the correct address and is functioning as expected.
4.  **Test on Testnets**: Thoroughly test your configuration and strategies on test networks like Sepolia or Holesky before deploying to the mainnet.

## Contributing

Contributions to 0xBuilder are **highly welcome and encouraged!** ❤️

**How to Contribute:**

1.  **Fork the Repository**: Fork the 0xBuilder repository to your own GitHub account.
2.  **Create a Branch**: Create a new branch for your feature or bug fix (e.g., `git checkout -b feature/new-strategy`).
3.  **Code and Test**: Implement your changes, ensuring your code adheres to PEP 8 style guidelines. Write comprehensive unit tests to cover your changes.
4.  **Submit a Pull Request**: Create a well-documented pull request from your branch to the main repository's `main` branch.  Clearly describe your changes and their benefits.

Please review `CONTRIBUTING.md` (if present in the repository) for more detailed contribution guidelines.

## License

0xBuilder is released under the [MIT License](LICENSE).  See the `LICENSE` file for details.

---

**Disclaimer:**  Trading and MEV strategies involve significant financial risk. Use 0xBuilder at your own risk. The developers are not responsible for any losses incurred while using this software. Always conduct thorough testing and understand the risks involved before deploying to the mainnet with real funds.
