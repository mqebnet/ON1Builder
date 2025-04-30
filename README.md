# ON1Builder

![ON1Builder Logo](https://github.com/user-attachments/assets/59e03abe-67ee-4195-9030-63f49c48e46f)

[![Built with Python 3.12](https://img.shields.io/badge/Built%20with-Python%203.12-blue?logo=python)](https://www.python.org/)
[![ON1Builder Wiki](https://img.shields.io/badge/ON1Builder-Wiki-blue?logo=GitHub&logoColor=white)](https://github.com/John0n1/ON1Builder/wiki)
[![Last Commit](https://img.shields.io/github/last-commit/John0n1/ON1Builder?display_timestamp=committer&logo=GitHub&color=white)](https://github.com/John0n1/ON1Builder/commits/main)


# ON1Builder

**High-performance, modular, and production-ready MEV (Maximum Extractable Value) searcher bot for Ethereum and EVM-compatible chains.**
---

**ON1Builder epresents a significant leap forward.**  this version has undergone a meticulous refactoring process (addressing issues transforming it into a robust, efficient, and safe platform engineered for the intense demands of mainnet MEV operations.)

It continuously monitors the Ethereum mempool (or pending blocks), identifies profitable opportunities (like front-running, back-running, sandwiches), evaluates their safety and profitability, and executes them strategically using sophisticated risk management, dynamic gas handling, and near-bulletproof nonce management.

**If you're serious about MEV, this is the foundation you need.**

## 🔥 Why ON1Builder V2 Stands Out

This isn't just code; it's carefully engineered software incorporating lessons learned from real-world MEV challenges:

1.  **🚀 Performance & Efficiency:**
    *   **Async Native:** Built entirely on Python's `asyncio` for superior concurrency and I/O handling.
    *   **Optimized Data Structures:** Uses `@dataclass(slots=True)` for lightweight objects and `TTLCache` for efficient, memory-bound caching, preventing leaks (A1, A9).
    *   **Non-Blocking Core:** Critical paths avoid blocking the event loop, ensuring responsiveness even under heavy load (A3 uses `random.choices` or `asyncio.to_thread`).
    *   **Efficient Logging:** Parameterized logging statements (`logger.debug("Value: %s", var)`) avoid costly f-string formatting unless the log level is active (A2).
    *   **Concurrent API Calls:** Fetches data from multiple external price APIs concurrently with timeouts for speed and resilience (A12).
    *   **Optimized Data Pipelines:** CSV training data is appended directly, avoiding full reloads, with options for stream processing (A13). Wei-native math used where appropriate (A14).

2.  **🛡️ Uncompromising Reliability & Safety:**
    *   **Bulletproof Nonce Management:** `NonceCore` employs locking (`asyncio.Lock`), caching with TTL, pending transaction tracking, and robust chain synchronization to *drastically* reduce nonce errors (the bane of many bots) (A7).
    *   **Dedicated SafetyNet:** A specialized component performs critical pre-flight checks:
        *   Accurate Profit Calculation: Uses `Decimal` types consistently to avoid floating-point precision issues (A8).
        *   Dynamic Slippage Control: Adjusts based on network congestion.
        *   Gas Price Ceilings: Prevents overpaying in volatile gas markets.
    *   **Atomic Persistence:** Reinforcement learning weights are saved atomically (temp-file + rename) to prevent corruption on crashes or shutdowns (A4).
    *   **Graceful Shutdown & Resource Management:** Handles `SIGINT`/`SIGTERM`, cancels all background tasks cleanly, awaits their completion, saves final states (like RL weights), and closes connections (A10). No dangling coroutines or resource leaks.
    *   **Correct Initialization:** Ensures components are initialized in the correct dependency order (A5).
    *   **Verified Signing:** Uses the correct `Account.sign_transaction` compatible with `AsyncWeb3` (A6).
    *   **Strict Type Safety:** Enforced with `mypy --strict`, catching potential errors before runtime.
    *   **Rigorous Testing:** Comprehensive `pytest` suite covering core functionalities, race conditions, and critical paths, ensuring high code coverage (A17).

3.  **🧩 Modular & Maintainable Design:**
    *   **Clear Separation of Concerns:** Functionality is cleanly divided into components (`MainCore`, `TransactionCore`, `StrategyNet`, `SafetyNet`, `MempoolMonitor`, etc.), making the codebase easier to understand, modify, and extend.
    *   **Clean Configuration:** Configuration is split into logical modules (core, paths, limits) accessed via a unified facade, simplifying management (A15).
    *   **Flexible Strategy Framework:** `StrategyNet` allows easy addition/modification of MEV strategies. Strategies now directly return `(success, profit_decimal)` for precise performance tracking and RL updates (A11).
    *   **Dependency Injection:** Components receive their dependencies during initialization, promoting clarity (A5).

4.  **💡 Intelligent Operation:**
    *   **Reinforcement Learning (RL):** `StrategyNet` learns! It adjusts strategy selection based on real-world performance (profit, success rate, execution time), prioritizing what actually works. Weights are persisted across restarts (A4, A11).
    *   **Adaptive Operation:** Dynamically adjusts gas fees (EIP-1559 aware) and slippage based on current network conditions.
    *   **Resilient Price Feeds:** Aggregates data from multiple APIs using weighted averaging for more reliable price information (A12).

5.  **👀 Superior Observability:**
    *   **Structured Logging:** Logs include `component` and `tx_hash` context, enabling powerful filtering, aggregation, and easier debugging, especially when feeding logs into analysis tools or the Web UI (A16).
    *   **Real-time Web UI:** Includes a Flask + SocketIO interface for live status monitoring, streaming structured logs, basic metrics display, and start/stop controls.

## 🏗️ Architecture V2

ON1Builder's refactored architecture emphasizes modularity and clear responsibilities:

*   **`MainCore`**: Central orchestrator; manages component lifecycle, main loop, signals, and shutdown.
*   **`Configuration` (`configuration/` facade)**: Loads and provides validated settings (core, paths, limits) from `.env` and files (A15).
*   **`APIConfig`**: Handles external API interactions (price feeds, exchanges) with concurrency, caching, and rate limiting (A12).
*   **`NonceCore`**: Provides thread-safe, reliable nonce management with caching and synchronization (A7).
*   **`SafetyNet`**: Performs pre-transaction risk and profitability checks using precise `Decimal` math (A8).
*   **`TransactionCore`**: Responsible for building, signing (A6), simulating, and sending transactions. Executes strategy logic, returning profit (A11, A14).
*   **`MarketMonitor`**: Monitors market data, manages historical data persistence (A13), and potentially trains/runs predictive models. Handles task cancellation correctly (A10).
*   **`MempoolMonitor`**: Watches for pending transactions, performs initial filtering/analysis, and uses `TTLCache` to avoid reprocessing (A9).
*   **`StrategyNet`**: Selects the best strategy via RL, executes it via `TransactionCore`, updates metrics based on direct profit return (A11), and persists weights atomically (A4). Uses non-blocking selection (A3).
*   **`Web UI (app.py)`**: Flask/SocketIO interface for monitoring and basic control, displaying structured logs (A16).
*   **`LoggingConfig`**: Centralized logging setup supporting colored console output and structured JSON (A2, A16).
*   **`ABIRegistry`**: Manages loading and validation of contract ABIs.

*(A5 ensures `TransactionCore` is instantiated before `MarketMonitor`)*

## 🚀 Getting Started

### Prerequisites

*   **Python:** 3.10+ (developed/tested with 3.10/3.11)
*   **pip:** For installing dependencies.
*   **Git:** For cloning the repository.
*   **OS:** Linux strongly recommended for performance and compatibility (`asyncio`, signal handling). macOS may work. Windows is not recommended for production deployments.
*   **Ethereum Node RPC Endpoint:** **Crucial.** You need reliable, low-latency access to an Ethereum node (e.g., private node, Flashbots Protect RPC, specialized RPC provider like Blocknative, Alchemy, Infura). WebSocket (`wss://`) is generally preferred over HTTP (`https://`) for mempool monitoring. Archive node capabilities might be needed depending on strategy complexity.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/John0n1/ON1Builder.git
    cd ON1Builder
    ```

2.  **Set up a Python virtual environment (Highly Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Configuration (`.env` file)

1.  **Copy the example environment file:**
    ```bash
    cp .env.example .env
    ```

2.  **Edit the `.env` file with YOUR details:**

    ```dotenv
    # ======================================
    #        CORE & NETWORK SETTINGS
    # ======================================
    # WebSocket Preferred for Mempool monitoring
    WEBSOCKET_ENDPOINT=wss://mainnet.infura.io/ws/v3/YOUR_INFURA_PROJECT_ID
    # Fallback HTTP Endpoint (Must provide at least one endpoint)
    HTTP_ENDPOINT=https://mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID
    # IPC_ENDPOINT=/path/to/your/node/geth.ipc # If using local IPC

    # ======================================
    #          WALLET CONFIGURATION
    # ======================================
    # Wallet Private Key (MUST start with 0x) - KEEP SECURE!
    WALLET_KEY=0xyour_64_char_hex_private_key_here_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # Wallet Address (derived from key, used for validation)
    WALLET_ADDRESS=0xYourCorrespondingWalletAddressHerexxxxxxxx

    # ======================================
    #          API KEYS (Recommended)
    # ======================================
    # Services like Etherscan, Coingecko Pro, etc., for better data/rate limits
    ETHERSCAN_API_KEY=YOUR_ETHERSCAN_API_KEY
    COINGECKO_API_KEY=YOUR_COINGECKO_PRO_API_KEY # Blank for free tier
    # COINMARKETCAP_API_KEY=...
    # CRYPTOCOMPARE_API_KEY=...
    # Add others if used...

    # ======================================
    #      LIMITS & THRESHOLDS (Review!)
    # ======================================
    # Max gas price in Gwei bot will ever use
    MAX_GAS_PRICE_GWEI=250
    # Minimum required profit in ETH (as float string) for SafetyNet check
    MIN_PROFIT_ETH="0.0005"
    # Minimum ETH balance required in wallet
    MIN_BALANCE_ETH="0.02"
    # Default slippage for swaps (e.g., 0.005 = 0.5%)
    SLIPPAGE_DEFAULT=0.005

    # ======================================
    #         OPERATIONAL SETTINGS
    # ======================================
    LOG_LEVEL=INFO # DEBUG, INFO, WARNING, ERROR
    # Directory for runtime files (weights, data) relative to project root
    RUNTIME_DIR="./runtime"

    # ======================================
    #      CONTRACT ADDRESSES (Mainnet)
    # ======================================
    # Review and ensure these match the target network (if not mainnet)
    WETH_ADDRESS=0xC02aaa39b223FE8D0a0e5C4F27eAD9083C756Cc2
    USDC_ADDRESS=0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48
    UNISWAP_ADDRESS=0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D # V2 Router
    SUSHISWAP_ADDRESS=0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F # V2 Router
    # AAVE_POOL_ADDRESS=... # Ensure correct Aave Pool address for network
    # AAVE_FLASHLOAN_ADDRESS=... # Address of YOUR deployed Flashloan receiver/contract

    # Other settings from .env.example can be adjusted as needed.
    ```

**🔒 SECURITY WARNING:** Your `WALLET_KEY` grants full control over your funds. **NEVER** share it or commit the `.env` file to version control (it's already in `.gitignore`). Secure it appropriately.

## ▶️ Running the Bot

1.  **Activate the virtual environment:**
    ```bash
    source venv/bin/activate
    ```

2.  **Run the main application module:**
    ```bash
    python -m ON1Builder.python.main
    ```
    *(This assumes you run it from the root `ON1Builder` directory)*

The bot will initialize, connect, and begin monitoring/searching. Logs will appear in your console.

3.  **Stopping Gracefully:**
    *   Press `Ctrl+C` in the terminal. The bot is designed to catch this (`SIGINT`) and perform a clean shutdown (cancelling tasks, saving weights, closing connections - A10, A4).
    *   Use the "Stop Bot" button in the Web UI (see below).

## 🖥️ Web UI & Interaction

Monitor and interact with your running ON1Builder instance via the built-in web interface:

1.  **Start the bot** (as described above).
2.  **Open a web browser** and go to `http://localhost:5000` (or the configured host/port).

**Features:**
*   **Status:** See if the bot is running and the health/status of its components.
*   **Live Logs:** View a real-time stream of logs, filterable thanks to structured logging (A16). See `component` and `tx_hash` context.
*   **Metrics:** View key operational data (balance, profit estimates, queue sizes, etc.).
*   **Controls:** Start and (most importantly) Stop the bot gracefully.

**API Endpoints (for programmatic interaction):**
*   `GET /status`: Bot and component status.
*   `GET /metrics`: Current operational metrics.
*   `POST /start`: Start the bot (if not running).
*   `POST /stop`: Request a graceful shutdown.
*   `GET /logs`: Get recent buffered logs as JSON.

## ռ MEV Strategies Implemented

ON1Builder V2 includes logic or robust placeholders for common MEV strategies:

*   **Front-running:** (Basic, Aggressive, Predictive, Volatility-based)
*   **Back-running:** (Basic, Price Dip, High Volume)
*   **Sandwich Attacks:** (Requires careful tuning and implementation)
*   **Flashloans:** Integrated with strategies like `flashloan_back_run` (Requires a deployed flashloan-capable contract configured via `AAVE_FLASHLOAN_ADDRESS`).

**Key Improvement (A11):** Strategies now return their calculated profit (`Decimal` ETH) directly, allowing `StrategyNet`'s RL component to learn based on actual, precise outcomes, not just success/failure or shared state variables.

**Note:** The profitability and success of these strategies are *highly* dependent on your configuration (gas limits, profit thresholds), network latency, gas price dynamics, and the specific implementation details within `TransactionCore`. **Default strategy implementations may require significant tuning or replacement for consistent profitability.**

## 🛠️ Development & Testing

### Development Setup

1.  Complete the **Installation** steps.
2.  Install development tools:
    ```bash
    pip install -r requirements-dev.txt
    ```

### Testing (A17)

ON1Builder V2 emphasizes testability.

*   Tests are located in the `tests/` directory.
*   Uses `pytest` with `pytest-asyncio` for async testing.
*   **Requires a local development blockchain** (e.g., Hardhat node `npx hardhat node`, Anvil `anvil`) for tests involving transaction signing and sending (like A6 signing test, nonce tests).
*   Configure your test node endpoint in a separate `.env.test` file (gitignored). `pytest` should ideally pick this up or be configured to use it.
*   **Run tests:**
    ```bash
    # Ensure dev chain is running & .env.test configured
    pytest tests/
    ```
*   **Run tests with coverage:**
    ```bash
    coverage run -m pytest tests/
    coverage report -m
    # coverage html # For a detailed HTML report
    ```
    The CI pipeline enforces >= 90% coverage.

### Code Quality & Style

We use standard Python tooling:

*   **Formatter:** `black .`
*   **Import Sorting:** `isort .`
*   **Linter:** `flake8 .`
*   **Type Checking:** `mypy --strict .` (run this!)

Ensure all checks pass before submitting pull requests.

## 🙌 Contributing

Your contributions make ON1Builder better! Please adhere to these guidelines:

1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feat/your-feature-name`).
3.  Write clear, concise code with type hints.
4.  **Write tests** for your changes! Ensure overall coverage remains high.
5.  Run linters/formatters (`black`, `isort`, `flake8`, `mypy`).
6.  Commit using [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, etc.).
7.  Push to your fork and submit a Pull Request to the `main` branch.
8.  Discuss significant changes in an Issue first.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**MEV IS EXTREMELY RISKY. USE THIS SOFTWARE ENTIRELY AT YOUR OWN RISK.**

*   **No Profit Guarantee:** Trading cryptocurrencies and engaging in MEV involves substantial risk of financial loss. There is absolutely **no guarantee of profit** when using ON1Builder. Market conditions, network latency, gas fees, and strategy effectiveness can change rapidly.
*   **Risk of Loss:** You can lose your entire investment. Smart contract interactions can fail, transactions can be front-run by others, and strategies may perform unexpectedly. **Do not risk funds you cannot afford to lose.**
*   **Code Understanding:** Ensure you understand the code you are running, especially the strategy logic and risk parameters in the configuration.
*   **Security:** Protect your private keys (`WALLET_KEY`). Compromise means total loss of funds in that wallet.
*   **No Liability:** The developers and contributors of ON1Builder are **not liable** for any financial losses, damages, or other issues arising from the use of this software.
*   **Test Thoroughly:** **Always** test extensively on testnets (e.g., Goerli, Sepolia) with test funds before deploying on the mainnet. Validate strategy logic and risk parameters carefully.

## 📞 Contact & Support

*   **Issues & Feature Requests:** Please use the [GitHub Issues](https://github.com/John0n1/ON1Builder/issues) tracker for the primary repository.
*   **Discussions:** Use the [GitHub Discussions](https://github.com/John0n1/ON1Builder/discussions) tab for questions and broader topics.

---