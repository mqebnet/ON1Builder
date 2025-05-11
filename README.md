# ON1Builder

ON1Builder is a powerful tool for building, testing, and deploying Ethereum trading strategies.

## Features

- **Real Transaction Execution**: Execute real transactions on Ethereum mainnet and testnets
- **Multi-Chain Support**: Run on multiple EVM-compatible chains simultaneously
- **Secure Secret Management**: Store sensitive information in HashiCorp Vault
- **Comprehensive Monitoring**: Prometheus metrics and Grafana dashboard
- **Alerting**: Slack and email alerts for profits and failures
- **Risk Management**: Slippage protection, gas price optimization, and profit thresholds
- **Dockerized Deployment**: Easy deployment with Docker Compose
- **CI/CD Integration**: GitHub Actions workflow for automated testing and deployment
- **Extensive Documentation**: API documentation, deployment guide, and security checklist

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Ethereum wallet with funds
- Infura or Alchemy API key
- Slack webhook URL (optional)
- SMTP server for email alerts (optional)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/John0n1/ON1Builder.git
   cd ON1Builder
   ```

2. Create a `.env` file with your configuration:
   ```bash
   # Ethereum RPC endpoints
   HTTP_ENDPOINT=https://sepolia.infura.io/v3/your-infura-key
   WEBSOCKET_ENDPOINT=wss://sepolia.infura.io/ws/v3/your-infura-key

   # Wallet configuration
   WALLET_ADDRESS=0xYourWalletAddress

   # Execution control
   DRY_RUN=true
   GO_LIVE=false
   ```

3. Start the application:
   ```bash
   docker-compose up -d
   ```

4. Check the application status:
   ```bash
   curl http://localhost:5001/healthz
   ```

## Production Deployment

For production deployment on Ethereum mainnet, follow these steps:

1. Install prerequisites (macOS):
   ```bash
   brew install --cask docker
   brew install docker-compose
   brew tap hashicorp/tap
   brew install hashicorp/tap/vault
   ```

2. Start Vault in dev mode for testing:
   ```bash
   vault server -dev -dev-root-token-id="root-token" &
   export VAULT_ADDR="http://127.0.0.1:8200"
   export VAULT_TOKEN="root-token"
   ```

3. Initialize Vault and store secrets:
   ```bash
   cd /path/to/ON1Builder
   chmod +x scripts/vault_init.sh
   ./scripts/vault_init.sh
   vault kv get secret/on1builder  # Verify secrets
   ```

4. Export required environment variables:
   ```bash
   export VAULT_ADDR="http://127.0.0.1:8200"
   export VAULT_TOKEN="root-token"  # or the token from vault_init.sh
   export HTTP_ENDPOINT="https://mainnet.infura.io/v3/your-infura-key"
   export WALLET_KEY="your-ethereum-private-key"
   export SMTP_PASSWORD="your-smtp-password"
   export SLACK_WEBHOOK_URL="your-slack-webhook-url"
   ```

5. Build and push the Docker image:
   ```bash
   chmod +x scripts/build_and_push.sh
   ./scripts/build_and_push.sh
   ```

6. Deploy with Docker Compose:
   ```bash
   chmod +x deploy_prod.sh
   ./deploy_prod.sh
   ```

7. Verify the deployment:
   ```bash
   # Health check
   curl -f http://localhost:5001/healthz

   # Metrics check
   curl http://localhost:5001/metrics | grep expected_profit_eth

   # Grafana dashboard
   open http://localhost:3000

   # Slack alert
   # Confirm receipt of test alert in Slack channel
   ```

For detailed deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).

## Configuration

ON1Builder can be configured using environment variables or a `.env` file:

### Core Configuration

- `HTTP_ENDPOINT`: Ethereum RPC endpoint URL
- `WEBSOCKET_ENDPOINT`: Ethereum WebSocket endpoint URL
- `WALLET_ADDRESS`: Ethereum wallet address
- `DRY_RUN`: Set to "true" to simulate transactions without sending them
- `GO_LIVE`: Set to "true" to send real transactions

### Vault Configuration

- `VAULT_ADDR`: URL of the HashiCorp Vault instance
- `VAULT_TOKEN`: Token for authenticating with Vault
- `VAULT_PATH`: Path in Vault where secrets are stored

### Alert Configuration

- `ALERT_THRESHOLD_ETH`: Threshold for profit alerts in ETH
- `SMTP_SERVER`: SMTP server address
- `SMTP_PORT`: SMTP server port
- `SMTP_USERNAME`: Username for the SMTP server
- `SMTP_PASSWORD`: Password for the SMTP server
- `ALERT_FROM_EMAIL`: Email address to send alerts from
- `ALERT_TO_EMAIL`: Email address to send alerts to
- `SLACK_WEBHOOK_URL`: Webhook URL for Slack notifications

### Transaction Configuration

- `MAX_GAS_PRICE_GWEI`: Maximum gas price in Gwei
- `SLIPPAGE_DEFAULT`: Default slippage tolerance
- `MIN_PROFIT`: Minimum profit threshold in ETH
- `MAX_TRANSACTION_VALUE`: Maximum transaction value in ETH
- `MAX_DAILY_TRANSACTIONS`: Maximum number of transactions per day

For a complete list of configuration options, see [config.yaml](config.yaml).

## Multi-Chain Support

ON1Builder supports running on multiple EVM-compatible chains simultaneously. To enable multi-chain support:

1. Configure the chains in `config_multi_chain.yaml` or using environment variables:
   ```yaml
   # Multi-chain settings
   CHAINS: "1,11155111,137"

   # Chain-specific settings for Ethereum Mainnet (Chain ID: 1)
   CHAIN_1_CHAIN_NAME: "Ethereum Mainnet"
   CHAIN_1_HTTP_ENDPOINT: "https://mainnet.infura.io/v3/your-infura-key"
   CHAIN_1_WEBSOCKET_ENDPOINT: "wss://mainnet.infura.io/ws/v3/your-infura-key"
   CHAIN_1_WALLET_ADDRESS: "0xYourMainnetWalletAddress"

   # Chain-specific settings for Sepolia Testnet (Chain ID: 11155111)
   CHAIN_11155111_CHAIN_NAME: "Sepolia Testnet"
   CHAIN_11155111_HTTP_ENDPOINT: "https://sepolia.infura.io/v3/your-infura-key"
   CHAIN_11155111_WEBSOCKET_ENDPOINT: "wss://sepolia.infura.io/ws/v3/your-infura-key"
   CHAIN_11155111_WALLET_ADDRESS: "0xYourSepoliaWalletAddress"

   # Chain-specific settings for Polygon Mainnet (Chain ID: 137)
   CHAIN_137_CHAIN_NAME: "Polygon Mainnet"
   CHAIN_137_HTTP_ENDPOINT: "https://polygon-rpc.com"
   CHAIN_137_WEBSOCKET_ENDPOINT: "wss://polygon-rpc.com"
   CHAIN_137_WALLET_ADDRESS: "0xYourPolygonWalletAddress"
   ```

2. Set the `CHAINS` environment variable in your `.env` file:
   ```bash
   CHAINS=1,11155111,137
   ```

3. Run the bot in dry-run mode:
   ```bash
   ON1Builder --dry-run
   ```

4. For production deployment with multi-chain support:
   ```bash
   GO_LIVE=true docker-compose -f docker-compose.multi-chain.yml up -d
   ```

5. Verify that the metrics endpoint shows data for all chains:
   ```bash
   curl http://localhost:5001/metrics | grep expected_profit_eth
   ```

The multi-chain support allows you to monitor and trade on multiple EVM-compatible chains simultaneously, with chain-specific configuration and metrics.

## API Endpoints

ON1Builder provides the following API endpoints:

- `GET /healthz`: Health check endpoint
- `GET /metrics`: Prometheus metrics endpoint
- `GET /status`: Current status of the bot
- `POST /start`: Start the bot
- `POST /stop`: Stop the bot
- `POST /api/test-alert`: Send a test alert
- `POST /api/simulate-transaction`: Simulate a transaction

For detailed API documentation, see [DOCS/API.md](DOCS/API.md).

## Security

ON1Builder takes security seriously. For security best practices and a comprehensive security checklist, see [SECURITY.md](SECURITY.md).

## Development

### Prerequisites

- Python 3.9+
- Node.js 14+
- Ethereum wallet with testnet funds

### Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

### Running Tests

```bash
pytest
```

### Building Documentation

```bash
cd docs
make html
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Ethereum Foundation](https://ethereum.org/)
- [Web3.py](https://web3py.readthedocs.io/)
- [HashiCorp Vault](https://www.vaultproject.io/)
- [Prometheus](https://prometheus.io/)
- [Grafana](https://grafana.com/)
- [Docker](https://www.docker.com/)
