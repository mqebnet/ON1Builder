# ON1Builder Deployment Guide

This document provides detailed instructions for deploying ON1Builder to production environments.

## Prerequisites

Before deploying ON1Builder, ensure you have the following:

- Docker and Docker Compose installed
- Access to a Docker registry (Docker Hub or private registry)
- HashiCorp Vault instance or access to deploy one
- Ethereum wallet with sufficient funds for mainnet transactions
- Infura or Alchemy API key for Ethereum RPC access
- SMTP server for email alerts
- Slack webhook URL for Slack alerts
- Domain name (optional, but recommended for production)
- SSL certificate (optional, but recommended for production)

## Step-by-Step Deployment

### 1. Install Prerequisites (macOS)

```bash
brew install --cask docker
brew install docker-compose
brew tap hashicorp/tap
brew install hashicorp/tap/vault
```

For Linux (Ubuntu/Debian):

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose
curl -fsSL https://apt.releases.hashicorp.com/gpg | sudo apt-key add -
sudo apt-add-repository "deb [arch=amd64] https://apt.releases.hashicorp.com $(lsb_release -cs) main"
sudo apt-get update
sudo apt-get install -y vault
```

### 2. Start Vault in Dev Mode

```bash
vault server -dev -dev-root-token-id="root-token" &
export VAULT_ADDR="http://127.0.0.1:8200"
export VAULT_TOKEN="root-token"
```

This will start Vault in development mode with a known root token. In a real production environment, you should use a proper Vault deployment with high availability and proper security measures.

### 3. Initialize Vault and Store Secrets

```bash
cd /path/to/ON1Builder
chmod +x scripts/vault_init.sh
./scripts/vault_init.sh
```

This script will:
- Check if Vault is running
- Initialize Vault if not already initialized
- Unseal Vault if sealed
- Enable the KV-v2 secrets engine at 'secret/'
- Write your secrets to Vault:
  - WALLET_KEY (your Ethereum private key)
  - SMTP_PASSWORD (for email alerts)
  - SLACK_WEBHOOK_URL (for Slack alerts)
  - ALERT_THRESHOLD_ETH (profit threshold for alerts)

Verify that your secrets were stored correctly:

```bash
vault kv get secret/on1builder
```

### 4. Export Required Environment Variables

```bash
export VAULT_ADDR="http://127.0.0.1:8200"
export VAULT_TOKEN="root-token"  # or the token from vault_init.sh
export HTTP_ENDPOINT="https://mainnet.infura.io/v3/your-infura-key"
export WALLET_KEY="your-ethereum-private-key"
export SMTP_PASSWORD="your-smtp-password"
export SLACK_WEBHOOK_URL="your-slack-webhook-url"
```

These environment variables are required for the deployment script to work. The VAULT_ADDR and VAULT_TOKEN are used to connect to Vault. The HTTP_ENDPOINT is used to connect to the Ethereum mainnet.

### 5. Build and Push Docker Image

```bash
chmod +x scripts/build_and_push.sh
./scripts/build_and_push.sh
```

This script will:
- Build the Docker image using Dockerfile.prod
- Tag the image as on1builder:prod
- Push the image to your Docker registry

### 6. Deploy with Docker Compose

#### Single-Chain Deployment

```bash
chmod +x deploy_prod.sh
./deploy_prod.sh
```

This script will:
- Validate all required environment variables
- Login to and unseal Vault
- Write secrets to Vault
- Build and push the Docker image
- Start the application with docker-compose -f docker-compose.prod.yml up -d
- Wait for the application to start
- Check the health endpoint
- Check the metrics endpoint
- Set up cron jobs
- Import the Grafana dashboard
- Trigger a test alert

During the deployment, you will be prompted to confirm receipt of a test alert in your Slack channel.

#### Multi-Chain Deployment

For deploying with multi-chain support:

```bash
# Set the CHAINS environment variable
export CHAINS="1,137"  # Ethereum Mainnet and Polygon Mainnet

# Deploy with multi-chain support
chmod +x deploy_prod.sh
./deploy_prod.sh --multi-chain
```

Alternatively, you can use the multi-chain Docker Compose file directly:

```bash
# Set required environment variables
export VAULT_ADDR="http://127.0.0.1:8200"
export VAULT_TOKEN="root-token"
export CHAINS="1,137"
export GO_LIVE=true

# Deploy with multi-chain Docker Compose file
docker-compose -f docker-compose.multi-chain.yml up -d
```

The multi-chain deployment will:
- Initialize Vault with secrets for each chain
- Start the application with multi-chain support
- Monitor all configured chains simultaneously
- Provide metrics for each chain separately

### 7. Verify Deployment

After the deployment is complete, verify that everything is working correctly:

```bash
# Health check
curl -f http://localhost:5001/healthz

# Metrics check
curl http://localhost:5001/metrics | grep expected_profit_eth

# Grafana dashboard
open http://localhost:3000
```

## Environment Variables

The following environment variables are required for production deployment:

### Required Environment Variables

- `VAULT_ADDR`: URL of the HashiCorp Vault instance
- `VAULT_TOKEN`: Token for authenticating with Vault
- `WALLET_KEY`: Ethereum wallet private key
- `SMTP_PASSWORD`: Password for the SMTP server
- `SLACK_WEBHOOK_URL`: Webhook URL for Slack notifications
- `HTTP_ENDPOINT`: Ethereum RPC endpoint URL

### Optional Environment Variables

- `WEBSOCKET_ENDPOINT`: Ethereum WebSocket endpoint URL
- `DRY_RUN`: Set to "false" for production
- `GO_LIVE`: Set to "true" for production
- `VAULT_PATH`: Path in Vault where secrets are stored (default: "secret/on1builder")
- `MAX_TRANSACTION_VALUE`: Maximum transaction value in ETH (default: 1.0)
- `MAX_DAILY_TRANSACTIONS`: Maximum number of transactions per day (default: 10)
- `MIN_PROFIT_POTENTIAL`: Minimum profit potential in ETH (default: 0.01)
- `ALERT_THRESHOLD_ETH`: Threshold for profit alerts in ETH (default: 0.01)
- `SMTP_SERVER`: SMTP server address (default: smtp.gmail.com)
- `SMTP_PORT`: SMTP server port (default: 587)
- `SMTP_USERNAME`: Username for the SMTP server
- `ALERT_FROM_EMAIL`: Email address to send alerts from
- `ALERT_TO_EMAIL`: Email address to send alerts to
- `GRAFANA_ADMIN_USER`: Grafana admin username (default: admin)
- `GRAFANA_ADMIN_PASSWORD`: Grafana admin password (default: admin)

## Maintenance Tasks

### Updating the Application

To update the application:

1. Pull the latest changes:
   ```bash
   git pull
   ```

2. Run the deployment script:
   ```bash
   ./deploy_prod.sh
   ```

### Backing Up Vault

Regularly back up Vault data:

```bash
docker-compose -f docker-compose.prod.yml exec vault vault operator raft snapshot save /vault/data/backup.snap
docker cp on1builder_vault_1:/vault/data/backup.snap ./backup.snap
```

### Rotating Secrets

Regularly rotate secrets:

1. Generate new secrets (wallet key, API keys, etc.)
2. Update the secrets in Vault:
   ```bash
   curl -X POST -H "X-Vault-Token: $VAULT_TOKEN" -d '{"data":{"WALLET_KEY":"new-key"}}' $VAULT_ADDR/v1/$VAULT_PATH
   ```

3. Restart the application:
   ```bash
   docker-compose -f docker-compose.prod.yml restart app
   ```

## Troubleshooting

### Container Fails to Start

Check the container logs:

```bash
docker-compose -f docker-compose.prod.yml logs app
```

### Health Check Fails

Check the health check endpoint:

```bash
curl http://localhost:5001/healthz
```

Common issues:
- Vault is not accessible
- Ethereum RPC is not accessible
- GO_LIVE is not set to true

### Transactions Not Being Executed

Check the application logs:

```bash
docker-compose -f docker-compose.prod.yml logs app
```

Common issues:
- Insufficient funds in wallet
- Gas price too high
- No profitable opportunities found

### Alerts Not Being Sent

Check the alert configuration:

```bash
docker-compose -f docker-compose.prod.yml exec app python -c "from alerts import AlertManager; print(AlertManager().send_alert('Test', 'INFO'))"
```

Common issues:
- SMTP server not accessible
- Slack webhook URL invalid
- Alert threshold too high

## Security Considerations

See [SECURITY.md](SECURITY.md) for a comprehensive security checklist and best practices.
