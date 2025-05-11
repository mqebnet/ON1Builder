# ON1Builder

ON1Builder is a fully-automated, production-grade, multi-chain trading bot running on Ethereum Mainnet and Polygon Mainnet.

## Features

- Multi-chain support for Ethereum Mainnet and Polygon Mainnet
- Secure secret management with HashiCorp Vault
- Comprehensive monitoring with Prometheus and Grafana
- Alerting via Slack and email
- Automated maintenance and security tasks
- Production-grade deployment with Docker and systemd

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ON1Builder.git
   cd ON1Builder
   ```

2. Run the secure deployment script:
   ```bash
   ./secure_deploy_multi_chain.sh
   ```

3. Follow the prompts to enter sensitive information.

4. Verify the deployment:
   ```bash
   ./scripts/verify_live.sh
   ```

## Documentation

- [Deployment Guide](DEPLOYMENT.md)
- [Security Guidelines](SECURITY.md)
- [Post-Deployment Checklist](post_deployment_checklist.md)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
