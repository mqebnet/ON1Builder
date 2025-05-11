# ON1Builder Post-Deployment Checklist

## Security Best Practices

### Wallet Key Management

- [ ] **Rotate wallet keys monthly**
  ```bash
  ./scripts/rotate_wallet_keys.sh
  ```
  This script will:
  - Back up current wallet keys
  - Generate new wallet keys
  - Update Vault with the new keys
  - Restart services to use the new keys
  - Verify that everything is working correctly

- [ ] **Back up wallet keys securely**
  ```bash
  ./scripts/backup_wallet_keys.sh
  ```
  This script will:
  - Create encrypted backups of wallet keys
  - Store them in a secure location
  - Verify that the backups are valid

### Monitoring and Alerting

- [ ] **Monitor Slack alerts**
  - Set up a dedicated Slack channel for alerts
  - Configure notification preferences to ensure alerts are seen promptly
  - Test alerts regularly to ensure they're working correctly

- [ ] **Monitor email alerts**
  - Configure email filters to ensure alerts don't go to spam
  - Set up email forwarding to ensure alerts are seen by multiple team members
  - Test email alerts regularly

- [ ] **Check Prometheus alarms**
  - Review Prometheus alert rules
  - Test alerts by triggering conditions
  - Ensure alerts are properly routed

- [ ] **Review Grafana dashboards**
  - Check that all metrics are being displayed correctly
  - Set up dashboard alerts for critical metrics
  - Share dashboards with relevant team members

### Regular Maintenance

- [ ] **Check logs regularly**
  - Review application logs for errors or warnings
  - Check system logs for any issues
  - Set up log rotation to prevent disk space issues

- [ ] **Update dependencies**
  - Keep Docker images up to date
  - Update Python dependencies
  - Apply security patches promptly

- [ ] **Perform regular backups**
  - Back up configuration files
  - Back up Vault data
  - Back up Prometheus and Grafana data

### Chain-Specific Checks

- [ ] **Monitor gas prices**
  - Set up alerts for gas price spikes
  - Adjust gas price strategies as needed
  - Consider implementing gas price optimization

- [ ] **Check wallet balances**
  - Ensure wallets have sufficient funds
  - Set up alerts for low balances
  - Monitor transaction costs

- [ ] **Verify chain connections**
  - Check that connections to all chains are stable
  - Monitor for chain-specific issues
  - Have fallback RPC endpoints configured

## Environment-Specific Policies

Review and update the SECURITY.md file with any environment-specific policies:

- [ ] **Access control**
  - Who has access to the production environment
  - How access is granted and revoked
  - Multi-factor authentication requirements

- [ ] **Incident response**
  - Who to contact in case of an incident
  - Steps to take to mitigate damage
  - How to report and document incidents

- [ ] **Compliance requirements**
  - Any regulatory requirements that apply
  - How compliance is monitored and enforced
  - Documentation requirements

## Regular Reviews

- [ ] **Weekly review**
  - Check system health
  - Review alerts from the past week
  - Verify that all cron jobs are running

- [ ] **Monthly review**
  - Perform a security audit
  - Review and rotate credentials
  - Check for any needed updates or improvements

- [ ] **Quarterly review**
  - Comprehensive system review
  - Test disaster recovery procedures
  - Update documentation as needed
