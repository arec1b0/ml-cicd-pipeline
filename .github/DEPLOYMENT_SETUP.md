# Deployment Setup Guide

This guide explains how to configure the CI/CD pipeline for secure production deployments.

## GitHub Actions Variables and Secrets

### Required Variables

Configure these in your repository settings under **Settings → Secrets and variables → Actions → Variables**:

| Variable Name | Description | Example Value |
|--------------|-------------|---------------|
| `MODEL_REGISTRY_NAME` | MLflow model registry name | `iris-random-forest` |
| `MODEL_STAGE` | MLflow model stage to deploy | `production` |

### Required Secrets

Configure these in your repository settings under **Settings → Secrets and variables → Actions → Secrets**:

| Secret Name | Description | How to Get |
|------------|-------------|-----------|
| `REGISTRY_HOST` | Container registry hostname | e.g., `gcr.io`, `docker.io` |
| `REGISTRY_USER` | Container registry username | From registry provider |
| `REGISTRY_TOKEN` | Container registry password/token | From registry provider |
| `REGISTRY_REPO` | Container registry repository | e.g., `myorg/ml-model` |
| `KUBE_CONFIG_DATA` | Base64-encoded kubeconfig | `cat ~/.kube/config \| base64` |
| `MLFLOW_TRACKING_URI` | MLflow tracking server URI | e.g., `https://mlflow.example.com` |
| `MLFLOW_EXPERIMENT_NAME` | MLflow experiment name | e.g., `iris-classification` |
| `MLFLOW_MODEL_NAME` | MLflow model name | e.g., `iris-random-forest` |
| `COSIGN_PRIVATE_KEY` | Cosign private key for signing | Generate with `cosign generate-key-pair` |
| `COSIGN_PASSWORD` | Password for Cosign private key | Set during key generation |

## Environment Protection Rules

To prevent accidental production deployments, configure environment protection rules:

### Step 1: Create a Production Environment

1. Go to **Settings → Environments** in your repository
2. Click **New environment**
3. Name it `production`
4. Click **Configure environment**

### Step 2: Configure Protection Rules

Add the following protection rules:

#### Required Reviewers
- **Purpose**: Require manual approval before production deployments
- **Configuration**:
  1. Check **Required reviewers**
  2. Add team members who can approve deployments (minimum 1)
  3. Recommended: Add at least 2 reviewers from different teams

#### Wait Timer
- **Purpose**: Add a delay before deployment starts
- **Configuration**:
  1. Check **Wait timer**
  2. Set delay (recommended: 5-10 minutes for production)
  3. This gives time to cancel accidental deployments

#### Deployment Branches
- **Purpose**: Only allow deployments from specific branches
- **Configuration**:
  1. Under **Deployment branches**, select **Selected branches**
  2. Add branch rules:
     - `main` (for production releases)
     - `release/*` (for release candidates)
  3. Do NOT allow feature branches to deploy to production

### Step 3: Update Workflow to Use Environment

The deployment workflow has been configured to use the `production` environment for the promote job. This means:

1. The canary deployment will proceed automatically after build
2. Manual approval is required before promoting canary to stable
3. Reviewers will be notified when approval is needed
4. The deployment can be reviewed and approved in the Actions tab

### Step 4: Test the Configuration

1. Trigger a deployment workflow
2. Verify that the build and canary steps complete
3. Verify that the promote job waits for approval
4. Approve the deployment as a reviewer
5. Verify that the promotion completes after approval

## Cosign Setup for Container Signing

### Generate Signing Keys

```bash
# Install Cosign
brew install cosign  # macOS
# or
wget "https://github.com/sigstore/cosign/releases/latest/download/cosign-linux-amd64"
chmod +x cosign-linux-amd64
sudo mv cosign-linux-amd64 /usr/local/bin/cosign

# Generate key pair
cosign generate-key-pair

# This creates:
# - cosign.key (private key)
# - cosign.pub (public key)
```

### Store Keys in GitHub Secrets

```bash
# Add private key to GitHub Secrets as COSIGN_PRIVATE_KEY
cat cosign.key  # Copy the entire content

# Add the password you used as COSIGN_PASSWORD
```

### Verify Signed Images

```bash
# Download public key from repository
curl -O https://raw.githubusercontent.com/your-org/your-repo/main/cosign.pub

# Verify an image signature
cosign verify --key cosign.pub your-registry/your-repo:tag

# Expected output:
# Verification for your-registry/your-repo:tag --
# The following checks were performed on each of these signatures:
#   - The cosign claims were validated
#   - The signatures were verified against the specified public key
```

### Store Public Key in Repository

Commit the public key to your repository so it can be used for verification:

```bash
cp cosign.pub .github/cosign.pub
git add .github/cosign.pub
git commit -m "Add Cosign public key for image verification"
```

## Security Scanning Configuration

### Trivy Configuration

Trivy is configured to scan container images for vulnerabilities. Configuration options:

```yaml
# .github/workflows/deploy-canary-and-promote.yml
- name: Run Trivy security scan
  uses: aquasecurity/trivy-action@master
  with:
    image-ref: ${{ secrets.REGISTRY_HOST }}/${{ secrets.REGISTRY_REPO }}:${{ github.sha }}
    format: 'sarif'
    output: 'trivy-results.sarif'
    severity: 'CRITICAL,HIGH'  # Adjust severity levels
    exit-code: '1'  # Fail on vulnerabilities
```

To adjust severity thresholds, modify the `severity` parameter in the workflow.

### pip-audit Configuration

pip-audit scans Python dependencies for known vulnerabilities. To ignore specific vulnerabilities:

```bash
# Create .pip-audit-ignore file
echo "VULN-ID-1234" > .pip-audit-ignore
echo "VULN-ID-5678" >> .pip-audit-ignore

# Update workflow to use ignore file
pip-audit --ignore-vuln VULN-ID-1234 --ignore-vuln VULN-ID-5678
```

## Rollback Workflow

The automated rollback workflow (`.github/workflows/rollback.yml`) allows quick recovery from failed deployments.

### When to Use Rollback

Use the rollback workflow when:
- Production deployment causes issues
- Metrics show degraded performance
- Critical bugs are discovered after deployment

### How to Trigger Rollback

1. Go to **Actions → Rollback to Previous Release**
2. Click **Run workflow**
3. Enter the target tag/SHA to rollback to (or leave empty for previous release)
4. Click **Run workflow**
5. Monitor the rollback process in the Actions tab

### What the Rollback Does

1. Identifies the previous stable deployment
2. Redeploys the previous container image
3. Removes any canary deployments
4. Verifies the rollback with health checks
5. Notifies the team of the rollback

## Monitoring and Alerts

### Deployment Notifications

Configure GitHub Actions notifications:
1. Go to **Settings → Notifications**
2. Enable **Actions** notifications
3. Configure Slack/email integrations if needed

### Metric Thresholds

The canary deployment evaluates model accuracy against a threshold. To adjust:

```yaml
# .github/workflows/deploy-canary-and-promote.yml
threshold=0.70  # Change this value (0.0 to 1.0)
```

Recommended thresholds:
- Development: 0.60
- Staging: 0.70
- Production: 0.80

## Troubleshooting

### Deployment Fails Immediately

**Cause**: Missing secrets or invalid configuration

**Solution**:
1. Verify all required secrets are configured
2. Check secret values are valid (no extra whitespace)
3. Test kubeconfig: `echo "$KUBE_CONFIG_DATA" | base64 -d | kubectl --kubeconfig=/dev/stdin get nodes`

### Approval Not Required

**Cause**: Environment not configured or workflow not using environment

**Solution**:
1. Verify `production` environment exists in Settings → Environments
2. Verify workflow job has `environment: production` configured
3. Check that required reviewers are added to the environment

### Cosign Verification Fails

**Cause**: Image not signed or wrong public key

**Solution**:
1. Verify `COSIGN_PRIVATE_KEY` and `COSIGN_PASSWORD` are set in secrets
2. Check workflow logs for signing step success
3. Ensure public key matches the private key used for signing

### Security Scan Blocks Deployment

**Cause**: Critical/high vulnerabilities found

**Solution**:
1. Review the scan results in the workflow logs
2. Update vulnerable dependencies in `pyproject.toml`
3. Run `poetry update` and test locally
4. If vulnerability is a false positive, adjust Trivy severity settings

## Best Practices

1. **Regular Key Rotation**: Rotate Cosign keys every 6-12 months
2. **Review Approvers**: Ensure multiple people can approve deployments
3. **Monitor Metrics**: Set up alerts for deployment failures and metric degradation
4. **Test Rollback**: Practice rollback procedures regularly
5. **Document Incidents**: Keep a log of deployments, rollbacks, and issues
6. **Least Privilege**: Only give deployment approval rights to necessary team members
7. **Audit Logs**: Regularly review deployment and approval logs

## Additional Resources

- [GitHub Environments Documentation](https://docs.github.com/en/actions/deployment/targeting-different-environments/using-environments-for-deployment)
- [Cosign Documentation](https://docs.sigstore.dev/cosign/overview/)
- [Trivy Documentation](https://aquasecurity.github.io/trivy/)
- [SLSA Framework](https://slsa.dev/)
