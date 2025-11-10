# Secret Management Guide

This document provides comprehensive guidance on managing secrets for the ML Model Helm chart. **Never commit secrets to version control.** All secrets must be created externally using one of the methods described below.

## Table of Contents

1. [Overview](#overview)
2. [External Secrets Operator (Recommended)](#external-secrets-operator-recommended)
3. [Sealed Secrets](#sealed-secrets)
4. [Manual Secret Creation](#manual-secret-creation)
5. [Migration Guide](#migration-guide)
6. [Secret Rotation Procedures](#secret-rotation-procedures)
7. [Troubleshooting](#troubleshooting)

## Overview

The ML Model chart requires two secrets:

1. **Admin Token Secret** (`ADMIN_API_TOKEN`)
   - Used for authenticating admin API endpoints (e.g., `/admin/reload`)
   - Configured via `env.adminTokenSecretName` and `env.adminTokenSecretKey`

2. **MLflow Credentials Secret** (`MLFLOW_TRACKING_PASSWORD`)
   - Used for authenticating with MLflow tracking server
   - Configured via `env.mlflow.trackingPasswordSecretName` and `env.mlflow.trackingPasswordSecretKey`

### Security Best Practices

- ✅ Use External Secrets Operator or Sealed Secrets for production
- ✅ Rotate secrets regularly (recommended: every 90 days)
- ✅ Use least-privilege access for secret stores
- ✅ Enable audit logging for secret access
- ❌ Never commit secrets to Git
- ❌ Never use inline secret values in values.yaml
- ❌ Never share secrets via unencrypted channels

## External Secrets Operator (Recommended)

The External Secrets Operator integrates with external secret management systems and automatically syncs secrets to Kubernetes.

### Prerequisites

1. Install External Secrets Operator:
```bash
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets external-secrets/external-secrets -n external-secrets-system --create-namespace
```

2. Create a SecretStore or ClusterSecretStore that connects to your secret backend.

### Supported Backends

- AWS Secrets Manager
- AWS Systems Manager Parameter Store
- HashiCorp Vault
- Azure Key Vault
- Google Secret Manager
- Kubernetes Secrets (for migration)
- And many more...

### Configuration

Enable External Secrets in `values.yaml`:

```yaml
externalSecrets:
  enabled: true
  storeRef: "aws-secrets-manager"  # Name of your SecretStore
  adminToken:
    secretName: "ml-model/admin-token"  # Path/key in external store
    secretKey: "token"
    refreshInterval: "1h"
  mlflowCredentials:
    secretName: "ml-model/mlflow-password"
    secretKey: "password"
    refreshInterval: "1h"

env:
  adminTokenSecretName: ""  # Will be auto-generated if empty
  mlflow:
    trackingPasswordSecretName: ""  # Will be auto-generated if empty
```

### Example: AWS Secrets Manager

1. Create a SecretStore:

```yaml
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: aws-secrets-manager
  namespace: default
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-east-1
      auth:
        jwt:
          serviceAccountRef:
            name: external-secrets-sa
```

2. Create secrets in AWS Secrets Manager:

```bash
# Admin token
aws secretsmanager create-secret \
  --name ml-model/admin-token \
  --secret-string '{"token":"your-secure-admin-token-here"}'

# MLflow password
aws secretsmanager create-secret \
  --name ml-model/mlflow-password \
  --secret-string '{"password":"your-mlflow-password-here"}'
```

3. Deploy the Helm chart with External Secrets enabled.

### Example: HashiCorp Vault

1. Create a SecretStore:

```yaml
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: vault-backend
  namespace: default
spec:
  provider:
    vault:
      server: "https://vault.example.com:8200"
      path: "secret"
      version: "v2"
      auth:
        kubernetes:
          mountPath: "kubernetes"
          role: "external-secrets"
          serviceAccountRef:
            name: external-secrets-sa
```

2. Store secrets in Vault:

```bash
vault kv put secret/ml-model/admin-token token="your-secure-admin-token-here"
vault kv put secret/ml-model/mlflow-password password="your-mlflow-password-here"
```

3. Deploy the Helm chart with External Secrets enabled.

### Example: Azure Key Vault

1. Create a SecretStore:

```yaml
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: azure-keyvault
  namespace: default
spec:
  provider:
    azurekv:
      vaultUrl: "https://your-keyvault.vault.azure.net"
      authType: "ManagedIdentity"
      identityId: "your-managed-identity-id"
```

2. Store secrets in Azure Key Vault:

```bash
az keyvault secret set --vault-name your-keyvault --name ml-model-admin-token --value "your-secure-admin-token-here"
az keyvault secret set --vault-name your-keyvault --name ml-model-mlflow-password --value "your-mlflow-password-here"
```

3. Update values.yaml:

```yaml
externalSecrets:
  enabled: true
  storeRef: "azure-keyvault"
  adminToken:
    secretName: "ml-model-admin-token"
  mlflowCredentials:
    secretName: "ml-model-mlflow-password"
```

## Sealed Secrets

Sealed Secrets allow you to encrypt Kubernetes Secrets so they can be safely stored in Git.

### Prerequisites

1. Install kubeseal CLI:
```bash
# macOS
brew install kubeseal

# Linux
wget https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.24.0/kubeseal-0.24.0-linux-amd64.tar.gz
tar -xzf kubeseal-0.24.0-linux-amd64.tar.gz
sudo install -m 755 kubeseal /usr/local/bin/kubeseal
```

2. Install Sealed Secrets controller:
```bash
kubectl apply -f https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.24.0/controller.yaml
```

### Creating Sealed Secrets

1. Create a regular Kubernetes Secret locally:

```yaml
# admin-token-secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: ml-model-admin-token
  namespace: default
type: Opaque
stringData:
  token: "your-secure-admin-token-here"
```

2. Seal the secret:

```bash
kubeseal -f admin-token-secret.yaml -w sealed-admin-token-secret.yaml
```

3. Commit the sealed secret to Git (it's safe - it's encrypted):

```bash
git add sealed-admin-token-secret.yaml
git commit -m "Add sealed admin token secret"
```

4. Apply the sealed secret:

```bash
kubectl apply -f sealed-admin-token-secret.yaml
```

The Sealed Secrets controller will automatically decrypt and create the regular Secret.

5. Configure values.yaml:

```yaml
env:
  adminTokenSecretName: "ml-model-admin-token"
  adminTokenSecretKey: "token"
  mlflow:
    trackingPasswordSecretName: "ml-model-mlflow-credentials"
    trackingPasswordSecretKey: "password"
```

### Sealed Secret Example

```yaml
apiVersion: bitnami.com/v1alpha1
kind: SealedSecret
metadata:
  name: ml-model-admin-token
  namespace: default
spec:
  encryptedData:
    token: AgBx...encrypted-data...xyz
```

## Manual Secret Creation

For development or testing, you can manually create secrets using kubectl.

### Creating Secrets with kubectl

```bash
# Admin token secret
kubectl create secret generic ml-model-admin-token \
  --from-literal=token="your-secure-admin-token-here" \
  --namespace=default

# MLflow credentials secret
kubectl create secret generic ml-model-mlflow-credentials \
  --from-literal=password="your-mlflow-password-here" \
  --namespace=default
```

### Creating Secrets from Files

```bash
# Create token file
echo -n "your-secure-admin-token-here" > admin-token.txt

# Create secret from file
kubectl create secret generic ml-model-admin-token \
  --from-file=token=admin-token.txt \
  --namespace=default

# Clean up
rm admin-token.txt
```

### Configuration

Update `values.yaml`:

```yaml
env:
  adminTokenSecretName: "ml-model-admin-token"
  adminTokenSecretKey: "token"
  mlflow:
    trackingPasswordSecretName: "ml-model-mlflow-credentials"
    trackingPasswordSecretKey: "password"
```

## Migration Guide

### Migrating from Inline Secrets

If you were previously using inline secret values (`adminTokenSecretValue` or `trackingPasswordSecretValue`), follow these steps:

1. **Choose a secret management method** (External Secrets Operator recommended for production)

2. **Create secrets using your chosen method** (see sections above)

3. **Update values.yaml**:
   - Remove `adminTokenSecretValue` and `trackingPasswordSecretValue`
   - Set `adminTokenSecretName` and `mlflow.trackingPasswordSecretName` to reference your secrets

4. **Verify secrets exist**:
```bash
kubectl get secret ml-model-admin-token
kubectl get secret ml-model-mlflow-credentials
```

5. **Upgrade the Helm release**:
```bash
helm upgrade ml-model ./infra/helm/ml-model-chart \
  -f values.yaml \
  --namespace=default
```

6. **Verify pods can access secrets**:
```bash
kubectl describe pod -l app=ml-model-chart
# Check for Secret volume mounts and environment variables
```

## Secret Rotation Procedures

### Rotating Admin Token

1. **Generate new token**:
```bash
NEW_TOKEN=$(openssl rand -hex 32)
echo "New token: $NEW_TOKEN"
```

2. **Update secret** (method depends on your secret management):

   **External Secrets Operator**: Update secret in external store (AWS, Vault, etc.)
   
   **Sealed Secrets**: 
   ```bash
   # Create new secret
   kubectl create secret generic ml-model-admin-token \
     --from-literal=token="$NEW_TOKEN" \
     --dry-run=client -o yaml | kubeseal -w sealed-admin-token-secret.yaml
   
   # Apply sealed secret
   kubectl apply -f sealed-admin-token-secret.yaml
   ```
   
   **Manual**:
   ```bash
   kubectl create secret generic ml-model-admin-token \
     --from-literal=token="$NEW_TOKEN" \
     --dry-run=client -o yaml | kubectl apply -f -
   ```

3. **Restart pods** to pick up new secret:
```bash
kubectl rollout restart deployment/ml-model-chart-stable
kubectl rollout restart deployment/ml-model-chart-canary
```

4. **Verify new token works**:
```bash
curl -X POST https://your-api/admin/reload \
  -H "X-Admin-Token: $NEW_TOKEN"
```

5. **Update any CI/CD pipelines** or systems that use the admin token

### Rotating MLflow Password

1. **Update password in MLflow server**

2. **Update secret** (same methods as above)

3. **Restart pods**:
```bash
kubectl rollout restart deployment/ml-model-chart-stable
kubectl rollout restart deployment/ml-model-chart-canary
```

4. **Verify MLflow connectivity**:
```bash
kubectl logs -l app=ml-model-chart -c api | grep -i mlflow
```

### Automated Rotation

For External Secrets Operator, secrets are automatically refreshed based on `refreshInterval`. For other methods, consider:

- **CronJob** to rotate secrets periodically
- **GitOps** workflow with automated secret updates
- **External secret rotation** in your secret store (AWS Secrets Manager rotation, etc.)

## Troubleshooting

### Secret Not Found

**Error**: `Secret "ml-model-admin-token" not found`

**Solution**:
1. Verify secret exists: `kubectl get secret ml-model-admin-token`
2. Check secret name matches `adminTokenSecretName` in values.yaml
3. Ensure secret is in the same namespace as the deployment

### Permission Denied

**Error**: `Permission denied` when accessing secrets

**Solution**:
1. Verify ServiceAccount has correct RBAC permissions:
```bash
kubectl describe role ml-model-chart
kubectl describe rolebinding ml-model-chart
```

2. Check ServiceAccount is assigned to pods:
```bash
kubectl describe pod <pod-name> | grep ServiceAccount
```

### External Secrets Not Syncing

**Error**: ExternalSecret status shows `SecretSyncedError`

**Solution**:
1. Check ExternalSecret status:
```bash
kubectl describe externalsecret ml-model-chart-admin-token
```

2. Verify SecretStore configuration:
```bash
kubectl describe secretstore <store-name>
```

3. Check External Secrets Operator logs:
```bash
kubectl logs -n external-secrets-system -l app.kubernetes.io/name=external-secrets
```

### Sealed Secret Not Decrypting

**Error**: SealedSecret shows `UnsealFailed`

**Solution**:
1. Verify Sealed Secrets controller is running:
```bash
kubectl get pods -n kube-system -l app.kubernetes.io/name=sealed-secrets
```

2. Check SealedSecret events:
```bash
kubectl describe sealedsecret <name>
```

3. Ensure you're using the correct sealing key (cluster-specific)

### Pod Cannot Read Secret

**Error**: Pod starts but cannot read secret values

**Solution**:
1. Verify secret key name matches configuration:
```bash
kubectl get secret ml-model-admin-token -o jsonpath='{.data}' | jq
```

2. Check environment variables in pod:
```bash
kubectl exec <pod-name> -- env | grep ADMIN_API_TOKEN
```

3. Verify secretKeyRef in deployment:
```bash
kubectl get deployment ml-model-chart-stable -o yaml | grep -A 5 secretKeyRef
```

## Additional Resources

- [External Secrets Operator Documentation](https://external-secrets.io/)
- [Sealed Secrets Documentation](https://github.com/bitnami-labs/sealed-secrets)
- [Kubernetes Secrets Best Practices](https://kubernetes.io/docs/concepts/configuration/secret/#best-practices)
- [CNCF Security Best Practices](https://github.com/cncf/tag-security/blob/main/supply-chain-security/guides/secrets-management.md)

