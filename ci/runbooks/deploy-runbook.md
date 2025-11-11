# Deploy runbook (canary)

## Overview

This deployment workflow implements a comprehensive CI/CD pipeline with:

- **Security Scanning**: Trivy container scanning for vulnerabilities
- **Artifact Attestation**: Cosign signing and SLSA provenance attestation
- **Canary Deployment**: Low-risk gradual rollout with metrics validation
- **Environment Protection**: Manual approval required for production promotion
- **Automated Rollback**: One-click rollback to previous releases
- **Error Handling**: Comprehensive error detection and automatic cleanup

## Prerequisites

### Secret Management Setup

Before deploying, ensure secrets are properly configured:

1. **Verify required secrets exist**:
   ```bash
   # Admin token secret (required)
   kubectl get secret <admin-token-secret-name> -n <namespace>
   
   # MLflow credentials secret (required if MLflow auth is enabled)
   kubectl get secret <mlflow-credentials-secret-name> -n <namespace>
   ```

2. **Secret management methods**:
   - **External Secrets Operator** (recommended): Secrets are synced automatically from external stores
   - **Sealed Secrets**: Secrets are encrypted and stored in Git
   - **Manual**: Secrets created via kubectl (development only)
   
   See `infra/helm/ml-model-chart/SECRET_MANAGEMENT.md` for detailed setup instructions.

3. **Verify ServiceAccount and RBAC**:
   ```bash
   kubectl get serviceaccount -n <namespace> | grep ml-model-chart
   kubectl get role,rolebinding -n <namespace> | grep ml-model-chart
   ```

## Deployment Steps

1. **Trigger workflow**: Go to Actions -> "Deploy Canary and Promote" -> Run workflow.

2. **Confirm prerequisites**:
   - Registry credentials and kubeconfig secrets exist in GitHub Secrets
   - Required application secrets exist in the target namespace
   - ServiceAccount and RBAC resources are configured

3. **Monitor Actions logs for**:
   - Image build success
   - Security scan (Trivy) passed
   - Image signing (Cosign) completed
   - SLSA provenance attestation generated
   - Canary deployment success
   - Smoke tests pass
   - Metrics evaluation pass
   - Secret validation (if External Secrets Operator is enabled)

4. **Approve production promotion**:
   - After canary validation passes, the workflow will pause and wait for manual approval
   - Go to **Actions** tab and find the running workflow
   - Review the canary validation results
   - Click **Review deployments** and approve the `production` environment
   - The workflow will then promote the canary to stable production

5. **Refresh the running pods with the new model** (skip if `MODEL_AUTO_REFRESH_SECONDS` is non-zero):
   - **Retrieve the admin token**:
     ```bash
     # Get secret name from values.yaml (env.adminTokenSecretName)
     SECRET_NAME="<admin-token-secret-name>"
     SECRET_KEY="token"  # or env.adminTokenSecretKey value
     
     # Retrieve token value
     ADMIN_TOKEN=$(kubectl get secret $SECRET_NAME -n <namespace> \
       -o jsonpath="{.data.$SECRET_KEY}" | base64 -d)
     
     echo "Admin token retrieved (length: ${#ADMIN_TOKEN})"
     ```
   
   - **Call the secured reload endpoint**:
     ```bash
     curl -X POST https://<host>/admin/reload \
       -H "X-Admin-Token: $ADMIN_TOKEN" \
       -v
     ```

6. **Verify container image signature** (optional):
   ```bash
   # Download public key from repository
   curl -O https://raw.githubusercontent.com/your-org/your-repo/main/.github/cosign.pub

   # Verify the signature
   cosign verify --key cosign.pub <registry>/<repo>:<tag>

   # Verify SLSA attestation
   cosign verify-attestation --key cosign.pub --type slsaprovenance <registry>/<repo>:<tag>
   ```

7. **If deployment fails**, workflow leaves stable unchanged and attempts to uninstall canary.

8. **Automated rollback**:
   - Go to **Actions → Rollback to Previous Release**
   - Click **Run workflow**
   - Options:
     - Leave `target_tag` empty to rollback to the previous release (automatic)
     - Or specify a `target_tag` to rollback to a specific version
     - Check `skip_verification` to skip health checks after rollback
   - Click **Run workflow** to execute
   - The rollback workflow will:
     - Determine the target version (if not specified)
     - Verify the target image exists
     - Remove any canary deployments
     - Rollback the stable deployment
     - Run health checks to verify the rollback
     - Generate a summary and upload rollback information

9. **Manual rollback** (if automated workflow is unavailable):
   ```bash
   # List deployment history
   kubectl --kubeconfig <kubeconfig> rollout history deployment/<release> -n <namespace>

   # Rollback to previous revision
   kubectl --kubeconfig <kubeconfig> rollout undo deployment/<release> -n <namespace>

   # Or rollback to specific revision
   kubectl --kubeconfig <kubeconfig> rollout undo deployment/<release> --to-revision=<revision> -n <namespace>

   # Remove canary if exists
   kubectl --kubeconfig <kubeconfig> delete deploy <release>-canary -n <namespace>
   ```

## Secret Rotation Procedures

### Rotating Admin Token

1. **Generate new secure token**:
   ```bash
   NEW_TOKEN=$(openssl rand -hex 32)
   echo "New token generated (length: ${#NEW_TOKEN})"
   ```

2. **Update secret** (method depends on secret management):
   
   **External Secrets Operator**: Update secret in external store (AWS Secrets Manager, Vault, etc.)
   
   **Sealed Secrets**: 
   ```bash
   kubectl create secret generic <admin-token-secret-name> \
     --from-literal=token="$NEW_TOKEN" \
     --dry-run=client -o yaml | kubeseal -w sealed-admin-token-secret.yaml
   kubectl apply -f sealed-admin-token-secret.yaml -n <namespace>
   ```
   
   **Manual**:
   ```bash
   kubectl create secret generic <admin-token-secret-name> \
     --from-literal=token="$NEW_TOKEN" \
     --dry-run=client -o yaml | kubectl apply -f - -n <namespace>
   ```

3. **Restart pods** to pick up new secret:
   ```bash
   kubectl rollout restart deployment/<release>-stable -n <namespace>
   kubectl rollout restart deployment/<release>-canary -n <namespace>
   
   # Wait for rollout to complete
   kubectl rollout status deployment/<release>-stable -n <namespace>
   ```

4. **Verify new token works**:
   ```bash
   curl -X POST https://<host>/admin/reload \
     -H "X-Admin-Token: $NEW_TOKEN"
   ```

5. **Update CI/CD pipelines** or systems that use the admin token

### Rotating MLflow Password

1. **Update password in MLflow server**

2. **Update secret** (same methods as admin token above)

3. **Restart pods**:
   ```bash
   kubectl rollout restart deployment/<release>-stable -n <namespace>
   kubectl rollout restart deployment/<release>-canary -n <namespace>
   ```

4. **Verify MLflow connectivity**:
   ```bash
   kubectl logs -l app=ml-model-chart -c api -n <namespace> | grep -i mlflow
   ```

## Troubleshooting

### Secret Not Found

**Symptoms**: Pods fail to start with "Secret not found" errors

**Diagnosis**:
```bash
# Check if secret exists
kubectl get secret <secret-name> -n <namespace>

# Check secret name in values.yaml matches
grep -A 2 adminTokenSecretName values.yaml
```

**Resolution**:
1. Verify secret name in values.yaml matches actual secret name
2. Ensure secret exists in the same namespace as deployment
3. For External Secrets Operator, check ExternalSecret status:
   ```bash
   kubectl describe externalsecret <secret-name> -n <namespace>
   ```

### Permission Denied

**Symptoms**: Pods start but cannot read secrets

**Diagnosis**:
```bash
# Check ServiceAccount assignment
kubectl describe pod <pod-name> -n <namespace> | grep ServiceAccount

# Check RBAC permissions
kubectl describe role <role-name> -n <namespace>
kubectl describe rolebinding <rolebinding-name> -n <namespace>
```

**Resolution**:
1. Verify ServiceAccount is created and assigned to pods
2. Ensure Role has `get` and `list` permissions for secrets
3. Check RoleBinding links Role to ServiceAccount

### External Secrets Not Syncing

**Symptoms**: ExternalSecret shows `SecretSyncedError` status

**Diagnosis**:
```bash
# Check ExternalSecret status
kubectl describe externalsecret <secret-name> -n <namespace>

# Check External Secrets Operator logs
kubectl logs -n external-secrets-system \
  -l app.kubernetes.io/name=external-secrets \
  --tail=100
```

**Resolution**:
1. Verify SecretStore configuration is correct
2. Check secret exists in external store with correct path/key
3. Verify authentication credentials are valid
4. Check External Secrets Operator controller logs for errors

### Pod Cannot Access Secret

**Symptoms**: Pod starts but environment variable is empty or missing

**Diagnosis**:
```bash
# Check environment variables in pod
kubectl exec <pod-name> -n <namespace> -- env | grep ADMIN_API_TOKEN

# Check secret key exists
kubectl get secret <secret-name> -n <namespace> -o jsonpath='{.data}' | jq
```

**Resolution**:
1. Verify secret key name matches `adminTokenSecretKey` in values.yaml
2. Check secret contains the expected key
3. Ensure secret is base64 encoded correctly
4. Restart pod after secret updates

### Additional Resources

- See `infra/helm/ml-model-chart/SECRET_MANAGEMENT.md` for comprehensive secret management guide
- [External Secrets Operator Documentation](https://external-secrets.io/)
- [Kubernetes Secrets Best Practices](https://kubernetes.io/docs/concepts/configuration/secret/#best-practices)
