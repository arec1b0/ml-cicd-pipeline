# TLS Configuration Guide

## Overview

This document provides guidance on configuring TLS/HTTPS for the ML CI/CD Pipeline inference service. By default, the application exposes HTTP on port 8000. For production deployments, it is **critical** to enable TLS to protect sensitive data in transit, including:

- Admin API tokens
- Model predictions
- Input features
- Authentication credentials

## Security Risk

**WARNING**: Without TLS/HTTPS, all data is transmitted in cleartext, making it vulnerable to:
- Man-in-the-middle (MITM) attacks
- Credential theft
- Data interception
- API token exposure

## Recommended Approaches

### 1. Reverse Proxy with TLS Termination (Recommended)

The most common and recommended approach is to use a reverse proxy that handles TLS termination in front of the application.

#### Option A: Nginx

```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # TLS Configuration
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    # Strong TLS configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384';
    ssl_prefer_server_ciphers on;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}
```

Deploy with Docker Compose:

```yaml
services:
  nginx:
    image: nginx:latest
    ports:
      - "443:443"
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certs:/etc/nginx/certs:ro
    depends_on:
      - api

  api:
    build:
      context: .
      dockerfile: docker/Dockerfile
    # Remove port exposure - only accessible via nginx
    expose:
      - "8000"
    environment:
      - ADMIN_API_TOKEN=${ADMIN_API_TOKEN:?Required}
      # ... other config
```

#### Option B: Traefik

```yaml
services:
  traefik:
    image: traefik:v2.9
    command:
      - "--providers.docker=true"
      - "--entrypoints.web.address=:80"
      - "--entrypoints.websecure.address=:443"
      - "--certificatesresolvers.myresolver.acme.tlschallenge=true"
      - "--certificatesresolvers.myresolver.acme.email=your-email@example.com"
      - "--certificatesresolvers.myresolver.acme.storage=/letsencrypt/acme.json"
    ports:
      - "443:443"
      - "80:80"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - ./letsencrypt:/letsencrypt

  api:
    build:
      context: .
      dockerfile: docker/Dockerfile
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.api.rule=Host(`your-domain.com`)"
      - "traefik.http.routers.api.entrypoints=websecure"
      - "traefik.http.routers.api.tls.certresolver=myresolver"
      - "traefik.http.services.api.loadbalancer.server.port=8000"
    environment:
      - ADMIN_API_TOKEN=${ADMIN_API_TOKEN:?Required}
```

### 2. Cloud Load Balancer with TLS

When deploying to cloud platforms, use managed load balancers with TLS termination:

#### AWS Application Load Balancer (ALB)

1. Create an ALB with HTTPS listener (port 443)
2. Configure SSL certificate from ACM (AWS Certificate Manager)
3. Target the ECS service or EC2 instances on port 8000
4. Configure security groups to only allow traffic from ALB to application

#### Google Cloud Load Balancer

1. Create a Google-managed SSL certificate or upload your own
2. Configure HTTPS frontend on port 443
3. Backend service points to your GKE service or Compute Engine instances
4. Firewall rules to restrict direct access to port 8000

#### Azure Application Gateway

1. Configure Application Gateway with HTTPS listener
2. Upload SSL certificate or use Azure Key Vault integration
3. Backend pool points to your AKS service or VMs
4. NSG rules to restrict direct access

### 3. Kubernetes Ingress with TLS

For Kubernetes deployments, use Ingress with TLS:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-inference-ingress
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - your-domain.com
    secretName: ml-inference-tls
  rules:
  - host: your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ml-inference-service
            port:
              number: 8000
```

With cert-manager for automatic certificate management:

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.12.0/cert-manager.yaml

# Create ClusterIssuer for Let's Encrypt
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: your-email@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

## Obtaining TLS Certificates

### Let's Encrypt (Free, Automated)

For public-facing services:
- Use [certbot](https://certbot.eff.org/) for standalone servers
- Use cert-manager for Kubernetes
- Certificates auto-renew every 90 days

### Self-Signed Certificates (Development Only)

**Not recommended for production**

```bash
# Generate self-signed certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout key.pem -out cert.pem \
  -subj "/CN=localhost"
```

### Commercial Certificate Authorities

For enterprise deployments:
- DigiCert
- GlobalSign
- Sectigo

Purchase certificates with appropriate validation level (DV, OV, or EV).

## Verification

After configuring TLS, verify the setup:

1. Check certificate validity:
```bash
curl -vI https://your-domain.com/health
```

2. Test TLS configuration:
- Use [SSL Labs](https://www.ssllabs.com/ssltest/) for comprehensive analysis
- Aim for A+ rating

3. Verify HTTP to HTTPS redirect:
```bash
curl -I http://your-domain.com/health
# Should return 301 redirect to https://
```

4. Test API functionality over HTTPS:
```bash
curl -X POST https://your-domain.com/predict \
  -H "Content-Type: application/json" \
  -H "X-Admin-Token: your-secure-token" \
  -d '{"features": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]}'
```

## Additional Security Considerations

1. **Always use TLS 1.2 or higher** - Disable TLS 1.0 and 1.1
2. **Enable HTTP Strict Transport Security (HSTS)** - Force browsers to use HTTPS
3. **Use strong cipher suites** - Prefer ECDHE with AES-GCM
4. **Rotate certificates before expiry** - Set up monitoring and alerts
5. **Protect private keys** - Use restrictive file permissions (600) or key vaults
6. **Consider mutual TLS (mTLS)** - For service-to-service communication

## References

- [Mozilla SSL Configuration Generator](https://ssl-config.mozilla.org/)
- [OWASP TLS Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Transport_Layer_Protection_Cheat_Sheet.html)
- [Let's Encrypt Documentation](https://letsencrypt.org/docs/)
