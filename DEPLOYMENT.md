# AICtrlNet Community Edition - Production Deployment Checklist

A step-by-step checklist for deploying AICtrlNet Community Edition to production.

## Pre-Deployment

### Infrastructure Requirements

- [ ] **PostgreSQL 15+** database provisioned
- [ ] **Redis 7+** instance provisioned
- [ ] **Server/Container** with 2GB+ RAM, 2+ CPU cores
- [ ] **Domain** configured with SSL certificate
- [ ] **Firewall** allows ports 8000 (API), 5432 (Postgres), 6379 (Redis)

### Security Credentials

Generate secure values for:

```bash
# Generate SECRET_KEY (JWT signing)
openssl rand -hex 32

# Generate MFA_ENCRYPTION_KEY (32 characters)
openssl rand -base64 24 | head -c 32

# Generate OAUTH2_ENCRYPTION_KEY (Fernet key)
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# Generate strong database password
openssl rand -base64 24
```

- [ ] `SECRET_KEY` generated and stored securely
- [ ] `MFA_ENCRYPTION_KEY` generated (if using MFA)
- [ ] `OAUTH2_ENCRYPTION_KEY` generated (if using OAuth2)
- [ ] Database password generated
- [ ] Redis password generated (if required)

---

## Deployment Steps

### 1. Environment Configuration

Create production `.env` file:

```bash
# Core
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false

# Security (REQUIRED - use generated values)
SECRET_KEY=<your-generated-secret-key>
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Database (REQUIRED)
DATABASE_URL=postgresql://user:password@db-host:5432/aictrlnet
# Or individual components:
POSTGRES_SERVER=db-host
POSTGRES_USER=aictrlnet
POSTGRES_PASSWORD=<your-db-password>
POSTGRES_DB=aictrlnet
POSTGRES_PORT=5432

# Redis (REQUIRED)
REDIS_URL=redis://:password@redis-host:6379/0
# Or individual components:
REDIS_HOST=redis-host
REDIS_PORT=6379
REDIS_PASSWORD=<your-redis-password>

# CORS (REQUIRED - your domain)
BACKEND_CORS_ORIGINS=https://your-domain.com,https://app.your-domain.com

# Optional: MFA
MFA_ENCRYPTION_KEY=<your-32-char-key>

# Optional: OAuth2
OAUTH2_ENCRYPTION_KEY=<your-fernet-key>
OAUTH2_REDIRECT_URI=https://your-domain.com/auth/callback

# Optional: Stripe
STRIPE_SECRET_KEY=sk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...
FRONTEND_URL=https://your-domain.com
```

- [ ] `.env` file created with all required values
- [ ] No default/dev values remain
- [ ] File permissions restricted (`chmod 600 .env`)

### 2. Database Setup

```bash
# Run database migrations
docker exec <container> alembic upgrade head

# Or if using direct deployment:
alembic upgrade head
```

- [ ] Migrations completed successfully
- [ ] Database tables created
- [ ] Initial admin user created (optional)

### 3. Deploy Application

**Option A: Docker Compose**
```bash
# Pull latest image
docker pull bodaty/aictrlnet-community:latest

# Start with your docker-compose.yml
docker-compose up -d
```

**Option B: Docker Run**
```bash
docker run -d \
  --name aictrlnet \
  -p 8000:8000 \
  --env-file .env \
  bodaty/aictrlnet-community:latest
```

**Option C: Kubernetes**
```bash
kubectl apply -f k8s/deployment.yaml
```

- [ ] Container/pod running
- [ ] No crash loops
- [ ] Logs show successful startup

### 4. Verify Deployment

```bash
# Health check
curl https://your-domain.com/health
# Expected: {"status":"ok","edition":"community","version":"2.0.0"}

# API docs accessible
curl -I https://your-domain.com/api/v1/openapi.json
# Expected: HTTP 200

# Database connectivity
curl https://your-domain.com/api/v1/health
# Expected: HTTP 200 with database status
```

- [ ] Health endpoint returns `ok`
- [ ] API responds to requests
- [ ] Database connection working
- [ ] Redis connection working

---

## Post-Deployment

### Security Hardening

- [ ] SSL/TLS certificate installed and valid
- [ ] HTTP redirects to HTTPS
- [ ] Security headers configured (HSTS, CSP, etc.)
- [ ] Rate limiting enabled
- [ ] Database not exposed to public internet
- [ ] Redis not exposed to public internet
- [ ] Environment variables not logged

### Monitoring Setup

- [ ] Application logs collected
- [ ] Error alerting configured
- [ ] Health check monitoring (uptime)
- [ ] Database monitoring
- [ ] Disk space monitoring

### Backup Configuration

- [ ] Database backup schedule configured
- [ ] Backup retention policy defined
- [ ] Backup restoration tested
- [ ] Redis persistence configured (if needed)

### Documentation

- [ ] Deployment documented for team
- [ ] Runbook created for common operations
- [ ] Incident response plan defined
- [ ] Contact information for on-call

---

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs <container-name>

# Common issues:
# - Missing environment variables
# - Database connection refused
# - Port already in use
```

### Database Connection Failed

```bash
# Test connectivity
docker exec <container> python -c "import psycopg2; psycopg2.connect('$DATABASE_URL')"

# Check:
# - Database host reachable
# - Credentials correct
# - Database exists
# - User has permissions
```

### Redis Connection Failed

```bash
# Test connectivity
docker exec <container> python -c "import redis; r=redis.from_url('$REDIS_URL'); r.ping()"

# Check:
# - Redis host reachable
# - Password correct (if set)
# - Firewall allows connection
```

### Health Check Failing

```bash
# Check what's failing
curl -v http://localhost:8000/health

# Check container health
docker inspect <container> | jq '.[0].State.Health'
```

---

## Quick Reference

| Component | Port | Health Check |
|-----------|------|--------------|
| API | 8000 | `GET /health` |
| PostgreSQL | 5432 | `pg_isready` |
| Redis | 6379 | `redis-cli ping` |

| Environment | Log Level | Debug |
|-------------|-----------|-------|
| development | DEBUG | true |
| staging | INFO | false |
| production | INFO/WARNING | false |

---

## Related Documentation

- [Environment Variables Reference](ENV_VARS.md)
- [Docker Compose Example](docker-compose.example.yml)
- [Security Policy](SECURITY.md)
