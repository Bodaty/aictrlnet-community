# AICtrlNet Community Edition - Environment Variables

Complete reference for all configurable environment variables.

## Quick Start

Create a `.env` file in your project root:

```bash
# Minimum required for production
DATABASE_URL=postgresql://user:password@localhost:5432/aictrlnet
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=your-secure-random-string-at-least-32-characters
ENVIRONMENT=production
```

## Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | `development` | Environment mode: `development`, `staging`, `production` |
| `DEBUG` | `false` | Enable debug mode (never in production) |
| `LOG_LEVEL` | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `AICTRLNET_EDITION` | `community` | Edition: `community`, `business`, `enterprise` |

## Database (PostgreSQL)

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | - | Full connection string (overrides individual settings) |
| `POSTGRES_SERVER` | `localhost` | Database hostname |
| `POSTGRES_PORT` | `5432` | Database port |
| `POSTGRES_USER` | `aictrlnet` | Database username |
| `POSTGRES_PASSWORD` | `local_dev_password` | Database password |
| `POSTGRES_DB` | `aictrlnet_community` | Database name |
| `SQLALCHEMY_DATABASE_URI` | - | Alternative URL (for Cloud SQL Unix sockets) |

**Example:**
```bash
# Option 1: Full URL
DATABASE_URL=postgresql://postgres:mypassword@db.example.com:5432/aictrlnet

# Option 2: Individual components
POSTGRES_SERVER=db.example.com
POSTGRES_USER=postgres
POSTGRES_PASSWORD=mypassword
POSTGRES_DB=aictrlnet
```

## Cache (Redis)

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | - | Full connection string (overrides individual settings) |
| `REDIS_HOST` | `localhost` | Redis hostname |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_PASSWORD` | - | Redis password (optional) |
| `REDIS_DB` | `0` | Redis database number |
| `CACHE_TTL` | `300` | Default cache TTL in seconds |

**Example:**
```bash
# Option 1: Full URL
REDIS_URL=redis://:password@redis.example.com:6379/0

# Option 2: Individual components
REDIS_HOST=redis.example.com
REDIS_PASSWORD=mypassword
REDIS_DB=0
```

## Security

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | dev key | JWT signing key (**must change in production**) |
| `ALGORITHM` | `HS256` | JWT algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | Token expiration time |
| `MFA_ENCRYPTION_KEY` | dev key | MFA secret encryption (32 chars) |
| `OAUTH2_ENCRYPTION_KEY` | dev key | OAuth2 secret encryption (Fernet key) |
| `OAUTH2_REDIRECT_URI` | `http://localhost:3000/auth/callback` | OAuth2 callback URL |

**Generate secure keys:**
```bash
# SECRET_KEY (32+ random characters)
openssl rand -hex 32

# MFA_ENCRYPTION_KEY (exactly 32 characters)
openssl rand -base64 24 | head -c 32

# OAUTH2_ENCRYPTION_KEY (Fernet key)
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

## CORS

| Variable | Default | Description |
|----------|---------|-------------|
| `BACKEND_CORS_ORIGINS` | localhost URLs | Comma-separated list of allowed origins |

**Example:**
```bash
BACKEND_CORS_ORIGINS=https://app.example.com,https://admin.example.com
```

## AI/ML Services

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `DEFAULT_LLM_MODEL` | `llama3.1:8b-instruct-q4_K_M` | Default LLM model ID |
| `LLM_SERVICE_URL` | - | External LLM service URL (optional) |
| `ML_SERVICE_URL` | - | ML microservice URL (for AI governance) |
| `OPENAI_API_KEY` | - | OpenAI API key (if using OpenAI) |

## Stripe (Payments)

| Variable | Default | Description |
|----------|---------|-------------|
| `STRIPE_SECRET_KEY` | test key | Stripe secret API key (or restricted key) |
| `STRIPE_WEBHOOK_SECRET` | test key | Stripe webhook signing secret |
| `STRIPE_PRICE_BUSINESS_STARTER` | - | Stripe Price ID for Business Starter plan |
| `STRIPE_PRICE_BUSINESS_GROWTH` | - | Stripe Price ID for Business Growth plan |
| `STRIPE_PRICE_BUSINESS_SCALE` | - | Stripe Price ID for Business Scale plan |
| `STRIPE_PRICE_ENTERPRISE` | - | Stripe Price ID for Enterprise plan |
| `FRONTEND_URL` | `http://localhost:3000` | Frontend URL for Stripe redirects |

**Setup Steps:**
1. Create products and prices in [Stripe Dashboard](https://dashboard.stripe.com/products)
2. Copy each Price ID (starts with `price_`)
3. Add webhook endpoint pointing to `/api/v1/billing/webhook`
4. Enable these webhook events: `checkout.session.completed`, `customer.subscription.created`, `customer.subscription.updated`, `customer.subscription.deleted`, `invoice.paid`, `invoice.payment_failed`

**Example:**
```bash
STRIPE_SECRET_KEY=rk_live_your_restricted_key
STRIPE_WEBHOOK_SECRET=whsec_your_webhook_secret
STRIPE_PRICE_BUSINESS_STARTER=price_1ABC123
STRIPE_PRICE_BUSINESS_GROWTH=price_1DEF456
STRIPE_PRICE_BUSINESS_SCALE=price_1GHI789
STRIPE_PRICE_ENTERPRISE=price_1JKL012
```

## Credentials Backend

| Variable | Default | Description |
|----------|---------|-------------|
| `CREDENTIAL_BACKEND` | `environment` | Backend: `environment`, `file`, `database`, `vault`. `environment` holds credentials as plaintext env vars and is refused under `AICTRLNET_PHI_MODE` |
| `CREDENTIAL_ENCRYPTION_KEY` | derived from `SECRET_KEY` | Fernet key for `CredentialService`'s file backend |
| `PLATFORM_CREDENTIAL_KEY` | derived from `SECRET_KEY` | Fernet key for `PlatformCredentialService`'s file and database backends |
| `CREDENTIAL_FILE_PATH` | `/app/data/credentials.json` | Path for file-based storage |
| `VAULT_URL` | - | HashiCorp Vault URL |
| `VAULT_TOKEN` | - | Vault authentication token |

**On the derived defaults.** When a credential key is unset, it is derived deterministically
from `SECRET_KEY` rather than randomly generated. Both backends previously called
`Fernet.generate_key()` at process start, so the key died with the process while the
ciphertext lived on in the database — stored credentials were unreadable after any restart,
and the platform decrypt path returned an empty dict rather than raising, so they came back
blank instead of erroring. No compose file or deploy script sets either variable, so this
affected every deployment.

Deriving keeps the key stable for the life of the deployment and distinct between
deployments, with no new provisioning. **Rotating `SECRET_KEY` changes the derived key and
makes existing credentials unreadable**, so any deployment holding credentials it cannot
afford to re-enter should set the explicit variable. `AICTRLNET_PHI_MODE` requires it.

## Performance

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_CONNECTIONS_COUNT` | `10` | Max database connections |
| `MIN_CONNECTIONS_COUNT` | `10` | Min database connections |
| `DATA_PATH` | `/tmp/aictrlnet` | Path for data, staged documents, state snapshots and the RBAC/security/validation JSON files. The self-hosted stack overrides this to the mounted data volume (`/data`) and mounts `DATA_VOLUME` there, so setting one without the other writes into the container's own layer |
| `STAGED_FILES_DIR` | `/tmp/aictrlnet/staged_files` | Where uploaded and generated documents are staged |

## PHI / HIPAA deployments

| Variable | Default | Description |
|----------|---------|-------------|
| `AICTRLNET_PHI_MODE` | `false` | Opt-in for deployments handling protected health information |

When `AICTRLNET_PHI_MODE` is set, the application **refuses to start** unless every one of
the following holds. The check runs in each edition's startup lifespan, and the error names
every setting that is wrong along with what it resolved to.

- `DATA_PATH` is an absolute path that does not resolve under `/tmp`
- `STAGED_FILES_DIR` is an absolute path that does not resolve under `/tmp`, and resolves
  under `DATA_PATH`
- `CREDENTIAL_BACKEND` is not `environment`
- the selected backend can actually encrypt: `file` and `database` require both
  `CREDENTIAL_ENCRYPTION_KEY` and `PLATFORM_CREDENTIAL_KEY`; `vault` requires
  `VAULT_URL` and `VAULT_TOKEN`
- `ALLOW_DEV_TOKENS` is false
- `ENVIRONMENT` names a real deployment (`production`, `prod`, `staging`, `stage`)

The key-material check exists because both credential services generate a throwaway Fernet
key at process start when their key variable is unset — no compose file sets either one — so
credentials would be encrypted with a key that dies with the process and become unreadable
after a restart. Outside PHI mode this behaviour is unchanged.

Containment is decided on resolved real paths, so symlinks and `..` segments cannot smuggle
a path into `/tmp`. The check validates path *shape* only — it does not verify the directory
exists or that the volume is genuinely encrypted, which is a provisioning control.

Leaving `AICTRLNET_PHI_MODE` unset changes nothing about how a deployment behaves.

## Cloud Platform Detection

These are automatically detected but can be overridden:

| Variable | Platform |
|----------|----------|
| `AWS_EXECUTION_ENV`, `AWS_LAMBDA_FUNCTION_NAME` | AWS Lambda |
| `AWS_REGION`, `AWS_DEFAULT_REGION` | AWS |
| `K_SERVICE`, `K_REVISION` | Google Cloud Run |
| `GCP_PROJECT`, `GOOGLE_CLOUD_PROJECT` | Google Cloud |
| `WEBSITE_INSTANCE_ID`, `WEBSITE_SITE_NAME` | Azure App Service |
| `FLY_APP_NAME`, `FLY_REGION` | Fly.io |
| `HEROKU_APP_NAME` | Heroku |
| `DO_APP_ID`, `DO_APP_NAME` | DigitalOcean App Platform |

## Production Checklist

Before deploying to production, ensure you have:

- [ ] Set `ENVIRONMENT=production`
- [ ] Generated a secure `SECRET_KEY`
- [ ] Configured `DATABASE_URL` with production credentials
- [ ] Configured `REDIS_URL` with production credentials
- [ ] Set `BACKEND_CORS_ORIGINS` to your domain(s)
- [ ] Configured `MFA_ENCRYPTION_KEY` if using MFA
- [ ] Configured `OAUTH2_ENCRYPTION_KEY` if using OAuth2
- [ ] Set up `STRIPE_SECRET_KEY` and `STRIPE_WEBHOOK_SECRET` if using payments
- [ ] Configure `STRIPE_PRICE_*` variables with your Stripe Price IDs

## Example .env Files

### Development
```bash
ENVIRONMENT=development
LOG_LEVEL=DEBUG
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/aictrlnet
REDIS_URL=redis://localhost:6379/0
```

### Production
```bash
ENVIRONMENT=production
LOG_LEVEL=INFO
SECRET_KEY=<generate-with-openssl-rand-hex-32>
DATABASE_URL=postgresql://user:secure-password@db.example.com:5432/aictrlnet
REDIS_URL=redis://:redis-password@cache.example.com:6379/0
BACKEND_CORS_ORIGINS=https://app.yourdomain.com
MFA_ENCRYPTION_KEY=<32-character-key>
OAUTH2_ENCRYPTION_KEY=<fernet-key>
```

### Docker Compose
See `docker-compose.example.yml` for a complete example with all services configured.
