# AICtrlNet Edition Migration Guide

This guide covers upgrading from Community Edition to Business or Enterprise editions. Business and Enterprise are commercial editions available from Bodaty.

**Contact:** sales@bodaty.com

---

## Table of Contents

1. [Edition Comparison](#edition-comparison)
2. [Migration Prerequisites](#migration-prerequisites)
3. [Community to Business Migration](#community-to-business-migration)
4. [Business to Enterprise Migration](#business-to-enterprise-migration)
5. [Data Migration](#data-migration)
6. [Configuration Changes](#configuration-changes)
7. [Licensing and Billing](#licensing-and-billing)
8. [Rollback Procedures](#rollback-procedures)
9. [Common Issues and Solutions](#common-issues-and-solutions)

---

## Edition Comparison

### Feature Matrix

| Feature Category | Community (MIT) | Business | Enterprise |
|-----------------|:---------------:|:--------:|:----------:|
| **Core Platform** |
| Visual Workflow Editor | Yes | Yes | Yes |
| API Endpoints | Yes | Yes | Yes |
| Basic Task Management | Yes | Yes | Yes |
| Webhook Support | Yes | Yes | Yes |
| **AI Frameworks** |
| LangChain Integration | Yes | Yes | Yes |
| AutoGPT, AutoGen, CrewAI | Yes | Yes | Yes |
| MCP Protocol Support | Yes | Yes | Yes |
| **Human-in-the-Loop** |
| Basic Approvals | Yes | Yes | Yes |
| Multi-step Approval Workflows | - | Yes | Yes |
| Approval Delegation | - | Yes | Yes |
| **AI Governance** |
| Basic Audit Logs | Yes | Yes | Yes |
| ML-Powered Risk Assessment | - | Yes | Yes |
| Policy Templates | - | Yes | Yes |
| Bias Detection | - | Yes | Yes |
| Compliance Dashboards | - | Yes | Yes |
| **Agents** |
| Basic Agents | Yes | Yes | Yes |
| Enhanced Agents (Memory, Learning) | - | Yes | Yes |
| Agent Pods & Swarms | - | Yes | Yes |
| Agent-to-Agent Protocol (A2A) | - | Yes | Yes |
| **Security** |
| API Key Authentication | Yes | Yes | Yes |
| OAuth2 Integration | - | Yes | Yes |
| MFA (Multi-Factor Auth) | - | Yes | Yes |
| SAML/SSO | - | - | Yes |
| **Data & Analytics** |
| Basic Metrics | Yes | Yes | Yes |
| Advanced Analytics | - | Yes | Yes |
| Custom Dashboards | - | Yes | Yes |
| Predictive Analytics | - | - | Yes |
| **Organization** |
| Single Tenant | Yes | Yes | Yes |
| Organizations & Departments | - | Yes | Yes |
| Team Collaboration | - | Yes | Yes |
| Multi-Tenancy | - | - | Yes |
| **Compliance** |
| Basic Logging | Yes | Yes | Yes |
| SOC2 Reports | - | Yes | Yes |
| HIPAA Compliance Tools | - | - | Yes |
| GDPR Compliance Tools | - | - | Yes |
| Geographic Routing | - | - | Yes |
| **Infrastructure** |
| Single Instance | Yes | Yes | Yes |
| Federation | - | - | Yes |
| Cross-Region Deployment | - | - | Yes |
| White-Label Support | - | - | Yes |
| **Support** |
| Community Support | Yes | Yes | Yes |
| Email Support | - | Yes | Yes |
| Priority Support | - | - | Yes |
| Dedicated Account Manager | - | - | Yes |

### Database Tables by Edition

| Edition | Approximate Table Count | Key Tables Added |
|---------|------------------------|------------------|
| Community | ~75 | users, tasks, workflows, webhooks, api_keys, mcp_servers, adapters |
| Business | ~150 | approval_workflows, enhanced_agents, ai_governance, oauth2, organizations, resource_pools |
| Enterprise | ~200 | tenant_groups, federation_nodes, compliance_standards, saml_providers, geographic_regions |

---

## Migration Prerequisites

### Before Starting Any Migration

1. **Backup Your Data**
   ```bash
   # Create a full database backup
   pg_dump -h localhost -U postgres aictrlnet > backup_$(date +%Y%m%d_%H%M%S).sql

   # Backup configuration files
   tar -czf config_backup_$(date +%Y%m%d).tar.gz .env docker-compose.yml
   ```

2. **Document Current State**
   - Record current workflow count
   - Note active integrations
   - List custom configurations
   - Export API key inventory

3. **System Requirements**

   | Edition | Min RAM | Min CPU | Recommended Storage |
   |---------|---------|---------|---------------------|
   | Community | 4 GB | 2 cores | 20 GB |
   | Business | 8 GB | 4 cores | 50 GB |
   | Enterprise | 16 GB | 8 cores | 100 GB+ |

4. **Obtain License**
   - Contact sales@bodaty.com
   - Receive license key and access credentials
   - Download edition-specific container images

5. **Schedule Maintenance Window**
   - Plan for 1-2 hours of downtime
   - Notify affected users
   - Prepare rollback plan

---

## Community to Business Migration

### Step 1: Prepare Environment

```bash
# Stop Community Edition
docker-compose down

# Verify backup exists
ls -la backup_*.sql

# Pull Business Edition images (after receiving access)
docker login registry.bodaty.com
docker pull registry.bodaty.com/aictrlnet-business:latest
docker pull registry.bodaty.com/aictrlnet-ml-service:latest
```

### Step 2: Update Configuration

Create or update your `.env` file:

```bash
# Add Business Edition settings
AICTRLNET_EDITION=business
AICTRLNET_LICENSE_KEY=your-license-key-here

# ML Service Configuration (new in Business)
ML_SERVICE_ENABLED=true
ML_SERVICE_URL=http://ml-service:8003

# OAuth2 Settings (new in Business)
OAUTH2_GOOGLE_CLIENT_ID=your-client-id
OAUTH2_GOOGLE_CLIENT_SECRET=your-client-secret

# Advanced Features
AI_GOVERNANCE_ENABLED=true
ENHANCED_AGENTS_ENABLED=true
```

### Step 3: Update Docker Compose

Replace your `docker-compose.yml` with the Business Edition version:

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: aictrlnet
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

  ml-service:
    image: registry.bodaty.com/aictrlnet-ml-service:latest
    ports:
      - "8003:8003"
    environment:
      - ML_MODEL_PATH=/models
    volumes:
      - ml_models:/models

  business:
    image: registry.bodaty.com/aictrlnet-business:latest
    ports:
      - "8001:8001"
    environment:
      - DATABASE_URL=postgresql://postgres:${DB_PASSWORD}@postgres:5432/aictrlnet
      - REDIS_URL=redis://redis:6379/0
      - ML_SERVICE_URL=http://ml-service:8003
      - AICTRLNET_LICENSE_KEY=${AICTRLNET_LICENSE_KEY}
    depends_on:
      - postgres
      - redis
      - ml-service

  frontend:
    image: registry.bodaty.com/aictrlnet-hitlai:latest
    ports:
      - "3000:80"
    environment:
      - REACT_APP_API_URL=http://localhost:8001
    depends_on:
      - business

volumes:
  postgres_data:
  redis_data:
  ml_models:
```

### Step 4: Run Database Migrations

```bash
# Start database services first
docker-compose up -d postgres redis

# Wait for postgres to be ready
sleep 10

# Run Business Edition migrations
docker-compose run --rm business alembic upgrade head

# Verify migration success
docker-compose run --rm business alembic current
```

### Step 5: Start Services and Verify

```bash
# Start all services
docker-compose up -d

# Check service health
curl http://localhost:8001/health

# Verify ML service
curl http://localhost:8003/health

# Test Business-specific endpoint
curl -H "Authorization: Bearer your-token" \
  http://localhost:8001/api/v1/ai-governance/ml-status
```

### Step 6: Post-Migration Tasks

1. **Configure OAuth2 Providers** (if using SSO)
   - Navigate to Settings > Authentication
   - Add Google, GitHub, or other providers

2. **Set Up Organizations**
   - Create organization structure
   - Assign users to departments

3. **Enable AI Governance**
   - Configure risk assessment thresholds
   - Set up policy templates
   - Enable bias detection

4. **Create Enhanced Agents**
   - Migrate basic agents to enhanced agents
   - Configure memory and learning settings

---

## Business to Enterprise Migration

### Step 1: Prepare Environment

```bash
# Stop Business Edition
docker-compose down

# Backup Business Edition data
pg_dump -h localhost -U postgres aictrlnet > backup_business_$(date +%Y%m%d_%H%M%S).sql

# Pull Enterprise Edition images
docker pull registry.bodaty.com/aictrlnet-enterprise:latest
```

### Step 2: Update Configuration

Add Enterprise-specific settings to `.env`:

```bash
# Update Edition
AICTRLNET_EDITION=enterprise

# Multi-Tenancy Settings
MULTI_TENANT_ENABLED=true
DEFAULT_TENANT_ID=your-tenant-uuid

# SAML/SSO Configuration
SAML_ENABLED=true
SAML_IDP_METADATA_URL=https://your-idp.com/metadata

# Federation Settings
FEDERATION_ENABLED=true
FEDERATION_NODE_ID=primary

# Compliance Settings
COMPLIANCE_MODE=strict
GEOGRAPHIC_ROUTING_ENABLED=true
DATA_RESIDENCY_REGION=us-east-1

# Advanced Security
AUDIT_LOG_RETENTION_DAYS=365
ENCRYPTION_KEY_ROTATION_DAYS=90
```

### Step 3: Update Docker Compose

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: aictrlnet
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  ml-service:
    image: registry.bodaty.com/aictrlnet-ml-service:latest
    ports:
      - "8003:8003"
    deploy:
      resources:
        limits:
          memory: 4G

  enterprise:
    image: registry.bodaty.com/aictrlnet-enterprise:latest
    ports:
      - "8002:8002"
    environment:
      - DATABASE_URL=postgresql://postgres:${DB_PASSWORD}@postgres:5432/aictrlnet
      - REDIS_URL=redis://redis:6379/0
      - ML_SERVICE_URL=http://ml-service:8003
      - AICTRLNET_LICENSE_KEY=${AICTRLNET_LICENSE_KEY}
      - MULTI_TENANT_ENABLED=true
    depends_on:
      - postgres
      - redis
      - ml-service

  frontend:
    image: registry.bodaty.com/aictrlnet-hitlai-enterprise:latest
    ports:
      - "3000:80"
    environment:
      - REACT_APP_API_URL=http://localhost:8002
      - REACT_APP_ENTERPRISE_MODE=true

volumes:
  postgres_data:
  redis_data:
```

### Step 4: Run Database Migrations

```bash
# Start database
docker-compose up -d postgres redis

# Run Enterprise migrations
docker-compose run --rm enterprise alembic upgrade head

# Verify all tables created
docker-compose run --rm enterprise alembic current
```

### Step 5: Configure Multi-Tenancy

```bash
# Create initial tenant
curl -X POST http://localhost:8002/api/v1/tenants \
  -H "Authorization: Bearer admin-token" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Primary Tenant",
    "slug": "primary",
    "settings": {
      "max_users": 100,
      "max_workflows": 1000
    }
  }'

# Migrate existing data to tenant
docker-compose run --rm enterprise python scripts/migrate_to_tenant.py --tenant-id primary
```

### Step 6: Configure SAML SSO

1. Obtain SAML metadata from your Identity Provider
2. Configure in Enterprise admin panel:
   - Navigate to Settings > Security > SAML
   - Upload IdP metadata
   - Configure attribute mappings
   - Test SSO flow

### Step 7: Set Up Federation (Optional)

```bash
# Register federation node
curl -X POST http://localhost:8002/api/v1/federation/nodes \
  -H "Authorization: Bearer admin-token" \
  -H "Content-Type: application/json" \
  -d '{
    "node_id": "primary",
    "region": "us-east-1",
    "endpoint": "https://primary.yourdomain.com"
  }'
```

---

## Data Migration

### Automatic Data Preservation

Both migration paths preserve existing data:

- **Users**: All user accounts retained
- **Workflows**: Workflow definitions and history preserved
- **API Keys**: Existing API keys continue to work
- **Webhooks**: Webhook configurations maintained
- **Audit Logs**: Historical logs preserved

### Data Schema Evolution

Migrations add new columns and tables without modifying existing data:

```
Community Tables → Business Tables Added → Enterprise Tables Added
     (75)              (+75 = 150)             (+50 = 200)
```

### Manual Data Migration Tasks

Some data may need manual attention:

1. **Agent Migration** (Community → Business)
   ```sql
   -- Basic agents can be enhanced with additional capabilities
   -- Run after Business migration
   INSERT INTO enhanced_agents (id, name, type, capabilities, created_at)
   SELECT id, name, 'ai', '{}', created_at
   FROM basic_agents
   WHERE id NOT IN (SELECT id FROM enhanced_agents);
   ```

2. **Organization Assignment** (Business → Enterprise)
   ```sql
   -- Assign existing users to default tenant
   UPDATE users SET tenant_id = 'your-default-tenant-id'
   WHERE tenant_id IS NULL;
   ```

### Data Export/Import

For custom data migration needs:

```bash
# Export specific tables
pg_dump -t workflows -t tasks aictrlnet > workflows_export.sql

# Import to new instance
psql -d aictrlnet < workflows_export.sql
```

---

## Configuration Changes

### Environment Variables by Edition

| Variable | Community | Business | Enterprise |
|----------|-----------|----------|------------|
| `DATABASE_URL` | Required | Required | Required |
| `REDIS_URL` | Required | Required | Required |
| `SECRET_KEY` | Required | Required | Required |
| `AICTRLNET_LICENSE_KEY` | - | Required | Required |
| `ML_SERVICE_URL` | - | Required | Required |
| `OAUTH2_*` | - | Optional | Optional |
| `MULTI_TENANT_ENABLED` | - | - | Required |
| `SAML_*` | - | - | Optional |
| `FEDERATION_*` | - | - | Optional |
| `COMPLIANCE_MODE` | - | - | Optional |

### Port Mappings

| Service | Community | Business | Enterprise |
|---------|-----------|----------|------------|
| API | 8000 | 8001 | 8002 |
| ML Service | - | 8003 | 8003 |
| Frontend | 3000 | 3000 | 3000 |

### API Endpoint Changes

Base URLs change by edition:

- Community: `http://localhost:8000/api/v1/`
- Business: `http://localhost:8001/api/v1/`
- Enterprise: `http://localhost:8002/api/v1/`

Update client applications accordingly.

---

## Licensing and Billing

### License Types

| Edition | License Model | Typical Pricing |
|---------|--------------|-----------------|
| Community | MIT (Free) | Free forever |
| Business | Annual Subscription | Contact sales |
| Enterprise | Annual Subscription | Contact sales |

### Obtaining a License

1. Contact sales@bodaty.com
2. Describe your use case and team size
3. Receive quote and license agreement
4. Complete purchase
5. Receive:
   - License key
   - Container registry credentials
   - Support portal access

### License Activation

```bash
# Set license key in environment
export AICTRLNET_LICENSE_KEY=your-license-key

# Or add to .env file
echo "AICTRLNET_LICENSE_KEY=your-license-key" >> .env

# Verify license on startup
docker-compose logs business | grep -i license
```

### License Validation

The license is validated:
- On service startup
- Every 24 hours during operation
- License includes: edition, seat count, expiration date

### Billing Management

- Invoices sent monthly or annually
- Self-service portal for:
  - Usage reports
  - Seat management
  - Invoice history
  - Payment method updates

---

## Rollback Procedures

### Rollback Community → Business

If you need to revert from Business to Community:

```bash
# Stop Business services
docker-compose down

# Restore Community configuration
cp docker-compose.community.backup.yml docker-compose.yml
cp .env.community.backup .env

# Restore database (if schema issues occur)
psql -d aictrlnet < backup_community_YYYYMMDD.sql

# Start Community Edition
docker-compose up -d
```

**Note:** Business-specific data (enhanced agents, AI governance records, etc.) will remain in the database but be inaccessible from Community Edition.

### Rollback Business → Enterprise

```bash
# Stop Enterprise services
docker-compose down

# Restore Business configuration
cp docker-compose.business.backup.yml docker-compose.yml
cp .env.business.backup .env

# No database restore needed - Enterprise tables are additive

# Start Business Edition
docker-compose up -d
```

### Emergency Rollback Script

Save this as `rollback.sh`:

```bash
#!/bin/bash
set -e

EDITION=$1
BACKUP_DATE=$2

if [ -z "$EDITION" ] || [ -z "$BACKUP_DATE" ]; then
    echo "Usage: ./rollback.sh [community|business] YYYYMMDD"
    exit 1
fi

echo "Rolling back to $EDITION edition from backup $BACKUP_DATE..."

# Stop current services
docker-compose down

# Restore configuration
cp docker-compose.${EDITION}.backup.yml docker-compose.yml
cp .env.${EDITION}.backup .env

# Restore database
psql -d aictrlnet < backup_${EDITION}_${BACKUP_DATE}.sql

# Start services
docker-compose up -d

echo "Rollback complete. Verify services at http://localhost:8000"
```

---

## Common Issues and Solutions

### Issue: License Validation Failed

**Symptoms:**
- Service fails to start
- Log shows "License validation failed"

**Solutions:**
1. Verify license key is correct
2. Check network connectivity to license server
3. Ensure system clock is accurate (NTP)
4. Contact support if persists

```bash
# Check license status
curl http://localhost:8001/api/v1/system/license
```

### Issue: Database Migration Errors

**Symptoms:**
- `alembic upgrade head` fails
- "Relation already exists" errors

**Solutions:**
1. Check current migration state:
   ```bash
   docker-compose run --rm business alembic current
   ```

2. Mark migration as complete if table exists:
   ```bash
   docker-compose run --rm business alembic stamp head
   ```

3. Manual recovery:
   ```sql
   -- Check which migrations have run
   SELECT * FROM alembic_version;
   SELECT * FROM alembic_version_business;
   ```

### Issue: ML Service Connection Failed

**Symptoms:**
- Business Edition cannot reach ML service
- AI governance features unavailable

**Solutions:**
1. Verify ML service is running:
   ```bash
   docker-compose logs ml-service
   curl http://localhost:8003/health
   ```

2. Check network configuration:
   ```bash
   docker network ls
   docker network inspect aictrlnet_default
   ```

3. Verify environment variable:
   ```bash
   docker-compose exec business env | grep ML_SERVICE
   ```

### Issue: OAuth2 Login Fails

**Symptoms:**
- "Invalid redirect URI" errors
- OAuth callback fails

**Solutions:**
1. Verify redirect URI matches OAuth provider configuration
2. Check CORS settings if using different domains
3. Verify client ID and secret are correct

### Issue: Multi-Tenant Data Isolation

**Symptoms:**
- Users see data from other tenants
- Permission errors

**Solutions:**
1. Verify tenant_id is set for all users
2. Check Row-Level Security (RLS) policies
3. Review tenant configuration:
   ```sql
   SELECT * FROM tenants WHERE id = 'your-tenant-id';
   ```

### Issue: SAML SSO Not Working

**Symptoms:**
- SAML redirect fails
- "Invalid assertion" errors

**Solutions:**
1. Verify IdP metadata is current
2. Check clock synchronization (SAML is time-sensitive)
3. Verify attribute mappings match IdP configuration
4. Check SAML debug logs:
   ```bash
   docker-compose logs enterprise | grep -i saml
   ```

### Issue: Performance Degradation After Migration

**Symptoms:**
- Slower API responses
- Database query timeouts

**Solutions:**
1. Run database maintenance:
   ```sql
   VACUUM ANALYZE;
   ```

2. Check for missing indexes:
   ```sql
   SELECT tablename, indexname FROM pg_indexes WHERE schemaname = 'public';
   ```

3. Review resource allocation (RAM, CPU)

4. Enable query logging to identify slow queries

---

## Getting Help

### Documentation Resources

- API Documentation: `/docs` endpoint on each service
- Knowledge Base: https://docs.bodaty.com/aictrlnet
- Release Notes: https://github.com/Bodaty/aictrlnet-community/releases

### Support Channels

| Channel | Community | Business | Enterprise |
|---------|-----------|----------|------------|
| GitHub Issues | Yes | Yes | Yes |
| Community Discord | Yes | Yes | Yes |
| Email Support | - | Yes | Yes |
| Phone Support | - | - | Yes |
| Dedicated Slack | - | - | Yes |

### Contact Information

- **Sales Inquiries:** sales@bodaty.com
- **Technical Support:** support@bodaty.com
- **General Questions:** team@aictrlnet.com

---

## Appendix: Quick Reference

### Migration Checklist

- [ ] Create full database backup
- [ ] Backup configuration files
- [ ] Verify system requirements
- [ ] Obtain license key
- [ ] Schedule maintenance window
- [ ] Notify affected users
- [ ] Stop current services
- [ ] Update configuration
- [ ] Update docker-compose.yml
- [ ] Run database migrations
- [ ] Start new services
- [ ] Verify health endpoints
- [ ] Test critical workflows
- [ ] Configure new features
- [ ] Document any issues
- [ ] Notify users of completion

### Useful Commands

```bash
# Check service health
curl http://localhost:800X/health

# View logs
docker-compose logs -f [service-name]

# Run migrations
docker-compose run --rm [service] alembic upgrade head

# Check migration status
docker-compose run --rm [service] alembic current

# Database console
docker-compose exec postgres psql -U postgres aictrlnet

# Restart specific service
docker-compose restart [service-name]
```
