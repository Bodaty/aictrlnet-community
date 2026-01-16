# Platform Integration Credential Backend Configuration

This document describes how to configure different credential storage backends for AICtrlNet Platform Integration.

## Overview

AICtrlNet supports multiple credential storage backends to accommodate different deployment scenarios:

1. **Environment Variables** - Simple, read-only storage for self-hosted deployments
2. **File-based** - Encrypted local file storage for development/small deployments
3. **Database** - Encrypted database storage for cloud deployments (default)
4. **HashiCorp Vault** - Enterprise-grade secret management for production deployments

## Configuration

Set the `CREDENTIAL_BACKEND` environment variable to choose your backend:

```bash
# Options: environment, file, database, vault
export CREDENTIAL_BACKEND=database
```

## Backend Details

### 1. Environment Variables Backend

**Use Case**: Self-hosted deployments where credentials are managed via environment variables.

**Configuration**:
```bash
export CREDENTIAL_BACKEND=environment

# Platform credentials format: PLATFORM_CRED_<KEY>
export PLATFORM_CRED_N8N_API='{"api_key":"your-key","base_url":"https://n8n.example.com"}'
export PLATFORM_CRED_ZAPIER_OAUTH='{"client_id":"id","client_secret":"secret","access_token":"token"}'
```

**Features**:
- Read-only (cannot create/update/delete at runtime)
- Simple and secure for static configurations
- No encryption needed (relies on OS/container security)

**Limitations**:
- Cannot dynamically add credentials via UI
- Requires restart to update credentials
- Not suitable for multi-tenant deployments

### 2. File-based Backend

**Use Case**: Development environments or small self-hosted deployments.

**Configuration**:
```bash
export CREDENTIAL_BACKEND=file
export CREDENTIAL_FILE_PATH=/app/data/credentials.json  # Optional, this is default
export CREDENTIAL_ENCRYPTION_KEY=your-base64-encryption-key  # Optional, auto-generated if not set
```

**Features**:
- Encrypted JSON file storage
- Supports full CRUD operations
- Automatic encryption with Fernet
- Good for single-instance deployments

**Security**:
- File is encrypted at rest
- Encryption key should be stored securely
- File permissions should be restricted (600)

### 3. Database Backend (Default)

**Use Case**: Cloud deployments, multi-user environments.

**Configuration**:
```bash
export CREDENTIAL_BACKEND=database
export PLATFORM_CREDENTIAL_KEY=your-base64-encryption-key  # Required for production
```

**Features**:
- Encrypted storage in PostgreSQL
- Full CRUD operations
- Multi-tenant support
- Audit trail (created_at, updated_at, last_used_at)
- Soft delete capability

**Security**:
- Credentials encrypted before storage
- Per-user isolation
- Database-level access controls

### 4. HashiCorp Vault Backend

**Use Case**: Enterprise deployments requiring advanced secret management.

**Configuration**:
```bash
export CREDENTIAL_BACKEND=vault
export VAULT_URL=https://vault.example.com:8200
export VAULT_TOKEN=your-vault-token
export VAULT_NAMESPACE=aictrlnet  # Optional, for Vault Enterprise

# Optional customization
export VAULT_MOUNT_POINT=secret  # Default: secret
export VAULT_PATH_PREFIX=aictrlnet/platforms  # Default: aictrlnet/platforms
```

**Features**:
- Enterprise-grade secret management
- Automatic encryption at rest
- Secret versioning and rotation
- Fine-grained access control
- Comprehensive audit logging
- Dynamic secret generation (future)
- High availability support

**Security**:
- Industry-standard secret management
- Supports various auth methods (token, AppRole, Kubernetes, etc.)
- Secret lease management
- Break-glass procedures

**Vault Setup**:
```bash
# Enable KV v2 secrets engine (done automatically by backend)
vault secrets enable -version=2 -path=secret kv

# Create policy for AICtrlNet
vault policy write aictrlnet-platform - <<EOF
path "secret/data/aictrlnet/platforms/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}
path "secret/metadata/aictrlnet/platforms/*" {
  capabilities = ["read", "list", "delete"]
}
EOF

# Create token with policy
vault token create -policy=aictrlnet-platform
```

## Migration Between Backends

### From Environment to Database
```python
# Script to migrate credentials
import os
import json
from aictrlnet.services import PlatformCredentialService

# Read from environment
for key, value in os.environ.items():
    if key.startswith("PLATFORM_CRED_"):
        platform_key = key[14:].lower()
        credential_data = json.loads(value)
        # Store in database using service
```

### From Database to Vault
```bash
# Export from database and import to Vault
# Use the platform integration API endpoints
```

## Best Practices

### Development
- Use file backend or database backend
- Auto-generate encryption keys
- Store test credentials only

### Staging
- Use database backend with proper encryption key
- Test credential rotation procedures
- Enable audit logging

### Production
- Use Vault backend for enterprise deployments
- Use database backend for cloud SaaS deployments
- Never auto-generate encryption keys
- Store encryption keys in secure key management service
- Enable comprehensive audit logging
- Implement credential rotation policies
- Use least-privilege access controls

## Troubleshooting

### Common Issues

1. **"No encryption key found"**
   - Set `PLATFORM_CREDENTIAL_KEY` for database backend
   - Set `CREDENTIAL_ENCRYPTION_KEY` for file backend

2. **"Failed to authenticate with Vault"**
   - Check `VAULT_TOKEN` is valid
   - Verify `VAULT_URL` is accessible
   - Ensure Vault is unsealed

3. **"Cannot store credentials to environment"**
   - Environment backend is read-only
   - Switch to file/database/vault backend for dynamic storage

4. **"Permission denied" for file backend**
   - Check file permissions (should be 600)
   - Ensure directory exists and is writable

## Security Considerations

1. **Encryption Keys**
   - Never commit encryption keys to version control
   - Use different keys for different environments
   - Rotate keys periodically
   - Store keys in secure key management service

2. **Access Control**
   - Implement least-privilege access
   - Use database row-level security for multi-tenant
   - Configure Vault policies carefully
   - Audit credential access regularly

3. **Network Security**
   - Use TLS for all connections
   - Restrict network access to credential stores
   - Use private endpoints for cloud deployments

4. **Compliance**
   - Enable audit logging for all credential operations
   - Implement credential rotation policies
   - Document access procedures
   - Regular security assessments