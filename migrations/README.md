# AICtrlNet Flask to FastAPI Migration Guide

This directory contains scripts and tools to migrate your AICtrlNet data from Flask-based deployment to FastAPI-based deployment.

## Prerequisites

1. Both Flask and FastAPI deployments should be running
2. Database backups should be created before migration
3. All active users should be notified of the migration window

## Migration Files

- `flask_to_fastapi.sql` - SQL script for direct database migration
- `migrate_flask_to_fastapi.py` - Python script for programmatic migration with transformations
- `rollback_migration.sql` - SQL script to rollback migration if needed
- `verify_migration.py` - Script to verify data integrity after migration

## Migration Process

### Step 1: Backup Databases

```bash
# Backup Flask database
pg_dump -h localhost -U aictrlnet -d aictrlnet_flask > flask_backup_$(date +%Y%m%d_%H%M%S).sql

# Backup FastAPI database (if it has any data)
pg_dump -h localhost -U aictrlnet -d aictrlnet_community > fastapi_backup_$(date +%Y%m%d_%H%M%S).sql
```

### Step 2: Stop Flask Services

```bash
docker-compose -f docker-compose-flask.yml down
```

### Step 3: Run Migration

#### Option A: Using SQL Script (Simple)

```bash
# For PostgreSQL
psql -h localhost -U aictrlnet -d aictrlnet_community < flask_to_fastapi.sql
psql -h localhost -U aictrlnet -d aictrlnet_business < flask_to_fastapi.sql
psql -h localhost -U aictrlnet -d aictrlnet_enterprise < flask_to_fastapi.sql
```

#### Option B: Using Python Script (Recommended)

```bash
# Install dependencies
pip install asyncpg sqlalchemy click

# Run migration for all editions
python migrate_flask_to_fastapi.py \
    --flask-db "postgresql://aictrlnet:password@localhost/aictrlnet_flask" \
    --fastapi-db "postgresql://aictrlnet:password@localhost/aictrlnet_community" \
    --edition all

# Or migrate specific edition
python migrate_flask_to_fastapi.py \
    --flask-db "postgresql://aictrlnet:password@localhost/aictrlnet_flask" \
    --fastapi-db "postgresql://aictrlnet:password@localhost/aictrlnet_business" \
    --edition business
```

### Step 4: Verify Migration

```bash
# Run verification script
python verify_migration.py \
    --flask-db "postgresql://aictrlnet:password@localhost/aictrlnet_flask" \
    --fastapi-db "postgresql://aictrlnet:password@localhost/aictrlnet_community"
```

### Step 5: Update Application Configuration

1. Update environment variables to point to FastAPI endpoints
2. Update any client applications to use new API endpoints
3. Update monitoring and alerting configurations

### Step 6: Start FastAPI Services

```bash
docker-compose -f docker-compose.yml up -d
```

## Schema Differences

The following fields have been renamed in FastAPI:

| Flask Field | FastAPI Field | Table |
|------------|---------------|--------|
| metadata | task_metadata | tasks |
| definition | workflow_definition | workflow_definitions |
| type | adapter_type | adapters |
| schema | tool_schema | mcp_tools |
| metadata | request_metadata | approval_requests |
| password_hash | hashed_password | users |
| is_admin | is_superuser | users |
| type | policy_type | policies |
| metadata | state_metadata | state_entries |
| metadata | tenant_metadata | tenants |
| metadata | instance_metadata | federated_instances |
| metadata | quota_metadata | resource_quotas |
| metadata | usage_metadata | resource_usage |

## Timestamp Handling

Some tables in FastAPI use Unix timestamps (float) instead of datetime:
- state_entries (created_at, updated_at, expires_at)
- mcp_service_regions (created_at)
- policy_violations (created_at)

The migration scripts handle these conversions automatically.

## Rollback Procedure

If you need to rollback the migration:

1. Stop FastAPI services
2. Run the rollback script:
   ```bash
   psql -h localhost -U aictrlnet -d aictrlnet_community < rollback_migration.sql
   ```
3. Restore Flask database from backup if needed
4. Restart Flask services

## Troubleshooting

### Common Issues

1. **Duplicate key errors**: The migration scripts use `ON CONFLICT DO NOTHING` to skip duplicates. This is normal if you're re-running migration.

2. **Foreign key violations**: Ensure you migrate tables in the correct order (parent tables before child tables).

3. **Connection errors**: Verify database credentials and network connectivity.

4. **Memory issues**: For large databases, consider migrating in batches or increasing system memory.

### Getting Help

- Check logs in `/var/log/aictrlnet/migration.log`
- Run migration with `--dry-run` flag to preview changes
- Contact support with migration ID and error messages

## Post-Migration Tasks

1. **Performance Testing**: Run performance tests to ensure FastAPI meets expectations
2. **User Acceptance Testing**: Have users verify their data and workflows
3. **Monitoring**: Set up monitoring for the new FastAPI deployment
4. **Documentation**: Update internal documentation with new endpoints
5. **Training**: Train team members on FastAPI-specific features

## API Endpoint Changes

| Operation | Flask Endpoint | FastAPI Endpoint |
|-----------|---------------|------------------|
| List Tasks | GET /api/tasks | GET /api/v1/tasks |
| Create Task | POST /api/tasks | POST /api/v1/tasks |
| WebSocket | Not available | ws://host/api/v1/ws |
| API Docs | Not available | GET /api/v1/docs |

Note: FastAPI includes automatic API documentation at `/api/v1/docs`