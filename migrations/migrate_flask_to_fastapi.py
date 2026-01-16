#!/usr/bin/env python3
"""
Migration script from Flask AICtrlNet to FastAPI AICtrlNet.

This script migrates data from Flask-based database to FastAPI-based database.
It handles schema differences and data transformations.
"""

import asyncio
import logging
import sys
from datetime import datetime
from typing import Dict, Any, Optional

import asyncpg
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import sessionmaker
import click

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlaskToFastAPIMigrator:
    """Handles migration from Flask to FastAPI database schemas."""
    
    def __init__(self, flask_db_url: str, fastapi_db_url: str):
        self.flask_db_url = flask_db_url
        self.fastapi_db_url = fastapi_db_url
        self.flask_engine = None
        self.fastapi_conn = None
        self.migrated_tables = []
        self.failed_tables = []
    
    async def connect(self):
        """Establish database connections."""
        # Connect to Flask database using SQLAlchemy
        self.flask_engine = create_engine(self.flask_db_url)
        
        # Connect to FastAPI database using asyncpg
        self.fastapi_conn = await asyncpg.connect(self.fastapi_db_url)
        
        logger.info("Connected to both databases")
    
    async def disconnect(self):
        """Close database connections."""
        if self.flask_engine:
            self.flask_engine.dispose()
        if self.fastapi_conn:
            await self.fastapi_conn.close()
        
        logger.info("Disconnected from databases")
    
    async def migrate_table(self, source_table: str, target_table: str, 
                          field_mappings: Optional[Dict[str, str]] = None,
                          transformations: Optional[Dict[str, callable]] = None):
        """Migrate data from one table to another with field mappings."""
        try:
            # Read from Flask database
            metadata = MetaData()
            table = Table(source_table, metadata, autoload_with=self.flask_engine)
            
            with self.flask_engine.connect() as conn:
                result = conn.execute(table.select())
                rows = result.fetchall()
            
            if not rows:
                logger.info(f"No data to migrate in {source_table}")
                return
            
            # Prepare column names
            source_columns = list(result.keys())
            
            # Apply field mappings
            if field_mappings:
                target_columns = [field_mappings.get(col, col) for col in source_columns]
            else:
                target_columns = source_columns
            
            # Prepare insert query
            placeholders = ', '.join([f'${i+1}' for i in range(len(target_columns))])
            columns_str = ', '.join(target_columns)
            insert_query = f"""
                INSERT INTO {target_table} ({columns_str})
                VALUES ({placeholders})
                ON CONFLICT (id) DO NOTHING
            """
            
            # Transform and insert data
            migrated_count = 0
            for row in rows:
                values = []
                for i, col in enumerate(source_columns):
                    value = row[i]
                    
                    # Apply transformations if specified
                    if transformations and col in transformations:
                        value = transformations[col](value)
                    
                    values.append(value)
                
                try:
                    await self.fastapi_conn.execute(insert_query, *values)
                    migrated_count += 1
                except Exception as e:
                    logger.error(f"Failed to migrate row in {target_table}: {e}")
            
            logger.info(f"Migrated {migrated_count}/{len(rows)} rows from {source_table} to {target_table}")
            self.migrated_tables.append(target_table)
            
        except Exception as e:
            logger.error(f"Failed to migrate table {source_table}: {e}")
            self.failed_tables.append(source_table)
    
    async def migrate_community_edition(self):
        """Migrate Community Edition tables."""
        logger.info("Migrating Community Edition tables...")
        
        # Tasks table
        await self.migrate_table(
            'tasks', 'tasks',
            field_mappings={'metadata': 'task_metadata'}
        )
        
        # Workflow definitions
        await self.migrate_table(
            'workflow_definitions', 'workflow_definitions',
            field_mappings={'definition': 'workflow_definition'}
        )
        
        # Workflow instances
        await self.migrate_table('workflow_instances', 'workflow_instances')
        
        # Workflow steps
        await self.migrate_table('workflow_steps', 'workflow_steps')
        
        # Adapters
        await self.migrate_table(
            'adapters', 'adapters',
            field_mappings={'type': 'adapter_type'}
        )
        
        # MCP servers
        await self.migrate_table('mcp_servers', 'mcp_servers')
        
        # MCP tools
        await self.migrate_table(
            'mcp_tools', 'mcp_tools',
            field_mappings={'schema': 'tool_schema'}
        )
        
        # Bridge connections
        await self.migrate_table('bridge_connections', 'bridge_connections')
    
    async def migrate_business_edition(self):
        """Migrate Business Edition tables."""
        logger.info("Migrating Business Edition tables...")
        
        # Approval workflows
        await self.migrate_table(
            'approval_workflows', 'approval_workflows',
            field_mappings={'approval_steps': 'approval_steps_config'}
        )
        
        # Approval steps
        await self.migrate_table('approval_steps', 'approval_steps')
        
        # Approval requests
        await self.migrate_table(
            'approval_requests', 'approval_requests',
            field_mappings={'metadata': 'request_metadata'}
        )
        
        # Users
        await self.migrate_table(
            'users', 'users',
            field_mappings={'password_hash': 'hashed_password', 'is_admin': 'is_superuser'}
        )
        
        # Roles
        await self.migrate_table('roles', 'roles')
        
        # User roles
        await self.migrate_table(
            'user_roles', 'user_roles',
            field_mappings={'created_at': 'granted_at'}
        )
        
        # Policies
        await self.migrate_table(
            'policies', 'policies',
            field_mappings={'type': 'policy_type'}
        )
        
        # State entries - with timestamp transformation
        def datetime_to_timestamp(dt):
            if dt is None:
                return None
            if isinstance(dt, datetime):
                return dt.timestamp()
            return dt
        
        await self.migrate_table(
            'state_entries', 'state_entries',
            field_mappings={'metadata': 'state_metadata'},
            transformations={
                'created_at': datetime_to_timestamp,
                'updated_at': datetime_to_timestamp,
                'expires_at': datetime_to_timestamp
            }
        )
        
        # Validation rules
        await self.migrate_table('validation_rules', 'validation_rules')
    
    async def migrate_enterprise_edition(self):
        """Migrate Enterprise Edition tables."""
        logger.info("Migrating Enterprise Edition tables...")
        
        # Tenants
        await self.migrate_table(
            'tenants', 'tenants',
            field_mappings={'metadata': 'tenant_metadata'}
        )
        
        # Tenant users
        await self.migrate_table('tenant_users', 'tenant_users')
        
        # Federated instances
        await self.migrate_table(
            'federated_instances', 'federated_instances',
            field_mappings={'metadata': 'instance_metadata'}
        )
        
        # Resource quotas
        await self.migrate_table(
            'resource_quotas', 'resource_quotas',
            field_mappings={'metadata': 'quota_metadata'}
        )
        
        # Resource usage
        await self.migrate_table(
            'resource_usage', 'resource_usage',
            field_mappings={'metadata': 'usage_metadata'}
        )
        
        # Audit logs
        await self.migrate_table('audit_logs', 'audit_logs')
    
    async def run_migration(self, edition: str = 'all'):
        """Run the migration for specified edition."""
        await self.connect()
        
        try:
            if edition in ['all', 'community']:
                await self.migrate_community_edition()
            
            if edition in ['all', 'business']:
                await self.migrate_business_edition()
            
            if edition in ['all', 'enterprise']:
                await self.migrate_enterprise_edition()
            
            logger.info(f"\nMigration Summary:")
            logger.info(f"Successfully migrated: {len(self.migrated_tables)} tables")
            logger.info(f"Failed migrations: {len(self.failed_tables)} tables")
            
            if self.failed_tables:
                logger.error(f"Failed tables: {', '.join(self.failed_tables)}")
            
        finally:
            await self.disconnect()


@click.command()
@click.option('--flask-db', required=True, help='Flask database URL')
@click.option('--fastapi-db', required=True, help='FastAPI database URL')
@click.option('--edition', default='all', type=click.Choice(['all', 'community', 'business', 'enterprise']))
@click.option('--dry-run', is_flag=True, help='Show what would be migrated without actually migrating')
def main(flask_db: str, fastapi_db: str, edition: str, dry_run: bool):
    """Migrate AICtrlNet data from Flask to FastAPI database."""
    if dry_run:
        logger.info("DRY RUN MODE - No data will be migrated")
    
    migrator = FlaskToFastAPIMigrator(flask_db, fastapi_db)
    
    # Run migration
    asyncio.run(migrator.run_migration(edition))


if __name__ == '__main__':
    main()