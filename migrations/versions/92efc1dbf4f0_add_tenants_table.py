"""add_tenants_table

Revision ID: 92efc1dbf4f0
Revises: 2bc046024965
Create Date: 2026-01-15

This migration adds the basic tenants table for multi-tenant SaaS infrastructure.
Enterprise Edition extends this table with additional columns via its own migration.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision = '92efc1dbf4f0'
down_revision = '2bc046024965'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Check if tenants table already exists (Enterprise may have created it)
    connection = op.get_bind()
    inspector = inspect(connection)
    existing_tables = inspector.get_table_names()

    if 'tenants' in existing_tables:
        # Table exists (likely from Enterprise), ensure our columns are present
        existing_columns = [col['name'] for col in inspector.get_columns('tenants')]

        if 'is_default' not in existing_columns:
            op.add_column('tenants', sa.Column('is_default', sa.Boolean(), server_default='false'))

        # Check if status column uses enum type (Enterprise uses tenantstatus enum)
        # Get column type info
        columns_info = {col['name']: col for col in inspector.get_columns('tenants')}
        status_col = columns_info.get('status', {})

        # Check if enum exists in database
        result = connection.execute(sa.text(
            "SELECT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'tenantstatus')"
        ))
        uses_enum = result.scalar()

        # Ensure default tenant exists with proper status value
        if uses_enum:
            # Enterprise uses enum type - use uppercase enum value
            op.execute("""
                INSERT INTO tenants (id, name, display_name, status, is_default, created_at, updated_at)
                SELECT 'default-tenant', 'default', 'Default Tenant', 'ACTIVE'::tenantstatus, true, NOW(), NOW()
                WHERE NOT EXISTS (SELECT 1 FROM tenants WHERE id = 'default-tenant')
            """)
        else:
            # Community uses string type
            op.execute("""
                INSERT INTO tenants (id, name, display_name, status, is_default, created_at, updated_at)
                SELECT 'default-tenant', 'default', 'Default Tenant', 'active', true, NOW(), NOW()
                WHERE NOT EXISTS (SELECT 1 FROM tenants WHERE id = 'default-tenant')
            """)
        return

    # Create basic tenants table
    op.create_table(
        'tenants',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(128), nullable=False, unique=True),
        sa.Column('display_name', sa.String(256), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, server_default='active'),
        sa.Column('settings', sa.JSON(), nullable=True),
        sa.Column('is_default', sa.Boolean(), server_default='false'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    )

    # Create index on name for faster lookups
    op.create_index('ix_tenants_name', 'tenants', ['name'])
    op.create_index('ix_tenants_status', 'tenants', ['status'])

    # Create default tenant for self-hosted mode
    op.execute("""
        INSERT INTO tenants (id, name, display_name, status, is_default, created_at, updated_at)
        VALUES ('default-tenant', 'default', 'Default Tenant', 'active', true, NOW(), NOW())
    """)


def downgrade() -> None:
    # Check if this was the migration that created the table
    connection = op.get_bind()
    inspector = inspect(connection)
    existing_tables = inspector.get_table_names()

    if 'tenants' in existing_tables:
        # Check if Enterprise added columns - if so, just remove our column
        existing_columns = [col['name'] for col in inspector.get_columns('tenants')]
        enterprise_columns = ['domain', 'config', 'tenant_metadata', 'parent_tenant_id']

        has_enterprise_columns = any(col in existing_columns for col in enterprise_columns)

        if has_enterprise_columns:
            # Enterprise owns this table, just remove our is_default column
            if 'is_default' in existing_columns:
                op.drop_column('tenants', 'is_default')
        else:
            # We own this table, drop it
            op.drop_index('ix_tenants_status', 'tenants')
            op.drop_index('ix_tenants_name', 'tenants')
            op.drop_table('tenants')
