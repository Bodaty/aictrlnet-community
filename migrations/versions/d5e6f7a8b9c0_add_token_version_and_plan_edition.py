"""add_token_version_and_plan_edition

Revision ID: d5e6f7a8b9c0
Revises: a1b2c3d4e5f6
Create Date: 2026-03-11 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'd5e6f7a8b9c0'
down_revision = 'a1b2c3d4e5f6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    # Add token_version to users table
    users_columns = [col['name'] for col in inspector.get_columns('users')]
    if 'token_version' not in users_columns:
        op.add_column('users', sa.Column('token_version', sa.Integer(), server_default='0', nullable=False))

    # Add edition to subscription_plans table
    plans_columns = [col['name'] for col in inspector.get_columns('subscription_plans')]
    if 'edition' not in plans_columns:
        op.add_column('subscription_plans', sa.Column('edition', sa.String(20), server_default='community', nullable=True))

        # Backfill edition from plan name
        op.execute("UPDATE subscription_plans SET edition = 'community' WHERE name LIKE 'community%'")
        op.execute("UPDATE subscription_plans SET edition = 'business' WHERE name LIKE 'business%'")
        op.execute("UPDATE subscription_plans SET edition = 'enterprise' WHERE name LIKE 'enterprise%'")
        op.execute("UPDATE subscription_plans SET edition = 'community' WHERE edition IS NULL")

    # Add MFA enforcement fields to tenants table (if table exists)
    if inspector.has_table('tenants'):
        tenants_columns = [col['name'] for col in inspector.get_columns('tenants')]
        if 'mfa_required' not in tenants_columns:
            op.add_column('tenants', sa.Column('mfa_required', sa.Boolean(), server_default='false', nullable=False))
        if 'mfa_grace_period_days' not in tenants_columns:
            op.add_column('tenants', sa.Column('mfa_grace_period_days', sa.Integer(), server_default='7', nullable=False))


def downgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    if inspector.has_table('tenants'):
        tenants_columns = [col['name'] for col in inspector.get_columns('tenants')]
        if 'mfa_grace_period_days' in tenants_columns:
            op.drop_column('tenants', 'mfa_grace_period_days')
        if 'mfa_required' in tenants_columns:
            op.drop_column('tenants', 'mfa_required')

    plans_columns = [col['name'] for col in inspector.get_columns('subscription_plans')]
    if 'edition' in plans_columns:
        op.drop_column('subscription_plans', 'edition')

    users_columns = [col['name'] for col in inspector.get_columns('users')]
    if 'token_version' in users_columns:
        op.drop_column('users', 'token_version')
