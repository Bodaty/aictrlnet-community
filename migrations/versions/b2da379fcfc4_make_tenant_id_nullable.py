"""make_tenant_id_nullable

Revision ID: b2da379fcfc4
Revises: 6dd2d6fbf47f
Create Date: 2025-10-26 17:57:41.398067

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'b2da379fcfc4'
down_revision = '6dd2d6fbf47f'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Make tenant_id nullable to support Community/Business editions
    # NULL = single-tenant (Community/Business)
    # Non-NULL = multi-tenant (Enterprise)
    op.alter_column('users', 'tenant_id',
                    existing_type=sa.String(),
                    nullable=True)


def downgrade() -> None:
    # Revert tenant_id to non-nullable
    # Note: This will fail if NULL values exist
    op.alter_column('users', 'tenant_id',
                    existing_type=sa.String(),
                    nullable=False)