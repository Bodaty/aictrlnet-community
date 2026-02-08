"""add_is_active_to_workflow_templates

Revision ID: 2785e5c2fac9
Revises: 71644546b9dd
Create Date: 2025-10-30 16:47:30.202845

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '2785e5c2fac9'
down_revision = '71644546b9dd'
branch_labels = None
depends_on = None


def upgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)
    existing_columns = [col['name'] for col in inspector.get_columns('workflow_templates')]

    if 'is_active' not in existing_columns:
        op.add_column('workflow_templates', sa.Column('is_active', sa.Boolean(), nullable=True, server_default='true'))
        op.execute("UPDATE workflow_templates SET is_active = true WHERE is_active IS NULL")
        op.alter_column('workflow_templates', 'is_active', nullable=False)


def downgrade() -> None:
    # Remove is_active column
    op.drop_column('workflow_templates', 'is_active')