"""add cancel_at_period_end to subscriptions

Revision ID: b3c9f1e2d4a7
Revises: ea884d0a4c00
Create Date: 2026-03-05 23:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'b3c9f1e2d4a7'
down_revision = 'ea884d0a4c00'
branch_labels = None
depends_on = None


def upgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)
    if 'subscriptions' not in inspector.get_table_names():
        return
    existing = [c['name'] for c in inspector.get_columns('subscriptions')]
    if 'cancel_at_period_end' not in existing:
        op.add_column('subscriptions',
            sa.Column('cancel_at_period_end', sa.Boolean(), server_default='false'))


def downgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)
    if 'subscriptions' not in inspector.get_table_names():
        return
    existing = [c['name'] for c in inspector.get_columns('subscriptions')]
    if 'cancel_at_period_end' in existing:
        op.drop_column('subscriptions', 'cancel_at_period_end')
