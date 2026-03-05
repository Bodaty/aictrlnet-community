"""add_iam_agent_type_values

Revision ID: ea884d0a4c00
Revises: f8b2d4e6a1c3
Create Date: 2026-03-05 22:20:18.577828

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'ea884d0a4c00'
down_revision = 'f8b2d4e6a1c3'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TYPE iamagenttype ADD VALUE IF NOT EXISTS 'ai'")
    op.execute("ALTER TYPE iamagenttype ADD VALUE IF NOT EXISTS 'human'")
    op.execute("ALTER TYPE iamagenttype ADD VALUE IF NOT EXISTS 'orchestrator'")


def downgrade() -> None:
    pass  # PostgreSQL does not support removing enum values