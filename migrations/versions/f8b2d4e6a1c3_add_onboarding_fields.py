"""Add onboarding_state and user_context columns to personal_agent_configs.

Revision ID: f8b2d4e6a1c3
Revises: c7a1b2d3e4f5
Create Date: 2026-02-16
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision = 'f8b2d4e6a1c3'
down_revision = 'c7a1b2d3e4f5'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Use inspector-pattern for idempotent migration
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    if inspector.has_table('personal_agent_configs'):
        existing_columns = [
            col['name'] for col in inspector.get_columns('personal_agent_configs')
        ]

        if 'onboarding_state' not in existing_columns:
            op.add_column(
                'personal_agent_configs',
                sa.Column('onboarding_state', sa.JSON(), nullable=True,
                          server_default='{"status": "not_started", "current_chapter": 1, "completed_chapters": []}')
            )

        if 'user_context' not in existing_columns:
            op.add_column(
                'personal_agent_configs',
                sa.Column('user_context', sa.JSON(), nullable=True, server_default='{}')
            )

        # Backfill existing rows that have NULL values
        op.execute(text(
            "UPDATE personal_agent_configs SET onboarding_state = "
            "'{\"status\": \"not_started\", \"current_chapter\": 1, \"completed_chapters\": []}' "
            "WHERE onboarding_state IS NULL"
        ))
        op.execute(text(
            "UPDATE personal_agent_configs SET user_context = '{}' WHERE user_context IS NULL"
        ))


def downgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    if inspector.has_table('personal_agent_configs'):
        existing_columns = [
            col['name'] for col in inspector.get_columns('personal_agent_configs')
        ]

        if 'user_context' in existing_columns:
            op.drop_column('personal_agent_configs', 'user_context')

        if 'onboarding_state' in existing_columns:
            op.drop_column('personal_agent_configs', 'onboarding_state')
