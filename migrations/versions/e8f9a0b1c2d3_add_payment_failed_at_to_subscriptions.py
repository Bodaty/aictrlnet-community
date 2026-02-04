"""add_payment_failed_at_to_subscriptions

Add payment_failed_at column to subscriptions table for tracking
payment failure grace periods in Stripe integration.

Revision ID: e8f9a0b1c2d3
Revises: d7e8f9a0b1c2
Create Date: 2026-02-04 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "e8f9a0b1c2d3"
down_revision: Union[str, None] = "d7e8f9a0b1c2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    # Check if subscriptions table exists
    existing_tables = inspector.get_table_names()
    if "subscriptions" not in existing_tables:
        # Table doesn't exist yet - it will be created by initial migration
        return

    # Check if column already exists
    existing_columns = [col["name"] for col in inspector.get_columns("subscriptions")]

    if "payment_failed_at" not in existing_columns:
        op.add_column(
            "subscriptions",
            sa.Column("payment_failed_at", sa.DateTime(), nullable=True)
        )


def downgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    existing_tables = inspector.get_table_names()
    if "subscriptions" not in existing_tables:
        return

    existing_columns = [col["name"] for col in inspector.get_columns("subscriptions")]

    if "payment_failed_at" in existing_columns:
        op.drop_column("subscriptions", "payment_failed_at")
