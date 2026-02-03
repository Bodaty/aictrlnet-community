"""add_marketplace_tables

Community Marketplace tables — items, reviews, and installations.
Uses inspector pattern to avoid conflicts with Business edition tables.

Revision ID: c5d6e7f8a9b0
Revises: b4c7d9e1f234
Create Date: 2026-02-02 12:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "c5d6e7f8a9b0"
down_revision: Union[str, None] = "b4c7d9e1f234"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)
    existing_tables = inspector.get_table_names()

    # ── marketplace_items ─────────────────────────────────────────────
    if "marketplace_items" not in existing_tables:
        op.create_table(
            "marketplace_items",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("name", sa.String(255), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("short_description", sa.String(500), nullable=True),
            sa.Column("author_id", sa.String(), sa.ForeignKey("users.id"), nullable=False),
            sa.Column("author_name", sa.String(255), nullable=False),
            sa.Column("category", sa.String(50), nullable=False),
            sa.Column("item_type", sa.String(100), nullable=True),
            sa.Column("version", sa.String(50), nullable=False, server_default="1.0.0"),
            sa.Column("tags", sa.JSON(), default=[]),
            sa.Column("config_schema", sa.JSON(), default={}),
            sa.Column("install_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("rating_avg", sa.Float(), nullable=True),
            sa.Column("rating_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("status", sa.String(50), nullable=False, server_default="draft"),
            sa.Column("visibility", sa.String(50), nullable=False, server_default="public"),
            sa.Column("resource_metadata", sa.JSON(), default={}),
            sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
        )
        op.create_index("ix_marketplace_items_category", "marketplace_items", ["category"])
        op.create_index("ix_marketplace_items_status", "marketplace_items", ["status"])
        op.create_index("ix_marketplace_items_visibility", "marketplace_items", ["visibility"])
        op.create_index("ix_marketplace_items_author", "marketplace_items", ["author_id"])
        op.create_index("ix_marketplace_items_rating", "marketplace_items", ["rating_avg"])

    # ── marketplace_reviews ───────────────────────────────────────────
    if "marketplace_reviews" not in existing_tables:
        op.create_table(
            "marketplace_reviews",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("item_id", sa.String(), sa.ForeignKey("marketplace_items.id"), nullable=False),
            sa.Column("user_id", sa.String(), sa.ForeignKey("users.id"), nullable=False),
            sa.Column("rating", sa.Integer(), nullable=False),
            sa.Column("comment", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        )
        op.create_index("ix_marketplace_reviews_item", "marketplace_reviews", ["item_id"])
        op.create_index("ix_marketplace_reviews_user", "marketplace_reviews", ["user_id"])

    # ── marketplace_installations ─────────────────────────────────────
    if "marketplace_installations" not in existing_tables:
        op.create_table(
            "marketplace_installations",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("item_id", sa.String(), sa.ForeignKey("marketplace_items.id"), nullable=False),
            sa.Column("user_id", sa.String(), sa.ForeignKey("users.id"), nullable=False),
            sa.Column("organization_id", sa.String(), nullable=True),
            sa.Column("version", sa.String(50), nullable=False),
            sa.Column("status", sa.String(50), nullable=False, server_default="installed"),
            sa.Column("installed_at", sa.DateTime(), server_default=sa.func.now()),
            sa.Column("uninstalled_at", sa.DateTime(), nullable=True),
        )
        op.create_index("ix_marketplace_installations_item", "marketplace_installations", ["item_id"])
        op.create_index("ix_marketplace_installations_user", "marketplace_installations", ["user_id"])
        op.create_index("ix_marketplace_installations_org", "marketplace_installations", ["organization_id"])
        op.create_index("ix_marketplace_installations_status", "marketplace_installations", ["status"])


def downgrade() -> None:
    op.drop_table("marketplace_installations")
    op.drop_table("marketplace_reviews")
    op.drop_table("marketplace_items")
