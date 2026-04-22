"""api key + oauth2 client: rate_limit_per_tool JSON column

Adds a per-tool rate-limit JSON column to ``api_keys`` and
``oauth2_clients``. Shape:

    {
        "<tool_name>": {"per_minute": 10, "per_day": 100},
        "*":           {"per_minute": 60, "per_day": 2000}
    }

Absent entries fall back to the existing per-minute / per-day columns
(where present) and finally to ``rate_bucket.DEFAULT_LIMITS``.

Revision ID: a3b4c5d6e7f8
Revises: f2b3c4d5e6a7
Create Date: 2026-04-22 13:22:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "a3b4c5d6e7f8"
down_revision = "f2b3c4d5e6a7"
branch_labels = None
depends_on = None


def _has_column(conn, table: str, column: str) -> bool:
    inspector = sa.inspect(conn)
    if table not in inspector.get_table_names():
        return False
    return column in {c["name"] for c in inspector.get_columns(table)}


def upgrade() -> None:
    conn = op.get_bind()

    if not _has_column(conn, "api_keys", "rate_limit_per_tool"):
        op.add_column(
            "api_keys",
            sa.Column("rate_limit_per_tool", sa.JSON(), nullable=True),
        )

    inspector = sa.inspect(conn)
    if "oauth2_clients" in inspector.get_table_names():
        if not _has_column(conn, "oauth2_clients", "rate_limit_per_tool"):
            op.add_column(
                "oauth2_clients",
                sa.Column("rate_limit_per_tool", sa.JSON(), nullable=True),
            )


def downgrade() -> None:
    conn = op.get_bind()

    if _has_column(conn, "api_keys", "rate_limit_per_tool"):
        op.drop_column("api_keys", "rate_limit_per_tool")

    if _has_column(conn, "oauth2_clients", "rate_limit_per_tool"):
        op.drop_column("oauth2_clients", "rate_limit_per_tool")
