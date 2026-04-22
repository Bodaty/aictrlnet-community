"""Add AI Control Spectrum autonomy columns to workflow_definitions.

Revision ID: a1b2c3d4e5f6
Revises: d5e6f7a8b9c0
Create Date: 2026-04-21

Adds `autonomy_level` (Int NULL, 0-100) and `autonomy_locked` (Boolean, default
false) to workflow_definitions. These support the per-workflow autonomy
override in the Control Spectrum hierarchy.

Idempotent: uses inspector-pattern existence checks per project policy.
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers
revision = "a7b8c9d0e1f2"
down_revision = "d5e6f7a8b9c0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    existing = {c["name"] for c in inspector.get_columns("workflow_definitions")}

    if "autonomy_level" not in existing:
        op.add_column(
            "workflow_definitions",
            sa.Column("autonomy_level", sa.Integer(), nullable=True),
        )

    if "autonomy_locked" not in existing:
        op.add_column(
            "workflow_definitions",
            sa.Column(
                "autonomy_locked",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("false"),
            ),
        )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    existing = {c["name"] for c in inspector.get_columns("workflow_definitions")}

    if "autonomy_locked" in existing:
        op.drop_column("workflow_definitions", "autonomy_locked")
    if "autonomy_level" in existing:
        op.drop_column("workflow_definitions", "autonomy_level")
