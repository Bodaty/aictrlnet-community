"""users flags get a server default and NOT NULL.

``is_active``, ``is_superuser``, ``email_verified`` and ``mfa_enabled`` had an ORM default only, so a
row written outside the ORM (seeds, harnesses, fixtures) held NULL: a NULL ``is_superuser`` made
``GET /auth/me`` return 500, and a NULL ``is_active`` could not sign in. NULLs are backfilled to the
ORM default, then each flag gets that default on the server and NOT NULL.

Revision ID: us1fl2ag3df4
Revises: ia1te2na3nt4
"""

import sqlalchemy as sa
from alembic import op

revision = "us1fl2ag3df4"
down_revision = "ia1te2na3nt4"
branch_labels = None
depends_on = None

FLAGS = {"is_active": "true", "is_superuser": "false", "email_verified": "false", "mfa_enabled": "false"}


def upgrade():
    # users has RLS; the backfill must see every tenant's rows.
    op.execute("SET LOCAL app.is_admin = 'true'")
    for column, default in FLAGS.items():
        op.execute(f"UPDATE users SET {column} = {default} WHERE {column} IS NULL")
        op.alter_column("users", column, existing_type=sa.Boolean(),
                        server_default=sa.text(default), nullable=False)


def downgrade():
    for column in FLAGS:
        op.alter_column("users", column, existing_type=sa.Boolean(), server_default=None, nullable=True)
