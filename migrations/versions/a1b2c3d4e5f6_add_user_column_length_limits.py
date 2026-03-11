"""add user column length limits

Revision ID: a1b2c3d4e5f6
Revises: b3c9f1e2d4a7
Create Date: 2026-03-11 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a1b2c3d4e5f6'
down_revision = 'b3c9f1e2d4a7'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # PostgreSQL ALTER TYPE for varchar length is safe on existing data —
    # it only adds a constraint check, no rewrite needed for widening.
    # Narrowing (String -> String(N)) on existing data will fail if any
    # row exceeds N, but our Pydantic schemas have been enforcing these
    # limits so existing data should be within bounds.
    op.alter_column('users', 'id', type_=sa.String(36), existing_type=sa.String())
    op.alter_column('users', 'email', type_=sa.String(254), existing_type=sa.String())
    op.alter_column('users', 'username', type_=sa.String(50), existing_type=sa.String())
    op.alter_column('users', 'hashed_password', type_=sa.String(255), existing_type=sa.String())
    op.alter_column('users', 'full_name', type_=sa.String(100), existing_type=sa.String())
    op.alter_column('users', 'tenant_id', type_=sa.String(100), existing_type=sa.String())
    op.alter_column('users', 'edition', type_=sa.String(20), existing_type=sa.String())


def downgrade() -> None:
    op.alter_column('users', 'id', type_=sa.String(), existing_type=sa.String(36))
    op.alter_column('users', 'email', type_=sa.String(), existing_type=sa.String(254))
    op.alter_column('users', 'username', type_=sa.String(), existing_type=sa.String(50))
    op.alter_column('users', 'hashed_password', type_=sa.String(), existing_type=sa.String(255))
    op.alter_column('users', 'full_name', type_=sa.String(), existing_type=sa.String(100))
    op.alter_column('users', 'tenant_id', type_=sa.String(), existing_type=sa.String(100))
    op.alter_column('users', 'edition', type_=sa.String(), existing_type=sa.String(20))
