"""Pin each workflow run to the definition version it started on.

Revision ID: dv2ex3pin4s5
Revises: dv1wf2ver3s4
"""

import sqlalchemy as sa
from alembic import op

revision = "dv2ex3pin4s5"
down_revision = "dv1wf2ver3s4"
branch_labels = None
depends_on = None

TABLE = "workflow_executions"
COLUMN = "definition_version_id"
INDEX = "ix_workflow_executions_definition_version_id"


def upgrade():
    inspector = sa.inspect(op.get_bind())
    if COLUMN not in {c["name"] for c in inspector.get_columns(TABLE)}:
        op.add_column(TABLE, sa.Column(COLUMN, sa.String(36), nullable=True))
    if INDEX not in {i["name"] for i in inspector.get_indexes(TABLE)}:
        op.create_index(INDEX, TABLE, [COLUMN])


def downgrade():
    inspector = sa.inspect(op.get_bind())
    if INDEX in {i["name"] for i in inspector.get_indexes(TABLE)}:
        op.drop_index(INDEX, table_name=TABLE)
    if COLUMN in {c["name"] for c in inspector.get_columns(TABLE)}:
        op.drop_column(TABLE, COLUMN)
