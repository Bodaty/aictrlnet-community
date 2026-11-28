"""workflow_definitions.run_as_user_id: whose connections an unattended run uses.

A scheduled, work-item or trigger start has no user, so a step that needs a person's
connection (a Gmail OAuth token, a saved key on the shared hosted tenant) had no one to
resolve it for. The owner is a server-only column, never taken from a request: set when a
workflow is created (its creator), by solution set-up after the admin confirms, and by a
later change-owner action. A deleted user leaves the workflow ownerless.

Revision ID: rr1un2as3ow4
Revises: us1fl2ag3df4
"""

import sqlalchemy as sa
from alembic import op

revision = "rr1un2as3ow4"
down_revision = "us1fl2ag3df4"
branch_labels = None
depends_on = None

FK = "fk_workflow_definitions_run_as_user_id"


def upgrade():
    columns = {c["name"] for c in sa.inspect(op.get_bind()).get_columns("workflow_definitions")}
    if "run_as_user_id" not in columns:
        op.add_column("workflow_definitions", sa.Column("run_as_user_id", sa.String(36), nullable=True))
        op.create_foreign_key(FK, "workflow_definitions", "users", ["run_as_user_id"], ["id"],
                              ondelete="SET NULL")


def downgrade():
    columns = {c["name"] for c in sa.inspect(op.get_bind()).get_columns("workflow_definitions")}
    if "run_as_user_id" in columns:
        op.drop_constraint(FK, "workflow_definitions", type_="foreignkey")
        op.drop_column("workflow_definitions", "run_as_user_id")
