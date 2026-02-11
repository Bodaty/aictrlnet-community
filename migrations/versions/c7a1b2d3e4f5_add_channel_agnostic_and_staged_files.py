"""Add channel-agnostic fields, staged_files, and channel_links tables.

Revision ID: c7a1b2d3e4f5
Revises: e8f9a0b1c2d3
Create Date: 2026-02-10

Adds:
- channel_bindings, primary_channel to conversation_sessions
- channel_type, external_message_id to conversation_messages
- staged_files table for file upload pipeline
- channel_links table for authenticated channel identities
- channel_link_codes table for linking verification codes
"""

revision = 'c7a1b2d3e4f5'
down_revision = 'e8f9a0b1c2d3'
branch_labels = None
depends_on = None

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from alembic import op


def upgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    # --- conversation_sessions: add channel columns ---
    existing_cols = [col['name'] for col in inspector.get_columns('conversation_sessions')]

    if 'channel_bindings' not in existing_cols:
        op.add_column('conversation_sessions',
                       sa.Column('channel_bindings', sa.JSON(), nullable=False, server_default='{}'))
    if 'primary_channel' not in existing_cols:
        op.add_column('conversation_sessions',
                       sa.Column('primary_channel', sa.String(50), nullable=False, server_default='web'))

    # --- conversation_messages: add channel columns ---
    msg_cols = [col['name'] for col in inspector.get_columns('conversation_messages')]

    if 'channel_type' not in msg_cols:
        op.add_column('conversation_messages',
                       sa.Column('channel_type', sa.String(50), nullable=False, server_default='web'))
    if 'external_message_id' not in msg_cols:
        op.add_column('conversation_messages',
                       sa.Column('external_message_id', sa.String(255), nullable=True))

    # --- staged_files table ---
    existing_tables = inspector.get_table_names()
    if 'staged_files' not in existing_tables:
        op.create_table(
            'staged_files',
            sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), nullable=False, index=True),
            sa.Column('filename', sa.String(500), nullable=False),
            sa.Column('content_type', sa.String(100), nullable=False),
            sa.Column('file_size', sa.Integer(), nullable=False),
            sa.Column('storage_path', sa.Text(), nullable=False),
            sa.Column('stage', sa.String(50), nullable=False, server_default='uploaded'),
            sa.Column('extracted_data', sa.JSON(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=False),
            sa.Column('expires_at', sa.DateTime(), nullable=False),
        )

    # --- channel_links table (authenticated channel identities) ---
    if 'channel_links' not in existing_tables:
        op.create_table(
            'channel_links',
            sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), nullable=False, index=True),
            sa.Column('channel_type', sa.String(50), nullable=False, index=True),
            sa.Column('channel_user_id', sa.String(255), nullable=False, index=True),
            sa.Column('display_name', sa.String(200), nullable=True),
            sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
            sa.Column('linked_at', sa.DateTime(), nullable=False),
            sa.Column('unlinked_at', sa.DateTime(), nullable=True),
            sa.UniqueConstraint('channel_type', 'channel_user_id', name='uq_channel_identity'),
        )

    # --- channel_link_codes table (short-lived verification codes) ---
    if 'channel_link_codes' not in existing_tables:
        op.create_table(
            'channel_link_codes',
            sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), nullable=False),
            sa.Column('code', sa.String(6), nullable=False, index=True),
            sa.Column('channel_type', sa.String(50), nullable=False),
            sa.Column('expires_at', sa.DateTime(), nullable=False),
            sa.Column('used', sa.Boolean(), nullable=False, server_default='false'),
            sa.Column('created_at', sa.DateTime(), nullable=False),
        )


def downgrade() -> None:
    op.drop_table('channel_link_codes')
    op.drop_table('channel_links')
    op.drop_table('staged_files')
    op.drop_column('conversation_messages', 'external_message_id')
    op.drop_column('conversation_messages', 'channel_type')
    op.drop_column('conversation_sessions', 'primary_channel')
    op.drop_column('conversation_sessions', 'channel_bindings')
