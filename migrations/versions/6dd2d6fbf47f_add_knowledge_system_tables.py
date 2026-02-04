"""add_knowledge_system_tables

Revision ID: 6dd2d6fbf47f
Revises: 43ec3e926d2e
Create Date: 2025-09-22 17:41:47.397916

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '6dd2d6fbf47f'
down_revision = '43ec3e926d2e'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Check if tables already exist before creating
    connection = op.get_bind()
    inspector = sa.inspect(connection)
    existing_tables = inspector.get_table_names()

    # knowledge_index table
    if 'knowledge_index' in existing_tables:
        print("knowledge_index table already exists, skipping creation...")
    else:
        op.create_table('knowledge_index',
            sa.Column('id', sa.UUID(), nullable=False),
            sa.Column('index_version', sa.String(length=20), nullable=False),
            sa.Column('index_type', sa.String(length=50), nullable=False),
            sa.Column('index_data', sa.JSON(), nullable=False),
            sa.Column('total_items', sa.Integer(), nullable=False),
            sa.Column('item_counts', sa.JSON(), nullable=False),
            sa.Column('built_at', sa.DateTime(), nullable=False),
            sa.Column('build_duration_ms', sa.Integer(), nullable=True),
            sa.Column('last_refresh', sa.DateTime(), nullable=True),
            sa.Column('index_config', sa.JSON(), nullable=False),
            sa.Column('is_current', sa.Boolean(), nullable=False),
            sa.Column('needs_rebuild', sa.Boolean(), nullable=False),
            sa.PrimaryKeyConstraint('id')
        )

    # knowledge_items table
    if 'knowledge_items' in existing_tables:
        print("knowledge_items table already exists, skipping creation...")
    else:
        op.create_table('knowledge_items',
            sa.Column('id', sa.UUID(), nullable=False),
            sa.Column('item_type', sa.String(length=50), nullable=False),
            sa.Column('category', sa.String(length=100), nullable=False),
            sa.Column('name', sa.String(length=200), nullable=False),
            sa.Column('description', sa.Text(), nullable=False),
            sa.Column('content', sa.JSON(), nullable=False),
            sa.Column('tags', sa.JSON(), nullable=False),
            sa.Column('keywords', sa.JSON(), nullable=False),
            sa.Column('semantic_embedding', sa.JSON(), nullable=True),
            sa.Column('usage_count', sa.Integer(), nullable=False),
            sa.Column('relevance_score', sa.Float(), nullable=False),
            sa.Column('success_rate', sa.Float(), nullable=True),
            sa.Column('source_file', sa.String(length=500), nullable=True),
            sa.Column('source_version', sa.String(length=50), nullable=True),
            sa.Column('edition_required', sa.String(length=20), nullable=False),
            sa.Column('related_items', sa.JSON(), nullable=False),
            sa.Column('dependencies', sa.JSON(), nullable=False),
            sa.Column('created_at', sa.DateTime(), nullable=False),
            sa.Column('updated_at', sa.DateTime(), nullable=False),
            sa.Column('last_accessed', sa.DateTime(), nullable=True),
            sa.Column('is_active', sa.Boolean(), nullable=False),
            sa.Column('is_deprecated', sa.Boolean(), nullable=False),
            sa.PrimaryKeyConstraint('id')
        )
        # Create indexes only if table was just created
        op.create_index('idx_knowledge_name_type', 'knowledge_items', ['name', 'item_type'], unique=False)
        op.create_index('idx_knowledge_type_category', 'knowledge_items', ['item_type', 'category'], unique=False)
        op.create_index('idx_knowledge_usage', 'knowledge_items', ['usage_count', 'relevance_score'], unique=False)
        op.create_index(op.f('ix_knowledge_items_category'), 'knowledge_items', ['category'], unique=False)
        op.create_index(op.f('ix_knowledge_items_item_type'), 'knowledge_items', ['item_type'], unique=False)
        op.create_index(op.f('ix_knowledge_items_name'), 'knowledge_items', ['name'], unique=False)

    # learned_patterns table
    if 'learned_patterns' in existing_tables:
        print("learned_patterns table already exists, skipping creation...")
    else:
        op.create_table('learned_patterns',
            sa.Column('id', sa.UUID(), nullable=False),
            sa.Column('pattern_type', sa.String(length=50), nullable=False),
            sa.Column('pattern_signature', sa.String(length=200), nullable=False),
            sa.Column('pattern_data', sa.JSON(), nullable=False),
            sa.Column('context_requirements', sa.JSON(), nullable=False),
            sa.Column('occurrence_count', sa.Integer(), nullable=False),
            sa.Column('success_count', sa.Integer(), nullable=False),
            sa.Column('confidence_score', sa.Float(), nullable=False),
            sa.Column('is_active', sa.Boolean(), nullable=False),
            sa.Column('activation_threshold', sa.Float(), nullable=False),
            sa.Column('last_applied', sa.DateTime(), nullable=True),
            sa.Column('application_count', sa.Integer(), nullable=False),
            sa.Column('first_observed', sa.DateTime(), nullable=False),
            sa.Column('last_observed', sa.DateTime(), nullable=False),
            sa.Column('is_validated', sa.Boolean(), nullable=False),
            sa.Column('validated_by', sa.String(length=36), nullable=True),
            sa.Column('validated_at', sa.DateTime(), nullable=True),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index('idx_pattern_active', 'learned_patterns', ['is_active', 'confidence_score'], unique=False)
        op.create_index('idx_pattern_signature', 'learned_patterns', ['pattern_signature', 'pattern_type'], unique=False)
        op.create_index(op.f('ix_learned_patterns_pattern_signature'), 'learned_patterns', ['pattern_signature'], unique=False)

    # system_manifests table
    if 'system_manifests' in existing_tables:
        print("system_manifests table already exists, skipping creation...")
    else:
        op.create_table('system_manifests',
            sa.Column('id', sa.UUID(), nullable=False),
            sa.Column('manifest_version', sa.String(length=20), nullable=False),
            sa.Column('manifest_type', sa.String(length=50), nullable=False),
            sa.Column('manifest_data', sa.JSON(), nullable=False),
            sa.Column('statistics', sa.JSON(), nullable=False),
            sa.Column('feature_count', sa.Integer(), nullable=False),
            sa.Column('endpoint_count', sa.Integer(), nullable=False),
            sa.Column('template_count', sa.Integer(), nullable=False),
            sa.Column('agent_count', sa.Integer(), nullable=False),
            sa.Column('adapter_count', sa.Integer(), nullable=False),
            sa.Column('generated_at', sa.DateTime(), nullable=False),
            sa.Column('generation_time_ms', sa.Integer(), nullable=True),
            sa.Column('is_current', sa.Boolean(), nullable=False),
            sa.Column('expires_at', sa.DateTime(), nullable=True),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index('idx_manifest_current', 'system_manifests', ['is_current', 'manifest_type'], unique=False)

    # knowledge_queries table
    if 'knowledge_queries' in existing_tables:
        print("knowledge_queries table already exists, skipping creation...")
    else:
        op.create_table('knowledge_queries',
            sa.Column('id', sa.UUID(), nullable=False),
            sa.Column('query_text', sa.Text(), nullable=False),
            sa.Column('query_type', sa.String(length=50), nullable=False),
            sa.Column('context', sa.JSON(), nullable=False),
            sa.Column('user_id', sa.String(length=36), nullable=True),
            sa.Column('session_id', sa.UUID(), nullable=True),
            sa.Column('results_returned', sa.JSON(), nullable=False),
            sa.Column('result_count', sa.Integer(), nullable=False),
            sa.Column('top_result_id', sa.UUID(), nullable=True),
            sa.Column('query_time_ms', sa.Integer(), nullable=True),
            sa.Column('retrieval_method', sa.String(length=50), nullable=True),
            sa.Column('was_helpful', sa.Boolean(), nullable=True),
            sa.Column('user_selected_item', sa.UUID(), nullable=True),
            sa.Column('feedback_notes', sa.Text(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(['session_id'], ['conversation_sessions.id'], ),
            sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index(op.f('ix_knowledge_queries_user_id'), 'knowledge_queries', ['user_id'], unique=False)


def downgrade() -> None:
    # Drop tables in reverse order (respecting foreign keys)
    connection = op.get_bind()
    inspector = sa.inspect(connection)
    existing_tables = inspector.get_table_names()

    if 'knowledge_queries' in existing_tables:
        op.drop_index(op.f('ix_knowledge_queries_user_id'), table_name='knowledge_queries')
        op.drop_table('knowledge_queries')

    if 'system_manifests' in existing_tables:
        op.drop_index('idx_manifest_current', table_name='system_manifests')
        op.drop_table('system_manifests')

    if 'learned_patterns' in existing_tables:
        op.drop_index(op.f('ix_learned_patterns_pattern_signature'), table_name='learned_patterns')
        op.drop_index('idx_pattern_signature', table_name='learned_patterns')
        op.drop_index('idx_pattern_active', table_name='learned_patterns')
        op.drop_table('learned_patterns')

    if 'knowledge_items' in existing_tables:
        op.drop_index(op.f('ix_knowledge_items_name'), table_name='knowledge_items')
        op.drop_index(op.f('ix_knowledge_items_item_type'), table_name='knowledge_items')
        op.drop_index(op.f('ix_knowledge_items_category'), table_name='knowledge_items')
        op.drop_index('idx_knowledge_usage', table_name='knowledge_items')
        op.drop_index('idx_knowledge_type_category', table_name='knowledge_items')
        op.drop_index('idx_knowledge_name_type', table_name='knowledge_items')
        op.drop_table('knowledge_items')

    if 'knowledge_index' in existing_tables:
        op.drop_table('knowledge_index')
