-- AICtrlNet Flask to FastAPI Migration Script
-- This script migrates data from Flask-based tables to FastAPI-based tables
-- Run this script after setting up the FastAPI database schema

-- ============================================
-- COMMUNITY EDITION MIGRATIONS
-- ============================================

-- Migrate tasks table
-- Note: The schema is mostly the same, but we need to handle some field differences
INSERT INTO tasks (
    id, name, description, status, priority, due_date, 
    assigned_to, task_metadata, tags, dependencies, 
    created_at, updated_at
)
SELECT 
    id, name, description, status, priority, due_date,
    assigned_to, metadata, tags, dependencies,
    created_at, updated_at
FROM flask_tasks
ON CONFLICT (id) DO NOTHING;

-- Migrate workflow_definitions table
INSERT INTO workflow_definitions (
    id, name, description, category, version, is_active,
    workflow_definition, created_by, created_at, updated_at
)
SELECT 
    id, name, description, category, version, is_active,
    definition, created_by, created_at, updated_at
FROM flask_workflow_definitions
ON CONFLICT (id) DO NOTHING;

-- Migrate workflow_instances table
INSERT INTO workflow_instances (
    id, workflow_id, name, status, context, current_step,
    started_at, completed_at, created_by, error_message,
    created_at, updated_at
)
SELECT 
    id, workflow_id, name, status, context, current_step,
    started_at, completed_at, created_by, error_message,
    created_at, updated_at
FROM flask_workflow_instances
ON CONFLICT (id) DO NOTHING;

-- Migrate workflow_steps table
INSERT INTO workflow_steps (
    id, instance_id, name, step_type, status, input_data,
    output_data, error_message, started_at, completed_at,
    created_at, updated_at
)
SELECT 
    id, instance_id, name, step_type, status, input_data,
    output_data, error_message, started_at, completed_at,
    created_at, updated_at
FROM flask_workflow_steps
ON CONFLICT (id) DO NOTHING;

-- Migrate adapters table
INSERT INTO adapters (
    id, name, adapter_type, description, config, status,
    version, created_at, updated_at
)
SELECT 
    id, name, type, description, config, status,
    version, created_at, updated_at
FROM flask_adapters
ON CONFLICT (id) DO NOTHING;

-- Migrate mcp_servers table
INSERT INTO mcp_servers (
    id, name, description, version, url, api_key, status,
    is_active, created_at, updated_at
)
SELECT 
    id, name, description, version, url, api_key, status,
    is_active, created_at, updated_at
FROM flask_mcp_servers
ON CONFLICT (id) DO NOTHING;

-- Migrate mcp_tools table
INSERT INTO mcp_tools (
    id, server_id, name, description, tool_schema, is_active,
    created_at, updated_at
)
SELECT 
    id, server_id, name, description, schema, is_active,
    created_at, updated_at
FROM flask_mcp_tools
ON CONFLICT (id) DO NOTHING;

-- Migrate bridge_connections table
INSERT INTO bridge_connections (
    id, name, source_type, source_config, destination_type,
    destination_config, mapping_rules, status, is_active,
    created_at, updated_at
)
SELECT 
    id, name, source_type, source_config, destination_type,
    destination_config, mapping_rules, status, is_active,
    created_at, updated_at
FROM flask_bridge_connections
ON CONFLICT (id) DO NOTHING;

-- ============================================
-- BUSINESS EDITION MIGRATIONS
-- ============================================

-- Migrate approval_workflows table
INSERT INTO approval_workflows (
    id, name, description, resource_type, conditions,
    approval_steps_config, enabled, owner_id,
    created_at, updated_at
)
SELECT 
    id, name, description, resource_type, conditions,
    approval_steps, enabled, owner_id,
    created_at, updated_at
FROM flask_approval_workflows
ON CONFLICT (id) DO NOTHING;

-- Migrate approval_steps table
INSERT INTO approval_steps (
    id, workflow_id, name, description, step_number,
    approver_type, approver_id, auto_approve, timeout_hours,
    created_at, updated_at
)
SELECT 
    id, workflow_id, name, description, step_number,
    approver_type, approver_id, auto_approve, timeout_hours,
    created_at, updated_at
FROM flask_approval_steps
ON CONFLICT (id) DO NOTHING;

-- Migrate approval_requests table
INSERT INTO approval_requests (
    id, workflow_id, resource_type, resource_id, status,
    current_step, request_metadata, requester_id,
    created_at, updated_at
)
SELECT 
    id, workflow_id, resource_type, resource_id, status,
    current_step, metadata, requester_id,
    created_at, updated_at
FROM flask_approval_requests
ON CONFLICT (id) DO NOTHING;

-- Migrate users table
INSERT INTO users (
    id, username, email, hashed_password, is_active,
    is_superuser, created_at, updated_at
)
SELECT 
    id, username, email, password_hash, is_active,
    is_admin, created_at, updated_at
FROM flask_users
ON CONFLICT (id) DO NOTHING;

-- Migrate roles table
INSERT INTO roles (
    id, name, description, permissions, created_at, updated_at
)
SELECT 
    id, name, description, permissions, created_at, updated_at
FROM flask_roles
ON CONFLICT (id) DO NOTHING;

-- Migrate user_roles table
INSERT INTO user_roles (
    id, user_id, role_id, granted_by, granted_at
)
SELECT 
    id, user_id, role_id, granted_by, created_at
FROM flask_user_roles
ON CONFLICT (id) DO NOTHING;

-- Migrate policies table
INSERT INTO policies (
    id, name, description, policy_type, resource_type,
    conditions, actions, effect, priority, enabled,
    created_by, created_at, updated_at
)
SELECT 
    id, name, description, type, resource_type,
    conditions, actions, effect, priority, enabled,
    created_by, created_at, updated_at
FROM flask_policies
ON CONFLICT (id) DO NOTHING;

-- Migrate state_entries table
-- Note: Handle timestamp conversion from datetime to float
INSERT INTO state_entries (
    id, key, value, owner, scope, scope_id, version,
    created_at, updated_at, expires_at, state_metadata, permissions
)
SELECT 
    id, key, value, owner, scope, scope_id, version,
    EXTRACT(EPOCH FROM created_at),
    EXTRACT(EPOCH FROM updated_at),
    CASE WHEN expires_at IS NOT NULL THEN EXTRACT(EPOCH FROM expires_at) ELSE NULL END,
    metadata, permissions
FROM flask_state_entries
ON CONFLICT (id) DO NOTHING;

-- Migrate validation_rules table
INSERT INTO validation_rules (
    id, name, description, rule_type, standard_reference,
    rule_definition, severity, component_types, enabled,
    owner_id, created_at, updated_at
)
SELECT 
    id, name, description, rule_type, standard_reference,
    rule_definition, severity, component_types, enabled,
    owner_id, created_at, updated_at
FROM flask_validation_rules
ON CONFLICT (id) DO NOTHING;

-- ============================================
-- ENTERPRISE EDITION MIGRATIONS
-- ============================================

-- Migrate tenants table
INSERT INTO tenants (
    id, name, display_name, description, domain, config,
    status, settings, tenant_metadata, created_at, updated_at
)
SELECT 
    id, name, display_name, description, domain, config,
    status, settings, metadata, created_at, updated_at
FROM flask_tenants
ON CONFLICT (id) DO NOTHING;

-- Migrate tenant_users table
INSERT INTO tenant_users (
    id, tenant_id, user_id, role, is_primary, permissions,
    joined_at, created_at, updated_at
)
SELECT 
    id, tenant_id, user_id, role, is_primary, permissions,
    joined_at, created_at, updated_at
FROM flask_tenant_users
ON CONFLICT (id) DO NOTHING;

-- Migrate federated_instances table
INSERT INTO federated_instances (
    id, name, description, base_url, api_key, status,
    last_sync, sync_interval_minutes, capabilities,
    instance_metadata, created_at, updated_at
)
SELECT 
    id, name, description, base_url, api_key, status,
    last_sync, sync_interval_minutes, capabilities,
    metadata, created_at, updated_at
FROM flask_federated_instances
ON CONFLICT (id) DO NOTHING;

-- Migrate resource_quotas table
INSERT INTO resource_quotas (
    id, tenant_id, resource_type, limit_value, period,
    soft_limit, hard_limit, quota_metadata,
    created_at, updated_at
)
SELECT 
    id, tenant_id, resource_type, limit_value, period,
    soft_limit, hard_limit, metadata,
    created_at, updated_at
FROM flask_resource_quotas
ON CONFLICT (id) DO NOTHING;

-- Migrate resource_usage table
INSERT INTO resource_usage (
    id, tenant_id, resource_type, usage_value,
    period_start, period_end, usage_metadata, recorded_at
)
SELECT 
    id, tenant_id, resource_type, usage_value,
    period_start, period_end, metadata, recorded_at
FROM flask_resource_usage
ON CONFLICT (id) DO NOTHING;

-- Migrate audit_logs table
-- Note: Handle timestamp conversion
INSERT INTO audit_logs (
    id, timestamp, event_type, component_id, resource_type,
    resource_id, action, ip_address, user_agent, request_id,
    endpoint, method, details, status
)
SELECT 
    id, timestamp, event_type, component_id, resource_type,
    resource_id, action, ip_address, user_agent, request_id,
    endpoint, method, details, status
FROM flask_audit_logs
ON CONFLICT (id) DO NOTHING;

-- ============================================
-- POST-MIGRATION CLEANUP
-- ============================================

-- Update sequences to ensure new records get proper IDs
-- This is PostgreSQL specific - adjust for other databases

-- Reset sequences for tables using serial/identity columns
-- Add more as needed based on your specific schema

-- Log migration completion
DO $$
BEGIN
    RAISE NOTICE 'Flask to FastAPI migration completed successfully';
    RAISE NOTICE 'Please verify data integrity and remove flask_ prefixed tables when ready';
END $$;