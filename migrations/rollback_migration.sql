-- AICtrlNet FastAPI to Flask Rollback Script
-- This script can be used to rollback the migration if needed
-- WARNING: This will delete all data in FastAPI tables!

-- ============================================
-- ROLLBACK PROCEDURE
-- ============================================

-- 1. First, ensure you have a backup of both databases
-- 2. Stop all FastAPI services
-- 3. Run this script to clear FastAPI tables
-- 4. Restart Flask services

-- ============================================
-- ENTERPRISE EDITION ROLLBACK
-- ============================================

-- Clear Enterprise tables (in reverse dependency order)
TRUNCATE TABLE federation_sync_logs CASCADE;
TRUNCATE TABLE resource_usage CASCADE;
TRUNCATE TABLE resource_quotas CASCADE;
TRUNCATE TABLE cross_tenant_policies CASCADE;
TRUNCATE TABLE tenant_users CASCADE;
TRUNCATE TABLE federated_instances CASCADE;
TRUNCATE TABLE tenants CASCADE;
TRUNCATE TABLE mcp_service_regions CASCADE;
TRUNCATE TABLE federation_connections CASCADE;
TRUNCATE TABLE audit_logs CASCADE;

-- ============================================
-- BUSINESS EDITION ROLLBACK
-- ============================================

-- Clear Business tables
TRUNCATE TABLE policy_violations CASCADE;
TRUNCATE TABLE state_entries CASCADE;
TRUNCATE TABLE validation_rules CASCADE;
TRUNCATE TABLE approval_decisions CASCADE;
TRUNCATE TABLE approval_requests CASCADE;
TRUNCATE TABLE approval_steps CASCADE;
TRUNCATE TABLE approval_workflows CASCADE;
TRUNCATE TABLE user_roles CASCADE;
TRUNCATE TABLE role_permissions CASCADE;
TRUNCATE TABLE permissions CASCADE;
TRUNCATE TABLE roles CASCADE;
TRUNCATE TABLE users CASCADE;
TRUNCATE TABLE policies CASCADE;

-- ============================================
-- COMMUNITY EDITION ROLLBACK
-- ============================================

-- Clear Community tables
TRUNCATE TABLE bridge_syncs CASCADE;
TRUNCATE TABLE bridge_connections CASCADE;
TRUNCATE TABLE mcp_invocations CASCADE;
TRUNCATE TABLE mcp_tools CASCADE;
TRUNCATE TABLE mcp_servers CASCADE;
TRUNCATE TABLE adapters CASCADE;
TRUNCATE TABLE workflow_steps CASCADE;
TRUNCATE TABLE workflow_instances CASCADE;
TRUNCATE TABLE workflow_definitions CASCADE;
TRUNCATE TABLE tasks CASCADE;

-- ============================================
-- VERIFICATION
-- ============================================

-- Verify all tables are empty
DO $$
DECLARE
    table_name text;
    row_count integer;
    total_rows integer := 0;
BEGIN
    FOR table_name IN 
        SELECT tablename 
        FROM pg_tables 
        WHERE schemaname = 'public' 
        AND tablename NOT LIKE 'alembic%'
    LOOP
        EXECUTE format('SELECT COUNT(*) FROM %I', table_name) INTO row_count;
        IF row_count > 0 THEN
            RAISE NOTICE 'Table % still has % rows', table_name, row_count;
            total_rows := total_rows + row_count;
        END IF;
    END LOOP;
    
    IF total_rows = 0 THEN
        RAISE NOTICE 'Rollback completed successfully - all tables are empty';
    ELSE
        RAISE WARNING 'Rollback incomplete - % rows remain in tables', total_rows;
    END IF;
END $$;