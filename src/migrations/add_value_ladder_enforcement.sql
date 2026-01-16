-- Value Ladder Enforcement Database Schema
-- This creates the necessary tables for usage tracking, license enforcement, and upgrade management

-- Usage metrics table for tracking all usage
CREATE TABLE IF NOT EXISTS usage_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    metric_type VARCHAR(50) NOT NULL,
    value DECIMAL(20,4) NOT NULL DEFAULT 1.0,
    count INTEGER DEFAULT 1,
    metadata JSONB DEFAULT '{}',
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes for efficient querying
    INDEX idx_usage_tenant_time (tenant_id, timestamp),
    INDEX idx_usage_type_time (metric_type, timestamp),
    INDEX idx_usage_tenant_type_time (tenant_id, metric_type, timestamp)
);

-- Tenant limits override for custom deals and special cases
CREATE TABLE IF NOT EXISTS tenant_limit_overrides (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    limit_type VARCHAR(50) NOT NULL,
    limit_value INTEGER NOT NULL,
    reason TEXT,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by UUID,
    
    -- Ensure one override per tenant per limit type
    UNIQUE(tenant_id, limit_type),
    INDEX idx_override_tenant (tenant_id),
    INDEX idx_override_expires (expires_at)
);

-- Feature trials for testing higher-tier features
CREATE TABLE IF NOT EXISTS feature_trials (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    edition_required VARCHAR(20) NOT NULL,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    converted BOOLEAN DEFAULT FALSE,
    converted_at TIMESTAMP,
    
    UNIQUE(tenant_id, feature_name),
    INDEX idx_trials_tenant (tenant_id),
    INDEX idx_trials_expires (expires_at),
    INDEX idx_trials_active (tenant_id, expires_at)
);

-- Upgrade prompts tracking for analytics
CREATE TABLE IF NOT EXISTS upgrade_prompts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    prompt_type VARCHAR(50) NOT NULL, -- limit_warning, feature_locked, trial_expiring, etc.
    trigger_reason VARCHAR(200),
    shown_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    action_taken VARCHAR(50), -- viewed, dismissed, clicked_upgrade, started_trial
    metadata JSONB DEFAULT '{}',
    
    INDEX idx_prompts_tenant (tenant_id),
    INDEX idx_prompts_type (prompt_type),
    INDEX idx_prompts_action (action_taken)
);

-- License cache for performance
CREATE TABLE IF NOT EXISTS license_cache (
    tenant_id UUID PRIMARY KEY,
    edition VARCHAR(50) NOT NULL,
    tier VARCHAR(50), -- For business: starter, growth, scale
    limits JSONB NOT NULL,
    features JSONB NOT NULL,
    expires_at TIMESTAMP,
    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_cache_expires (expires_at)
);

-- Billing events for Stripe integration
CREATE TABLE IF NOT EXISTS billing_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    event_type VARCHAR(50) NOT NULL, -- subscription_created, upgraded, downgraded, cancelled
    stripe_event_id VARCHAR(255) UNIQUE,
    previous_edition VARCHAR(50),
    new_edition VARCHAR(50),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_billing_tenant (tenant_id),
    INDEX idx_billing_stripe (stripe_event_id)
);

-- Monthly usage summaries for billing
CREATE TABLE IF NOT EXISTS usage_summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    month DATE NOT NULL,
    metric_type VARCHAR(50) NOT NULL,
    total_value DECIMAL(20,4) NOT NULL,
    total_count BIGINT NOT NULL,
    daily_breakdown JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(tenant_id, month, metric_type),
    INDEX idx_summary_tenant_month (tenant_id, month)
);

-- Create function to get current usage for a metric
CREATE OR REPLACE FUNCTION get_current_usage(
    p_tenant_id UUID,
    p_metric_type VARCHAR,
    p_period_start TIMESTAMP DEFAULT NULL
)
RETURNS DECIMAL AS $$
DECLARE
    v_usage DECIMAL;
BEGIN
    -- Default to start of current month if not specified
    IF p_period_start IS NULL THEN
        p_period_start := date_trunc('month', CURRENT_TIMESTAMP);
    END IF;
    
    SELECT COALESCE(SUM(value), 0) INTO v_usage
    FROM usage_metrics
    WHERE tenant_id = p_tenant_id
      AND metric_type = p_metric_type
      AND timestamp >= p_period_start;
    
    RETURN v_usage;
END;
$$ LANGUAGE plpgsql;

-- Create function to check if limit exceeded
CREATE OR REPLACE FUNCTION check_limit_exceeded(
    p_tenant_id UUID,
    p_limit_type VARCHAR,
    p_current_value INTEGER DEFAULT NULL
)
RETURNS TABLE(
    exceeded BOOLEAN,
    current_usage INTEGER,
    limit_value INTEGER,
    percentage_used DECIMAL
) AS $$
DECLARE
    v_limit INTEGER;
    v_current INTEGER;
BEGIN
    -- Get limit from override or defaults
    SELECT COALESCE(o.limit_value, 
        CASE 
            -- Default limits by edition (would be better from config)
            WHEN lc.edition = 'community' THEN
                CASE p_limit_type
                    WHEN 'workflows' THEN 10
                    WHEN 'adapters' THEN 5
                    WHEN 'users' THEN 1
                    WHEN 'api_calls' THEN 10000
                    ELSE 0
                END
            WHEN lc.edition = 'business' THEN
                CASE p_limit_type
                    WHEN 'workflows' THEN 100
                    WHEN 'adapters' THEN 20
                    WHEN 'users' THEN 5
                    WHEN 'api_calls' THEN 1000000
                    ELSE 0
                END
            ELSE 999999 -- Enterprise unlimited
        END
    ) INTO v_limit
    FROM license_cache lc
    LEFT JOIN tenant_limit_overrides o 
        ON o.tenant_id = p_tenant_id 
        AND o.limit_type = p_limit_type
        AND (o.expires_at IS NULL OR o.expires_at > CURRENT_TIMESTAMP)
    WHERE lc.tenant_id = p_tenant_id;
    
    -- Get current usage if not provided
    IF p_current_value IS NULL THEN
        v_current := get_current_usage(p_tenant_id, p_limit_type);
    ELSE
        v_current := p_current_value;
    END IF;
    
    RETURN QUERY SELECT 
        v_current >= v_limit,
        v_current,
        v_limit,
        CASE WHEN v_limit > 0 THEN (v_current::DECIMAL / v_limit * 100) ELSE 0 END;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to summarize usage monthly
CREATE OR REPLACE FUNCTION summarize_monthly_usage()
RETURNS TRIGGER AS $$
BEGIN
    -- Summarize previous month's data on the 1st of each month
    IF date_part('day', CURRENT_TIMESTAMP) = 1 THEN
        INSERT INTO usage_summaries (tenant_id, month, metric_type, total_value, total_count, daily_breakdown)
        SELECT 
            tenant_id,
            date_trunc('month', timestamp - interval '1 month')::DATE,
            metric_type,
            SUM(value),
            SUM(count),
            jsonb_object_agg(
                date_trunc('day', timestamp)::DATE::TEXT,
                json_build_object('value', SUM(value), 'count', SUM(count))
            )
        FROM usage_metrics
        WHERE timestamp >= date_trunc('month', CURRENT_TIMESTAMP - interval '1 month')
          AND timestamp < date_trunc('month', CURRENT_TIMESTAMP)
        GROUP BY tenant_id, metric_type
        ON CONFLICT (tenant_id, month, metric_type) 
        DO UPDATE SET 
            total_value = EXCLUDED.total_value,
            total_count = EXCLUDED.total_count,
            daily_breakdown = EXCLUDED.daily_breakdown;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Add columns to tenants table if not exists
ALTER TABLE tenants ADD COLUMN IF NOT EXISTS stripe_customer_id VARCHAR(255) UNIQUE;
ALTER TABLE tenants ADD COLUMN IF NOT EXISTS stripe_subscription_id VARCHAR(255) UNIQUE;
ALTER TABLE tenants ADD COLUMN IF NOT EXISTS billing_email VARCHAR(255);
ALTER TABLE tenants ADD COLUMN IF NOT EXISTS trial_ends_at TIMESTAMP;