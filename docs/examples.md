# AICtrlNet Community Edition -- Practical Examples

This document walks through five real-world scenarios using the AICtrlNet Community Edition API. Every `curl` command targets `http://localhost:8000` and is copy-paste ready once you replace `YOUR_TOKEN` with a valid Bearer token.

All endpoints live under the `/api/v1` prefix.

---

## Table of Contents

1. [Content Review Pipeline](#1-content-review-pipeline)
2. [Customer Support Triage](#2-customer-support-triage)
3. [Data Quality Validation](#3-data-quality-validation)
4. [Multi-Model AI Chain](#4-multi-model-ai-chain)
5. [Webhook-Triggered Workflow](#5-webhook-triggered-workflow)

---

## 1. Content Review Pipeline

**Scenario:** AI generates a blog post, a human editor reviews and approves or rejects it, and then AI publishes the approved content. This demonstrates the native human-in-the-loop capability that makes AICtrlNet unique.

### Step 1 -- Create the workflow

```bash
curl -X POST http://localhost:8000/api/v1/workflows/ \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Content Review Pipeline",
    "description": "AI generates content, human reviews, AI publishes",
    "category": "content",
    "tags": ["content", "review", "hitl"],
    "status": "active",
    "definition": {
      "nodes": [
        {
          "id": "generate",
          "type": "ai_task",
          "name": "Generate Blog Post",
          "data": {
            "model": "gpt-4",
            "prompt_template": "Write a blog post about {{topic}}",
            "parameters": {"temperature": 0.7, "max_tokens": 2000}
          }
        },
        {
          "id": "review",
          "type": "human_task",
          "name": "Editorial Review",
          "data": {
            "role": "editor",
            "action": "approve",
            "instructions": "Review the generated content for accuracy, tone, and brand alignment.",
            "timeout_hours": 24
          }
        },
        {
          "id": "publish",
          "type": "ai_task",
          "name": "Publish Content",
          "data": {
            "model": "gpt-4",
            "prompt_template": "Format and prepare this content for publication: {{approved_content}}"
          }
        }
      ],
      "edges": [
        {"from": "generate", "to": "review", "label": "content_ready"},
        {"from": "review", "to": "publish", "label": "approved", "condition": "approval_status == approved"}
      ]
    }
  }'
```

**Expected response:**

```json
{
  "id": "a1b2c3d4-...",
  "name": "Content Review Pipeline",
  "description": "AI generates content, human reviews, AI publishes",
  "category": "content",
  "tags": ["content", "review", "hitl"],
  "status": "active",
  "version": 1,
  "definition": { "..." },
  "tenant_id": "default-tenant",
  "created_at": "2025-07-15T10:00:00Z",
  "updated_at": "2025-07-15T10:00:00Z"
}
```

### Step 2 -- Execute the workflow

Use the `id` returned above as `WORKFLOW_ID`:

```bash
curl -X POST http://localhost:8000/api/v1/workflows/WORKFLOW_ID/execute \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input_data": {
      "topic": "The Future of Human-AI Collaboration in Enterprise Software"
    },
    "trigger_source": "manual"
  }'
```

**Expected response:**

```json
{
  "id": "exec-uuid-...",
  "workflow_id": "WORKFLOW_ID",
  "status": "running",
  "input_data": {"topic": "The Future of Human-AI Collaboration in Enterprise Software"},
  "started_at": "2025-07-15T10:01:00Z",
  "created_at": "2025-07-15T10:01:00Z"
}
```

### Step 3 -- Check execution status

```bash
curl http://localhost:8000/api/v1/workflows/WORKFLOW_ID/status \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Expected response (while waiting for human review):**

```json
{
  "workflow_id": "WORKFLOW_ID",
  "instance_id": "instance-uuid-...",
  "status": "running",
  "started_at": "2025-07-15T10:01:00Z",
  "context": {"current_node": "review", "awaiting": "human_approval"},
  "outputs": {}
}
```

### Customization notes

- Change `timeout_hours` in the `review` node to adjust how long the human reviewer has.
- Add more `human_task` nodes for multi-level approval chains (e.g., editor then legal review).
- Replace `gpt-4` with any model you have configured through the adapter system.
- Add a `condition` on the rejection edge to route back to the `generate` node for revision.

---

## 2. Customer Support Triage

**Scenario:** Incoming support tickets are created as tasks. AI categorizes each ticket by topic and urgency, then routes it to the appropriate human agent team based on classification confidence.

### Step 1 -- Create the triage workflow

```bash
curl -X POST http://localhost:8000/api/v1/workflows/ \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Support Ticket Triage",
    "description": "AI classifies tickets and routes to human agents by confidence",
    "category": "support",
    "tags": ["support", "triage", "routing"],
    "status": "active",
    "definition": {
      "nodes": [
        {
          "id": "classify",
          "type": "ai_task",
          "name": "Classify Ticket",
          "data": {
            "model": "gpt-4",
            "prompt_template": "Classify this support ticket. Return JSON with category (billing, technical, account, general), urgency (critical, high, medium, low), and confidence (0-1).\n\nTicket: {{ticket_text}}"
          }
        },
        {
          "id": "high_confidence_route",
          "type": "ai_task",
          "name": "Auto-Route (High Confidence)",
          "data": {
            "model": "gpt-4",
            "prompt_template": "Generate a suggested response for this {{category}} ticket: {{ticket_text}}"
          }
        },
        {
          "id": "human_review",
          "type": "human_task",
          "name": "Manual Classification Review",
          "data": {
            "role": "support_lead",
            "action": "classify_and_route",
            "instructions": "AI confidence was below threshold. Please review the classification and route manually."
          }
        }
      ],
      "edges": [
        {"from": "classify", "to": "high_confidence_route", "label": "auto_route", "condition": "confidence >= 0.85"},
        {"from": "classify", "to": "human_review", "label": "needs_review", "condition": "confidence < 0.85"}
      ]
    }
  }'
```

### Step 2 -- Create a support ticket as a task

```bash
curl -X POST http://localhost:8000/api/v1/tasks/ \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Support Ticket #1042",
    "description": "Customer unable to access billing portal after password reset",
    "metadata": {
      "source": "email",
      "customer_id": "cust-9876",
      "customer_tier": "premium",
      "ticket_text": "I reset my password yesterday but I still cannot log into the billing portal. I keep getting a 403 error. I need to update my payment method before my subscription renews tomorrow."
    }
  }'
```

**Expected response:**

```json
{
  "id": "task-uuid-...",
  "name": "Support Ticket #1042",
  "description": "Customer unable to access billing portal after password reset",
  "status": "pending",
  "metadata": {
    "source": "email",
    "customer_id": "cust-9876",
    "customer_tier": "premium",
    "ticket_text": "..."
  },
  "tenant_id": "default-tenant",
  "created_at": "2025-07-15T11:00:00Z",
  "updated_at": "2025-07-15T11:00:00Z"
}
```

### Step 3 -- Execute the triage workflow with the ticket data

```bash
curl -X POST http://localhost:8000/api/v1/workflows/WORKFLOW_ID/execute \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input_data": {
      "ticket_text": "I reset my password yesterday but I still cannot log into the billing portal. I keep getting a 403 error. I need to update my payment method before my subscription renews tomorrow.",
      "customer_tier": "premium",
      "task_id": "task-uuid-..."
    }
  }'
```

### Step 4 -- Update the task status after triage

```bash
curl -X PUT http://localhost:8000/api/v1/tasks/TASK_ID \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "status": "in_progress",
    "metadata": {
      "category": "billing",
      "urgency": "high",
      "confidence": 0.92,
      "assigned_team": "billing_support",
      "ai_suggested_response": "...",
      "routed_by": "auto"
    }
  }'
```

**Expected response:**

```json
{
  "id": "task-uuid-...",
  "name": "Support Ticket #1042",
  "status": "in_progress",
  "metadata": {
    "category": "billing",
    "urgency": "high",
    "confidence": 0.92,
    "assigned_team": "billing_support"
  },
  "updated_at": "2025-07-15T11:02:00Z"
}
```

### Customization notes

- Adjust the `confidence` threshold in the edge condition (0.85) to match your quality needs.
- Add additional classification categories to the AI prompt for your domain.
- Use task `status` values (`pending`, `in_progress`, `completed`, `failed`, `cancelled`) to track ticket lifecycle.
- Chain multiple AI models by adding more `ai_task` nodes (e.g., sentiment analysis before classification).

---

## 3. Data Quality Validation

**Scenario:** Before processing a data import, run it through quality checks. AI validates entries against rules, flags anomalies for human review, and produces a quality score using the ISO 25012-compliant data quality framework.

### Step 1 -- Check available quality dimensions

```bash
curl http://localhost:8000/api/v1/quality/dimensions \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Expected response:**

```json
[
  {
    "name": "accuracy",
    "category": "inherent",
    "description": "Assess data accuracy",
    "available_in_edition": true,
    "measurement_method": "Automated accuracy assessment"
  },
  {
    "name": "completeness",
    "category": "inherent",
    "description": "Assess data completeness",
    "available_in_edition": true,
    "measurement_method": "Automated completeness assessment"
  },
  {
    "name": "consistency",
    "category": "inherent",
    "description": "Assess data consistency",
    "available_in_edition": false,
    "measurement_method": "Automated consistency assessment"
  }
]
```

Note: Community Edition includes `accuracy` and `completeness` dimensions. Business adds 8 more (10 total), Enterprise adds 5 more (15 total).

### Step 2 -- Create a quality rule

```bash
curl -X POST http://localhost:8000/api/v1/quality/rules \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Email Format Validation",
    "description": "Validates that email fields match standard email format",
    "dimension": "accuracy",
    "rule_type": "regex",
    "rule_definition": {
      "field": "email",
      "pattern": "^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$"
    },
    "severity": "error"
  }'
```

**Expected response:**

```json
{
  "id": "rule-uuid-...",
  "name": "Email Format Validation",
  "dimension": "accuracy",
  "rule_type": "regex",
  "rule_definition": {
    "field": "email",
    "pattern": "^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$"
  },
  "severity": "error",
  "is_active": true,
  "is_system": false,
  "created_at": "2025-07-15T12:00:00Z"
}
```

### Step 3 -- Create a range rule for numeric validation

```bash
curl -X POST http://localhost:8000/api/v1/quality/rules \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Age Range Check",
    "description": "Flags age values outside reasonable range",
    "dimension": "accuracy",
    "rule_type": "range",
    "rule_definition": {
      "field": "age",
      "min": 0,
      "max": 150
    },
    "severity": "warning"
  }'
```

### Step 4 -- Run a quality assessment

```bash
curl -X POST http://localhost:8000/api/v1/quality/assess \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"name": "Alice Johnson", "email": "alice@example.com", "age": 34, "department": "Engineering"},
      {"name": "Bob", "email": "not-an-email", "age": 28, "department": ""},
      {"name": "", "email": "carol@example.com", "age": -5, "department": "Sales"},
      {"name": "Dave Wilson", "email": "dave@example.com", "age": 45, "department": "Marketing"}
    ],
    "dimensions": ["accuracy", "completeness"],
    "include_suggestions": true,
    "include_profile": true
  }'
```

**Expected response:**

```json
{
  "id": "assessment-uuid-...",
  "overall_score": 0.72,
  "dimension_scores": {
    "accuracy": 0.68,
    "completeness": 0.75
  },
  "issues_found": [
    {"field": "email", "row": 1, "issue": "Invalid email format", "severity": "error"},
    {"field": "name", "row": 2, "issue": "Empty required field", "severity": "error"},
    {"field": "age", "row": 2, "issue": "Value -5 outside valid range [0, 150]", "severity": "warning"},
    {"field": "department", "row": 1, "issue": "Empty field", "severity": "warning"}
  ],
  "suggestions": [
    {"type": "fix", "description": "Fix email format in row 2"},
    {"type": "fix", "description": "Provide name in row 3"},
    {"type": "review", "description": "Verify age value in row 3"}
  ],
  "data_profile": {
    "row_count": 4,
    "column_count": 4,
    "null_percentage": 6.25,
    "unique_percentage": 87.5
  },
  "assessment_time": "2025-07-15T12:05:00Z",
  "dimensions_assessed": ["accuracy", "completeness"],
  "edition": "community"
}
```

### Step 5 -- Create a workflow that includes quality gating

```bash
curl -X POST http://localhost:8000/api/v1/workflows/ \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Data Import with Quality Gate",
    "description": "Validates imported data quality before processing, routes failures to human review",
    "category": "data_processing",
    "tags": ["data-quality", "import", "validation"],
    "status": "active",
    "definition": {
      "nodes": [
        {
          "id": "validate",
          "type": "ai_task",
          "name": "Quality Assessment",
          "data": {
            "action": "quality_assess",
            "dimensions": ["accuracy", "completeness"],
            "threshold": 0.8
          }
        },
        {
          "id": "process",
          "type": "ai_task",
          "name": "Process Import",
          "data": {
            "action": "process_data",
            "target": "main_database"
          }
        },
        {
          "id": "human_fix",
          "type": "human_task",
          "name": "Fix Data Issues",
          "data": {
            "role": "data_steward",
            "action": "review_and_fix",
            "instructions": "Data quality score below threshold. Please review and fix flagged issues."
          }
        }
      ],
      "edges": [
        {"from": "validate", "to": "process", "label": "quality_pass", "condition": "overall_score >= 0.8"},
        {"from": "validate", "to": "human_fix", "label": "quality_fail", "condition": "overall_score < 0.8"},
        {"from": "human_fix", "to": "validate", "label": "revalidate"}
      ]
    }
  }'
```

### Step 6 -- View the quality dashboard

```bash
curl "http://localhost:8000/api/v1/quality/dashboard?time_range=30" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Expected response:**

```json
{
  "total_assessments": 15,
  "average_score": 0.78,
  "assessments_by_dimension": {"accuracy": 15, "completeness": 12},
  "scores_by_dimension": {"accuracy": 0.82, "completeness": 0.74},
  "top_issues": [
    {"issue": "Missing required fields", "count": 23, "severity": "error"},
    {"issue": "Format validation failures", "count": 18, "severity": "warning"}
  ],
  "quality_trend": [
    {"date": "2025-07-01", "score": 0.71},
    {"date": "2025-07-08", "score": 0.75},
    {"date": "2025-07-15", "score": 0.82}
  ]
}
```

### Customization notes

- Community Edition supports `accuracy` and `completeness` dimensions. Upgrade to Business for consistency, credibility, currentness, and more.
- The quality threshold (0.8) in the workflow can be adjusted per use case.
- Combine `regex`, `range`, `validation`, and `pattern` rule types for comprehensive checks.
- Use `include_profile: true` to get statistical profiling of your data alongside quality scores.

---

## 4. Multi-Model AI Chain

**Scenario:** Register multiple AI model adapters (e.g., OpenAI for analysis, Anthropic Claude for summarization), then create a workflow that chains them together -- one model analyzes raw data and another produces a human-readable summary.

### Step 1 -- Register the OpenAI adapter configuration

```bash
curl -X POST http://localhost:8000/api/v1/adapter-config/config \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "adapter_type": "openai",
    "name": "OpenAI Analysis Model",
    "display_name": "GPT-4 for Data Analysis",
    "credentials": {
      "api_key": "sk-your-openai-api-key"
    },
    "settings": {
      "model": "gpt-4",
      "temperature": 0.3,
      "max_tokens": 4000,
      "use_case": "analysis"
    },
    "enabled": true
  }'
```

**Expected response:**

```json
{
  "id": "config-uuid-1-...",
  "adapter_type": "openai",
  "name": "OpenAI Analysis Model",
  "display_name": "GPT-4 for Data Analysis",
  "credentials": null,
  "settings": {"model": "gpt-4", "temperature": 0.3, "max_tokens": 4000},
  "enabled": true,
  "user_id": "user-uuid-...",
  "test_status": null,
  "created_at": "2025-07-15T13:00:00Z",
  "updated_at": "2025-07-15T13:00:00Z"
}
```

Note: Credentials are stored encrypted and are never returned in API responses.

### Step 2 -- Register the Anthropic adapter configuration

```bash
curl -X POST http://localhost:8000/api/v1/adapter-config/config \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "adapter_type": "anthropic",
    "name": "Claude Summary Model",
    "display_name": "Claude for Summarization",
    "credentials": {
      "api_key": "sk-ant-your-anthropic-key"
    },
    "settings": {
      "model": "claude-sonnet-4-20250514",
      "temperature": 0.5,
      "max_tokens": 2000,
      "use_case": "summarization"
    },
    "enabled": true
  }'
```

### Step 3 -- Test adapter configurations

```bash
curl -X POST http://localhost:8000/api/v1/adapter-config/config/CONFIG_UUID_1/test \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"timeout": 30}'
```

**Expected response:**

```json
{
  "status": "success",
  "message": "Adapter connection verified successfully",
  "details": {"model": "gpt-4", "response_time_ms": 245},
  "tested_at": "2025-07-15T13:05:00Z",
  "duration_ms": 312
}
```

### Step 4 -- Activate the adapters

```bash
curl -X POST http://localhost:8000/api/v1/adapter-config/config/CONFIG_UUID_1/activate \
  -H "Authorization: Bearer YOUR_TOKEN"

curl -X POST http://localhost:8000/api/v1/adapter-config/config/CONFIG_UUID_2/activate \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Step 5 -- Create a multi-model chain workflow

```bash
curl -X POST http://localhost:8000/api/v1/workflows/ \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Multi-Model Analysis Pipeline",
    "description": "OpenAI analyzes data, Claude summarizes findings for stakeholders",
    "category": "ai_ml",
    "tags": ["multi-model", "analysis", "summarization"],
    "status": "active",
    "definition": {
      "nodes": [
        {
          "id": "ingest",
          "type": "data_input",
          "name": "Receive Raw Data",
          "data": {
            "accepts": ["json", "csv", "text"],
            "max_size_mb": 10
          }
        },
        {
          "id": "analyze",
          "type": "ai_task",
          "name": "Deep Analysis (GPT-4)",
          "data": {
            "adapter": "openai",
            "model": "gpt-4",
            "prompt_template": "Analyze the following data thoroughly. Identify key patterns, anomalies, trends, and risks. Structure your analysis with clear sections.\n\nData:\n{{raw_data}}",
            "parameters": {"temperature": 0.3, "max_tokens": 4000}
          }
        },
        {
          "id": "summarize",
          "type": "ai_task",
          "name": "Executive Summary (Claude)",
          "data": {
            "adapter": "anthropic",
            "model": "claude-sonnet-4-20250514",
            "prompt_template": "Create a concise executive summary from this analysis. Use bullet points, highlight the 3 most important findings, and include recommended actions.\n\nAnalysis:\n{{analysis_output}}",
            "parameters": {"temperature": 0.5, "max_tokens": 2000}
          }
        },
        {
          "id": "review",
          "type": "human_task",
          "name": "Stakeholder Review",
          "data": {
            "role": "analyst",
            "action": "approve",
            "instructions": "Review the AI-generated analysis and summary. Approve for distribution or send back for revision."
          }
        }
      ],
      "edges": [
        {"from": "ingest", "to": "analyze", "label": "data_ready"},
        {"from": "analyze", "to": "summarize", "label": "analysis_complete"},
        {"from": "summarize", "to": "review", "label": "summary_ready"}
      ]
    }
  }'
```

### Step 6 -- List your adapter configurations

```bash
curl "http://localhost:8000/api/v1/adapter-config/config?enabled_only=true" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Expected response:**

```json
{
  "configs": [
    {
      "id": "config-uuid-1-...",
      "adapter_type": "openai",
      "name": "OpenAI Analysis Model",
      "display_name": "GPT-4 for Data Analysis",
      "enabled": true,
      "test_status": "success"
    },
    {
      "id": "config-uuid-2-...",
      "adapter_type": "anthropic",
      "name": "Claude Summary Model",
      "display_name": "Claude for Summarization",
      "enabled": true,
      "test_status": "success"
    }
  ],
  "total": 2
}
```

### Step 7 -- Browse the adapter registry for more adapters

```bash
curl "http://localhost:8000/api/v1/adapters/registry/list?category=ai" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Customization notes

- AICtrlNet supports 27+ adapters including OpenAI, Claude, Slack, Teams, Stripe, and more. Browse the registry to discover what is available.
- Adjust `temperature` for each model -- lower (0.1-0.3) for deterministic analysis, higher (0.5-0.8) for creative summarization.
- Add more models to the chain by inserting additional `ai_task` nodes (e.g., a fact-checking step between analysis and summary).
- Use the bulk test endpoint (`POST /api/v1/adapter-config/config/test-bulk`) to verify all adapters at once.

---

## 5. Webhook-Triggered Workflow

**Scenario:** An external system (e.g., a CI/CD pipeline, a monitoring tool, or a third-party SaaS) sends an event via webhook to AICtrlNet. The webhook triggers a workflow that processes the event, and notifies your team via a configured webhook on completion.

### Step 1 -- Create a webhook to receive external events

```bash
curl -X POST http://localhost:8000/api/v1/webhooks \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "CI/CD Pipeline Events",
    "description": "Receives deployment events from CI/CD system",
    "url": "http://localhost:8000/api/v1/webhooks",
    "events": ["workflow.started", "workflow.completed", "workflow.failed"],
    "secret": "my-webhook-secret-key",
    "custom_headers": {
      "X-Source": "cicd-pipeline"
    },
    "max_retries": 5,
    "retry_delay_seconds": 120,
    "timeout_seconds": 30
  }'
```

**Expected response:**

```json
{
  "id": "webhook-uuid-...",
  "name": "CI/CD Pipeline Events",
  "description": "Receives deployment events from CI/CD system",
  "url": "http://localhost:8000/api/v1/webhooks",
  "events": ["workflow.started", "workflow.completed", "workflow.failed"],
  "secret": "my-webhook-secret-key",
  "is_active": true,
  "consecutive_failures": 0,
  "total_deliveries": 0,
  "total_failures": 0,
  "created_at": "2025-07-15T14:00:00Z",
  "updated_at": "2025-07-15T14:00:00Z"
}
```

Note: The `secret` field is only returned in the create response. Store it securely -- it is used to verify webhook signatures via HMAC-SHA256 in the `X-Webhook-Signature` header.

### Step 2 -- Create a notification webhook (outbound)

This webhook sends notifications to your team's endpoint when workflow events occur:

```bash
curl -X POST http://localhost:8000/api/v1/webhooks \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Team Notifications",
    "description": "Notifies team Slack channel on workflow completion",
    "url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
    "events": ["workflow.completed", "workflow.failed", "task.failed"],
    "secret": "notification-secret-456",
    "max_retries": 3,
    "retry_delay_seconds": 60,
    "timeout_seconds": 15
  }'
```

### Step 3 -- Create a workflow that processes external events

```bash
curl -X POST http://localhost:8000/api/v1/workflows/ \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Deployment Review Pipeline",
    "description": "Processes CI/CD deployment events, runs validation, and notifies team",
    "category": "integration",
    "tags": ["cicd", "deployment", "webhook"],
    "status": "active",
    "definition": {
      "nodes": [
        {
          "id": "receive_event",
          "type": "data_input",
          "name": "Receive Deployment Event",
          "data": {
            "source": "webhook",
            "event_types": ["deployment.completed"]
          }
        },
        {
          "id": "validate_deployment",
          "type": "ai_task",
          "name": "Validate Deployment",
          "data": {
            "model": "gpt-4",
            "prompt_template": "Analyze this deployment event and check for potential issues:\n\nEnvironment: {{environment}}\nVersion: {{version}}\nChanges: {{changes}}\n\nProvide: risk_level (low/medium/high), issues found, and recommendation."
          }
        },
        {
          "id": "auto_approve",
          "type": "ai_task",
          "name": "Auto-Approve (Low Risk)",
          "data": {
            "action": "mark_approved",
            "notify": true
          }
        },
        {
          "id": "manual_review",
          "type": "human_task",
          "name": "Manual Deployment Review",
          "data": {
            "role": "devops_lead",
            "action": "approve",
            "instructions": "AI flagged this deployment as medium/high risk. Review the analysis and approve or rollback.",
            "timeout_hours": 2
          }
        }
      ],
      "edges": [
        {"from": "receive_event", "to": "validate_deployment", "label": "event_received"},
        {"from": "validate_deployment", "to": "auto_approve", "label": "low_risk", "condition": "risk_level == low"},
        {"from": "validate_deployment", "to": "manual_review", "label": "needs_review", "condition": "risk_level != low"}
      ]
    }
  }'
```

### Step 4 -- Create a trigger for the workflow

```bash
curl -X POST http://localhost:8000/api/v1/workflows/WORKFLOW_ID/triggers \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "trigger_type": "webhook",
    "config": {
      "webhook_path": "/deployments",
      "event_filter": "deployment.*",
      "authentication": "hmac-sha256"
    },
    "is_active": true
  }'
```

**Expected response:**

```json
{
  "id": "trigger-uuid-...",
  "workflow_id": "WORKFLOW_ID",
  "trigger_type": "webhook",
  "config": {
    "webhook_path": "/deployments",
    "event_filter": "deployment.*",
    "authentication": "hmac-sha256"
  },
  "is_active": true
}
```

### Step 5 -- Test the webhook

```bash
curl -X POST http://localhost:8000/api/v1/webhooks/WEBHOOK_ID/test \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "workflow.completed",
    "payload": {
      "workflow_id": "WORKFLOW_ID",
      "status": "completed",
      "environment": "staging",
      "version": "2.1.0",
      "timestamp": "2025-07-15T14:30:00Z"
    }
  }'
```

**Expected response:**

```json
{
  "success": true,
  "status_code": 200,
  "response_time_ms": 142,
  "error_message": null,
  "response_body": "{\"ok\": true}"
}
```

### Step 6 -- View webhook delivery history

```bash
curl "http://localhost:8000/api/v1/webhooks/WEBHOOK_ID/deliveries?limit=10" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Expected response:**

```json
{
  "deliveries": [
    {
      "id": "delivery-uuid-...",
      "webhook_id": "WEBHOOK_ID",
      "event_type": "workflow.completed",
      "attempt_number": 1,
      "status_code": 200,
      "response_time_ms": 142,
      "is_success": true,
      "created_at": "2025-07-15T14:30:00Z",
      "delivered_at": "2025-07-15T14:30:00Z"
    }
  ],
  "total": 1
}
```

### Step 7 -- Manage webhook lifecycle

Disable a webhook temporarily:

```bash
curl -X POST http://localhost:8000/api/v1/webhooks/WEBHOOK_ID/disable \
  -H "Authorization: Bearer YOUR_TOKEN"
```

Re-enable it:

```bash
curl -X POST http://localhost:8000/api/v1/webhooks/WEBHOOK_ID/enable \
  -H "Authorization: Bearer YOUR_TOKEN"
```

List all webhooks with filters:

```bash
curl "http://localhost:8000/api/v1/webhooks?is_active=true&event_type=workflow.*" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Customization notes

- Supported event patterns: `task.*`, `workflow.*`, `agent.*`, `system.*`, and `*` (all events). Individual events like `task.created` or `workflow.failed` are also valid.
- The `secret` field enables HMAC-SHA256 signature verification. The signature is sent in the `X-Webhook-Signature` header.
- Set `max_retries` (0-10) and `retry_delay_seconds` (10-3600) to control retry behavior for failed deliveries.
- Combine with workflow schedules for time-based triggers in addition to event-based webhooks.

---

## Authentication Reference

All examples use `Bearer YOUR_TOKEN` for authentication. To obtain a token:

```bash
# Login to get a token
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "dev@aictrlnet.com",
    "password": "testpass123"
  }'
```

For local development, you can also use the development token:

```bash
Authorization: Bearer dev-token-for-testing
```

---

## Next Steps

- Explore the full API by visiting `http://localhost:8000/docs` (interactive Swagger UI).
- Check the [Adapter Guide](adapters.md) for details on configuring all 27+ supported adapters.
- Read the [Architecture Overview](architecture.md) to understand how the control plane, adapters, and execution engine interact.
- For ML-enhanced features (AI governance, risk assessment, bias detection), see the [Business Edition](https://aictrlnet.com).
