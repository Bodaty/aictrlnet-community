# AICtrlNet Community Edition -- Guides

Practical guides for working with the AICtrlNet Community Edition API. All examples assume the API is running on `http://localhost:8000` and that you have a valid authentication token. For development and testing, the token `dev-token-for-testing` is available.

Replace `YOUR_TOKEN` in the examples below with your actual token.

---

## Table of Contents

1. [Working with Adapters](#working-with-adapters)
2. [Human-in-the-Loop Workflows](#human-in-the-loop-workflows)
3. [MCP Integration](#mcp-integration)
4. [AI Governance](#ai-governance)
5. [Authentication and API Keys](#authentication-and-api-keys)
6. [Deployment](#deployment)

---

## Working with Adapters

Adapters are the integration layer that connects AICtrlNet to external services. Each adapter wraps a specific provider or communication channel and exposes a uniform interface so that workflows can route tasks without being tightly coupled to any single vendor.

### Adapter Categories

AICtrlNet Community Edition ships with adapters organized into the following categories:

| Category | Adapters | Purpose |
|----------|----------|---------|
| **AI** | OpenAI, Claude (Anthropic), Ollama, HuggingFace, MCP Client, ML Service | LLM inference and AI model access |
| **Communication** | Slack, Discord, Email, Webhook | Notifications, alerts, and messaging |
| **Human** | Upwork, Fiverr, TaskRabbit | Routing tasks to human workers |
| **Database** | Database adapters | Persistent data access |
| **Payment** | Payment adapters (e.g., Stripe) | Payment processing |

### Listing Available Adapters

Use the adapter registry endpoint to discover adapters available in your edition:

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8000/api/v1/adapters/registry/list
```

You can filter by category or search by name:

```bash
# Filter by category
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "http://localhost:8000/api/v1/adapters/registry/list?category=ai"

# Search by name
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "http://localhost:8000/api/v1/adapters/registry/list?search=openai"
```

The response includes an `available` flag per adapter. Some adapters require a Business or Enterprise edition license and will appear with `available: false` in the Community Edition.

### Configuring an Adapter

Before using an adapter in a workflow, create a configuration that stores your credentials and settings. Credentials are stored encrypted at rest.

```bash
curl -X POST http://localhost:8000/api/v1/adapters/config \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "adapter_type": "openai",
    "name": "My OpenAI Config",
    "credentials": {
      "api_key": "sk-..."
    },
    "settings": {
      "default_model": "gpt-4",
      "max_tokens": 4096
    },
    "enabled": true
  }'
```

For self-hosted models via Ollama:

```bash
curl -X POST http://localhost:8000/api/v1/adapters/config \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "adapter_type": "ollama",
    "name": "Local Ollama",
    "settings": {
      "base_url": "http://localhost:11434",
      "default_model": "llama3"
    },
    "enabled": true
  }'
```

### Listing Your Configurations

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8000/api/v1/adapters/config
```

Filter to only enabled configurations:

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "http://localhost:8000/api/v1/adapters/config?enabled_only=true"
```

### Testing an Adapter Configuration

Verify that your credentials work before using the adapter in production workflows:

```bash
curl -X POST http://localhost:8000/api/v1/adapters/config/CONFIG_ID/test \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "test_message": "Hello, can you respond?"
  }'
```

### Checking Adapter Availability

Check whether a specific set of adapters is available in your edition:

```bash
curl -X POST http://localhost:8000/api/v1/adapters/check-availability \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "adapter_ids": ["openai", "claude", "slack"]
  }'
```

---

## Human-in-the-Loop Workflows

AICtrlNet treats humans and AI as equal participants in workflows. Rather than bolting human review onto an AI pipeline as an afterthought, you define explicit human nodes that receive tasks, make decisions, and feed results back into the workflow.

### Creating a Workflow with Human Approval

The following example creates a three-step content review workflow: AI generates content, a human reviewer approves or rejects it, and then the system publishes the approved content.

```bash
curl -X POST http://localhost:8000/api/v1/workflows/ \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Content Review Pipeline",
    "description": "AI drafts content, human reviews, then publish",
    "definition": {
      "nodes": [
        {
          "id": "draft",
          "type": "ai",
          "data": {
            "model": "gpt-4",
            "task": "generate_content",
            "prompt": "Write a blog post about AI governance"
          },
          "position": {"x": 100, "y": 100}
        },
        {
          "id": "review",
          "type": "human",
          "data": {
            "role": "reviewer",
            "action": "approve",
            "instructions": "Review the AI-generated content for accuracy and tone"
          },
          "position": {"x": 400, "y": 100}
        },
        {
          "id": "publish",
          "type": "ai",
          "data": {
            "task": "publish_content",
            "channel": "blog"
          },
          "position": {"x": 700, "y": 100}
        }
      ],
      "edges": [
        {"source": "draft", "target": "review"},
        {"source": "review", "target": "publish"}
      ]
    }
  }'
```

### Confidence-Based Routing

You can design workflows that route to human reviewers only when the AI's confidence falls below a threshold. High-confidence results proceed automatically, while uncertain results are escalated for human judgment.

```bash
curl -X POST http://localhost:8000/api/v1/workflows/ \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Confidence-Based Review",
    "description": "Route to human when AI confidence is low",
    "definition": {
      "nodes": [
        {
          "id": "classify",
          "type": "ai",
          "data": {
            "model": "gpt-4",
            "task": "classify_ticket",
            "output_confidence": true
          },
          "position": {"x": 100, "y": 200}
        },
        {
          "id": "confidence_check",
          "type": "condition",
          "data": {
            "condition": "confidence >= 0.85",
            "true_target": "auto_route",
            "false_target": "human_review"
          },
          "position": {"x": 400, "y": 200}
        },
        {
          "id": "auto_route",
          "type": "ai",
          "data": {"task": "route_ticket"},
          "position": {"x": 700, "y": 100}
        },
        {
          "id": "human_review",
          "type": "human",
          "data": {
            "role": "support_lead",
            "action": "review_and_route",
            "instructions": "AI was uncertain. Please review and route this ticket."
          },
          "position": {"x": 700, "y": 300}
        }
      ],
      "edges": [
        {"source": "classify", "target": "confidence_check"},
        {"source": "confidence_check", "target": "auto_route", "label": "high_confidence"},
        {"source": "confidence_check", "target": "human_review", "label": "low_confidence"}
      ]
    }
  }'
```

### Escalation Paths

Workflows can include escalation nodes that transfer a task up the chain when it requires higher authority or specialized expertise.

```bash
curl -X POST http://localhost:8000/api/v1/workflows/ \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Tiered Approval Workflow",
    "description": "Multi-level approval with escalation",
    "definition": {
      "nodes": [
        {
          "id": "initial_review",
          "type": "human",
          "data": {
            "role": "analyst",
            "action": "review",
            "escalation": {
              "enabled": true,
              "timeout_minutes": 60,
              "escalate_to": "manager_review"
            }
          },
          "position": {"x": 100, "y": 200}
        },
        {
          "id": "manager_review",
          "type": "human",
          "data": {
            "role": "manager",
            "action": "approve",
            "escalation": {
              "enabled": true,
              "timeout_minutes": 120,
              "escalate_to": "director_review"
            }
          },
          "position": {"x": 400, "y": 200}
        },
        {
          "id": "director_review",
          "type": "human",
          "data": {
            "role": "director",
            "action": "final_approve"
          },
          "position": {"x": 700, "y": 200}
        }
      ],
      "edges": [
        {"source": "initial_review", "target": "manager_review", "label": "escalate"},
        {"source": "manager_review", "target": "director_review", "label": "escalate"}
      ]
    }
  }'
```

### Executing a Workflow

Once created, trigger execution with:

```bash
curl -X POST http://localhost:8000/api/v1/workflows/WORKFLOW_ID/execute \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input_data": {
      "content_topic": "AI in healthcare"
    }
  }'
```

### Checking Workflow Status

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8000/api/v1/workflows/WORKFLOW_ID/status
```

When a workflow reaches a human node, the status will indicate it is waiting for human input. Use the approval endpoints to submit the decision.

### Pausing and Resuming Workflows

```bash
# Pause a running workflow
curl -X POST http://localhost:8000/api/v1/workflows/WORKFLOW_ID/pause \
  -H "Authorization: Bearer YOUR_TOKEN"

# Resume a paused workflow
curl -X POST http://localhost:8000/api/v1/workflows/WORKFLOW_ID/resume \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Validating a Workflow Before Saving

Check a workflow definition for errors before committing it:

```bash
curl -X POST http://localhost:8000/api/v1/workflows/validate \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "nodes": [
      {"id": "step1", "type": "ai", "data": {"model": "gpt-4"}},
      {"id": "step2", "type": "human", "data": {"role": "reviewer"}}
    ],
    "edges": [
      {"source": "step1", "target": "step2"}
    ]
  }'
```

The response includes validation errors (if any), resource requirement estimates, and estimated execution time.

---

## MCP Integration

### What is MCP?

The Model Context Protocol (MCP) is an open standard that defines how AI applications communicate with external tools and data sources. Rather than building custom integrations for every service, MCP provides a uniform protocol based on JSON-RPC 2.0 that any tool provider can implement.

AICtrlNet has native MCP support, meaning you can register any MCP-compatible server and immediately use its tools and resources inside your workflows.

Specification reference: https://modelcontextprotocol.io/specification/2025-03-26

### Supported Transport Types

AICtrlNet supports two MCP transport types:

- **stdio** (recommended): Spawns the MCP server as a subprocess and communicates via stdin/stdout. This is the standard MCP transport and works with tools like `npx`, `python`, or any executable.
- **http_sse**: Connects to a remote MCP server via HTTP with Server-Sent Events.

### Registering an MCP Server

Register a stdio-based MCP server (the most common type):

```bash
curl -X POST http://localhost:8000/api/v1/mcp/servers \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "filesystem-server",
    "transport_type": "stdio",
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp/workspace"],
    "service_type": "filesystem"
  }'
```

Register an HTTP/SSE-based MCP server:

```bash
curl -X POST http://localhost:8000/api/v1/mcp/servers \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "remote-tools",
    "transport_type": "http_sse",
    "url": "https://mcp.example.com/sse",
    "api_key": "your-api-key",
    "service_type": "custom"
  }'
```

During registration, AICtrlNet will attempt to connect to stdio servers automatically and discover their capabilities (tools, resources, prompts).

### Listing Registered MCP Servers

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8000/api/v1/mcp/servers
```

Filter by transport type or status:

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "http://localhost:8000/api/v1/mcp/servers?transport_type=stdio&server_status=connected"
```

### Connecting to an MCP Server

If a server was not connected during registration, connect explicitly:

```bash
curl -X POST http://localhost:8000/api/v1/mcp/servers/SERVER_ID/connect \
  -H "Authorization: Bearer YOUR_TOKEN"
```

The response includes the discovered tools, resources, and server capabilities.

### Discovering Tools from a Connected Server

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8000/api/v1/mcp/servers/SERVER_ID/tools
```

Each tool includes:
- `name`: The tool identifier
- `description`: What the tool does
- `input_schema`: JSON Schema describing the expected arguments
- `output_schema`: (when available) JSON Schema describing the return value
- `annotations`: Hints about side effects (read-only, destructive, idempotent)

### Calling an MCP Tool

```bash
curl -X POST http://localhost:8000/api/v1/mcp/servers/SERVER_ID/tools/read_file/call \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "path": "/tmp/workspace/example.txt"
  }'
```

The response contains the tool result as a `content` array, which may include text, images, or other content types per the MCP specification.

### Listing Server Resources

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8000/api/v1/mcp/servers/SERVER_ID/resources
```

### Health Checks

Check whether an MCP server is responsive:

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8000/api/v1/mcp/servers/SERVER_ID/health
```

### Viewing Active Connections

See all currently connected MCP servers:

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8000/api/v1/mcp/connections
```

### Async Tasks (Long-Running Operations)

For MCP operations that take longer than a single request/response cycle, use async tasks:

```bash
# Create an async task
curl -X POST http://localhost:8000/api/v1/mcp/tasks \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "server_id": "SERVER_ID",
    "tool_id": "long_running_tool",
    "method": "tools/call",
    "request_params": {"input": "large dataset"},
    "timeout_seconds": 600
  }'

# Poll for status using the task token
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8000/api/v1/mcp/tasks/TASK_TOKEN

# Cancel a running task
curl -X POST http://localhost:8000/api/v1/mcp/tasks/TASK_TOKEN/cancel \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Disconnecting from an MCP Server

```bash
curl -X POST http://localhost:8000/api/v1/mcp/servers/SERVER_ID/disconnect \
  -H "Authorization: Bearer YOUR_TOKEN"
```

---

## AI Governance

AICtrlNet builds governance into the platform rather than treating it as an afterthought. The Community Edition includes foundational governance capabilities; Business and Enterprise editions add ML-powered risk assessment, bias detection, and compliance dashboards.

### 5-Layer AI Workflow Security Gateway

Every workflow in AICtrlNet passes through a security gateway that evaluates prompts and actions at multiple layers:

1. **Input Validation** -- Prompt length limits, forbidden pattern detection (credential access attempts, code injection)
2. **Rate Limiting** -- Per-user request limits (Community: 5/min, 30/hour, 100/day)
3. **Content Filtering** -- Detection of prompt injection and data exfiltration patterns
4. **Execution Sandboxing** -- Workflows execute in isolated contexts
5. **Output Auditing** -- All actions and results are logged for audit trail

### Audit Trails

All API operations in AICtrlNet are logged. Audit logs capture:
- Who performed the action (user ID, email)
- What was done (endpoint, method, parameters)
- When it happened (timestamp)
- The result (success/failure, status code)

This logging is automatic and requires no additional configuration. Logs are stored in the database and can be queried for compliance reporting.

### Workflow Validation

Before a workflow is executed, the validation service checks:
- Node types are valid for the current edition
- Required connections between nodes exist
- No orphaned nodes
- Resource requirements are within limits

```bash
curl -X POST http://localhost:8000/api/v1/workflows/validate \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "nodes": [
      {"id": "n1", "type": "ai", "data": {"model": "gpt-4"}}
    ],
    "edges": []
  }'
```

The response includes:
- `is_valid`: Whether the workflow passes validation
- `errors`: List of specific issues found
- `resource_requirements`: Estimated compute and memory needs
- `estimated_execution_time`: Rough time estimate in seconds

### Usage Tracking

AICtrlNet tracks resource usage per tenant, including workflow executions, API calls, and AI model invocations. This data supports governance reporting and capacity planning.

### Compliance Frameworks

The Community Edition provides the foundational audit logging and security controls. For full compliance framework support, the editions break down as follows:

| Capability | Community | Business | Enterprise |
|-----------|-----------|----------|------------|
| Basic audit logs | Yes | Yes | Yes |
| Security gateway | Yes | Yes | Yes |
| Usage tracking | Yes | Yes | Yes |
| ML-powered risk assessment | -- | Yes | Yes |
| Bias detection and monitoring | -- | Yes | Yes |
| Policy templates | -- | Yes | Yes |
| SOC2 compliance reports | -- | Yes | Yes |
| HIPAA compliance tools | -- | -- | Yes |
| GDPR compliance tools | -- | -- | Yes |
| Geographic data routing | -- | -- | Yes |

To upgrade, see the [Edition Migration Guide](EDITION_MIGRATION.md).

---

## Authentication and API Keys

AICtrlNet uses JWT-based authentication for all API access. There are two ways to authenticate: user login (email/password returning a JWT) and API keys (for programmatic access).

### Registering a User

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "you@example.com",
    "password": "your-secure-password",
    "username": "yourname",
    "full_name": "Your Full Name"
  }'
```

### Logging In

The login endpoint uses OAuth2 password form encoding:

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=you@example.com&password=your-secure-password"
```

On success, the response contains an `access_token`. Use it as a Bearer token in subsequent requests:

```bash
curl -H "Authorization: Bearer ACCESS_TOKEN_HERE" \
  http://localhost:8000/api/v1/auth/me
```

If MFA is enabled for your account, the login response will include `mfa_required: true` and a `session_token`. Submit the MFA code to complete authentication:

```bash
curl -X POST http://localhost:8000/api/v1/auth/mfa/verify \
  -H "Content-Type: application/json" \
  -d '{
    "session_token": "SESSION_TOKEN_FROM_LOGIN",
    "code": "123456"
  }'
```

### Refreshing Tokens

Access tokens expire after a configurable period. Use the refresh token to obtain a new access token without re-entering credentials:

```bash
curl -X POST http://localhost:8000/api/v1/auth/token/refresh \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "YOUR_REFRESH_TOKEN"
  }'
```

### Changing Your Password

```bash
curl -X POST http://localhost:8000/api/v1/auth/password/change \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "current_password": "old-password",
    "new_password": "new-secure-password"
  }'
```

### Password Reset

Request a reset token (sent via email in production):

```bash
curl -X POST http://localhost:8000/api/v1/auth/password-reset/request \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com"}'
```

Confirm the reset with the token:

```bash
curl -X POST http://localhost:8000/api/v1/auth/password-reset/confirm \
  -H "Content-Type: application/json" \
  -d '{
    "token": "RESET_TOKEN",
    "new_password": "new-secure-password"
  }'
```

### Creating API Keys

API keys are useful for programmatic access, CI/CD pipelines, and service-to-service communication. Unlike JWT tokens, API keys do not expire by default (unless you set an expiration).

```bash
curl -X POST http://localhost:8000/api/v1/api-keys \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "CI Pipeline Key",
    "description": "Used by GitHub Actions for deployment",
    "scopes": ["read:workflows", "write:workflows", "read:tasks"],
    "expires_in_days": 90
  }'
```

The response contains the full API key value. Save it immediately -- it cannot be retrieved again after this response.

Available scopes:
- `read:all` / `write:all` -- Full read or write access
- `read:tasks` / `write:tasks` -- Task operations
- `read:workflows` / `write:workflows` -- Workflow operations
- `admin` -- Full administrative access

### Listing API Keys

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8000/api/v1/api-keys
```

Include revoked keys:

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "http://localhost:8000/api/v1/api-keys?include_inactive=true"
```

### Revoking an API Key

Revoked keys are kept in the system for audit purposes but can no longer authenticate requests:

```bash
curl -X POST http://localhost:8000/api/v1/api-keys/KEY_ID/revoke \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"reason": "Rotating credentials per security policy"}'
```

### Regenerating an API Key

This revokes the existing key and creates a new one with the same configuration:

```bash
curl -X POST http://localhost:8000/api/v1/api-keys/KEY_ID/regenerate \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Deleting an API Key

Permanent deletion (cannot be undone). Prefer revoking instead unless you need to remove the record entirely:

```bash
curl -X DELETE http://localhost:8000/api/v1/api-keys/KEY_ID \
  -H "Authorization: Bearer YOUR_TOKEN"
```

---

## Deployment

### Quick Start with Docker Compose

The fastest way to run AICtrlNet Community Edition with all dependencies (PostgreSQL, Redis):

```bash
# Download the example compose file
curl -O https://raw.githubusercontent.com/Bodaty/aictrlnet-community/main/docker-compose.example.yml

# Start all services
docker-compose -f docker-compose.example.yml up -d

# Verify the API is running
curl http://localhost:8000/health
```

### Docker Hub

Run the container directly if you have external PostgreSQL and Redis:

```bash
docker pull bodaty/aictrlnet-community:latest

docker run -d -p 8000:8000 \
  -e DATABASE_URL=postgresql://user:pass@host:5432/aictrlnet \
  -e REDIS_URL=redis://host:6379/0 \
  -e SECRET_KEY=your-secret-key \
  bodaty/aictrlnet-community:latest
```

### Manual Installation

```bash
git clone https://github.com/Bodaty/aictrlnet-community.git
cd aictrlnet-community

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# Initialize the database
alembic upgrade head

# Start the server
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `REDIS_URL` | Yes | Redis connection string |
| `SECRET_KEY` | Yes | Secret key for JWT signing |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | No | JWT expiration (default: 30) |

### Database Migrations

Migrations are managed with Alembic and run automatically on `docker-compose up`. To run them manually:

```bash
alembic upgrade head
```

### Health Check

```bash
curl http://localhost:8000/health
```

### Port Reference

| Service | Port |
|---------|------|
| Community API | 8000 |
| PostgreSQL | 5432 |
| Redis | 6379 |

For production deployment considerations (TLS, reverse proxy, scaling), refer to the repository documentation and Docker configuration files.

---

## Next Steps

- Explore the interactive API documentation at `http://localhost:8000/docs` (Swagger UI)
- Browse workflow templates with `GET /api/v1/workflows/create`
- Check the [Edition Migration Guide](EDITION_MIGRATION.md) if you need Business or Enterprise features
- Join the community on [GitHub Discussions](https://github.com/Bodaty/aictrlnet-community/discussions)
