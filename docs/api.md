# AICtrlNet Community Edition API Reference

This document covers the REST API for AICtrlNet Community Edition. All endpoints are served under the `/api/v1` prefix.

**Interactive API docs** are available when running locally:

- Swagger UI: [http://localhost:8000/api/v1/docs](http://localhost:8000/api/v1/docs)
- ReDoc: [http://localhost:8000/api/v1/redoc](http://localhost:8000/api/v1/redoc)

**Base URL**: `http://localhost:8000`

---

## Authentication

AICtrlNet supports two authentication methods:

### Bearer Token (JWT)

Obtain a token via the `/api/v1/auth/login` endpoint and include it in the `Authorization` header.

```bash
curl -H "Authorization: Bearer <your-jwt-token>" \
  http://localhost:8000/api/v1/health
```

For local development, a built-in token is available:

```bash
curl -H "Authorization: Bearer dev-token-for-testing" \
  http://localhost:8000/api/v1/health
```

### API Keys

API keys can be created through the `/api/v1/users/api-keys` endpoint. Include the key in the `Authorization` header with the `Bearer` scheme.

---

## Health & System Info

### Check system health

```
GET /api/v1/health
```

Returns the status of the API, database, and connected services.

```bash
curl http://localhost:8000/api/v1/health
```

**Response**:

```json
{
  "status": "ok",
  "edition": "community",
  "version": "1.0.0",
  "services": {
    "api": "ok",
    "database": "ok",
    "edition": "community"
  }
}
```

### Get edition information

```
GET /api/v1/edition
```

Returns the current edition, version, and feature flags.

```bash
curl http://localhost:8000/api/v1/edition
```

---

## Auth

All auth endpoints are under `/api/v1/auth`.

### Register a new user

```
POST /api/v1/auth/register
```

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "securepassword",
    "username": "myuser",
    "full_name": "Test User"
  }'
```

**Response**:

```json
{
  "id": "uuid-string",
  "email": "user@example.com",
  "username": "myuser",
  "full_name": "Test User",
  "is_active": true,
  "is_superuser": false,
  "edition": "community",
  "created_at": "2025-01-23T10:00:00"
}
```

### Login

```
POST /api/v1/auth/login
```

Uses OAuth2 password form. If MFA is enabled for the user, returns a session token instead of an access token.

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -d "username=user@example.com&password=securepassword"
```

**Response (no MFA)**:

```json
{
  "access_token": "eyJ...",
  "token_type": "bearer",
  "mfa_required": false
}
```

**Response (MFA required)**:

```json
{
  "mfa_required": true,
  "session_token": "random-session-token",
  "expires_in": 300
}
```

### Verify MFA code

```
POST /api/v1/auth/mfa/verify
```

```bash
curl -X POST http://localhost:8000/api/v1/auth/mfa/verify \
  -H "Content-Type: application/json" \
  -d '{
    "session_token": "random-session-token",
    "code": "123456"
  }'
```

### Get current user

```
GET /api/v1/auth/me
```

```bash
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/api/v1/auth/me
```

### Logout

```
POST /api/v1/auth/logout
```

Client should discard the token after calling this endpoint.

### Refresh token

```
POST /api/v1/auth/token/refresh
```

```bash
curl -X POST http://localhost:8000/api/v1/auth/token/refresh \
  -H "Content-Type: application/json" \
  -d '{"refresh_token": "your-refresh-token"}'
```

### Change password

```
POST /api/v1/auth/password/change
```

```bash
curl -X POST http://localhost:8000/api/v1/auth/password/change \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "current_password": "oldpass",
    "new_password": "newpass"
  }'
```

### Request password reset

```
POST /api/v1/auth/password-reset/request
```

```bash
curl -X POST http://localhost:8000/api/v1/auth/password-reset/request \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com"}'
```

### Confirm password reset

```
POST /api/v1/auth/password-reset/confirm
```

```bash
curl -X POST http://localhost:8000/api/v1/auth/password-reset/confirm \
  -H "Content-Type: application/json" \
  -d '{"token": "reset-token", "new_password": "newpass"}'
```

### Verify email

```
POST /api/v1/auth/verify-email/{token}
```

### Resend verification email

```
POST /api/v1/auth/resend-verification
```

Requires authentication.

---

## Users

All user endpoints are under `/api/v1/users`. Most endpoints require authentication.

### Get current user info

```
GET /api/v1/users/me
```

```bash
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/api/v1/users/me
```

### Get/update user preferences

```
GET  /api/v1/users/me/preferences
PUT  /api/v1/users/me/preferences
```

```bash
curl -X PUT http://localhost:8000/api/v1/users/me/preferences \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "preferences": {
      "notifications": {"email": true, "in_app": true},
      "ui_preferences": {"theme": "dark"}
    }
  }'
```

### Get/update application settings

```
GET    /api/v1/users/app-settings
PUT    /api/v1/users/app-settings
PATCH  /api/v1/users/app-settings
```

Application settings include AI model preferences, theme, notifications, and workspace configuration.

```bash
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/api/v1/users/app-settings
```

### User management (admin only)

These endpoints require superuser privileges.

```
GET    /api/v1/users                  # List users (paginated, searchable)
GET    /api/v1/users/{user_id}        # Get user by ID
POST   /api/v1/users                  # Create user
PUT    /api/v1/users/{user_id}        # Update user
DELETE /api/v1/users/{user_id}        # Delete user
```

```bash
curl -H "Authorization: Bearer <token>" \
  "http://localhost:8000/api/v1/users?limit=10&offset=0&search=test"
```

---

## MFA (Multi-Factor Authentication)

MFA endpoints are under `/api/v1/users`.

```
GET  /api/v1/users/me/mfa             # Get MFA status for current user
GET  /api/v1/users/{user_id}/mfa      # Get MFA status for a user
POST /api/v1/users/{user_id}/mfa/init # Initialize MFA enrollment
```

```bash
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/api/v1/users/me/mfa
```

---

## API Keys

API key endpoints are nested under `/api/v1/users`. All require authentication.

### List API keys

```
GET /api/v1/users/api-keys
```

```bash
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/api/v1/users/api-keys
```

### Create an API key

```
POST /api/v1/users/api-keys
```

The full key value is only returned once on creation.

```bash
curl -X POST http://localhost:8000/api/v1/users/api-keys \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Integration Key",
    "scopes": ["read:all", "write:workflows"],
    "expires_in_days": 90
  }'
```

Available scopes: `read:all`, `write:all`, `read:tasks`, `write:tasks`, `read:workflows`, `write:workflows`, `admin`.

### Get, update, revoke, regenerate, or delete a key

```
GET    /api/v1/users/api-keys/{key_id}
PUT    /api/v1/users/api-keys/{key_id}
POST   /api/v1/users/api-keys/{key_id}/revoke
POST   /api/v1/users/api-keys/{key_id}/regenerate
DELETE /api/v1/users/api-keys/{key_id}
```

---

## Webhooks

Webhook endpoints are nested under `/api/v1/users`. All require authentication.

### List webhooks

```
GET /api/v1/users/webhooks
```

```bash
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/api/v1/users/webhooks
```

### Create a webhook

```
POST /api/v1/users/webhooks
```

```bash
curl -X POST http://localhost:8000/api/v1/users/webhooks \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Webhook",
    "url": "https://example.com/webhook",
    "event_types": ["workflow.completed", "task.*"],
    "secret": "my-webhook-secret"
  }'
```

Event patterns support wildcards: `task.created`, `task.*`, `workflow.*`, `agent.*`, `system.*`, `*`.

### Manage webhooks

```
GET    /api/v1/users/webhooks/{webhook_id}
PUT    /api/v1/users/webhooks/{webhook_id}
DELETE /api/v1/users/webhooks/{webhook_id}
POST   /api/v1/users/webhooks/{webhook_id}/test
POST   /api/v1/users/webhooks/{webhook_id}/enable
POST   /api/v1/users/webhooks/{webhook_id}/disable
GET    /api/v1/users/webhooks/{webhook_id}/deliveries
```

---

## Tasks

Task endpoints are under `/api/v1/tasks`.

### List tasks

```
GET /api/v1/tasks/
```

```bash
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/api/v1/tasks/
```

### Create a task

```
POST /api/v1/tasks/
```

```bash
curl -X POST http://localhost:8000/api/v1/tasks/ \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Review document",
    "description": "Review the Q4 report for accuracy",
    "task_metadata": {"priority": "high"}
  }'
```

### Route a task (with MCP support)

```
POST /api/v1/tasks/route
```

Routes a task to the appropriate handler. If the task targets an MCP server, it is routed through MCP.

```bash
curl -X POST http://localhost:8000/api/v1/tasks/route \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "destination": "ai",
    "payload": {"messages": [{"role": "user", "content": "Summarize this"}]}
  }'
```

### MCP task endpoints

```
GET /api/v1/tasks/mcp           # List MCP tasks
GET /api/v1/tasks/mcp/{task_id} # Get MCP task details
```

---

## Workflows

Workflow endpoints are under `/api/v1/workflows`.

### List workflows

```
GET /api/v1/workflows/
```

Supports query parameters: `skip`, `limit`, `category`, `is_template`, `search`.

```bash
curl -H "Authorization: Bearer <token>" \
  "http://localhost:8000/api/v1/workflows/?limit=10&search=invoice"
```

### Create a workflow

```
POST /api/v1/workflows/
```

Create from scratch with a definition or from a template using `template_id`.

```bash
curl -X POST http://localhost:8000/api/v1/workflows/ \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Document Review",
    "description": "Human-AI document review pipeline",
    "definition": {
      "nodes": [
        {"id": "n1", "type": "input", "position": {"x": 0, "y": 0}},
        {"id": "n2", "type": "ai_review", "position": {"x": 200, "y": 0}},
        {"id": "n3", "type": "human_approval", "position": {"x": 400, "y": 0}}
      ],
      "edges": [
        {"source": "n1", "target": "n2"},
        {"source": "n2", "target": "n3"}
      ]
    }
  }'
```

### Create from template

```
POST /api/v1/workflows/from-template
```

```bash
curl -X POST http://localhost:8000/api/v1/workflows/from-template \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "template_id": "template-uuid",
    "name": "My Workflow from Template",
    "description": "Customized from template"
  }'
```

### Create a manual workflow (with enhancements)

```
POST /api/v1/workflows/create-manual
```

Supports user-selected enhancement options applied through the unified enhancement pipeline.

### Get, update, delete a workflow

```
GET    /api/v1/workflows/{workflow_id}
PUT    /api/v1/workflows/{workflow_id}
DELETE /api/v1/workflows/{workflow_id}
```

### Workflow execution status

```
GET /api/v1/workflows/{workflow_id}/status
```

```bash
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/api/v1/workflows/my-workflow-id/status
```

**Response**:

```json
{
  "workflow_id": "my-workflow-id",
  "status": "not_started",
  "message": "Workflow has not been executed yet"
}
```

### Execute a workflow

```
POST /api/v1/workflows/{workflow_id}/execute
```

```bash
curl -X POST http://localhost:8000/api/v1/workflows/my-workflow-id/execute \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"input_data": {"document_url": "https://example.com/doc.pdf"}}'
```

### Pause, resume, or cancel a running workflow

```
POST /api/v1/workflows/{workflow_id}/pause
POST /api/v1/workflows/{workflow_id}/resume
POST /api/v1/workflows/{workflow_id}/cancel
```

### Workflow instances and executions

```
GET /api/v1/workflows/{workflow_id}/instances
GET /api/v1/workflows/{workflow_id}/executions
GET /api/v1/workflows/executions/{execution_id}
```

### Triggers and schedules

```
POST /api/v1/workflows/{workflow_id}/triggers
POST /api/v1/workflows/{workflow_id}/triggers/webhook
POST /api/v1/workflows/{workflow_id}/triggers/event
GET  /api/v1/workflows/triggers/available
POST /api/v1/workflows/{workflow_id}/trigger           # Manual trigger
GET  /api/v1/workflows/{workflow_id}/schedules
POST /api/v1/workflows/{workflow_id}/schedules
POST /api/v1/workflows/{workflow_id}/schedule
```

### Validate a workflow definition

```
POST /api/v1/workflows/validate
```

```bash
curl -X POST http://localhost:8000/api/v1/workflows/validate \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "nodes": [{"id": "n1", "type": "input"}],
    "edges": []
  }'
```

### Get workflow catalog

```
GET /api/v1/workflows/catalog
```

Returns the dynamic node catalog with available node types, adapters, and MCP tools for building workflows.

### Get workflow creation info

```
GET /api/v1/workflows/create
```

Returns templates, node types, and categories available for workflow creation.

---

## Workflow Templates

Template endpoints are under `/api/v1/workflow-templates`.

### List templates

```
GET /api/v1/workflow-templates
```

Supports query parameters: `category`, `tags`, `edition`, `complexity`, `search`, `sort_by`, `sort_desc`, `skip`, `limit`.

```bash
curl -H "Authorization: Bearer <token>" \
  "http://localhost:8000/api/v1/workflow-templates?category=data_processing&limit=20"
```

### Get a specific template

```
GET /api/v1/workflow-templates/{template_id}
```

Use `?load_definition=true` to include the full workflow definition.

### Instantiate a template

```
POST /api/v1/workflow-templates/{template_id}/instantiate
```

Creates a new workflow from the template with optional parameter customization.

```bash
curl -X POST http://localhost:8000/api/v1/workflow-templates/{template_id}/instantiate \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Instance",
    "description": "Created from template",
    "parameters": {}
  }'
```

### Preview a template

```
GET /api/v1/workflow-templates/{template_id}/preview
```

### Add a review to a template

```
POST /api/v1/workflow-templates/{template_id}/reviews
```

### Create a custom template

```
POST /api/v1/workflow-templates
```

### Template system health

```
GET /api/v1/workflow-templates/health
```

### Initialize system templates

```
POST /api/v1/workflow-templates/initialize-system-templates
```

Scans the template directory and registers all found templates as system templates.

---

## Adapters

Adapter endpoints are under `/api/v1/adapters`.

### List adapters

```
GET /api/v1/adapters/
```

```bash
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/api/v1/adapters/
```

### Get adapter details

```
GET /api/v1/adapters/{adapter_id}
```

### Get adapter categories

```
GET /api/v1/adapters/categories
```

### Check adapter availability

```
POST /api/v1/adapters/check-availability
```

```bash
curl -X POST http://localhost:8000/api/v1/adapters/check-availability \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"adapter_ids": ["slack", "jira", "github"]}'
```

---

## Adapter Configuration

Manage user-specific adapter settings under `/api/v1/adapter-config`.

```
GET    /api/v1/adapter-config/config                     # List configurations
POST   /api/v1/adapter-config/config                     # Create configuration
GET    /api/v1/adapter-config/config/{config_id}         # Get configuration
PUT    /api/v1/adapter-config/config/{config_id}         # Update configuration
DELETE /api/v1/adapter-config/config/{config_id}         # Delete configuration
POST   /api/v1/adapter-config/config/{config_id}/test    # Test connection
```

---

## MCP (Model Context Protocol)

MCP endpoints are under `/api/v1/mcp`. These manage MCP server connections, tool execution, and async tasks.

### System info

```
GET /api/v1/mcp/info
```

```bash
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/api/v1/mcp/info
```

**Response**:

```json
{
  "version": "1.0.0",
  "supported_providers": ["openai", "anthropic", "google", "cohere", "huggingface", "local", "custom"],
  "features": ["context-management", "token-optimization", "multi-provider", "task-routing"],
  "status": "active"
}
```

### MCP Server Management

```
POST   /api/v1/mcp/servers                          # Register a new MCP server
GET    /api/v1/mcp/servers                          # List registered servers
GET    /api/v1/mcp/servers/{server_id}              # Get server details
PATCH  /api/v1/mcp/servers/{server_id}              # Update a server
DELETE /api/v1/mcp/servers/{server_id}              # Delete a server
```

Register an MCP server (stdio transport):

```bash
curl -X POST http://localhost:8000/api/v1/mcp/servers \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "filesystem-server",
    "transport_type": "stdio",
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    "service_type": "filesystem"
  }'
```

### MCP Server Connections

```
POST /api/v1/mcp/servers/{server_id}/connect       # Connect to server (starts subprocess for stdio)
POST /api/v1/mcp/servers/{server_id}/disconnect     # Disconnect from server
GET  /api/v1/mcp/connections                        # List active connections
```

### MCP Server Health

```
GET /api/v1/mcp/servers/{server_id}/health
```

### MCP Tools

```
GET  /api/v1/mcp/servers/{server_id}/tools                     # List tools from connected server
POST /api/v1/mcp/servers/{server_id}/tools/{tool_name}/call    # Call a specific tool
```

Call a tool:

```bash
curl -X POST http://localhost:8000/api/v1/mcp/servers/{server_id}/tools/read_file/call \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"path": "/tmp/test.txt"}'
```

### MCP Resources

```
GET /api/v1/mcp/servers/{server_id}/resources       # List resources from connected server
```

### MCP Task Execution

```
POST /api/v1/mcp/execute                            # Execute a task via MCP
POST /api/v1/mcp/servers/{server_id}/test           # Test server connection
GET  /api/v1/mcp/discovery                          # Discover MCP resources and capabilities
GET  /api/v1/mcp/test                               # Test MCP connection to a URL
```

### MCP Async Tasks (SEP-1686)

For tracking long-running MCP operations:

```
POST   /api/v1/mcp/tasks                                    # Create an async task
GET    /api/v1/mcp/tasks                                    # List async tasks
GET    /api/v1/mcp/tasks/{task_token}                       # Get task status by token
PATCH  /api/v1/mcp/tasks/{task_token}                       # Update task progress/result
POST   /api/v1/mcp/tasks/{task_token}/cancel                # Cancel a running task
DELETE /api/v1/mcp/tasks/{task_token}                       # Delete a completed task
GET    /api/v1/mcp/tasks/server/{server_id}/active          # Get active tasks for a server
```

### MCP Sampling (SEP-1577)

Enables agentic workflows where MCP servers request LLM sampling:

```
POST /api/v1/mcp/sampling/create                    # Handle a sampling request
GET  /api/v1/mcp/sampling/capabilities              # Get sampling capabilities
```

### MCP Elicitation (SEP-1036)

Secure out-of-band credential flows:

```
POST /api/v1/mcp/elicitation/request                        # Create an elicitation request
GET  /api/v1/mcp/elicitation/{request_id}/status            # Check elicitation status
POST /api/v1/mcp/elicitation/{request_id}/complete          # Complete an elicitation
```

---

## NLP (Natural Language Processing)

NLP endpoints are under `/api/v1/nlp`.

### Process natural language into a workflow

```
POST /api/v1/nlp/process
```

```bash
curl -X POST http://localhost:8000/api/v1/nlp/process \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Create a workflow to review invoices with AI and human approval",
    "context": {}
  }'
```

---

## Conversation

Multi-turn conversation endpoints are under `/api/v1/conversation`. These provide the conversational AI assistant interface.

### Session management

```
POST   /api/v1/conversation/sessions                        # Create a new session
GET    /api/v1/conversation/sessions                        # List sessions
GET    /api/v1/conversation/sessions/{session_id}           # Get session with full history
POST   /api/v1/conversation/sessions/{session_id}/end       # End a session
DELETE /api/v1/conversation/sessions/{session_id}           # End/delete a session
```

Create a session:

```bash
curl -X POST http://localhost:8000/api/v1/conversation/sessions \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "initial_message": "Help me set up a document review workflow"
  }'
```

### Send messages

```
POST /api/v1/conversation/sessions/{session_id}/messages
```

```bash
curl -X POST http://localhost:8000/api/v1/conversation/sessions/{session_id}/messages \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"content": "I need it to handle PDF documents"}'
```

### Streaming message endpoints (SSE)

These return Server-Sent Events for real-time progress updates:

```
POST /api/v1/conversation/sessions/{session_id}/messages/stream
POST /api/v1/conversation/sessions/{session_id}/messages/tools/stream
POST /api/v1/conversation/sessions/{session_id}/chat          # v5 unified endpoint
```

### Send message without a session

```
POST /api/v1/conversation/messages
```

Automatically creates or reuses the most recent active session.

### Intent detection

```
POST /api/v1/conversation/detect-intent?text=Create+a+workflow
```

### Intent management

```
POST /api/v1/conversation/intents                   # Create a predefined intent
GET  /api/v1/conversation/intents                   # List intents
```

### Conversation patterns

```
GET  /api/v1/conversation/patterns                  # List learned patterns
POST /api/v1/conversation/patterns/{pattern_id}/promote  # Promote a pattern
```

### Action execution

```
POST /api/v1/conversation/sessions/{session_id}/actions                        # Execute an action
POST /api/v1/conversation/sessions/{session_id}/plan                           # Create action plan
GET  /api/v1/conversation/sessions/{session_id}/actions/{action_id}/progress   # Get action progress
POST /api/v1/conversation/sessions/{session_id}/actions/{action_id}/rollback   # Rollback an action
```

---

## Data Quality (ISO 25012)

Data quality endpoints are available at both `/api/v1/quality` and `/api/v1/data-quality` (backward compatibility).

### Get available quality dimensions

```
GET /api/v1/quality/dimensions
```

Community edition supports `accuracy` and `completeness` dimensions. Business and Enterprise add additional dimensions.

```bash
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/api/v1/quality/dimensions
```

### Assess data quality

```
POST /api/v1/quality/assess
```

```bash
curl -X POST http://localhost:8000/api/v1/quality/assess \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "data": {"records": [{"name": "Test", "email": "test@example.com"}]},
    "dimensions": ["accuracy", "completeness"]
  }'
```

### Get a specific assessment

```
GET /api/v1/quality/assessment/{assessment_id}
```

### Quality dashboard metrics

```
GET /api/v1/quality/dashboard?time_range=30
```

### Quality rules

```
GET  /api/v1/quality/rules                          # List quality rules
POST /api/v1/quality/rules                          # Create a quality rule
```

### Usage statistics

```
GET /api/v1/quality/usage
```

### Quality profiles and audit trail

```
GET /api/v1/quality/profiles                        # Business/Enterprise only
GET /api/v1/quality/audit-trail                     # Business/Enterprise only
```

---

## Knowledge Service

Knowledge endpoints are under `/api/v1/knowledge`. These power the intelligent assistant with system awareness.

### Get system capabilities

```
GET /api/v1/knowledge/capabilities
```

Returns counts of templates, agents, adapters, and automation coverage.

```bash
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/api/v1/knowledge/capabilities
```

### Retrieve relevant knowledge

```
POST /api/v1/knowledge/retrieve
```

Uses RAG-based retrieval to find relevant templates, agents, and features for a given query.

```bash
curl -X POST http://localhost:8000/api/v1/knowledge/retrieve \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I create a document review workflow?",
    "context": {},
    "limit": 5
  }'
```

---

## Bridge

Bridge endpoints are under `/api/v1/bridge` for system integration.

```
POST /api/v1/bridge/connections                     # Create a bridge connection
GET  /api/v1/bridge/connections                     # List connections
GET  /api/v1/bridge/connections/{connection_id}     # Get connection details
GET  /api/v1/bridge/connections/{connection_id}/status  # Get connection status
POST /api/v1/bridge/sync                            # Trigger a sync
```

---

## Platform Integration

Platform integration endpoints are under `/api/v1/platform-integration`.

These endpoints manage connections to external platforms (Slack, Jira, GitHub, etc.) for workflow integration.

---

## Resource Pools

Resource pool endpoints are under `/api/v1/resource-pools`.

```
GET    /api/v1/resource-pools/configs                    # List resource pool configurations
GET    /api/v1/resource-pools/configs/{config_id}        # Get a configuration
POST   /api/v1/resource-pools/configs                    # Create a configuration
PUT    /api/v1/resource-pools/configs/{config_id}        # Update a configuration
DELETE /api/v1/resource-pools/configs/{config_id}        # Delete a configuration
```

---

## Usage Tracking

Usage tracking endpoints are under `/api/v1/usage`.

### Get usage status

```
GET /api/v1/usage/status
```

Returns current usage counts and limits for the Community edition.

```bash
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/api/v1/usage/status
```

### Check specific resource limits

```
GET /api/v1/usage/check-limits?resource_type=workflows
```

Returns 402 if the limit has been reached, with upgrade information.

---

## License Management

License endpoints are under `/api/v1/license`.

### Get current license

```
GET /api/v1/license/current
```

```bash
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/api/v1/license/current
```

**Response**:

```json
{
  "subscription": {
    "id": "sub_community",
    "plan": "community",
    "status": "active",
    "current_period_start": "2025-01-01T00:00:00Z",
    "current_period_end": "2025-01-31T23:59:59Z",
    "features": {
      "max_users": 1,
      "max_workflows": 10,
      "max_executions_per_month": 1000,
      "ai_governance": false,
      "advanced_analytics": false
    }
  }
}
```

### Request an upgrade

```
POST /api/v1/license/upgrade
```

---

## Upgrade & Billing

Upgrade endpoints are under `/api/v1/upgrade`.

### Get upgrade options

```
GET /api/v1/upgrade/options
```

Returns available upgrade paths, pricing, and current usage summary.

```bash
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/api/v1/upgrade/options
```

---

## IAM (Internal Agent Messaging)

IAM endpoints are under `/api/v1/iam` for managing internal agent communication.

### Agent management

```
POST /api/v1/iam/agents                             # Create an agent
GET  /api/v1/iam/agents                             # List agents
GET  /api/v1/iam/agents/active                      # Get active agents
GET  /api/v1/iam/agents/discover?capabilities=nlp   # Discover agents by capability
```

```bash
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/api/v1/iam/agents
```

---

## AI Agent (Basic)

Basic AI agent endpoints are under `/api/v1/ai-agent`. Community edition provides limited agent functionality with rate limits.

### Get AI agent status

```
GET /api/v1/ai-agent/status
```

```bash
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/api/v1/ai-agent/status
```

---

## Agent Execution

Agent execution endpoints are under `/api/v1/agent-execution` for running and monitoring agent tasks.

---

## Memory (Basic)

Basic memory endpoints are under `/api/v1/memory` for agent memory management. Community edition provides limited memory features.

---

## Cache (Basic)

Basic cache management endpoints are under `/api/v1/cache`.

---

## Google A2A Protocol

Google Agent-to-Agent (A2A) protocol endpoints are under `/api/v1/a2a`. Community edition supports discovery features.

---

## LLM Service

Unified LLM service endpoints are under `/api/v1/llm` for interacting with language models.

---

## WebSocket

WebSocket endpoints are available for real-time communication:

```
WS /api/v1/ws/workflow/{workflow_id}    # Real-time workflow execution updates
```

---

## Error Responses

All endpoints return errors in a consistent format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

Common HTTP status codes:

| Code | Meaning |
|------|---------|
| 400  | Bad request (invalid input) |
| 401  | Unauthorized (missing or invalid token) |
| 402  | Payment required (limit reached, upgrade needed) |
| 403  | Forbidden (insufficient permissions) |
| 404  | Not found |
| 422  | Validation error (missing required fields) |
| 500  | Internal server error |

---

## Rate Limits

Community edition enforces the following limits:

| Resource | Limit |
|----------|-------|
| Workflows | 10 |
| Executions per month | 1,000 |
| Users | 1 |
| Quality assessments per month | 1,000 |

Upgrade to Business or Enterprise for higher or unlimited limits. When a limit is reached, the API returns HTTP 402 with upgrade information.

---

## Pagination

List endpoints support pagination via query parameters:

- `skip` or `offset`: Number of items to skip (default: 0)
- `limit` or `per_page`: Maximum items to return (default varies, max usually 1000)
- `page`: Page number (1-indexed, used by some endpoints)

Paginated responses typically include `total` count for calculating pages.
