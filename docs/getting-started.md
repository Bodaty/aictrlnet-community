# Getting Started with AICtrlNet Community Edition

This guide walks you through installing AICtrlNet Community Edition, verifying it works, creating your first human-in-the-loop workflow, and connecting AI models.

AICtrlNet is an open-source (MIT) AI orchestration platform that treats humans and AI as equal participants in workflows. It provides built-in governance, audit trails, and visual orchestration out of the box.

---

## Prerequisites

Before you begin, make sure you have the following installed:

| Requirement | Minimum Version | Purpose |
|-------------|----------------|---------|
| Python | 3.10+ | Runtime (manual install only) |
| Docker | 20.10+ | Container runtime |
| Docker Compose | 2.0+ | Multi-container orchestration |
| PostgreSQL | 14+ | Primary database (Docker provides this) |
| Redis | 7+ | Caching and task queues (Docker provides this) |

If you use Docker Compose (recommended), PostgreSQL and Redis are included automatically. You only need them installed separately for manual/pip installations.

Verify your Docker setup:

```bash
docker --version
docker compose version
```

---

## Installation

### Method 1: Docker Compose (Recommended)

This is the fastest way to get a fully working stack with PostgreSQL and Redis included.

```bash
# Download the example compose file
curl -O https://raw.githubusercontent.com/Bodaty/aictrlnet-community/main/docker-compose.example.yml

# Copy and optionally customize
cp docker-compose.example.yml docker-compose.yml

# Start all services
docker compose up -d
```

This starts three containers:
- **aictrlnet** -- the FastAPI application on port 8000
- **postgres** -- PostgreSQL 16 on port 5432
- **redis** -- Redis 7 on port 6379

Alternatively, clone the full repository:

```bash
git clone https://github.com/Bodaty/aictrlnet-community.git
cd aictrlnet-community
docker compose up -d
```

### Method 2: Docker Standalone

If you already have PostgreSQL and Redis running elsewhere:

```bash
docker pull bodaty/aictrlnet-community:latest

docker run -d -p 8000:8000 \
  -e DATABASE_URL=postgresql://user:pass@host:5432/aictrlnet \
  -e REDIS_URL=redis://host:6379/0 \
  -e SECRET_KEY=$(openssl rand -hex 32) \
  bodaty/aictrlnet-community:latest
```

Replace the `DATABASE_URL` and `REDIS_URL` values with your actual connection strings.

### Method 3: Manual (pip)

For development or when you want full control:

```bash
# Clone the repository
git clone https://github.com/Bodaty/aictrlnet-community.git
cd aictrlnet-community

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set required environment variables
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/aictrlnet
export REDIS_URL=redis://localhost:6379/0
export SECRET_KEY=$(openssl rand -hex 32)

# Run database migrations
alembic upgrade head

# Start the server
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

For a complete list of environment variables, see [ENV_VARS.md](../ENV_VARS.md).

---

## Verify Installation

Once the server is running, confirm everything is working:

### Health Check

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status": "healthy"}
```

### API Documentation

Open your browser and navigate to the interactive API docs:

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

These are auto-generated from the FastAPI application and provide a complete reference for every endpoint, including request/response schemas.

### Check Running Containers (Docker)

```bash
docker compose ps
```

You should see three containers in a "running" or "healthy" state: `aictrlnet`, `postgres`, and `redis`.

---

## Your First Workflow

This section walks you through the core workflow: register a component, create a task, and build a workflow that includes human approval.

### Step 1: Register a Component

Components are the building blocks of AICtrlNet -- AI agents, human workers, or software services. Register an AI agent:

```bash
curl -X POST http://localhost:8000/api/v1/control/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Content Generator",
    "type": "ai"
  }'
```

The response includes a component ID. Save it for later use.

### Step 2: Create a Task

Tasks are units of work routed to components. Create a task that sends a prompt to an AI model:

```bash
curl -X POST http://localhost:8000/api/v1/nodes/tasks \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "destination": "ai",
    "payload": {
      "model": "gpt-4",
      "messages": [
        {"role": "user", "content": "Write a short product description for a new AI orchestration tool."}
      ]
    }
  }'
```

Replace `YOUR_TOKEN` with a valid API token. In development, you can generate one through the auth endpoints at `/api/v1/auth/`.

### Step 3: Create a Workflow with Human Approval

This is where AICtrlNet stands apart. Define a multi-step workflow where AI generates content, a human reviews it, and then AI publishes:

```bash
curl -X POST http://localhost:8000/api/v1/workflows \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Content Review Pipeline",
    "nodes": [
      {
        "type": "ai",
        "model": "gpt-4",
        "task": "generate_content"
      },
      {
        "type": "human",
        "role": "reviewer",
        "action": "approve"
      },
      {
        "type": "ai",
        "model": "gpt-4",
        "task": "publish"
      }
    ]
  }'
```

This workflow does three things in sequence:

1. **AI node** -- generates content using GPT-4
2. **Human node** -- pauses execution and waits for a human reviewer to approve or reject
3. **AI node** -- publishes the approved content

The human-in-the-loop step is native to AICtrlNet, not a workaround. The workflow engine holds state, notifies the assigned reviewer, and resumes execution only after explicit approval.

---

## Understanding the Architecture

AICtrlNet Community Edition is built on four core components:

```
                 +------------------+
                 |   API Layer      |
                 |   (FastAPI)      |
                 |   Port 8000      |
                 +--------+---------+
                          |
          +---------------+---------------+
          |                               |
+---------+---------+          +----------+----------+
|   PostgreSQL      |          |   Redis             |
|   Primary DB      |          |   Cache / Queues    |
|   Port 5432       |          |   Port 6379         |
+-------------------+          +---------------------+
```

- **FastAPI Application** -- handles all API requests, workflow orchestration, and business logic. Endpoints are organized under `/api/v1/` with groups for control, nodes, workflows, auth, and more.

- **PostgreSQL** -- stores all persistent data: users, workflows, tasks, audit logs, API keys, and webhook configurations. Schema is managed through Alembic migrations.

- **Redis** -- provides caching, session management, and task queue support. Configurable TTL defaults to 300 seconds.

- **Adapters** -- 27+ built-in adapters connect AICtrlNet to external services: OpenAI, Anthropic Claude, Slack, Microsoft Teams, Stripe, and others. Adapters are pluggable -- you can add custom ones for any service.

### Project Structure

```
aictrlnet-community/
├── src/
│   ├── api/              # API endpoint definitions
│   ├── core/             # Configuration, database, security
│   ├── models/           # SQLAlchemy database models
│   ├── schemas/          # Pydantic request/response schemas
│   ├── services/         # Business logic layer
│   └── main.py           # FastAPI app entry point
├── migrations/           # Alembic database migrations
├── tests/                # Test suite
└── docker-compose.example.yml
```

---

## Connecting AI Models

AICtrlNet supports multiple AI providers through its adapter system. Here is how to connect the most common ones.

### OpenAI (GPT-4, GPT-4o, etc.)

Set the `OPENAI_API_KEY` environment variable:

```bash
# Docker Compose: add to your docker-compose.yml environment section
- OPENAI_API_KEY=sk-your-openai-api-key

# Manual install: export in your shell
export OPENAI_API_KEY=sk-your-openai-api-key
```

Then use OpenAI models in tasks:

```bash
curl -X POST http://localhost:8000/api/v1/nodes/tasks \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "destination": "ai",
    "payload": {
      "model": "gpt-4",
      "messages": [{"role": "user", "content": "Summarize this document."}]
    }
  }'
```

### Anthropic Claude

Set the `ANTHROPIC_API_KEY` environment variable:

```bash
- ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
```

### Ollama (Local Models)

Ollama lets you run models locally with no API keys required. Install Ollama from [ollama.com](https://ollama.com), pull a model, and point AICtrlNet to it:

```bash
# Install and start Ollama
ollama pull llama3.1:8b-instruct-q4_K_M
ollama serve

# Configure AICtrlNet
export OLLAMA_URL=http://localhost:11434
export OLLAMA_MODEL=llama3.1:8b-instruct-q4_K_M
```

If running AICtrlNet in Docker and Ollama on the host machine, use `host.docker.internal` instead of `localhost`:

```bash
- OLLAMA_URL=http://host.docker.internal:11434
```

You can also set a default model for all AI tasks:

```bash
- DEFAULT_LLM_MODEL=llama3.1:8b-instruct-q4_K_M
```

---

## Next Steps

Now that AICtrlNet is running, here is where to go next:

- **API Reference** -- explore all endpoints interactively at [http://localhost:8000/docs](http://localhost:8000/docs) (Swagger UI)
- **Environment Variables** -- see [ENV_VARS.md](../ENV_VARS.md) for the complete configuration reference
- **Redis Caching** -- see [Redis Caching Guide](REDIS_CACHING.md) for cache configuration details
- **Credential Management** -- see [Credential Backend Configuration](CREDENTIAL_BACKEND_CONFIGURATION.md) for secure credential storage
- **Contributing** -- see [CONTRIBUTING.md](../CONTRIBUTING.md) to get involved
- **GitHub Discussions** -- ask questions and share ideas at [github.com/Bodaty/aictrlnet-community/discussions](https://github.com/Bodaty/aictrlnet-community/discussions)
- **Report Issues** -- file bugs at [github.com/Bodaty/aictrlnet-community/issues](https://github.com/Bodaty/aictrlnet-community/issues)

### Upgrading to Business or Enterprise

The Community Edition covers core orchestration, human-in-the-loop workflows, and essential adapters. If you need ML-enhanced features, RAG pipelines, industry packs, or the HitLai visual workflow editor, see the [Business Edition](https://aictrlnet.com). For multi-tenancy, federation, and compliance frameworks (HIPAA, GDPR, SOC2), see the [Enterprise Edition](https://aictrlnet.com). Contact sales@aictrlnet.com for details.
