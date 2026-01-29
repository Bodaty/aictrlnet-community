# AICtrlNet Community Edition

**Build AI workflows where humans and AI work together** — with governance, audit trails, and visual orchestration.

An MIT-licensed AI orchestration platform with native MCP support and human-in-the-loop workflow design.

<!-- Badges - Update URLs after launch -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/Bodaty/aictrlnet-community?style=social)](https://github.com/Bodaty/aictrlnet-community)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)

<!-- TODO: Add screenshot or GIF of HitLai workflow editor here -->
<!-- ![AICtrlNet Workflow Editor](docs/images/workflow-editor.gif) -->

## Why AICtrlNet?

The AI orchestration market is exploding — $7.63B in 2025, projected $50.31B by 2030. But existing tools fall into two camps:

| Category | Tools | The Problem |
|----------|-------|-------------|
| **Code-first frameworks** | LangChain, CrewAI, AutoGen, OpenAI Agents SDK | Powerful for developers, but governance and audit trails require custom implementation |
| **Visual automation** | n8n, Zapier, Make, Dify, Flowise | Easy to connect services, but limited AI governance and compliance controls |

**AICtrlNet bridges this gap** — combining visual workflow design, native human-in-the-loop, and AI governance in a single MIT-licensed platform.

### What Makes AICtrlNet Different

**1. Human-in-the-Loop Native, Not Afterthought**

Unlike tools that treat humans as fallback for AI failures, AICtrlNet orchestrates humans and AI as equal participants. Define approval workflows, escalation paths, and human validation points directly in your workflows.

**2. AI Governance Built-In**

AICtrlNet includes governance at every tier:
- Audit logging and decision tracking (Community)
- Bias detection and monitoring (Business)
- 5-layer AI Workflow Security Gateway (Business/Enterprise)
- Compliance frameworks — HIPAA, GDPR, SOC2 (Business/Enterprise)

**3. Model Context Protocol (MCP) Support**

Native MCP integration for standardized AI model communication — connect any MCP-compatible model or service without custom adapters.

**4. True MIT License**

Unlike n8n's restrictive "fair-code" or commercial licenses with usage caps, AICtrlNet Community is genuinely MIT licensed. Build commercial products, modify freely, no vendor lock-in.

**5. Visual + Code = Best of Both**

Visual workflow designer for rapid prototyping, with full API access for programmatic control. Use what works for your team.

### AICtrlNet vs. Specific Alternatives

<details>
<summary><strong>vs. LangChain / LangGraph</strong></summary>

LangChain is the dominant AI framework and excellent for developers building custom AI applications. LangGraph adds agent orchestration with HITL support.

**Choose LangChain when:** You want maximum control and are comfortable with code-first development.

**Choose AICtrlNet when:** You need built-in governance, audit trails, or want non-developers to participate in workflow design.

*AICtrlNet uses LangChain under the hood for AI execution — we're complementary, not competing.*
</details>

<details>
<summary><strong>vs. CrewAI</strong></summary>

CrewAI excels at role-based AI teams with multi-agent collaboration and has recently added HITL support.

**Choose CrewAI when:** Your primary use case is AI agent teams with defined roles.

**Choose AICtrlNet when:** You need enterprise governance, audit trails, or unified human + AI orchestration across multiple frameworks.
</details>

<details>
<summary><strong>vs. n8n</strong></summary>

n8n is a mature automation platform with 400+ integrations, MCP support, and multi-agent capabilities.

**Choose n8n when:** You need broad automation integrations with AI capabilities, and fair-code licensing works for you.

**Choose AICtrlNet when:** AI governance and audit trails are requirements, or you need true MIT open source. *We integrate with n8n — use both together.*
</details>

<details>
<summary><strong>vs. Dify / Flowise</strong></summary>

Dify and Flowise are excellent visual AI workflow builders.

**Choose Dify/Flowise when:** You want the simplest path to a RAG chatbot or basic AI workflow.

**Choose AICtrlNet when:** You need human-in-the-loop, enterprise governance, multi-tenancy, or compliance features. We go deeper on orchestration where they go wider on accessibility.
</details>

<details>
<summary><strong>vs. Temporal / Prefect / Dagster</strong></summary>

These are workflow orchestration powerhouses for data and microservices.

**Choose Temporal when:** You need bulletproof durability for mission-critical microservices.

**Choose AICtrlNet when:** Your workflows are AI-centric and need human involvement, governance, and visual design. Different tools for different jobs.
</details>

<details>
<summary><strong>vs. Lindy AI</strong></summary>

Lindy is a no-code platform for creating AI "employees" with 4,000+ integrations.

**Choose Lindy when:** You want the easiest possible no-code AI automation with minimal setup.

**Choose AICtrlNet when:** You need self-hosted control, true open source, custom governance, or human-in-the-loop beyond simple approval chains.
</details>

## Key Features

- **Hybrid Architecture**: Centralized control plane with distributed execution
- **AI Agent Orchestration**: Route tasks to AI models, human workers, or software components
- **Human-in-the-Loop**: Workflow nodes for human tasks; approval workflows in Business/Enterprise
- **Model Context Protocol (MCP)**: Standardized context handling for AI interactions
- **5 AI Frameworks**: LangChain, AutoGPT, AutoGen, CrewAI, Semantic Kernel
- **Quality Framework**: Multi-dimensional assessment and validation
- **Governance Controls**: Audit logging (Community); bias detection and compliance (Business/Enterprise)
- **17+ Adapters**: OpenAI, Claude, Slack, Teams, and more (47+ across all editions)

## Editions

| Edition | Description | License |
|---------|-------------|---------|
| **Community** | Core framework, essential adapters | MIT (this repo) |
| **Business** | ML-enhanced features, RAG, 43 industry packs, HitLai UI | Commercial |
| **Enterprise** | Multi-tenancy, federation, compliance, white-label | Commercial |

For Business or Enterprise editions, contact sales@aictrlnet.com.

## Quick Start

### Using Docker (Recommended)

**Option 1: Docker Hub (Fastest)**
```bash
# Pull and run (requires external PostgreSQL and Redis)
docker pull bodaty/aictrlnet-community:latest
docker run -d -p 8000:8000 \
  -e DATABASE_URL=postgresql://user:pass@host:5432/aictrlnet \
  -e REDIS_URL=redis://host:6379/0 \
  bodaty/aictrlnet-community:latest

# API available at http://localhost:8000
# Health check: curl http://localhost:8000/health
```

**Option 2: Docker Compose (Full Stack)**
```bash
# Download the example compose file
curl -O https://raw.githubusercontent.com/Bodaty/aictrlnet-community/main/docker-compose.example.yml

# Copy and customize (optional)
cp docker-compose.example.yml docker-compose.yml

# Start all services (API + PostgreSQL + Redis)
docker-compose up -d

# API available at http://localhost:8000
```

Or clone the full repository:
```bash
git clone https://github.com/Bodaty/aictrlnet-community.git
cd aictrlnet-community
docker-compose up -d
```

### Using pip

```bash
pip install aictrlnet
```

### Manual Installation

```bash
# Clone the repository
git clone https://github.com/Bodaty/aictrlnet-community.git
cd aictrlnet-community

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize database
alembic upgrade head

# Start the server
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## Project Structure

```
aictrlnet-community/
├── src/
│   ├── api/              # API endpoints
│   ├── core/             # Core functionality
│   ├── models/           # SQLAlchemy models
│   ├── schemas/          # Pydantic schemas
│   ├── services/         # Business logic
│   └── main.py           # FastAPI application
├── migrations/           # Alembic database migrations
├── tests/                # Test suite
└── docker/               # Docker configuration
```

## Basic Usage

### Create a Task

```bash
curl -X POST http://localhost:8000/api/v1/tasks/ \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Review data pipeline",
    "description": "Audit the ETL pipeline for data quality issues"
  }'
```

### Create a Workflow

```bash
curl -X POST http://localhost:8000/api/v1/workflows/ \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Content Review Workflow",
    "definition": {
      "nodes": [
        {"id": "generate", "type": "ai_process", "name": "Generate Content"},
        {"id": "review", "type": "human_task", "name": "Human Review"},
        {"id": "publish", "type": "ai_process", "name": "Publish"}
      ],
      "edges": [
        {"from": "generate", "to": "review"},
        {"from": "review", "to": "publish"}
      ]
    }
  }'
```

### Register an MCP Server

```bash
curl -X POST http://localhost:8000/api/v1/mcp/servers \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "filesystem-server",
    "transport_type": "stdio",
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/data"]
  }'
```

## Documentation

- [Getting Started](docs/getting-started.md)
- [API Reference](docs/api.md)
- [Guides](docs/guides.md)
- [Examples](docs/examples.md)
- [Contributing Guide](CONTRIBUTING.md)

## Community

- **GitHub Discussions**: [Ask questions, share ideas](https://github.com/Bodaty/aictrlnet-community/discussions)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes and add tests
4. Run tests: `pytest`
5. Submit a pull request

## Roadmap

- [ ] Additional AI framework support
- [ ] More adapters (submit requests via Issues!)
- [ ] Expanded MCP capabilities
- [ ] Webhook and event-driven workflow triggers

Have an idea? [Open a feature request](https://github.com/Bodaty/aictrlnet-community/issues/new?template=feature_request.md).

## License

MIT License - see [LICENSE](LICENSE) for details.

**This means you can:**
- Use AICtrlNet commercially
- Modify the source code
- Distribute your modifications
- Use it privately
- No attribution required (though appreciated!)

## Contact

- Email: team@aictrlnet.com
- GitHub: [Bodaty/aictrlnet-community](https://github.com/Bodaty/aictrlnet-community)
- Organization: [Bodaty](https://github.com/Bodaty)

## Related Projects

- **HitLai**: React-based visual UI for AICtrlNet (included in Business/Enterprise editions)
- **AICtrlNet Business**: ML-enhanced features, RAG, and 43 industry packs
- **AICtrlNet Enterprise**: Multi-tenant and compliance features

---

<p align="center">
  <strong>Ready to orchestrate AI and humans together?</strong><br>
  <a href="#quick-start">Get Started</a> ·
  <a href="https://github.com/Bodaty/aictrlnet-community/discussions">Discussions</a> ·
  <a href="https://github.com/Bodaty/aictrlnet-community/issues">Report Bug</a>
</p>
