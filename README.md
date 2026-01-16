# AICtrlNet Community Edition

**Build AI workflows where humans and AI work together** — with governance, audit trails, and visual orchestration.

The only MIT-licensed AI orchestration platform with native MCP support and true human-in-the-loop capabilities.

<!-- Badges - Update URLs after launch -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/Bodaty/aictrlnet-community?style=social)](https://github.com/Bodaty/aictrlnet-community)
[![Discord](https://img.shields.io/badge/Discord-Join%20Community-7289da)](https://discord.gg/aictrlnet)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)

<!-- TODO: Add screenshot or GIF of HitLai workflow editor here -->
<!-- ![AICtrlNet Workflow Editor](docs/images/workflow-editor.gif) -->

## Why AICtrlNet?

The AI orchestration market is exploding — $7.63B in 2025, projected $50.31B by 2030. But existing tools fall into two camps:

| Category | Tools | The Problem |
|----------|-------|-------------|
| **Code-first frameworks** | LangChain, CrewAI, AutoGen, OpenAI Agents SDK | Great for developers, but no visual orchestration, no governance, steep learning curve |
| **Visual automation** | n8n, Zapier, Make, Dify, Flowise | Easy to use, but AI is bolted-on, limited governance, restrictive licenses |

**AICtrlNet bridges this gap:**

| | LangChain | CrewAI | n8n | Dify | **AICtrlNet** |
|---|:---:|:---:|:---:|:---:|:---:|
| Visual workflow editor | - | - | Yes | Yes | **Yes** |
| Native HITL (Human-in-the-Loop) | - | - | - | - | **Yes** |
| AI governance & compliance | - | - | - | - | **Yes** |
| MCP protocol support | - | - | - | Yes | **Yes** |
| Multi-agent orchestration | Yes | Yes | - | Yes | **Yes** |
| Enterprise multi-tenancy | - | - | - | - | **Yes** |
| True open source (MIT) | Yes | Yes | Fair-code | Apache | **MIT** |
| Human + AI unified platform | - | - | - | - | **Yes** |

### What Makes AICtrlNet Different

**1. Human-in-the-Loop Native, Not Afterthought**

Unlike tools that treat humans as fallback for AI failures, AICtrlNet orchestrates humans and AI as equal participants. Define approval workflows, escalation paths, and human validation points directly in your workflows.

**2. AI Governance Built-In**

While others add governance as an enterprise upsell, AICtrlNet includes:
- 5-layer AI Workflow Security Gateway
- Bias detection and monitoring
- Complete audit trails
- Compliance frameworks (HIPAA, GDPR, SOC2)

**3. Model Context Protocol (MCP) Support**

Native MCP integration for standardized AI model communication — connect any MCP-compatible model or service without custom adapters.

**4. True MIT License**

Unlike n8n's restrictive "fair-code" or commercial licenses with usage caps, AICtrlNet Community is genuinely MIT licensed. Build commercial products, modify freely, no vendor lock-in.

**5. Visual + Code = Best of Both**

Visual workflow designer for rapid prototyping, with full API access for programmatic control. Use what works for your team.

### AICtrlNet vs. Specific Alternatives

<details>
<summary><strong>vs. LangChain / LangGraph</strong></summary>

LangChain is the dominant framework (80K+ GitHub stars) and excellent for developers building custom AI applications.

**Choose LangChain when:** You want maximum control, are comfortable with code, and don't need visual orchestration.

**Choose AICtrlNet when:** You need visual workflows, human-in-the-loop, governance, or want non-developers to participate in workflow design.

*AICtrlNet actually uses LangChain under the hood for AI execution — we're complementary, not competing.*
</details>

<details>
<summary><strong>vs. CrewAI</strong></summary>

CrewAI excels at role-based AI teams that collaborate autonomously.

**Choose CrewAI when:** Your use case is purely AI-to-AI collaboration with defined roles.

**Choose AICtrlNet when:** You need humans in the loop, enterprise governance, or visual workflow design. AICtrlNet supports CrewAI as one of 5 AI frameworks.
</details>

<details>
<summary><strong>vs. n8n</strong></summary>

n8n is a mature automation platform with 400+ integrations.

**Choose n8n when:** You need traditional automation with occasional AI, and fair-code licensing works for you.

**Choose AICtrlNet when:** AI is central (not peripheral), you need governance, or you want true MIT open source. *We actually integrate with n8n — use both together.*
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
- **Human-in-the-Loop**: Built-in approval workflows and escalation paths
- **Model Context Protocol (MCP)**: Standardized context handling for AI interactions
- **5 AI Frameworks**: LangChain, AutoGPT, AutoGen, CrewAI, Semantic Kernel
- **Quality Framework**: Multi-dimensional assessment and validation
- **Governance Controls**: Comprehensive audit logging and compliance
- **27+ Adapters**: OpenAI, Claude, Slack, Teams, Stripe, and more

## Editions

| Edition | Description | License |
|---------|-------------|---------|
| **Community** | Core framework, essential adapters | MIT (this repo) |
| **Business** | ML-enhanced features, RAG, 43 industry packs, HitLai UI | Commercial |
| **Enterprise** | Multi-tenancy, federation, compliance, white-label | Commercial |

For Business or Enterprise editions, contact sales@bodaty.com.

## Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/Bodaty/aictrlnet-community.git
cd aictrlnet-community

# Start with Docker Compose
docker-compose up -d

# API available at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
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

### Register a Component

```bash
curl -X POST http://localhost:8000/api/v1/control/register \
  -H "Content-Type: application/json" \
  -d '{"name": "My AI Agent", "type": "ai"}'
```

### Create a Task

```bash
curl -X POST http://localhost:8000/api/v1/nodes/tasks \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "destination": "ai",
    "payload": {
      "model": "gpt-4",
      "messages": [{"role": "user", "content": "Hello!"}]
    }
  }'
```

### Create a Workflow with Human Approval

```bash
curl -X POST http://localhost:8000/api/v1/workflows \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Content Review Workflow",
    "nodes": [
      {"type": "ai", "model": "gpt-4", "task": "generate_content"},
      {"type": "human", "role": "reviewer", "action": "approve"},
      {"type": "ai", "model": "gpt-4", "task": "publish"}
    ]
  }'
```

## Documentation

- [API Reference](docs/api.md)
- [Architecture Overview](docs/architecture.md)
- [Adapter Guide](docs/adapters.md)
- [MCP Integration](docs/mcp.md)
- [Contributing Guide](CONTRIBUTING.md)

## Community

- **Discord**: [Join our community](https://discord.gg/aictrlnet) <!-- Update link -->
- **GitHub Discussions**: [Ask questions, share ideas](https://github.com/Bodaty/aictrlnet-community/discussions)
- **Twitter/X**: [@aictrlnet](https://twitter.com/aictrlnet) <!-- Update handle -->

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes and add tests
4. Run tests: `pytest`
5. Submit a pull request

## Roadmap

- [ ] Visual workflow designer (Community Edition)
- [ ] Additional AI framework support
- [ ] More adapters (submit requests via Issues!)
- [ ] Expanded MCP capabilities

See our [public roadmap](https://github.com/Bodaty/aictrlnet-community/projects) for what's coming next.

## License

MIT License - see [LICENSE](LICENSE) for details.

**This means you can:**
- Use AICtrlNet commercially
- Modify the source code
- Distribute your modifications
- Use it privately
- No attribution required (though appreciated!)

## Contact

- Email: bobby@bodaty.com
- GitHub: [@bobbykoritala](https://github.com/bobbykoritala)
- Organization: [Bodaty](https://github.com/Bodaty)

## Related Projects

- **HitLai**: React-based visual UI for AICtrlNet (included in Business/Enterprise editions)
- **AICtrlNet Business**: ML-enhanced features, RAG, and 43 industry packs
- **AICtrlNet Enterprise**: Multi-tenant and compliance features

---

<p align="center">
  <strong>Ready to orchestrate AI and humans together?</strong><br>
  <a href="#quick-start">Get Started</a> ·
  <a href="https://discord.gg/aictrlnet">Join Discord</a> ·
  <a href="https://github.com/Bodaty/aictrlnet-community/issues">Report Bug</a>
</p>
