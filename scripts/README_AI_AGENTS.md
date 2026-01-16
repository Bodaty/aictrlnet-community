# AI Agent Adapter Scripts

This directory contains scripts for managing AI agent framework adapters in AICtrlNet.

## Available Scripts

### seed_adapters.py
Seeds AI agent adapters into the system with proper edition assignments.

```bash
# Seed adapters for Business Edition (includes Community adapters)
python seed_adapters.py --edition business

# List currently registered adapters
python seed_adapters.py --list

# Clear and reseed adapters
python seed_adapters.py --clear --edition business
```

### test_ai_agents.py
Tests AI agent adapters to ensure they're working correctly.

```bash
# Test all adapters
python test_ai_agents.py

# Test specific adapter
python test_ai_agents.py --adapter langchain
python test_ai_agents.py --adapter autogpt
python test_ai_agents.py --adapter crewai
```

## AI Agent Adapters by Edition

### Community Edition
- **LangChain** (Limited): 100 requests/day limit, basic features only

### Business Edition
- **LangChain** (Full): Unlimited requests, all features including streaming and callbacks
- **AutoGPT**: Autonomous AI agent with goal-oriented task execution
- **AutoGen**: Multi-agent conversation framework with role-based agents
- **CrewAI**: Crew-based task execution with agent collaboration
- **Semantic Kernel**: Microsoft's AI orchestration framework with skills and planners

### Enterprise Edition
- Includes all Business Edition adapters
- Additional enterprise-specific adapters (to be implemented)

## Database Migration

AI agent adapters are also added via Alembic migration:
```bash
# Run migration to add AI agent adapters
docker exec dev-community-1 alembic upgrade head
```

The migration file `c1234567890b_add_ai_agent_adapters.py` adds:
- All AI agent adapter records
- Proper edition assignments
- Sample Enhanced Agent configurations
- Indexes for performance

## Integration with Enhanced Agents

AI agent adapters integrate with the Enhanced Agent system:

1. Each adapter is registered with appropriate capabilities
2. Enhanced Agents can use AI agent adapters as their backend
3. Workflows can orchestrate multiple AI agents
4. Memory sharing enables agent collaboration

## Usage Examples

### Creating a LangChain Agent
```python
from adapters.factory import AdapterFactory

# Create LangChain adapter
adapter = await AdapterFactory.create_and_register_adapter(
    "langchain",
    {
        "name": "my-langchain",
        "rate_limit_per_minute": 20
    }
)

# Create agent
agent = await adapter.create_agent({
    "name": "Assistant",
    "model": "gpt-3.5-turbo",
    "temperature": 0.7,
    "tools": ["web_search", "calculator"]
})

# Execute agent
result = await adapter.execute_agent(agent, {
    "input": "What's the weather like?"
})
```

### Creating a CrewAI Team
```python
# Create CrewAI adapter
adapter = await AdapterFactory.create_and_register_adapter(
    "crewai",
    {"name": "my-crew"}
)

# Create agents
researcher = await adapter.create_agent({
    "role": "Researcher",
    "goal": "Find accurate information"
})

writer = await adapter.create_agent({
    "role": "Writer", 
    "goal": "Create engaging content"
})

# Create crew
crew = await adapter.create_crew({
    "agents": [researcher, writer],
    "tasks": [
        {"description": "Research the topic"},
        {"description": "Write an article"}
    ],
    "process": "sequential"
})

# Execute crew
result = await adapter.execute_agent(None, {
    "crew_id": crew["id"],
    "inputs": {"topic": "AI trends"}
})
```

## Troubleshooting

### Import Errors
If you get import errors when running scripts:
1. Ensure you're in the correct directory
2. The scripts add the src directory to Python path automatically
3. For Business Edition adapters, ensure the Business Edition is installed

### Adapter Not Found
If an adapter type is not found:
1. Check that it's registered in the appropriate factory.py
2. Verify the edition requirements
3. Run seed_adapters.py to register adapters

### Rate Limiting
Community Edition has a 100/day limit for AI agent requests:
- The limit resets daily at midnight UTC
- Business Edition has no limits
- Check usage with the usage tracking API