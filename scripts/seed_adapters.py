#!/usr/bin/env python3
"""Seed adapters for Community Edition.

This script seeds all Community Edition adapters directly to the database.
"""

import asyncio
import logging
from typing import Dict, Any
import sys
from pathlib import Path
import os

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select
from models.community_complete import Adapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Community Edition adapter definitions
COMMUNITY_ADAPTERS = {
    "community": [
        # AI Provider Adapters
        {"name": "Ollama", "type": "ollama", "category": "ai", "provider": "Ollama", "required_edition": "community"},
        {"name": "Claude", "type": "claude", "category": "ai", "provider": "Anthropic", "required_edition": "community"},
        {"name": "OpenAI", "type": "openai", "category": "ai", "provider": "OpenAI", "required_edition": "community"},
        {"name": "Hugging Face", "type": "huggingface", "category": "ai", "provider": "HuggingFace", "required_edition": "community"},
        
        # Internal Service Adapters
        {"name": "LLM Service", "type": "llm-service", "category": "ai", "provider": "AICtrlNet", "required_edition": "community"},
        {"name": "ML Service", "type": "ml-service", "category": "ai", "provider": "AICtrlNet", "required_edition": "community"},
        {"name": "MCP Client", "type": "mcp-client", "category": "ai", "provider": "AICtrlNet", "required_edition": "community"},
        
        # Communication Adapters
        {"name": "Slack", "type": "slack", "category": "communication", "provider": "Slack", "required_edition": "community"},
        {"name": "Discord", "type": "discord", "category": "communication", "provider": "Discord", "required_edition": "community"},
        {"name": "Email", "type": "email", "category": "communication", "provider": "SMTP", "required_edition": "community"},
        {"name": "Webhook", "type": "webhook", "category": "communication", "provider": "HTTP", "required_edition": "community"},
        
        # Human Service Adapters
        {"name": "Upwork", "type": "upwork", "category": "human", "provider": "Upwork", "required_edition": "community"},
        {"name": "Fiverr", "type": "fiverr", "category": "human", "provider": "Fiverr", "required_edition": "community"},
        {"name": "TaskRabbit", "type": "taskrabbit", "category": "human", "provider": "TaskRabbit", "required_edition": "community"},
        
        # Payment Adapters
        {"name": "Stripe", "type": "stripe", "category": "payment", "provider": "Stripe", "required_edition": "community"},
        
        # AI Agent Adapters (Community Limited Versions)
        {"name": "LangChain (Community)", "type": "langchain", "category": "ai_agents", "provider": "LangChain", "required_edition": "community"},
        {"name": "AutoGPT (Community)", "type": "autogpt", "category": "ai_agents", "provider": "AutoGPT", "required_edition": "community"},
        {"name": "AutoGen (Community)", "type": "autogen", "category": "ai_agents", "provider": "Microsoft", "required_edition": "community"},
        {"name": "CrewAI (Community)", "type": "crewai", "category": "ai_agents", "provider": "CrewAI", "required_edition": "community"},
        {"name": "Semantic Kernel (Community)", "type": "semantic-kernel", "category": "ai_agents", "provider": "Microsoft", "required_edition": "community"},
    ]
}


async def seed_adapters_to_db(session: AsyncSession, edition: str = "community") -> int:
    """Seed adapters to database for discovery."""
    created_count = 0
    
    # Community only seeds Community adapters
    adapters_to_seed = []
    if edition == "community":
        adapters_to_seed.extend(COMMUNITY_ADAPTERS["community"])
    
    for adapter_data in adapters_to_seed:
        # Check if adapter already exists
        existing = await session.execute(
            select(Adapter).where(Adapter.name == adapter_data["name"])
        )
        if existing.scalar():
            logger.info(f"  ⏭️  Adapter already exists: {adapter_data['name']}")
            continue
        
        # Create adapter record
        adapter = Adapter(
            name=adapter_data["name"],
            category=adapter_data["category"],
            description=f"{adapter_data['provider']} {adapter_data['category']} adapter",
            version="1.0.0",
            min_edition=adapter_data.get("required_edition", "community"),
            enabled=True,
            adapter_metadata={
                "adapter_type": adapter_data["type"],
                "provider": adapter_data["provider"],
                "auto_registered": True,
                "source": "community_seed_script"
            },
            capabilities=[adapter_data["category"]],
            tags=[adapter_data["provider"], adapter_data["category"]],
            available=True,
            installed=True
        )
        session.add(adapter)
        created_count += 1
        logger.info(f"  ✅ Created adapter: {adapter_data['name']}")
    
    await session.commit()
    return created_count


def register_adapters_in_runtime(edition: str = "community") -> int:
    """Register adapters in runtime for immediate use.
    
    Note: Runtime registration is handled by the application startup.
    This function is kept for compatibility but doesn't do anything.
    Database seeding is sufficient for adapter discovery.
    """
    # Runtime registration is handled by app.py during startup
    # We only need database seeding for discovery
    return 0


async def seed_adapters_for_edition(edition: str = "community") -> Dict[str, Any]:
    """Seed adapters for a specific edition."""
    logger.info(f"🚀 Seeding adapters for {edition} edition...")

    # Database connection - try DATABASE_URL first, then SQLALCHEMY_DATABASE_URI, then fallback
    DATABASE_URL = os.getenv('DATABASE_URL') or os.getenv('SQLALCHEMY_DATABASE_URI', 'postgresql+asyncpg://postgres:postgres@postgres:5432/aictrlnet')
    
    engine = create_async_engine(DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    results = {
        "edition": edition,
        "seeded_to_db": 0,
        "registered_in_runtime": 0,
        "errors": []
    }
    
    try:
        # Seed to database
        async with async_session() as session:
            results["seeded_to_db"] = await seed_adapters_to_db(session, edition)
        
        # Register in runtime
        results["registered_in_runtime"] = register_adapters_in_runtime(edition)
        
    except Exception as e:
        logger.error(f"Error seeding adapters: {e}")
        results["errors"].append(str(e))
    finally:
        await engine.dispose()
    
    logger.info(f"✅ Seeded {results['seeded_to_db']} adapters to DB")
    logger.info(f"✅ Registered {results['registered_in_runtime']} adapters in runtime")
    
    return results


async def main():
    """Main function for standalone execution."""
    import argparse
    parser = argparse.ArgumentParser(description="Seed Community Edition adapters")
    parser.add_argument("--edition", default="community", choices=["community"],
                       help="Edition to seed (default: community)")
    args = parser.parse_args()
    
    results = await seed_adapters_for_edition(args.edition)
    
    if results["errors"]:
        logger.error(f"❌ Errors occurred: {results['errors']}")
        sys.exit(1)
    else:
        logger.info("✨ Adapter seeding complete!")


if __name__ == "__main__":
    asyncio.run(main())