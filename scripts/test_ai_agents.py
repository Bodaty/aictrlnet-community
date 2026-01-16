#!/usr/bin/env python3
"""
Test AI Agent Adapters

This script tests the AI agent framework adapters to ensure they're properly initialized.

Usage:
    python test_ai_agents.py [--adapter langchain|autogpt|autogen|crewai|semantic-kernel]
"""

import asyncio
import sys
import os
from pathlib import Path
import logging
import argparse
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from adapters.factory import AdapterFactory
from adapters.models import AdapterConfig, AdapterCategory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_langchain_adapter():
    """Test LangChain adapter."""
    logger.info("\n=== Testing LangChain Adapter ===")
    
    try:
        # Create adapter
        config = {
            "name": "langchain-test",
            "category": AdapterCategory.AI_AGENT,
            "version": "1.0.0",
            "description": "Test LangChain adapter",
            "required_edition": Edition.COMMUNITY,
            "rate_limit_per_minute": 20,
        }
        
        adapter = AdapterFactory.create_adapter("langchain", config, auto_start=False)
        
        # Initialize
        await adapter.initialize()
        logger.info("✓ Adapter initialized successfully")
        
        # Create agent
        agent = await adapter.create_agent({
            "name": "Test Agent",
            "model": "gpt-3.5-turbo",
            "temperature": 0.7
        })
        logger.info(f"✓ Agent created: {agent}")
        
        # Execute agent
        result = await adapter.execute_agent(agent, {
            "input": "What is 2+2?"
        })
        logger.info(f"✓ Agent executed: {result}")
        
        # Check capabilities
        capabilities = adapter.get_capabilities()
        logger.info(f"✓ Capabilities: {[cap.name for cap in capabilities]}")
        
        # Health check
        health = await adapter.health_check()
        logger.info(f"✓ Health check: {health}")
        
        # Shutdown
        await adapter.shutdown()
        logger.info("✓ Adapter shutdown successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ LangChain test failed: {str(e)}")
        return False


async def test_autogpt_adapter():
    """Test AutoGPT adapter."""
    logger.info("\n=== Testing AutoGPT Adapter ===")
    
    try:
        # Import Business adapter
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'aictrlnet-fastapi-business' / 'src'))
        from adapters.implementations.ai_agents.autogpt_adapter import AutoGPTAdapter
        
        # Create adapter
        adapter = AutoGPTAdapter()
        
        # Initialize
        await adapter.initialize()
        logger.info("✓ Adapter initialized successfully")
        
        # Create agent
        agent = await adapter.create_agent({
            "name": "AutoGPT Test",
            "role": "Task Executor",
            "goals": ["Complete the test task"],
            "max_iterations": 5
        })
        logger.info(f"✓ Agent created: {agent['id']}")
        
        # Execute agent
        result = await adapter.execute_agent(agent, {
            "input": "Analyze this sentence: The quick brown fox",
            "autonomous_mode": False,
            "max_iterations": 3
        })
        logger.info(f"✓ Agent executed: {result.get('output', 'No output')}")
        
        # Get state
        state = await adapter.get_agent_state(agent)
        logger.info(f"✓ Agent state: iterations={state.get('total_iterations', 0)}")
        
        # Health check
        health = await adapter.health_check()
        logger.info(f"✓ Health check: {health}")
        
        # Shutdown
        await adapter.shutdown()
        logger.info("✓ Adapter shutdown successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ AutoGPT test failed: {str(e)}")
        return False


async def test_crewai_adapter():
    """Test CrewAI adapter."""
    logger.info("\n=== Testing CrewAI Adapter ===")
    
    try:
        # Import Business adapter
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'aictrlnet-fastapi-business' / 'src'))
        from adapters.implementations.ai_agents.crewai_adapter import CrewAIAdapter
        
        # Create adapter
        adapter = CrewAIAdapter()
        
        # Initialize
        await adapter.initialize()
        logger.info("✓ Adapter initialized successfully")
        
        # Create agents
        writer = await adapter.create_agent({
            "role": "Content Writer",
            "goal": "Write engaging content",
            "backstory": "An experienced content creator"
        })
        editor = await adapter.create_agent({
            "role": "Editor",
            "goal": "Review and improve content",
            "backstory": "A meticulous editor"
        })
        logger.info("✓ Agents created")
        
        # Create tasks
        task1 = await adapter.create_task({
            "description": "Write a short paragraph about AI",
            "agent_id": writer["id"]
        })
        task2 = await adapter.create_task({
            "description": "Review and edit the paragraph",
            "agent_id": editor["id"]
        })
        logger.info("✓ Tasks created")
        
        # Create crew
        crew = await adapter.create_crew({
            "agents": [writer, editor],
            "tasks": [task1, task2],
            "process": "sequential"
        })
        logger.info(f"✓ Crew created: {crew['id']}")
        
        # Execute crew
        result = await adapter.execute_agent(None, {
            "crew_id": crew["id"],
            "inputs": {"topic": "artificial intelligence"}
        })
        logger.info(f"✓ Crew executed: {result.get('output', 'No output')}")
        
        # Health check
        health = await adapter.health_check()
        logger.info(f"✓ Health check: {health}")
        
        # Shutdown
        await adapter.shutdown()
        logger.info("✓ Adapter shutdown successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ CrewAI test failed: {str(e)}")
        return False


async def test_all_adapters():
    """Test all AI agent adapters."""
    results = {}
    
    # Test each adapter
    results["langchain"] = await test_langchain_adapter()
    results["autogpt"] = await test_autogpt_adapter()
    results["crewai"] = await test_crewai_adapter()
    
    # Summary
    logger.info("\n=== Test Summary ===")
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for adapter, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        logger.info(f"{adapter}: {status}")
    
    logger.info(f"\nTotal: {passed}/{total} passed")
    
    return passed == total


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test AI Agent Adapters"
    )
    parser.add_argument(
        "--adapter",
        choices=["langchain", "autogpt", "autogen", "crewai", "semantic-kernel", "all"],
        default="all",
        help="Adapter to test (default: all)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.adapter == "all":
            success = await test_all_adapters()
        elif args.adapter == "langchain":
            success = await test_langchain_adapter()
        elif args.adapter == "autogpt":
            success = await test_autogpt_adapter()
        elif args.adapter == "crewai":
            success = await test_crewai_adapter()
        else:
            logger.error(f"Test not implemented for {args.adapter}")
            success = False
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logger.error(f"Test error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())