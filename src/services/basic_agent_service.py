"""Service for managing Basic Agents in Community Edition."""

import logging
from typing import List, Optional
from uuid import UUID, uuid4
from datetime import datetime
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from models.basic_agent import BasicAgent, COMMUNITY_AGENT_LIMIT
from models.user import User
from schemas.basic_agent import (
    BasicAgentCreate,
    BasicAgentUpdate,
    BasicAgentResponse,
    BasicAgentLimitResponse,
    BasicAgentExecuteRequest,
    BasicAgentExecuteResponse
)
from llm import LLMService

logger = logging.getLogger(__name__)


class BasicAgentService:
    """Service for managing basic agents in Community Edition."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.llm_service = LLMService()  # Singleton - no params needed

    async def check_agent_limit(self, user_id: str) -> BasicAgentLimitResponse:
        """Check if user has reached agent limit."""
        result = await self.db.execute(
            select(func.count(BasicAgent.id))
            .where(BasicAgent.user_id == user_id)
        )
        current_count = result.scalar() or 0

        can_create = current_count < COMMUNITY_AGENT_LIMIT

        response = BasicAgentLimitResponse(
            current_count=current_count,
            max_allowed=COMMUNITY_AGENT_LIMIT,
            can_create=can_create
        )

        if not can_create:
            response.upgrade_message = (
                f"You've reached the Community Edition limit of {COMMUNITY_AGENT_LIMIT} agents. "
                "Upgrade to Business Edition for up to 50 agents with enhanced capabilities!"
            )

        return response

    async def create_agent(self, user_id: str, agent_data: BasicAgentCreate) -> BasicAgentResponse:
        """Create a new basic agent."""
        # Check limit
        limit_check = await self.check_agent_limit(user_id)
        if not limit_check.can_create:
            raise ValueError(limit_check.upgrade_message)

        # Create agent
        agent = BasicAgent(
            name=agent_data.name,
            description=agent_data.description,
            agent_type=agent_data.agent_type,
            model=agent_data.model,
            temperature=int(agent_data.temperature * 10),  # Convert to integer scale
            system_prompt=agent_data.system_prompt,
            capabilities=agent_data.capabilities,
            user_id=user_id
        )

        self.db.add(agent)
        try:
            await self.db.commit()
            await self.db.refresh(agent)
            logger.info(f"Created basic agent {agent.id} for user {user_id}")
            return BasicAgentResponse.from_orm(agent)
        except IntegrityError as e:
            await self.db.rollback()
            raise ValueError(f"Failed to create agent: {str(e)}")

    async def get_agent(self, agent_id: UUID, user_id: str) -> Optional[BasicAgentResponse]:
        """Get a specific agent by ID."""
        result = await self.db.execute(
            select(BasicAgent)
            .where(BasicAgent.id == agent_id)
            .where(BasicAgent.user_id == user_id)
        )
        agent = result.scalar_one_or_none()

        if agent:
            return BasicAgentResponse.from_orm(agent)
        return None

    async def list_agents(self, user_id: str) -> List[BasicAgentResponse]:
        """List all agents for a user."""
        result = await self.db.execute(
            select(BasicAgent)
            .where(BasicAgent.user_id == user_id)
            .order_by(BasicAgent.created_at.desc())
        )
        agents = result.scalars().all()
        return [BasicAgentResponse.from_orm(agent) for agent in agents]

    async def update_agent(
        self,
        agent_id: UUID,
        user_id: str,
        update_data: BasicAgentUpdate
    ) -> Optional[BasicAgentResponse]:
        """Update an existing agent."""
        result = await self.db.execute(
            select(BasicAgent)
            .where(BasicAgent.id == agent_id)
            .where(BasicAgent.user_id == user_id)
        )
        agent = result.scalar_one_or_none()

        if not agent:
            return None

        # Update fields
        update_dict = update_data.dict(exclude_unset=True)

        # Convert temperature if present
        if 'temperature' in update_dict:
            update_dict['temperature'] = int(update_dict['temperature'] * 10)

        for key, value in update_dict.items():
            setattr(agent, key, value)

        agent.updated_at = datetime.utcnow()

        await self.db.commit()
        await self.db.refresh(agent)

        logger.info(f"Updated agent {agent_id}")
        return BasicAgentResponse.from_orm(agent)

    async def delete_agent(self, agent_id: UUID, user_id: str) -> bool:
        """Delete an agent."""
        result = await self.db.execute(
            select(BasicAgent)
            .where(BasicAgent.id == agent_id)
            .where(BasicAgent.user_id == user_id)
        )
        agent = result.scalar_one_or_none()

        if not agent:
            return False

        await self.db.delete(agent)
        await self.db.commit()

        logger.info(f"Deleted agent {agent_id}")
        return True

    async def execute_agent(
        self,
        agent_id: UUID,
        user_id: str,
        request: BasicAgentExecuteRequest
    ) -> BasicAgentExecuteResponse:
        """Execute an agent with a prompt."""
        # Get agent
        result = await self.db.execute(
            select(BasicAgent)
            .where(BasicAgent.id == agent_id)
            .where(BasicAgent.user_id == user_id)
        )
        agent = result.scalar_one_or_none()

        if not agent:
            raise ValueError(f"Agent {agent_id} not found")

        if not agent.is_active:
            raise ValueError(f"Agent {agent.name} is not active")

        # Build the full prompt
        full_prompt = ""
        if agent.system_prompt:
            full_prompt = f"{agent.system_prompt}\n\n"
        full_prompt += request.prompt

        # Execute via LLM service
        start_time = datetime.utcnow()

        try:
            response = await self.llm_service.generate(
                prompt=full_prompt,
                model=agent.model,
                temperature=agent.temperature / 10.0,
                task_type="basic_agent",
                context=request.context
            )

            execution_time = (datetime.utcnow() - start_time).total_seconds()

            # Update agent stats
            agent.execution_count += 1
            agent.last_executed = datetime.utcnow()
            await self.db.commit()

            return BasicAgentExecuteResponse(
                agent_id=agent.id,
                execution_id=str(uuid4()),
                response=response.text,
                model_used=agent.model,
                execution_time=execution_time,
                tokens_used=response.usage.get('total_tokens') if response.usage else None,
                timestamp=datetime.utcnow()
            )
        except Exception as e:
            logger.error(f"Failed to execute agent {agent_id}: {str(e)}")
            raise ValueError(f"Agent execution failed: {str(e)}")

    async def create_agent_from_progressive_executor(
        self,
        user_id: str,
        name: str,
        agent_type: str = "assistant",
        description: Optional[str] = None,
        capabilities: Optional[List[str]] = None
    ) -> dict:
        """Create agent from Progressive Executor (simplified interface)."""
        agent_data = BasicAgentCreate(
            name=name,
            description=description or f"A {agent_type} agent created via conversation",
            agent_type=agent_type if agent_type in ["assistant", "support", "task"] else "assistant",
            capabilities=capabilities or ["chat", "assist"]
        )

        try:
            agent = await self.create_agent(user_id, agent_data)
            return {
                "success": True,
                "agent_id": str(agent.id),
                "name": agent.name,
                "type": agent.agent_type,
                "model": agent.model,
                "capabilities": agent.capabilities,
                "message": f"Created basic {agent.agent_type} agent: {agent.name}"
            }
        except ValueError as e:
            if "limit" in str(e).lower():
                return {
                    "success": False,
                    "error": "agent_limit_reached",
                    "message": str(e),
                    "requires_upgrade": True
                }
            else:
                return {
                    "success": False,
                    "error": str(e),
                    "message": f"Failed to create agent: {str(e)}"
                }