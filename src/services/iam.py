"""Service layer for Internal Agent Messaging (IAM)."""

from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc
from sqlalchemy.orm import selectinload
from datetime import datetime, timedelta
import uuid
import asyncio
import logging

from models.iam import (
    IAMAgent, IAMMessage, IAMSession, IAMEventLog, IAMMetric,
    IAMMessageType, IAMMessageStatus, IAMAgentStatus
)
from schemas.iam import (
    IAMAgentCreate, IAMAgentUpdate, IAMAgentResponse,
    IAMMessageCreate, IAMMessageResponse, IAMMessageFilter,
    IAMSessionCreate, IAMSessionUpdate, IAMSessionResponse,
    IAMFlowData, IAMSystemMetrics, IAMAgentMetrics,
    IAMCommunicationPatterns, IAMErrorLog
)

logger = logging.getLogger(__name__)


class IAMService:
    """Service for managing Internal Agent Messaging."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    # Agent management
    async def create_agent(self, agent_data: IAMAgentCreate) -> IAMAgentResponse:
        """Create a new IAM agent."""
        agent = IAMAgent(**agent_data.model_dump())
        self.db.add(agent)
        
        # Log agent creation
        await self._log_event(
            event_type="agent_created",
            agent_id=agent.id,
            event_data={"name": agent.name, "type": agent.agent_type}
        )
        
        await self.db.commit()
        await self.db.refresh(agent)
        return IAMAgentResponse.model_validate(agent)
    
    async def get_agent(self, agent_id: uuid.UUID) -> Optional[IAMAgentResponse]:
        """Get an IAM agent by ID."""
        result = await self.db.execute(
            select(IAMAgent).where(IAMAgent.id == agent_id)
        )
        agent = result.scalar_one_or_none()
        return IAMAgentResponse.model_validate(agent) if agent else None
    
    async def list_agents(
        self,
        agent_type: Optional[str] = None,
        status: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[IAMAgentResponse]:
        """List IAM agents with optional filters."""
        query = select(IAMAgent)
        
        if agent_type:
            query = query.where(IAMAgent.agent_type == agent_type)
        if status:
            query = query.where(IAMAgent.status == status)
        
        query = query.offset(skip).limit(limit)
        result = await self.db.execute(query)
        agents = result.scalars().all()
        
        return [IAMAgentResponse.model_validate(agent) for agent in agents]
    
    async def update_agent(
        self,
        agent_id: uuid.UUID,
        agent_update: IAMAgentUpdate
    ) -> Optional[IAMAgentResponse]:
        """Update an IAM agent."""
        result = await self.db.execute(
            select(IAMAgent).where(IAMAgent.id == agent_id)
        )
        agent = result.scalar_one_or_none()
        
        if not agent:
            return None
        
        update_data = agent_update.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(agent, field, value)
        
        agent.updated_at = datetime.utcnow()
        
        await self._log_event(
            event_type="agent_updated",
            agent_id=agent_id,
            event_data=update_data
        )
        
        await self.db.commit()
        await self.db.refresh(agent)
        return IAMAgentResponse.model_validate(agent)
    
    async def update_agent_heartbeat(self, agent_id: uuid.UUID) -> bool:
        """Update agent heartbeat timestamp."""
        result = await self.db.execute(
            select(IAMAgent).where(IAMAgent.id == agent_id)
        )
        agent = result.scalar_one_or_none()
        
        if not agent:
            return False
        
        agent.last_heartbeat = datetime.utcnow()
        agent.status = IAMAgentStatus.ACTIVE
        await self.db.commit()
        return True
    
    # Message handling
    async def send_message(
        self,
        sender_id: uuid.UUID,
        message_data: IAMMessageCreate
    ) -> IAMMessageResponse:
        """Send an IAM message."""
        message = IAMMessage(
            sender_id=sender_id,
            **message_data.model_dump()
        )
        
        # Set expiration if TTL is specified
        if message.ttl_seconds:
            message.expires_at = datetime.utcnow() + timedelta(seconds=message.ttl_seconds)
        
        self.db.add(message)
        
        # Log message creation
        await self._log_event(
            event_type="message_sent",
            agent_id=sender_id,
            message_id=message.id,
            event_data={
                "type": message.message_type,
                "recipient": str(message.recipient_id) if message.recipient_id else "broadcast"
            }
        )
        
        await self.db.commit()
        await self.db.refresh(message)
        
        # Process message delivery asynchronously
        asyncio.create_task(self._process_message_delivery(message.id))
        
        return IAMMessageResponse.model_validate(message)
    
    async def get_messages(
        self,
        filter_params: IAMMessageFilter
    ) -> Dict[str, Any]:
        """Get messages with filtering."""
        query = select(IAMMessage).options(
            selectinload(IAMMessage.sender),
            selectinload(IAMMessage.recipient)
        )
        
        # Apply filters
        conditions = []
        if filter_params.source_id:
            conditions.append(IAMMessage.sender_id == filter_params.source_id)
        if filter_params.destination_id:
            conditions.append(IAMMessage.recipient_id == filter_params.destination_id)
        if filter_params.message_type:
            conditions.append(IAMMessage.message_type == filter_params.message_type)
        if filter_params.status:
            conditions.append(IAMMessage.status == filter_params.status)
        if filter_params.correlation_id:
            conditions.append(IAMMessage.correlation_id == filter_params.correlation_id)
        if filter_params.session_id:
            conditions.append(IAMMessage.session_id == filter_params.session_id)
        if filter_params.start_time:
            conditions.append(IAMMessage.created_at >= filter_params.start_time)
        if filter_params.end_time:
            conditions.append(IAMMessage.created_at <= filter_params.end_time)
        
        if conditions:
            query = query.where(and_(*conditions))
        
        # Get total count
        count_query = select(func.count()).select_from(IAMMessage)
        if conditions:
            count_query = count_query.where(and_(*conditions))
        total_result = await self.db.execute(count_query)
        total = total_result.scalar()
        
        # Apply pagination
        query = query.order_by(desc(IAMMessage.created_at))
        query = query.offset(filter_params.offset).limit(filter_params.limit)
        
        result = await self.db.execute(query)
        messages = result.scalars().all()
        
        return {
            "messages": [IAMMessageResponse.model_validate(msg) for msg in messages],
            "total": total,
            "has_more": (filter_params.offset + filter_params.limit) < total
        }
    
    async def get_message_details(self, message_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """Get detailed information about a message."""
        result = await self.db.execute(
            select(IAMMessage)
            .options(
                selectinload(IAMMessage.sender),
                selectinload(IAMMessage.recipient),
                selectinload(IAMMessage.session)
            )
            .where(IAMMessage.id == message_id)
        )
        message = result.scalar_one_or_none()
        
        if not message:
            return None
        
        # Get trace data (delivery attempts, etc.)
        trace_result = await self.db.execute(
            select(IAMEventLog)
            .where(IAMEventLog.message_id == message_id)
            .order_by(IAMEventLog.created_at)
        )
        trace_events = trace_result.scalars().all()
        
        trace_data = {
            "hops": [
                {
                    "agent": event.event_data.get("agent", "system"),
                    "action": event.event_type,
                    "timestamp": event.created_at.isoformat()
                }
                for event in trace_events
            ]
        }
        
        response = IAMMessageResponse.model_validate(message).model_dump()
        response.update({
            "trace_data": trace_data,
            "retries": message.retry_count,
            "delivery_attempts": len([e for e in trace_events if e.event_type == "delivery_attempt"])
        })
        
        return response
    
    # Session management
    async def create_session(
        self,
        initiator_id: uuid.UUID,
        session_data: IAMSessionCreate
    ) -> IAMSessionResponse:
        """Create a new IAM session."""
        session = IAMSession(
            initiator_id=initiator_id,
            **session_data.model_dump()
        )
        self.db.add(session)
        
        await self._log_event(
            event_type="session_created",
            agent_id=initiator_id,
            session_id=session.id,
            event_data={"type": session.session_type}
        )
        
        await self.db.commit()
        await self.db.refresh(session)
        return IAMSessionResponse.model_validate(session)
    
    async def get_session(self, session_id: uuid.UUID) -> Optional[IAMSessionResponse]:
        """Get a session by ID."""
        result = await self.db.execute(
            select(IAMSession)
            .options(selectinload(IAMSession.initiator))
            .where(IAMSession.id == session_id)
        )
        session = result.scalar_one_or_none()
        return IAMSessionResponse.model_validate(session) if session else None
    
    async def update_session(
        self,
        session_id: uuid.UUID,
        session_update: IAMSessionUpdate
    ) -> Optional[IAMSessionResponse]:
        """Update a session."""
        result = await self.db.execute(
            select(IAMSession).where(IAMSession.id == session_id)
        )
        session = result.scalar_one_or_none()
        
        if not session:
            return None
        
        update_data = session_update.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(session, field, value)
        
        session.updated_at = datetime.utcnow()
        
        if update_data.get("is_active") is False:
            session.ended_at = datetime.utcnow()
        
        await self.db.commit()
        await self.db.refresh(session)
        return IAMSessionResponse.model_validate(session)
    
    # Monitoring and metrics
    async def get_flow_data(
        self,
        time_range: str = "1h",
        agent_ids: Optional[List[uuid.UUID]] = None
    ) -> IAMFlowData:
        """Get flow data for visualization."""
        # Parse time range
        time_delta = self._parse_time_range(time_range)
        start_time = datetime.utcnow() - time_delta
        
        # Build query
        query = select(IAMMessage).where(
            IAMMessage.created_at >= start_time
        )
        
        if agent_ids:
            query = query.where(
                or_(
                    IAMMessage.sender_id.in_(agent_ids),
                    IAMMessage.recipient_id.in_(agent_ids)
                )
            )
        
        result = await self.db.execute(query)
        messages = result.scalars().all()
        
        # Build flow data
        nodes = {}
        edges = {}
        
        for msg in messages:
            # Add sender node
            if msg.sender_id not in nodes:
                sender = await self.get_agent(msg.sender_id)
                if sender:
                    nodes[str(msg.sender_id)] = {
                        "id": str(msg.sender_id),
                        "type": sender.agent_type,
                        "name": sender.name,
                        "messages_sent": 0,
                        "messages_received": 0
                    }
            
            if str(msg.sender_id) in nodes:
                nodes[str(msg.sender_id)]["messages_sent"] += 1
            
            # Add recipient node if not broadcast
            if msg.recipient_id:
                if msg.recipient_id not in nodes:
                    recipient = await self.get_agent(msg.recipient_id)
                    if recipient:
                        nodes[str(msg.recipient_id)] = {
                            "id": str(msg.recipient_id),
                            "type": recipient.agent_type,
                            "name": recipient.name,
                            "messages_sent": 0,
                            "messages_received": 0
                        }
                
                if str(msg.recipient_id) in nodes:
                    nodes[str(msg.recipient_id)]["messages_received"] += 1
                
                # Add edge
                edge_key = f"{msg.sender_id}->{msg.recipient_id}"
                if edge_key not in edges:
                    edges[edge_key] = {
                        "source": str(msg.sender_id),
                        "target": str(msg.recipient_id),
                        "message_count": 0,
                        "avg_size_bytes": 0
                    }
                edges[edge_key]["message_count"] += 1
        
        return IAMFlowData(
            nodes=list(nodes.values()),
            edges=list(edges.values()),
            metrics={
                "total_messages": len(messages),
                "unique_agents": len(nodes),
                "time_range": time_range
            }
        )
    
    async def get_system_metrics(self) -> IAMSystemMetrics:
        """Get system-wide metrics."""
        # Get basic counts
        total_messages = await self.db.scalar(select(func.count(IAMMessage.id)))
        active_agents = await self.db.scalar(
            select(func.count(IAMAgent.id)).where(IAMAgent.status == IAMAgentStatus.ACTIVE)
        )
        
        # Calculate messages per minute (last hour)
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_messages = await self.db.scalar(
            select(func.count(IAMMessage.id)).where(IAMMessage.created_at >= one_hour_ago)
        )
        messages_per_minute = recent_messages / 60.0 if recent_messages else 0
        
        # Calculate error rate
        error_messages = await self.db.scalar(
            select(func.count(IAMMessage.id)).where(IAMMessage.status == IAMMessageStatus.FAILED)
        )
        error_rate = error_messages / total_messages if total_messages > 0 else 0
        
        # TODO: Calculate actual metrics from data
        return IAMSystemMetrics(
            total_messages=total_messages,
            messages_per_minute=messages_per_minute,
            active_agents=active_agents,
            avg_message_size_bytes=425,  # Placeholder
            avg_response_time_ms=1350,   # Placeholder
            error_rate=error_rate,
            uptime_seconds=86400          # Placeholder
        )
    
    async def get_agent_metrics(self, agent_id: uuid.UUID) -> Optional[IAMAgentMetrics]:
        """Get metrics for a specific agent."""
        # Check agent exists
        agent = await self.get_agent(agent_id)
        if not agent:
            return None
        
        # Get message counts
        sent_count = await self.db.scalar(
            select(func.count(IAMMessage.id)).where(IAMMessage.sender_id == agent_id)
        )
        received_count = await self.db.scalar(
            select(func.count(IAMMessage.id)).where(IAMMessage.recipient_id == agent_id)
        )
        
        # Get error rate
        error_count = await self.db.scalar(
            select(func.count(IAMMessage.id)).where(
                and_(
                    IAMMessage.sender_id == agent_id,
                    IAMMessage.status == IAMMessageStatus.FAILED
                )
            )
        )
        error_rate = error_count / sent_count if sent_count > 0 else 0
        
        # Get last activity
        last_message_result = await self.db.execute(
            select(IAMMessage.created_at)
            .where(
                or_(
                    IAMMessage.sender_id == agent_id,
                    IAMMessage.recipient_id == agent_id
                )
            )
            .order_by(desc(IAMMessage.created_at))
            .limit(1)
        )
        last_activity = last_message_result.scalar() or agent.created_at
        
        # Calculate health score
        health_score = 1.0 - error_rate
        if agent.status != IAMAgentStatus.ACTIVE:
            health_score *= 0.5
        
        return IAMAgentMetrics(
            agent_id=agent_id,
            messages_sent=sent_count,
            messages_received=received_count,
            avg_response_time_ms=1250,  # Placeholder
            error_rate=error_rate,
            last_activity=last_activity,
            health_score=health_score
        )
    
    async def get_communication_patterns(
        self,
        time_range: str = "24h"
    ) -> IAMCommunicationPatterns:
        """Analyze communication patterns."""
        time_delta = self._parse_time_range(time_range)
        start_time = datetime.utcnow() - time_delta
        
        # Get message type distribution
        type_result = await self.db.execute(
            select(
                IAMMessage.message_type,
                func.count(IAMMessage.id).label("count")
            )
            .where(IAMMessage.created_at >= start_time)
            .group_by(IAMMessage.message_type)
        )
        type_counts = type_result.all()
        
        total_messages = sum(count for _, count in type_counts)
        patterns = [
            {
                "pattern": msg_type.value,
                "frequency": count,
                "percentage": count / total_messages if total_messages > 0 else 0
            }
            for msg_type, count in type_counts
        ]
        
        # Get busiest routes
        route_result = await self.db.execute(
            select(
                IAMMessage.sender_id,
                IAMMessage.recipient_id,
                func.count(IAMMessage.id).label("count")
            )
            .where(
                and_(
                    IAMMessage.created_at >= start_time,
                    IAMMessage.recipient_id.isnot(None)
                )
            )
            .group_by(IAMMessage.sender_id, IAMMessage.recipient_id)
            .order_by(desc("count"))
            .limit(10)
        )
        route_counts = route_result.all()
        
        # Get agent names for routes
        routes = []
        for sender_id, recipient_id, count in route_counts:
            sender = await self.get_agent(sender_id)
            recipient = await self.get_agent(recipient_id)
            if sender and recipient:
                routes.append({
                    "from_agent": sender.name,
                    "to_agent": recipient.name,
                    "message_count": count
                })
        
        return IAMCommunicationPatterns(
            most_common_patterns=patterns,
            busiest_routes=routes
        )
    
    async def get_error_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[IAMErrorLog]:
        """Get error logs."""
        query = select(IAMMessage).where(
            IAMMessage.status == IAMMessageStatus.FAILED
        )
        
        if start_time:
            query = query.where(IAMMessage.created_at >= start_time)
        if end_time:
            query = query.where(IAMMessage.created_at <= end_time)
        
        query = query.order_by(desc(IAMMessage.created_at)).limit(limit)
        result = await self.db.execute(query)
        failed_messages = result.scalars().all()
        
        error_logs = []
        for msg in failed_messages:
            sender = await self.get_agent(msg.sender_id)
            recipient = await self.get_agent(msg.recipient_id) if msg.recipient_id else None
            
            error_logs.append(IAMErrorLog(
                id=msg.id,
                timestamp=msg.created_at,
                message_id=msg.id,
                error_type="delivery_failure",
                error_message=msg.error_message or "Message delivery failed",
                source_agent=sender.name if sender else str(msg.sender_id),
                destination_agent=recipient.name if recipient else "broadcast",
                severity="error"
            ))
        
        return error_logs
    
    async def get_active_agents(self) -> List[IAMAgentResponse]:
        """Get currently active agents."""
        result = await self.db.execute(
            select(IAMAgent).where(IAMAgent.status == IAMAgentStatus.ACTIVE)
        )
        agents = result.scalars().all()
        return [IAMAgentResponse.model_validate(agent) for agent in agents]
    
    # Private methods
    async def _log_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        agent_id: Optional[uuid.UUID] = None,
        message_id: Optional[uuid.UUID] = None,
        session_id: Optional[uuid.UUID] = None
    ):
        """Log an IAM event."""
        event = IAMEventLog(
            event_type=event_type,
            event_data=event_data,
            agent_id=agent_id,
            message_id=message_id,
            session_id=session_id
        )
        self.db.add(event)
    
    async def _process_message_delivery(self, message_id: uuid.UUID):
        """Process message delivery asynchronously."""
        # This would be implemented to handle actual message delivery
        # For now, we just mark messages as delivered
        async with AsyncSession(self.db.get_bind()) as session:
            result = await session.execute(
                select(IAMMessage).where(IAMMessage.id == message_id)
            )
            message = result.scalar_one_or_none()
            
            if message:
                message.status = IAMMessageStatus.DELIVERED
                message.delivered_at = datetime.utcnow()
                
                await self._log_event(
                    event_type="message_delivered",
                    message_id=message_id,
                    event_data={"delivered_at": message.delivered_at.isoformat()}
                )
                
                await session.commit()
    
    def _parse_time_range(self, time_range: str) -> timedelta:
        """Parse time range string to timedelta."""
        unit = time_range[-1]
        value = int(time_range[:-1])
        
        if unit == "h":
            return timedelta(hours=value)
        elif unit == "d":
            return timedelta(days=value)
        elif unit == "m":
            return timedelta(minutes=value)
        else:
            return timedelta(hours=1)  # Default to 1 hour
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get IAM system health status."""
        # Get active agents count
        active_agents_result = await self.db.execute(
            select(func.count(IAMAgent.id)).where(IAMAgent.status == IAMAgentStatus.ACTIVE)
        )
        active_agents = active_agents_result.scalar() or 0
        
        # Get recent message stats
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_messages_result = await self.db.execute(
            select(func.count(IAMMessage.id)).where(IAMMessage.created_at >= one_hour_ago)
        )
        recent_messages = recent_messages_result.scalar() or 0
        
        # Get failed messages count
        failed_messages_result = await self.db.execute(
            select(func.count(IAMMessage.id)).where(
                and_(
                    IAMMessage.created_at >= one_hour_ago,
                    IAMMessage.status == IAMMessageStatus.FAILED
                )
            )
        )
        failed_messages = failed_messages_result.scalar() or 0
        
        # Calculate error rate
        error_rate = (failed_messages / recent_messages * 100) if recent_messages > 0 else 0
        
        # Determine overall health
        if active_agents == 0:
            health = "critical"
            status_message = "No active agents"
        elif error_rate > 20:
            health = "warning"
            status_message = f"High error rate: {error_rate:.1f}%"
        else:
            health = "healthy"
            status_message = "System operating normally"
        
        return {
            "status": health,
            "message": status_message,
            "metrics": {
                "active_agents": active_agents,
                "messages_last_hour": recent_messages,
                "failed_messages": failed_messages,
                "error_rate": f"{error_rate:.1f}%"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def get_events(
        self,
        event_type: Optional[str] = None,
        agent_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get IAM events."""
        # Query event logs
        query = select(IAMEventLog)
        
        if event_type:
            query = query.where(IAMEventLog.event_type == event_type)
        
        if agent_id:
            query = query.where(IAMEventLog.agent_id == agent_id)
        
        query = query.order_by(desc(IAMEventLog.timestamp)).offset(skip).limit(limit)
        result = await self.db.execute(query)
        events = result.scalars().all()
        
        return [
            {
                "id": str(event.id),
                "event_type": event.event_type,
                "agent_id": str(event.agent_id) if event.agent_id else None,
                "session_id": str(event.session_id) if event.session_id else None,
                "message_id": str(event.message_id) if event.message_id else None,
                "event_data": event.event_data,
                "timestamp": event.timestamp.isoformat()
            }
            for event in events
        ]
    
    async def discover_agents(self, capabilities: List[str]) -> List[IAMAgentResponse]:
        """Discover agents by capabilities."""
        # Query agents that have all requested capabilities
        query = select(IAMAgent).where(IAMAgent.status == IAMAgentStatus.ACTIVE)
        
        # Filter by capabilities using PostgreSQL array operations
        for capability in capabilities:
            query = query.where(IAMAgent.capabilities.contains([capability]))
        
        result = await self.db.execute(query)
        agents = result.scalars().all()
        
        return [IAMAgentResponse.model_validate(agent) for agent in agents]