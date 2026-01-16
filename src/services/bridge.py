"""Bridge service for system integration."""

from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
import time
import asyncio
import json
from datetime import datetime

from models.community import BridgeConnection, BridgeSync


class BridgeService:
    """Service for bridge-related operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    # Supported bridge types
    SUPPORTED_TYPES = {
        "source": [
            "database", "api", "file", "webhook", "message_queue",
            "kafka", "rabbitmq", "redis", "s3", "ftp", "sftp"
        ],
        "target": [
            "database", "api", "file", "webhook", "message_queue",
            "kafka", "rabbitmq", "redis", "s3", "ftp", "sftp",
            "elasticsearch", "mongodb", "postgres", "mysql"
        ]
    }
    
    async def list_connections(
        self,
        skip: int = 0,
        limit: int = 100,
        status: Optional[str] = None
    ) -> List[BridgeConnection]:
        """List bridge connections."""
        query = select(BridgeConnection)
        
        if status:
            query = query.where(BridgeConnection.status == status)
        
        query = query.order_by(BridgeConnection.created_at.desc()).offset(skip).limit(limit)
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def get_connection(self, connection_id: str) -> Optional[BridgeConnection]:
        """Get a specific bridge connection."""
        query = select(BridgeConnection).where(BridgeConnection.id == connection_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def create_connection(
        self,
        name: str,
        source_type: str,
        target_type: str,
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> BridgeConnection:
        """Create a new bridge connection."""
        # Validate types
        if source_type not in self.SUPPORTED_TYPES["source"]:
            raise ValueError(f"Unsupported source type: {source_type}")
        if target_type not in self.SUPPORTED_TYPES["target"]:
            raise ValueError(f"Unsupported target type: {target_type}")
        
        connection = BridgeConnection(
            name=name,
            source_type=source_type,
            target_type=target_type,
            config=config,
            bridge_metadata=metadata,
            status="active"
        )
        
        self.db.add(connection)
        await self.db.commit()
        await self.db.refresh(connection)
        
        return connection
    
    async def get_connection_status(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get bridge connection status."""
        # Get connection
        connection = await self.get_connection(connection_id)
        if not connection:
            return None
        
        # Get latest sync
        sync_query = select(BridgeSync).where(
            BridgeSync.connection_id == connection_id
        ).order_by(BridgeSync.created_at.desc()).limit(1)
        
        sync_result = await self.db.execute(sync_query)
        latest_sync = sync_result.scalar_one_or_none()
        
        # Get sync statistics
        stats_query = select(
            func.count(BridgeSync.id).label("sync_count"),
            func.count(BridgeSync.id).filter(BridgeSync.status == "error").label("error_count")
        ).where(BridgeSync.connection_id == connection_id)
        
        stats_result = await self.db.execute(stats_query)
        stats = stats_result.first()
        
        return {
            "connection_id": connection_id,
            "status": connection.status,
            "last_sync": latest_sync.created_at.isoformat() if latest_sync else None,
            "sync_count": stats.sync_count if stats else 0,
            "error_count": stats.error_count if stats else 0,
            "last_error": latest_sync.error_message if latest_sync and latest_sync.status == "error" else None
        }
    
    async def sync_connection(
        self,
        connection_id: str,
        force: bool = False,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Synchronize a bridge connection."""
        connection = await self.get_connection(connection_id)
        if not connection:
            raise ValueError(f"Bridge connection '{connection_id}' not found")
        
        if connection.status != "active" and not force:
            raise ValueError(f"Connection is not active (status: {connection.status})")
        
        start_time = time.time()
        
        # Create sync record
        sync_record = BridgeSync(
            connection_id=connection_id,
            status="running",
            sync_options=options or {},
            started_at=datetime.utcnow()
        )
        
        self.db.add(sync_record)
        await self.db.commit()
        await self.db.refresh(sync_record)
        
        try:
            # Perform the actual sync
            result = await self._perform_sync(connection, options or {})
            
            # Update sync record with success
            sync_record.status = "completed"
            sync_record.completed_at = datetime.utcnow()
            sync_record.items_processed = result["items_processed"]
            sync_record.items_created = result["items_created"]
            sync_record.items_updated = result["items_updated"]
            sync_record.items_failed = result["items_failed"]
            
            await self.db.commit()
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            return {
                "success": True,
                "message": f"Sync completed successfully",
                "sync_id": sync_record.id,
                "items_processed": result["items_processed"],
                "items_created": result["items_created"],
                "items_updated": result["items_updated"],
                "items_failed": result["items_failed"],
                "duration_ms": duration_ms
            }
            
        except Exception as e:
            # Update sync record with error
            sync_record.status = "error"
            sync_record.completed_at = datetime.utcnow()
            sync_record.error_message = str(e)
            
            await self.db.commit()
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            return {
                "success": False,
                "message": f"Sync failed: {str(e)}",
                "sync_id": sync_record.id,
                "items_processed": 0,
                "items_created": 0,
                "items_updated": 0,
                "items_failed": 0,
                "duration_ms": duration_ms
            }
    
    async def _perform_sync(
        self,
        connection: BridgeConnection,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform the actual synchronization."""
        # Simulate sync operation based on connection types
        await asyncio.sleep(0.5)  # Simulate processing time
        
        source_type = connection.source_type
        target_type = connection.target_type
        
        # Mock data processing based on types
        if source_type == "database":
            items_processed = options.get("batch_size", 100)
        elif source_type == "api":
            items_processed = options.get("page_size", 50)
        elif source_type == "file":
            items_processed = options.get("chunk_size", 1000)
        else:
            items_processed = 25
        
        # Simulate some processing results
        items_created = int(items_processed * 0.3)  # 30% new items
        items_updated = int(items_processed * 0.6)  # 60% updated items
        items_failed = items_processed - items_created - items_updated  # remainder failed
        
        return {
            "items_processed": items_processed,
            "items_created": items_created,
            "items_updated": items_updated,
            "items_failed": items_failed
        }
    
    async def get_connection_data(
        self,
        connection_id: str,
        skip: int = 0,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Get data from a bridge connection."""
        connection = await self.get_connection(connection_id)
        if not connection:
            raise ValueError(f"Bridge connection '{connection_id}' not found")
        
        # Mock data retrieval
        mock_items = []
        for i in range(min(limit, 50)):  # Cap at 50 for demo
            mock_items.append({
                "id": f"item_{skip + i + 1}",
                "name": f"Item {skip + i + 1}",
                "type": connection.source_type,
                "data": {
                    "source": connection.source_type,
                    "target": connection.target_type,
                    "timestamp": time.time()
                },
                "metadata": {
                    "connection_id": connection_id,
                    "sync_source": connection.name
                }
            })
        
        # Mock total count
        total = 1000  # Simulate large dataset
        
        return {
            "items": mock_items,
            "total": total,
            "page": (skip // limit) + 1,
            "limit": limit,
            "connection_id": connection_id
        }
    
    async def delete_connection(self, connection_id: str) -> bool:
        """Delete a bridge connection."""
        connection = await self.get_connection(connection_id)
        if not connection:
            return False
        
        # Delete related sync records first
        await self.db.execute(
            "DELETE FROM bridge_syncs WHERE connection_id = :connection_id",
            {"connection_id": connection_id}
        )
        
        # Delete the connection
        await self.db.delete(connection)
        await self.db.commit()
        
        return True
    
    async def get_supported_types(self) -> Dict[str, List[str]]:
        """Get supported bridge types."""
        return self.SUPPORTED_TYPES
    
    async def get_connection_logs(
        self,
        connection_id: str,
        skip: int = 0,
        limit: int = 50,
        level: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get logs for a bridge connection."""
        connection = await self.get_connection(connection_id)
        if not connection:
            raise ValueError(f"Bridge connection '{connection_id}' not found")
        
        # Mock log entries
        log_levels = ["info", "warning", "error", "debug"]
        if level and level not in log_levels:
            level = None
        
        logs = []
        for i in range(min(limit, 20)):  # Cap at 20 for demo
            log_level = level if level else log_levels[i % len(log_levels)]
            logs.append({
                "timestamp": (time.time() - (i * 300)),  # 5 minutes apart
                "level": log_level,
                "message": f"Bridge {connection.name}: {log_level.title()} message {i + 1}",
                "source": "bridge_service",
                "connection_id": connection_id,
                "metadata": {
                    "source_type": connection.source_type,
                    "target_type": connection.target_type
                }
            })
        
        return logs
    
    async def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get active bridge sessions."""
        # Get all active connections
        query = select(BridgeConnection).where(BridgeConnection.status == "active")
        result = await self.db.execute(query)
        connections = result.scalars().all()
        
        sessions = []
        for conn in connections:
            # Get last sync info
            sync_query = select(BridgeSync).where(
                BridgeSync.connection_id == conn.id
            ).order_by(BridgeSync.created_at.desc()).limit(1)
            sync_result = await self.db.execute(sync_query)
            last_sync = sync_result.scalar_one_or_none()
            
            session = {
                "session_id": f"session-{conn.id}",
                "connection_id": conn.id,
                "connection_name": conn.name,
                "status": "active" if last_sync and last_sync.status == "completed" else "idle",
                "started_at": conn.created_at.isoformat(),
                "last_activity": last_sync.created_at.isoformat() if last_sync else None,
                "source_type": conn.source_type,
                "target_type": conn.target_type
            }
            sessions.append(session)
        
        return sessions
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get bridge system metrics."""
        # Get connection counts
        total_query = select(func.count(BridgeConnection.id))
        total_result = await self.db.execute(total_query)
        total_connections = total_result.scalar() or 0
        
        active_query = select(func.count(BridgeConnection.id)).where(
            BridgeConnection.status == "active"
        )
        active_result = await self.db.execute(active_query)
        active_connections = active_result.scalar() or 0
        
        # Get sync metrics
        sync_query = select(func.count(BridgeSync.id))
        sync_result = await self.db.execute(sync_query)
        total_syncs = sync_result.scalar() or 0
        
        successful_query = select(func.count(BridgeSync.id)).where(
            BridgeSync.status == "completed"
        )
        successful_result = await self.db.execute(successful_query)
        successful_syncs = successful_result.scalar() or 0
        
        failed_query = select(func.count(BridgeSync.id)).where(
            BridgeSync.status == "failed"
        )
        failed_result = await self.db.execute(failed_query)
        failed_syncs = failed_result.scalar() or 0
        
        # Get last sync time
        last_sync_query = select(BridgeSync.created_at).order_by(
            BridgeSync.created_at.desc()
        ).limit(1)
        last_sync_result = await self.db.execute(last_sync_query)
        last_sync = last_sync_result.scalar_one_or_none()
        
        # Calculate average sync duration (mock for now)
        avg_duration_ms = 1250
        
        # Calculate data transferred (mock for now)
        data_transferred = total_syncs * 1024 * 1024  # Mock: 1MB per sync
        
        return {
            "total_connections": total_connections,
            "active_connections": active_connections,
            "total_syncs": total_syncs,
            "successful_syncs": successful_syncs,
            "failed_syncs": failed_syncs,
            "last_sync_time": last_sync.isoformat() if last_sync else None,
            "average_sync_duration_ms": avg_duration_ms,
            "data_transferred_bytes": data_transferred
        }