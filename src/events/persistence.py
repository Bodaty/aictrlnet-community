"""Event persistence for replay and audit."""

from datetime import datetime
from typing import List, Optional, Dict, Any
import json
import asyncio
from pathlib import Path

from .models import Event


class EventPersistence:
    """Simple file-based event persistence."""
    
    def __init__(self, storage_path: str = "/tmp/aictrlnet_events"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self._lock = asyncio.Lock()
    
    async def save_event(self, event: Event):
        """Save an event to persistent storage."""
        if not event.persistent:
            return
        
        async with self._lock:
            # Group events by date for easier management
            date_str = event.timestamp.strftime("%Y-%m-%d")
            file_path = self.storage_path / f"events_{date_str}.jsonl"
            
            # Append event to file
            event_data = event.dict()
            event_data["timestamp"] = event.timestamp.isoformat()
            
            # Use asyncio to write file
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._write_event,
                file_path,
                event_data
            )
    
    def _write_event(self, file_path: Path, event_data: dict):
        """Write event to file (sync operation)."""
        with open(file_path, "a") as f:
            f.write(json.dumps(event_data) + "\n")
    
    async def get_events(
        self,
        start_time: datetime,
        end_time: datetime,
        event_types: Optional[List[str]] = None
    ) -> List[Event]:
        """Retrieve events from persistent storage."""
        events = []
        
        # Get all relevant files
        start_date = start_time.date()
        end_date = end_time.date()
        
        current_date = start_date
        while current_date <= end_date:
            file_path = self.storage_path / f"events_{current_date}.jsonl"
            if file_path.exists():
                file_events = await self._read_events_from_file(
                    file_path,
                    start_time,
                    end_time,
                    event_types
                )
                events.extend(file_events)
            
            # Move to next day
            current_date = current_date.replace(day=current_date.day + 1)
        
        return events
    
    async def _read_events_from_file(
        self,
        file_path: Path,
        start_time: datetime,
        end_time: datetime,
        event_types: Optional[List[str]] = None
    ) -> List[Event]:
        """Read events from a file."""
        loop = asyncio.get_event_loop()
        events_data = await loop.run_in_executor(
            None,
            self._read_file,
            file_path
        )
        
        events = []
        for line in events_data.splitlines():
            if line.strip():
                try:
                    event_data = json.loads(line)
                    event_time = datetime.fromisoformat(event_data["timestamp"])
                    
                    # Check time range
                    if start_time <= event_time <= end_time:
                        # Check event type filter
                        if event_types is None or event_data["type"] in event_types:
                            event = Event(**event_data)
                            events.append(event)
                except Exception:
                    # Skip invalid events
                    pass
        
        return events
    
    def _read_file(self, file_path: Path) -> str:
        """Read file contents (sync operation)."""
        with open(file_path, "r") as f:
            return f.read()
    
    async def cleanup_old_events(self, days_to_keep: int = 30):
        """Clean up events older than specified days."""
        cutoff_date = datetime.now().date()
        cutoff_date = cutoff_date.replace(day=cutoff_date.day - days_to_keep)
        
        for file_path in self.storage_path.glob("events_*.jsonl"):
            # Extract date from filename
            date_str = file_path.stem.replace("events_", "")
            try:
                file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                if file_date < cutoff_date:
                    file_path.unlink()
            except Exception:
                # Skip files with invalid names
                pass