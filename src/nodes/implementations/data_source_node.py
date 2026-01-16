"""DataSource node implementation for data input from various sources."""

import json
import csv
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
import aiofiles
import httpx
from pathlib import Path

from ..base_node import BaseNode
from ..models import NodeConfig, NodeExecutionResult, NodeStatus
from events.event_bus import event_bus
from adapters.registry import adapter_registry


logger = logging.getLogger(__name__)


class DataSourceNode(BaseNode):
    """Node for loading data from various sources.
    
    Supports loading data from:
    - Files (JSON, CSV, TXT)
    - APIs (REST endpoints)
    - Databases (via adapters)
    - Static data
    """
    
    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> NodeExecutionResult:
        """Execute the data source node."""
        start_time = datetime.utcnow()
        
        try:
            # Get source configuration
            source_type = self.config.parameters.get("source_type", "static")
            
            # Load data based on source type
            if source_type == "file":
                output_data = await self._load_from_file()
            elif source_type == "api":
                output_data = await self._load_from_api()
            elif source_type == "database":
                output_data = await self._load_from_database()
            elif source_type == "static":
                output_data = await self._load_static_data()
            elif source_type == "input":
                output_data = await self._load_from_input(input_data)
            else:
                raise ValueError(f"Unsupported source type: {source_type}")
            
            # Apply any transformations
            if self.config.parameters.get("transform"):
                output_data = await self._apply_transformations(output_data)
            
            # Calculate duration
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Publish completion event
            await event_bus.publish(
                "node.executed",
                {
                    "node_id": self.config.id,
                    "node_type": "dataSource",
                    "source_type": source_type,
                    "records_loaded": len(output_data.get("data", [])) if isinstance(output_data.get("data"), list) else 1,
                    "duration_ms": duration_ms
                }
            )
            
            return NodeExecutionResult(
                node_instance_id=self.config.id,
                status=NodeStatus.COMPLETED,
                output_data=output_data,
                duration_ms=duration_ms,
                events_published=1
            )
            
        except Exception as e:
            logger.error(f"DataSource node {self.config.id} failed: {str(e)}")
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return NodeExecutionResult(
                node_instance_id=self.config.id,
                status=NodeStatus.FAILED,
                error=str(e),
                duration_ms=duration_ms
            )
    
    async def _load_from_file(self) -> Dict[str, Any]:
        """Load data from a file."""
        file_path = self.config.parameters.get("file_path")
        if not file_path:
            raise ValueError("file_path parameter is required for file source")
        
        file_format = self.config.parameters.get("file_format", "auto")
        
        # Auto-detect format from extension
        if file_format == "auto":
            path = Path(file_path)
            ext = path.suffix.lower()
            if ext == ".json":
                file_format = "json"
            elif ext in [".csv", ".tsv"]:
                file_format = "csv"
            elif ext in [".txt", ".text"]:
                file_format = "text"
            else:
                file_format = "text"  # Default
        
        # Read file based on format
        if file_format == "json":
            return await self._read_json_file(file_path)
        elif file_format == "csv":
            return await self._read_csv_file(file_path)
        elif file_format == "text":
            return await self._read_text_file(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    
    async def _read_json_file(self, file_path: str) -> Dict[str, Any]:
        """Read JSON file."""
        async with aiofiles.open(file_path, 'r') as f:
            content = await f.read()
            data = json.loads(content)
            
            # Ensure data is wrapped in standard format
            if isinstance(data, list):
                return {"data": data}
            elif isinstance(data, dict):
                return {"data": data}
            else:
                return {"data": {"value": data}}
    
    async def _read_csv_file(self, file_path: str) -> Dict[str, Any]:
        """Read CSV file."""
        delimiter = self.config.parameters.get("delimiter", ",")
        has_header = self.config.parameters.get("has_header", True)
        
        rows = []
        async with aiofiles.open(file_path, 'r') as f:
            content = await f.read()
            
        # Parse CSV
        lines = content.strip().split('\n')
        reader = csv.reader(lines, delimiter=delimiter)
        
        if has_header:
            headers = next(reader)
            for row in reader:
                if row:  # Skip empty rows
                    rows.append(dict(zip(headers, row)))
        else:
            for row in reader:
                if row:  # Skip empty rows
                    rows.append(row)
        
        return {"data": rows}
    
    async def _read_text_file(self, file_path: str) -> Dict[str, Any]:
        """Read text file."""
        async with aiofiles.open(file_path, 'r') as f:
            content = await f.read()
        
        # Check if we should split into lines
        split_lines = self.config.parameters.get("split_lines", False)
        
        if split_lines:
            lines = content.strip().split('\n')
            return {"data": lines}
        else:
            return {"data": content}
    
    async def _load_from_api(self) -> Dict[str, Any]:
        """Load data from API endpoint."""
        url = self.config.parameters.get("url")
        if not url:
            raise ValueError("url parameter is required for API source")
        
        method = self.config.parameters.get("method", "GET").upper()
        headers = self.config.parameters.get("headers", {})
        params = self.config.parameters.get("params", {})
        body = self.config.parameters.get("body")
        timeout = self.config.parameters.get("timeout", 30)
        
        # Add authentication if provided
        auth_type = self.config.parameters.get("auth_type")
        if auth_type == "bearer":
            token = self.config.parameters.get("auth_token")
            if token:
                headers["Authorization"] = f"Bearer {token}"
        elif auth_type == "api_key":
            api_key = self.config.parameters.get("api_key")
            api_key_header = self.config.parameters.get("api_key_header", "X-API-Key")
            if api_key:
                headers[api_key_header] = api_key
        
        # Make request
        async with httpx.AsyncClient() as client:
            kwargs = {
                "method": method,
                "url": url,
                "headers": headers,
                "params": params,
                "timeout": timeout
            }
            
            if body and method in ["POST", "PUT", "PATCH"]:
                if isinstance(body, dict):
                    kwargs["json"] = body
                else:
                    kwargs["content"] = body
            
            response = await client.request(**kwargs)
            response.raise_for_status()
            
            # Parse response
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                data = response.json()
            else:
                data = response.text
            
            # Wrap in standard format
            if isinstance(data, list):
                return {"data": data}
            elif isinstance(data, dict):
                return {"data": data}
            else:
                return {"data": {"response": data}}
    
    async def _load_from_database(self) -> Dict[str, Any]:
        """Load data from database using adapter."""
        adapter_id = self.config.parameters.get("adapter_id")
        if not adapter_id:
            raise ValueError("adapter_id parameter is required for database source")
        
        query = self.config.parameters.get("query")
        if not query:
            raise ValueError("query parameter is required for database source")
        
        # Get database adapter class from registry
        adapter_class = adapter_registry.get_adapter_class(adapter_id)
        if not adapter_class:
            raise ValueError(f"Database adapter {adapter_id} not found")
        
        # Create adapter instance
        adapter = adapter_class({})
        
        # Execute query
        from adapters.models import AdapterRequest
        request = AdapterRequest(
            capability="query",
            parameters={
                "query": query,
                "params": self.config.parameters.get("query_params", {})
            }
        )
        
        response = await adapter.execute(request)
        
        if response.status == "error":
            raise Exception(f"Database query failed: {response.error}")
        
        # Extract rows from response
        rows = response.data.get("rows", [])
        return {"data": rows}
    
    async def _load_static_data(self) -> Dict[str, Any]:
        """Load static data from configuration."""
        data = self.config.parameters.get("data")
        if data is None:
            raise ValueError("data parameter is required for static source")
        
        # Wrap in standard format if needed
        if isinstance(data, list):
            return {"data": data}
        elif isinstance(data, dict):
            return {"data": data}
        else:
            return {"data": {"value": data}}
    
    async def _load_from_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Load data from node input."""
        # Get field from input data
        field = self.config.parameters.get("input_field", "data")
        
        if field in input_data:
            data = input_data[field]
            # Wrap in standard format if needed
            if isinstance(data, list):
                return {"data": data}
            elif isinstance(data, dict):
                return {"data": data}
            else:
                return {"data": {"value": data}}
        else:
            # Return all input data if field not found
            return {"data": input_data}
    
    async def _apply_transformations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transformations to loaded data."""
        transformations = self.config.parameters.get("transform", [])
        if not isinstance(transformations, list):
            transformations = [transformations]
        
        result = data
        
        for transform in transformations:
            if isinstance(transform, dict):
                transform_type = transform.get("type")
                
                if transform_type == "filter":
                    # Filter data
                    field = transform.get("field")
                    value = transform.get("value")
                    operator = transform.get("operator", "equals")
                    
                    if "data" in result and isinstance(result["data"], list):
                        filtered = []
                        for item in result["data"]:
                            if isinstance(item, dict) and field in item:
                                item_value = item[field]
                                if operator == "equals" and item_value == value:
                                    filtered.append(item)
                                elif operator == "not_equals" and item_value != value:
                                    filtered.append(item)
                                elif operator == "contains" and value in str(item_value):
                                    filtered.append(item)
                                elif operator == "greater_than" and item_value > value:
                                    filtered.append(item)
                                elif operator == "less_than" and item_value < value:
                                    filtered.append(item)
                        result["data"] = filtered
                
                elif transform_type == "limit":
                    # Limit number of records
                    limit = transform.get("limit", 10)
                    if "data" in result and isinstance(result["data"], list):
                        result["data"] = result["data"][:limit]
                
                elif transform_type == "sort":
                    # Sort data
                    field = transform.get("field")
                    reverse = transform.get("reverse", False)
                    
                    if "data" in result and isinstance(result["data"], list) and field:
                        try:
                            result["data"] = sorted(
                                result["data"],
                                key=lambda x: x.get(field) if isinstance(x, dict) else None,
                                reverse=reverse
                            )
                        except Exception:
                            pass  # Skip if sorting fails
                
                elif transform_type == "map":
                    # Map fields
                    mappings = transform.get("mappings", {})
                    if "data" in result and isinstance(result["data"], list):
                        mapped = []
                        for item in result["data"]:
                            if isinstance(item, dict):
                                new_item = {}
                                for old_key, new_key in mappings.items():
                                    if old_key in item:
                                        new_item[new_key] = item[old_key]
                                mapped.append(new_item)
                        result["data"] = mapped
        
        return result
    
    def validate_config(self) -> bool:
        """Validate node configuration."""
        source_type = self.config.parameters.get("source_type")
        if not source_type:
            raise ValueError("source_type parameter is required")
        
        # Validate based on source type
        if source_type == "file" and not self.config.parameters.get("file_path"):
            raise ValueError("file_path parameter is required for file source")
        elif source_type == "api" and not self.config.parameters.get("url"):
            raise ValueError("url parameter is required for API source")
        elif source_type == "database":
            if not self.config.parameters.get("adapter_id"):
                raise ValueError("adapter_id parameter is required for database source")
            if not self.config.parameters.get("query"):
                raise ValueError("query parameter is required for database source")
        elif source_type == "static" and "data" not in self.config.parameters:
            raise ValueError("data parameter is required for static source")
        
        return True