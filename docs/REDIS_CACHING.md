# Redis Caching Integration

## Overview

AICtrlNet FastAPI now includes Redis caching support to improve performance and reduce database load. The caching layer is transparent and automatically manages cache invalidation.

## Features

- **Automatic caching** of frequently accessed data
- **Cache invalidation** on data modifications
- **Performance monitoring** through cache statistics
- **Edition-based features** (Business/Enterprise only)
- **Per-edition isolation** using separate Redis databases

## Configuration

### Environment Variables

```bash
# Redis connection settings
REDIS_URL=redis://localhost:6379/0
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=  # Optional
REDIS_DB=0      # 0 for Community, 1 for Business, 2 for Enterprise
CACHE_TTL=300   # Default cache TTL in seconds (5 minutes)
```

### Docker Compose

The Redis service is automatically configured in `docker-compose.yml`:

```yaml
redis-fastapi:
  image: redis:7-alpine
  command: redis-server --appendonly yes
  ports:
    - "6380:6379"
  volumes:
    - redis_data:/data
  healthcheck:
    test: ["CMD", "redis-cli", "ping"]
    interval: 10s
    timeout: 5s
    retries: 5
```

## Usage

### Automatic Caching

The following endpoints automatically use caching:

1. **GET /api/v1/tasks/{task_id}** - Individual task retrieval
   - Cache key: `task:{task_id}`
   - TTL: 5 minutes
   - Invalidated on: task update/delete

### Cache Decorators

Use these decorators to add caching to your endpoints:

```python
from src.core.cache import cache_result, invalidate_cache

# Cache results
@cache_result(prefix="task", expire=300, key_builder=lambda task_id, **kwargs: f"task:{task_id}")
async def get_task(task_id: str):
    # Fetch task from database
    pass

# Invalidate cache on modifications
@invalidate_cache(patterns=["task:*", "tasks:*"])
async def update_task(task_id: str, data: dict):
    # Update task in database
    pass
```

### Manual Cache Operations

```python
from src.core.cache import get_cache

cache = await get_cache()

# Set value
await cache.set("key", {"data": "value"}, expire=300)

# Get value
value = await cache.get("key")

# Delete key
await cache.delete("key")

# Clear pattern
await cache.clear_pattern("task:*")
```

## Cache Management API

### Get Cache Statistics (Business/Enterprise)

```bash
GET /api/v1/cache/stats
Authorization: Bearer {token}

Response:
{
  "status": "enabled",
  "server": {
    "version": "7.0.12",
    "uptime_seconds": 3600,
    "connected_clients": 5
  },
  "memory": {
    "used_memory": "2.5MB",
    "peak_memory": "5MB",
    "total_system_memory": "16GB"
  },
  "stats": {
    "total_keys": 150,
    "hits": 1200,
    "misses": 300,
    "hit_rate": 80.0
  }
}
```

### Clear Cache (Business/Enterprise)

```bash
# Clear specific pattern
DELETE /api/v1/cache/clear?pattern=task:*
Authorization: Bearer {token}

# Clear all cache (use with caution!)
DELETE /api/v1/cache/clear
Authorization: Bearer {token}
```

### List Cache Keys (Enterprise only)

```bash
GET /api/v1/cache/keys?pattern=task:*&limit=100
Authorization: Bearer {token}

Response:
{
  "pattern": "task:*",
  "total_found": 25,
  "limit": 100,
  "keys": [
    "task:123e4567-e89b-12d3-a456-426614174000",
    "task:223e4567-e89b-12d3-a456-426614174001",
    ...
  ]
}
```

### Get Cache Key Value (Enterprise only)

```bash
GET /api/v1/cache/key/task:123e4567-e89b-12d3-a456-426614174000
Authorization: Bearer {token}

Response:
{
  "key": "task:123e4567-e89b-12d3-a456-426614174000",
  "value": {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "name": "Sample Task",
    "status": "pending"
  },
  "ttl": 245,
  "type": "dict"
}
```

## Performance Testing

Run the cache test script to verify performance:

```bash
cd aictrlnet-fastapi
python test_cache.py
```

This script will:
1. Check cache connectivity
2. Test cache hit/miss performance
3. Verify cache invalidation
4. Run concurrent request tests
5. Display performance metrics

## Best Practices

1. **Cache Key Naming**: Use consistent, hierarchical key patterns
   ```
   entity:id           # Single entity
   entities:filter     # Lists with filters
   ```

2. **TTL Strategy**: 
   - Short TTL (60s) for frequently changing data
   - Medium TTL (300s) for standard data
   - Long TTL (3600s) for rarely changing data

3. **Invalidation Patterns**:
   - Invalidate specific keys when possible
   - Use wildcard patterns sparingly
   - Consider cascading invalidation for related data

4. **Memory Management**:
   - Monitor Redis memory usage
   - Set appropriate max memory limits
   - Configure eviction policies

## Monitoring

### Using Redis CLI

```bash
# Connect to Redis
docker exec -it redis-fastapi redis-cli

# Check server info
INFO

# Monitor commands in real-time
MONITOR

# Check specific key
GET "task:123e4567-e89b-12d3-a456-426614174000"

# List keys by pattern
KEYS "task:*"
```

### Performance Metrics

Monitor these key metrics:
- **Hit Rate**: Should be > 70% for effective caching
- **Memory Usage**: Keep below 80% of available memory
- **Key Count**: Monitor for unexpected growth
- **Response Times**: Compare cached vs uncached requests

## Troubleshooting

### Cache Not Working

1. Check Redis connectivity:
   ```bash
   docker exec -it redis-fastapi redis-cli ping
   ```

2. Verify environment variables are set correctly

3. Check application logs for Redis connection errors

### High Memory Usage

1. Review TTL settings - reduce if too long
2. Implement key expiration policies
3. Clear unused cache patterns regularly

### Low Hit Rate

1. Analyze access patterns
2. Adjust cache keys to match common queries
3. Increase TTL for frequently accessed data

## Security Considerations

1. **Network Security**: Redis should not be exposed to public internet
2. **Authentication**: Use Redis AUTH in production
3. **Encryption**: Use TLS for Redis connections in production
4. **Access Control**: Limit cache management API to authorized users

## Future Enhancements

- [ ] Add cache warming on startup
- [ ] Implement cache tags for grouped invalidation
- [ ] Add cache metrics to Prometheus exporter
- [ ] Support for Redis Cluster
- [ ] Implement cache stampede protection