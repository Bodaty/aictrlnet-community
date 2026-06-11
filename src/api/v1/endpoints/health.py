"""Health check endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from core.database import get_db
from core.config import get_settings
from schemas.common import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(db: AsyncSession = Depends(get_db)):
    """Check system health."""
    settings = get_settings()
    services = {
        "api": "ok",
        "edition": settings.EDITION,
    }
    
    # Check database
    try:
        await db.execute(text("SELECT 1"))
        services["database"] = "ok"
    except Exception:
        services["database"] = "error"
    
    # Check Redis if configured
    if settings.REDIS_URL:
        services["redis"] = "not_implemented"
    
    # Check Ollama if AI enabled
    if settings.get_edition_features().get("ai_enabled"):
        services["ai"] = "not_implemented"
    
    return HealthResponse(
        status="ok" if all(v == "ok" for v in services.values() if v != "not_implemented") else "degraded",
        edition=settings.EDITION,
        version=settings.VERSION,
        services=services,
    )


@router.get("/health/detail")
async def health_detail():
    """Worker-level health internals for wedge diagnosis.

    Reports event-loop lag, default-executor and slow-sync pool saturation,
    and DB pool usage. Curl repeatedly to sample all gunicorn workers (pid
    distinguishes them).
    """
    import os

    from core.executors import blocking_executor_stats
    from core.loop_monitor import default_executor_stats, loop_monitor_stats

    detail = {
        "pid": os.getpid(),
        "event_loop": loop_monitor_stats(),
        "default_executor": default_executor_stats(),
        "blocking_executor": blocking_executor_stats(),
    }

    try:
        from core.database import get_engine
        pool = get_engine().pool
        detail["db_pool"] = {
            "size": pool.size(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "status": pool.status(),
        }
    except Exception as e:
        detail["db_pool"] = {"error": str(e)}

    return detail


@router.get("/edition")
async def get_edition_info():
    """Get edition information."""
    settings = get_settings()
    return {
        "edition": settings.EDITION,
        "features": settings.get_edition_features(),
        "version": settings.VERSION,
    }