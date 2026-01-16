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


@router.get("/edition")
async def get_edition_info():
    """Get edition information."""
    settings = get_settings()
    return {
        "edition": settings.EDITION,
        "features": settings.get_edition_features(),
        "version": settings.VERSION,
    }