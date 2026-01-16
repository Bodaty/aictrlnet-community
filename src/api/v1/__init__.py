"""API v1 router aggregation - Community Edition."""

# Re-export the community router
from .community_router import api_router

__all__ = ["api_router"]