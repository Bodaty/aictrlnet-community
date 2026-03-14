"""Community OAuth2 login endpoints.

Provides social login (Google, Microsoft, GitHub) at the Community level.
Delegates to the Business OAuth2ServiceAsync if available. When Business
edition is not present, returns a clear error so the frontend can hide
OAuth buttons.
"""

import sys
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(tags=["oauth2-login"])

# Try to import Business OAuth2 service — graceful fallback if unavailable
_OAuth2ServiceAsync = None
_OAuth2AuthorizationUrl = None
try:
    if '/workspace/editions/business/src' not in sys.path:
        sys.path.insert(0, '/workspace/editions/business/src')
    from aictrlnet_business.services.oauth2_service_async import OAuth2ServiceAsync
    from aictrlnet_business.schemas.oauth2 import OAuth2AuthorizationUrl
    _OAuth2ServiceAsync = OAuth2ServiceAsync
    _OAuth2AuthorizationUrl = OAuth2AuthorizationUrl
except ImportError:
    logger.info("Business OAuth2 service not available — social login disabled in Community-only mode")


async def _get_oauth2_service(db: AsyncSession = Depends(get_db)):
    if _OAuth2ServiceAsync is None:
        raise HTTPException(
            status_code=501,
            detail="Social login requires Business or Enterprise edition"
        )
    return _OAuth2ServiceAsync(db)


async def _find_provider_by_name(service, provider_name: str):
    providers = await service.list_providers(active_only=True)
    return next(
        (p for p in providers if p.provider_name.lower() == provider_name.lower()),
        None,
    )


@router.get("/login/authorize/{provider_name}")
async def login_authorize(
    provider_name: str,
    redirect_uri: str = Query(..., description="Redirect URI after authorization"),
    state: Optional[str] = Query(None, description="State parameter for CSRF protection"),
    code_challenge: Optional[str] = Query(None, description="PKCE code challenge"),
    code_challenge_method: Optional[str] = Query(None, description="PKCE code challenge method"),
    service=Depends(_get_oauth2_service),
):
    """Get authorization URL for OAuth2 social login."""
    from core.exceptions import NotFoundError, ValidationError

    try:
        provider = await _find_provider_by_name(service, provider_name)
        if not provider:
            raise NotFoundError(f"Provider '{provider_name}' not found")

        result = await service.get_authorization_url(
            provider.id, redirect_uri, state,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
        )
        return result
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get authorization URL: {str(e)}")


@router.post("/login/callback/{provider_name}")
async def login_callback(
    provider_name: str,
    code: str = Query(..., description="Authorization code from provider"),
    state: str = Query(..., description="State parameter for CSRF validation"),
    redirect_uri: str = Query(..., description="Redirect URI used in authorization"),
    code_verifier: Optional[str] = Query(None, description="PKCE code verifier"),
    service=Depends(_get_oauth2_service),
    db: AsyncSession = Depends(get_db),
):
    """Handle OAuth2 callback for social login."""
    # Delegate to Business edition's full implementation
    # Import here to avoid circular imports at module load
    try:
        from aictrlnet_business.api.v1.endpoints.oauth2_login import login_callback as biz_callback
        # The Business endpoint handles user creation, account linking, JWT issuance
        return await biz_callback(
            provider_name=provider_name,
            code=code,
            state=state,
            redirect_uri=redirect_uri,
            code_verifier=code_verifier,
            service=service,
            db=db,
        )
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="Social login callback requires Business or Enterprise edition"
        )
