"""GCP credentials helper for Vertex AI integration.

This module provides two-tier credential management:

1. SYSTEM TIER (for AICtrlNet platform internal operations):
   - Uses GCP default credentials automatically from:
     - Cloud Run metadata server (on GCP)
     - GOOGLE_APPLICATION_CREDENTIALS environment variable (local dev)
   - Used for: intent detection, workflow generation, semantic matching
   - Hidden from customers - they cannot access these credentials

2. CUSTOMER TIER (for customer workflow AI nodes):
   - Requires explicit credentials provided by the customer
   - Customers must provide their own GCP project credentials
   - Prevents customers from piggybacking on AICtrlNet's GCP billing
"""

import os
import logging
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Cache for system credentials to avoid repeated metadata server calls
_system_credentials_cache: Optional[Dict[str, Any]] = None
_system_credentials_expiry: Optional[datetime] = None
_CACHE_DURATION_MINUTES = 45  # GCP tokens last ~60 min, refresh at 45


def is_running_on_gcp() -> bool:
    """
    Check if running on GCP (Cloud Run, GCE, GKE, etc.).

    GCP metadata server is only available on GCP infrastructure.
    """
    # Check for Cloud Run environment variables
    if os.environ.get('K_SERVICE') or os.environ.get('CLOUD_RUN_JOB'):
        return True

    # Check for GCE/GKE metadata
    if os.environ.get('GCE_METADATA_HOST'):
        return True

    # Quick check: try to access metadata server (with short timeout)
    try:
        import httpx
        response = httpx.get(
            'http://metadata.google.internal/computeMetadata/v1/',
            headers={'Metadata-Flavor': 'Google'},
            timeout=0.5
        )
        return response.status_code == 200
    except Exception:
        return False


async def get_system_credentials() -> Optional[Dict[str, Any]]:
    """
    Get GCP credentials for SYSTEM TIER operations.

    This is used internally by AICtrlNet for platform operations like:
    - Intent detection
    - Workflow generation
    - Semantic matching
    - AI-powered features

    Credentials are obtained from:
    1. GCP metadata server (on Cloud Run/GCE/GKE)
    2. GOOGLE_APPLICATION_CREDENTIALS file (local development)
    3. Environment variables (fallback)

    Returns:
        Dict with 'access_token', 'project_id', 'location' or None if unavailable

    Note:
        These credentials are HIDDEN from customers. The AICtrlNet platform
        uses its own GCP project for internal AI operations. Customers cannot
        use these credentials for their workflow AI nodes.
    """
    global _system_credentials_cache, _system_credentials_expiry

    # Check cache
    if _system_credentials_cache and _system_credentials_expiry:
        if datetime.utcnow() < _system_credentials_expiry:
            logger.debug("Using cached system credentials")
            return _system_credentials_cache

    credentials = None

    # Try different methods
    if is_running_on_gcp():
        credentials = await _get_credentials_from_metadata_server()

    if not credentials:
        credentials = await _get_credentials_from_adc()

    if not credentials:
        credentials = _get_credentials_from_env()

    if credentials:
        # Cache the credentials
        _system_credentials_cache = credentials
        _system_credentials_expiry = datetime.utcnow() + timedelta(minutes=_CACHE_DURATION_MINUTES)
        logger.info(f"System credentials obtained for project: {credentials.get('project_id', 'unknown')}")
    else:
        logger.warning("No system GCP credentials available - Vertex AI will not work")

    return credentials


async def _get_credentials_from_metadata_server() -> Optional[Dict[str, Any]]:
    """
    Get credentials from GCP metadata server (Cloud Run, GCE, GKE).

    This uses the default service account associated with the Cloud Run service.
    """
    try:
        import httpx

        headers = {'Metadata-Flavor': 'Google'}
        base_url = 'http://metadata.google.internal/computeMetadata/v1'

        async with httpx.AsyncClient(timeout=5.0) as client:
            # Get access token
            token_response = await client.get(
                f'{base_url}/instance/service-accounts/default/token',
                headers=headers
            )
            if token_response.status_code != 200:
                logger.warning(f"Failed to get access token from metadata: {token_response.status_code}")
                return None

            token_data = token_response.json()
            access_token = token_data.get('access_token')

            # Get project ID
            project_response = await client.get(
                f'{base_url}/project/project-id',
                headers=headers
            )
            project_id = project_response.text if project_response.status_code == 200 else None

            # Get zone (to derive region)
            zone_response = await client.get(
                f'{base_url}/instance/zone',
                headers=headers
            )
            zone = zone_response.text if zone_response.status_code == 200 else None
            # Zone format: projects/123/zones/us-central1-a -> extract us-central1
            location = zone.split('/')[-1].rsplit('-', 1)[0] if zone else 'us-central1'

            if access_token and project_id:
                logger.info(f"Got credentials from metadata server: project={project_id}, location={location}")
                return {
                    'access_token': access_token,
                    'project_id': project_id,
                    'location': location,
                    'source': 'metadata_server'
                }

    except Exception as e:
        logger.debug(f"Metadata server not available: {e}")

    return None


async def _get_credentials_from_adc() -> Optional[Dict[str, Any]]:
    """
    Get credentials from Application Default Credentials (ADC).

    This uses the google-auth library to get credentials from:
    - GOOGLE_APPLICATION_CREDENTIALS environment variable
    - User credentials from gcloud auth
    - Service account key file
    """
    try:
        import google.auth
        import google.auth.transport.requests

        credentials, project = google.auth.default(
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )

        # Refresh to get access token
        auth_request = google.auth.transport.requests.Request()
        credentials.refresh(auth_request)

        if credentials.token and project:
            location = os.environ.get('GCP_LOCATION', 'us-central1')
            logger.info(f"Got credentials from ADC: project={project}, location={location}")
            return {
                'access_token': credentials.token,
                'project_id': project,
                'location': location,
                'source': 'adc'
            }

    except ImportError:
        logger.debug("google-auth library not available")
    except Exception as e:
        logger.debug(f"ADC not available: {e}")

    return None


def _get_credentials_from_env() -> Optional[Dict[str, Any]]:
    """
    Get credentials from environment variables (fallback).

    Environment variables:
    - GCP_ACCESS_TOKEN: OAuth2 access token
    - GCP_PROJECT_ID or GOOGLE_CLOUD_PROJECT: Project ID
    - GCP_LOCATION: Region (default: us-central1)
    """
    access_token = os.environ.get('GCP_ACCESS_TOKEN')
    project_id = os.environ.get('GCP_PROJECT_ID') or os.environ.get('GOOGLE_CLOUD_PROJECT')
    location = os.environ.get('GCP_LOCATION', 'us-central1')

    if access_token and project_id:
        logger.info(f"Got credentials from environment: project={project_id}, location={location}")
        return {
            'access_token': access_token,
            'project_id': project_id,
            'location': location,
            'source': 'environment'
        }

    return None


def validate_customer_credentials(credentials: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate customer-provided credentials for CUSTOMER TIER operations.

    Customers must provide their own credentials for workflow AI nodes.
    This prevents them from using AICtrlNet's GCP billing.

    Args:
        credentials: Dict with customer's credentials

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not credentials:
        return False, "Customer credentials are required for AI workflow nodes"

    # Check for required fields
    required_fields = ['project_id']  # access_token or service_account_key required

    for field in required_fields:
        if not credentials.get(field):
            return False, f"Missing required credential field: {field}"

    # Must have either access_token or service_account credentials
    has_auth = (
        credentials.get('access_token') or
        credentials.get('service_account_key') or
        credentials.get('api_key')  # For direct Gemini API
    )

    if not has_auth:
        return False, "Customer must provide access_token, service_account_key, or api_key"

    return True, None


def get_credential_tier(context: Optional[Dict[str, Any]] = None) -> str:
    """
    Determine which credential tier to use based on context.

    Args:
        context: Optional context dict with 'credential_tier' or 'is_system_operation'

    Returns:
        'system' or 'customer'
    """
    if not context:
        return 'system'  # Default to system for internal operations

    # Explicit tier specification
    if context.get('credential_tier'):
        return context['credential_tier']

    # Check if this is a system operation
    if context.get('is_system_operation', False):
        return 'system'

    # Check if this is a workflow node execution (customer tier)
    if context.get('workflow_node_id') or context.get('workflow_execution_id'):
        return 'customer'

    # Default to system for internal operations
    return 'system'


# Sync version for non-async code
def get_system_credentials_sync() -> Optional[Dict[str, Any]]:
    """
    Synchronous version of get_system_credentials.

    Uses cached credentials or falls back to environment variables.
    For full functionality on GCP, use the async version.
    """
    global _system_credentials_cache, _system_credentials_expiry

    # Check cache
    if _system_credentials_cache and _system_credentials_expiry:
        if datetime.utcnow() < _system_credentials_expiry:
            return _system_credentials_cache

    # Try environment variables (sync-safe)
    credentials = _get_credentials_from_env()

    if credentials:
        _system_credentials_cache = credentials
        _system_credentials_expiry = datetime.utcnow() + timedelta(minutes=_CACHE_DURATION_MINUTES)

    return credentials
