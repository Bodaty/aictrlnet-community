"""Main application entry point with new infrastructure integration."""

import os
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.app import create_app
from integration import get_infrastructure_lifespan, integrate_routes


# Create the FastAPI app with integrated infrastructure
def create_integrated_app():
    """Create FastAPI app with new infrastructure components."""
    # Create base app
    app = create_app()
    
    # Get the original lifespan if it exists
    original_lifespan = getattr(app, 'router').lifespan_context
    
    # Create new lifespan that includes infrastructure
    integrated_lifespan = get_infrastructure_lifespan(original_lifespan)
    
    # Update the app's lifespan
    app.router.lifespan_context = integrated_lifespan
    
    # Add any additional routes (control plane enhancement for community)
    edition = os.getenv("EDITION", "community")
    integrate_routes(app, edition)
    
    return app


# Create the app instance
app = create_integrated_app()


if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "main_with_integration:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )