"""An unhandled exception must reach the browser as a 500 it can read.

Without the error boundary, Starlette's ServerErrorMiddleware renders the 500
outside CORSMiddleware; the browser reports a CORS failure and the SPA swaps
every page for "Can't reach the server".
"""

from fastapi.testclient import TestClient

from core.app import create_app

ORIGIN = "http://localhost:3000"


def _boom():
    raise RuntimeError("boom")


def test_unhandled_exception_is_a_500_with_cors_headers():
    app = create_app()
    app.add_api_route("/__test__/boom", _boom)
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.get("/__test__/boom", headers={"Origin": ORIGIN})

    assert resp.status_code == 500
    assert resp.json() == {"detail": "Internal server error"}
    assert resp.headers.get("access-control-allow-origin") == ORIGIN


def test_error_boundary_sits_directly_inside_cors():
    from fastapi.middleware.cors import CORSMiddleware
    from middleware.error_boundary import ErrorBoundaryMiddleware

    classes = [m.cls for m in create_app().user_middleware]
    cors = classes.index(CORSMiddleware)
    assert classes[cors + 1] is ErrorBoundaryMiddleware
