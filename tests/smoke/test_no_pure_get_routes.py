"""Make-it-real gate, route surface: every GET route that answers 2xx with a
non-empty body must have touched the (mocked) database, a socket or a
subprocess inside its endpoint function. The sweep runs in a fresh interpreter
so caches warmed by earlier tests can't change the verdict. See
tests/smoke_common/pure_routes.py and .claude/plans/make-it-real-ledger.md §1.
"""

import os

from smoke_common.pure_routes import assert_ledger, sweep_cold

EDITION = "community"
APP_IMPORT = 'from core.app import create_app\n_app = create_app()'

# Routes answering from literals today. Keys are "GET <filled path>"; values the ledger id.
KNOWN_FABRICATED_ROUTES = {
    'GET /api/v1/a2a/agents': 'MR-086 demo-agent-1 registry',
    'GET /api/v1/a2a/health': 'MR-086 unconditional healthy',
    'GET /api/v1/cache/basic-stats': 'MR-087 reports only /cache/simulate injections',
    'GET /api/v1/cache/health': 'MR-087 unconditional healthy',
    'GET /api/v1/cache/info': 'MR-087 usage from the injectable stats',
    'GET /api/v1/llm/models/{model_name:path}/provider-status': 'MR-094 configured flags not checked',
    'GET /api/v1/llm/status': 'MR-094 healthy + cache_enabled literal',
    'GET /api/v1/llm/usage/stats': 'MR-093 cache_hit_rate placeholder',
    'GET /api/v1/mcp/info': "MR-107 status 'active' literal",
    'GET /api/v1/nlp/models': "MR-094 model status 'available' with no check",
    'GET /api/v1/users/preferences': 'MR-108 literal defaults; PATCH does not persist',
}
# Routes whose answer is legitimately static (D-1): one-line reason each.
DECLARED_PURE_ROUTES = {
    'GET /': 'static service banner',
    'GET /.well-known/agent.json': 'static agent card from edition config',
    'GET /api/v1/a2a/.well-known/agent.json': 'static A2A agent card',
    'GET /api/v1/a2a/capabilities': 'static edition capability matrix',
    'GET /api/v1/api-keys/available-scopes': 'static scope catalog',
    'GET /api/v1/autonomy/phases': 'static autonomy taxonomy',
    'GET /api/v1/bridge/types': 'static type catalog',
    'GET /api/v1/canvas/templates': 'static canvas template catalog',
    'GET /api/v1/conversation/budgets': 'static latency budget config',
    'GET /api/v1/edition': 'edition config',
    'GET /api/v1/health/detail': 'in-process runtime stats (event loop, executors, pool)',
    'GET /api/v1/knowledge/manifest': 'generated from the static feature manifest',
    'GET /api/v1/llm/models': 'in-process model registry',
    'GET /api/v1/llm/models/system-defaults': 'model tier config',
    'GET /api/v1/marketplace/categories': 'static category catalog',
    'GET /api/v1/mcp-server/discovery': 'static MCP service catalog',
    'GET /api/v1/mcp/agent/tools': 'in-process MCP tool cache (empty is honest)',
    'GET /api/v1/mcp/agent/tools/summary': 'in-process MCP tool cache',
    'GET /api/v1/mcp/sampling/capabilities': 'static sampling capability',
    'GET /api/v1/mcp/v1/models': 'in-process model registry',
    'GET /api/v1/memory/': 'honest available=False for Community',
    'GET /api/v1/memory/health': 'in-process store, labelled in-memory, persistence false',
    'GET /api/v1/memory/stats': 'in-process store counts',
    'GET /api/v1/nlp/templates/categories': 'static category catalog',
    'GET /api/v1/platform-integration/cache/statistics': 'in-process cache counters',
    'GET /api/v1/platform-integration/cloud/config': 'environment config',
    'GET /api/v1/platform-integration/cloud/info': 'environment config',
    'GET /api/v1/platform-integration/credentials/backend/info': 'static description of the credential backend',
    'GET /api/v1/platform-integration/monitoring/metrics/realtime': 'in-process execution counters',
    'GET /api/v1/resource-pools/types': 'static type catalog',
    'GET /api/v1/users/api-keys/available-scopes': 'static scope catalog',
    'GET /api/v1/users/me/preferences': 'read from the user row (merged with defaults)',
    'GET /api/v1/users/profile': 'read from the authenticated user',
    'GET /api/v1/workflows/triggers/available': 'static trigger catalog',
    'GET /health': 'liveness',
}


def test_no_get_route_answers_from_literals_outside_the_ledger():
    rows = sweep_cold(EDITION, APP_IMPORT, cwd=os.path.dirname(os.path.abspath(__file__)))
    assert_ledger(rows, KNOWN_FABRICATED_ROUTES, DECLARED_PURE_ROUTES)
