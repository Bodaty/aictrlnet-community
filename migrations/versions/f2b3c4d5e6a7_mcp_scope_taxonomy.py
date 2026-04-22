"""mcp scope taxonomy — phase A

Rewrites legacy ``read:all`` / ``write:all`` scopes to the new
per-resource taxonomy. Leaves a ``scopes_legacy`` / ``allowed_scopes_legacy``
snapshot in place so Phase B can drop the dual-taxonomy acceptance
safely after 24h of observation.

Tables touched (all ``IF EXISTS`` — Business/Enterprise tables may not
be present in a Community-only deploy):

- ``api_keys.scopes`` (JSON list) -> expand legacy scopes
- ``oauth2_clients.allowed_scopes`` (ARRAY) -> expand legacy scopes +
  default-to-READ when empty
- ``oauth2_access_tokens.scopes`` (ARRAY) -> expand legacy scopes

The taxonomy map here is a *frozen copy* of
``mcp_server.scopes.LEGACY_SCOPE_MAP`` — migrations must not import
application code.

Revision ID: f2b3c4d5e6a7
Revises: e1a2b3c4d5f6
Create Date: 2026-04-22 13:20:00.000000
"""

import json

from alembic import op
import sqlalchemy as sa


revision = "f2b3c4d5e6a7"
down_revision = "e1a2b3c4d5f6"
branch_labels = None
depends_on = None


# Frozen copy of mcp_server/scopes.py — DO NOT sync later; any taxonomy
# additions after this migration ran are applied by their own migration.

_READ = [
    "read:workflows", "read:adapters", "read:templates", "read:policies",
    "read:approvals", "read:tasks", "read:conversations", "read:knowledge",
    "read:memory", "read:files", "read:analytics", "read:audit",
    "read:usage", "read:compliance", "read:institute", "read:subscription",
    "read:self_extending", "read:autonomy", "read:patterns", "read:org",
    "read:agents", "read:notifications", "read:llm", "read:fleet",
    "read:license",
]

_WRITE = [
    "write:workflows", "write:adapters", "write:self_extending",
    "write:browser", "write:approvals", "write:tasks", "write:conversations",
    "write:knowledge", "write:memory", "write:files", "write:messaging",
    "write:policies", "write:autonomy", "write:patterns", "write:org",
    "write:company", "write:agents", "write:notifications", "write:quality",
    "write:institute",
]

_LEGACY_MAP = {
    "read:all": _READ,
    "write:all": _WRITE,
    "read:workflows": ["read:workflows", "read:templates"],
    "write:workflows": ["write:workflows"],
}


def _expand(scopes):
    if not scopes:
        return []
    result: set[str] = set()
    for s in scopes:
        if s in _LEGACY_MAP:
            result.update(_LEGACY_MAP[s])
        elif s in _READ or s in _WRITE:
            result.add(s)
        # unknown scopes silently dropped
    return sorted(result)


def _column_exists(conn, table: str, column: str) -> bool:
    inspector = sa.inspect(conn)
    if table not in inspector.get_table_names():
        return False
    return column in {c["name"] for c in inspector.get_columns(table)}


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = set(inspector.get_table_names())

    # ------------ api_keys ------------
    if "api_keys" in tables:
        if not _column_exists(conn, "api_keys", "scopes_legacy"):
            op.add_column("api_keys", sa.Column("scopes_legacy", sa.JSON(), nullable=True))

        # Snapshot current scopes, then rewrite
        rows = conn.execute(sa.text("SELECT id, scopes FROM api_keys")).fetchall()
        for row_id, scopes in rows:
            # psql JSON column comes back as Python list/dict already
            if isinstance(scopes, str):
                try:
                    scopes = json.loads(scopes)
                except Exception:
                    scopes = []
            scopes = scopes or []
            expanded = _expand(scopes)
            conn.execute(
                sa.text(
                    "UPDATE api_keys SET scopes_legacy = :legacy, scopes = :new WHERE id = :id"
                ),
                {
                    "legacy": json.dumps(scopes),
                    "new": json.dumps(expanded),
                    "id": row_id,
                },
            )

    # ------------ oauth2_clients (Business edition) ------------
    if "oauth2_clients" in tables:
        if not _column_exists(conn, "oauth2_clients", "allowed_scopes_legacy"):
            op.add_column(
                "oauth2_clients",
                sa.Column(
                    "allowed_scopes_legacy",
                    sa.ARRAY(sa.String()),
                    nullable=True,
                ),
            )
        rows = conn.execute(
            sa.text("SELECT id, allowed_scopes FROM oauth2_clients")
        ).fetchall()
        for row_id, scopes in rows:
            scopes = list(scopes or [])
            expanded = _expand(scopes) or list(_READ)  # default to READ when empty
            conn.execute(
                sa.text(
                    "UPDATE oauth2_clients "
                    "SET allowed_scopes_legacy = :legacy, allowed_scopes = :new "
                    "WHERE id = :id"
                ),
                {"legacy": scopes, "new": expanded, "id": row_id},
            )

    # ------------ oauth2_access_tokens (Business edition) ------------
    if "oauth2_access_tokens" in tables:
        if not _column_exists(conn, "oauth2_access_tokens", "scopes_legacy"):
            op.add_column(
                "oauth2_access_tokens",
                sa.Column("scopes_legacy", sa.ARRAY(sa.String()), nullable=True),
            )
        rows = conn.execute(
            sa.text("SELECT id, scopes FROM oauth2_access_tokens WHERE revoked = false")
        ).fetchall()
        for row_id, scopes in rows:
            scopes = list(scopes or [])
            expanded = _expand(scopes)
            conn.execute(
                sa.text(
                    "UPDATE oauth2_access_tokens "
                    "SET scopes_legacy = :legacy, scopes = :new "
                    "WHERE id = :id"
                ),
                {"legacy": scopes, "new": expanded, "id": row_id},
            )


def downgrade() -> None:
    """Phase-A rollback: copy ``*_legacy`` columns back and drop them.

    Phase-B migration (once it ships) will drop the ``_legacy`` columns
    for real; until then this downgrade restores full pre-migration state.
    """
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = set(inspector.get_table_names())

    if "api_keys" in tables and _column_exists(conn, "api_keys", "scopes_legacy"):
        rows = conn.execute(
            sa.text("SELECT id, scopes_legacy FROM api_keys WHERE scopes_legacy IS NOT NULL")
        ).fetchall()
        for row_id, legacy in rows:
            conn.execute(
                sa.text("UPDATE api_keys SET scopes = :legacy WHERE id = :id"),
                {"legacy": legacy if isinstance(legacy, str) else json.dumps(legacy), "id": row_id},
            )
        op.drop_column("api_keys", "scopes_legacy")

    if "oauth2_clients" in tables and _column_exists(conn, "oauth2_clients", "allowed_scopes_legacy"):
        rows = conn.execute(
            sa.text(
                "SELECT id, allowed_scopes_legacy FROM oauth2_clients "
                "WHERE allowed_scopes_legacy IS NOT NULL"
            )
        ).fetchall()
        for row_id, legacy in rows:
            conn.execute(
                sa.text(
                    "UPDATE oauth2_clients SET allowed_scopes = :legacy WHERE id = :id"
                ),
                {"legacy": list(legacy or []), "id": row_id},
            )
        op.drop_column("oauth2_clients", "allowed_scopes_legacy")

    if "oauth2_access_tokens" in tables and _column_exists(conn, "oauth2_access_tokens", "scopes_legacy"):
        rows = conn.execute(
            sa.text(
                "SELECT id, scopes_legacy FROM oauth2_access_tokens "
                "WHERE scopes_legacy IS NOT NULL"
            )
        ).fetchall()
        for row_id, legacy in rows:
            conn.execute(
                sa.text(
                    "UPDATE oauth2_access_tokens SET scopes = :legacy WHERE id = :id"
                ),
                {"legacy": list(legacy or []), "id": row_id},
            )
        op.drop_column("oauth2_access_tokens", "scopes_legacy")
