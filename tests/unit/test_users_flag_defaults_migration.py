"""The users-flag migration backfills NULL flags to the ORM defaults and gives each flag a server
default and NOT NULL, so a row written outside the ORM gets the defaults. Run against a throwaway
database (never the shared one), as the editions' own non-superuser role, with RLS forced on users
as it is on a real stack."""

import os
import subprocess
import uuid
from pathlib import Path

import psycopg2
import pytest
from psycopg2 import sql

COMMUNITY_DIR = Path(__file__).resolve().parents[2]
PREVIOUS_HEAD = "ia1te2na3nt4"
NEW_HEAD = "us1fl2ag3df4"
FLAGS = {"is_active": True, "is_superuser": False, "email_verified": False, "mfa_enabled": False}
ROLE = os.environ.get("POSTGRES_USER", "app")
ROLE_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")

# users as the earlier migrations left it: the four flags nullable, with no server default.
OLD_TABLE = """
CREATE TABLE users (
    id VARCHAR(36) PRIMARY KEY,
    email VARCHAR(254) NOT NULL UNIQUE,
    hashed_password VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(100),
    is_active BOOLEAN,
    is_superuser BOOLEAN,
    email_verified BOOLEAN,
    mfa_enabled BOOLEAN,
    token_version INTEGER NOT NULL DEFAULT 0
);
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE users FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation_users ON users FOR ALL
    USING (tenant_id::text = current_setting('app.current_tenant_id', true))
    WITH CHECK (tenant_id::text = current_setting('app.current_tenant_id', true));
CREATE POLICY admin_bypass_users ON users FOR ALL
    USING (current_setting('app.is_admin', true)::boolean = true);
"""


def _conn(dbname, user="postgres", password="postgres"):
    conn = psycopg2.connect(
        host=os.environ.get("POSTGRES_SERVER", "postgres"), port=int(os.environ.get("POSTGRES_PORT", "5432")),
        user=user, password=password, dbname=dbname)
    conn.autocommit = True
    return conn


def _alembic(db_name, *args):
    env = dict(os.environ)
    env["SQLALCHEMY_DATABASE_URI"] = (
        f"postgresql+asyncpg://{ROLE}:{ROLE_PASSWORD}@{os.environ.get('POSTGRES_SERVER', 'postgres')}"
        f":{os.environ.get('POSTGRES_PORT', '5432')}/{db_name}")
    return subprocess.run(["alembic", *args], cwd=COMMUNITY_DIR, env=env, capture_output=True, text=True)


@pytest.fixture
def throwaway_db():
    name = f"users_flags_{uuid.uuid4().hex[:10]}"
    admin = _conn("postgres")
    admin.cursor().execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    try:
        conn = _conn(name)
        cur = conn.cursor()
        cur.execute(OLD_TABLE)
        cur.execute(sql.SQL("ALTER TABLE users OWNER TO {}").format(sql.Identifier(ROLE)))
        cur.execute(sql.SQL("GRANT ALL ON SCHEMA public TO {}").format(sql.Identifier(ROLE)))
        cur.execute(
            "INSERT INTO users (id, email, hashed_password, tenant_id, is_active, is_superuser, email_verified, mfa_enabled) "
            "VALUES ('nulls', 'nulls@example.com', 'x', 't1', NULL, NULL, NULL, NULL), "
            "       ('set', 'set@example.com', 'x', 't2', false, true, true, true)")
        conn.close()
        yield name
    finally:
        admin.cursor().execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name)))
        admin.close()


def _flag_columns(cur):
    cur.execute("SELECT column_name, is_nullable, column_default FROM information_schema.columns "
                "WHERE table_name = 'users' AND column_name = ANY(%s)", (list(FLAGS),))
    return {r[0]: (r[1], r[2]) for r in cur.fetchall()}


def _flags(cur, user_id):
    cur.execute("SELECT is_active, is_superuser, email_verified, mfa_enabled FROM users WHERE id = %s", (user_id,))
    return dict(zip(FLAGS, cur.fetchone()))


def test_migration_backfills_and_a_raw_insert_gets_the_defaults(throwaway_db):
    assert _alembic(throwaway_db, "stamp", PREVIOUS_HEAD).returncode == 0

    up = _alembic(throwaway_db, "upgrade", NEW_HEAD)
    assert up.returncode == 0, up.stderr

    cur = _conn(throwaway_db).cursor()
    assert _flags(cur, "nulls") == FLAGS
    assert _flags(cur, "set") == {"is_active": False, "is_superuser": True, "email_verified": True, "mfa_enabled": True}
    for column, (nullable, default) in _flag_columns(cur).items():
        assert nullable == "NO", column
        assert default == str(FLAGS[column]).lower(), column

    cur.execute("INSERT INTO users (id, email, hashed_password) VALUES ('raw', 'raw@example.com', 'x')")
    assert _flags(cur, "raw") == FLAGS
    with pytest.raises(psycopg2.errors.NotNullViolation):
        cur.execute("INSERT INTO users (id, email, hashed_password, is_superuser) "
                    "VALUES ('null', 'null@example.com', 'x', NULL)")


def test_downgrade_drops_the_defaults_and_keeps_the_values(throwaway_db):
    assert _alembic(throwaway_db, "stamp", PREVIOUS_HEAD).returncode == 0
    assert _alembic(throwaway_db, "upgrade", NEW_HEAD).returncode == 0

    down = _alembic(throwaway_db, "downgrade", PREVIOUS_HEAD)
    assert down.returncode == 0, down.stderr

    cur = _conn(throwaway_db).cursor()
    assert _flag_columns(cur) == {column: ("YES", None) for column in FLAGS}
    assert _flags(cur, "nulls") == FLAGS


def test_the_model_declares_what_the_migration_makes():
    from models.user import User

    for column, value in FLAGS.items():
        col = User.__table__.c[column]
        assert col.nullable is False, column
        assert str(col.server_default.arg) == str(value).lower(), column
        assert col.default.arg is value, column
