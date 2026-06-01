#!/bin/bash

# AICtrlNet Community Edition Startup Script
# This script handles migrations and seed data loading automatically

echo "🚀 Starting AICtrlNet Community Edition initialization..."
echo "=============================================="

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL to be ready..."
until python -c "import psycopg2; psycopg2.connect(host='postgres', database='aictrlnet', user='postgres', password='postgres')" 2>/dev/null; do
  sleep 1
done
echo "✅ PostgreSQL is ready!"

# Wait for Redis to be ready
echo "⏳ Waiting for Redis to be ready..."
until python -c "import redis; r = redis.Redis(host='redis', port=6379); r.ping()" 2>/dev/null; do
  sleep 1
done
echo "✅ Redis is ready!"

# Run database migrations
# DISABLED: Migrations should be run via 'make migrate' or 'make fresh', not automatically on container start
# echo ""
# echo "📊 Running database migrations..."
# alembic upgrade head
# if [ $? -eq 0 ]; then
#     echo "✅ Migrations completed successfully!"
# else
#     echo "❌ Migration failed! Check logs for details."
#     # Don't exit - let's see what happens
# fi

# Load seed data
# DISABLED: Run manually with 'make seed' or 'make seed-community'
# echo ""
# echo "🌱 Loading seed data..."
# python scripts/seed_community_data.py
# if [ $? -eq 0 ]; then
#     echo "✅ Seed data loaded successfully!"
# else
#     echo "⚠️  Seed data may already exist or partially failed"
# fi

echo ""
echo "=============================================="
echo "✅ Initialization complete! Starting application..."
echo ""

# Start the FastAPI application
# Note: PYTHONPATH should include /app/src for production builds
# gunicorn manages UvicornWorker processes: a worker whose event loop wedges
# (blocking call) stops heart-beating and is killed+respawned after --timeout,
# so a single blocking path can't silently freeze the edition (see
# .claude/plans/worker-wedge-permanent-fix.md).
exec gunicorn main:app \
    -k uvicorn.workers.UvicornWorker \
    --workers ${WORKERS:-2} \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    --graceful-timeout 30 \
    --keep-alive 5 \
    --forwarded-allow-ips="*"