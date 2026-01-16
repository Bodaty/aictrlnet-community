#!/bin/bash

echo "🚀 Starting AICtrlNet FastAPI Proof of Concept"
echo "=============================================="

# Build the Docker image
echo ""
echo "📦 Building Docker image..."
docker-compose build

# Start the services
echo ""
echo "🔧 Starting services..."
docker-compose up -d

# Wait for services to be ready
echo ""
echo "⏳ Waiting for services to start..."
sleep 10

# Check health
echo ""
echo "🏥 Checking service health..."
for port in 8010 8001 8002; do
    if curl -s http://localhost:$port/health > /dev/null; then
        echo "✅ Service on port $port is healthy"
    else
        echo "❌ Service on port $port is not responding"
    fi
done

echo ""
echo "📚 API Documentation available at:"
echo "   Community: http://localhost:8010/api/v1/docs"
echo "   Business:  http://localhost:8001/api/v1/docs"
echo "   Enterprise: http://localhost:8002/api/v1/docs"

echo ""
echo "🧪 To run tests:"
echo "   python test_fastapi.py"

echo ""
echo "🛑 To stop services:"
echo "   docker-compose down"

echo ""
echo "=============================================="
echo "✅ FastAPI services are ready!"