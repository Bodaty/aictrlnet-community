#!/usr/bin/env python3
"""Test WebSocket functionality."""

import asyncio
import json
import websockets
import sys
from datetime import datetime


async def test_websocket_connection(port: int = 8010, edition: str = "community"):
    """Test WebSocket connection and messaging."""
    uri = f"ws://localhost:{port}/api/v1/ws?token=dev-token-for-testing&edition={edition}"
    
    print(f"\n🔌 Testing WebSocket connection to {edition} edition on port {port}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            # Wait for connection message
            message = await websocket.recv()
            data = json.loads(message)
            print(f"✅ Connected: {data['message']}")
            
            # Test ping/pong
            print("\n📡 Testing ping/pong...")
            await websocket.send(json.dumps({
                "type": "ping"
            }))
            
            response = await websocket.recv()
            data = json.loads(response)
            if data["type"] == "pong":
                print("✅ Ping/pong successful")
            
            # Test subscription
            print("\n📨 Testing event subscription...")
            await websocket.send(json.dumps({
                "type": "subscribe",
                "events": ["task_created", "workflow_updated", "approval_required"]
            }))
            
            response = await websocket.recv()
            data = json.loads(response)
            if data["type"] == "subscribed":
                print(f"✅ Subscribed to events: {data['events']}")
            
            # Test broadcast (will fail for non-admin)
            print("\n📢 Testing broadcast (expected to fail for regular user)...")
            await websocket.send(json.dumps({
                "type": "broadcast",
                "message": "Hello everyone!"
            }))
            
            response = await websocket.recv()
            data = json.loads(response)
            if data["type"] == "error":
                print(f"✅ Broadcast correctly rejected: {data['message']}")
            
            # Keep connection open for a bit to receive any events
            print("\n⏳ Listening for events for 5 seconds...")
            try:
                await asyncio.wait_for(websocket.recv(), timeout=5.0)
            except asyncio.TimeoutError:
                print("✅ No events received (expected in test)")
            
            print("\n✅ WebSocket test completed successfully!")
            
    except websockets.exceptions.WebSocketException as e:
        print(f"❌ WebSocket error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    return True


async def test_multiple_connections():
    """Test multiple simultaneous WebSocket connections."""
    print("\n🔌 Testing multiple simultaneous connections...")
    
    tasks = []
    editions = [
        (8010, "community"),
        (8001, "business"),
        (8002, "enterprise")
    ]
    
    for port, edition in editions:
        task = asyncio.create_task(test_websocket_connection(port, edition))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    success_count = sum(1 for r in results if r is True)
    print(f"\n📊 Summary: {success_count}/{len(editions)} connections successful")


async def main():
    """Main test function."""
    print("🧪 AICtrlNet FastAPI WebSocket Test Suite")
    print("=" * 50)
    
    # Test single connection
    print("\n1️⃣ Testing single WebSocket connection...")
    success = await test_websocket_connection()
    
    if success:
        # Test multiple connections
        print("\n2️⃣ Testing multiple simultaneous connections...")
        await test_multiple_connections()
    
    print("\n✅ All WebSocket tests completed!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)