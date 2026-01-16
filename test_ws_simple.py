#!/usr/bin/env python3
"""Simple WebSocket test."""

import asyncio
import websockets
import json


async def test():
    """Test WebSocket connection."""
    uri = "ws://localhost:8010/api/v1/ws?token=dev-token-for-testing&edition=community"
    
    try:
        async with websockets.connect(uri) as websocket:
            # Wait for connection message
            message = await websocket.recv()
            data = json.loads(message)
            print(f"Connected: {data['message']}")
            
            # Send ping
            await websocket.send(json.dumps({"type": "ping"}))
            
            # Wait for pong
            response = await websocket.recv()
            data = json.loads(response)
            print(f"Received: {data['type']}")
            
            print("WebSocket test successful!")
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(test())