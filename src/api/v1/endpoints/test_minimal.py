"""Minimal test endpoint to debug POST issues."""

from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def get_test():
    """GET test."""
    return {"method": "GET", "message": "GET works!"}

@router.post("/")
async def post_test():
    """POST test."""
    return {"method": "POST", "message": "POST works!"}

@router.put("/")
async def put_test():
    """PUT test."""
    return {"method": "PUT", "message": "PUT works!"}

@router.delete("/")
async def delete_test():
    """DELETE test."""
    return {"method": "DELETE", "message": "DELETE works!"}