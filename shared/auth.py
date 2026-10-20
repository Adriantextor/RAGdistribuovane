import secrets

from fastapi import Header, HTTPException

from shared.config import settings


def require_node(authorization: str = Header(default="")):
    expected = f"Bearer {settings().node_api_token.get_secret_value()}"
    if not secrets.compare_digest(authorization, expected):
        raise HTTPException(401, "Invalid node token")
