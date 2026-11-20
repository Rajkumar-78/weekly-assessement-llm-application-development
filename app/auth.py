"""x-api-key authentication: 401 = missing, 403 = wrong. Constant-time comparison."""
import secrets
from typing import Optional

from fastapi import Depends, Header, HTTPException

from .config import Settings, get_settings


async def verify_api_key(x_api_key: Optional[str] = Header(None),
                         settings: Settings = Depends(get_settings)) -> None:
    if not settings.api_key:
        raise HTTPException(500, "Server API key is not configured")
    if not x_api_key:
        raise HTTPException(401, "Missing x-api-key header", headers={"WWW-Authenticate": "ApiKey"})
    if not secrets.compare_digest(x_api_key.encode(), settings.api_key.encode()):
        raise HTTPException(403, "Invalid API key")
