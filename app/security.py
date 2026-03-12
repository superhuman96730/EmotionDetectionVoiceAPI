"""
API authentication and security module
"""

from datetime import datetime, timedelta
from typing import Optional
import jwt
from functools import wraps

class TokenManager:
    """Manage JWT authentication tokens"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_expiry = timedelta(hours=24)
    
    def create_token(self, user_id: str, data: dict = None) -> str:
        """Create JWT token"""
        payload = {
            "user_id": user_id,
            "exp": datetime.utcnow() + self.token_expiry,
            "iat": datetime.utcnow()
        }
        if data:
            payload.update(data)
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token
    
    def verify_token(self, token: str) -> dict:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")

class APIKeyManager:
    """Manage API keys for authentication"""
    
    def __init__(self):
        self.keys = {}
    
    def generate_key(self, client_id: str) -> str:
        """Generate new API key"""
        import secrets
        key = secrets.token_urlsafe(32)
        self.keys[key] = client_id
        return key
    
    def validate_key(self, key: str) -> bool:
        """Validate API key"""
        return key in self.keys
