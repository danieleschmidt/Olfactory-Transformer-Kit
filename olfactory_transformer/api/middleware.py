"""Security and rate limiting middleware for the API."""

import time
import logging
from typing import Dict, Optional
from collections import defaultdict, deque
import hashlib
import hmac

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..utils.security import security_manager


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for API protection."""
    
    def __init__(self, app, secret_key: Optional[str] = None):
        super().__init__(app)
        self.secret_key = secret_key or "secure-olfactory-api-key"
        self.blocked_ips = set()
        self.suspicious_patterns = {
            "sql_injection": ["'", "union", "select", "--", "/*"],
            "xss": ["<script", "javascript:", "onerror=", "onload="],
            "path_traversal": ["../", "..\\", "%2e%2e"],
            "command_injection": [";", "|", "&", "`", "$"]
        }
    
    async def dispatch(self, request: Request, call_next):
        """Process request through security filters."""
        client_ip = self._get_client_ip(request)
        
        # Check blocked IPs
        if client_ip in self.blocked_ips:
            logging.warning(f"Blocked request from IP: {client_ip}")
            return JSONResponse(
                status_code=403,
                content={"error": "Access denied"}
            )
        
        # Validate request
        if not await self._validate_request(request):
            self._mark_suspicious(client_ip)
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid request"}
            )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Add security headers
            self._add_security_headers(response)
            
            return response
            
        except Exception as e:
            logging.error(f"Request processing failed: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"}
            )
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    async def _validate_request(self, request: Request) -> bool:
        """Validate request for security threats."""
        try:
            # Check request size
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB limit
                logging.warning(f"Request too large: {content_length}")
                return False
            
            # Check for suspicious patterns in URL
            url_path = str(request.url.path).lower()
            query_params = str(request.url.query).lower()
            
            for category, patterns in self.suspicious_patterns.items():
                for pattern in patterns:
                    if pattern in url_path or pattern in query_params:
                        logging.warning(f"Suspicious {category} pattern detected: {pattern}")
                        return False
            
            # Validate content type for POST requests
            if request.method == "POST":
                content_type = request.headers.get("content-type", "")
                if not content_type.startswith(("application/json", "application/x-www-form-urlencoded")):
                    logging.warning(f"Invalid content type: {content_type}")
                    return False
            
            # Check for request body patterns
            if request.method in ["POST", "PUT", "PATCH"]:
                try:
                    body = await request.body()
                    if body:
                        body_str = body.decode("utf-8", errors="ignore").lower()
                        
                        for category, patterns in self.suspicious_patterns.items():
                            for pattern in patterns:
                                if pattern in body_str:
                                    logging.warning(f"Suspicious {category} pattern in body: {pattern}")
                                    return False
                except Exception:
                    # If body can't be read, continue with request
                    pass
            
            return True
            
        except Exception as e:
            logging.error(f"Request validation failed: {e}")
            return False
    
    def _mark_suspicious(self, ip: str) -> None:
        """Mark IP as suspicious and potentially block."""
        # Simple implementation - in production, use Redis or database
        if not hasattr(self, '_suspicious_ips'):
            self._suspicious_ips = defaultdict(int)
        
        self._suspicious_ips[ip] += 1
        
        # Block after 5 suspicious requests
        if self._suspicious_ips[ip] >= 5:
            self.blocked_ips.add(ip)
            logging.warning(f"Blocked suspicious IP: {ip}")
    
    def _add_security_headers(self, response: Response) -> None:
        """Add security headers to response."""
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""
    
    def __init__(self, app, requests_per_minute: int = 60, burst_limit: int = 10):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.rate_limit_window = 60  # 1 minute
        
        # Store request history per IP
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque())
        self.burst_history: Dict[str, deque] = defaultdict(lambda: deque())
    
    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting."""
        client_ip = self._get_client_ip(request)
        current_time = time.time()
        
        # Check rate limits
        if not self._check_rate_limit(client_ip, current_time):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "retry_after": 60
                },
                headers={"Retry-After": "60"}
            )
        
        # Check burst limit
        if not self._check_burst_limit(client_ip, current_time):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Burst limit exceeded",
                    "retry_after": 10
                },
                headers={"Retry-After": "10"}
            )
        
        # Record request
        self._record_request(client_ip, current_time)
        
        response = await call_next(request)
        
        # Add rate limit headers
        self._add_rate_limit_headers(response, client_ip, current_time)
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
    
    def _check_rate_limit(self, ip: str, current_time: float) -> bool:
        """Check if IP is within rate limits."""
        history = self.request_history[ip]
        
        # Remove old requests
        while history and current_time - history[0] > self.rate_limit_window:
            history.popleft()
        
        return len(history) < self.requests_per_minute
    
    def _check_burst_limit(self, ip: str, current_time: float) -> bool:
        """Check burst limit (last 10 seconds)."""
        burst_history = self.burst_history[ip]
        burst_window = 10  # 10 seconds
        
        # Remove old requests
        while burst_history and current_time - burst_history[0] > burst_window:
            burst_history.popleft()
        
        return len(burst_history) < self.burst_limit
    
    def _record_request(self, ip: str, current_time: float) -> None:
        """Record request timestamp."""
        self.request_history[ip].append(current_time)
        self.burst_history[ip].append(current_time)
    
    def _add_rate_limit_headers(self, response: Response, ip: str, current_time: float) -> None:
        """Add rate limit information to response headers."""
        history = self.request_history[ip]
        remaining = max(0, self.requests_per_minute - len(history))
        
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(current_time + self.rate_limit_window))


class APIKeyMiddleware(BaseHTTPMiddleware):
    """API key authentication middleware."""
    
    def __init__(self, app, api_keys: Optional[Dict[str, Dict]] = None):
        super().__init__(app)
        self.api_keys = api_keys or {}
        self.public_endpoints = {"/health", "/docs", "/redoc", "/openapi.json"}
    
    async def dispatch(self, request: Request, call_next):
        """Validate API key for protected endpoints."""
        path = request.url.path
        
        # Skip authentication for public endpoints
        if path in self.public_endpoints or path.startswith("/docs"):
            return await call_next(request)
        
        # Extract API key
        api_key = self._extract_api_key(request)
        
        if not api_key:
            return JSONResponse(
                status_code=401,
                content={"error": "API key required"}
            )
        
        # Validate API key
        key_info = self.api_keys.get(api_key)
        if not key_info:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid API key"}
            )
        
        # Check key permissions
        if not self._check_permissions(key_info, request):
            return JSONResponse(
                status_code=403,
                content={"error": "Insufficient permissions"}
            )
        
        # Add key info to request state
        request.state.api_key_info = key_info
        
        return await call_next(request)
    
    def _extract_api_key(self, request: Request) -> Optional[str]:
        """Extract API key from request."""
        # Check Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:]
        
        # Check X-API-Key header
        api_key_header = request.headers.get("X-API-Key")
        if api_key_header:
            return api_key_header
        
        # Check query parameter
        return request.query_params.get("api_key")
    
    def _check_permissions(self, key_info: Dict, request: Request) -> bool:
        """Check if API key has required permissions."""
        # Simple permission checking
        permissions = key_info.get("permissions", [])
        
        if "admin" in permissions:
            return True
        
        method = request.method.lower()
        path = request.url.path
        
        # Map endpoints to required permissions
        endpoint_permissions = {
            "predict": ["read", "predict"],
            "stream": ["read", "stream"],
            "batch": ["read", "batch"],
            "models": ["read"],
            "metrics": ["admin", "metrics"]
        }
        
        for endpoint, required_perms in endpoint_permissions.items():
            if endpoint in path:
                return any(perm in permissions for perm in required_perms)
        
        return False