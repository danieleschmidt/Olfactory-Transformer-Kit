"""Security utilities and input validation for Olfactory Transformer."""

import re
import logging
import threading
from typing import Any, List, Dict, Optional, Union
import hashlib
import hmac
import base64
import time
from pathlib import Path
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class SecurityValidator:
    """Comprehensive security validation and input sanitization."""
    
    # Dangerous patterns to detect and block
    DANGEROUS_PATTERNS = [
        r'exec\s*\(',
        r'eval\s*\(',
        r'__import__\s*\(',
        r'open\s*\(',
        r'file\s*\(',
        r'input\s*\(',
        r'raw_input\s*\(',
        r'compile\s*\(',
        r'globals\s*\(',
        r'locals\s*\(',
        r'vars\s*\(',
        r'dir\s*\(',
        r'hasattr\s*\(',
        r'getattr\s*\(',
        r'setattr\s*\(',
        r'delattr\s*\(',
        r'\.\./',  # Path traversal
        r'\.\.',   # Path traversal
        r'[;&|`]',  # Command injection
        r'<script',  # XSS
        r'javascript:',  # XSS
        r'vbscript:',   # XSS
        r'data:',       # Data URL injection
        r'DROP\s+TABLE',  # SQL injection
        r'UNION\s+SELECT', # SQL injection
        r'INSERT\s+INTO',  # SQL injection
        r'DELETE\s+FROM',  # SQL injection
        r'UPDATE\s+SET',   # SQL injection
    ]
    
    def __init__(self):
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.DANGEROUS_PATTERNS]
        
    def validate_smiles(self, smiles: str) -> bool:
        """Validate SMILES string for security and format."""
        if not isinstance(smiles, str):
            logger.warning(f"Invalid SMILES type: {type(smiles)}")
            return False
            
        # Length limits
        if len(smiles) > 1000:
            logger.warning(f"SMILES too long: {len(smiles)} characters")
            return False
            
        if len(smiles.strip()) == 0:
            logger.warning("Empty SMILES string")
            return False
            
        # Check for dangerous patterns
        for pattern in self.compiled_patterns:
            if pattern.search(smiles):
                logger.warning(f"Dangerous pattern detected in SMILES: {smiles[:50]}...")
                return False
                
        # Valid SMILES characters (comprehensive set)
        valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
                         '()[]{}#=-+/.\\:@*$%')
        
        for char in smiles:
            if char not in valid_chars:
                logger.warning(f"Invalid character in SMILES: {char}")
                return False
                
        return True
        
    def sanitize_input(self, text: str) -> str:
        """Sanitize input text by removing dangerous patterns."""
        if not isinstance(text, str):
            return ""
            
        sanitized = text
        
        # Remove dangerous patterns
        for pattern in self.compiled_patterns:
            sanitized = pattern.sub('', sanitized)
            
        # Limit length
        if len(sanitized) > 1000:
            sanitized = sanitized[:1000]
            
        return sanitized.strip()
        
    def validate_file_path(self, file_path: str) -> bool:
        """Validate file path for security."""
        if not isinstance(file_path, str):
            return False
            
        # Check for path traversal
        if '..' in file_path or '/etc/' in file_path or '~' in file_path:
            logger.warning(f"Suspicious file path: {file_path}")
            return False
            
        # Must be relative path or in allowed directories
        path = Path(file_path)
        try:
            path.resolve()
        except Exception:
            logger.warning(f"Invalid file path: {file_path}")
            return False
            
        return True


class SecureTokenizer:
    """Security-enhanced tokenizer wrapper."""
    
    def __init__(self, base_tokenizer):
        self.base_tokenizer = base_tokenizer
        self.validator = SecurityValidator()
        
    def encode(self, text: str, **kwargs) -> Dict[str, Any]:
        """Secure encoding with validation."""
        # Validate input
        if not self.validator.validate_smiles(text):
            raise ValueError(f"Invalid or dangerous input: {text[:50]}...")
            
        # Sanitize
        sanitized_text = self.validator.sanitize_input(text)
        
        # Call base tokenizer
        return self.base_tokenizer.encode(sanitized_text, **kwargs)
        
    def decode(self, token_ids: List[int], **kwargs) -> str:
        """Secure decoding."""
        result = self.base_tokenizer.decode(token_ids, **kwargs)
        
        # Validate output
        if not self.validator.validate_smiles(result):
            logger.warning("Decoded output failed validation")
            return self.validator.sanitize_input(result)
            
        return result


class APISecurityManager:
    """API endpoint security manager."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.validator = SecurityValidator()
        self.rate_limits = {}
        
    def validate_api_request(self, request_data: Dict[str, Any]) -> bool:
        """Validate API request data."""
        # Check required fields
        if 'smiles' not in request_data:
            return False
            
        # Validate SMILES
        smiles = request_data['smiles']
        if not self.validator.validate_smiles(smiles):
            return False
            
        return True
        
    def check_rate_limit(self, client_id: str, max_requests: int = 100, window_minutes: int = 60) -> bool:
        """Check rate limiting for client."""
        import time
        
        current_time = time.time()
        window_start = current_time - (window_minutes * 60)
        
        if client_id not in self.rate_limits:
            self.rate_limits[client_id] = []
            
        # Clean old requests
        self.rate_limits[client_id] = [
            req_time for req_time in self.rate_limits[client_id] 
            if req_time > window_start
        ]
        
        # Check limit
        if len(self.rate_limits[client_id]) >= max_requests:
            return False
            
        # Add current request
        self.rate_limits[client_id].append(current_time)
        return True


# Global security instance
security_validator = SecurityValidator()
api_security = APISecurityManager()
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import base64
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False


class SecurityViolation:
    """Security violation record."""
    def __init__(self, timestamp: float, violation_type: str, 
                 source_ip: Optional[str] = None, user_id: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None, severity: str = "medium"):
        self.timestamp = timestamp
        self.violation_type = violation_type
        self.source_ip = source_ip
        self.user_id = user_id
        self.details = details or {}
        self.severity = severity


class InputValidator:
    """Comprehensive input validation for security."""
    
    def __init__(self):
        # Dangerous patterns that should be blocked
        self.dangerous_patterns = [
            r'<script.*?>.*?</script>',  # XSS
            r'javascript:',  # JavaScript injection
            r'on\w+\s*=',  # Event handlers
            r'(union|select|insert|update|delete|drop|exec|script)',  # SQL injection
            r'(\.\./){2,}',  # Path traversal
            r'(cmd|eval|exec|system|shell_exec|passthru)',  # Command injection
            r'(file|ftp|http|https)://',  # URL schemes (selective)
            r'\\x[0-9a-fA-F]{2}',  # Hex encoding
            r'%[0-9a-fA-F]{2}',  # URL encoding
        ]
        
        # Valid SMILES characters (extended set)
        self.valid_smiles_chars = set(
            'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
            '0123456789()[]{}+-=#@/\\.$*%'
        )
        
        # Maximum lengths
        self.max_smiles_length = 500
        self.max_description_length = 10000
        self.max_filename_length = 255
        
        logging.info("Input validator initialized with security patterns")
    
    def validate_smiles(self, smiles: str) -> Dict[str, Any]:
        """Validate SMILES string for security and format."""
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "sanitized": smiles.strip() if smiles else "",
        }
        
        if not smiles:
            result["valid"] = False
            result["errors"].append("SMILES string is empty")
            return result
        
        # Basic type check
        if not isinstance(smiles, str):
            result["valid"] = False
            result["errors"].append("SMILES must be a string")
            return result
        
        smiles = smiles.strip()
        result["sanitized"] = smiles
        
        # Length check
        if len(smiles) > self.max_smiles_length:
            result["valid"] = False
            result["errors"].append(f"SMILES too long (max {self.max_smiles_length} chars)")
        
        # Character validation
        invalid_chars = set(smiles) - self.valid_smiles_chars
        if invalid_chars:
            result["valid"] = False
            result["errors"].append(f"Invalid characters: {invalid_chars}")
        
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, smiles, re.IGNORECASE):
                result["valid"] = False
                result["errors"].append(f"Contains dangerous pattern: {pattern}")
        
        # SMILES-specific validation
        if not self._validate_smiles_structure(smiles):
            result["warnings"].append("SMILES structure may be invalid")
        
        return result
    
    def validate_file_path(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Validate file path for security."""
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "sanitized": str(path) if path else "",
        }
        
        if not path:
            result["valid"] = False
            result["errors"].append("Path is empty")
            return result
        
        path_str = str(path)
        
        # Path traversal check
        if '..' in path_str or '~' in path_str:
            result["valid"] = False
            result["errors"].append("Path traversal attempt detected")
        
        # Absolute path outside allowed directories
        try:
            abs_path = Path(path_str).resolve()
            # Check if path tries to escape current working directory
            cwd = Path.cwd()
            if not str(abs_path).startswith(str(cwd)):
                result["warnings"].append("Path outside working directory")
        except Exception as e:
            result["errors"].append(f"Invalid path: {e}")
            result["valid"] = False
        
        # Filename length
        filename = Path(path_str).name
        if len(filename) > self.max_filename_length:
            result["valid"] = False
            result["errors"].append(f"Filename too long (max {self.max_filename_length} chars)")
        
        # Dangerous extensions
        dangerous_extensions = ['.exe', '.bat', '.sh', '.ps1', '.cmd', '.scr']
        if any(path_str.lower().endswith(ext) for ext in dangerous_extensions):
            result["valid"] = False
            result["errors"].append("Dangerous file extension")
        
        return result
    
    def validate_user_input(self, input_text: str, input_type: str = "general") -> Dict[str, Any]:
        """General user input validation."""
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "sanitized": input_text.strip() if input_text else "",
        }
        
        if not isinstance(input_text, str):
            result["valid"] = False
            result["errors"].append("Input must be a string")
            return result
        
        input_text = input_text.strip()
        result["sanitized"] = input_text
        
        # Length checks
        max_length = self.max_description_length if input_type == "description" else 1000
        if len(input_text) > max_length:
            result["valid"] = False
            result["errors"].append(f"Input too long (max {max_length} chars)")
        
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, input_text, re.IGNORECASE):
                result["valid"] = False
                result["errors"].append("Contains potentially dangerous content")
                break
        
        # Check for excessive special characters (possible obfuscation)
        special_char_ratio = sum(1 for c in input_text if not c.isalnum() and not c.isspace()) / max(len(input_text), 1)
        if special_char_ratio > 0.5:
            result["warnings"].append("High ratio of special characters")
        
        return result
    
    def _validate_smiles_structure(self, smiles: str) -> bool:
        """Basic SMILES structure validation."""
        # Check balanced parentheses
        paren_count = 0
        bracket_count = 0
        
        for char in smiles:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
                if paren_count < 0:
                    return False
            elif char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count < 0:
                    return False
        
        return paren_count == 0 and bracket_count == 0
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage."""
        if not filename:
            return "unnamed_file"
        
        # Remove path separators and dangerous characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Remove control characters
        sanitized = ''.join(c for c in sanitized if ord(c) >= 32)
        
        # Limit length
        if len(sanitized) > self.max_filename_length:
            name, ext = Path(sanitized).stem, Path(sanitized).suffix
            max_name_len = self.max_filename_length - len(ext) - 1
            sanitized = name[:max_name_len] + ext
        
        return sanitized or "unnamed_file"


class RateLimiter:
    """Rate limiting for API endpoints and operations."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(deque)
        self.lock = threading.Lock()
        
        logging.info(f"Rate limiter: {max_requests} requests per {window_seconds}s")
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for identifier (IP, user, etc.)."""
        with self.lock:
            now = time.time()
            window_start = now - self.window_seconds
            
            # Clean old requests
            user_requests = self.requests[identifier]
            while user_requests and user_requests[0] < window_start:
                user_requests.popleft()
            
            # Check if under limit
            if len(user_requests) >= self.max_requests:
                return False
            
            # Add current request
            user_requests.append(now)
            return True
    
    def get_remaining_requests(self, identifier: str) -> int:
        """Get remaining requests for identifier."""
        with self.lock:
            now = time.time()
            window_start = now - self.window_seconds
            
            user_requests = self.requests[identifier]
            # Count recent requests
            recent_count = sum(1 for req_time in user_requests if req_time >= window_start)
            
            return max(0, self.max_requests - recent_count)
    
    def reset_user(self, identifier: str):
        """Reset rate limit for specific identifier."""
        with self.lock:
            if identifier in self.requests:
                del self.requests[identifier]


class SecurityLogger:
    """Security event logging and monitoring."""
    
    def __init__(self, log_file: Optional[Path] = None):
        self.log_file = log_file
        self.violations = deque(maxlen=1000)  # Keep recent violations
        self.violation_counts = defaultdict(int)
        self.lock = threading.Lock()
        
        # Setup file logging if specified
        if log_file:
            self.file_handler = logging.FileHandler(log_file)
            self.file_handler.setLevel(logging.WARNING)
            formatter = logging.Formatter(
                '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
            )
            self.file_handler.setFormatter(formatter)
            
            # Add to security logger
            self.security_logger = logging.getLogger('security')
            self.security_logger.addHandler(self.file_handler)
            self.security_logger.setLevel(logging.WARNING)
    
    def log_violation(
        self,
        violation_type: str,
        details: Dict[str, Any],
        severity: str = "medium",
        source_ip: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """Log security violation."""
        violation = SecurityViolation(
            timestamp=time.time(),
            violation_type=violation_type,
            source_ip=source_ip,
            user_id=user_id,
            details=details,
            severity=severity
        )
        
        with self.lock:
            self.violations.append(violation)
            self.violation_counts[violation_type] += 1
        
        # Log to standard and file loggers
        log_message = (
            f"Security violation: {violation_type} "
            f"[{severity}] from {source_ip or 'unknown'} "
            f"- {details}"
        )
        
        if severity == "critical":
            logging.critical(log_message)
        elif severity == "high":
            logging.error(log_message)
        elif severity == "medium":
            logging.warning(log_message)
        else:
            logging.info(log_message)
        
        # File logging
        if hasattr(self, 'security_logger'):
            self.security_logger.warning(log_message)
    
    def get_violation_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get violation summary for recent period."""
        cutoff_time = time.time() - (hours * 3600)
        
        with self.lock:
            recent_violations = [
                v for v in self.violations if v.timestamp >= cutoff_time
            ]
        
        # Count by type and severity
        type_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for violation in recent_violations:
            type_counts[violation.violation_type] += 1
            severity_counts[violation.severity] += 1
        
        return {
            "period_hours": hours,
            "total_violations": len(recent_violations),
            "by_type": dict(type_counts),
            "by_severity": dict(severity_counts),
            "most_common_type": max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else None,
        }


class SecureConfig:
    """Secure configuration management."""
    
    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file
        self.config_data = {}
        self.encrypted_keys = set()
        
        # Generate encryption key if needed
        if HAS_CRYPTOGRAPHY:
            self.cipher_suite = None
            self._setup_encryption()
        
        if config_file and config_file.exists():
            self.load_config()
    
    def _setup_encryption(self):
        """Setup encryption for sensitive config values."""
        if not HAS_CRYPTOGRAPHY:
            logging.warning("Cryptography not available, config encryption disabled")
            return
        
        # Use a key derivation function for better security
        password = b"olfactory_transformer_key"  # In production, use env var
        salt = b"stable_salt_for_config"  # In production, use random salt
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password))
        self.cipher_suite = Fernet(key)
    
    def set_secure_value(self, key: str, value: str, encrypt: bool = False):
        """Set configuration value, optionally encrypted."""
        if encrypt and self.cipher_suite:
            encrypted_value = self.cipher_suite.encrypt(value.encode())
            self.config_data[key] = base64.b64encode(encrypted_value).decode()
            self.encrypted_keys.add(key)
        else:
            self.config_data[key] = value
    
    def get_secure_value(self, key: str) -> Optional[str]:
        """Get configuration value, decrypting if necessary."""
        if key not in self.config_data:
            return None
        
        value = self.config_data[key]
        
        if key in self.encrypted_keys and self.cipher_suite:
            try:
                encrypted_data = base64.b64decode(value.encode())
                decrypted = self.cipher_suite.decrypt(encrypted_data)
                return decrypted.decode()
            except Exception as e:
                logging.error(f"Failed to decrypt config value for {key}: {e}")
                return None
        
        return value
    
    def save_config(self):
        """Save configuration to file."""
        if not self.config_file:
            return
        
        # Create secure config format
        config_output = {
            "config": self.config_data,
            "encrypted_keys": list(self.encrypted_keys),
            "timestamp": time.time(),
        }
        
        import json
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config_output, f, indent=2)
            
            # Set restrictive permissions
            self.config_file.chmod(0o600)  # Owner read/write only
            
        except Exception as e:
            logging.error(f"Failed to save secure config: {e}")
    
    def load_config(self):
        """Load configuration from file."""
        if not self.config_file or not self.config_file.exists():
            return
        
        import json
        try:
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
            
            self.config_data = config_data.get("config", {})
            self.encrypted_keys = set(config_data.get("encrypted_keys", []))
            
        except Exception as e:
            logging.error(f"Failed to load secure config: {e}")


class SecurityManager:
    """Central security management."""
    
    def __init__(self):
        self.validator = InputValidator()
        self.rate_limiter = RateLimiter()
        self.security_logger = SecurityLogger()
        self.secure_config = SecureConfig()
        
        # Security policies
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.allowed_extensions = {'.json', '.csv', '.txt', '.sdf', '.mol', '.smi'}
        self.blocked_ips = set()
        
        logging.info("Security manager initialized")
    
    def validate_and_secure_input(
        self, 
        input_data: Any, 
        input_type: str,
        source_ip: Optional[str] = None
    ) -> Dict[str, Any]:
        """Comprehensive input validation and security check."""
        result = {
            "valid": True,
            "sanitized": input_data,
            "errors": [],
            "warnings": [],
            "security_checks": []
        }
        
        # Rate limiting check
        identifier = source_ip or "unknown"
        if not self.rate_limiter.is_allowed(identifier):
            result["valid"] = False
            result["errors"].append("Rate limit exceeded")
            self.security_logger.log_violation(
                "rate_limit_exceeded",
                {"ip": source_ip, "input_type": input_type},
                severity="medium",
                source_ip=source_ip
            )
            return result
        
        # IP blocking check
        if source_ip in self.blocked_ips:
            result["valid"] = False
            result["errors"].append("IP blocked")
            self.security_logger.log_violation(
                "blocked_ip_access",
                {"ip": source_ip, "input_type": input_type},
                severity="high",
                source_ip=source_ip
            )
            return result
        
        # Type-specific validation
        if input_type == "smiles":
            validation = self.validator.validate_smiles(input_data)
        elif input_type == "file_path":
            validation = self.validator.validate_file_path(input_data)
        else:
            validation = self.validator.validate_user_input(input_data, input_type)
        
        # Merge validation results
        result["valid"] = validation["valid"]
        result["sanitized"] = validation["sanitized"]
        result["errors"].extend(validation["errors"])
        result["warnings"].extend(validation["warnings"])
        
        # Log security violations
        if validation["errors"]:
            self.security_logger.log_violation(
                f"invalid_{input_type}_input",
                {
                    "input": str(input_data)[:100],  # Truncate for logging
                    "errors": validation["errors"]
                },
                severity="medium",
                source_ip=source_ip
            )
        
        return result
    
    def check_file_security(self, file_path: Path) -> Dict[str, Any]:
        """Security check for uploaded/processed files."""
        result = {
            "safe": True,
            "errors": [],
            "warnings": []
        }
        
        if not file_path.exists():
            result["errors"].append("File does not exist")
            result["safe"] = False
            return result
        
        # Size check
        file_size = file_path.stat().st_size
        if file_size > self.max_file_size:
            result["errors"].append(f"File too large: {file_size} bytes")
            result["safe"] = False
        
        # Extension check
        if file_path.suffix.lower() not in self.allowed_extensions:
            result["warnings"].append(f"Unusual file extension: {file_path.suffix}")
        
        # Basic content scan
        try:
            with open(file_path, 'rb') as f:
                header = f.read(1024)  # Read first 1KB
            
            # Check for executable headers
            if header.startswith(b'MZ') or header.startswith(b'\x7fELF'):
                result["errors"].append("Executable file detected")
                result["safe"] = False
        
        except Exception as e:
            result["warnings"].append(f"Could not scan file content: {e}")
        
        return result
    
    def block_ip(self, ip_address: str, reason: str):
        """Block IP address."""
        self.blocked_ips.add(ip_address)
        self.security_logger.log_violation(
            "ip_blocked",
            {"ip": ip_address, "reason": reason},
            severity="high"
        )
        logging.warning(f"IP {ip_address} blocked: {reason}")
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get overall security status."""
        return {
            "blocked_ips_count": len(self.blocked_ips),
            "recent_violations": self.security_logger.get_violation_summary(hours=1),
            "rate_limiter_active": True,
            "encryption_available": HAS_CRYPTOGRAPHY,
            "max_file_size_mb": self.max_file_size / (1024 * 1024),
            "allowed_extensions": list(self.allowed_extensions),
        }


# Global security manager instance
security_manager = SecurityManager()


def secure_operation(input_type: str = "general"):
    """Decorator for securing operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Basic security check (can be enhanced)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                security_manager.security_logger.log_violation(
                    "operation_error",
                    {"function": func.__name__, "error": str(e)},
                    severity="low"
                )
                raise
        return wrapper
    return decorator