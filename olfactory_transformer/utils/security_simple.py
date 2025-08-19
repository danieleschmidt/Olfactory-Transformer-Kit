"""Simple security utilities for Olfactory Transformer."""

import re
import logging
from typing import Any, List, Dict, Optional

logger = logging.getLogger(__name__)


class SecurityValidator:
    """Simple security validation and input sanitization."""
    
    # Dangerous patterns to detect and block
    DANGEROUS_PATTERNS = [
        r'exec\s*\(',
        r'eval\s*\(',
        r'__import__\s*\(',
        r'open\s*\(',
        r'file\s*\(',
        r'subprocess',
        r'os\.system',
        r'\.\./',  # Path traversal
        r'\.\.',   # Path traversal
        r'[;&|`]',  # Command injection
        r'<script',  # XSS
        r'javascript:',  # XSS
        r'DROP\s+TABLE',  # SQL injection
        r'UNION\s+SELECT', # SQL injection
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


# Global security instance
security_validator = SecurityValidator()