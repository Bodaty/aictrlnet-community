"""Basic Validation service for Community Edition.

Provides rule-based validation without database persistence.
Advanced features like custom validation rules and validation history
are available in Business Edition.
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from pathlib import Path
from email_validator import validate_email, EmailNotValidError

from core.config import settings


logger = logging.getLogger(__name__)


class ValidationService:
    """Basic validation service for Community Edition.
    
    Provides:
    - Common validation rules
    - Schema validation
    - Input sanitization
    
    Advanced features available in Business Edition:
    - Custom validation rules with database storage
    - Validation history and analytics
    - Complex multi-field validation
    - Async validation with external services
    - Validation templates
    """
    
    # Built-in validation patterns
    PATTERNS = {
        "email": re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
        "phone": re.compile(r'^\+?1?\d{9,15}$'),
        "url": re.compile(r'^https?://[^\s/$.?#].[^\s]*$'),
        "alphanumeric": re.compile(r'^[a-zA-Z0-9]+$'),
        "uuid": re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'),
        "date": re.compile(r'^\d{4}-\d{2}-\d{2}$'),
        "time": re.compile(r'^\d{2}:\d{2}(:\d{2})?$'),
        "datetime": re.compile(r'^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}'),
        "json": re.compile(r'^[\{\[].*[\}\]]$')
    }
    
    def __init__(self):
        """Initialize the basic validation service."""
        # Load custom rules from file
        self.rules_file = Path(settings.DATA_PATH) / "validation" / "rules.json"
        self.custom_rules = {}
        self._load_custom_rules()
        
        # Validation functions
        self._validators: Dict[str, Callable] = {
            "required": self._validate_required,
            "type": self._validate_type,
            "pattern": self._validate_pattern,
            "length": self._validate_length,
            "range": self._validate_range,
            "enum": self._validate_enum,
            "email": self._validate_email,
            "url": self._validate_url,
            "custom": self._validate_custom
        }
    
    def _load_custom_rules(self):
        """Load custom validation rules from file."""
        if self.rules_file.exists():
            try:
                self.custom_rules = json.loads(self.rules_file.read_text())
            except Exception as e:
                logger.warning(f"Failed to load custom rules: {e}")
        else:
            self.rules_file.parent.mkdir(parents=True, exist_ok=True)
            self._save_custom_rules()
    
    def _save_custom_rules(self):
        """Save custom rules to file."""
        try:
            self.rules_file.write_text(json.dumps(self.custom_rules, indent=2))
        except Exception as e:
            logger.error(f"Failed to save custom rules: {e}")
    
    async def validate(
        self,
        data: Any,
        rules: Dict[str, Any],
        field_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate data against rules."""
        errors = []
        warnings = []
        
        for rule_type, rule_config in rules.items():
            if rule_type in self._validators:
                result = self._validators[rule_type](data, rule_config, field_name)
                if not result["valid"]:
                    if result.get("warning"):
                        warnings.extend(result["errors"])
                    else:
                        errors.extend(result["errors"])
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "field": field_name
        }
    
    async def validate_schema(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate data against a schema."""
        all_errors = []
        all_warnings = []
        field_errors = {}
        
        # Validate each field
        for field_name, field_rules in schema.items():
            field_value = data.get(field_name)
            result = await self.validate(field_value, field_rules, field_name)
            
            if not result["valid"]:
                all_errors.extend(result["errors"])
                field_errors[field_name] = result["errors"]
            
            if result.get("warnings"):
                all_warnings.extend(result["warnings"])
        
        # Check for unexpected fields
        schema_fields = set(schema.keys())
        data_fields = set(data.keys())
        unexpected = data_fields - schema_fields
        
        if unexpected:
            all_warnings.append(f"Unexpected fields: {', '.join(unexpected)}")
        
        return {
            "valid": len(all_errors) == 0,
            "errors": all_errors,
            "warnings": all_warnings,
            "field_errors": field_errors
        }
    
    async def sanitize(
        self,
        data: Any,
        sanitization_rules: Dict[str, Any]
    ) -> Any:
        """Sanitize data according to rules."""
        if isinstance(data, str):
            sanitized = data
            
            # Trim whitespace
            if sanitization_rules.get("trim", True):
                sanitized = sanitized.strip()
            
            # Convert case
            if "case" in sanitization_rules:
                if sanitization_rules["case"] == "lower":
                    sanitized = sanitized.lower()
                elif sanitization_rules["case"] == "upper":
                    sanitized = sanitized.upper()
                elif sanitization_rules["case"] == "title":
                    sanitized = sanitized.title()
            
            # Remove HTML tags
            if sanitization_rules.get("strip_html", False):
                import html
                sanitized = re.sub(r'<[^>]+>', '', sanitized)
                sanitized = html.unescape(sanitized)
            
            # Escape special characters
            if sanitization_rules.get("escape", False):
                import html
                sanitized = html.escape(sanitized)
            
            # Max length
            if "max_length" in sanitization_rules:
                max_len = sanitization_rules["max_length"]
                if len(sanitized) > max_len:
                    sanitized = sanitized[:max_len]
            
            return sanitized
        
        elif isinstance(data, dict):
            # Recursively sanitize dict values
            return {
                key: await self.sanitize(value, sanitization_rules)
                for key, value in data.items()
            }
        
        elif isinstance(data, list):
            # Recursively sanitize list items
            return [
                await self.sanitize(item, sanitization_rules)
                for item in data
            ]
        
        return data
    
    # Validation functions
    
    def _validate_required(self, data: Any, config: Any, field_name: Optional[str]) -> Dict[str, Any]:
        """Validate required field."""
        is_required = config if isinstance(config, bool) else True
        
        if is_required and (data is None or data == ""):
            return {
                "valid": False,
                "errors": [f"{field_name or 'Field'} is required"]
            }
        return {"valid": True, "errors": []}
    
    def _validate_type(self, data: Any, expected_type: str, field_name: Optional[str]) -> Dict[str, Any]:
        """Validate data type."""
        if data is None:
            return {"valid": True, "errors": []}
        
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict
        }
        
        expected = type_map.get(expected_type)
        if expected and not isinstance(data, expected):
            return {
                "valid": False,
                "errors": [f"{field_name or 'Field'} must be of type {expected_type}"]
            }
        return {"valid": True, "errors": []}
    
    def _validate_pattern(self, data: Any, pattern: str, field_name: Optional[str]) -> Dict[str, Any]:
        """Validate against regex pattern."""
        if not isinstance(data, str):
            return {"valid": True, "errors": []}
        
        # Check built-in patterns
        if pattern in self.PATTERNS:
            regex = self.PATTERNS[pattern]
        else:
            try:
                regex = re.compile(pattern)
            except re.error:
                return {
                    "valid": False,
                    "errors": [f"Invalid pattern: {pattern}"]
                }
        
        if not regex.match(data):
            return {
                "valid": False,
                "errors": [f"{field_name or 'Field'} does not match pattern {pattern}"]
            }
        return {"valid": True, "errors": []}
    
    def _validate_length(self, data: Any, config: Dict[str, int], field_name: Optional[str]) -> Dict[str, Any]:
        """Validate length constraints."""
        if not isinstance(data, (str, list, dict)):
            return {"valid": True, "errors": []}
        
        length = len(data)
        errors = []
        
        if "min" in config and length < config["min"]:
            errors.append(f"{field_name or 'Field'} must have at least {config['min']} items/characters")
        
        if "max" in config and length > config["max"]:
            errors.append(f"{field_name or 'Field'} must have at most {config['max']} items/characters")
        
        if "exact" in config and length != config["exact"]:
            errors.append(f"{field_name or 'Field'} must have exactly {config['exact']} items/characters")
        
        return {"valid": len(errors) == 0, "errors": errors}
    
    def _validate_range(self, data: Any, config: Dict[str, Any], field_name: Optional[str]) -> Dict[str, Any]:
        """Validate numeric range."""
        if not isinstance(data, (int, float)):
            return {"valid": True, "errors": []}
        
        errors = []
        
        if "min" in config and data < config["min"]:
            errors.append(f"{field_name or 'Field'} must be at least {config['min']}")
        
        if "max" in config and data > config["max"]:
            errors.append(f"{field_name or 'Field'} must be at most {config['max']}")
        
        return {"valid": len(errors) == 0, "errors": errors}
    
    def _validate_enum(self, data: Any, allowed_values: List[Any], field_name: Optional[str]) -> Dict[str, Any]:
        """Validate against allowed values."""
        if data not in allowed_values:
            return {
                "valid": False,
                "errors": [f"{field_name or 'Field'} must be one of: {', '.join(map(str, allowed_values))}"]
            }
        return {"valid": True, "errors": []}
    
    def _validate_email(self, data: Any, config: Any, field_name: Optional[str]) -> Dict[str, Any]:
        """Validate email address."""
        if not isinstance(data, str):
            return {"valid": True, "errors": []}
        
        try:
            validate_email(data)
            return {"valid": True, "errors": []}
        except EmailNotValidError:
            return {
                "valid": False,
                "errors": [f"{field_name or 'Field'} is not a valid email address"]
            }
    
    def _validate_url(self, data: Any, config: Any, field_name: Optional[str]) -> Dict[str, Any]:
        """Validate URL."""
        if not isinstance(data, str):
            return {"valid": True, "errors": []}
        
        if not self.PATTERNS["url"].match(data):
            return {
                "valid": False,
                "errors": [f"{field_name or 'Field'} is not a valid URL"]
            }
        return {"valid": True, "errors": []}
    
    def _validate_custom(self, data: Any, rule_name: str, field_name: Optional[str]) -> Dict[str, Any]:
        """Validate against custom rule."""
        if rule_name not in self.custom_rules:
            return {
                "valid": False,
                "errors": [f"Unknown custom rule: {rule_name}"]
            }
        
        # In Community Edition, custom rules are simple patterns
        # Business Edition supports complex validation logic
        rule = self.custom_rules[rule_name]
        if "pattern" in rule:
            return self._validate_pattern(data, rule["pattern"], field_name)
        
        return {"valid": True, "errors": []}
    
    async def add_custom_rule(
        self,
        name: str,
        pattern: str,
        description: Optional[str] = None
    ) -> bool:
        """Add a custom validation rule (Community Edition: pattern-based only)."""
        try:
            # Test the pattern
            re.compile(pattern)
            
            self.custom_rules[name] = {
                "pattern": pattern,
                "description": description,
                "created_at": datetime.utcnow().isoformat()
            }
            
            self._save_custom_rules()
            return True
            
        except re.error:
            logger.error(f"Invalid regex pattern for rule {name}: {pattern}")
            return False
    
    async def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            "built_in_patterns": len(self.PATTERNS),
            "custom_rules": len(self.custom_rules),
            "validators": list(self._validators.keys()),
            "features": {
                "schema_validation": True,
                "input_sanitization": True,
                "custom_rules": "pattern-based",
                "validation_history": False,
                "async_validation": False
            },
            "upgrade_available": True,
            "upgrade_benefits": [
                "Database-stored custom rules",
                "Complex multi-field validation",
                "Validation history and analytics",
                "Async validation with external services",
                "Validation templates"
            ]
        }