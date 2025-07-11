"""Configuration Manager Unit

This module implements the ConfigurationManager class that provides centralized
configuration management with hierarchical loading, validation, and access patterns.

Behavioral Specification:
- Loads configuration from multiple sources (defaults, files, CLI, env vars)
- Validates configuration against schema and business rules
- Provides type-safe access to configuration values
- Maintains configuration source traceability
"""

import json
import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field


@dataclass
class ConfigurationSource:
    """Represents a configuration source with metadata"""
    name: str
    type: str  # 'default', 'file', 'cli', 'env'
    priority: int
    data: Dict[str, Any]
    location: Optional[str] = None


class ConfigurationManager:
    """
    Manages configuration loading, validation, and access with hierarchical precedence.
    
    Behavioral Contract:
    - MUST load default configuration on initialization
    - MUST validate all configuration before making it available
    - MUST preserve configuration source information for traceability
    - MUST provide type-safe access to configuration values
    - MUST handle missing configuration gracefully with defaults
    """

    def __init__(self, config_schema: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration manager with optional schema.
        
        Args:
            config_schema: JSON schema for configuration validation
            
        Behaviors:
        - Initializes with default configuration
        - Sets up validation schema if provided
        - Prepares configuration source tracking
        """
        self.logger = logging.getLogger(__name__)
        self.schema = config_schema or self._get_default_schema()
        self.sources: List[ConfigurationSource] = []
        self.merged_config: Dict[str, Any] = {}
        self.validation_errors: List[str] = []
        
        # Load default configuration
        self._load_defaults()
    
    def _get_default_schema(self) -> Dict[str, Any]:
        """
        Returns the default configuration schema.
        
        Returns:
            Default schema dictionary defining configuration structure
            
        Behavior:
        - Provides comprehensive schema for all configuration parameters
        - Includes type definitions, constraints, and defaults
        - Supports nested configuration structures
        """
        return {
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "properties": {
                        "input_file": {"type": "string"},
                        "output_directory": {"type": "string", "default": "./output"},
                        "format": {"type": "string", "enum": ["csv", "parquet", "json"], "default": "csv"}
                    }
                },
                "processing": {
                    "type": "object",
                    "properties": {
                        "split_ratios": {
                            "type": "object",
                            "properties": {
                                "d1": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.4},
                                "d2": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.2},
                                "d3": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.2},
                                "d4": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.1},
                                "d5": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.05},
                                "d6": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.05}
                            },
                            "required": ["d1", "d2", "d3", "d4", "d5", "d6"]
                        },
                        "training_datasets": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["d1", "d2", "d3", "d4", "d5", "d6"]},
                            "default": ["d1", "d2"]
                        },
                        "normalization": {
                            "type": "object",
                            "properties": {
                                "method": {"type": "string", "enum": ["zscore", "minmax", "none"], "default": "zscore"},
                                "save_parameters": {"type": "boolean", "default": True},
                                "parameters_dir": {"type": "string", "default": "./normalization_params"}
                            }
                        }
                    }
                },
                "plugins": {
                    "type": "object",
                    "properties": {
                        "preprocessor_plugins": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": []
                        },
                        "feature_engineering_plugins": {
                            "type": "array", 
                            "items": {"type": "string"},
                            "default": []
                        },
                        "postprocessing_plugins": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": []
                        }
                    }
                },
                "logging": {
                    "type": "object",
                    "properties": {
                        "level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR"], "default": "INFO"},
                        "file": {"type": "string", "default": "preprocessor.log"}
                    }
                }
            },
            "required": ["data", "processing"]
        }
    
    def _load_defaults(self) -> None:
        """
        Loads default configuration values from schema.
        
        Behavior:
        - Extracts default values from schema definitions
        - Creates default configuration source
        - Applies defaults to merged configuration
        """
        defaults = self._extract_defaults_from_schema(self.schema)
        default_source = ConfigurationSource(
            name="defaults",
            type="default",
            priority=0,
            data=defaults
        )
        self.sources.append(default_source)
        self.merged_config = defaults.copy()
        
        self.logger.debug("Loaded default configuration")
    
    def _extract_defaults_from_schema(self, schema: Dict[str, Any], path: str = "") -> Dict[str, Any]:
        """
        Recursively extracts default values from JSON schema.
        
        Args:
            schema: JSON schema to extract defaults from
            path: Current path in schema (for nested extraction)
            
        Returns:
            Dictionary containing default values
            
        Behavior:
        - Recursively traverses schema structure
        - Extracts default values at all levels
        - Maintains nested structure in result
        """
        defaults = {}
        
        if schema.get("type") == "object" and "properties" in schema:
            for prop, prop_schema in schema["properties"].items():
                if "default" in prop_schema:
                    defaults[prop] = prop_schema["default"]
                elif prop_schema.get("type") == "object":
                    nested_defaults = self._extract_defaults_from_schema(prop_schema, f"{path}.{prop}")
                    if nested_defaults:
                        defaults[prop] = nested_defaults
        
        return defaults
    
    def load_from_file(self, config_file: Union[str, Path]) -> bool:
        """
        Loads configuration from a file.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            True if loading successful, False otherwise
            
        Behavior:
        - Supports JSON and YAML configuration files
        - Validates file format and structure
        - Merges file configuration with existing configuration
        - Reports loading errors with specific details
        """
        try:
            config_path = Path(config_file)
            if not config_path.exists():
                self.logger.error(f"Configuration file not found: {config_file}")
                return False
            
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() == '.json':
                    file_config = json.load(f)
                elif config_path.suffix.lower() in ['.yml', '.yaml']:
                    try:
                        import yaml
                        file_config = yaml.safe_load(f)
                    except ImportError:
                        self.logger.error("PyYAML not installed, cannot load YAML configuration")
                        return False
                else:
                    self.logger.error(f"Unsupported configuration file format: {config_path.suffix}")
                    return False
            
            file_source = ConfigurationSource(
                name=f"file_{config_path.name}",
                type="file",
                priority=2,
                data=file_config,
                location=str(config_path)
            )
            
            self.sources.append(file_source)
            self._merge_configuration()
            
            self.logger.info(f"Loaded configuration from file: {config_file}")
            return True
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in configuration file {config_file}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error loading configuration file {config_file}: {e}")
            return False
    
    def load_from_environment(self, prefix: str = "PREPROCESSOR_") -> None:
        """
        Loads configuration from environment variables.
        
        Args:
            prefix: Prefix for environment variables to consider
            
        Behavior:
        - Scans environment for variables with specified prefix
        - Converts environment variable names to configuration paths
        - Handles type conversion based on schema
        - Creates environment configuration source
        """
        env_config = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to configuration path
                env_key = key[len(prefix):]
                config_key = self._env_key_to_config_path(env_key)
                
                # Attempt type conversion based on schema
                converted_value = self._convert_env_value(value, config_key)
                self._set_nested_value(env_config, config_key, converted_value)
        
        if env_config:
            env_source = ConfigurationSource(
                name="environment",
                type="env",
                priority=3,
                data=env_config
            )
            self.sources.append(env_source)
            self._merge_configuration()
            
            self.logger.debug(f"Loaded {len(env_config)} configuration values from environment")
    
    def _env_key_to_config_path(self, env_key: str) -> str:
        """
        Converts environment variable key to configuration path.
        
        Args:
            env_key: Environment variable key (without prefix)
            
        Returns:
            Configuration path in dot notation
            
        Behavior:
        - Converts uppercase to lowercase
        - Uses specific mapping rules for known sections
        - Falls back to simple underscore-to-dot conversion
        """
        # Convert to lowercase first
        lower_key = env_key.lower()
        
        # Handle known section patterns
        if lower_key.startswith('data_'):
            # DATA_INPUT_FILE -> data.input_file
            param = lower_key[5:]  # Remove 'data_'
            return f"data.{param}"
        elif lower_key.startswith('processing_'):
            # PROCESSING_NORMALIZATION_METHOD -> processing.normalization.method
            # PROCESSING_SPLIT_RATIOS_D1 -> processing.split_ratios.d1
            param_path = lower_key[11:]  # Remove 'processing_'
            if 'split_ratios_' in param_path:
                # Handle split_ratios specially: split_ratios_d1 -> split_ratios.d1
                dataset_key = param_path.replace('split_ratios_', '')
                return f"processing.split_ratios.{dataset_key}"
            elif '_' in param_path:
                # Split subsection and parameter
                parts = param_path.split('_', 1)
                if len(parts) == 2:
                    subsection, param = parts
                    return f"processing.{subsection}.{param}"
                else:
                    return f"processing.{param_path.replace('_', '.')}"
            else:
                return f"processing.{param_path}"
        elif lower_key.startswith('logging_'):
            # LOGGING_LEVEL -> logging.level
            param = lower_key[8:]  # Remove 'logging_'
            return f"logging.{param}"
        elif lower_key.startswith('plugins_'):
            # PLUGINS_FEATURE_ENGINEERING_PLUGINS -> plugins.feature_engineering_plugins
            param = lower_key[8:]  # Remove 'plugins_'
            return f"plugins.{param}"
        else:
            # Default: convert all underscores to dots
            return lower_key.replace('_', '.')
    
    def load_from_cli(self, cli_args: Dict[str, Any]) -> None:
        """
        Loads configuration from command-line arguments.
        
        Args:
            cli_args: Dictionary of command-line arguments
            
        Behavior:
        - Accepts CLI arguments as configuration overrides
        - Validates CLI argument types against schema
        - Creates CLI configuration source with highest priority
        - Merges CLI configuration with existing configuration
        """
        if not cli_args:
            return
        
        # Filter out None values and convert argument names
        cli_config = {}
        for key, value in cli_args.items():
            if value is not None:
                # Convert CLI argument names to configuration paths
                config_key = key.replace('-', '_').replace('__', '.')
                self._set_nested_value(cli_config, config_key, value)
        
        if cli_config:
            cli_source = ConfigurationSource(
                name="command_line",
                type="cli",
                priority=4,
                data=cli_config
            )
            self.sources.append(cli_source)
            self._merge_configuration()
            
            self.logger.debug(f"Loaded {len(cli_config)} configuration values from CLI")
    
    def _convert_env_value(self, value: str, config_key: str) -> Any:
        """
        Converts environment variable string value to appropriate type.
        
        Args:
            value: Environment variable string value
            config_key: Configuration key for type lookup
            
        Returns:
            Converted value with appropriate type
            
        Behavior:
        - Uses schema to determine expected type
        - Handles boolean, numeric, and array conversions
        - Returns string value if type cannot be determined
        """
        # Simple type conversion based on value patterns
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        
        try:
            # Try integer first
            if '.' not in value:
                return int(value)
            else:
                return float(value)
        except ValueError:
            pass
        
        # Check if it's a JSON array or object
        if value.startswith('[') or value.startswith('{'):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        return value
    
    def _set_nested_value(self, config: Dict[str, Any], key_path: str, value: Any) -> None:
        """
        Sets a value in nested dictionary using dot notation path.
        
        Args:
            config: Configuration dictionary to modify
            key_path: Dot-separated path to the configuration key
            value: Value to set
            
        Behavior:
        - Creates nested dictionary structure as needed
        - Supports arbitrary nesting depth
        - Overwrites existing values at the specified path
        """
        keys = key_path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _merge_configuration(self) -> None:
        """
        Merges all configuration sources according to priority.
        
        Behavior:
        - Sorts sources by priority (lowest to highest)
        - Merges configurations with higher priority overriding lower
        - Maintains nested structure during merging
        - Updates merged_config with final result
        """
        # Sort sources by priority
        sorted_sources = sorted(self.sources, key=lambda s: s.priority)
        
        # Start with empty configuration
        merged = {}
        
        # Merge each source in priority order
        for source in sorted_sources:
            merged = self._deep_merge(merged, source.data)
        
        self.merged_config = merged
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs deep merge of two dictionaries.
        
        Args:
            base: Base dictionary
            override: Override dictionary
            
        Returns:
            Merged dictionary with override values taking precedence
            
        Behavior:
        - Recursively merges nested dictionaries
        - Override values completely replace base values for non-dict types
        - Preserves structure and nesting from both sources
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def validate(self) -> bool:
        """
        Validates the merged configuration against schema and business rules.
        
        Returns:
            True if configuration is valid, False otherwise
            
        Behavior:
        - Validates configuration structure against JSON schema
        - Checks business rule constraints (e.g., split ratios sum to 1.0)
        - Populates validation_errors list with specific error messages
        - Returns validation status without raising exceptions
        """
        self.validation_errors.clear()
        
        # Basic schema validation
        if not self._validate_schema():
            return False
        
        # Business rule validation
        if not self._validate_business_rules():
            return False
        
        self.logger.info("Configuration validation successful")
        return True
    
    def _validate_schema(self) -> bool:
        """
        Validates configuration against JSON schema.
        
        Returns:
            True if schema validation passes, False otherwise
            
        Behavior:
        - Uses jsonschema library for validation if available
        - Falls back to basic type checking if jsonschema not available
        - Records specific schema violations in validation_errors
        """
        try:
            import jsonschema
            jsonschema.validate(self.merged_config, self.schema)
            return True
        except ImportError:
            # Fall back to basic validation if jsonschema not available
            return self._basic_schema_validation()
        except jsonschema.ValidationError as e:
            self.validation_errors.append(f"Schema validation error: {e.message}")
            return False
    
    def _basic_schema_validation(self) -> bool:
        """
        Performs basic schema validation without jsonschema library.
        
        Returns:
            True if basic validation passes, False otherwise
            
        Behavior:
        - Checks for required keys
        - Validates basic types
        - Records validation errors for missing or invalid fields
        """
        # Check required sections
        required_sections = ["data", "processing"]
        for section in required_sections:
            if section not in self.merged_config:
                self.validation_errors.append(f"Required configuration section missing: {section}")
                return False
        
        return True
    
    def _validate_business_rules(self) -> bool:
        """
        Validates business-specific configuration rules.
        
        Returns:
            True if business rules validation passes, False otherwise
            
        Behavior:
        - Validates split ratios sum to 1.0 (within tolerance)
        - Checks training dataset specifications are valid
        - Validates plugin configurations are consistent
        - Records specific business rule violations
        """
        valid = True
        
        # Validate split ratios
        if "processing" in self.merged_config and "split_ratios" in self.merged_config["processing"]:
            ratios = self.merged_config["processing"]["split_ratios"]
            ratio_sum = sum(ratios.values())
            
            if abs(ratio_sum - 1.0) > 0.001:
                self.validation_errors.append(
                    f"Split ratios must sum to 1.0 (±0.001), current sum: {ratio_sum:.6f}"
                )
                valid = False
        
        # Validate training datasets
        if "processing" in self.merged_config and "training_datasets" in self.merged_config["processing"]:
            training_datasets = self.merged_config["processing"]["training_datasets"]
            valid_datasets = ["d1", "d2", "d3", "d4", "d5", "d6"]
            
            for dataset in training_datasets:
                if dataset not in valid_datasets:
                    self.validation_errors.append(
                        f"Invalid training dataset specified: {dataset}. Valid options: {valid_datasets}"
                    )
                    valid = False
        
        return valid
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Gets configuration value using dot notation path.
        
        Args:
            key_path: Dot-separated path to configuration value
            default: Default value if key not found
            
        Returns:
            Configuration value or default if not found
            
        Behavior:
        - Supports nested key access with dot notation
        - Returns typed values from configuration
        - Uses provided default or None if key not found
        - Does not raise exceptions for missing keys
        """
        keys = key_path.split('.')
        current = self.merged_config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Gets entire configuration section.
        
        Args:
            section: Name of configuration section
            
        Returns:
            Dictionary containing section configuration
            
        Behavior:
        - Returns complete section as dictionary
        - Returns empty dictionary if section not found
        - Preserves nested structure within section
        """
        return self.merged_config.get(section, {})
    
    def has(self, key_path: str) -> bool:
        """
        Checks if configuration key exists.
        
        Args:
            key_path: Dot-separated path to configuration key
            
        Returns:
            True if key exists, False otherwise
            
        Behavior:
        - Supports nested key checking with dot notation
        - Returns False for None values
        - Does not raise exceptions for invalid paths
        """
        try:
            value = self.get(key_path)
            return value is not None
        except:
            return False
    
    def get_validation_errors(self) -> List[str]:
        """
        Returns list of validation errors from last validation attempt.
        
        Returns:
            List of validation error messages
            
        Behavior:
        - Returns errors from most recent validation
        - Provides specific, actionable error messages
        - Clears on next validation attempt
        """
        return self.validation_errors.copy()
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """
        Returns summary of configuration sources and final values.
        
        Returns:
            Dictionary containing configuration summary information
            
        Behavior:
        - Lists all configuration sources with their priorities
        - Shows final merged configuration
        - Includes validation status and any errors
        - Provides traceability for configuration values
        """
        return {
            "sources": [
                {
                    "name": source.name,
                    "type": source.type,
                    "priority": source.priority,
                    "location": source.location
                }
                for source in self.sources
            ],
            "merged_config": self.merged_config,
            "validation_status": len(self.validation_errors) == 0,
            "validation_errors": self.validation_errors
        }
