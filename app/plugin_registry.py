"""
Plugin Registry for Preprocessor System

This module manages plugin registration, metadata, and discovery
with comprehensive validation and capability tracking.
"""

import json
import importlib.util
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum


class PluginType(Enum):
    """Plugin type categories."""
    PREPROCESSOR = "preprocessor"
    FEATURE_EXTRACTOR = "feature_extractor"
    POST_PROCESSOR = "post_processor"
    VALIDATOR = "validator"


class PluginStatus(Enum):
    """Plugin registration status."""
    REGISTERED = "registered"
    VALIDATED = "validated"
    ACTIVE = "active"
    DISABLED = "disabled"
    ERROR = "error"


@dataclass
class PluginMetadata:
    """Comprehensive plugin metadata."""
    name: str
    version: str
    plugin_type: PluginType
    description: str
    author: str
    capabilities: List[str]
    requirements: List[str]
    parameters: Dict[str, Any]
    entry_point: str
    file_path: str
    status: PluginStatus = PluginStatus.REGISTERED
    error_message: Optional[str] = None


class PluginRegistry:
    """
    Central registry for managing plugin metadata, capabilities,
    and lifecycle with comprehensive validation and discovery.
    
    Behavioral Requirements:
    - BR-PR-001: Register plugins with complete metadata validation
    - BR-PR-002: Support plugin querying by capabilities and type
    - BR-PR-003: Maintain plugin status and lifecycle information
    - BR-PR-004: Provide plugin dependency resolution
    """
    
    def __init__(self, registry_file: Optional[str] = None):
        self.registry_file = registry_file or "plugin_registry.json"
        self.plugins: Dict[str, PluginMetadata] = {}
        self.capability_index: Dict[str, Set[str]] = {}
        self.type_index: Dict[PluginType, Set[str]] = {}
        self._load_registry()
    
    def register_plugin(self, metadata: PluginMetadata) -> Dict[str, Any]:
        """
        Register a plugin with comprehensive metadata validation.
        
        Args:
            metadata: Complete plugin metadata
            
        Returns:
            Registration result with validation details
        """
        validation_result = self._validate_plugin_metadata(metadata)
        
        if not validation_result['is_valid']:
            return {
                'success': False,
                'plugin_id': metadata.name,
                'validation_errors': validation_result['errors'],
                'missing_metadata': validation_result['missing_fields']
            }
        
        # Check for conflicts
        if metadata.name in self.plugins:
            existing = self.plugins[metadata.name]
            if existing.version == metadata.version:
                return {
                    'success': False,
                    'plugin_id': metadata.name,
                    'error': f"Plugin {metadata.name} version {metadata.version} already registered"
                }
        
        # Register plugin
        self.plugins[metadata.name] = metadata
        
        # Update indexes
        self._update_capability_index(metadata)
        self._update_type_index(metadata)
        
        # Persist registry
        self._save_registry()
        
        return {
            'success': True,
            'plugin_id': metadata.name,
            'metadata_validated': True,
            'capabilities_indexed': len(metadata.capabilities),
            'status': metadata.status.value
        }
    
    def _validate_plugin_metadata(self, metadata: PluginMetadata) -> Dict[str, Any]:
        """Validate plugin metadata completeness and consistency."""
        errors = []
        missing_fields = []
        
        # Required fields validation
        required_fields = ['name', 'version', 'plugin_type', 'description', 'entry_point']
        for field in required_fields:
            if not getattr(metadata, field, None):
                missing_fields.append(field)
        
        # Validate plugin type
        if not isinstance(metadata.plugin_type, PluginType):
            errors.append("Invalid plugin_type, must be PluginType enum")
        
        # Validate version format
        if metadata.version and not self._is_valid_version(metadata.version):
            errors.append("Invalid version format, should be semantic versioning")
        
        # Validate capabilities
        if not isinstance(metadata.capabilities, list):
            errors.append("Capabilities must be a list")
        
        # Validate entry point
        if metadata.entry_point and not metadata.entry_point.endswith('.py'):
            errors.append("Entry point must be a Python file")
        
        # Validate file path exists
        if metadata.file_path and not Path(metadata.file_path).exists():
            errors.append(f"Plugin file not found: {metadata.file_path}")
        
        return {
            'is_valid': len(errors) == 0 and len(missing_fields) == 0,
            'errors': errors,
            'missing_fields': missing_fields
        }
    
    def _is_valid_version(self, version: str) -> bool:
        """Validate semantic version format."""
        try:
            parts = version.split('.')
            return len(parts) >= 2 and all(part.isdigit() for part in parts[:3])
        except:
            return False
    
    def _update_capability_index(self, metadata: PluginMetadata):
        """Update capability-based index."""
        for capability in metadata.capabilities:
            if capability not in self.capability_index:
                self.capability_index[capability] = set()
            self.capability_index[capability].add(metadata.name)
    
    def _update_type_index(self, metadata: PluginMetadata):
        """Update type-based index."""
        if metadata.plugin_type not in self.type_index:
            self.type_index[metadata.plugin_type] = set()
        self.type_index[metadata.plugin_type].add(metadata.name)
    
    def query_by_capabilities(self, capabilities: List[str], 
                            match_all: bool = True) -> Dict[str, Any]:
        """
        Query plugins by their capabilities.
        
        Args:
            capabilities: List of required capabilities
            match_all: If True, plugin must have all capabilities; if False, any capability
            
        Returns:
            Query result with matching plugins
        """
        matching_plugins = []
        
        for plugin_name, metadata in self.plugins.items():
            if metadata.status != PluginStatus.ACTIVE:
                continue
            
            plugin_capabilities = set(metadata.capabilities)
            required_capabilities = set(capabilities)
            
            if match_all:
                matches = required_capabilities.issubset(plugin_capabilities)
            else:
                matches = bool(required_capabilities.intersection(plugin_capabilities))
            
            if matches:
                matching_plugins.append(metadata)
        
        return {
            'success': True,
            'matching_plugins': matching_plugins,
            'total_matches': len(matching_plugins),
            'query_criteria': {
                'capabilities': capabilities,
                'match_all': match_all
            }
        }
    
    def query_by_type(self, plugin_type: PluginType) -> Dict[str, Any]:
        """Query plugins by type."""
        matching_plugins = []
        
        if plugin_type in self.type_index:
            for plugin_name in self.type_index[plugin_type]:
                if plugin_name in self.plugins:
                    metadata = self.plugins[plugin_name]
                    if metadata.status == PluginStatus.ACTIVE:
                        matching_plugins.append(metadata)
        
        return {
            'success': True,
            'matching_plugins': matching_plugins,
            'total_matches': len(matching_plugins),
            'plugin_type': plugin_type.value
        }
    
    def get_plugin(self, plugin_name: str) -> Optional[PluginMetadata]:
        """Get plugin metadata by name."""
        return self.plugins.get(plugin_name)
    
    def update_plugin_status(self, plugin_name: str, status: PluginStatus, 
                           error_message: Optional[str] = None) -> bool:
        """Update plugin status."""
        if plugin_name not in self.plugins:
            return False
        
        self.plugins[plugin_name].status = status
        if error_message:
            self.plugins[plugin_name].error_message = error_message
        
        self._save_registry()
        return True
    
    def unregister_plugin(self, plugin_name: str) -> bool:
        """Unregister a plugin and clean up indexes."""
        if plugin_name not in self.plugins:
            return False
        
        metadata = self.plugins[plugin_name]
        
        # Clean up capability index
        for capability in metadata.capabilities:
            if capability in self.capability_index:
                self.capability_index[capability].discard(plugin_name)
                if not self.capability_index[capability]:
                    del self.capability_index[capability]
        
        # Clean up type index
        if metadata.plugin_type in self.type_index:
            self.type_index[metadata.plugin_type].discard(plugin_name)
            if not self.type_index[metadata.plugin_type]:
                del self.type_index[metadata.plugin_type]
        
        # Remove plugin
        del self.plugins[plugin_name]
        
        self._save_registry()
        return True
    
    def list_plugins(self, status_filter: Optional[PluginStatus] = None) -> List[PluginMetadata]:
        """List all plugins with optional status filtering."""
        if status_filter:
            return [metadata for metadata in self.plugins.values() 
                   if metadata.status == status_filter]
        return list(self.plugins.values())
    
    def get_registry_statistics(self) -> Dict[str, Any]:
        """Get comprehensive registry statistics."""
        status_counts = {}
        type_counts = {}
        
        for metadata in self.plugins.values():
            # Count by status
            status = metadata.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            
            # Count by type
            plugin_type = metadata.plugin_type.value
            type_counts[plugin_type] = type_counts.get(plugin_type, 0) + 1
        
        return {
            'total_plugins': len(self.plugins),
            'by_status': status_counts,
            'by_type': type_counts,
            'total_capabilities': len(self.capability_index),
            'active_plugins': len([p for p in self.plugins.values() 
                                 if p.status == PluginStatus.ACTIVE])
        }
    
    def _load_registry(self):
        """Load registry from persistent storage."""
        try:
            if Path(self.registry_file).exists():
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                    
                for plugin_data in data.get('plugins', []):
                    # Convert dict back to PluginMetadata
                    plugin_data['plugin_type'] = PluginType(plugin_data['plugin_type'])
                    plugin_data['status'] = PluginStatus(plugin_data['status'])
                    metadata = PluginMetadata(**plugin_data)
                    
                    self.plugins[metadata.name] = metadata
                    self._update_capability_index(metadata)
                    self._update_type_index(metadata)
        except Exception as e:
            # Log error but continue with empty registry
            print(f"Warning: Could not load plugin registry: {e}")
    
    def _save_registry(self):
        """Save registry to persistent storage."""
        try:
            registry_data = {
                'plugins': [asdict(metadata) for metadata in self.plugins.values()],
                'version': '1.0',
                'timestamp': str(Path(self.registry_file).stat().st_mtime if Path(self.registry_file).exists() else 0)
            }
            
            # Convert enums to strings for JSON serialization
            for plugin_data in registry_data['plugins']:
                plugin_data['plugin_type'] = plugin_data['plugin_type'].value
                plugin_data['status'] = plugin_data['status'].value
            
            with open(self.registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save plugin registry: {e}")
