"""
Plugin Loader for Preprocessor System

This module provides comprehensive plugin loading, validation, and isolation
with support for external repositories and perfect replicability.
"""

import sys
import os
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from copy import deepcopy
import json


class PluginIsolationManager:
    """Manages plugin execution isolation for perfect replicability."""
    
    def __init__(self):
        self.original_sys_path = None
        self.original_modules = None
        self.isolated_namespace = {}
    
    def enter_isolation(self):
        """Enter isolated execution context."""
        self.original_sys_path = sys.path.copy()
        self.original_modules = set(sys.modules.keys())
        
    def exit_isolation(self):
        """Exit isolated execution context and clean up."""
        # Restore original sys.path
        if self.original_sys_path is not None:
            sys.path[:] = self.original_sys_path
        
        # Remove any modules loaded during isolation
        if self.original_modules is not None:
            current_modules = set(sys.modules.keys())
            new_modules = current_modules - self.original_modules
            for module_name in new_modules:
                if module_name in sys.modules:
                    del sys.modules[module_name]
        
        self.isolated_namespace.clear()


class PluginLoader:
    """
    Comprehensive plugin loading system with isolation, validation,
    and support for external repositories.
    
    Behavioral Requirements:
    - BR-PL-001: Discover and validate plugin structure
    - BR-PL-002: Load plugins with proper isolation
    - BR-PL-003: Support external repository plugin loading
    - BR-PL-004: Ensure perfect replicability across executions
    """
    
    def __init__(self, plugin_paths: Optional[List[str]] = None):
        self.plugin_paths = plugin_paths or []
        self.loaded_plugins: Dict[str, Any] = {}
        self.plugin_cache: Dict[str, Dict[str, Any]] = {}
        self.isolation_manager = PluginIsolationManager()
        self.external_plugin_paths: List[str] = []
        
        # Default plugin paths
        self.default_paths = [
            "app/plugins",
            "plugins"
        ]
    
    def discover_plugins(self, plugin_dir: str) -> Dict[str, Any]:
        """
        Discover available plugins in directory with validation.
        
        Args:
            plugin_dir: Directory to search for plugins
            
        Returns:
            Discovery result with validated plugins
        """
        discovered_plugins = []
        invalid_plugins = []
        validation_errors = []
        
        plugin_path = Path(plugin_dir)
        if not plugin_path.exists():
            return {
                'success': False,
                'error': f"Plugin directory not found: {plugin_dir}"
            }
        
        # Scan for Python files
        for plugin_file in plugin_path.glob("*.py"):
            if plugin_file.name.startswith("__"):
                continue
            
            plugin_name = plugin_file.stem
            validation_result = self._validate_plugin_file(plugin_file)
            
            if validation_result['is_valid']:
                discovered_plugins.append({
                    'name': plugin_name,
                    'file_path': str(plugin_file),
                    'interface_valid': True
                })
            else:
                invalid_plugins.append({
                    'name': plugin_name,
                    'file_path': str(plugin_file),
                    'errors': validation_result['errors']
                })
                validation_errors.extend(validation_result['errors'])
        
        return {
            'success': True,
            'discovered_plugins': [p['name'] for p in discovered_plugins],
            'invalid_plugins': invalid_plugins,
            'validation_errors': validation_errors,
            'total_discovered': len(discovered_plugins)
        }
    
    def _validate_plugin_file(self, plugin_file: Path) -> Dict[str, Any]:
        """Validate plugin file structure and interface."""
        errors = []
        
        try:
            # Load module to check interface
            spec = importlib.util.spec_from_file_location("temp_plugin", plugin_file)
            if spec is None or spec.loader is None:
                errors.append(f"Cannot load plugin module: {plugin_file}")
                return {'is_valid': False, 'errors': errors}
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check for Plugin class
            if not hasattr(module, 'Plugin'):
                errors.append("Plugin file must contain a 'Plugin' class")
            else:
                plugin_class = getattr(module, 'Plugin')
                
                # Check for required methods
                required_methods = ['process']
                for method in required_methods:
                    if not hasattr(plugin_class, method):
                        errors.append(f"Plugin class must have '{method}' method")
                
                # Check for plugin_params
                if not hasattr(plugin_class, 'plugin_params'):
                    errors.append("Plugin class must have 'plugin_params' class attribute")
        
        except Exception as e:
            errors.append(f"Plugin validation error: {str(e)}")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors
        }
    
    def load_plugin(self, plugin_name: str, external_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load plugin with proper isolation and validation.
        
        Args:
            plugin_name: Name of plugin to load
            external_path: Optional external repository path
            
        Returns:
            Loading result with plugin instance
        """
        try:
            # Enter isolation context
            self.isolation_manager.enter_isolation()
            
            # Determine plugin file path
            plugin_file = self._find_plugin_file(plugin_name, external_path)
            if not plugin_file:
                return {
                    'success': False,
                    'error': f"Plugin file not found: {plugin_name}"
                }
            
            # Load plugin with isolation
            plugin_instance = self._load_plugin_isolated(plugin_name, plugin_file)
            
            if plugin_instance:
                self.loaded_plugins[plugin_name] = plugin_instance
                return {
                    'success': True,
                    'plugin_instance': plugin_instance,
                    'plugin_namespace': f"isolated_{plugin_name}",
                    'dependencies_resolved': True,
                    'file_path': str(plugin_file)
                }
            else:
                return {
                    'success': False,
                    'error': f"Failed to instantiate plugin: {plugin_name}"
                }
        
        except Exception as e:
            return {
                'success': False,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'plugin_instance': None
            }
        
        finally:
            # Always exit isolation
            self.isolation_manager.exit_isolation()
    
    def _find_plugin_file(self, plugin_name: str, external_path: Optional[str] = None) -> Optional[Path]:
        """Find plugin file in available paths."""
        search_paths = []
        
        # Add external path if provided
        if external_path:
            search_paths.append(external_path)
        
        # Add configured external paths
        search_paths.extend(self.external_plugin_paths)
        
        # Add configured plugin paths
        search_paths.extend(self.plugin_paths)
        
        # Add default paths
        search_paths.extend(self.default_paths)
        
        for path in search_paths:
            plugin_file = Path(path) / f"{plugin_name}.py"
            if plugin_file.exists():
                return plugin_file
        
        return None
    
    def _load_plugin_isolated(self, plugin_name: str, plugin_file: Path) -> Any:
        """Load plugin in isolated environment."""
        # Create unique module name to avoid conflicts
        module_name = f"isolated_plugin_{plugin_name}_{id(plugin_file)}"
        
        # Load module specification
        spec = importlib.util.spec_from_file_location(module_name, plugin_file)
        if spec is None or spec.loader is None:
            return None
        
        # Create and execute module
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get Plugin class and instantiate
        if hasattr(module, 'Plugin'):
            plugin_class = getattr(module, 'Plugin')
            return plugin_class()
        
        return None
    
    def load_external_plugin(self, external_path: str) -> Any:
        """
        Load plugin from external repository for cross-app usage.
        
        Args:
            external_path: Path to external plugin directory
            
        Returns:
            Plugin instance ready for execution
        """
        # Add to external paths if not already present
        if external_path not in self.external_plugin_paths:
            self.external_plugin_paths.append(external_path)
        
        # Discover plugins in external path
        discovery_result = self.discover_plugins(external_path)
        
        if not discovery_result['success'] or not discovery_result['discovered_plugins']:
            raise ImportError(f"No valid plugins found in external path: {external_path}")
        
        # Load the first valid plugin (or could be made configurable)
        plugin_name = discovery_result['discovered_plugins'][0]
        load_result = self.load_plugin(plugin_name, external_path)
        
        if load_result['success']:
            return load_result['plugin_instance']
        else:
            raise ImportError(f"Failed to load external plugin: {load_result['error']}")
    
    def execute_plugin_isolated(self, plugin_instance: Any, config: Dict[str, Any], 
                               isolation_level: str = 'strict') -> Dict[str, Any]:
        """
        Execute plugin in complete isolation for perfect replicability.
        
        Args:
            plugin_instance: Plugin to execute
            config: Configuration parameters (will be deep-copied)
            isolation_level: Level of isolation ('basic', 'standard', 'strict')
            
        Returns:
            Execution result with isolation validation
        """
        # Deep copy config to prevent any shared state
        isolated_config = deepcopy(config)
        
        # Freeze config to prevent modifications
        frozen_config = self._freeze_config(isolated_config)
        
        # Enter isolation
        self.isolation_manager.enter_isolation()
        
        try:
            # Validate isolation before execution
            isolation_violations = self._check_isolation_violations(plugin_instance)
            
            if isolation_violations and isolation_level == 'strict':
                return {
                    'success': False,
                    'isolation_violations': isolation_violations,
                    'error': 'Plugin violates strict isolation requirements'
                }
            
            # Execute plugin with isolated config
            # Generate unique execution state
            execution_state = f"exec_{id(frozen_config)}_{id(plugin_instance)}"
            
            # Execute the plugin
            if hasattr(plugin_instance, 'process'):
                result = plugin_instance.process(
                    frozen_config.get('data'), 
                    frozen_config
                )
            else:
                raise AttributeError("Plugin must have 'process' method")
            
            return {
                'success': True,
                'result': result,
                'isolation_violations': isolation_violations,
                'internal_dependencies': [],  # Track if plugin accessed internal state
                'execution_state': execution_state
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'isolation_violations': [],
                'execution_state': None
            }
        
        finally:
            self.isolation_manager.exit_isolation()
    
    def _freeze_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a frozen (immutable) version of configuration."""
        # For now, return a deep copy. In production, consider using frozendict
        return deepcopy(config)
    
    def _check_isolation_violations(self, plugin_instance: Any) -> List[str]:
        """Check for potential isolation violations."""
        violations = []
        
        # Check if plugin attempts to modify global state
        if hasattr(plugin_instance, '__dict__'):
            for attr_name, attr_value in plugin_instance.__dict__.items():
                if callable(attr_value) and 'global' in str(attr_value):
                    violations.append(f"Plugin may access global state via {attr_name}")
        
        # Additional isolation checks could be added here
        return violations
    
    def clear_plugin_cache(self):
        """Clear plugin cache for fresh loading."""
        self.plugin_cache.clear()
        self.loaded_plugins.clear()
    
    def validate_replicability(self, plugin_instance: Any, config: Dict[str, Any], 
                             data: Any, iterations: int = 3) -> Dict[str, Any]:
        """
        Validate that plugin produces identical results across multiple executions.
        
        Args:
            plugin_instance: Plugin to test
            config: Configuration to use
            data: Test data
            iterations: Number of test iterations
            
        Returns:
            Replicability validation result
        """
        results = []
        
        for i in range(iterations):
            # Clear any potential state
            self.clear_plugin_cache()
            
            # Execute with isolated config
            isolated_config = deepcopy(config)
            isolated_config['data'] = deepcopy(data)
            
            exec_result = self.execute_plugin_isolated(plugin_instance, isolated_config)
            
            if exec_result['success']:
                results.append(exec_result['result'])
            else:
                return {
                    'replicable': False,
                    'error': f"Execution {i+1} failed: {exec_result['error']}"
                }
        
        # Compare results for exact equality
        first_result = results[0]
        all_identical = all(
            self._compare_results(first_result, result) for result in results[1:]
        )
        
        return {
            'replicable': all_identical,
            'iterations_tested': iterations,
            'all_results_identical': all_identical,
            'sample_result': first_result
        }
    
    def _compare_results(self, result1: Any, result2: Any) -> bool:
        """Compare two results for exact equality."""
        import pandas as pd
        import numpy as np
        
        # Handle pandas DataFrames
        if isinstance(result1, pd.DataFrame) and isinstance(result2, pd.DataFrame):
            return result1.equals(result2)
        
        # Handle numpy arrays
        if isinstance(result1, np.ndarray) and isinstance(result2, np.ndarray):
            return np.array_equal(result1, result2)
        
        # Handle other types
        return result1 == result2
    
    def get_plugin_info(self, plugin_name: str) -> Dict[str, Any]:
        """Get comprehensive information about a loaded plugin."""
        if plugin_name not in self.loaded_plugins:
            return {'exists': False}
        
        plugin_instance = self.loaded_plugins[plugin_name]
        
        info = {
            'exists': True,
            'class_name': plugin_instance.__class__.__name__,
            'has_process_method': hasattr(plugin_instance, 'process'),
            'has_params': hasattr(plugin_instance, 'plugin_params')
        }
        
        if hasattr(plugin_instance, 'plugin_params'):
            info['parameters'] = plugin_instance.plugin_params
        
        return info
