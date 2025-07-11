"""Plugin Loader Unit

This module implements the PluginLoader class that provides plugin discovery,
loading, and lifecycle management capabilities.

Behavioral Specification:
- Discovers plugins in configured directories automatically
- Validates plugin interfaces for compliance
- Manages plugin dependencies and loading order
- Provides plugin lifecycle management (initialize, execute, cleanup)
"""

import logging
import importlib.util
import inspect
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Tuple
from dataclasses import dataclass, field
from enum import Enum


class PluginState(Enum):
    """Plugin lifecycle states"""
    DISCOVERED = "discovered"
    LOADED = "loaded"
    INITIALIZED = "initialized"
    READY = "ready"
    FAILED = "failed"


@dataclass
class PluginMetadata:
    """Plugin metadata container"""
    name: str
    version: str
    description: str
    author: str
    plugin_type: str
    dependencies: List[str] = field(default_factory=list)
    file_path: str = ""
    class_name: str = ""
    state: PluginState = PluginState.DISCOVERED


class BasePlugin(ABC):
    """
    Base class for all plugins.
    
    Behavioral Contract:
    - MUST implement execute method for plugin functionality
    - MUST provide metadata through get_metadata method
    - MUST handle configuration through configure method
    - MUST support cleanup through cleanup method
    """
    
    def __init__(self):
        """Initialize base plugin"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.config = {}
        self.initialized = False
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """
        Returns plugin metadata.
        
        Returns:
            PluginMetadata object with plugin information
            
        Behavior:
        - MUST return complete metadata information
        - MUST include accurate plugin type and dependencies
        - MUST provide version and description
        """
        pass
    
    @abstractmethod
    def execute(self, data: Any, **kwargs) -> Any:
        """
        Executes plugin functionality.
        
        Args:
            data: Input data to process
            **kwargs: Additional execution parameters
            
        Returns:
            Processed data
            
        Behavior:
        - MUST process input data according to plugin purpose
        - MUST return data in expected output format
        - MUST handle errors gracefully without crashing
        - MUST preserve data integrity during processing
        """
        pass
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """
        Configures plugin with provided parameters.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if configuration successful, False otherwise
            
        Behavior:
        - Validates configuration parameters
        - Stores configuration for use during execution
        - Returns configuration status
        """
        try:
            self.config = config.copy()
            self.initialized = True
            self.logger.debug(f"Plugin {self.__class__.__name__} configured successfully")
            return True
        except Exception as e:
            self.logger.error(f"Plugin {self.__class__.__name__} configuration failed: {e}")
            return False
    
    def cleanup(self) -> None:
        """
        Performs plugin cleanup operations.
        
        Behavior:
        - Releases any allocated resources
        - Clears internal state
        - Prepares plugin for shutdown
        """
        self.config.clear()
        self.initialized = False
        self.logger.debug(f"Plugin {self.__class__.__name__} cleanup completed")
    
    def validate_input(self, data: Any) -> bool:
        """
        Validates input data format and requirements.
        
        Args:
            data: Input data to validate
            
        Returns:
            True if input is valid, False otherwise
            
        Behavior:
        - Checks input data format requirements
        - Validates data structure and types
        - Returns validation status without raising exceptions
        """
        # Base implementation - subclasses should override for specific validation
        return data is not None
    
    def get_output_schema(self) -> Dict[str, Any]:
        """
        Returns expected output data schema.
        
        Returns:
            Dictionary describing output data structure
            
        Behavior:
        - Provides schema for output data validation
        - Enables downstream components to validate plugin output
        - Supports integration testing and validation
        """
        return {"type": "any", "description": "Plugin output"}


class PluginLoader:
    """
    Manages plugin discovery, loading, and lifecycle.
    
    Behavioral Contract:
    - MUST discover plugins automatically from configured directories
    - MUST validate plugin interfaces before loading
    - MUST resolve plugin dependencies correctly
    - MUST provide plugin lifecycle management
    - MUST isolate plugin failures from core system
    """
    
    def __init__(self, plugin_directories: List[str] = None):
        """
        Initialize plugin loader.
        
        Args:
            plugin_directories: List of directories to search for plugins
            
        Behaviors:
        - Sets up plugin discovery directories
        - Initializes plugin tracking structures
        - Prepares dependency resolution system
        """
        self.logger = logging.getLogger(__name__)
        self.plugin_directories = plugin_directories or []
        self.discovered_plugins: Dict[str, PluginMetadata] = {}
        self.loaded_plugins: Dict[str, BasePlugin] = {}
        self.plugin_instances: Dict[str, BasePlugin] = {}
        self.dependency_graph: Dict[str, List[str]] = {}
        self.loading_errors: List[str] = []
    
    def add_plugin_directory(self, directory: str) -> None:
        """
        Adds a directory to the plugin search path.
        
        Args:
            directory: Path to directory containing plugins
            
        Behavior:
        - Validates directory exists and is accessible
        - Adds directory to search path if valid
        - Logs directory addition for traceability
        """
        dir_path = Path(directory)
        if dir_path.exists() and dir_path.is_dir():
            if directory not in self.plugin_directories:
                self.plugin_directories.append(directory)
                self.logger.info(f"Added plugin directory: {directory}")
        else:
            self.logger.warning(f"Plugin directory not found or not accessible: {directory}")
    
    def discover_plugins(self) -> int:
        """
        Discovers all plugins in configured directories.
        
        Returns:
            Number of plugins discovered
            
        Behavior:
        - Scans all configured directories for Python files
        - Identifies files containing valid plugin classes
        - Extracts plugin metadata without loading plugins
        - Returns count of discovered plugins
        """
        self.discovered_plugins.clear()
        self.loading_errors.clear()
        
        plugin_count = 0
        
        for directory in self.plugin_directories:
            plugin_count += self._discover_plugins_in_directory(directory)
        
        self.logger.info(f"Discovered {plugin_count} plugins in {len(self.plugin_directories)} directories")
        return plugin_count
    
    def _discover_plugins_in_directory(self, directory: str) -> int:
        """
        Discovers plugins in a specific directory.
        
        Args:
            directory: Directory path to search
            
        Returns:
            Number of plugins discovered in directory
            
        Behavior:
        - Scans directory for Python files
        - Inspects Python files for plugin classes
        - Extracts metadata from valid plugin classes
        - Handles discovery errors gracefully
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            return 0
        
        plugin_count = 0
        
        for file_path in dir_path.glob("*.py"):
            if file_path.name.startswith("__"):
                continue  # Skip __init__.py and other special files
            
            try:
                plugins_found = self._inspect_plugin_file(str(file_path))
                plugin_count += plugins_found
            except Exception as e:
                error_msg = f"Error inspecting plugin file {file_path}: {e}"
                self.loading_errors.append(error_msg)
                self.logger.warning(error_msg)
        
        return plugin_count
    
    def _inspect_plugin_file(self, file_path: str) -> int:
        """
        Inspects a Python file for plugin classes.
        
        Args:
            file_path: Path to Python file to inspect
            
        Returns:
            Number of plugin classes found in file
            
        Behavior:
        - Loads Python module from file
        - Inspects module for classes inheriting from BasePlugin
        - Extracts metadata from valid plugin classes
        - Handles inspection errors without crashing
        """
        spec = importlib.util.spec_from_file_location("plugin_module", file_path)
        if spec is None or spec.loader is None:
            return 0
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        plugin_count = 0
        
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (issubclass(obj, BasePlugin) and 
                obj != BasePlugin and 
                not inspect.isabstract(obj)):
                
                try:
                    # Create temporary instance to get metadata
                    temp_instance = obj()
                    metadata = temp_instance.get_metadata()
                    metadata.file_path = file_path
                    metadata.class_name = name
                    
                    self.discovered_plugins[metadata.name] = metadata
                    plugin_count += 1
                    
                    self.logger.debug(f"Discovered plugin: {metadata.name} ({metadata.plugin_type})")
                    
                except Exception as e:
                    error_msg = f"Error getting metadata from plugin class {name} in {file_path}: {e}"
                    self.loading_errors.append(error_msg)
                    self.logger.warning(error_msg)
        
        return plugin_count
    
    def validate_plugin_interface(self, plugin_name: str) -> bool:
        """
        Validates that a plugin implements the required interface correctly.
        
        Args:
            plugin_name: Name of plugin to validate
            
        Returns:
            True if plugin interface is valid, False otherwise
            
        Behavior:
        - Checks that plugin implements all required methods
        - Validates method signatures match base class
        - Verifies plugin metadata is complete and valid
        - Returns validation status without raising exceptions
        """
        if plugin_name not in self.discovered_plugins:
            self.logger.error(f"Plugin {plugin_name} not found in discovered plugins")
            return False
        
        metadata = self.discovered_plugins[plugin_name]
        
        try:
            # Load the plugin class
            plugin_class = self._load_plugin_class(metadata)
            if plugin_class is None:
                return False
            
            # Check required methods exist
            required_methods = ['get_metadata', 'execute']
            for method_name in required_methods:
                if not hasattr(plugin_class, method_name):
                    self.logger.error(f"Plugin {plugin_name} missing required method: {method_name}")
                    return False
                
                method = getattr(plugin_class, method_name)
                if not callable(method):
                    self.logger.error(f"Plugin {plugin_name} method {method_name} is not callable")
                    return False
            
            # Validate metadata completeness
            if not all([metadata.name, metadata.version, metadata.plugin_type]):
                self.logger.error(f"Plugin {plugin_name} has incomplete metadata")
                return False
            
            self.logger.debug(f"Plugin {plugin_name} interface validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Plugin {plugin_name} interface validation failed: {e}")
            return False
    
    def _load_plugin_class(self, metadata: PluginMetadata) -> Optional[Type[BasePlugin]]:
        """
        Loads a plugin class from its file.
        
        Args:
            metadata: Plugin metadata containing file and class information
            
        Returns:
            Plugin class or None if loading fails
            
        Behavior:
        - Loads Python module from plugin file
        - Extracts specified plugin class from module
        - Returns class object for instantiation
        - Handles loading errors gracefully
        """
        try:
            spec = importlib.util.spec_from_file_location("plugin_module", metadata.file_path)
            if spec is None or spec.loader is None:
                return None
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if hasattr(module, metadata.class_name):
                plugin_class = getattr(module, metadata.class_name)
                if issubclass(plugin_class, BasePlugin):
                    return plugin_class
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error loading plugin class {metadata.class_name} from {metadata.file_path}: {e}")
            return None
    
    def build_dependency_graph(self) -> bool:
        """
        Builds dependency graph for all discovered plugins.
        
        Returns:
            True if dependency graph is valid, False if circular dependencies detected
            
        Behavior:
        - Analyzes plugin dependencies from metadata
        - Constructs dependency graph
        - Detects circular dependencies
        - Validates all dependencies are available
        """
        self.dependency_graph.clear()
        
        # Build initial graph
        for plugin_name, metadata in self.discovered_plugins.items():
            self.dependency_graph[plugin_name] = metadata.dependencies.copy()
        
        # Validate dependencies exist
        for plugin_name, dependencies in self.dependency_graph.items():
            for dep in dependencies:
                if dep not in self.discovered_plugins:
                    error_msg = f"Plugin {plugin_name} depends on unknown plugin: {dep}"
                    self.loading_errors.append(error_msg)
                    self.logger.error(error_msg)
                    return False
        
        # Check for circular dependencies
        if self._has_circular_dependencies():
            error_msg = "Circular dependencies detected in plugin graph"
            self.loading_errors.append(error_msg)
            self.logger.error(error_msg)
            return False
        
        self.logger.info("Plugin dependency graph validated successfully")
        return True
    
    def _has_circular_dependencies(self) -> bool:
        """
        Detects circular dependencies in the plugin graph.
        
        Returns:
            True if circular dependencies exist, False otherwise
            
        Behavior:
        - Uses depth-first search to detect cycles
        - Tracks visited and recursion stack
        - Returns True if any cycle is detected
        """
        visited = set()
        rec_stack = set()
        
        def dfs(node: str) -> bool:
            if node in rec_stack:
                return True  # Cycle detected
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for dependency in self.dependency_graph.get(node, []):
                if dfs(dependency):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for plugin_name in self.dependency_graph:
            if plugin_name not in visited:
                if dfs(plugin_name):
                    return True
        
        return False
    
    def get_loading_order(self) -> List[str]:
        """
        Determines the correct order for loading plugins based on dependencies.
        
        Returns:
            List of plugin names in dependency-resolved order
            
        Behavior:
        - Uses topological sorting on dependency graph
        - Ensures dependencies are loaded before dependents
        - Returns empty list if dependency resolution fails
        """
        if not self.dependency_graph:
            return list(self.discovered_plugins.keys())
        
        # Topological sort using Kahn's algorithm
        in_degree = {plugin: 0 for plugin in self.dependency_graph}
        
        # Calculate in-degrees: if plugin A depends on plugin B, then A has an incoming edge from B
        for plugin, dependencies in self.dependency_graph.items():
            in_degree[plugin] = len(dependencies)  # Number of dependencies = in-degree
        
        # Find plugins with no dependencies (in-degree 0)
        queue = [plugin for plugin, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            # For each plugin that depends on current, reduce its in-degree
            for plugin, dependencies in self.dependency_graph.items():
                if current in dependencies:
                    in_degree[plugin] -= 1
                    if in_degree[plugin] == 0:
                        queue.append(plugin)
        
        # Check if all plugins were included (no cycles)
        if len(result) != len(self.dependency_graph):
            self.logger.error("Unable to resolve plugin loading order - circular dependencies exist")
            return []
        
        return result
    
    def load_plugin(self, plugin_name: str) -> bool:
        """
        Loads a specific plugin.
        
        Args:
            plugin_name: Name of plugin to load
            
        Returns:
            True if loading successful, False otherwise
            
        Behavior:
        - Validates plugin interface before loading
        - Instantiates plugin class
        - Stores plugin instance for later use
        - Updates plugin state tracking
        """
        if plugin_name not in self.discovered_plugins:
            self.logger.error(f"Cannot load unknown plugin: {plugin_name}")
            return False
        
        if plugin_name in self.loaded_plugins:
            self.logger.debug(f"Plugin {plugin_name} already loaded")
            return True
        
        metadata = self.discovered_plugins[plugin_name]
        
        # Validate interface
        if not self.validate_plugin_interface(plugin_name):
            metadata.state = PluginState.FAILED
            return False
        
        try:
            # Load plugin class
            plugin_class = self._load_plugin_class(metadata)
            if plugin_class is None:
                metadata.state = PluginState.FAILED
                return False
            
            # Instantiate plugin
            plugin_instance = plugin_class()
            
            # Store loaded plugin
            self.loaded_plugins[plugin_name] = plugin_instance
            self.plugin_instances[plugin_name] = plugin_instance
            metadata.state = PluginState.LOADED
            
            self.logger.info(f"Successfully loaded plugin: {plugin_name}")
            return True
            
        except Exception as e:
            error_msg = f"Error loading plugin {plugin_name}: {e}"
            self.loading_errors.append(error_msg)
            self.logger.error(error_msg)
            metadata.state = PluginState.FAILED
            return False
    
    def load_all_plugins(self) -> Tuple[int, int]:
        """
        Loads all discovered plugins in dependency order.
        
        Returns:
            Tuple of (successful_loads, failed_loads)
            
        Behavior:
        - Builds dependency graph first
        - Loads plugins in dependency-resolved order
        - Continues loading even if some plugins fail
        - Returns statistics on loading results
        """
        if not self.build_dependency_graph():
            self.logger.error("Cannot load plugins - dependency graph validation failed")
            return 0, len(self.discovered_plugins)
        
        loading_order = self.get_loading_order()
        if not loading_order:
            self.logger.error("Cannot determine plugin loading order")
            return 0, len(self.discovered_plugins)
        
        successful = 0
        failed = 0
        
        for plugin_name in loading_order:
            if self.load_plugin(plugin_name):
                successful += 1
            else:
                failed += 1
        
        self.logger.info(f"Plugin loading completed: {successful} successful, {failed} failed")
        return successful, failed
    
    def initialize_plugin(self, plugin_name: str, config: Dict[str, Any] = None) -> bool:
        """
        Initializes a loaded plugin with configuration.
        
        Args:
            plugin_name: Name of plugin to initialize
            config: Configuration dictionary for plugin
            
        Returns:
            True if initialization successful, False otherwise
            
        Behavior:
        - Verifies plugin is loaded before initialization
        - Passes configuration to plugin configure method
        - Updates plugin state tracking
        - Handles initialization errors gracefully
        """
        if plugin_name not in self.loaded_plugins:
            self.logger.error(f"Cannot initialize unloaded plugin: {plugin_name}")
            return False
        
        plugin = self.loaded_plugins[plugin_name]
        metadata = self.discovered_plugins[plugin_name]
        
        try:
            plugin_config = config or {}
            if plugin.configure(plugin_config):
                metadata.state = PluginState.INITIALIZED
                self.logger.info(f"Successfully initialized plugin: {plugin_name}")
                return True
            else:
                metadata.state = PluginState.FAILED
                self.logger.error(f"Plugin {plugin_name} configuration failed")
                return False
                
        except Exception as e:
            error_msg = f"Error initializing plugin {plugin_name}: {e}"
            self.loading_errors.append(error_msg)
            self.logger.error(error_msg)
            metadata.state = PluginState.FAILED
            return False
    
    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """
        Gets a loaded plugin instance by name.
        
        Args:
            plugin_name: Name of plugin to retrieve
            
        Returns:
            Plugin instance or None if not found
            
        Behavior:
        - Returns plugin instance if loaded and initialized
        - Returns None if plugin not found or not loaded
        - Provides access to plugin for execution
        """
        return self.loaded_plugins.get(plugin_name)
    
    def get_plugins_by_type(self, plugin_type: str) -> List[BasePlugin]:
        """
        Gets all loaded plugins of a specific type.
        
        Args:
            plugin_type: Type of plugins to retrieve
            
        Returns:
            List of plugin instances of the specified type
            
        Behavior:
        - Filters loaded plugins by type
        - Returns only initialized plugins
        - Maintains loading order within type
        """
        plugins = []
        for plugin_name, plugin in self.loaded_plugins.items():
            metadata = self.discovered_plugins[plugin_name]
            if (metadata.plugin_type == plugin_type and 
                metadata.state in [PluginState.INITIALIZED, PluginState.READY]):
                plugins.append(plugin)
        return plugins
    
    def get_plugin_metadata(self, plugin_name: str) -> Optional[PluginMetadata]:
        """
        Gets metadata for a discovered plugin.
        
        Args:
            plugin_name: Name of plugin
            
        Returns:
            Plugin metadata or None if not found
            
        Behavior:
        - Returns complete metadata for specified plugin
        - Includes current state and loading information
        - Provides plugin information for management decisions
        """
        return self.discovered_plugins.get(plugin_name)
    
    def get_loading_errors(self) -> List[str]:
        """
        Gets list of errors encountered during plugin loading.
        
        Returns:
            List of error messages
            
        Behavior:
        - Returns all errors from discovery and loading phases
        - Provides detailed error information for troubleshooting
        - Clears on next discovery operation
        """
        return self.loading_errors.copy()
    
    def cleanup_plugins(self) -> None:
        """
        Performs cleanup for all loaded plugins.
        
        Behavior:
        - Calls cleanup method on all loaded plugins
        - Handles cleanup errors gracefully
        - Clears plugin tracking structures
        - Prepares loader for shutdown
        """
        for plugin_name, plugin in self.loaded_plugins.items():
            try:
                plugin.cleanup()
                self.logger.debug(f"Cleaned up plugin: {plugin_name}")
            except Exception as e:
                self.logger.error(f"Error cleaning up plugin {plugin_name}: {e}")
        
        self.loaded_plugins.clear()
        self.plugin_instances.clear()
        self.logger.info("Plugin cleanup completed")
    
    def get_plugin_summary(self) -> Dict[str, Any]:
        """
        Returns summary of plugin loading status.
        
        Returns:
            Dictionary containing plugin loading summary
            
        Behavior:
        - Provides overview of discovery and loading results
        - Includes error information and state statistics
        - Useful for monitoring and troubleshooting
        """
        state_counts = {}
        for metadata in self.discovered_plugins.values():
            state = metadata.state.value
            state_counts[state] = state_counts.get(state, 0) + 1
        
        return {
            "total_discovered": len(self.discovered_plugins),
            "total_loaded": len(self.loaded_plugins),
            "state_distribution": state_counts,
            "loading_errors": len(self.loading_errors),
            "plugin_directories": self.plugin_directories.copy()
        }
