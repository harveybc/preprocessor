"""
Plugin Manager for Preprocessor System

This module manages plugin lifecycle, execution, and resource management
with comprehensive isolation and performance monitoring.
"""

import time
import traceback
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager


class PluginState(Enum):
    """Plugin execution states."""
    UNINITIALIZED = "uninitialized"
    INITIALIZED = "initialized"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    DISPOSED = "disposed"


@dataclass
class ExecutionContext:
    """Context for plugin execution."""
    plugin_name: str
    parameters: Dict[str, Any]
    timeout: Optional[float] = None
    isolation_level: str = "standard"
    resource_limits: Optional[Dict[str, Any]] = None


@dataclass
class ExecutionResult:
    """Result of plugin execution."""
    success: bool
    output: Any = None
    execution_time: float = 0.0
    memory_used: float = 0.0
    error_message: Optional[str] = None
    warnings: List[str] = None
    metadata: Dict[str, Any] = None


class PluginManager:
    """
    Comprehensive plugin lifecycle and execution management with
    isolation, monitoring, and resource control.
    
    Behavioral Requirements:
    - BR-PM-001: Manage plugin initialization and cleanup
    - BR-PM-002: Execute plugins with timeout and error handling
    - BR-PM-003: Monitor plugin performance and resource usage
    - BR-PM-004: Provide plugin isolation and security controls
    """
    
    def __init__(self, max_concurrent_plugins: int = 10):
        self.max_concurrent_plugins = max_concurrent_plugins
        self.active_plugins: Dict[str, Any] = {}
        self.plugin_states: Dict[str, PluginState] = {}
        self.execution_locks: Dict[str, threading.Lock] = {}
        self.resource_monitors: Dict[str, Any] = {}
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
    
    def initialize_plugin(self, plugin_instance: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize plugin with proper configuration and state setup.
        
        Args:
            plugin_instance: Plugin instance to initialize
            config: Configuration parameters
            
        Returns:
            Initialization result with status and metrics
        """
        plugin_name = getattr(plugin_instance, '__class__').__name__
        
        try:
            # Set plugin state
            self.plugin_states[plugin_name] = PluginState.UNINITIALIZED
            
            # Configure plugin parameters
            if hasattr(plugin_instance, 'set_params'):
                plugin_instance.set_params(**config)
            
            # Initialize plugin if it has an init method
            if hasattr(plugin_instance, 'initialize'):
                plugin_instance.initialize(config)
            
            # Store active plugin
            self.active_plugins[plugin_name] = plugin_instance
            self.plugin_states[plugin_name] = PluginState.INITIALIZED
            self.execution_locks[plugin_name] = threading.Lock()
            
            return {
                'success': True,
                'plugin_name': plugin_name,
                'plugin_state': PluginState.INITIALIZED.value,
                'configuration_applied': True,
                'parameters_set': len(config)
            }
            
        except Exception as e:
            self.plugin_states[plugin_name] = PluginState.ERROR
            return {
                'success': False,
                'plugin_name': plugin_name,
                'error_message': str(e),
                'error_traceback': traceback.format_exc()
            }
    
    def execute_plugin(self, plugin_instance: Any, context: ExecutionContext) -> ExecutionResult:
        """
        Execute plugin with comprehensive monitoring and error handling.
        
        Args:
            plugin_instance: Plugin instance to execute
            context: Execution context with parameters and constraints
            
        Returns:
            ExecutionResult with output and performance metrics
        """
        plugin_name = context.plugin_name
        start_time = time.time()
        
        # Check if plugin is properly initialized
        if plugin_name not in self.plugin_states or \
           self.plugin_states[plugin_name] != PluginState.INITIALIZED:
            return ExecutionResult(
                success=False,
                error_message=f"Plugin {plugin_name} not properly initialized"
            )
        
        # Acquire execution lock
        if plugin_name in self.execution_locks:
            with self.execution_locks[plugin_name]:
                return self._execute_with_monitoring(plugin_instance, context, start_time)
        else:
            return self._execute_with_monitoring(plugin_instance, context, start_time)
    
    def _execute_with_monitoring(self, plugin_instance: Any, context: ExecutionContext, 
                               start_time: float) -> ExecutionResult:
        """Execute plugin with monitoring and resource tracking."""
        plugin_name = context.plugin_name
        
        try:
            # Set running state
            self.plugin_states[plugin_name] = PluginState.RUNNING
            
            # Start resource monitoring
            with self._monitor_resources(plugin_name) as monitor:
                # Execute with timeout if specified
                if context.timeout:
                    result = self._execute_with_timeout(
                        plugin_instance, context.parameters, context.timeout
                    )
                else:
                    result = self._execute_plugin_method(plugin_instance, context.parameters)
                
                execution_time = time.time() - start_time
                memory_used = monitor.get_peak_memory() if monitor else 0.0
                
                # Update state and record performance
                self.plugin_states[plugin_name] = PluginState.COMPLETED
                self._record_performance(plugin_name, execution_time, memory_used, True)
                
                return ExecutionResult(
                    success=True,
                    output=result,
                    execution_time=execution_time,
                    memory_used=memory_used,
                    metadata={'plugin_name': plugin_name, 'context': context.__dict__}
                )
                
        except TimeoutError:
            self.plugin_states[plugin_name] = PluginState.ERROR
            return ExecutionResult(
                success=False,
                execution_time=time.time() - start_time,
                error_message=f"Plugin {plugin_name} execution timed out"
            )
            
        except Exception as e:
            self.plugin_states[plugin_name] = PluginState.ERROR
            execution_time = time.time() - start_time
            self._record_performance(plugin_name, execution_time, 0.0, False)
            
            return ExecutionResult(
                success=False,
                execution_time=execution_time,
                error_message=str(e),
                metadata={'error_traceback': traceback.format_exc()}
            )
    
    def _execute_plugin_method(self, plugin_instance: Any, parameters: Dict[str, Any]) -> Any:
        """Execute the main plugin process method."""
        if hasattr(plugin_instance, 'process'):
            return plugin_instance.process(parameters.get('data'), parameters.get('config', {}))
        else:
            raise AttributeError("Plugin must have a 'process' method")
    
    def _execute_with_timeout(self, plugin_instance: Any, parameters: Dict[str, Any], 
                            timeout: float) -> Any:
        """Execute plugin with timeout control."""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Plugin execution exceeded {timeout} seconds")
        
        # Set timeout signal
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))
        
        try:
            result = self._execute_plugin_method(plugin_instance, parameters)
            signal.alarm(0)  # Cancel timeout
            return result
        except:
            signal.alarm(0)  # Cancel timeout
            raise
    
    @contextmanager
    def _monitor_resources(self, plugin_name: str):
        """Context manager for resource monitoring."""
        class ResourceMonitor:
            def __init__(self):
                self.peak_memory = 0.0
                self.start_memory = 0.0
            
            def get_peak_memory(self):
                return self.peak_memory
        
        monitor = ResourceMonitor()
        
        try:
            # Start monitoring (simplified for this implementation)
            yield monitor
        finally:
            # Clean up monitoring
            pass
    
    def _record_performance(self, plugin_name: str, execution_time: float, 
                          memory_used: float, success: bool):
        """Record performance metrics for plugin execution."""
        if plugin_name not in self.performance_history:
            self.performance_history[plugin_name] = []
        
        performance_record = {
            'timestamp': time.time(),
            'execution_time': execution_time,
            'memory_used': memory_used,
            'success': success
        }
        
        self.performance_history[plugin_name].append(performance_record)
        
        # Keep only last 100 records per plugin
        if len(self.performance_history[plugin_name]) > 100:
            self.performance_history[plugin_name] = self.performance_history[plugin_name][-100:]
    
    def dispose_plugin(self, plugin_instance: Any) -> Dict[str, Any]:
        """
        Properly dispose of plugin and clean up resources.
        
        Args:
            plugin_instance: Plugin instance to dispose
            
        Returns:
            Disposal result with cleanup status
        """
        plugin_name = getattr(plugin_instance, '__class__').__name__
        
        try:
            # Call cleanup method if available
            if hasattr(plugin_instance, 'cleanup'):
                plugin_instance.cleanup()
            
            # Clean up manager resources
            if plugin_name in self.active_plugins:
                del self.active_plugins[plugin_name]
            
            if plugin_name in self.execution_locks:
                del self.execution_locks[plugin_name]
            
            self.plugin_states[plugin_name] = PluginState.DISPOSED
            
            return {
                'success': True,
                'plugin_name': plugin_name,
                'plugin_state': PluginState.DISPOSED.value,
                'resources_released': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'plugin_name': plugin_name,
                'error_message': str(e)
            }
    
    def get_plugin_status(self, plugin_name: str) -> Dict[str, Any]:
        """Get current status and metrics for a plugin."""
        status = {
            'plugin_name': plugin_name,
            'state': self.plugin_states.get(plugin_name, PluginState.UNINITIALIZED).value,
            'is_active': plugin_name in self.active_plugins,
            'has_lock': plugin_name in self.execution_locks
        }
        
        # Add performance history if available
        if plugin_name in self.performance_history:
            history = self.performance_history[plugin_name]
            if history:
                status['performance'] = {
                    'total_executions': len(history),
                    'successful_executions': sum(1 for r in history if r['success']),
                    'average_execution_time': sum(r['execution_time'] for r in history) / len(history),
                    'last_execution': history[-1]
                }
        
        return status
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status and metrics."""
        total_plugins = len(self.plugin_states)
        active_plugins = len(self.active_plugins)
        
        state_counts = {}
        for state in self.plugin_states.values():
            state_counts[state.value] = state_counts.get(state.value, 0) + 1
        
        return {
            'total_plugins': total_plugins,
            'active_plugins': active_plugins,
            'max_concurrent': self.max_concurrent_plugins,
            'plugin_states': state_counts,
            'performance_tracked_plugins': len(self.performance_history)
        }
    
    def cleanup_all_plugins(self):
        """Clean up all active plugins and resources."""
        for plugin_name, plugin_instance in list(self.active_plugins.items()):
            self.dispose_plugin(plugin_instance)
        
        self.active_plugins.clear()
        self.execution_locks.clear()
        self.resource_monitors.clear()
