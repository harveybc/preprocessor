"""Postprocessing Plugin Base Classes

This module provides abstract base classes for postprocessing plugins,
following the plugin interface specification from the design documentation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import logging
from datetime import datetime


class PostprocessingPlugin(ABC):
    """
    Abstract base class for postprocessing plugins.
    
    All postprocessing plugins must inherit from this class and implement
    the required methods according to the behavioral contracts specified in the
    unit-level design documentation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the postprocessing plugin.
        
        Args:
            config: Plugin-specific configuration parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.metadata = self._generate_metadata()
        self.is_initialized = False
        self.processing_history = []
        
    @abstractmethod
    def get_plugin_info(self) -> Dict[str, Any]:
        """
        Get plugin information and metadata.
        
        Returns:
            Dictionary containing plugin information:
            - name: Plugin name
            - version: Plugin version
            - description: Plugin description
            - author: Plugin author
            - dependencies: List of plugin dependencies
            - execution_conditions: Conditions for plugin execution
            - data_requirements: Data requirements
        """
        pass
    
    @abstractmethod
    def should_execute(self, data: Dict[str, pd.DataFrame], 
                      metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Determine if this plugin should execute based on data characteristics.
        
        Args:
            data: Dictionary of datasets to evaluate
            metadata: Optional metadata about the data
            
        Returns:
            True if plugin should execute, False otherwise
        """
        pass
    
    @abstractmethod
    def validate_input(self, data: Dict[str, pd.DataFrame]) -> Tuple[bool, List[str]]:
        """
        Validate input data meets plugin requirements.
        
        Args:
            data: Dictionary of input DataFrames to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        pass
    
    @abstractmethod
    def postprocess(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Apply postprocessing transformations to datasets.
        
        Args:
            data: Dictionary of datasets to postprocess
            
        Returns:
            Dictionary of postprocessed datasets
            
        Behavior:
        - Apply transformations consistently across all datasets
        - Preserve data integrity and relationships
        - Handle missing data gracefully
        - Maintain temporal ordering if applicable
        """
        pass
    
    @abstractmethod
    def get_transformation_summary(self) -> Dict[str, Any]:
        """
        Get summary of transformations that will be applied.
        
        Returns:
            Dictionary describing the transformations
        """
        pass
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize the plugin with configuration.
        
        Args:
            config: Plugin configuration parameters
            
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            if config:
                self.config.update(config)
            
            # Validate configuration
            validation_result = self._validate_configuration()
            if not validation_result[0]:
                self.logger.error(f"Configuration validation failed: {validation_result[1]}")
                return False
            
            # Perform plugin-specific initialization
            self._plugin_specific_initialization()
            
            self.is_initialized = True
            self.logger.info(f"Plugin {self.__class__.__name__} initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Plugin initialization failed: {e}")
            self.is_initialized = False
            return False
    
    def cleanup(self) -> None:
        """
        Clean up plugin resources.
        
        Behavior:
        - Release any held resources
        - Clear temporary data
        - Reset plugin state if needed
        """
        try:
            self._plugin_specific_cleanup()
        except Exception as e:
            self.logger.error(f"Plugin cleanup failed: {e}")
        finally:
            # Always reset state regardless of cleanup success
            self.processing_history.clear()
            self.is_initialized = False
            self.logger.info(f"Plugin {self.__class__.__name__} cleaned up successfully")
    
    def process(self, data: Dict[str, pd.DataFrame], 
               metadata: Optional[Dict[str, Any]] = None) -> Dict[str, pd.DataFrame]:
        """
        Main processing method that orchestrates postprocessing.
        
        Args:
            data: Dictionary of input datasets
            metadata: Optional metadata about the data
            
        Returns:
            Dictionary of postprocessed datasets
            
        Raises:
            RuntimeError: If plugin is not initialized
            ValueError: If input validation fails
        """
        if not self.is_initialized:
            raise RuntimeError(f"Plugin {self.__class__.__name__} is not initialized")
        
        # Check if plugin should execute
        if not self.should_execute(data, metadata):
            self.logger.info(f"Plugin {self.__class__.__name__} skipped - execution conditions not met")
            return data.copy()
        
        # Validate input
        is_valid, errors = self.validate_input(data)
        if not is_valid:
            raise ValueError(f"Input validation failed: {errors}")
        
        # Record processing start
        processing_start = datetime.now()
        
        try:
            # Apply postprocessing
            result = self.postprocess(data)
            
            # Validate output
            if not self._validate_output(data, result):
                raise ValueError("Output validation failed")
            
            # Record processing history
            self._record_processing(data, result, processing_start, metadata)
            
            self.logger.info(f"Postprocessing completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Postprocessing failed: {e}")
            raise
    
    def get_processing_history(self) -> List[Dict[str, Any]]:
        """
        Get processing history for this plugin.
        
        Returns:
            List of processing history entries
        """
        return self.processing_history.copy()
    
    def _generate_metadata(self) -> Dict[str, Any]:
        """Generate plugin metadata."""
        return {
            'class_name': self.__class__.__name__,
            'module': self.__class__.__module__,
            'created_at': datetime.now().isoformat(),
            'plugin_type': 'postprocessing'
        }
    
    def _validate_configuration(self) -> Tuple[bool, List[str]]:
        """
        Validate plugin configuration.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        # Default implementation - plugins can override
        return True, []
    
    def _plugin_specific_initialization(self) -> None:
        """Plugin-specific initialization logic."""
        # Default implementation - plugins can override
        pass
    
    def _plugin_specific_cleanup(self) -> None:
        """Plugin-specific cleanup logic."""
        # Default implementation - plugins can override
        pass
    
    def _validate_output(self, input_data: Dict[str, pd.DataFrame], 
                        output_data: Dict[str, pd.DataFrame]) -> bool:
        """
        Validate output data.
        
        Args:
            input_data: Original input data
            output_data: Processed output data
            
        Returns:
            True if output is valid, False otherwise
        """
        try:
            # Check that output is a dictionary
            if not isinstance(output_data, dict):
                self.logger.error("Output is not a dictionary")
                return False
            
            # Check that all input datasets are present in output
            missing_datasets = set(input_data.keys()) - set(output_data.keys())
            if missing_datasets:
                self.logger.error(f"Missing datasets in output: {missing_datasets}")
                return False
            
            # Check each dataset
            for dataset_name, input_df in input_data.items():
                if dataset_name not in output_data:
                    continue
                
                output_df = output_data[dataset_name]
                
                # Check that output is a DataFrame
                if not isinstance(output_df, pd.DataFrame):
                    self.logger.error(f"Output dataset '{dataset_name}' is not a pandas DataFrame")
                    return False
                
                # Check basic structural integrity (can be overridden by plugins)
                preserve_structure = self.config.get('preserve_data_structure', True)
                if preserve_structure:
                    if len(output_df.columns) != len(input_df.columns):
                        self.logger.warning(f"Dataset '{dataset_name}' has different number of columns")
                    
                    if len(output_df) != len(input_df):
                        self.logger.warning(f"Dataset '{dataset_name}' has different number of rows")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Output validation error: {e}")
            return False
    
    def _record_processing(self, input_data: Dict[str, pd.DataFrame], 
                          output_data: Dict[str, pd.DataFrame], 
                          start_time: datetime,
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record processing history entry."""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate dataset changes
        dataset_changes = {}
        for dataset_name in input_data.keys():
            if dataset_name in output_data:
                input_shape = input_data[dataset_name].shape
                output_shape = output_data[dataset_name].shape
                dataset_changes[dataset_name] = {
                    'input_shape': input_shape,
                    'output_shape': output_shape,
                    'rows_changed': output_shape[0] - input_shape[0],
                    'columns_changed': output_shape[1] - input_shape[1]
                }
        
        history_entry = {
            'timestamp': start_time.isoformat(),
            'processing_time_seconds': processing_time,
            'datasets_processed': len(input_data),
            'dataset_changes': dataset_changes,
            'metadata': metadata,
            'config_hash': hash(str(sorted(self.config.items())))
        }
        
        self.processing_history.append(history_entry)


class PostprocessingPipeline:
    """
    Pipeline for executing multiple postprocessing plugins in sequence.
    """
    
    def __init__(self, plugins: Optional[List[PostprocessingPlugin]] = None):
        """
        Initialize postprocessing pipeline.
        
        Args:
            plugins: List of postprocessing plugins
        """
        self.plugins = plugins or []
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.processing_history = []
    
    def add_plugin(self, plugin: PostprocessingPlugin) -> None:
        """Add a plugin to the pipeline."""
        if not isinstance(plugin, PostprocessingPlugin):
            raise TypeError("Plugin must inherit from PostprocessingPlugin")
        
        self.plugins.append(plugin)
        self.logger.info(f"Added plugin {plugin.__class__.__name__} to pipeline")
    
    def process(self, data: Dict[str, pd.DataFrame], 
               metadata: Optional[Dict[str, Any]] = None) -> Dict[str, pd.DataFrame]:
        """
        Process data through the postprocessing pipeline.
        
        Args:
            data: Dictionary of input datasets
            metadata: Optional metadata about the data
            
        Returns:
            Dictionary of postprocessed datasets
        """
        if not self.plugins:
            self.logger.warning("No plugins in pipeline, returning original data")
            return {k: v.copy() for k, v in data.items()}
        
        result = {k: v.copy() for k, v in data.items()}
        pipeline_start = datetime.now()
        plugin_results = []
        
        for plugin in self.plugins:
            plugin_start = datetime.now()
            
            try:
                result = plugin.process(result, metadata)
                plugin_time = (datetime.now() - plugin_start).total_seconds()
                
                plugin_results.append({
                    'plugin_name': plugin.__class__.__name__,
                    'processing_time_seconds': plugin_time,
                    'executed': True,
                    'success': True
                })
                
                self.logger.info(f"Plugin {plugin.__class__.__name__} completed successfully")
                
            except Exception as e:
                self.logger.error(f"Plugin {plugin.__class__.__name__} failed: {e}")
                
                plugin_results.append({
                    'plugin_name': plugin.__class__.__name__,
                    'processing_time_seconds': (datetime.now() - plugin_start).total_seconds(),
                    'executed': True,
                    'success': False,
                    'error': str(e)
                })
                
                # Continue with next plugin
                continue
        
        # Record pipeline processing history
        pipeline_time = (datetime.now() - pipeline_start).total_seconds()
        
        history_entry = {
            'timestamp': pipeline_start.isoformat(),
            'total_processing_time_seconds': pipeline_time,
            'input_datasets': len(data),
            'output_datasets': len(result),
            'plugins_executed': len([r for r in plugin_results if r['executed']]),
            'plugins_succeeded': len([r for r in plugin_results if r['success']]),
            'plugins_failed': len([r for r in plugin_results if r.get('executed') and not r['success']]),
            'plugin_results': plugin_results
        }
        
        self.processing_history.append(history_entry)
        
        self.logger.info(f"Pipeline processing completed successfully")
        
        return result
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline and its plugins."""
        return {
            'plugin_count': len(self.plugins),
            'plugins': [plugin.get_plugin_info() for plugin in self.plugins],
            'total_processing_runs': len(self.processing_history)
        }
    
    def initialize_plugins(self) -> bool:
        """Initialize all plugins in the pipeline."""
        success = True
        
        for plugin in self.plugins:
            if not plugin.initialize():
                self.logger.error(f"Failed to initialize plugin {plugin.__class__.__name__}")
                success = False
        
        return success
    
    def cleanup_plugins(self) -> None:
        """Clean up all plugins in the pipeline."""
        for plugin in self.plugins:
            try:
                plugin.cleanup()
            except Exception as e:
                self.logger.error(f"Failed to cleanup plugin {plugin.__class__.__name__}: {e}")
    
    def get_execution_plan(self, data: Dict[str, pd.DataFrame], 
                          metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Get execution plan showing which plugins will execute.
        
        Args:
            data: Dictionary of datasets to evaluate
            metadata: Optional metadata about the data
            
        Returns:
            List of execution plan entries for each plugin
        """
        plan = []
        
        for plugin in self.plugins:
            try:
                will_execute = plugin.should_execute(data, metadata)
                plan.append({
                    'plugin_name': plugin.__class__.__name__,
                    'will_execute': will_execute,
                    'plugin_info': plugin.get_plugin_info(),
                    'transformation_summary': plugin.get_transformation_summary() if will_execute else None
                })
            except Exception as e:
                plan.append({
                    'plugin_name': plugin.__class__.__name__,
                    'will_execute': False,
                    'error': f"Error evaluating execution conditions: {e}"
                })
        
        return plan
