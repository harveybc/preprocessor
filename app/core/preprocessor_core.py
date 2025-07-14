"""Preprocessor Core Integration Component

This module implements the main processing orchestrator that integrates all core units
following the integration-level design specifications from the BDD documentation.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from datetime import datetime
from pathlib import Path

from app.core.configuration_manager import ConfigurationManager
from app.core.plugin_loader import PluginLoader
from app.core.data_handler import DataHandler
from app.core.data_processor import DataProcessor
from app.core.normalization_handler import NormalizationHandler
from app.core.feature_engineering_plugin_base import FeatureEngineeringPipeline
from app.core.postprocessing_plugin_base import PostprocessingPipeline


class PreprocessorCore:
    """
    Main processing orchestrator that integrates all core preprocessing components.
    
    This class provides the primary interface for the preprocessing system,
    orchestrating data loading, splitting, normalization, feature engineering,
    and postprocessing through a unified API.
    """
    
    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        """
        Initialize the preprocessor core.
        
        Args:
            config_manager: Optional configuration manager instance
        """
        self.config_manager = config_manager or ConfigurationManager()
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # Initialize core components
        self.plugin_loader = PluginLoader()
        self.data_handler = DataHandler()
        self.data_processor = DataProcessor()
        self.normalization_handler = NormalizationHandler()
        
        # Initialize plugin pipelines
        self.feature_engineering_pipeline = FeatureEngineeringPipeline()
        self.postprocessing_pipeline = PostprocessingPipeline()
        
        # Processing state
        self.is_initialized = False
        self._data_loaded = False
        self.processing_history = []
        self.current_datasets = {}
        self.current_metadata = {}
        
        # If a configuration manager is provided and has configuration loaded, configure components
        if config_manager and hasattr(config_manager, 'merged_config') and config_manager.merged_config:
            self._configure_components()
            self.is_initialized = True
        
    def initialize(self, config_files: Optional[List[str]] = None,
                  cli_args: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize the preprocessor with configuration and plugins.
        
        Args:
            config_files: List of configuration file paths
            cli_args: Command-line arguments
            
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing preprocessor core...")
            
            # Load configuration
            if config_files:
                for config_file in config_files:
                    self.config_manager.load_from_file(config_file)
            
            if cli_args:
                self.config_manager.load_from_cli_args(cli_args)
            
            # Validate configuration
            if not self.config_manager.validate():
                self.logger.error("Configuration validation failed")
                return False
            
            # Initialize plugin loader
            try:
                plugin_config = self.config_manager.get_section('plugins')
            except (KeyError, ValueError):
                plugin_config = {}
            self._initialize_plugins(plugin_config)
            
            # Configure core components
            self._configure_components()
            
            self.is_initialized = True
            self.logger.info("Preprocessor core initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize preprocessor core: {e}")
            return False
    
    def process_data(self, input_path: str, output_path: str,
                    config_overrides: Optional[Dict[str, Any]] = None) -> bool:
        """
        Process data through the complete preprocessing pipeline.
        
        Args:
            input_path: Path to input data file
            output_path: Path for output files
            config_overrides: Optional configuration overrides
            
        Returns:
            True if processing successful, False otherwise
        """
        if not self.is_initialized:
            raise RuntimeError("Preprocessor core is not initialized")
        
        processing_start = datetime.now()
        
        try:
            self.logger.info(f"Starting data processing: {input_path} -> {output_path}")
            
            # Apply configuration overrides if provided
            if config_overrides:
                self._apply_config_overrides(config_overrides)
            
            # Step 1: Load and validate data
            self.logger.info("Step 1: Loading and validating data...")
            if not self._load_and_validate_data(input_path):
                return False
            
            # Step 2: Apply feature engineering
            self.logger.info("Step 2: Applying feature engineering...")
            if not self._apply_feature_engineering():
                return False
            
            # Step 3: Split data into datasets
            self.logger.info("Step 3: Splitting data into datasets...")
            if not self._split_data():
                return False
            
            # Step 4: Compute normalization parameters and normalize
            self.logger.info("Step 4: Computing normalization and normalizing...")
            if not self._normalize_data():
                return False
            
            # Step 5: Apply postprocessing
            self.logger.info("Step 5: Applying postprocessing...")
            if not self._apply_postprocessing():
                return False
            
            # Step 6: Export results
            self.logger.info("Step 6: Exporting results...")
            if not self._export_results(output_path):
                return False
            
            # Record processing history
            self._record_processing_completion(processing_start, input_path, output_path)
            
            self.logger.info("Data processing completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Data processing failed: {e}")
            self._record_processing_failure(processing_start, input_path, str(e))
            return False
    
    def get_processing_status(self) -> Dict[str, Any]:
        """
        Get current processing status and metrics.
        
        Returns:
            Dictionary containing processing status information
        """
        return {
            'is_initialized': self.is_initialized,
            'current_datasets': len(self.current_datasets),
            'processing_history_count': len(self.processing_history),
            'plugins_loaded': {
                'feature_engineering': len(self.feature_engineering_pipeline.plugins),
                'postprocessing': len(self.postprocessing_pipeline.plugins)
            },
            'current_metadata': self.current_metadata.copy()
        }
    
    def get_processing_history(self) -> List[Dict[str, Any]]:
        """Get processing history."""
        return self.processing_history.copy()
    
    def cleanup(self) -> None:
        """Clean up resources and reset state."""
        try:
            self.logger.info("Cleaning up preprocessor core...")
            
            # Cleanup plugin pipelines
            self.feature_engineering_pipeline.cleanup_plugins()
            self.postprocessing_pipeline.cleanup_plugins()
            
            # Clear state
            self.current_datasets.clear()
            self.current_metadata.clear()
            self.processing_history.clear()
            
            # Reset core components
            self.data_handler.clear_data()
            self.data_processor.clear_processing_state()
            self.normalization_handler.clear_parameters()
            
            self.is_initialized = False
            self.logger.info("Preprocessor core cleaned up successfully")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def _initialize_plugins(self, plugin_config: Dict[str, Any]) -> None:
        """Initialize and load plugins."""
        # Add plugin directories
        plugin_dirs = plugin_config.get('directories', [])
        for plugin_dir in plugin_dirs:
            self.plugin_loader.add_plugin_directory(plugin_dir)
        
        # Discover plugins
        self.plugin_loader.discover_plugins()
        
        # Load feature engineering plugins
        fe_config = plugin_config.get('feature_engineering', {})
        if isinstance(fe_config, dict) and fe_config.get('enabled', False):
            fe_plugins = fe_config.get('plugins', [])
            for plugin_name in fe_plugins:
                if self.plugin_loader.load_plugin(plugin_name):
                    plugin = self.plugin_loader.get_plugin(plugin_name)
                    if plugin:
                        self.feature_engineering_pipeline.add_plugin(plugin)
        elif isinstance(fe_config, list):
            # Support legacy list format
            for plugin_name in fe_config:
                if self.plugin_loader.load_plugin(plugin_name):
                    plugin = self.plugin_loader.get_plugin(plugin_name)
                    if plugin:
                        self.feature_engineering_pipeline.add_plugin(plugin)
        
        # Load postprocessing plugins
        pp_config = plugin_config.get('postprocessing', {})
        if isinstance(pp_config, dict) and pp_config.get('enabled', False):
            pp_plugins = pp_config.get('plugins', [])
            for plugin_name in pp_plugins:
                if self.plugin_loader.load_plugin(plugin_name):
                    plugin = self.plugin_loader.get_plugin(plugin_name)
                    if plugin:
                        self.postprocessing_pipeline.add_plugin(plugin)
        elif isinstance(pp_config, list):
            # Support legacy list format
            for plugin_name in pp_config:
                if self.plugin_loader.load_plugin(plugin_name):
                    plugin = self.plugin_loader.get_plugin(plugin_name)
                    if plugin:
                        self.postprocessing_pipeline.add_plugin(plugin)
        
        # Initialize plugin pipelines
        self.feature_engineering_pipeline.initialize_plugins()
        self.postprocessing_pipeline.initialize_plugins()
    
    def _configure_components(self) -> None:
        """Configure core components with loaded configuration."""
        # Configure data handler
        try:
            data_config = self.config_manager.get_section('data_handling')
        except (KeyError, ValueError):
            data_config = {}
        validation_rules = data_config.get('validation_rules', {})
        for rule_name, rule_value in validation_rules.items():
            self.data_handler.set_validation_rules({rule_name: rule_value})
        
        # Configure data processor
        try:
            split_config = self.config_manager.get_section('data_splitting')
        except (KeyError, ValueError):
            split_config = {}
        # Store split config for later use
        self.split_config_data = split_config
        
        # Configure normalization handler
        try:
            norm_config = self.config_manager.get_section('normalization')
        except (KeyError, ValueError):
            norm_config = {}
        tolerance = norm_config.get('tolerance', 1e-8)
        feature_exclusions = norm_config.get('feature_exclusions', [])
        self.normalization_handler = NormalizationHandler(tolerance)
        self.normalization_handler.set_feature_exclusions(feature_exclusions)
        
        # Configure and initialize plugins
        try:
            plugin_config = self.config_manager.get_section('plugins')
        except (KeyError, ValueError):
            plugin_config = {}
        self._initialize_plugins(plugin_config)
    
    def _load_and_validate_data(self, input_path: str) -> bool:
        """Load and validate input data."""
        try:
            # Load data
            if not self.data_handler.load_data(input_path):
                self.logger.error("Failed to load input data")
                return False
            
            # Validate data
            if not self.data_handler.validate_data_integrity():
                self.logger.error("Data validation failed")
                return False
            
            # Store metadata
            self.current_metadata.update({
                'input_path': input_path,
                'input_shape': self.data_handler.loaded_data.shape,
                'input_features': list(self.data_handler.loaded_data.columns),
                'load_timestamp': datetime.now().isoformat()
            })
            
            # Mark data as loaded
            self._data_loaded = True
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading/validating data: {e}")
            return False
    
    def _apply_feature_engineering(self) -> bool:
        """Apply feature engineering pipeline."""
        try:
            if len(self.feature_engineering_pipeline.plugins) == 0:
                self.logger.info("No feature engineering plugins configured, skipping")
                return True
            
            # Apply feature engineering
            original_data = self.data_handler.get_data()
            if original_data is None:
                self.logger.error("No data available for feature engineering")
                return False
                
            original_data = original_data.copy()
            engineered_data = self.feature_engineering_pipeline.process(original_data)
            
            # Update data handler with engineered features
            if not self.data_handler.update_data(engineered_data, update_metadata=True):
                self.logger.error("Failed to update data handler with engineered features")
                return False
            
            # Update metadata
            self.current_metadata.update({
                'features_added': len(engineered_data.columns) - len(original_data.columns),
                'final_features': list(engineered_data.columns),
                'feature_engineering_completed': datetime.now().isoformat()
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}")
            return False
    
    def _split_data(self) -> bool:
        """Split data into datasets."""
        try:
            # Set data in processor
            self.data_processor.set_data(self.data_handler.loaded_data)
            
            # Create split configuration from stored config
            from app.core.data_processor import SplitConfiguration
            split_config = None
            if hasattr(self, 'split_config_data') and self.split_config_data:
                split_ratios = self.split_config_data.get('split_ratios', {})
                if split_ratios:
                    split_config = SplitConfiguration(
                        ratios=split_ratios,
                        temporal_split=self.split_config_data.get('temporal_split', True),
                        temporal_column=self.split_config_data.get('temporal_column', 'timestamp'),
                        shuffle=self.split_config_data.get('shuffle', False),
                        random_seed=self.split_config_data.get('random_seed', None)
                    )
            
            # Execute split with configuration
            if split_config:
                split_result = self.data_processor.execute_split(split_config)
            else:
                split_result = self.data_processor.execute_split()
            
            # Store split datasets
            self.current_datasets = split_result.datasets
            
            # Update metadata
            self.current_metadata.update({
                'split_metadata': split_result.split_metadata,
                'datasets_created': len(split_result.datasets),
                'splitting_completed': datetime.now().isoformat()
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Data splitting failed: {e}")
            return False
    
    def _normalize_data(self) -> bool:
        """Compute normalization parameters and normalize datasets."""
        try:
            # Get training datasets configuration
            norm_config = self.config_manager.get_section('normalization')
            if not norm_config:
                norm_config = {}
            training_keys = norm_config.get('training_datasets', ['d1', 'd2'])
            
            # Extract training datasets
            training_datasets = {}
            for key in training_keys:
                if key in self.current_datasets:
                    training_datasets[key] = self.current_datasets[key]
            
            if not training_datasets:
                self.logger.error("No training datasets found for normalization")
                return False
            
            # Compute normalization parameters only if not already loaded
            if self.normalization_handler.parameters is None:
                parameters = self.normalization_handler.compute_parameters(training_datasets)
                
                # Persist parameters if configured
                storage_config = norm_config.get('storage', {})
                if storage_config:
                    self.normalization_handler.persist_parameters(storage_config)
            else:
                self.logger.info("Using existing normalization parameters (loaded from file)")
                parameters = self.normalization_handler.parameters
            
            # Apply normalization to all datasets
            self.current_datasets = self.normalization_handler.apply_normalization(self.current_datasets)
            
            # Update DataProcessor with normalized datasets
            self.data_processor.split_datasets = self.current_datasets.copy()
            
            # Validate normalization quality
            validation_results = self.normalization_handler.validate_normalization_quality(
                self.current_datasets, training_keys
            )
            
            # Update metadata
            self.current_metadata.update({
                'normalization_parameters': {
                    'feature_count': len(parameters.features),
                    'training_datasets': training_keys,
                    'computation_timestamp': parameters.computation_timestamp.isoformat()
                },
                'normalization_quality': validation_results,
                'normalization_completed': datetime.now().isoformat()
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Normalization failed: {e}")
            return False
    
    def _apply_postprocessing(self) -> bool:
        """Apply postprocessing pipeline."""
        try:
            if len(self.postprocessing_pipeline.plugins) == 0:
                self.logger.info("No postprocessing plugins configured, skipping")
                return True
            
            # Apply postprocessing
            processed_datasets = self.postprocessing_pipeline.process(
                self.current_datasets, self.current_metadata
            )
            
            # Update datasets
            self.current_datasets = processed_datasets
            
            # Update metadata
            self.current_metadata.update({
                'postprocessing_completed': datetime.now().isoformat()
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Postprocessing failed: {e}")
            return False
    
    def _export_results(self, output_path: str) -> bool:
        """Export final results."""
        try:
            # Export split datasets
            export_config = self.config_manager.get_section('export')
            if not export_config:
                export_config = {}
            format_type = export_config.get('format', 'csv')
            
            success = self.data_processor.export_split_datasets(output_path, format_type)
            
            if not success:
                self.logger.error("Failed to export split datasets")
                return False
            
            # Export normalization parameters if configured
            norm_export = export_config.get('export_normalization_params', True)
            if norm_export and self.normalization_handler.parameters:
                param_storage = {
                    'means_file': str(Path(output_path) / 'means.json'),
                    'stds_file': str(Path(output_path) / 'stds.json')
                }
                self.normalization_handler.persist_parameters(param_storage)
            
            # Update metadata
            self.current_metadata.update({
                'output_path': output_path,
                'export_format': format_type,
                'export_completed': datetime.now().isoformat()
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            return False
    
    def _apply_config_overrides(self, overrides: Dict[str, Any]) -> None:
        """Apply configuration overrides."""
        for key, value in overrides.items():
            self.config_manager.set_config_value(key, value)
        
        # Reconfigure components with new settings
        self._configure_components()
    
    def _record_processing_completion(self, start_time: datetime, 
                                    input_path: str, output_path: str) -> None:
        """Record successful processing completion."""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        history_entry = {
            'timestamp': start_time.isoformat(),
            'processing_time_seconds': processing_time,
            'input_path': input_path,
            'output_path': output_path,
            'status': 'success',
            'datasets_processed': len(self.current_datasets),
            'metadata': self.current_metadata.copy()
        }
        
        self.processing_history.append(history_entry)
    
    def _record_processing_failure(self, start_time: datetime, 
                                 input_path: str, error_message: str) -> None:
        """Record processing failure."""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        history_entry = {
            'timestamp': start_time.isoformat(),
            'processing_time_seconds': processing_time,
            'input_path': input_path,
            'status': 'failed',
            'error_message': error_message,
            'metadata': self.current_metadata.copy()
        }
        
        self.processing_history.append(history_entry)
    
    # CLI-compatible interface methods
    
    def load_data(self, input_path: str) -> bool:
        """
        Load data from the specified input path.
        
        Args:
            input_path: Path to the input data file
            
        Returns:
            bool: True if data loaded successfully, False otherwise
        """
        try:
            return self._load_and_validate_data(input_path)
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            return False
    
    def process_data(self) -> bool:
        """
        Process the loaded data through the complete preprocessing pipeline.
        
        Returns:
            bool: True if processing completed successfully, False otherwise
        """
        try:
            if not hasattr(self, '_data_loaded') or not self._data_loaded:
                self.logger.error("No data loaded. Call load_data() first.")
                return False
            
            start_time = datetime.now()
            
            # Apply feature engineering
            if not self._apply_feature_engineering():
                self.logger.error("Feature engineering failed")
                return False
            
            # Split data
            if not self._split_data():
                self.logger.error("Data splitting failed")
                return False
            
            # Normalize data
            if not self._normalize_data():
                self.logger.error("Data normalization failed")
                return False
            
            # Apply postprocessing
            if not self._apply_postprocessing():
                self.logger.error("Postprocessing failed")
                return False
            
            # Log completion
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Processing completed successfully in {processing_time:.2f} seconds")
            return True
            
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            return False
    
    def export_results(self, output_path: str, format: str = 'csv', 
                      include_metadata: bool = False) -> bool:
        """
        Export the processed results to the specified output path.
        
        Args:
            output_path: Directory path for output files
            format: Output format ('csv', 'json', 'parquet')
            include_metadata: Whether to include processing metadata
            
        Returns:
            bool: True if export completed successfully, False otherwise
        """
        try:
            # Export the results using the internal method
            return self._export_results(output_path)
            
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            return False
