"""Unit Tests for Feature Engineering Plugin Base Classes

This module implements comprehensive unit tests for the feature engineering plugin
base classes, following BDD methodology and testing all behavioral contracts.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import logging
from datetime import datetime
from unittest.mock import MagicMock, patch

from app.core.feature_engineering_plugin_base import FeatureEngineeringPlugin, FeatureEngineeringPipeline


class MockFeatureEngineeringPlugin(FeatureEngineeringPlugin):
    """Mock plugin for testing"""
    
    def get_plugin_info(self):
        return {
            'name': 'MockPlugin',
            'version': '1.0.0',
            'description': 'Mock plugin for testing',
            'author': 'Test Author',
            'dependencies': [],
            'input_requirements': ['numeric_data'],
            'output_schema': ['feature_1', 'feature_2', 'mock_feature']
        }
    
    def validate_input(self, data):
        if data.empty:
            return False, ["Data is empty"]
        if len(data.columns) == 0:
            return False, ["No columns in data"]
        return True, []
    
    def engineer_features(self, data):
        result = data.copy()
        # Use plugin_id to make feature names unique
        plugin_id = self.config.get('plugin_id', 'default')
        feature_name = f'mock_feature_{plugin_id}'
        result[feature_name] = result.iloc[:, 0] * 2  # Simple feature engineering
        return result
    
    def get_output_features(self):
        plugin_id = self.config.get('plugin_id', 'default')
        return [f'mock_feature_{plugin_id}']


class FailingMockPlugin(FeatureEngineeringPlugin):
    """Mock plugin that fails for testing error handling"""
    
    def get_plugin_info(self):
        return {
            'name': 'FailingMockPlugin',
            'version': '1.0.0',
            'description': 'Plugin that fails',
            'author': 'Test Author',
            'dependencies': [],
            'input_requirements': [],
            'output_schema': []
        }
    
    def validate_input(self, data):
        return True, []
    
    def engineer_features(self, data):
        raise ValueError("Simulated plugin failure")
    
    def get_output_features(self):
        return []


class TestFeatureEngineeringPluginBase(unittest.TestCase):
    """Test FeatureEngineeringPlugin base class behaviors"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.plugin = MockFeatureEngineeringPlugin()
        self.test_data = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100)
        })
    
    def test_plugin_initialization_default(self):
        """Test default plugin initialization"""
        plugin = MockFeatureEngineeringPlugin()
        
        self.assertIsNotNone(plugin.config)
        self.assertIsInstance(plugin.config, dict)
        self.assertIsNotNone(plugin.logger)
        self.assertIsNotNone(plugin.metadata)
        self.assertFalse(plugin.is_initialized)
        self.assertEqual(len(plugin.processing_history), 0)
    
    def test_plugin_initialization_with_config(self):
        """Test plugin initialization with custom config"""
        config = {'param1': 'value1', 'param2': 42}
        plugin = MockFeatureEngineeringPlugin(config)
        
        self.assertEqual(plugin.config, config)
        self.assertFalse(plugin.is_initialized)
    
    def test_plugin_metadata_generation(self):
        """Test plugin metadata generation"""
        metadata = self.plugin.metadata
        
        self.assertIn('class_name', metadata)
        self.assertIn('module', metadata)
        self.assertIn('created_at', metadata)
        self.assertIn('plugin_type', metadata)
        self.assertEqual(metadata['class_name'], 'MockFeatureEngineeringPlugin')
        self.assertEqual(metadata['plugin_type'], 'feature_engineering')
    
    def test_initialize_plugin_success(self):
        """Test successful plugin initialization"""
        result = self.plugin.initialize()
        
        self.assertTrue(result)
        self.assertTrue(self.plugin.is_initialized)
    
    def test_initialize_plugin_with_config_update(self):
        """Test plugin initialization with config update"""
        initial_config = {'param1': 'value1'}
        plugin = MockFeatureEngineeringPlugin(initial_config)
        
        update_config = {'param2': 'value2'}
        result = plugin.initialize(update_config)
        
        self.assertTrue(result)
        self.assertEqual(plugin.config['param1'], 'value1')
        self.assertEqual(plugin.config['param2'], 'value2')
    
    def test_cleanup_plugin(self):
        """Test plugin cleanup"""
        # Initialize plugin first
        self.plugin.initialize()
        self.plugin.processing_history.append({'test': 'data'})
        
        # Clean up
        self.plugin.cleanup()
        
        self.assertFalse(self.plugin.is_initialized)
        self.assertEqual(len(self.plugin.processing_history), 0)
    
    def test_process_uninitialized_plugin(self):
        """Test processing with uninitialized plugin"""
        with self.assertRaises(RuntimeError) as context:
            self.plugin.process(self.test_data)
        
        self.assertIn("not initialized", str(context.exception))
    
    def test_process_success(self):
        """Test successful feature engineering processing"""
        self.plugin.initialize()
        
        result = self.plugin.process(self.test_data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.test_data))
        self.assertIn('mock_feature_default', result.columns)
        self.assertTrue(len(result.columns) > len(self.test_data.columns))
    
    def test_process_input_validation_failure(self):
        """Test processing with input validation failure"""
        self.plugin.initialize()
        
        # Use empty DataFrame to trigger validation failure
        empty_data = pd.DataFrame()
        
        with self.assertRaises(ValueError) as context:
            self.plugin.process(empty_data)
        
        self.assertIn("Input validation failed", str(context.exception))
    
    def test_process_records_history(self):
        """Test that processing records history"""
        self.plugin.initialize()
        initial_history_count = len(self.plugin.processing_history)
        
        self.plugin.process(self.test_data)
        
        self.assertEqual(len(self.plugin.processing_history), initial_history_count + 1)
        
        history_entry = self.plugin.processing_history[-1]
        self.assertIn('timestamp', history_entry)
        self.assertIn('processing_time_seconds', history_entry)
        self.assertIn('input_shape', history_entry)
        self.assertIn('output_shape', history_entry)
        self.assertIn('features_added', history_entry)
        self.assertIn('new_features', history_entry)
    
    def test_get_processing_history(self):
        """Test getting processing history"""
        self.plugin.initialize()
        
        # Process some data
        self.plugin.process(self.test_data)
        
        history = self.plugin.get_processing_history()
        
        self.assertIsInstance(history, list)
        self.assertEqual(len(history), 1)
        self.assertIsInstance(history[0], dict)
        
        # Verify it's a copy (modifying returned history shouldn't affect original)
        history.clear()
        self.assertEqual(len(self.plugin.get_processing_history()), 1)
    
    def test_output_validation_invalid_type(self):
        """Test output validation with invalid type"""
        self.plugin.initialize()
        
        # Mock engineer_features to return invalid type
        with patch.object(self.plugin, 'engineer_features', return_value="invalid"):
            with self.assertRaises(ValueError) as context:
                self.plugin.process(self.test_data)
            
            self.assertIn("Output validation failed", str(context.exception))
    
    def test_output_validation_fewer_rows(self):
        """Test output validation with fewer rows"""
        self.plugin.initialize()
        
        # Mock engineer_features to return DataFrame with fewer rows
        fewer_rows_df = self.test_data.iloc[:50].copy()
        with patch.object(self.plugin, 'engineer_features', return_value=fewer_rows_df):
            with self.assertRaises(ValueError) as context:
                self.plugin.process(self.test_data)
            
            self.assertIn("Output validation failed", str(context.exception))
    
    def test_preserve_original_features_default(self):
        """Test that original features are preserved by default"""
        self.plugin.initialize()
        
        result = self.plugin.process(self.test_data)
        
        # All original columns should be present
        for col in self.test_data.columns:
            self.assertIn(col, result.columns)
    
    def test_get_output_features_interface(self):
        """Test get_output_features interface"""
        features = self.plugin.get_output_features()
        
        self.assertIsInstance(features, list)
        self.assertEqual(features, ['mock_feature_default'])


class TestFeatureEngineeringPipeline(unittest.TestCase):
    """Test FeatureEngineeringPipeline behaviors"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.pipeline = FeatureEngineeringPipeline()
        self.test_data = pd.DataFrame({
            'feature_1': np.random.randn(50),
            'feature_2': np.random.randn(50)
        })
        
        # Create multiple plugins
        self.plugin1 = MockFeatureEngineeringPlugin({'plugin_id': 1})
        self.plugin2 = MockFeatureEngineeringPlugin({'plugin_id': 2})
    
    def test_pipeline_initialization_empty(self):
        """Test empty pipeline initialization"""
        pipeline = FeatureEngineeringPipeline()
        
        self.assertEqual(len(pipeline.plugins), 0)
        self.assertIsNotNone(pipeline.logger)
        self.assertEqual(len(pipeline.processing_history), 0)
    
    def test_pipeline_initialization_with_plugins(self):
        """Test pipeline initialization with plugins"""
        plugins = [self.plugin1, self.plugin2]
        pipeline = FeatureEngineeringPipeline(plugins)
        
        self.assertEqual(len(pipeline.plugins), 2)
        self.assertIs(pipeline.plugins[0], self.plugin1)
        self.assertIs(pipeline.plugins[1], self.plugin2)
    
    def test_add_plugin_valid(self):
        """Test adding valid plugin to pipeline"""
        self.pipeline.add_plugin(self.plugin1)
        
        self.assertEqual(len(self.pipeline.plugins), 1)
        self.assertIs(self.pipeline.plugins[0], self.plugin1)
    
    def test_add_plugin_invalid_type(self):
        """Test adding invalid plugin type to pipeline"""
        with self.assertRaises(TypeError) as context:
            self.pipeline.add_plugin("not a plugin")
        
        self.assertIn("must inherit from FeatureEngineeringPlugin", str(context.exception))
    
    def test_process_empty_pipeline(self):
        """Test processing with empty pipeline"""
        with patch.object(self.pipeline.logger, 'warning') as mock_warning:
            result = self.pipeline.process(self.test_data)
        
        mock_warning.assert_called_once()
        pd.testing.assert_frame_equal(result, self.test_data)
    
    def test_process_single_plugin(self):
        """Test processing with single plugin"""
        self.plugin1.initialize()
        self.pipeline.add_plugin(self.plugin1)
        
        result = self.pipeline.process(self.test_data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.test_data))
        self.assertIn('mock_feature_1', result.columns)
        self.assertEqual(len(self.pipeline.processing_history), 1)
    
    def test_process_multiple_plugins(self):
        """Test processing with multiple plugins in sequence"""
        self.plugin1.initialize()
        self.plugin2.initialize()
        self.pipeline.add_plugin(self.plugin1)
        self.pipeline.add_plugin(self.plugin2)
        
        result = self.pipeline.process(self.test_data)
        
        # Should have original features plus features from both plugins
        expected_columns = len(self.test_data.columns) + 2  # Each plugin adds 1 feature
        self.assertEqual(len(result.columns), expected_columns)
        
        # Check processing history
        self.assertEqual(len(self.pipeline.processing_history), 1)
        history = self.pipeline.processing_history[0]
        self.assertEqual(history['plugins_executed'], 2)
        self.assertEqual(history['plugins_failed'], 0)
    
    def test_process_with_plugin_failure(self):
        """Test processing with plugin failure"""
        failing_plugin = FailingMockPlugin()
        failing_plugin.initialize()
        
        self.plugin1.initialize()
        
        self.pipeline.add_plugin(self.plugin1)
        self.pipeline.add_plugin(failing_plugin)
        
        # Processing should continue despite plugin failure
        result = self.pipeline.process(self.test_data)
        
        # Should have features from successful plugin
        self.assertIn('mock_feature_1', result.columns)
        
        # Check processing history reflects the failure
        history = self.pipeline.processing_history[0]
        self.assertEqual(history['plugins_executed'], 1)
        self.assertEqual(history['plugins_failed'], 1)
        
        # Check plugin results
        plugin_results = history['plugin_results']
        self.assertEqual(len(plugin_results), 2)
        self.assertTrue(plugin_results[0]['success'])
        self.assertFalse(plugin_results[1]['success'])
        self.assertIn('error', plugin_results[1])
    
    def test_get_pipeline_info(self):
        """Test getting pipeline information"""
        self.pipeline.add_plugin(self.plugin1)
        self.pipeline.add_plugin(self.plugin2)
        
        info = self.pipeline.get_pipeline_info()
        
        self.assertIn('plugin_count', info)
        self.assertIn('plugins', info)
        self.assertIn('total_processing_runs', info)
        
        self.assertEqual(info['plugin_count'], 2)
        self.assertEqual(len(info['plugins']), 2)
        self.assertEqual(info['total_processing_runs'], 0)  # No processing runs yet
    
    def test_initialize_plugins_success(self):
        """Test successful initialization of all plugins"""
        self.pipeline.add_plugin(self.plugin1)
        self.pipeline.add_plugin(self.plugin2)
        
        result = self.pipeline.initialize_plugins()
        
        self.assertTrue(result)
        self.assertTrue(self.plugin1.is_initialized)
        self.assertTrue(self.plugin2.is_initialized)
    
    def test_initialize_plugins_with_failure(self):
        """Test plugin initialization with some failures"""
        # Create a plugin that will fail initialization
        failing_plugin = FailingMockPlugin()
        
        # Mock the initialize method to fail
        with patch.object(failing_plugin, 'initialize', return_value=False):
            self.pipeline.add_plugin(self.plugin1)
            self.pipeline.add_plugin(failing_plugin)
            
            result = self.pipeline.initialize_plugins()
            
            self.assertFalse(result)  # Should return False if any plugin fails
    
    def test_cleanup_plugins(self):
        """Test cleanup of all plugins in pipeline"""
        self.plugin1.initialize()
        self.plugin2.initialize()
        
        self.pipeline.add_plugin(self.plugin1)
        self.pipeline.add_plugin(self.plugin2)
        
        self.pipeline.cleanup_plugins()
        
        self.assertFalse(self.plugin1.is_initialized)
        self.assertFalse(self.plugin2.is_initialized)
    
    def test_cleanup_plugins_with_failure(self):
        """Test cleanup continues even if some plugins fail"""
        self.plugin1.initialize()
        self.plugin2.initialize()
        
        # Mock plugin1 cleanup to fail
        with patch.object(self.plugin1, 'cleanup', side_effect=Exception("Cleanup failed")):
            self.pipeline.add_plugin(self.plugin1)
            self.pipeline.add_plugin(self.plugin2)
            
            # Should not raise exception
            self.pipeline.cleanup_plugins()
            
            # plugin2 should still be cleaned up
            self.assertFalse(self.plugin2.is_initialized)
    
    def test_processing_history_tracking(self):
        """Test that pipeline tracks processing history correctly"""
        self.plugin1.initialize()
        self.pipeline.add_plugin(self.plugin1)
        
        # Process multiple times
        self.pipeline.process(self.test_data)
        self.pipeline.process(self.test_data)
        
        self.assertEqual(len(self.pipeline.processing_history), 2)
        
        for history_entry in self.pipeline.processing_history:
            self.assertIn('timestamp', history_entry)
            self.assertIn('total_processing_time_seconds', history_entry)
            self.assertIn('input_shape', history_entry)
            self.assertIn('output_shape', history_entry)
            self.assertIn('total_features_added', history_entry)


class TestFeatureEngineeringPluginErrorHandling(unittest.TestCase):
    """Test error handling in feature engineering plugins"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.plugin = MockFeatureEngineeringPlugin()
        self.test_data = pd.DataFrame({
            'feature_1': np.random.randn(50),
            'feature_2': np.random.randn(50)
        })
    
    def test_plugin_configuration_validation_failure(self):
        """Test plugin initialization with configuration validation failure"""
        # Mock configuration validation to fail
        with patch.object(self.plugin, '_validate_configuration', return_value=(False, ["Invalid config"])):
            result = self.plugin.initialize()
            
            self.assertFalse(result)
            self.assertFalse(self.plugin.is_initialized)
    
    def test_plugin_initialization_exception(self):
        """Test plugin initialization with exception"""
        # Mock plugin-specific initialization to raise exception
        with patch.object(self.plugin, '_plugin_specific_initialization', side_effect=Exception("Init failed")):
            result = self.plugin.initialize()
            
            self.assertFalse(result)
            self.assertFalse(self.plugin.is_initialized)
    
    def test_plugin_cleanup_exception(self):
        """Test plugin cleanup with exception"""
        self.plugin.initialize()
        
        # Mock plugin-specific cleanup to raise exception
        with patch.object(self.plugin, '_plugin_specific_cleanup', side_effect=Exception("Cleanup failed")):
            # Should not raise exception
            self.plugin.cleanup()
            
            # Should still mark as not initialized
            self.assertFalse(self.plugin.is_initialized)
    
    def test_feature_engineering_exception(self):
        """Test feature engineering with exception"""
        self.plugin.initialize()
        
        # Mock engineer_features to raise exception
        with patch.object(self.plugin, 'engineer_features', side_effect=Exception("Engineering failed")):
            with self.assertRaises(Exception) as context:
                self.plugin.process(self.test_data)
            
            self.assertIn("Engineering failed", str(context.exception))
    
    def test_output_validation_exception(self):
        """Test output validation with exception"""
        self.plugin.initialize()
        
        # Mock output validation to raise exception
        with patch.object(self.plugin, '_validate_output', side_effect=Exception("Validation failed")):
            with self.assertRaises(Exception) as context:
                self.plugin.process(self.test_data)
            
            self.assertIn("Validation failed", str(context.exception))


if __name__ == '__main__':
    unittest.main()
