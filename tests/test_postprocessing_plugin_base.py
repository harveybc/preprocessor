"""Unit Tests for Postprocessing Plugin Base Classes

This module implements comprehensive unit tests for the postprocessing plugin
base classes, following BDD methodology and testing all behavioral contracts.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import logging
from datetime import datetime
from unittest.mock import MagicMock, patch

from app.core.postprocessing_plugin_base import PostprocessingPlugin, PostprocessingPipeline


class MockPostprocessingPlugin(PostprocessingPlugin):
    """Mock plugin for testing"""
    
    def get_plugin_info(self):
        return {
            'name': 'MockPostprocessingPlugin',
            'version': '1.0.0',
            'description': 'Mock postprocessing plugin for testing',
            'author': 'Test Author',
            'dependencies': [],
            'execution_conditions': ['always'],
            'data_requirements': ['numeric_data']
        }
    
    def should_execute(self, data, metadata=None):
        # Simple condition: execute if any dataset has more than 10 rows
        return any(len(df) > 10 for df in data.values())
    
    def validate_input(self, data):
        if not data:
            return False, ["No datasets provided"]
        
        for name, df in data.items():
            if df.empty:
                return False, [f"Dataset '{name}' is empty"]
        
        return True, []
    
    def postprocess(self, data):
        result = {}
        for name, df in data.items():
            processed_df = df.copy()
            # Simple postprocessing: add a processed flag column
            processed_df['postprocessed'] = True
            result[name] = processed_df
        return result
    
    def get_transformation_summary(self):
        return {
            'transformations': ['add_postprocessed_flag'],
            'affects_all_datasets': True,
            'preserves_structure': True
        }


class ConditionalMockPlugin(PostprocessingPlugin):
    """Mock plugin with conditional execution"""
    
    def get_plugin_info(self):
        return {
            'name': 'ConditionalMockPlugin',
            'version': '1.0.0',
            'description': 'Conditionally executing mock plugin',
            'author': 'Test Author',
            'dependencies': [],
            'execution_conditions': ['outliers_detected'],
            'data_requirements': []
        }
    
    def should_execute(self, data, metadata=None):
        # Execute only if metadata indicates outliers
        if metadata and metadata.get('outliers_detected', False):
            return True
        return False
    
    def validate_input(self, data):
        return True, []
    
    def postprocess(self, data):
        result = {}
        for name, df in data.items():
            processed_df = df.copy()
            processed_df['outlier_processed'] = True
            result[name] = processed_df
        return result
    
    def get_transformation_summary(self):
        return {
            'transformations': ['outlier_processing'],
            'conditional': True
        }


class FailingMockPlugin(PostprocessingPlugin):
    """Mock plugin that fails for testing error handling"""
    
    def get_plugin_info(self):
        return {
            'name': 'FailingMockPlugin',
            'version': '1.0.0',
            'description': 'Plugin that fails',
            'author': 'Test Author',
            'dependencies': [],
            'execution_conditions': [],
            'data_requirements': []
        }
    
    def should_execute(self, data, metadata=None):
        return True
    
    def validate_input(self, data):
        return True, []
    
    def postprocess(self, data):
        raise ValueError("Simulated plugin failure")
    
    def get_transformation_summary(self):
        return {'transformations': ['failing_transformation']}


class TestPostprocessingPluginBase(unittest.TestCase):
    """Test PostprocessingPlugin base class behaviors"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.plugin = MockPostprocessingPlugin()
        self.test_data = {
            'd1': pd.DataFrame({
                'feature_1': np.random.randn(50),
                'feature_2': np.random.randn(50)
            }),
            'd2': pd.DataFrame({
                'feature_1': np.random.randn(30),
                'feature_2': np.random.randn(30)
            })
        }
    
    def test_plugin_initialization_default(self):
        """Test default plugin initialization"""
        plugin = MockPostprocessingPlugin()
        
        self.assertIsNotNone(plugin.config)
        self.assertIsInstance(plugin.config, dict)
        self.assertIsNotNone(plugin.logger)
        self.assertIsNotNone(plugin.metadata)
        self.assertFalse(plugin.is_initialized)
        self.assertEqual(len(plugin.processing_history), 0)
    
    def test_plugin_initialization_with_config(self):
        """Test plugin initialization with custom config"""
        config = {'param1': 'value1', 'param2': 42}
        plugin = MockPostprocessingPlugin(config)
        
        self.assertEqual(plugin.config, config)
        self.assertFalse(plugin.is_initialized)
    
    def test_plugin_metadata_generation(self):
        """Test plugin metadata generation"""
        metadata = self.plugin.metadata
        
        self.assertIn('class_name', metadata)
        self.assertIn('module', metadata)
        self.assertIn('created_at', metadata)
        self.assertIn('plugin_type', metadata)
        self.assertEqual(metadata['class_name'], 'MockPostprocessingPlugin')
        self.assertEqual(metadata['plugin_type'], 'postprocessing')
    
    def test_initialize_plugin_success(self):
        """Test successful plugin initialization"""
        result = self.plugin.initialize()
        
        self.assertTrue(result)
        self.assertTrue(self.plugin.is_initialized)
    
    def test_initialize_plugin_with_config_update(self):
        """Test plugin initialization with config update"""
        initial_config = {'param1': 'value1'}
        plugin = MockPostprocessingPlugin(initial_config)
        
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
        """Test successful postprocessing"""
        self.plugin.initialize()
        
        result = self.plugin.process(self.test_data)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), len(self.test_data))
        
        # Check that all datasets have the postprocessed flag
        for dataset_name, df in result.items():
            self.assertIn('postprocessed', df.columns)
            self.assertTrue(all(df['postprocessed']))
    
    def test_process_skipped_execution(self):
        """Test processing when execution conditions are not met"""
        # Use small datasets to trigger condition failure
        small_data = {
            'd1': pd.DataFrame({'feature_1': [1, 2, 3]}),  # Only 3 rows
            'd2': pd.DataFrame({'feature_1': [4, 5]})      # Only 2 rows
        }
        
        self.plugin.initialize()
        
        result = self.plugin.process(small_data)
        
        # Should return original data unchanged
        self.assertEqual(len(result), len(small_data))
        for name, df in result.items():
            self.assertNotIn('postprocessed', df.columns)
    
    def test_process_input_validation_failure(self):
        """Test processing with input validation failure"""
        # Create a mock plugin that always executes but has strict validation
        class StrictValidationPlugin(MockPostprocessingPlugin):
            def should_execute(self, data, metadata=None):
                return True  # Always execute
            
            def validate_input(self, data):
                if not data:
                    return False, ["No datasets provided"]
                return False, ["Strict validation failure"]  # Always fail validation
        
        strict_plugin = StrictValidationPlugin()
        strict_plugin.initialize()
        
        with self.assertRaises(ValueError) as context:
            strict_plugin.process(self.test_data)
        
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
        self.assertIn('datasets_processed', history_entry)
        self.assertIn('dataset_changes', history_entry)
    
    def test_conditional_execution(self):
        """Test conditional plugin execution"""
        conditional_plugin = ConditionalMockPlugin()
        conditional_plugin.initialize()
        
        # Test without metadata (should not execute)
        result1 = conditional_plugin.process(self.test_data)
        for df in result1.values():
            self.assertNotIn('outlier_processed', df.columns)
        
        # Test with metadata indicating outliers (should execute)
        metadata = {'outliers_detected': True}
        result2 = conditional_plugin.process(self.test_data, metadata)
        for df in result2.values():
            self.assertIn('outlier_processed', df.columns)
    
    def test_get_processing_history(self):
        """Test getting processing history"""
        self.plugin.initialize()
        
        # Process some data
        self.plugin.process(self.test_data)
        
        history = self.plugin.get_processing_history()
        
        self.assertIsInstance(history, list)
        self.assertEqual(len(history), 1)
        self.assertIsInstance(history[0], dict)
        
        # Verify it's a copy
        history.clear()
        self.assertEqual(len(self.plugin.get_processing_history()), 1)
    
    def test_output_validation_invalid_type(self):
        """Test output validation with invalid type"""
        self.plugin.initialize()
        
        # Mock postprocess to return invalid type
        with patch.object(self.plugin, 'postprocess', return_value="invalid"):
            with self.assertRaises(ValueError) as context:
                self.plugin.process(self.test_data)
            
            self.assertIn("Output validation failed", str(context.exception))
    
    def test_output_validation_missing_datasets(self):
        """Test output validation with missing datasets"""
        self.plugin.initialize()
        
        # Mock postprocess to return incomplete result
        with patch.object(self.plugin, 'postprocess', return_value={'d1': self.test_data['d1']}):
            with self.assertRaises(ValueError) as context:
                self.plugin.process(self.test_data)
            
            self.assertIn("Output validation failed", str(context.exception))
    
    def test_get_transformation_summary_interface(self):
        """Test get_transformation_summary interface"""
        summary = self.plugin.get_transformation_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('transformations', summary)


class TestPostprocessingPipeline(unittest.TestCase):
    """Test PostprocessingPipeline behaviors"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.pipeline = PostprocessingPipeline()
        self.test_data = {
            'd1': pd.DataFrame({
                'feature_1': np.random.randn(50),
                'feature_2': np.random.randn(50)
            }),
            'd2': pd.DataFrame({
                'feature_1': np.random.randn(30),
                'feature_2': np.random.randn(30)
            })
        }
        
        # Create multiple plugins
        self.plugin1 = MockPostprocessingPlugin({'plugin_id': 1})
        self.plugin2 = MockPostprocessingPlugin({'plugin_id': 2})
    
    def test_pipeline_initialization_empty(self):
        """Test empty pipeline initialization"""
        pipeline = PostprocessingPipeline()
        
        self.assertEqual(len(pipeline.plugins), 0)
        self.assertIsNotNone(pipeline.logger)
        self.assertEqual(len(pipeline.processing_history), 0)
    
    def test_pipeline_initialization_with_plugins(self):
        """Test pipeline initialization with plugins"""
        plugins = [self.plugin1, self.plugin2]
        pipeline = PostprocessingPipeline(plugins)
        
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
        
        self.assertIn("must inherit from PostprocessingPlugin", str(context.exception))
    
    def test_process_empty_pipeline(self):
        """Test processing with empty pipeline"""
        with patch.object(self.pipeline.logger, 'warning') as mock_warning:
            result = self.pipeline.process(self.test_data)
        
        mock_warning.assert_called_once()
        
        # Should return copies of original data
        self.assertEqual(len(result), len(self.test_data))
        for name, df in result.items():
            pd.testing.assert_frame_equal(df, self.test_data[name])
    
    def test_process_single_plugin(self):
        """Test processing with single plugin"""
        self.plugin1.initialize()
        self.pipeline.add_plugin(self.plugin1)
        
        result = self.pipeline.process(self.test_data)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), len(self.test_data))
        
        # Check that postprocessing was applied
        for df in result.values():
            self.assertIn('postprocessed', df.columns)
        
        self.assertEqual(len(self.pipeline.processing_history), 1)
    
    def test_process_multiple_plugins(self):
        """Test processing with multiple plugins in sequence"""
        self.plugin1.initialize()
        self.plugin2.initialize()
        self.pipeline.add_plugin(self.plugin1)
        self.pipeline.add_plugin(self.plugin2)
        
        result = self.pipeline.process(self.test_data)
        
        # Both plugins should have been applied
        for df in result.values():
            self.assertIn('postprocessed', df.columns)
        
        # Check processing history
        self.assertEqual(len(self.pipeline.processing_history), 1)
        history = self.pipeline.processing_history[0]
        self.assertEqual(history['plugins_executed'], 2)
        self.assertEqual(history['plugins_succeeded'], 2)
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
        
        # Should have transformations from successful plugin
        for df in result.values():
            self.assertIn('postprocessed', df.columns)
        
        # Check processing history reflects the failure
        history = self.pipeline.processing_history[0]
        self.assertEqual(history['plugins_executed'], 2)
        self.assertEqual(history['plugins_succeeded'], 1)
        self.assertEqual(history['plugins_failed'], 1)
        
        # Check plugin results
        plugin_results = history['plugin_results']
        self.assertEqual(len(plugin_results), 2)
        self.assertTrue(plugin_results[0]['success'])
        self.assertFalse(plugin_results[1]['success'])
        self.assertIn('error', plugin_results[1])
    
    def test_process_with_metadata(self):
        """Test processing with metadata"""
        conditional_plugin = ConditionalMockPlugin()
        conditional_plugin.initialize()
        self.pipeline.add_plugin(conditional_plugin)
        
        metadata = {'outliers_detected': True}
        result = self.pipeline.process(self.test_data, metadata)
        
        # Plugin should have executed because of metadata
        for df in result.values():
            self.assertIn('outlier_processed', df.columns)
    
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
        self.assertEqual(info['total_processing_runs'], 0)
    
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
        failing_plugin = FailingMockPlugin()
        
        # Mock the initialize method to fail
        with patch.object(failing_plugin, 'initialize', return_value=False):
            self.pipeline.add_plugin(self.plugin1)
            self.pipeline.add_plugin(failing_plugin)
            
            result = self.pipeline.initialize_plugins()
            
            self.assertFalse(result)
    
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
    
    def test_get_execution_plan(self):
        """Test getting execution plan"""
        self.plugin1.initialize()
        conditional_plugin = ConditionalMockPlugin()
        conditional_plugin.initialize()
        
        self.pipeline.add_plugin(self.plugin1)
        self.pipeline.add_plugin(conditional_plugin)
        
        # Test without metadata (conditional plugin should not execute)
        plan1 = self.pipeline.get_execution_plan(self.test_data)
        
        self.assertEqual(len(plan1), 2)
        self.assertTrue(plan1[0]['will_execute'])  # Regular plugin
        self.assertFalse(plan1[1]['will_execute'])  # Conditional plugin
        
        # Test with metadata (both should execute)
        metadata = {'outliers_detected': True}
        plan2 = self.pipeline.get_execution_plan(self.test_data, metadata)
        
        self.assertTrue(plan2[0]['will_execute'])
        self.assertTrue(plan2[1]['will_execute'])
    
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
            self.assertIn('input_datasets', history_entry)
            self.assertIn('output_datasets', history_entry)


class TestPostprocessingPluginErrorHandling(unittest.TestCase):
    """Test error handling in postprocessing plugins"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.plugin = MockPostprocessingPlugin()
        self.test_data = {
            'd1': pd.DataFrame({
                'feature_1': np.random.randn(50),
                'feature_2': np.random.randn(50)
            })
        }
    
    def test_plugin_configuration_validation_failure(self):
        """Test plugin initialization with configuration validation failure"""
        with patch.object(self.plugin, '_validate_configuration', return_value=(False, ["Invalid config"])):
            result = self.plugin.initialize()
            
            self.assertFalse(result)
            self.assertFalse(self.plugin.is_initialized)
    
    def test_plugin_initialization_exception(self):
        """Test plugin initialization with exception"""
        with patch.object(self.plugin, '_plugin_specific_initialization', side_effect=Exception("Init failed")):
            result = self.plugin.initialize()
            
            self.assertFalse(result)
            self.assertFalse(self.plugin.is_initialized)
    
    def test_plugin_cleanup_exception(self):
        """Test plugin cleanup with exception"""
        self.plugin.initialize()
        
        with patch.object(self.plugin, '_plugin_specific_cleanup', side_effect=Exception("Cleanup failed")):
            # Should not raise exception
            self.plugin.cleanup()
            
            # Should still mark as not initialized
            self.assertFalse(self.plugin.is_initialized)
    
    def test_postprocessing_exception(self):
        """Test postprocessing with exception"""
        self.plugin.initialize()
        
        with patch.object(self.plugin, 'postprocess', side_effect=Exception("Postprocessing failed")):
            with self.assertRaises(Exception) as context:
                self.plugin.process(self.test_data)
            
            self.assertIn("Postprocessing failed", str(context.exception))
    
    def test_execution_condition_exception(self):
        """Test execution condition evaluation with exception"""
        plugin = ConditionalMockPlugin()
        plugin.initialize()
        
        with patch.object(plugin, 'should_execute', side_effect=Exception("Condition failed")):
            with self.assertRaises(Exception):
                plugin.process(self.test_data)
    
    def test_output_validation_exception(self):
        """Test output validation with exception"""
        self.plugin.initialize()
        
        with patch.object(self.plugin, '_validate_output', side_effect=Exception("Validation failed")):
            with self.assertRaises(Exception) as context:
                self.plugin.process(self.test_data)
            
            self.assertIn("Validation failed", str(context.exception))


if __name__ == '__main__':
    unittest.main()
