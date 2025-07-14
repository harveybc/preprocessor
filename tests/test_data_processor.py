"""Unit Tests for DataProcessor

This module implements comprehensive unit tests for the DataProcessor class,
following BDD methodology and testing all behavioral contracts specified in the design.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from app.core.data_processor import DataProcessor, SplitConfiguration, SplitResult, ProcessingState


class TestDataProcessorInitialization(unittest.TestCase):
    """Test DataProcessor initialization behaviors"""
    
    def test_default_initialization(self):
        """Test default DataProcessor initialization"""
        processor = DataProcessor()
        
        self.assertIsNotNone(processor.logger)
        self.assertIsNone(processor.random_seed)
        self.assertEqual(len(processor.processing_history), 0)
        self.assertIsNone(processor.current_data)
        self.assertIsNone(processor.split_datasets)
        self.assertIsNone(processor.split_metadata)
        
        # Check default split configuration
        self.assertIsNotNone(processor.default_split_config)
        self.assertEqual(len(processor.default_split_config.ratios), 6)
        self.assertAlmostEqual(sum(processor.default_split_config.ratios.values()), 1.0, places=3)
    
    def test_initialization_with_random_seed(self):
        """Test DataProcessor initialization with random seed"""
        processor = DataProcessor(random_seed=42)
        
        self.assertEqual(processor.random_seed, 42)
        self.assertEqual(processor.default_split_config.random_seed, 42)
    
    def test_default_split_configuration_validity(self):
        """Test that default split configuration is valid"""
        processor = DataProcessor()
        config = processor.default_split_config
        
        # Test ratio validity
        self.assertEqual(set(config.ratios.keys()), {'d1', 'd2', 'd3', 'd4', 'd5', 'd6'})
        self.assertAlmostEqual(sum(config.ratios.values()), 1.0, places=3)
        
        for key, value in config.ratios.items():
            self.assertGreater(value, 0)
            self.assertLess(value, 1)


class TestSplitConfiguration(unittest.TestCase):
    """Test SplitConfiguration validation behaviors"""
    
    def test_valid_split_configuration(self):
        """Test creation of valid split configuration"""
        ratios = {'d1': 0.4, 'd2': 0.2, 'd3': 0.2, 'd4': 0.1, 'd5': 0.05, 'd6': 0.05}
        config = SplitConfiguration(ratios=ratios, random_seed=42)
        
        self.assertEqual(config.ratios, ratios)
        self.assertEqual(config.random_seed, 42)
        self.assertTrue(config.shuffle)
        self.assertFalse(config.temporal_split)
    
    def test_invalid_ratio_count(self):
        """Test split configuration with wrong number of ratios"""
        ratios = {'d1': 0.5, 'd2': 0.5}  # Only 2 ratios instead of 6
        
        with self.assertRaises(ValueError) as context:
            SplitConfiguration(ratios=ratios)
        
        self.assertIn("exactly 6 dataset ratios", str(context.exception))
    
    def test_invalid_ratio_keys(self):
        """Test split configuration with invalid keys"""
        ratios = {'train': 0.4, 'val': 0.2, 'test': 0.2, 'other1': 0.1, 'other2': 0.05, 'other3': 0.05}
        
        with self.assertRaises(ValueError) as context:
            SplitConfiguration(ratios=ratios)
        
        self.assertIn("must have keys", str(context.exception))
    
    def test_invalid_ratio_sum(self):
        """Test split configuration with ratios not summing to 1.0"""
        ratios = {'d1': 0.4, 'd2': 0.2, 'd3': 0.2, 'd4': 0.1, 'd5': 0.05, 'd6': 0.1}  # Sum = 1.05
        
        with self.assertRaises(ValueError) as context:
            SplitConfiguration(ratios=ratios)
        
        self.assertIn("must sum to 1.0", str(context.exception))
    
    def test_negative_ratio_values(self):
        """Test split configuration with negative ratio values"""
        ratios = {'d1': 0.4, 'd2': -0.1, 'd3': 0.3, 'd4': 0.2, 'd5': 0.1, 'd6': 0.1}
        
        with self.assertRaises(ValueError) as context:
            SplitConfiguration(ratios=ratios)
        
        self.assertIn("must be between 0 and 1", str(context.exception))
    
    def test_temporal_split_configuration(self):
        """Test temporal split configuration"""
        ratios = {'d1': 0.4, 'd2': 0.2, 'd3': 0.2, 'd4': 0.1, 'd5': 0.05, 'd6': 0.05}
        config = SplitConfiguration(
            ratios=ratios,
            temporal_split=True,
            temporal_column='timestamp',
            shuffle=False
        )
        
        self.assertTrue(config.temporal_split)
        self.assertEqual(config.temporal_column, 'timestamp')
        self.assertFalse(config.shuffle)


class TestDataProcessorDataManagement(unittest.TestCase):
    """Test DataProcessor data management behaviors"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = DataProcessor(random_seed=42)
        self.test_data = self.create_test_data(1000, 5)
    
    def create_test_data(self, rows=100, columns=3):
        """Helper to create test data"""
        np.random.seed(42)
        data = pd.DataFrame({
            f'feature_{i}': np.random.randn(rows) for i in range(columns)
        })
        # Add timestamp column for temporal tests
        start_date = datetime(2023, 1, 1)
        data['timestamp'] = [start_date + timedelta(hours=i) for i in range(rows)]
        return data
    
    def test_set_data_valid(self):
        """Test setting valid data"""
        self.processor.set_data(self.test_data)
        
        self.assertIsNotNone(self.processor.current_data)
        self.assertEqual(len(self.processor.current_data), len(self.test_data))
        self.assertEqual(list(self.processor.current_data.columns), list(self.test_data.columns))
        self.assertEqual(len(self.processor.processing_history), 0)
    
    def test_set_data_none(self):
        """Test setting None data"""
        with self.assertRaises(ValueError) as context:
            self.processor.set_data(None)
        
        self.assertIn("cannot be None", str(context.exception))
    
    def test_set_data_empty(self):
        """Test setting empty data"""
        empty_data = pd.DataFrame()
        
        with self.assertRaises(ValueError) as context:
            self.processor.set_data(empty_data)
        
        self.assertIn("cannot be None or empty", str(context.exception))
    
    def test_set_data_small_dataset_warning(self):
        """Test warning for small datasets"""
        small_data = self.create_test_data(30, 3)  # Less than 60 samples
        
        with self.assertLogs(level='WARNING') as log:
            self.processor.set_data(small_data)
        
        self.assertTrue(any("only 30 samples" in message for message in log.output))
    
    def test_set_data_resets_state(self):
        """Test that setting new data resets processing state"""
        # First, set data and create some state
        self.processor.set_data(self.test_data)
        self.processor.split_datasets = {'d1': pd.DataFrame()}
        self.processor.split_metadata = {'test': 'metadata'}
        self.processor.processing_history.append(
            ProcessingState('test', (100, 5), (100, 5), 'test_transform', datetime.now(), 1.0, 100.0)
        )
        
        # Set new data
        new_data = self.create_test_data(500, 3)
        self.processor.set_data(new_data)
        
        # Check state is reset
        self.assertIsNone(self.processor.split_datasets)
        self.assertIsNone(self.processor.split_metadata)
        self.assertEqual(len(self.processor.processing_history), 0)


class TestDataProcessorSplitValidation(unittest.TestCase):
    """Test DataProcessor split validation behaviors"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = DataProcessor(random_seed=42)
        self.test_data = self.create_test_data(1000, 5)
        self.processor.set_data(self.test_data)
        
        self.valid_config = SplitConfiguration(
            ratios={'d1': 0.4, 'd2': 0.2, 'd3': 0.2, 'd4': 0.1, 'd5': 0.05, 'd6': 0.05}
        )
    
    def create_test_data(self, rows=100, columns=3):
        """Helper to create test data"""
        np.random.seed(42)
        data = pd.DataFrame({
            f'feature_{i}': np.random.randn(rows) for i in range(columns)
        })
        data['timestamp'] = pd.date_range('2023-01-01', periods=rows, freq='1h')
        return data
    
    def test_validate_split_configuration_valid(self):
        """Test validation of valid split configuration"""
        is_valid, errors = self.processor.validate_split_configuration(self.valid_config)
        
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_validate_split_configuration_no_data(self):
        """Test validation when no data is loaded"""
        processor_no_data = DataProcessor()
        
        is_valid, errors = processor_no_data.validate_split_configuration(self.valid_config)
        
        self.assertFalse(is_valid)
        self.assertIn("No data loaded", errors[0])
    
    def test_validate_split_insufficient_samples(self):
        """Test validation with insufficient samples for splitting"""
        # Create configuration that would result in too few samples per split
        small_data = self.create_test_data(30, 3)  # Only 30 samples
        processor = DataProcessor()
        processor.set_data(small_data)
        
        # Configuration requiring minimum 10 samples per split
        config = SplitConfiguration(
            ratios={'d1': 0.1, 'd2': 0.1, 'd3': 0.1, 'd4': 0.1, 'd5': 0.3, 'd6': 0.3}
        )
        
        is_valid, errors = processor.validate_split_configuration(config)
        
        self.assertFalse(is_valid)
        self.assertTrue(any("would have" in error and "minimum required is 10" in error for error in errors))
    
    def test_validate_temporal_split_no_column(self):
        """Test validation of temporal split without temporal column"""
        config = SplitConfiguration(
            ratios={'d1': 0.4, 'd2': 0.2, 'd3': 0.2, 'd4': 0.1, 'd5': 0.05, 'd6': 0.05},
            temporal_split=True
        )
        
        is_valid, errors = self.processor.validate_split_configuration(config)
        
        self.assertFalse(is_valid)
        self.assertIn("temporal_column to be specified", errors[0])
    
    def test_validate_temporal_split_invalid_column(self):
        """Test validation of temporal split with non-existent column"""
        config = SplitConfiguration(
            ratios={'d1': 0.4, 'd2': 0.2, 'd3': 0.2, 'd4': 0.1, 'd5': 0.05, 'd6': 0.05},
            temporal_split=True,
            temporal_column='nonexistent_column'
        )
        
        is_valid, errors = self.processor.validate_split_configuration(config)
        
        self.assertFalse(is_valid)
        self.assertIn("not found in data", errors[0])
    
    def test_validate_temporal_split_null_values(self):
        """Test validation of temporal split with null values in temporal column"""
        data_with_nulls = self.test_data.copy()
        data_with_nulls.loc[0:10, 'timestamp'] = pd.NaT
        self.processor.set_data(data_with_nulls)
        
        config = SplitConfiguration(
            ratios={'d1': 0.4, 'd2': 0.2, 'd3': 0.2, 'd4': 0.1, 'd5': 0.05, 'd6': 0.05},
            temporal_split=True,
            temporal_column='timestamp'
        )
        
        is_valid, errors = self.processor.validate_split_configuration(config)
        
        self.assertFalse(is_valid)
        self.assertIn("contains null values", errors[0])
    
    def test_validate_stratify_column_invalid(self):
        """Test validation of stratification with non-existent column"""
        config = SplitConfiguration(
            ratios={'d1': 0.4, 'd2': 0.2, 'd3': 0.2, 'd4': 0.1, 'd5': 0.05, 'd6': 0.05},
            stratify_column='nonexistent_column'
        )
        
        is_valid, errors = self.processor.validate_split_configuration(config)
        
        self.assertFalse(is_valid)
        self.assertIn("not found in data", errors[0])
    
    def test_validate_stratify_column_too_many_unique_values(self):
        """Test validation of stratification with too many unique values"""
        # Add a column with many unique values
        data_with_many_uniques = self.test_data.copy()
        data_with_many_uniques['unique_id'] = range(len(data_with_many_uniques))
        self.processor.set_data(data_with_many_uniques)
        
        config = SplitConfiguration(
            ratios={'d1': 0.4, 'd2': 0.2, 'd3': 0.2, 'd4': 0.1, 'd5': 0.05, 'd6': 0.05},
            stratify_column='unique_id'
        )
        
        is_valid, errors = self.processor.validate_split_configuration(config)
        
        self.assertFalse(is_valid)
        self.assertIn("too many unique values", errors[0])


class TestDataProcessorSplitExecution(unittest.TestCase):
    """Test DataProcessor split execution behaviors"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = DataProcessor(random_seed=42)
        self.test_data = self.create_test_data(1000, 5)
        self.processor.set_data(self.test_data)
        
        self.valid_config = SplitConfiguration(
            ratios={'d1': 0.4, 'd2': 0.2, 'd3': 0.2, 'd4': 0.1, 'd5': 0.05, 'd6': 0.05}
        )
    
    def create_test_data(self, rows=100, columns=3):
        """Helper to create test data"""
        np.random.seed(42)
        data = pd.DataFrame({
            f'feature_{i}': np.random.randn(rows) for i in range(columns)
        })
        data['timestamp'] = pd.date_range('2023-01-01', periods=rows, freq='1h')
        return data
    
    def test_execute_split_success(self):
        """Test successful split execution"""
        result = self.processor.execute_split(self.valid_config)
        
        self.assertIsInstance(result, SplitResult)
        self.assertEqual(len(result.datasets), 6)
        self.assertIn('d1', result.datasets)
        self.assertIn('d6', result.datasets)
        
        # Check sample counts
        total_samples = sum(len(ds) for ds in result.datasets.values())
        self.assertEqual(total_samples, len(self.test_data))
        
        # Check approximate ratios
        d1_ratio = len(result.datasets['d1']) / len(self.test_data)
        self.assertAlmostEqual(d1_ratio, 0.4, delta=0.01)
    
    def test_execute_split_no_data(self):
        """Test split execution with no data loaded"""
        processor = DataProcessor()
        
        with self.assertRaises(ValueError) as context:
            processor.execute_split(self.valid_config)
        
        self.assertIn("No data loaded", str(context.exception))
    
    def test_execute_split_invalid_configuration(self):
        """Test split execution with invalid configuration"""
        # Test that validation catches configuration errors before execution
        small_data = self.create_test_data(30, 3)  # Small dataset
        processor = DataProcessor()
        processor.set_data(small_data)
        
        # Configuration requiring minimum 10 samples per split, but data too small
        config = SplitConfiguration(
            ratios={'d1': 0.1, 'd2': 0.1, 'd3': 0.1, 'd4': 0.1, 'd5': 0.3, 'd6': 0.3}
        )
        
        with self.assertRaises(ValueError) as context:
            processor.execute_split(config)
        
        self.assertIn("Invalid split configuration", str(context.exception))
    
    def test_execute_split_default_configuration(self):
        """Test split execution with default configuration"""
        result = self.processor.execute_split()  # No config provided
        
        self.assertIsInstance(result, SplitResult)
        self.assertEqual(len(result.datasets), 6)
        
        # Check that default configuration was used
        d1_ratio = len(result.datasets['d1']) / len(self.test_data)
        self.assertAlmostEqual(d1_ratio, 0.5, delta=0.01)  # Default d1 ratio is 0.5
    
    def test_execute_split_temporal_ordering(self):
        """Test split execution with temporal ordering"""
        config = SplitConfiguration(
            ratios={'d1': 0.4, 'd2': 0.2, 'd3': 0.2, 'd4': 0.1, 'd5': 0.05, 'd6': 0.05},
            temporal_split=True,
            temporal_column='timestamp',
            shuffle=False
        )
        
        result = self.processor.execute_split(config)
        
        # Check temporal ordering within each dataset
        for dataset_key, dataset in result.datasets.items():
            if not dataset.empty and len(dataset) > 1:
                timestamps = dataset['timestamp']
                self.assertTrue(timestamps.is_monotonic_increasing, 
                              f"Dataset {dataset_key} is not temporally ordered")
        
        # Check temporal boundaries
        self.assertIsNotNone(result.temporal_boundaries)
        self.assertIn('d1', result.temporal_boundaries)
    
    def test_execute_split_shuffling(self):
        """Test split execution with shuffling"""
        config = SplitConfiguration(
            ratios={'d1': 0.4, 'd2': 0.2, 'd3': 0.2, 'd4': 0.1, 'd5': 0.05, 'd6': 0.05},
            shuffle=True,
            random_seed=42
        )
        
        result1 = self.processor.execute_split(config)
        
        # Reset data and execute again with same seed
        self.processor.set_data(self.test_data)
        result2 = self.processor.execute_split(config)
        
        # Results should be identical with same random seed
        pd.testing.assert_frame_equal(result1.datasets['d1'], result2.datasets['d1'])
    
    def test_execute_split_sample_count_preservation(self):
        """Test that split execution preserves total sample count"""
        result = self.processor.execute_split(self.valid_config)
        
        original_count = len(self.test_data)
        split_count = sum(len(ds) for ds in result.datasets.values())
        
        self.assertEqual(original_count, split_count)
        
        # Check integrity verification
        integrity = result.split_metadata['integrity_verification']
        self.assertTrue(integrity['total_samples_preserved'])
        self.assertEqual(integrity['sample_count_difference'], 0)
    
    def test_execute_split_feature_preservation(self):
        """Test that split execution preserves all features"""
        result = self.processor.execute_split(self.valid_config)
        
        original_features = set(self.test_data.columns)
        
        for dataset_key, dataset in result.datasets.items():
            if not dataset.empty:
                dataset_features = set(dataset.columns)
                self.assertEqual(original_features, dataset_features, 
                               f"Features not preserved in dataset {dataset_key}")
    
    def test_execute_split_metadata_generation(self):
        """Test that split execution generates comprehensive metadata"""
        result = self.processor.execute_split(self.valid_config)
        
        metadata = result.split_metadata
        self.assertIn('split_timestamp', metadata)
        self.assertIn('execution_time_seconds', metadata)
        self.assertIn('configuration', metadata)
        self.assertIn('dataset_statistics', metadata)
        self.assertIn('integrity_verification', metadata)
        
        # Check dataset statistics
        stats = metadata['dataset_statistics']
        self.assertEqual(len(stats), 6)
        
        for dataset_key in ['d1', 'd2', 'd3', 'd4', 'd5', 'd6']:
            self.assertIn(dataset_key, stats)
            self.assertIn('sample_count', stats[dataset_key])
            self.assertIn('actual_ratio', stats[dataset_key])
            self.assertIn('expected_ratio', stats[dataset_key])
    
    def test_execute_split_processing_history_tracking(self):
        """Test that split execution records processing history"""
        initial_history_length = len(self.processor.processing_history)
        
        self.processor.execute_split(self.valid_config)
        
        self.assertEqual(len(self.processor.processing_history), initial_history_length + 1)
        
        latest_state = self.processor.processing_history[-1]
        self.assertEqual(latest_state.stage, "split_execution")
        self.assertEqual(latest_state.transformation_applied, "six_dataset_split")
        self.assertEqual(latest_state.input_shape, self.test_data.shape)


class TestDataProcessorUtilityMethods(unittest.TestCase):
    """Test DataProcessor utility and helper methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = DataProcessor(random_seed=42)
        self.test_data = self.create_test_data(1000, 5)
        self.processor.set_data(self.test_data)
        
        # Execute split to have datasets available
        config = SplitConfiguration(
            ratios={'d1': 0.4, 'd2': 0.2, 'd3': 0.2, 'd4': 0.1, 'd5': 0.05, 'd6': 0.05}
        )
        self.split_result = self.processor.execute_split(config)
    
    def create_test_data(self, rows=100, columns=3):
        """Helper to create test data"""
        np.random.seed(42)
        data = pd.DataFrame({
            f'feature_{i}': np.random.randn(rows) for i in range(columns)
        })
        data['timestamp'] = pd.date_range('2023-01-01', periods=rows, freq='1h')
        return data
    
    def test_get_split_datasets(self):
        """Test getting split datasets"""
        datasets = self.processor.get_split_datasets()
        
        self.assertIsNotNone(datasets)
        self.assertEqual(len(datasets), 6)
        self.assertEqual(set(datasets.keys()), {'d1', 'd2', 'd3', 'd4', 'd5', 'd6'})
    
    def test_get_split_datasets_no_split(self):
        """Test getting split datasets when no split has been performed"""
        processor = DataProcessor()
        
        datasets = processor.get_split_datasets()
        
        self.assertIsNone(datasets)
    
    def test_get_split_metadata(self):
        """Test getting split metadata"""
        metadata = self.processor.get_split_metadata()
        
        self.assertIsNotNone(metadata)
        self.assertIn('split_timestamp', metadata)
        self.assertIn('dataset_statistics', metadata)
    
    def test_get_split_metadata_no_split(self):
        """Test getting split metadata when no split has been performed"""
        processor = DataProcessor()
        
        metadata = processor.get_split_metadata()
        
        self.assertIsNone(metadata)
    
    def test_get_processing_history(self):
        """Test getting processing history"""
        history = self.processor.get_processing_history()
        
        self.assertIsInstance(history, list)
        self.assertGreater(len(history), 0)
        
        # Check that it's a copy (modifications don't affect original)
        original_length = len(self.processor.processing_history)
        history.append("test")
        self.assertEqual(len(self.processor.processing_history), original_length)
    
    def test_get_training_datasets_default(self):
        """Test getting training datasets with default keys"""
        training_datasets = self.processor.get_training_datasets()
        
        self.assertEqual(set(training_datasets.keys()), {'d1', 'd2'})
        self.assertIsInstance(training_datasets['d1'], pd.DataFrame)
        self.assertIsInstance(training_datasets['d2'], pd.DataFrame)
    
    def test_get_training_datasets_custom_keys(self):
        """Test getting training datasets with custom keys"""
        training_datasets = self.processor.get_training_datasets(['d1', 'd3', 'd5'])
        
        self.assertEqual(set(training_datasets.keys()), {'d1', 'd3', 'd5'})
    
    def test_get_training_datasets_invalid_key(self):
        """Test getting training datasets with invalid key"""
        with self.assertLogs(level='WARNING') as log:
            training_datasets = self.processor.get_training_datasets(['d1', 'invalid_key'])
        
        self.assertEqual(set(training_datasets.keys()), {'d1'})
        self.assertTrue(any("not found" in message for message in log.output))
    
    def test_get_training_datasets_no_split(self):
        """Test getting training datasets when no split has been performed"""
        processor = DataProcessor()
        
        with self.assertRaises(ValueError) as context:
            processor.get_training_datasets()
        
        self.assertIn("No split datasets available", str(context.exception))
    
    def test_verify_data_integrity_success(self):
        """Test data integrity verification when integrity is maintained"""
        integrity_report = self.processor.verify_data_integrity()
        
        self.assertEqual(integrity_report['integrity_status'], 'verified')
        self.assertEqual(len(integrity_report['issues_found']), 0)
        self.assertIn('sample_count_preservation', integrity_report['checks_performed'])
        self.assertIn('feature_consistency', integrity_report['checks_performed'])
        self.assertIn('data_type_preservation', integrity_report['checks_performed'])
    
    def test_verify_data_integrity_no_original_data(self):
        """Test data integrity verification when no original data is available"""
        processor = DataProcessor()
        
        integrity_report = processor.verify_data_integrity()
        
        self.assertEqual(integrity_report['integrity_status'], 'cannot_verify')
        self.assertIn("No original data available", integrity_report['issues_found'][0])
    
    def test_verify_data_integrity_no_split_data(self):
        """Test data integrity verification when no split data is available"""
        processor = DataProcessor()
        processor.set_data(self.test_data)
        
        integrity_report = processor.verify_data_integrity()
        
        self.assertEqual(integrity_report['integrity_status'], 'cannot_verify')
        self.assertIn("No split datasets available", integrity_report['issues_found'][0])
    
    def test_clear_processing_state(self):
        """Test clearing processing state"""
        # Verify state exists
        self.assertIsNotNone(self.processor.current_data)
        self.assertIsNotNone(self.processor.split_datasets)
        self.assertGreater(len(self.processor.processing_history), 0)
        
        # Clear state
        self.processor.clear_processing_state()
        
        # Verify state is cleared
        self.assertIsNone(self.processor.current_data)
        self.assertIsNone(self.processor.split_datasets)
        self.assertIsNone(self.processor.split_metadata)
        self.assertEqual(len(self.processor.processing_history), 0)


class TestDataProcessorExportFunctionality(unittest.TestCase):
    """Test DataProcessor export functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = DataProcessor(random_seed=42)
        self.test_data = self.create_test_data(500, 3)  # Larger dataset for export tests
        self.processor.set_data(self.test_data)
        
        # Execute split to have datasets available
        config = SplitConfiguration(
            ratios={'d1': 0.5, 'd2': 0.2, 'd3': 0.15, 'd4': 0.1, 'd5': 0.03, 'd6': 0.02}
        )
        self.processor.execute_split(config)
        
        # Create temporary directory for exports
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_data(self, rows=100, columns=3):
        """Helper to create test data"""
        np.random.seed(42)
        data = pd.DataFrame({
            f'feature_{i}': np.random.randn(rows) for i in range(columns)
        })
        data['timestamp'] = pd.date_range('2023-01-01', periods=rows, freq='1h')
        return data
    
    def test_export_split_datasets_csv(self):
        """Test exporting split datasets to CSV format"""
        result = self.processor.export_split_datasets(self.temp_dir, format='csv')
        
        self.assertTrue(result)
        
        # Check that files were created
        output_path = Path(self.temp_dir)
        expected_files = ['d1.csv', 'd2.csv', 'd3.csv', 'd4.csv', 'd5.csv', 'd6.csv', 'split_metadata.json']
        
        for filename in expected_files:
            file_path = output_path / filename
            self.assertTrue(file_path.exists(), f"File {filename} was not created")
    
    def test_export_split_datasets_parquet(self):
        """Test exporting split datasets to Parquet format"""
        try:
            import pyarrow
            parquet_available = True
        except ImportError:
            parquet_available = False
        
        result = self.processor.export_split_datasets(self.temp_dir, format='parquet')
        
        if parquet_available:
            self.assertTrue(result)
            
            # Check that files were created
            output_path = Path(self.temp_dir)
            parquet_files = ['d1.parquet', 'd2.parquet', 'd3.parquet', 'd4.parquet', 'd5.parquet', 'd6.parquet']
            
            for filename in parquet_files:
                file_path = output_path / filename
                self.assertTrue(file_path.exists(), f"File {filename} was not created")
        else:
            # When pyarrow is not available, export should fail gracefully
            self.assertFalse(result)
    
    def test_export_split_datasets_json(self):
        """Test exporting split datasets to JSON format"""
        result = self.processor.export_split_datasets(self.temp_dir, format='json')
        
        self.assertTrue(result)
        
        # Check that files were created
        output_path = Path(self.temp_dir)
        json_files = ['d1.json', 'd2.json', 'd3.json', 'd4.json', 'd5.json', 'd6.json']
        
        for filename in json_files:
            file_path = output_path / filename
            self.assertTrue(file_path.exists(), f"File {filename} was not created")
    
    def test_export_split_datasets_unsupported_format(self):
        """Test exporting split datasets with unsupported format"""
        result = self.processor.export_split_datasets(self.temp_dir, format='xlsx')
        
        self.assertFalse(result)
    
    def test_export_split_datasets_no_split_data(self):
        """Test exporting when no split datasets are available"""
        processor = DataProcessor()
        
        result = processor.export_split_datasets(self.temp_dir)
        
        self.assertFalse(result)
    
    def test_export_split_datasets_creates_directory(self):
        """Test that export creates output directory if it doesn't exist"""
        nonexistent_dir = Path(self.temp_dir) / "new_subdir"
        self.assertFalse(nonexistent_dir.exists())
        
        result = self.processor.export_split_datasets(nonexistent_dir)
        
        self.assertTrue(result)
        self.assertTrue(nonexistent_dir.exists())
    
    def test_export_metadata_included(self):
        """Test that metadata is included in export"""
        self.processor.export_split_datasets(self.temp_dir, format='csv')
        
        metadata_path = Path(self.temp_dir) / "split_metadata.json"
        self.assertTrue(metadata_path.exists())
        
        # Verify metadata content
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.assertIn('split_timestamp', metadata)
        self.assertIn('dataset_statistics', metadata)
        self.assertIn('integrity_verification', metadata)


class TestDataProcessorCalculationHelpers(unittest.TestCase):
    """Test DataProcessor internal calculation helper methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = DataProcessor(random_seed=42)
    
    def test_calculate_split_indices_exact_division(self):
        """Test split index calculation with exact division"""
        ratios = {'d1': 0.5, 'd2': 0.2, 'd3': 0.2, 'd4': 0.1, 'd5': 0.0, 'd6': 0.0}
        total_samples = 100
        
        indices = self.processor._calculate_split_indices(total_samples, ratios)
        
        self.assertEqual(indices['d1'], 50)
        self.assertEqual(indices['d2'], 20)
        self.assertEqual(indices['d3'], 20)
        self.assertEqual(indices['d4'], 10)
        self.assertEqual(indices['d5'], 0)
        self.assertEqual(indices['d6'], 0)
        
        # Check total preservation
        self.assertEqual(sum(indices.values()), total_samples)
    
    def test_calculate_split_indices_with_remainder(self):
        """Test split index calculation with remainder distribution"""
        ratios = {'d1': 0.34, 'd2': 0.33, 'd3': 0.33, 'd4': 0.0, 'd5': 0.0, 'd6': 0.0}
        total_samples = 100
        
        indices = self.processor._calculate_split_indices(total_samples, ratios)
        
        # Check total preservation
        self.assertEqual(sum(indices.values()), total_samples)
        
        # Check that largest ratios get the remainder
        self.assertGreaterEqual(indices['d1'], 34)  # Should get remainder due to largest ratio
        self.assertGreaterEqual(indices['d2'], 33)
        self.assertGreaterEqual(indices['d3'], 33)
    
    def test_calculate_split_indices_small_dataset(self):
        """Test split index calculation with small dataset"""
        ratios = {'d1': 0.4, 'd2': 0.2, 'd3': 0.2, 'd4': 0.1, 'd5': 0.05, 'd6': 0.05}
        total_samples = 7  # Very small dataset
        
        indices = self.processor._calculate_split_indices(total_samples, ratios)
        
        # Check total preservation
        self.assertEqual(sum(indices.values()), total_samples)
        
        # Check that all indices are non-negative
        for key, value in indices.items():
            self.assertGreaterEqual(value, 0)


if __name__ == '__main__':
    unittest.main()
