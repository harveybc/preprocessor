"""Unit Tests for NormalizationHandler

This module implements comprehensive unit tests for the NormalizationHandler class,
following BDD methodology and testing all behavioral contracts specified in the design.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import patch

from app.core.normalization_handler import NormalizationHandler, NormalizationParameters


class TestNormalizationParameters(unittest.TestCase):
    """Test NormalizationParameters dataclass behaviors"""
    
    def test_valid_parameters_creation(self):
        """Test creation of valid normalization parameters"""
        means = {'feature_1': 10.0, 'feature_2': -5.0}
        stds = {'feature_1': 2.0, 'feature_2': 1.5}
        features = ['feature_1', 'feature_2']
        timestamp = datetime.now()
        
        params = NormalizationParameters(
            means=means,
            stds=stds,
            features=features,
            computation_timestamp=timestamp,
            source_datasets=['d1', 'd2'],
            sample_count=1000,
            feature_count=2
        )
        
        self.assertEqual(params.means, means)
        self.assertEqual(params.stds, stds)
        self.assertEqual(params.features, features)
        self.assertEqual(params.source_datasets, ['d1', 'd2'])
        self.assertIsNotNone(params.checksum)
    
    def test_mismatched_means_stds_keys(self):
        """Test parameters creation with mismatched means and stds keys"""
        means = {'feature_1': 10.0, 'feature_2': -5.0}
        stds = {'feature_1': 2.0, 'feature_3': 1.5}  # Different key
        features = ['feature_1', 'feature_2']
        
        with self.assertRaises(ValueError) as context:
            NormalizationParameters(
                means=means,
                stds=stds,
                features=features,
                computation_timestamp=datetime.now(),
                source_datasets=['d1'],
                sample_count=100,
                feature_count=2
            )
        
        self.assertIn("identical feature sets", str(context.exception))
    
    def test_mismatched_features_keys(self):
        """Test parameters creation with mismatched features and parameter keys"""
        means = {'feature_1': 10.0, 'feature_2': -5.0}
        stds = {'feature_1': 2.0, 'feature_2': 1.5}
        features = ['feature_1', 'feature_3']  # Different feature
        
        with self.assertRaises(ValueError) as context:
            NormalizationParameters(
                means=means,
                stds=stds,
                features=features,
                computation_timestamp=datetime.now(),
                source_datasets=['d1'],
                sample_count=100,
                feature_count=2
            )
        
        self.assertIn("Feature list must match parameter keys", str(context.exception))
    
    def test_negative_standard_deviation(self):
        """Test parameters creation with negative standard deviation"""
        means = {'feature_1': 10.0}
        stds = {'feature_1': -2.0}  # Negative std
        features = ['feature_1']
        
        with self.assertRaises(ValueError) as context:
            NormalizationParameters(
                means=means,
                stds=stds,
                features=features,
                computation_timestamp=datetime.now(),
                source_datasets=['d1'],
                sample_count=100,
                feature_count=1
            )
        
        self.assertIn("must be positive", str(context.exception))
    
    def test_zero_standard_deviation(self):
        """Test parameters creation with zero standard deviation"""
        means = {'feature_1': 10.0}
        stds = {'feature_1': 0.0}  # Zero std
        features = ['feature_1']
        
        with self.assertRaises(ValueError) as context:
            NormalizationParameters(
                means=means,
                stds=stds,
                features=features,
                computation_timestamp=datetime.now(),
                source_datasets=['d1'],
                sample_count=100,
                feature_count=1
            )
        
        self.assertIn("must be positive", str(context.exception))
    
    def test_checksum_computation_and_verification(self):
        """Test checksum computation and integrity verification"""
        params = NormalizationParameters(
            means={'feature_1': 10.0},
            stds={'feature_1': 2.0},
            features=['feature_1'],
            computation_timestamp=datetime.now(),
            source_datasets=['d1'],
            sample_count=100,
            feature_count=1
        )
        
        # Test that checksum is computed
        self.assertIsNotNone(params.checksum)
        
        # Test integrity verification
        self.assertTrue(params.verify_integrity())
        
        # Test that modification breaks integrity
        params.means['feature_1'] = 15.0  # Modify parameter
        self.assertFalse(params.verify_integrity())


class TestNormalizationHandlerInitialization(unittest.TestCase):
    """Test NormalizationHandler initialization behaviors"""
    
    def test_default_initialization(self):
        """Test default NormalizationHandler initialization"""
        handler = NormalizationHandler()
        
        self.assertIsNotNone(handler.logger)
        self.assertEqual(handler.tolerance, 1e-6)
        self.assertIsNone(handler.parameters)
        self.assertEqual(len(handler.normalization_history), 0)
        self.assertEqual(len(handler.feature_exclusions), 0)
    
    def test_initialization_with_tolerance(self):
        """Test NormalizationHandler initialization with custom tolerance"""
        tolerance = 1e-4
        handler = NormalizationHandler(tolerance=tolerance)
        
        self.assertEqual(handler.tolerance, tolerance)
    
    def test_set_feature_exclusions(self):
        """Test setting feature exclusions"""
        handler = NormalizationHandler()
        exclusions = ['categorical_feature', 'id_feature']
        
        handler.set_feature_exclusions(exclusions)
        
        self.assertEqual(handler.feature_exclusions, exclusions)


class TestNormalizationHandlerParameterComputation(unittest.TestCase):
    """Test NormalizationHandler parameter computation behaviors"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.handler = NormalizationHandler()
        self.training_data = self.create_training_datasets()
    
    def create_training_datasets(self):
        """Helper to create training datasets"""
        np.random.seed(42)
        
        # Create datasets with known statistics
        d1 = pd.DataFrame({
            'feature_1': np.random.normal(10.0, 2.0, 500),  # mean=10, std=2
            'feature_2': np.random.normal(-5.0, 1.5, 500),  # mean=-5, std=1.5
            'feature_3': np.random.normal(0.0, 3.0, 500),   # mean=0, std=3
            'categorical': ['A'] * 250 + ['B'] * 250        # Non-numeric
        })
        
        d2 = pd.DataFrame({
            'feature_1': np.random.normal(10.0, 2.0, 300),
            'feature_2': np.random.normal(-5.0, 1.5, 300),
            'feature_3': np.random.normal(0.0, 3.0, 300),
            'categorical': ['A'] * 150 + ['B'] * 150
        })
        
        return {'d1': d1, 'd2': d2}
    
    def test_compute_parameters_success(self):
        """Test successful parameter computation"""
        parameters = self.handler.compute_parameters(self.training_data)
        
        self.assertIsInstance(parameters, NormalizationParameters)
        self.assertEqual(len(parameters.means), 3)  # 3 numeric features
        self.assertEqual(len(parameters.stds), 3)
        self.assertEqual(set(parameters.features), {'feature_1', 'feature_2', 'feature_3'})
        self.assertEqual(parameters.source_datasets, ['d1', 'd2'])
        self.assertEqual(parameters.sample_count, 800)  # 500 + 300
        self.assertEqual(parameters.feature_count, 3)
        
        # Verify parameters are stored
        self.assertEqual(self.handler.parameters, parameters)
        
        # Verify history is recorded
        self.assertEqual(len(self.handler.normalization_history), 1)
    
    def test_compute_parameters_with_exclusions(self):
        """Test parameter computation with feature exclusions"""
        self.handler.set_feature_exclusions(['feature_3'])
        
        parameters = self.handler.compute_parameters(self.training_data)
        
        self.assertEqual(len(parameters.features), 2)
        self.assertEqual(set(parameters.features), {'feature_1', 'feature_2'})
        self.assertNotIn('feature_3', parameters.means)
    
    def test_compute_parameters_with_config_exclusions(self):
        """Test parameter computation with config-based exclusions"""
        config = {'exclude_features': ['feature_2']}
        
        parameters = self.handler.compute_parameters(self.training_data, config)
        
        self.assertEqual(len(parameters.features), 2)
        self.assertEqual(set(parameters.features), {'feature_1', 'feature_3'})
        self.assertNotIn('feature_2', parameters.means)
    
    def test_compute_parameters_empty_datasets(self):
        """Test parameter computation with empty training datasets"""
        with self.assertRaises(ValueError) as context:
            self.handler.compute_parameters({})
        
        self.assertIn("cannot be empty", str(context.exception))
    
    def test_compute_parameters_no_numeric_features(self):
        """Test parameter computation when no numeric features available"""
        categorical_data = {
            'd1': pd.DataFrame({'category': ['A', 'B', 'C']})
        }
        
        with self.assertRaises(ValueError) as context:
            self.handler.compute_parameters(categorical_data)
        
        self.assertIn("No numeric features available", str(context.exception))
    
    def test_compute_parameters_constant_feature(self):
        """Test parameter computation with constant (zero variance) feature"""
        constant_data = {
            'd1': pd.DataFrame({
                'feature_1': [5.0] * 100,  # Constant feature
                'feature_2': np.random.randn(100)
            })
        }
        
        with self.assertLogs(level='WARNING') as log:
            parameters = self.handler.compute_parameters(constant_data)
        
        # Should handle constant feature gracefully
        self.assertEqual(parameters.means['feature_1'], 5.0)
        self.assertEqual(parameters.stds['feature_1'], 1.0)  # Set to 1.0 for zero variance
        self.assertTrue(any("zero variance" in message for message in log.output))
    
    def test_compute_parameters_with_nan_values(self):
        """Test parameter computation with NaN values"""
        data_with_nan = self.training_data.copy()
        # Fix pandas chained assignment warning
        data_with_nan['d1'].loc[data_with_nan['d1'].index[:10], 'feature_1'] = np.nan
        
        parameters = self.handler.compute_parameters(data_with_nan)
        
        # Should compute parameters excluding NaN values
        self.assertIsInstance(parameters.means['feature_1'], float)
        self.assertTrue(np.isfinite(parameters.means['feature_1']))
    
    def test_compute_parameters_feature_with_all_nan(self):
        """Test parameter computation when feature has all NaN values"""
        # Create data with one feature having all NaN - should continue with other features
        data_with_all_nan = {
            'd1': pd.DataFrame({
                'feature_1': [np.nan] * 100,
                'feature_2': np.random.randn(100)
            })
        }
        
        # Should succeed and exclude the all-NaN feature
        parameters = self.handler.compute_parameters(data_with_all_nan)
        self.assertIsNotNone(parameters)
        self.assertIn('feature_2', parameters.features)
        self.assertNotIn('feature_1', parameters.features)
        
        # Test case where ALL features have all NaN - should raise error
        data_all_nan = {
            'd1': pd.DataFrame({
                'feature_1': [np.nan] * 100,
                'feature_2': [np.nan] * 100
            })
        }
        
        with self.assertRaises(ValueError) as context:
            self.handler.compute_parameters(data_all_nan)
        
        self.assertIn("No numeric features available", str(context.exception))


class TestNormalizationHandlerParameterPersistence(unittest.TestCase):
    """Test NormalizationHandler parameter persistence behaviors"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.handler = NormalizationHandler()
        self.temp_dir = tempfile.mkdtemp()
        
        # Compute test parameters
        training_data = {
            'd1': pd.DataFrame({
                'feature_1': np.random.normal(10.0, 2.0, 100),
                'feature_2': np.random.normal(-5.0, 1.5, 100)
            })
        }
        self.handler.compute_parameters(training_data)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_persist_parameters_success(self):
        """Test successful parameter persistence"""
        storage_config = {
            'means_file': str(Path(self.temp_dir) / 'means.json'),
            'stds_file': str(Path(self.temp_dir) / 'stds.json'),
            'backup': False
        }
        
        result = self.handler.persist_parameters(storage_config)
        
        self.assertTrue(result)
        
        # Verify files were created
        means_path = Path(storage_config['means_file'])
        stds_path = Path(storage_config['stds_file'])
        
        self.assertTrue(means_path.exists())
        self.assertTrue(stds_path.exists())
        
        # Verify file contents
        with open(means_path, 'r') as f:
            means_data = json.load(f)
        
        with open(stds_path, 'r') as f:
            stds_data = json.load(f)
        
        self.assertIn('metadata', means_data)
        self.assertIn('means', means_data)
        self.assertIn('metadata', stds_data)
        self.assertIn('stds', stds_data)
        
        self.assertEqual(means_data['means'], self.handler.parameters.means)
        self.assertEqual(stds_data['stds'], self.handler.parameters.stds)
    
    def test_persist_parameters_with_backup(self):
        """Test parameter persistence with backup creation"""
        means_file = str(Path(self.temp_dir) / 'means.json')
        stds_file = str(Path(self.temp_dir) / 'stds.json')
        
        # Create existing files
        with open(means_file, 'w') as f:
            json.dump({'old': 'data'}, f)
        with open(stds_file, 'w') as f:
            json.dump({'old': 'data'}, f)
        
        storage_config = {
            'means_file': means_file,
            'stds_file': stds_file,
            'backup': True
        }
        
        result = self.handler.persist_parameters(storage_config)
        
        self.assertTrue(result)
        
        # Verify backup files were created
        backup_means = Path(self.temp_dir) / 'means.backup.json'
        backup_stds = Path(self.temp_dir) / 'stds.backup.json'
        
        self.assertTrue(backup_means.exists())
        self.assertTrue(backup_stds.exists())
    
    def test_persist_parameters_no_parameters(self):
        """Test parameter persistence when no parameters computed"""
        handler_no_params = NormalizationHandler()
        
        storage_config = {
            'means_file': str(Path(self.temp_dir) / 'means.json'),
            'stds_file': str(Path(self.temp_dir) / 'stds.json')
        }
        
        result = handler_no_params.persist_parameters(storage_config)
        
        self.assertFalse(result)
    
    def test_persist_parameters_directory_creation(self):
        """Test that persistence creates directories if needed"""
        subdir = Path(self.temp_dir) / 'subdir' / 'nested'
        storage_config = {
            'means_file': str(subdir / 'means.json'),
            'stds_file': str(subdir / 'stds.json'),
            'backup': False
        }
        
        result = self.handler.persist_parameters(storage_config)
        
        self.assertTrue(result)
        self.assertTrue(subdir.exists())
        self.assertTrue((subdir / 'means.json').exists())
        self.assertTrue((subdir / 'stds.json').exists())


class TestNormalizationHandlerParameterLoading(unittest.TestCase):
    """Test NormalizationHandler parameter loading behaviors"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.handler = NormalizationHandler()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create and persist test parameters
        training_data = {
            'd1': pd.DataFrame({
                'feature_1': np.random.normal(10.0, 2.0, 100),
                'feature_2': np.random.normal(-5.0, 1.5, 100)
            })
        }
        self.original_parameters = self.handler.compute_parameters(training_data)
        
        self.storage_config = {
            'means_file': str(Path(self.temp_dir) / 'means.json'),
            'stds_file': str(Path(self.temp_dir) / 'stds.json'),
            'backup': False
        }
        
        self.handler.persist_parameters(self.storage_config)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_parameters_success(self):
        """Test successful parameter loading"""
        # Clear current parameters
        self.handler.clear_parameters()
        self.assertIsNone(self.handler.parameters)
        
        # Load parameters
        result = self.handler.load_parameters(self.storage_config)
        
        self.assertTrue(result)
        self.assertIsNotNone(self.handler.parameters)
        
        # Verify loaded parameters match original
        loaded_params = self.handler.parameters
        self.assertEqual(loaded_params.means, self.original_parameters.means)
        self.assertEqual(loaded_params.stds, self.original_parameters.stds)
        self.assertEqual(loaded_params.features, self.original_parameters.features)
        self.assertEqual(loaded_params.checksum, self.original_parameters.checksum)
    
    def test_load_parameters_nonexistent_files(self):
        """Test parameter loading with non-existent files"""
        # Clear any existing parameters
        self.handler.parameters = None
        
        storage_config = {
            'means_file': str(Path(self.temp_dir) / 'nonexistent_means.json'),
            'stds_file': str(Path(self.temp_dir) / 'nonexistent_stds.json')
        }
        
        result = self.handler.load_parameters(storage_config)
        
        self.assertFalse(result)
        self.assertIsNone(self.handler.parameters)
    
    def test_load_parameters_corrupted_json(self):
        """Test parameter loading with corrupted JSON files"""
        # Clear any existing parameters
        self.handler.parameters = None
        
        # Create corrupted files
        with open(self.storage_config['means_file'], 'w') as f:
            f.write('invalid json content')
        
        result = self.handler.load_parameters(self.storage_config)
        
        self.assertFalse(result)
        self.assertIsNone(self.handler.parameters)
    
    def test_load_parameters_inconsistent_checksums(self):
        """Test parameter loading with inconsistent checksums between files"""
        # Load and modify one file to create checksum mismatch
        with open(self.storage_config['means_file'], 'r') as f:
            means_data = json.load(f)
        
        # Modify checksum
        means_data['metadata']['checksum'] = 'different_checksum'
        
        with open(self.storage_config['means_file'], 'w') as f:
            json.dump(means_data, f)
        
        result = self.handler.load_parameters(self.storage_config)
        
        self.assertFalse(result)
    
    def test_load_parameters_integrity_verification(self):
        """Test that loaded parameters pass integrity verification"""
        self.handler.clear_parameters()
        
        result = self.handler.load_parameters(self.storage_config)
        
        self.assertTrue(result)
        self.assertTrue(self.handler.parameters.verify_integrity())


class TestNormalizationHandlerNormalizationApplication(unittest.TestCase):
    """Test NormalizationHandler normalization application behaviors"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.handler = NormalizationHandler()
        
        # Create training data with known statistics
        np.random.seed(42)
        training_data = {
            'd1': pd.DataFrame({
                'feature_1': np.random.normal(10.0, 2.0, 500),
                'feature_2': np.random.normal(-5.0, 1.5, 500)
            }),
            'd2': pd.DataFrame({
                'feature_1': np.random.normal(10.0, 2.0, 300),
                'feature_2': np.random.normal(-5.0, 1.5, 300)
            })
        }
        
        # Compute parameters
        self.parameters = self.handler.compute_parameters(training_data)
        
        # Create test datasets for normalization
        self.test_datasets = {
            'd1': training_data['d1'].copy(),
            'd2': training_data['d2'].copy(),
            'd3': pd.DataFrame({
                'feature_1': np.random.normal(12.0, 3.0, 200),
                'feature_2': np.random.normal(-3.0, 2.0, 200)
            })
        }
    
    def test_apply_normalization_success(self):
        """Test successful normalization application"""
        normalized = self.handler.apply_normalization(self.test_datasets)
        
        self.assertEqual(len(normalized), 3)
        self.assertIn('d1', normalized)
        self.assertIn('d2', normalized)
        self.assertIn('d3', normalized)
        
        # Verify normalization was applied
        for dataset_name, dataset in normalized.items():
            self.assertIn('feature_1', dataset.columns)
            self.assertIn('feature_2', dataset.columns)
            
            # Check that normalization changed the values
            original_dataset = self.test_datasets[dataset_name]
            self.assertFalse(dataset['feature_1'].equals(original_dataset['feature_1']))
            self.assertFalse(dataset['feature_2'].equals(original_dataset['feature_2']))
    
    def test_apply_normalization_training_data_statistics(self):
        """Test that normalized training data has expected statistics"""
        training_datasets = {'d1': self.test_datasets['d1'], 'd2': self.test_datasets['d2']}
        normalized = self.handler.apply_normalization(training_datasets)
        
        # Combine normalized training data
        combined_normalized = pd.concat([normalized['d1'], normalized['d2']], ignore_index=True)
        
        # Check statistics for each feature
        for feature in self.parameters.features:
            feature_data = combined_normalized[feature].dropna()
            
            mean = feature_data.mean()
            std = feature_data.std(ddof=1)
            
            # Mean should be approximately 0
            self.assertAlmostEqual(mean, 0.0, places=2)
            
            # Standard deviation should be approximately 1
            self.assertAlmostEqual(std, 1.0, places=2)
    
    def test_apply_normalization_no_parameters(self):
        """Test normalization application without computed parameters"""
        handler_no_params = NormalizationHandler()
        
        with self.assertRaises(ValueError) as context:
            handler_no_params.apply_normalization(self.test_datasets)
        
        self.assertIn("No normalization parameters available", str(context.exception))
    
    def test_apply_normalization_empty_dataset(self):
        """Test normalization application with empty dataset"""
        datasets_with_empty = self.test_datasets.copy()
        datasets_with_empty['empty'] = pd.DataFrame()
        
        with self.assertLogs(level='WARNING') as log:
            normalized = self.handler.apply_normalization(datasets_with_empty)
        
        self.assertIn('empty', normalized)
        self.assertTrue(normalized['empty'].empty)
        self.assertTrue(any("empty" in message for message in log.output))
    
    def test_apply_normalization_missing_features(self):
        """Test normalization application with missing features"""
        dataset_missing_feature = {
            'incomplete': pd.DataFrame({
                'feature_1': np.random.randn(100)
                # Missing feature_2
            })
        }
        
        with self.assertLogs(level='WARNING') as log:
            normalized = self.handler.apply_normalization(dataset_missing_feature)
        
        # Should normalize available features
        self.assertIn('feature_1', normalized['incomplete'].columns)
        self.assertTrue(any("missing features" in message for message in log.output))
    
    def test_apply_normalization_extra_features(self):
        """Test normalization application with extra features"""
        dataset_extra_feature = {
            'extended': pd.DataFrame({
                'feature_1': np.random.randn(100),
                'feature_2': np.random.randn(100),
                'extra_feature': np.random.randn(100)
            })
        }
        
        with self.assertLogs(level='INFO') as log:
            normalized = self.handler.apply_normalization(dataset_extra_feature)
        
        # Should preserve extra feature unchanged
        self.assertIn('extra_feature', normalized['extended'].columns)
        self.assertTrue(any("extra numeric features" in message for message in log.output))
    
    def test_apply_normalization_non_numeric_features(self):
        """Test normalization application preserves non-numeric features"""
        dataset_with_categorical = {
            'mixed': pd.DataFrame({
                'feature_1': np.random.randn(100),
                'feature_2': np.random.randn(100),
                'category': ['A'] * 50 + ['B'] * 50
            })
        }
        
        normalized = self.handler.apply_normalization(dataset_with_categorical)
        
        # Non-numeric feature should be preserved unchanged
        self.assertIn('category', normalized['mixed'].columns)
        pd.testing.assert_series_equal(
            normalized['mixed']['category'],
            dataset_with_categorical['mixed']['category']
        )


class TestNormalizationHandlerDenormalization(unittest.TestCase):
    """Test NormalizationHandler denormalization behaviors"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.handler = NormalizationHandler()
        
        # Create test data with known values
        np.random.seed(42)
        self.original_data = pd.DataFrame({
            'feature_1': np.random.normal(10.0, 2.0, 100),
            'feature_2': np.random.normal(-5.0, 1.5, 100)
        })
        
        # Compute parameters and normalize
        training_data = {'d1': self.original_data}
        self.handler.compute_parameters(training_data)
        
        self.normalized_data = self.handler.apply_normalization({'test': self.original_data})['test']
    
    def test_denormalize_single_dataset(self):
        """Test denormalization of single dataset"""
        denormalized = self.handler.denormalize_data(self.normalized_data)
        
        # Should recover original values within tolerance
        for feature in self.handler.parameters.features:
            original_values = self.original_data[feature]
            denormalized_values = denormalized[feature]
            
            # Check that values are recovered within 0.001% accuracy
            relative_error = np.abs((denormalized_values - original_values) / original_values)
            max_error = relative_error.max()
            
            self.assertLess(max_error, 0.00001, f"Denormalization error too large for {feature}: {max_error}")
    
    def test_denormalize_multiple_datasets(self):
        """Test denormalization of multiple datasets"""
        normalized_datasets = {
            'test1': self.normalized_data,
            'test2': self.normalized_data.copy()
        }
        
        denormalized = self.handler.denormalize_data(normalized_datasets)
        
        self.assertIsInstance(denormalized, dict)
        self.assertEqual(len(denormalized), 2)
        self.assertIn('test1', denormalized)
        self.assertIn('test2', denormalized)
    
    def test_denormalize_no_parameters(self):
        """Test denormalization without parameters"""
        handler_no_params = NormalizationHandler()
        
        with self.assertRaises(ValueError) as context:
            handler_no_params.denormalize_data(self.normalized_data)
        
        self.assertIn("No normalization parameters available", str(context.exception))
    
    def test_denormalize_preserves_non_normalized_features(self):
        """Test that denormalization preserves non-normalized features"""
        # Add non-numeric feature
        data_with_category = self.normalized_data.copy()
        data_with_category['category'] = ['A'] * 50 + ['B'] * 50
        
        denormalized = self.handler.denormalize_data(data_with_category)
        
        # Non-normalized feature should be unchanged
        pd.testing.assert_series_equal(
            denormalized['category'],
            data_with_category['category']
        )


class TestNormalizationHandlerQualityValidation(unittest.TestCase):
    """Test NormalizationHandler quality validation behaviors"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.handler = NormalizationHandler()
        
        # Create training data
        np.random.seed(42)
        training_data = {
            'd1': pd.DataFrame({
                'feature_1': np.random.normal(10.0, 2.0, 500),
                'feature_2': np.random.normal(-5.0, 1.5, 500)
            }),
            'd2': pd.DataFrame({
                'feature_1': np.random.normal(10.0, 2.0, 300),
                'feature_2': np.random.normal(-5.0, 1.5, 300)
            })
        }
        
        # Compute parameters and normalize
        self.handler.compute_parameters(training_data)
        
        self.test_datasets = {
            'd1': training_data['d1'],
            'd2': training_data['d2'],
            'd3': pd.DataFrame({
                'feature_1': np.random.normal(12.0, 3.0, 200),
                'feature_2': np.random.normal(-3.0, 2.0, 200)
            })
        }
        
        self.normalized_datasets = self.handler.apply_normalization(self.test_datasets)
    
    def test_validate_normalization_quality_success(self):
        """Test successful normalization quality validation"""
        validation_results = self.handler.validate_normalization_quality(
            self.normalized_datasets, training_keys=['d1', 'd2']
        )
        
        self.assertIsInstance(validation_results, dict)
        self.assertIn('validation_passed', validation_results)
        self.assertIn('training_validation', validation_results)
        self.assertIn('consistency_validation', validation_results)
        
        # Training datasets should pass validation with realistic tolerances
        training_validation = validation_results['training_validation']
        self.assertIn('d1', training_validation)
        self.assertIn('d2', training_validation)
        
        # At least the overall validation should work with the updated tolerances
        for dataset_key in ['d1', 'd2']:
            dataset_validation = training_validation[dataset_key]
            self.assertTrue(dataset_validation['validation_passed'])
    
    def test_validate_normalization_quality_no_parameters(self):
        """Test quality validation without parameters"""
        handler_no_params = NormalizationHandler()
        
        with self.assertRaises(ValueError) as context:
            handler_no_params.validate_normalization_quality(self.normalized_datasets)
        
        self.assertIn("No parameters available", str(context.exception))
    
    def test_validate_training_statistics_quality(self):
        """Test validation of training statistics quality"""
        # Combine training data as normalization does
        combined_training = pd.concat([
            self.test_datasets['d1'],
            self.test_datasets['d2']
        ], ignore_index=True)
        
        # Normalize the combined training data
        combined_normalized = self.handler.apply_normalization({'combined': combined_training})['combined']
        
        # Check that combined training data has mean ≈ 0 and std ≈ 1
        for feature in self.handler.parameters.features:
            feature_data = combined_normalized[feature].dropna()
            
            mean_val = feature_data.mean()
            std_val = feature_data.std(ddof=1)
            
            # Check mean is close to 0 (stricter tolerance for combined data)
            self.assertLess(abs(mean_val), 0.01, 
                          f"Feature {feature} combined training mean {mean_val} not close to 0")
            
            # Check std is close to 1
            self.assertLess(abs(std_val - 1.0), 0.01,
                          f"Feature {feature} combined training std {std_val} not close to 1")


class TestNormalizationHandlerUtilityMethods(unittest.TestCase):
    """Test NormalizationHandler utility and helper methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.handler = NormalizationHandler()
    
    def test_get_parameters_no_parameters(self):
        """Test getting parameters when none computed"""
        parameters = self.handler.get_parameters()
        
        self.assertIsNone(parameters)
    
    def test_get_parameters_with_computed_parameters(self):
        """Test getting parameters after computation"""
        training_data = {
            'd1': pd.DataFrame({
                'feature_1': np.random.randn(100)
            })
        }
        
        computed_params = self.handler.compute_parameters(training_data)
        retrieved_params = self.handler.get_parameters()
        
        self.assertEqual(computed_params, retrieved_params)
    
    def test_get_normalization_history_empty(self):
        """Test getting normalization history when empty"""
        history = self.handler.get_normalization_history()
        
        self.assertIsInstance(history, list)
        self.assertEqual(len(history), 0)
    
    def test_get_normalization_history_with_computations(self):
        """Test getting normalization history after computations"""
        training_data = {
            'd1': pd.DataFrame({
                'feature_1': np.random.randn(100)
            })
        }
        
        self.handler.compute_parameters(training_data)
        history = self.handler.get_normalization_history()
        
        self.assertEqual(len(history), 1)
        self.assertIn('timestamp', history[0])
        self.assertIn('training_datasets', history[0])
        self.assertIn('feature_count', history[0])
        
        # Verify it's a copy (modifications don't affect original)
        original_length = len(self.handler.normalization_history)
        history.append({'test': 'entry'})
        self.assertEqual(len(self.handler.normalization_history), original_length)
    
    def test_clear_parameters(self):
        """Test clearing parameters and history"""
        # First compute parameters
        training_data = {
            'd1': pd.DataFrame({
                'feature_1': np.random.randn(100)
            })
        }
        
        self.handler.compute_parameters(training_data)
        self.assertIsNotNone(self.handler.parameters)
        self.assertGreater(len(self.handler.normalization_history), 0)
        
        # Clear parameters
        self.handler.clear_parameters()
        
        self.assertIsNone(self.handler.parameters)
        self.assertEqual(len(self.handler.normalization_history), 0)


if __name__ == '__main__':
    unittest.main()
