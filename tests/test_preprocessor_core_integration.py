"""
Integration tests for PreprocessorCore - main orchestrator component.

These tests verify that components work correctly together following
our BDD integration-level test plan specifications.
"""

import pytest
import pandas as pd
import numpy as np
import json
import tempfile
import shutil
from pathlib import Path

from app.core.preprocessor_core import PreprocessorCore
from app.core.configuration_manager import ConfigurationManager
from app.core.data_handler import DataHandler
from app.core.data_processor import DataProcessor
from app.core.normalization_handler import NormalizationHandler


class TestPreprocessorCoreBasicIntegration:
    """Basic integration tests for PreprocessorCore orchestration."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test data (larger dataset to accommodate six-dataset split minimums)
        np.random.seed(42)  # For reproducible tests
        self.test_data = pd.DataFrame({
            'feature1': np.random.randn(200),  # Increased from 100 to 200
            'feature2': np.random.randn(200), 
            'feature3': np.random.randn(200),
            'target': np.random.randn(200)
        })
        
        self.test_csv_path = self.temp_path / "test_data.csv"
        self.test_data.to_csv(self.test_csv_path, index=False)
        
        # Create configuration
        self.config = {
            "input": {
                "csv_file": str(self.test_csv_path),
                "format": "csv"
            },
            "processing": {
                "split_ratios": {
                    "d1": 0.4,
                    "d2": 0.2, 
                    "d3": 0.2,
                    "d4": 0.1,
                    "d5": 0.05,
                    "d6": 0.05
                },
                "random_seed": 42
            },
            "normalization": {
                "method": "z_score"
            }
        }
        
        self.config_path = self.temp_path / "config.json"
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_preprocessor_core_initialization(self):
        """Test TC-INT-001-A: Component initialization and configuration distribution."""
        # Given: Valid configuration
        config_manager = ConfigurationManager()
        config_manager.load_from_file(str(self.config_path))
        
        # When: PreprocessorCore is initialized 
        core = PreprocessorCore(config_manager)
        
        # Then: All core components are properly initialized
        assert core.config_manager is not None
        assert core.data_handler is not None
        assert core.data_processor is not None
        assert core.normalization_handler is not None
        assert core.plugin_loader is not None
        
        # And: Configuration is accessible
        config = core.config_manager.merged_config
        assert config["input"]["csv_file"] == str(self.test_csv_path)
        assert config["processing"]["split_ratios"]["d1"] == 0.4

    def test_data_loading_to_processing_handoff(self):
        """Test TC-INT-005-A: Data loading to processing handoff."""
        # Given: PreprocessorCore with valid configuration
        config_manager = ConfigurationManager()
        config_manager.load_from_file(str(self.config_path))
        core = PreprocessorCore(config_manager)
        
        # When: Data is loaded via DataHandler
        success = core.data_handler.load_data(str(self.test_csv_path))
        assert success, "Data loading should succeed"
        
        loaded_data = core.data_handler.get_data()
        
        # Then: Data format matches processor expectations
        assert isinstance(loaded_data, pd.DataFrame)
        assert len(loaded_data) == len(self.test_data)
        assert list(loaded_data.columns) == list(self.test_data.columns)
        
        # And: Data integrity is maintained during transfer
        pd.testing.assert_frame_equal(loaded_data, self.test_data, check_dtype=False)
        
        # And: Data can be processed by data processor
        from app.core.data_processor import SplitConfiguration
        split_config = SplitConfiguration(
            ratios=self.config["processing"]["split_ratios"],
            random_seed=42
        )
        core.data_processor.set_data(loaded_data)
        result = core.data_processor.execute_split(split_config)
        split_datasets = result.datasets
        assert 'd1' in split_datasets
        assert 'd2' in split_datasets 
        assert 'd3' in split_datasets

    def test_normalization_parameter_computation_and_application(self):
        """Test TC-INT-007-A & 007-B: Normalization parameter computation and cross-dataset application."""
        # Given: PreprocessorCore with loaded data
        config_manager = ConfigurationManager()
        config_manager.load_from_file(str(self.config_path))
        core = PreprocessorCore(config_manager)
        
        success = core.data_handler.load_data(str(self.test_csv_path))
        assert success, "Data loading should succeed"
        loaded_data = core.data_handler.get_data()
        
        from app.core.data_processor import SplitConfiguration
        split_config = SplitConfiguration(
            ratios=self.config["processing"]["split_ratios"],
            random_seed=42
        )
        core.data_processor.set_data(loaded_data)
        result = core.data_processor.execute_split(split_config)
        split_data = result.datasets
        
        # When: Normalization parameters are computed from training data
        # Using d1 and d2 as training datasets per BDD requirements
        training_datasets = {
            'd1': split_data['d1'],
            'd2': split_data['d2']
        }
        norm_params = core.normalization_handler.compute_parameters(training_datasets)
        
        # Then: Per-feature means and standard deviations are computed
        assert 'feature1' in norm_params.means
        assert 'feature2' in norm_params.means
        assert 'feature3' in norm_params.means
        assert 'feature1' in norm_params.stds
        assert 'feature2' in norm_params.stds
        assert 'feature3' in norm_params.stds
        
        # And: Statistical accuracy is maintained
        # Extract training features for validation
        train_d1_features = split_data['d1'].drop(columns=['target'])
        train_d2_features = split_data['d2'].drop(columns=['target'])
        combined_train_features = pd.concat([train_d1_features, train_d2_features])
        expected_mean = combined_train_features['feature1'].mean()
        expected_std = combined_train_features['feature1'].std()
        assert abs(norm_params.means['feature1'] - expected_mean) < 1e-10
        assert abs(norm_params.stds['feature1'] - expected_std) < 1e-10
        
        # And: Same parameters are used for all datasets (cross-dataset consistency)
        normalized_datasets = core.normalization_handler.apply_normalization(split_data, norm_params)
        
        # Verify normalization was applied correctly
        assert len(normalized_datasets) == len(split_data)
        for dataset_name in split_data.keys():
            assert dataset_name in normalized_datasets
            assert normalized_datasets[dataset_name].shape == split_data[dataset_name].shape

    def test_six_dataset_generation_consistency(self):
        """Test TC-INT-009-A & 009-B: Six-dataset generation and consistency."""
        # Given: PreprocessorCore with configuration for three datasets (standard split)
        config_manager = ConfigurationManager()
        config_manager.load_from_file(str(self.config_path))
        core = PreprocessorCore(config_manager)
        
        success = core.data_handler.load_data(str(self.test_csv_path))
        assert success, "Data loading should succeed"
        loaded_data = core.data_handler.get_data()
        
        # When: Datasets are generated
        from app.core.data_processor import SplitConfiguration
        split_config = SplitConfiguration(
            ratios=self.config["processing"]["split_ratios"],
            random_seed=42
        )
        core.data_processor.set_data(loaded_data)
        result = core.data_processor.execute_split(split_config)
        split_datasets = result.datasets
        
        # Then: All six datasets are present
        expected_datasets = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6']
        for dataset_name in expected_datasets:
            assert dataset_name in split_datasets, f"Missing dataset: {dataset_name}"
        
        # And: Data is correctly partitioned with no overlap
        all_indices = set()
        for dataset_name, dataset in split_datasets.items():
            current_indices = set(dataset.index)
            assert len(current_indices.intersection(all_indices)) == 0, f"Overlap found in {dataset_name}"
            all_indices.update(current_indices)
        
        # And: All datasets maintain feature consistency
        expected_columns = list(self.test_data.columns)
        for dataset_name, dataset in split_datasets.items():
            assert list(dataset.columns) == expected_columns, f"Column mismatch in {dataset_name}"
        
        # And: Total data size is preserved
        total_size = sum(len(dataset) for dataset in split_datasets.values())
        assert total_size == len(loaded_data), "Total data size not preserved"

    def test_complete_pipeline_integration(self):
        """Test complete end-to-end pipeline integration."""
        # Given: PreprocessorCore with full configuration
        config_manager = ConfigurationManager()
        config_manager.load_from_file(str(self.config_path))
        core = PreprocessorCore(config_manager)
        
        # When: Complete pipeline is executed
        # Step 1: Data loading
        success = core.data_handler.load_data(str(self.test_csv_path))
        assert success, "Data loading should succeed"
        loaded_data = core.data_handler.get_data()
        
        # Step 2: Data splitting
        from app.core.data_processor import SplitConfiguration
        split_config = SplitConfiguration(
            ratios=self.config["processing"]["split_ratios"],
            random_seed=42
        )
        core.data_processor.set_data(loaded_data)
        result = core.data_processor.execute_split(split_config)
        split_data = result.datasets
        
        # Step 3: Normalization parameter computation
        # Using d1 and d2 as training datasets per BDD requirements
        training_datasets = {
            'd1': split_data['d1'],
            'd2': split_data['d2']
        }
        norm_params = core.normalization_handler.compute_parameters(training_datasets)
        
        # Step 4: Apply normalization to all datasets
        normalized_data = core.normalization_handler.apply_normalization(split_data, norm_params)
        
        # Then: Pipeline execution completes successfully
        assert len(normalized_data) == len(split_data)
        
        # And: Data integrity is maintained throughout pipeline
        for dataset_name in split_data.keys():
            original_shape = split_data[dataset_name].shape
            normalized_shape = normalized_data[dataset_name].shape
            assert original_shape == normalized_shape, f"Shape mismatch in {dataset_name}"
        
        # And: Normalization is applied correctly
        for dataset_name, dataset in normalized_data.items():
            if dataset_name in ['d1', 'd2']:  # Training datasets
                # Training data should be approximately normalized (mean≈0, std≈1)
                features = dataset.drop(columns=['target'])
                # Combine d1 and d2 features to check overall normalization
                if dataset_name == 'd1':
                    d1_features = features
                elif dataset_name == 'd2':
                    d2_features = features
        
        # Check normalization on combined training data
        combined_features = pd.concat([d1_features, d2_features])
        for col in combined_features.columns:
            col_mean = combined_features[col].mean()
            col_std = combined_features[col].std()
            assert abs(col_mean) < 1e-10, f"Training {col} mean not normalized: {col_mean}"
            assert abs(col_std - 1.0) < 1e-10, f"Training {col} std not normalized: {col_std}"

    def test_error_handling_integration(self):
        """Test TC-INT-011-A & 011-B: Cross-component error propagation."""
        # Given: PreprocessorCore with invalid configuration
        invalid_config = self.config.copy()
        invalid_config["input"]["csv_file"] = "/nonexistent/file.csv"
        
        invalid_config_path = self.temp_path / "invalid_config.json"
        with open(invalid_config_path, 'w') as f:
            json.dump(invalid_config, f)
        
        config_manager = ConfigurationManager()
        config_manager.load_from_file(str(invalid_config_path))
        core = PreprocessorCore(config_manager)
        
        # When: Error occurs during data loading
        # Then: Error is properly handled by returning False
        success = core.data_handler.load_data("/nonexistent/file.csv")
        assert not success, "Loading nonexistent file should fail"
        
        # Given: Invalid split ratios
        invalid_split_config = self.config.copy()
        invalid_split_config["processing"]["split_ratios"] = {
            "d1": 0.5,
            "d2": 0.3,
            "d3": 0.3,  # This makes sum > 1.0 
            "d4": 0.1,
            "d5": 0.05,
            "d6": 0.05
        }
        
        invalid_split_config_path = self.temp_path / "invalid_split_config.json"
        with open(invalid_split_config_path, 'w') as f:
            json.dump(invalid_split_config, f)
        
        config_manager2 = ConfigurationManager()
        config_manager2.load_from_file(str(invalid_split_config_path))
        core2 = PreprocessorCore(config_manager2)
        
        success = core2.data_handler.load_data(str(self.test_csv_path))
        assert success, "Data loading should succeed"
        loaded_data = core2.data_handler.get_data()
        
        # When: Processing error occurs due to invalid ratios
        # Then: Error is caught and handled appropriately
        from app.core.data_processor import SplitConfiguration
        core2.data_processor.set_data(loaded_data)
        with pytest.raises(ValueError):
            invalid_split_config_obj = SplitConfiguration(
                ratios=invalid_split_config["processing"]["split_ratios"],
                random_seed=42
            )


class TestPreprocessorCorePluginIntegration:
    """Integration tests for plugin system integration."""
    
    def setup_method(self):
        """Set up test environment with mock plugins."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create plugin directories
        self.feature_plugin_dir = self.temp_path / "feature_engineering_plugins"
        self.feature_plugin_dir.mkdir()
        
        self.postproc_plugin_dir = self.temp_path / "postprocessing_plugins"
        self.postproc_plugin_dir.mkdir()
        
        # Create test data
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'feature1': np.random.randn(200),  # Increased from 100 to 200
            'feature2': np.random.randn(200),
            'target': np.random.randn(200)
        })
        
        self.test_csv_path = self.temp_path / "test_data.csv"
        self.test_data.to_csv(self.test_csv_path, index=False)
        
        # Create configuration with plugin paths
        self.config = {
            "input": {
                "csv_file": str(self.test_csv_path),
                "format": "csv"
            },
            "processing": {
                "split_ratios": {
                    "d1": 0.4,
                    "d2": 0.2, 
                    "d3": 0.2,
                    "d4": 0.1,
                    "d5": 0.05,
                    "d6": 0.05
                },
                "random_seed": 42
            },
            "plugins": {
                "directories": [str(self.feature_plugin_dir), str(self.postproc_plugin_dir)],
                "feature_engineering": [],
                "postprocessing": []
            }
        }
        
        self.config_path = self.temp_path / "config.json"
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_plugin_system_initialization(self):
        """Test TC-INT-003-A: Plugin discovery and loading integration."""
        # Given: PreprocessorCore with plugin configuration
        config_manager = ConfigurationManager()
        config_manager.load_from_file(str(self.config_path))
        core = PreprocessorCore(config_manager)
        
        # When: Plugin system is initialized
        core.initialize()
        
        # Then: Plugin directories are accessible
        assert core.plugin_loader is not None
        
        # And: Plugin discovery works without errors
        try:
            # Plugin directories should be set up during initialization
            num_plugins = core.plugin_loader.discover_plugins()
            
            # Empty directories should return 0 plugins without errors
            assert isinstance(num_plugins, int)
            assert num_plugins >= 0
            
            # Verify plugin directories were configured
            expected_dirs = [str(self.feature_plugin_dir), str(self.postproc_plugin_dir)]
            for expected_dir in expected_dirs:
                assert expected_dir in core.plugin_loader.plugin_directories
            
        except Exception as e:
            pytest.fail(f"Plugin discovery failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
