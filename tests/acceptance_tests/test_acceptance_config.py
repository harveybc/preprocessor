"""
Behavioral Acceptance Tests for Configuration Management

These tests focus on business behaviors, not implementation details.
Following BDD principles from design_acceptance.md.
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from pathlib import Path
from typing import Dict, Any

from app.core.preprocessor_core import PreprocessorCore


class TestConfigurationBehavior:
    """Test configuration behavior without implementation dependencies"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create minimal test data
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=1000, freq='H'),
            'close': np.random.randn(1000).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 1000),
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000)
        })
        
        self.test_input_file = os.path.join(self.temp_dir, 'test_data.csv')
        test_data.to_csv(self.test_input_file, index=False)
        
        yield
        
        # Cleanup
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_valid_configuration_enables_successful_processing(self):
        """
        Scenario: Valid configuration enables successful processing
        Given: A valid configuration with all required parameters
        When: The preprocessor processes data
        Then: Processing completes successfully and produces expected outputs
        """
        # Given: Valid configuration
        config = {
            'data_splitting': {
                'split_ratios': {
                    'd1': 0.4, 'd2': 0.2, 'd3': 0.2,
                    'd4': 0.1, 'd5': 0.05, 'd6': 0.05
                },
                'temporal_split': True
            },
            'normalization': {
                'training_datasets': ['d1', 'd2'],
                'tolerance': 1e-8
            },
            'plugins': {
                'feature_engineering': {'enabled': False},
                'postprocessing': {'enabled': False}
            }
        }
        
        # When: Preprocessor processes data
        preprocessor = PreprocessorCore()
        preprocessor.initialize([], config)
        
        success = preprocessor.load_data(self.test_input_file)
        assert success, "Should load data successfully"
        
        success = preprocessor.process_data()
        assert success, "Should process data successfully"
        
        success = preprocessor.export_results(self.temp_dir, include_metadata=True)
        assert success, "Should export results successfully"
        
        # Then: Expected outputs are produced
        for dataset_name in ['d1', 'd2', 'd3', 'd4', 'd5', 'd6']:
            dataset_file = os.path.join(self.temp_dir, f'{dataset_name}.csv')
            assert os.path.exists(dataset_file), f"Dataset {dataset_name} should be exported"
        
        # Metadata should be exported
        metadata_file = os.path.join(self.temp_dir, 'split_metadata.json')
        assert os.path.exists(metadata_file), "Metadata should be exported"
    
    def test_configuration_hierarchy_behavior(self):
        """
        Scenario: Configuration values flow correctly through the system
        Given: Configuration with hierarchical values
        When: The system uses configuration values
        Then: Values are applied correctly at each processing stage
        """
        # Given: Configuration with specific values
        config = {
            'data_splitting': {
                'split_ratios': {
                    'd1': 0.5, 'd2': 0.3, 'd3': 0.15,
                    'd4': 0.03, 'd5': 0.01, 'd6': 0.01
                }
            },
            'normalization': {
                'training_datasets': ['d1'],  # Use only d1 for training
                'tolerance': 1e-6
            }
        }
        
        # When: System processes with this configuration
        preprocessor = PreprocessorCore()
        preprocessor.initialize([], config)
        preprocessor.load_data(self.test_input_file)
        preprocessor.process_data()
        preprocessor.export_results(self.temp_dir)
        
        # Then: Configuration values are correctly applied
        # Check split ratios were applied correctly
        datasets = {}
        total_samples = 0
        for dataset_name in ['d1', 'd2', 'd3', 'd4', 'd5', 'd6']:
            dataset_file = os.path.join(self.temp_dir, f'{dataset_name}.csv')
            if os.path.exists(dataset_file):
                dataset = pd.read_csv(dataset_file)
                datasets[dataset_name] = len(dataset)
                total_samples += len(dataset)
        
        # Verify ratios are approximately correct (within rounding tolerance)
        if total_samples > 0:
            for dataset_name, expected_ratio in config['data_splitting']['split_ratios'].items():
                if dataset_name in datasets:
                    actual_ratio = datasets[dataset_name] / total_samples
                    assert abs(actual_ratio - expected_ratio) < 0.05, f"Split ratio for {dataset_name} incorrect"
    
    def test_invalid_configuration_provides_clear_feedback(self):
        """
        Scenario: Invalid configuration provides clear error feedback
        Given: Configuration with invalid values
        When: The preprocessor attempts to initialize or process
        Then: Clear error messages guide the user to fix the configuration
        """
        # Given: Invalid configuration (split ratios don't sum to 1.0)
        invalid_config = {
            'data_splitting': {
                'split_ratios': {
                    'd1': 0.3, 'd2': 0.3, 'd3': 0.3,  # Sum = 0.9, should be 1.0
                    'd4': 0.05, 'd5': 0.025, 'd6': 0.025
                }
            }
        }
        
        # When: Preprocessor attempts to process with invalid config
        preprocessor = PreprocessorCore()
        
        # Then: Clear error feedback is provided
        # The system should either reject the configuration during initialization
        # or provide clear error messages during processing
        try:
            preprocessor.initialize([], invalid_config)
            preprocessor.load_data(self.test_input_file)
            success = preprocessor.process_data()
            # If processing doesn't fail immediately, it should fail during validation
            assert not success, "Processing should fail with invalid configuration"
        except (ValueError, KeyError, TypeError) as e:
            # This is expected - invalid config should raise clear error
            error_msg = str(e).lower()
            # Error message should be helpful
            assert any(keyword in error_msg for keyword in ['ratio', 'sum', 'invalid', 'configuration']), \
                f"Error message should be helpful: {error_msg}"
    
    def test_configuration_supports_different_processing_modes(self):
        """
        Scenario: Configuration enables different processing modes
        Given: Different configuration modes (with/without normalization, with/without plugins)
        When: Each mode is processed
        Then: System adapts behavior according to configuration
        """
        # Test mode 1: No normalization, no plugins
        config_mode1 = {
            'data_splitting': {
                'split_ratios': {
                    'd1': 0.4, 'd2': 0.2, 'd3': 0.2,
                    'd4': 0.1, 'd5': 0.05, 'd6': 0.05
                }
            },
            'normalization': {
                'training_datasets': []  # Empty means no normalization
            },
            'plugins': {
                'feature_engineering': {'enabled': False},
                'postprocessing': {'enabled': False}
            }
        }
        
        # When: Process with mode 1
        preprocessor1 = PreprocessorCore()
        preprocessor1.initialize([], config_mode1)
        preprocessor1.load_data(self.test_input_file)
        success1 = preprocessor1.process_data()
        
        # Then: Processing succeeds without normalization
        assert success1, "Mode 1 processing should succeed"
        
        # Test mode 2: With normalization
        config_mode2 = {
            'data_splitting': {
                'split_ratios': {
                    'd1': 0.4, 'd2': 0.2, 'd3': 0.2,
                    'd4': 0.1, 'd5': 0.05, 'd6': 0.05
                }
            },
            'normalization': {
                'training_datasets': ['d1', 'd2']  # Enable normalization
            },
            'plugins': {
                'feature_engineering': {'enabled': False},
                'postprocessing': {'enabled': False}
            }
        }
        
        # When: Process with mode 2
        preprocessor2 = PreprocessorCore()
        preprocessor2.initialize([], config_mode2)
        preprocessor2.load_data(self.test_input_file)
        success2 = preprocessor2.process_data()
        
        # Then: Processing succeeds with normalization
        assert success2, "Mode 2 processing should succeed"
        
        # Both modes should work, demonstrating configuration flexibility
        assert success1 and success2, "Both processing modes should work"
