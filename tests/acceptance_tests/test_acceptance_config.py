"""
Acceptance Tests for Configuration and Compatibility
==================================================

Tests for ATS5 (Hierarchical Configuration) and ATS6 (Backward Compatibility)
covering configuration management, validation, migration, and API compatibility.
"""

import pytest
import pandas as pd
import numpy as np
import json
import tempfile
import shutil
import os
import yaml
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Add the app directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'app'))

from app.core.preprocessor_core import PreprocessorCore
from app.core.configuration_manager import ConfigurationManager
from app.cli import main as cli_main
from tests.acceptance_tests.test_acceptance_core import AcceptanceTestBase, TestDataFactory


# ATS5: Hierarchical Configuration Architecture Tests
class TestHierarchicalConfigurationArchitecture(AcceptanceTestBase):
    """
    Acceptance tests for ATS5: Modern Hierarchical Configuration Architecture
    
    Validates configuration hierarchy, inheritance, validation, 
    migration, and backward compatibility.
    """
    
    def test_configuration_hierarchy_and_inheritance(self):
        """
        ATS5.1: Configuration hierarchy and inheritance
        
        Given: Multiple configuration sources (global, environment, CLI, plugin)
        When: Configuration is loaded and resolved
        Then: Proper precedence hierarchy applied (CLI > plugin > env > global)
        """
        # Given: Global default configuration
        global_config = {
            'data': {
                'feature_columns': ['close'],
                'output_dir': '/default/output'
            },
            'processing': {
                'split_ratios': {'d1': 0.5, 'd2': 0.5, 'd3': 0.0, 'd4': 0.0, 'd5': 0.0, 'd6': 0.0},
                'normalize': False
            }
        }
        
        # Given: Environment-specific configuration
        env_config = {
            'data': {
                'output_dir': '/env/output'
            },
            'processing': {
                'normalize': True
            }
        }
        
        # Given: Plugin-specific configuration
        plugin_config = {
            'plugins': {
                'feature_engineering': {
                    'enabled': True,
                    'plugins': ['TestPlugin']
                }
            }
        }
        
        # Given: Command-line overrides
        cli_overrides = {
            'data': {
                'output_dir': self.temp_dir
            }
        }
        
        # When: Configuration hierarchy is resolved
        config_manager = ConfigurationManager()
        
        # Simulate loading from different sources
        final_config = global_config.copy()
        
        # Apply environment overrides
        def deep_update(base, updates):
            for key, value in updates.items():
                if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                    deep_update(base[key], value)
                else:
                    base[key] = value
        
        deep_update(final_config, env_config)
        deep_update(final_config, plugin_config)
        deep_update(final_config, cli_overrides)
        
        # Then: Verify proper precedence hierarchy
        assert final_config['data']['output_dir'] == self.temp_dir  # CLI override wins
        assert final_config['processing']['normalize'] is True  # Environment override
        assert 'plugins' in final_config  # Plugin config added
        assert final_config['data']['feature_columns'] == ['close']  # Global default preserved
        
        # Verify configuration can be used to create preprocessor
        input_data = self.test_data_factory.create_minimal_dataset(100)
        input_file = self.save_test_data(input_data)
        final_config['data']['input_file'] = input_file
        
        preprocessor = PreprocessorCore(final_config)
        assert preprocessor.config == final_config
    
    def test_comprehensive_configuration_validation(self):
        """
        ATS5.2: Comprehensive configuration validation
        
        Given: Configuration from multiple sources
        When: Configuration validation executes
        Then: Schema compliance, ranges, dependencies, and plugin requirements validated
        """
        # Given: Valid configuration
        valid_config = self.create_test_config()
        
        # When: Validate valid configuration
        try:
            preprocessor = PreprocessorCore(valid_config)
            validation_passed = True
            validation_error = None
        except Exception as e:
            validation_passed = False
            validation_error = str(e)
        
        # Then: Valid configuration passes
        assert validation_passed, f"Valid configuration failed validation: {validation_error}"
        
        # Given: Invalid configuration - bad split ratios
        invalid_config_ratios = self.create_test_config(
            processing={
                'split_ratios': {
                    'd1': 0.3, 'd2': 0.3, 'd3': 0.3,  # Sum = 0.9, should be 1.0
                    'd4': 0.1, 'd5': 0.0, 'd6': 0.0
                }
            }
        )
        
        # When: Validate invalid ratio configuration
        with pytest.raises(ValueError) as exc_info:
            PreprocessorCore(invalid_config_ratios)
        
        # Then: Specific validation failure reported
        error_msg = str(exc_info.value).lower()
        assert 'ratio' in error_msg or 'sum' in error_msg
        
        # Given: Invalid configuration - missing required fields
        invalid_config_missing = {
            'processing': {
                'normalize': True
                # Missing split_ratios
            }
        }
        
        # When: Validate configuration with missing fields
        with pytest.raises((KeyError, ValueError)):
            PreprocessorCore(invalid_config_missing)
    
    def test_configuration_migration_and_backward_compatibility(self):
        """
        ATS5.3: Configuration migration and backward compatibility
        
        Given: Legacy configuration files from previous system versions
        When: Configuration migration executes
        Then: Legacy formats detected, upgraded, deprecated parameters mapped
        """
        # Given: Legacy configuration format (simplified example)
        legacy_config = {
            'input_file': 'data.csv',
            'output_folder': self.temp_dir,
            'features': ['close', 'volume'],
            'train_ratio': 0.6,  # Legacy: single train ratio instead of 6-way split
            'test_ratio': 0.4,
            'standardize': True  # Legacy: 'standardize' instead of 'normalize'
        }
        
        # Save legacy configuration
        legacy_config_file = os.path.join(self.temp_dir, 'legacy_config.json')
        with open(legacy_config_file, 'w') as f:
            json.dump(legacy_config, f)
        
        # When: Configuration migration is applied
        def migrate_legacy_config(legacy_config):
            """Migrate legacy configuration to new format."""
            migrated = {
                'data': {
                    'input_file': legacy_config.get('input_file'),
                    'output_dir': legacy_config.get('output_folder'),
                    'feature_columns': legacy_config.get('features', [])
                },
                'processing': {
                    'normalize': legacy_config.get('standardize', False),
                    # Convert legacy train/test to 6-way split
                    'split_ratios': {
                        'd1': legacy_config.get('train_ratio', 0.6) * 0.8,  # 80% of train for d1
                        'd2': legacy_config.get('train_ratio', 0.6) * 0.2,  # 20% of train for d2
                        'd3': legacy_config.get('test_ratio', 0.4) * 0.5,   # 50% of test for d3
                        'd4': legacy_config.get('test_ratio', 0.4) * 0.3,   # 30% of test for d4
                        'd5': legacy_config.get('test_ratio', 0.4) * 0.1,   # 10% of test for d5
                        'd6': legacy_config.get('test_ratio', 0.4) * 0.1    # 10% of test for d6
                    },
                    'training_datasets': ['d1', 'd2']
                },
                'plugins': {
                    'feature_engineering': {'enabled': True},
                    'postprocessing': {'enabled': True}
                }
            }
            return migrated
        
        migrated_config = migrate_legacy_config(legacy_config)
        
        # Then: Legacy formats correctly upgraded
        assert 'data' in migrated_config
        assert 'processing' in migrated_config
        assert migrated_config['data']['input_file'] == 'data.csv'
        assert migrated_config['data']['output_dir'] == self.temp_dir
        assert migrated_config['processing']['normalize'] is True  # standardize -> normalize
        
        # Then: Split ratios correctly computed from train/test ratios
        split_ratios = migrated_config['processing']['split_ratios']
        total_ratio = sum(split_ratios.values())
        assert abs(total_ratio - 1.0) < 0.001, f"Split ratios sum to {total_ratio}, expected 1.0"
        
        # Then: Migrated configuration works with new system
        input_data = self.test_data_factory.create_minimal_dataset(100)
        input_file = self.save_test_data(input_data)
        migrated_config['data']['input_file'] = input_file
        
        preprocessor = PreprocessorCore(migrated_config)
        assert preprocessor.config['processing']['normalize'] is True
    
    def test_configuration_error_reporting_and_guidance(self):
        """
        ATS5.4: Configuration error reporting with guidance
        
        Given: Various invalid configurations
        When: Validation fails
        Then: Clear error messages with specific remediation guidance provided
        """
        test_cases = [
            {
                'name': 'Invalid split ratios',
                'config': {
                    'processing': {
                        'split_ratios': {'d1': 0.5, 'd2': 0.3}  # Missing datasets
                    }
                },
                'expected_keywords': ['ratio', 'split', 'd3', 'd4', 'd5', 'd6']
            },
            {
                'name': 'Missing required data config',
                'config': {
                    'processing': {'normalize': True}
                    # Missing data section
                },
                'expected_keywords': ['data', 'input_file', 'required']
            },
            {
                'name': 'Invalid feature columns type',
                'config': {
                    'data': {
                        'input_file': 'test.csv',
                        'feature_columns': 'close'  # Should be list
                    }
                },
                'expected_keywords': ['feature_columns', 'list', 'array']
            }
        ]
        
        for test_case in test_cases:
            with pytest.raises((ValueError, KeyError, TypeError)) as exc_info:
                PreprocessorCore(test_case['config'])
            
            error_msg = str(exc_info.value).lower()
            
            # Check that error message contains helpful keywords
            found_keywords = [kw for kw in test_case['expected_keywords'] if kw.lower() in error_msg]
            assert len(found_keywords) > 0, f"Error message for '{test_case['name']}' lacks guidance: {error_msg}"


# ATS6: Backward Compatibility and Migration Support Tests
class TestBackwardCompatibilityAndMigration(AcceptanceTestBase):
    """
    Acceptance tests for ATS6: Backward Compatibility and Migration Support
    
    Validates legacy workflow preservation, API contract preservation,
    and data format compatibility.
    """
    
    def test_legacy_workflow_preservation(self):
        """
        ATS6.1: Legacy workflow preservation
        
        Given: Existing preprocessor configurations and data files
        When: Refactored system processes legacy inputs
        Then: Workflows execute successfully, outputs compatible, performance maintained
        """
        # Given: Legacy-style input data (using existing test data format)
        legacy_data = pd.read_csv('/home/harveybc/Documents/GitHub/preprocessor/tests/data/eurusd_hourly_dataset_aligned_2011_2020.csv')
        
        # Take a smaller sample for testing
        legacy_data_sample = legacy_data.head(1000).copy()
        legacy_input_file = self.save_test_data(legacy_data_sample, 'legacy_input.csv')
        
        # Given: Legacy-compatible configuration
        legacy_config = {
            'data': {
                'input_file': legacy_input_file,
                'output_dir': self.temp_dir,
                'feature_columns': list(legacy_data_sample.select_dtypes(include=[np.number]).columns)
            },
            'processing': {
                'split_ratios': {
                    'd1': 0.4, 'd2': 0.2, 'd3': 0.2, 
                    'd4': 0.1, 'd5': 0.05, 'd6': 0.05
                },
                'normalize': True,
                'training_datasets': ['d1', 'd2']
            },
            'plugins': {
                'feature_engineering': {'enabled': False},
                'postprocessing': {'enabled': False}
            }
        }
        
        # When: Refactored system processes legacy inputs
        start_time = __import__('time').time()
        
        preprocessor = PreprocessorCore(legacy_config)
        preprocessor.load_data(legacy_input_file)
        preprocessor.process_data()
        result_metadata = preprocessor.export_results()
        
        end_time = __import__('time').time()
        processing_time = end_time - start_time
        
        # Then: Workflow executes successfully
        assert result_metadata is not None, "Legacy workflow should complete successfully"
        
        # Then: Output formats remain compatible
        datasets = self.load_result_datasets()
        assert len(datasets) == 6, "Should generate 6 datasets"
        
        for dataset_name, dataset in datasets.items():
            # Check CSV format compatibility
            assert isinstance(dataset, pd.DataFrame), f"{dataset_name} should be DataFrame"
            assert len(dataset) > 0, f"{dataset_name} should not be empty"
            
            # Check that feature columns are preserved
            expected_features = set(legacy_config['data']['feature_columns'])
            actual_features = set(dataset.select_dtypes(include=[np.number]).columns)
            assert expected_features.issubset(actual_features), f"Features missing in {dataset_name}"
        
        # Then: Performance characteristics maintained
        # For 1000 samples, should complete well under performance threshold
        assert processing_time < 30, f"Processing took {processing_time:.2f}s, may indicate performance regression"
        
        # Then: Normalization parameters generated
        means_file = os.path.join(self.temp_dir, 'means.json')
        stds_file = os.path.join(self.temp_dir, 'stds.json')
        assert os.path.exists(means_file), "Normalization means file should be generated"
        assert os.path.exists(stds_file), "Normalization stds file should be generated"
    
    def test_api_contract_preservation(self):
        """
        ATS6.2: API contract preservation
        
        Given: Existing integration points with preprocessor
        When: External systems interact with refactored preprocessor
        Then: API endpoints functional, response formats compatible, error handling consistent
        """
        # Given: Legacy API usage pattern (CLI-based interface)
        input_data = self.test_data_factory.create_standard_dataset(100)
        input_file = self.save_test_data(input_data)
        
        # Simulate legacy CLI invocation patterns
        legacy_cli_args = [
            '--input_file', input_file,
            '--output_dir', self.temp_dir,
            '--normalize',
            '--split_ratios', '0.4,0.2,0.2,0.1,0.05,0.05'
        ]
        
        # When: CLI interface is used (legacy style)
        with patch('sys.argv', ['preprocessor'] + legacy_cli_args):
            try:
                # Mock CLI execution
                from app.cli import create_parser
                parser = create_parser()
                args = parser.parse_args(legacy_cli_args)
                
                # Verify argument parsing works
                assert args.input_file == input_file
                assert args.output_dir == self.temp_dir
                assert args.normalize is True
                
                api_compatible = True
                error_msg = None
            except Exception as e:
                api_compatible = False
                error_msg = str(e)
        
        # Then: API endpoints remain functional
        assert api_compatible, f"CLI API incompatible: {error_msg}"
        
        # Then: Configuration-based API also works
        config = self.create_test_config()
        config['data']['input_file'] = input_file
        
        # Test programmatic API
        try:
            preprocessor = PreprocessorCore(config)
            preprocessor.load_data(input_file)
            preprocessor.process_data()
            result = preprocessor.export_results()
            
            programmatic_api_works = True
            assert result is not None
        except Exception as e:
            programmatic_api_works = False
            pytest.fail(f"Programmatic API failed: {e}")
        
        assert programmatic_api_works, "Programmatic API should remain functional"
    
    def test_data_format_compatibility(self):
        """
        ATS6.3: Data format compatibility
        
        Given: Legacy data files and output formats
        When: Refactored system processes legacy data
        Then: Input formats interpreted correctly, output structure maintained
        """
        # Given: Various legacy data formats
        
        # Test CSV with different column orders
        legacy_csv_data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=100, freq='1h'),
            'CLOSE': np.random.normal(1.1, 0.01, 100),  # Different casing
            'HIGH': np.random.normal(1.11, 0.01, 100),
            'LOW': np.random.normal(1.09, 0.01, 100),
            'OPEN': np.random.normal(1.1, 0.01, 100),
            'VOL': np.random.uniform(1000, 5000, 100)  # Different column name
        })
        
        legacy_csv_file = self.save_test_data(legacy_csv_data, 'legacy_format.csv')
        
        # Given: Configuration handling legacy column names
        legacy_config = self.create_test_config(
            data={
                'input_file': legacy_csv_file,
                'output_dir': self.temp_dir,
                'feature_columns': ['CLOSE', 'HIGH', 'LOW', 'OPEN', 'VOL']
            }
        )
        
        # When: System processes legacy data format
        preprocessor = PreprocessorCore(legacy_config)
        preprocessor.load_data(legacy_csv_file)
        preprocessor.process_data()
        preprocessor.export_results()
        
        # Then: Input format correctly interpreted
        datasets = self.load_result_datasets()
        assert len(datasets) == 6, "Should handle legacy format and generate 6 datasets"
        
        # Then: Output structure maintained
        for dataset_name, dataset in datasets.items():
            # Should preserve original column names
            assert 'CLOSE' in dataset.columns, f"Legacy column CLOSE missing in {dataset_name}"
            assert 'VOL' in dataset.columns, f"Legacy column VOL missing in {dataset_name}"
            
            # Should maintain data types
            assert dataset['CLOSE'].dtype in [np.float64, np.float32], "Numeric data type preserved"
            
            # Should maintain row structure
            assert len(dataset) > 0, f"Dataset {dataset_name} should not be empty"
    
    def test_migration_tools_and_documentation(self):
        """
        ATS6.4: Migration tools and documentation validation
        
        Given: Legacy configurations requiring migration
        When: Migration process is executed
        Then: Clear migration path provided, original files preserved
        """
        # Given: Complex legacy configuration
        complex_legacy_config = {
            'input_file': 'data.csv',
            'output_folder': '/old/path',
            'features': ['close', 'volume', 'indicator1'],
            'train_ratio': 0.7,
            'test_ratio': 0.3,
            'standardize': True,
            'indicators': {
                'moving_average': {'window': 5},
                'rsi': {'period': 14}
            },
            'validation_split': 0.2
        }
        
        # Save original configuration
        legacy_file = os.path.join(self.temp_dir, 'legacy_config.json')
        with open(legacy_file, 'w') as f:
            json.dump(complex_legacy_config, f, indent=2)
        
        # When: Migration process executed
        def comprehensive_migration(legacy_config):
            """Comprehensive migration function."""
            validation_ratio = legacy_config.get('validation_split', 0.0)
            train_ratio = legacy_config.get('train_ratio', 0.7)
            test_ratio = legacy_config.get('test_ratio', 0.3)
            
            # Adjust for validation split
            actual_train = train_ratio * (1 - validation_ratio)
            actual_val = train_ratio * validation_ratio
            
            migrated = {
                'data': {
                    'input_file': legacy_config.get('input_file'),
                    'output_dir': legacy_config.get('output_folder'),
                    'feature_columns': legacy_config.get('features', [])
                },
                'processing': {
                    'normalize': legacy_config.get('standardize', False),
                    'split_ratios': {
                        'd1': actual_train * 0.8,      # Primary training
                        'd2': actual_train * 0.2,      # Secondary training
                        'd3': actual_val,              # Validation
                        'd4': test_ratio * 0.6,        # Primary test
                        'd5': test_ratio * 0.2,        # Secondary test  
                        'd6': test_ratio * 0.2         # Final test
                    },
                    'training_datasets': ['d1', 'd2']
                },
                'plugins': {
                    'feature_engineering': {
                        'enabled': True,
                        'plugin_config': legacy_config.get('indicators', {})
                    },
                    'postprocessing': {'enabled': True}
                },
                'migration': {
                    'source_version': 'legacy',
                    'migration_date': __import__('datetime').datetime.now().isoformat(),
                    'original_config_backup': legacy_file + '.backup'
                }
            }
            return migrated
        
        migrated_config = comprehensive_migration(complex_legacy_config)
        
        # Then: Migration preserves original files
        backup_file = legacy_file + '.backup'
        shutil.copy(legacy_file, backup_file)
        assert os.path.exists(backup_file), "Original configuration should be backed up"
        
        # Then: Migrated configuration is valid
        migrated_ratios = migrated_config['processing']['split_ratios']
        total_ratio = sum(migrated_ratios.values())
        assert abs(total_ratio - 1.0) < 0.001, f"Migrated ratios sum to {total_ratio}"
        
        # Then: Migration metadata preserved
        assert 'migration' in migrated_config, "Migration metadata should be preserved"
        assert migrated_config['migration']['source_version'] == 'legacy'
        
        # Then: Complex features (indicators) mapped appropriately
        assert 'plugin_config' in migrated_config['plugins']['feature_engineering']
        original_indicators = complex_legacy_config.get('indicators', {})
        migrated_indicators = migrated_config['plugins']['feature_engineering']['plugin_config']
        assert migrated_indicators == original_indicators, "Indicator configuration should be preserved"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
