"""Unit Tests for ConfigurationManager

This module implements comprehensive unit tests for the ConfigurationManager class,
following BDD methodology and the specifications from test_unit.md.

Test Coverage:
- Default configuration initialization
- Configuration loading from files
- Configuration validation behavior
- Configuration access patterns
- Configuration merging
- Error handling and edge cases
"""

import unittest
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import patch, mock_open

from app.core.configuration_manager import ConfigurationManager, ConfigurationSource


class TestConfigurationManagerDefaultBehavior(unittest.TestCase):
    """Test ConfigurationManager default configuration behavior"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config_manager = ConfigurationManager()
    
    def test_default_configuration_initialization(self):
        """
        UNIT-001-A: Default configuration initialization
        Given: ConfigurationManager with no external configuration
        When: Default configuration is loaded
        Then: All default values are correctly set
        """
        # Verify default configuration is loaded
        self.assertIsNotNone(self.config_manager.merged_config)
        
        # Verify default split ratios
        expected_ratios = {"d1": 0.4, "d2": 0.2, "d3": 0.2, "d4": 0.1, "d5": 0.05, "d6": 0.05}
        actual_ratios = self.config_manager.get("processing.split_ratios")
        self.assertEqual(actual_ratios, expected_ratios)
        
        # Verify default training datasets
        expected_training = ["d1", "d2"]
        actual_training = self.config_manager.get("processing.training_datasets")
        self.assertEqual(actual_training, expected_training)
        
        # Verify default normalization method
        self.assertEqual(self.config_manager.get("processing.normalization.method"), "zscore")
        
        # Verify default output directory
        self.assertEqual(self.config_manager.get("data.output_directory"), "./output")
    
    def test_configuration_schema_validation(self):
        """
        UNIT-001-A: Configuration schema is valid
        Given: ConfigurationManager with default configuration
        When: Configuration schema is examined
        Then: Configuration schema is valid and complete
        """
        schema = self.config_manager.schema
        
        # Verify schema structure
        self.assertIn("type", schema)
        self.assertEqual(schema["type"], "object")
        self.assertIn("properties", schema)
        
        # Verify required sections in schema
        properties = schema["properties"]
        self.assertIn("data", properties)
        self.assertIn("processing", properties)
        self.assertIn("plugins", properties)
        self.assertIn("logging", properties)
    
    def test_default_configuration_passes_validation(self):
        """
        UNIT-001-A: Default configuration passes validation
        Given: ConfigurationManager with default configuration
        When: Validation is performed
        Then: Default configuration passes validation
        """
        validation_result = self.config_manager.validate()
        
        self.assertTrue(validation_result)
        self.assertEqual(len(self.config_manager.get_validation_errors()), 0)


class TestConfigurationManagerFileLoading(unittest.TestCase):
    """Test ConfigurationManager file loading behavior"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config_manager = ConfigurationManager()
    
    def test_valid_json_file_loading(self):
        """
        UNIT-001-B: Configuration loading from file
        Given: Valid configuration file
        When: Configuration is loaded from file
        Then: File contents are correctly parsed
        """
        # Create temporary JSON configuration file
        config_data = {
            "data": {
                "input_file": "test_data.csv",
                "output_directory": "./test_output"
            },
            "processing": {
                "split_ratios": {
                    "d1": 0.5, "d2": 0.2, "d3": 0.15, "d4": 0.1, "d5": 0.03, "d6": 0.02
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_file = f.name
        
        try:
            # Load configuration from file
            result = self.config_manager.load_from_file(temp_file)
            
            # Verify loading succeeded
            self.assertTrue(result)
            
            # Verify file configuration overrides defaults
            self.assertEqual(self.config_manager.get("data.input_file"), "test_data.csv")
            self.assertEqual(self.config_manager.get("data.output_directory"), "./test_output")
            self.assertEqual(self.config_manager.get("processing.split_ratios.d1"), 0.5)
            
            # Verify file source was added
            file_sources = [s for s in self.config_manager.sources if s.type == "file"]
            self.assertEqual(len(file_sources), 1)
            self.assertEqual(file_sources[0].priority, 2)
            
        finally:
            os.unlink(temp_file)
    
    def test_invalid_json_file_handling(self):
        """
        UNIT-001-B: File parsing errors are handled gracefully
        Given: Invalid JSON configuration file
        When: Configuration loading is attempted
        Then: Loading fails gracefully with error reporting
        """
        # Create temporary file with invalid JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"invalid": json content}')
            temp_file = f.name
        
        try:
            # Attempt to load invalid configuration
            result = self.config_manager.load_from_file(temp_file)
            
            # Verify loading failed
            self.assertFalse(result)
            
            # Verify no file source was added
            file_sources = [s for s in self.config_manager.sources if s.type == "file"]
            self.assertEqual(len(file_sources), 0)
            
        finally:
            os.unlink(temp_file)
    
    def test_nonexistent_file_handling(self):
        """
        UNIT-001-B: Missing file errors are handled gracefully
        Given: Non-existent configuration file path
        When: Configuration loading is attempted
        Then: Loading fails gracefully with appropriate error
        """
        # Attempt to load non-existent file
        result = self.config_manager.load_from_file("/nonexistent/path/config.json")
        
        # Verify loading failed
        self.assertFalse(result)
        
        # Verify no file source was added
        file_sources = [s for s in self.config_manager.sources if s.type == "file"]
        self.assertEqual(len(file_sources), 0)


class TestConfigurationManagerValidation(unittest.TestCase):
    """Test ConfigurationManager validation behavior"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config_manager = ConfigurationManager()
    
    def test_valid_configuration_validation(self):
        """
        UNIT-001-C: Configuration validation behavior
        Given: Configuration with valid parameter combinations
        When: Validation is performed
        Then: Valid configurations pass validation
        """
        # Load valid configuration
        valid_config = {
            "data": {"input_file": "test.csv"},
            "processing": {
                "split_ratios": {"d1": 0.4, "d2": 0.2, "d3": 0.2, "d4": 0.1, "d5": 0.05, "d6": 0.05},
                "training_datasets": ["d1", "d2"]
            }
        }
        
        # Manually set merged config for testing
        self.config_manager.merged_config = valid_config
        
        # Validate configuration
        result = self.config_manager.validate()
        
        # Verify validation passed
        self.assertTrue(result)
        self.assertEqual(len(self.config_manager.get_validation_errors()), 0)
    
    def test_invalid_split_ratios_validation(self):
        """
        UNIT-001-C: Invalid configurations are rejected with specific errors
        Given: Configuration with invalid split ratios
        When: Validation is performed
        Then: Configuration is rejected with specific error message
        """
        # Configuration with split ratios that don't sum to 1.0
        invalid_config = {
            "data": {"input_file": "test.csv"},
            "processing": {
                "split_ratios": {"d1": 0.5, "d2": 0.2, "d3": 0.2, "d4": 0.1, "d5": 0.05, "d6": 0.05},  # sums to 1.1
                "training_datasets": ["d1", "d2"]
            }
        }
        
        # Manually set merged config for testing
        self.config_manager.merged_config = invalid_config
        
        # Validate configuration
        result = self.config_manager.validate()
        
        # Verify validation failed
        self.assertFalse(result)
        
        # Verify specific error message
        errors = self.config_manager.get_validation_errors()
        self.assertTrue(len(errors) > 0)
        self.assertTrue(any("Split ratios must sum to 1.0" in error for error in errors))
    
    def test_invalid_training_datasets_validation(self):
        """
        UNIT-001-C: Invalid training dataset specifications are rejected
        Given: Configuration with invalid training datasets
        When: Validation is performed
        Then: Configuration is rejected with specific error message
        """
        # Configuration with invalid training dataset
        invalid_config = {
            "data": {"input_file": "test.csv"},
            "processing": {
                "split_ratios": {"d1": 0.4, "d2": 0.2, "d3": 0.2, "d4": 0.1, "d5": 0.05, "d6": 0.05},
                "training_datasets": ["d1", "d7"]  # d7 is invalid
            }
        }
        
        # Manually set merged config for testing
        self.config_manager.merged_config = invalid_config
        
        # Validate configuration
        result = self.config_manager.validate()
        
        # Verify validation failed
        self.assertFalse(result)
        
        # Verify specific error message
        errors = self.config_manager.get_validation_errors()
        self.assertTrue(len(errors) > 0)
        self.assertTrue(any("Invalid training dataset specified: d7" in error for error in errors))
    
    def test_missing_required_sections_validation(self):
        """
        UNIT-001-C: Missing required sections are detected
        Given: Configuration missing required sections
        When: Validation is performed
        Then: Validation fails with specific error about missing sections
        """
        # Configuration missing required section
        invalid_config = {
            "data": {"input_file": "test.csv"}
            # Missing "processing" section
        }
        
        # Manually set merged config for testing
        self.config_manager.merged_config = invalid_config
        
        # Validate configuration
        result = self.config_manager.validate()
        
        # Verify validation failed
        self.assertFalse(result)
        
        # Verify specific error message
        errors = self.config_manager.get_validation_errors()
        self.assertTrue(len(errors) > 0)
        self.assertTrue(any("Required configuration section missing: processing" in error for error in errors))


class TestConfigurationManagerAccess(unittest.TestCase):
    """Test ConfigurationManager access patterns"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config_manager = ConfigurationManager()
        
        # Set up test configuration
        test_config = {
            "data": {
                "input_file": "test.csv",
                "output_directory": "./test_output",
                "format": "csv"
            },
            "processing": {
                "split_ratios": {"d1": 0.4, "d2": 0.2, "d3": 0.2, "d4": 0.1, "d5": 0.05, "d6": 0.05},
                "training_datasets": ["d1", "d2"],
                "normalization": {
                    "method": "zscore",
                    "save_parameters": True
                }
            }
        }
        self.config_manager.merged_config = test_config
    
    def test_simple_key_access(self):
        """
        UNIT-001-D: Configuration access patterns
        Given: Loaded and validated configuration
        When: Configuration values are accessed
        Then: Values are returned in correct types
        """
        # Test simple key access
        self.assertEqual(self.config_manager.get("data.input_file"), "test.csv")
        self.assertEqual(self.config_manager.get("data.format"), "csv")
    
    def test_nested_key_access(self):
        """
        UNIT-001-D: Nested configuration is accessible
        Given: Configuration with nested structures
        When: Nested values are accessed
        Then: Nested values are correctly returned
        """
        # Test nested key access
        self.assertEqual(self.config_manager.get("processing.normalization.method"), "zscore")
        self.assertTrue(self.config_manager.get("processing.normalization.save_parameters"))
        
        # Test deeply nested access
        self.assertEqual(self.config_manager.get("processing.split_ratios.d1"), 0.4)
        self.assertEqual(self.config_manager.get("processing.split_ratios.d6"), 0.05)
    
    def test_nonexistent_key_access(self):
        """
        UNIT-001-D: Non-existent keys return appropriate defaults or errors
        Given: Configuration access for non-existent keys
        When: Non-existent keys are accessed
        Then: Default values are returned appropriately
        """
        # Test non-existent key returns None by default
        self.assertIsNone(self.config_manager.get("nonexistent.key"))
        
        # Test non-existent key returns custom default
        self.assertEqual(self.config_manager.get("nonexistent.key", "default_value"), "default_value")
    
    def test_section_access(self):
        """
        UNIT-001-D: Complete sections can be accessed
        Given: Configuration with defined sections
        When: Section access is requested
        Then: Complete section dictionary is returned
        """
        # Test section access
        data_section = self.config_manager.get_section("data")
        expected_data = {
            "input_file": "test.csv",
            "output_directory": "./test_output",
            "format": "csv"
        }
        self.assertEqual(data_section, expected_data)
        
        # Test non-existent section returns empty dict
        empty_section = self.config_manager.get_section("nonexistent")
        self.assertEqual(empty_section, {})
    
    def test_key_existence_checking(self):
        """
        UNIT-001-D: Key existence can be checked
        Given: Configuration with various keys
        When: Key existence is checked
        Then: Correct existence status is returned
        """
        # Test existing keys
        self.assertTrue(self.config_manager.has("data.input_file"))
        self.assertTrue(self.config_manager.has("processing.split_ratios.d1"))
        
        # Test non-existent keys
        self.assertFalse(self.config_manager.has("nonexistent.key"))
        self.assertFalse(self.config_manager.has("data.nonexistent"))


class TestConfigurationManagerMerging(unittest.TestCase):
    """Test ConfigurationManager merging behavior"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config_manager = ConfigurationManager()
    
    def test_configuration_source_priority(self):
        """
        UNIT-002-A: Simple configuration merging
        Given: Multiple configuration sources
        When: Configurations are merged
        Then: Higher precedence sources override lower precedence
        """
        # Add file configuration (priority 2)
        file_source = ConfigurationSource(
            name="test_file",
            type="file",
            priority=2,
            data={"data": {"input_file": "file_override.csv"}}
        )
        self.config_manager.sources.append(file_source)
        
        # Add CLI configuration (priority 4)
        cli_source = ConfigurationSource(
            name="cli",
            type="cli",
            priority=4,
            data={"data": {"input_file": "cli_override.csv"}}
        )
        self.config_manager.sources.append(cli_source)
        
        # Merge configurations
        self.config_manager._merge_configuration()
        
        # Verify CLI (higher priority) overrides file
        self.assertEqual(self.config_manager.get("data.input_file"), "cli_override.csv")
    
    def test_nested_configuration_merging(self):
        """
        UNIT-002-B: Complex nested merging
        Given: Configurations with nested structures
        When: Deep merging is performed
        Then: Nested structures are correctly combined
        """
        # Add source with partial nested override
        override_source = ConfigurationSource(
            name="override",
            type="file",
            priority=2,
            data={
                "processing": {
                    "split_ratios": {"d1": 0.5},  # Only override d1
                    "normalization": {"method": "minmax"}  # Override method only
                }
            }
        )
        self.config_manager.sources.append(override_source)
        
        # Merge configurations
        self.config_manager._merge_configuration()
        
        # Verify partial override preserves non-overridden values
        self.assertEqual(self.config_manager.get("processing.split_ratios.d1"), 0.5)  # Overridden
        self.assertEqual(self.config_manager.get("processing.split_ratios.d2"), 0.2)  # Preserved from default
        self.assertEqual(self.config_manager.get("processing.normalization.method"), "minmax")  # Overridden
        self.assertTrue(self.config_manager.get("processing.normalization.save_parameters"))  # Preserved from default
    
    def test_environment_variable_loading(self):
        """
        Test environment variable configuration loading
        Given: Environment variables with preprocessor prefix
        When: Environment configuration is loaded
        Then: Environment variables are correctly converted and loaded
        """
        # Mock environment variables
        env_vars = {
            "PREPROCESSOR_DATA_INPUT_FILE": "env_input.csv",
            "PREPROCESSOR_PROCESSING_NORMALIZATION_METHOD": "minmax",
            "PREPROCESSOR_PROCESSING_SPLIT_RATIOS_D1": "0.6",
            "PREPROCESSOR_LOGGING_LEVEL": "DEBUG"
        }
        
        with patch.dict(os.environ, env_vars):
            self.config_manager.load_from_environment()
        
        # Verify environment variables were loaded and converted
        self.assertEqual(self.config_manager.get("data.input_file"), "env_input.csv")
        self.assertEqual(self.config_manager.get("processing.normalization.method"), "minmax")
        self.assertEqual(self.config_manager.get("processing.split_ratios.d1"), 0.6)  # Converted to float
        self.assertEqual(self.config_manager.get("logging.level"), "DEBUG")
    
    def test_cli_argument_loading(self):
        """
        Test CLI argument configuration loading
        Given: CLI arguments dictionary
        When: CLI configuration is loaded
        Then: CLI arguments override other configuration sources
        """
        # Load CLI arguments
        cli_args = {
            "input-file": "cli_input.csv",
            "output-directory": "./cli_output",
            "processing__normalization__method": "none"  # Double underscore for nested
        }
        
        self.config_manager.load_from_cli(cli_args)
        
        # Verify CLI arguments were loaded with highest priority
        self.assertEqual(self.config_manager.get("input_file"), "cli_input.csv")
        self.assertEqual(self.config_manager.get("output_directory"), "./cli_output")
        self.assertEqual(self.config_manager.get("processing.normalization.method"), "none")


class TestConfigurationManagerErrorHandling(unittest.TestCase):
    """Test ConfigurationManager error handling and edge cases"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config_manager = ConfigurationManager()
    
    def test_configuration_summary_generation(self):
        """
        Test configuration summary generation
        Given: Configuration manager with loaded sources
        When: Configuration summary is requested
        Then: Complete summary with sources and validation status is returned
        """
        # Add a test source
        test_source = ConfigurationSource(
            name="test",
            type="file",
            priority=2,
            data={"test": "value"},
            location="/test/path"
        )
        self.config_manager.sources.append(test_source)
        
        # Generate summary
        summary = self.config_manager.get_configuration_summary()
        
        # Verify summary structure
        self.assertIn("sources", summary)
        self.assertIn("merged_config", summary)
        self.assertIn("validation_status", summary)
        self.assertIn("validation_errors", summary)
        
        # Verify source information
        self.assertEqual(len(summary["sources"]), 2)  # default + test
        test_source_info = next(s for s in summary["sources"] if s["name"] == "test")
        self.assertEqual(test_source_info["type"], "file")
        self.assertEqual(test_source_info["priority"], 2)
        self.assertEqual(test_source_info["location"], "/test/path")
    
    def test_type_conversion_edge_cases(self):
        """
        Test environment variable type conversion edge cases
        Given: Environment variables with various value types
        When: Type conversion is performed
        Then: Values are correctly converted to appropriate types
        """
        # Test boolean conversion
        self.assertTrue(self.config_manager._convert_env_value("true", "test.key"))
        self.assertFalse(self.config_manager._convert_env_value("false", "test.key"))
        self.assertTrue(self.config_manager._convert_env_value("True", "test.key"))
        self.assertFalse(self.config_manager._convert_env_value("False", "test.key"))
        
        # Test numeric conversion
        self.assertEqual(self.config_manager._convert_env_value("42", "test.key"), 42)
        self.assertEqual(self.config_manager._convert_env_value("3.14", "test.key"), 3.14)
        
        # Test JSON conversion
        json_array = self.config_manager._convert_env_value('["a", "b", "c"]', "test.key")
        self.assertEqual(json_array, ["a", "b", "c"])
        
        json_object = self.config_manager._convert_env_value('{"key": "value"}', "test.key")
        self.assertEqual(json_object, {"key": "value"})
        
        # Test string fallback
        self.assertEqual(self.config_manager._convert_env_value("plain_string", "test.key"), "plain_string")


if __name__ == "__main__":
    unittest.main()
