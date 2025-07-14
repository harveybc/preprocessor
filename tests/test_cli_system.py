"""System-Level Tests for CLI Integration

This module implements system-level tests for the CLI interface,
verifying end-to-end functionality following the BDD test specifications
from test_system.md.
"""

import unittest
import tempfile
import shutil
import subprocess
from pathlib import Path
import json
import pandas as pd

from app.cli import CLIInterface


class TestCLISystemIntegration(unittest.TestCase):
    """System tests for CLI integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_data_dir = Path("tests/data")
        self.test_data_file = self.test_data_dir / "eurusd_hourly_dataset_aligned_2011_2020.csv"
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Ensure test data exists
        if not self.test_data_file.exists():
            self.skipTest(f"Test data file not found: {self.test_data_file}")
    
    def tearDown(self):
        """Clean up test environment"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_cli_help_command(self):
        """
        SYS-001-A: System provides comprehensive help information
        Given: CLI interface is available
        When: Help command is executed
        Then: Complete usage information is displayed
        """
        result = subprocess.run(
            ["python", "-m", "app.main", "--help"],
            capture_output=True,
            text=True,
            cwd=Path.cwd()
        )
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("Preprocessor System", result.stdout)
        self.assertIn("positional arguments", result.stdout)
        self.assertIn("input_file", result.stdout)
        self.assertIn("Examples:", result.stdout)
    
    def test_cli_validation_only(self):
        """
        SYS-002-A: Configuration validation without processing
        Given: Valid input file and configuration
        When: Validation-only mode is executed
        Then: Configuration is validated successfully without processing
        """
        result = subprocess.run([
            "python", "-m", "app.main",
            str(self.test_data_file),
            "--validate-only"
        ], capture_output=True, text=True, cwd=Path.cwd())
        
        self.assertEqual(result.returncode, 0)
        output = result.stderr + result.stdout  # Check both streams
        self.assertIn("Configuration validation completed successfully", output)
        self.assertNotIn("Executing preprocessing pipeline", output)
    
    def test_cli_dry_run(self):
        """
        SYS-003-A: Dry run executes full pipeline without output
        Given: Valid input data
        When: Dry run mode is executed
        Then: Complete pipeline runs without creating output files
        """
        result = subprocess.run([
            "python", "-m", "app.main",
            str(self.test_data_file),
            "--dry-run",
            "--output-dir", str(self.temp_dir)
        ], capture_output=True, text=True, cwd=Path.cwd())
        
        self.assertEqual(result.returncode, 0)
        output = result.stderr + result.stdout
        self.assertIn("Processing completed successfully", output)
        self.assertIn("Dry run completed - no files written", output)
        
        # Verify no output files were created
        output_files = list(self.temp_dir.glob("*"))
        self.assertEqual(len(output_files), 0)
    
    def test_cli_full_processing_pipeline(self):
        """
        SYS-005-A: Complete six-dataset pipeline
        Given: Input dataset and default configuration
        When: System processes data through full pipeline
        Then: Six datasets are generated with consistent processing
        """
        output_dir = self.temp_dir / "output"
        
        result = subprocess.run([
            "python", "-m", "app.main",
            str(self.test_data_file),
            "--output-dir", str(output_dir)
        ], capture_output=True, text=True, cwd=Path.cwd())
        
        self.assertEqual(result.returncode, 0)
        output = result.stderr + result.stdout
        self.assertIn("Processing completed successfully", output)
        self.assertIn("Preprocessing completed successfully", output)
        
        # Verify all expected output files exist
        expected_files = [
            "d1.csv", "d2.csv", "d3.csv", "d4.csv", "d5.csv", "d6.csv",
            "means.json", "stds.json", "split_metadata.json"
        ]
        
        for filename in expected_files:
            file_path = output_dir / filename
            self.assertTrue(file_path.exists(), f"Missing output file: {filename}")
        
        # Verify dataset files contain data
        for i in range(1, 7):
            dataset_file = output_dir / f"d{i}.csv"
            df = pd.read_csv(dataset_file)
            self.assertGreater(len(df), 0, f"Dataset d{i} is empty")
            self.assertGreater(len(df.columns), 0, f"Dataset d{i} has no columns")
    
    def test_cli_configuration_file_loading(self):
        """
        SYS-001-B: Configuration file loading and merging
        Given: Configuration file with custom parameters
        When: CLI loads configuration file
        Then: Parameters are correctly applied to processing
        """
        # Create a test configuration file
        config_file = self.temp_dir / "test_config.json"
        test_config = {
            "data_processing": {
                "split_ratios": {
                    "d1": 0.5, "d2": 0.2, "d3": 0.2,
                    "d4": 0.05, "d5": 0.025, "d6": 0.025
                }
            },
            "normalization": {
                "method": "z-score"
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(test_config, f, indent=2)
        
        output_dir = self.temp_dir / "output"
        
        result = subprocess.run([
            "python", "-m", "app.main",
            str(self.test_data_file),
            "--config", str(config_file),
            "--output-dir", str(output_dir)
        ], capture_output=True, text=True, cwd=Path.cwd())
        
        self.assertEqual(result.returncode, 0)
        
        # Verify the configuration was applied by checking split metadata
        metadata_file = output_dir / "split_metadata.json"
        self.assertTrue(metadata_file.exists())
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # The custom split ratios should be reflected in the metadata
        self.assertIn("configuration", metadata)
        # Check that the custom ratio was applied (d1 should be 0.5, not default)
        config = metadata["configuration"]
        self.assertEqual(config["ratios"]["d1"], 0.5)
    
    def test_cli_error_handling_invalid_input(self):
        """
        SYS-007-A: Graceful error handling for invalid input
        Given: Invalid input file path
        When: CLI attempts to process non-existent file
        Then: Clear error message is provided and system exits gracefully
        """
        invalid_file = self.temp_dir / "nonexistent.csv"
        
        result = subprocess.run([
            "python", "-m", "app.main",
            str(invalid_file),
            "--output-dir", str(self.temp_dir)
        ], capture_output=True, text=True, cwd=Path.cwd())
        
        self.assertNotEqual(result.returncode, 0)
        output = result.stderr + result.stdout
        self.assertIn("Input file not found", output)
    
    def test_cli_verbose_output(self):
        """
        SYS-004-A: Verbose logging provides detailed processing information
        Given: Valid input data and verbose flag
        When: Processing is executed with verbose output
        Then: Detailed logging information is provided
        """
        result = subprocess.run([
            "python", "-m", "app.main",
            str(self.test_data_file),
            "--dry-run",
            "--verbose"
        ], capture_output=True, text=True, cwd=Path.cwd())
        
        self.assertEqual(result.returncode, 0)
        output = result.stderr + result.stdout
        self.assertIn("DEBUG", output)
        self.assertIn("Configuration validation successful", output)
        self.assertIn("Successfully split data into 6 datasets", output)
        self.assertIn("Applied normalization to 6 datasets", output)


class TestCLIInterface(unittest.TestCase):
    """Unit tests for CLI interface components"""
    
    def setUp(self):
        """Set up test environment"""
        self.cli = CLIInterface()
    
    def test_argument_parser_creation(self):
        """Test argument parser is created correctly"""
        parser = self.cli.create_argument_parser()
        self.assertIsNotNone(parser)
        
        # Test parsing help to ensure no exceptions
        with self.assertRaises(SystemExit):
            parser.parse_args(["--help"])
    
    def test_split_ratios_parsing(self):
        """Test split ratios parsing functionality"""
        # Valid ratios
        ratios = self.cli.parse_split_ratios("0.6,0.15,0.15,0.05,0.025,0.025")
        expected = {
            'd1': 0.6, 'd2': 0.15, 'd3': 0.15,
            'd4': 0.05, 'd5': 0.025, 'd6': 0.025
        }
        self.assertEqual(ratios, expected)
        
        # Invalid ratios - wrong count
        with self.assertRaises(ValueError):
            self.cli.parse_split_ratios("0.5,0.5")
        
        # Invalid ratios - sum not 1.0
        with self.assertRaises(ValueError):
            self.cli.parse_split_ratios("0.2,0.2,0.2,0.2,0.1,0.1")
    
    def test_plugin_list_parsing(self):
        """Test plugin list parsing functionality"""
        # Valid plugin list
        plugins = self.cli.parse_plugin_list("plugin1,plugin2,plugin3")
        self.assertEqual(plugins, ["plugin1", "plugin2", "plugin3"])
        
        # Empty plugin list
        plugins = self.cli.parse_plugin_list("")
        self.assertEqual(plugins, [])
        
        # None plugin list
        plugins = self.cli.parse_plugin_list(None)
        self.assertEqual(plugins, [])
        
        # Whitespace handling
        plugins = self.cli.parse_plugin_list(" plugin1 , plugin2 , plugin3 ")
        self.assertEqual(plugins, ["plugin1", "plugin2", "plugin3"])


if __name__ == '__main__':
    unittest.main()
