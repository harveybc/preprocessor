"""Unit tests for DataHandler

Tests verify DataHandler behaviors in isolation:
- Data loading from various file formats
- Data validation and quality checking  
- Data format conversion and standardization
- Metadata management and tracking
- Error handling and recovery

Based on UNIT-007 test scenarios from test_unit.md
"""

import unittest
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import json
from datetime import datetime
from unittest.mock import Mock, patch, mock_open

# Import the module under test
from app.core.data_handler import DataHandler, DataMetadata


class TestDataHandlerInitialization(unittest.TestCase):
    """Test DataHandler initialization and setup"""
    
    def test_default_initialization(self):
        """Test DataHandler initializes with default settings"""
        handler = DataHandler()
        
        # Verify default supported formats
        expected_formats = ['csv', 'parquet', 'json']
        self.assertEqual(handler.supported_formats, expected_formats)
        
        # Verify initial state
        self.assertIsNone(handler.loaded_data)
        self.assertIsNone(handler.metadata)
        self.assertEqual(handler.loading_errors, [])
        
        # Verify default validation rules are set
        self.assertIsNotNone(handler.validation_rules)
        self.assertIn('min_rows', handler.validation_rules)
        self.assertIn('min_columns', handler.validation_rules)
        self.assertIn('max_missing_percentage', handler.validation_rules)
    
    def test_custom_initialization(self):
        """Test DataHandler initializes with custom supported formats"""
        custom_formats = ['csv', 'parquet']
        handler = DataHandler(supported_formats=custom_formats)
        
        self.assertEqual(handler.supported_formats, custom_formats)
    
    def test_validation_rules_setup(self):
        """Test default validation rules are properly configured"""
        handler = DataHandler()
        
        expected_rules = {
            'min_rows', 'min_columns', 'max_missing_percentage',
            'required_numeric_features', 'allowed_data_types',
            'max_file_size_mb', 'check_duplicates', 'check_outliers'
        }
        
        self.assertTrue(expected_rules.issubset(handler.validation_rules.keys()))
        
        # Verify rule types and reasonable defaults
        self.assertIsInstance(handler.validation_rules['min_rows'], int)
        self.assertIsInstance(handler.validation_rules['max_missing_percentage'], float)
        self.assertIsInstance(handler.validation_rules['allowed_data_types'], list)


class TestDataHandlerValidationRules(unittest.TestCase):
    """Test validation rule management"""
    
    def setUp(self):
        self.handler = DataHandler()
    
    def test_set_validation_rules_valid(self):
        """Test setting valid validation rules"""
        new_rules = {
            'min_rows': 20,
            'max_missing_percentage': 30.0
        }
        
        self.handler.set_validation_rules(new_rules)
        
        self.assertEqual(self.handler.validation_rules['min_rows'], 20)
        self.assertEqual(self.handler.validation_rules['max_missing_percentage'], 30.0)
    
    def test_set_validation_rules_unknown_rule(self):
        """Test setting unknown validation rule logs warning"""
        with patch.object(self.handler.logger, 'warning') as mock_warning:
            self.handler.set_validation_rules({'unknown_rule': 'value'})
            mock_warning.assert_called_once()
    
    def test_set_validation_rules_preserves_other_rules(self):
        """Test setting validation rules preserves unmodified rules"""
        original_min_columns = self.handler.validation_rules['min_columns']
        
        self.handler.set_validation_rules({'min_rows': 50})
        
        self.assertEqual(self.handler.validation_rules['min_columns'], original_min_columns)
        self.assertEqual(self.handler.validation_rules['min_rows'], 50)


class TestDataHandlerFileFormatDetection(unittest.TestCase):
    """Test file format detection capabilities"""
    
    def setUp(self):
        self.handler = DataHandler()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_detect_csv_format(self):
        """Test CSV format detection by extension"""
        test_files = [
            'data.csv', 'data.tsv', 'data.txt'
        ]
        
        for filename in test_files:
            file_path = Path(self.temp_dir) / filename
            result = self.handler._detect_file_format(file_path)
            self.assertEqual(result, 'csv')
    
    def test_detect_parquet_format(self):
        """Test Parquet format detection by extension"""
        test_files = ['data.parquet', 'data.pq']
        
        for filename in test_files:
            file_path = Path(self.temp_dir) / filename
            result = self.handler._detect_file_format(file_path)
            self.assertEqual(result, 'parquet')
    
    def test_detect_json_format(self):
        """Test JSON format detection by extension"""
        test_files = ['data.json', 'data.jsonl']
        
        for filename in test_files:
            file_path = Path(self.temp_dir) / filename
            result = self.handler._detect_file_format(file_path)
            self.assertEqual(result, 'json')
    
    def test_detect_unsupported_format(self):
        """Test unsupported format returns None"""
        file_path = Path(self.temp_dir) / 'data.xlsx'
        result = self.handler._detect_file_format(file_path)
        self.assertIsNone(result)
    
    def test_detect_format_by_content_csv(self):
        """Test content-based format detection for CSV"""
        # Create a test file with CSV content
        test_file = Path(self.temp_dir) / 'data.dat'
        with open(test_file, 'w') as f:
            f.write('col1,col2,col3\n1,2,3\n4,5,6\n')
        
        result = self.handler._detect_format_by_content(test_file)
        self.assertEqual(result, 'csv')
    
    def test_detect_format_by_content_json(self):
        """Test content-based format detection for JSON"""
        test_file = Path(self.temp_dir) / 'data.dat'
        with open(test_file, 'w') as f:
            f.write('{"key": "value", "data": [1, 2, 3]}')
        
        result = self.handler._detect_format_by_content(test_file)
        self.assertEqual(result, 'json')


class TestDataHandlerCSVSeparatorDetection(unittest.TestCase):
    """Test CSV separator detection"""
    
    def setUp(self):
        self.handler = DataHandler()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_detect_comma_separator(self):
        """Test detection of comma separator"""
        test_file = Path(self.temp_dir) / 'test.csv'
        with open(test_file, 'w') as f:
            f.write('col1,col2,col3\n1,2,3\n4,5,6\n')
        
        result = self.handler._detect_csv_separator(test_file)
        self.assertEqual(result, ',')
    
    def test_detect_semicolon_separator(self):
        """Test detection of semicolon separator"""
        test_file = Path(self.temp_dir) / 'test.csv'
        with open(test_file, 'w') as f:
            f.write('col1;col2;col3\n1;2;3\n4;5;6\n')
        
        result = self.handler._detect_csv_separator(test_file)
        self.assertEqual(result, ';')
    
    def test_detect_tab_separator(self):
        """Test detection of tab separator"""
        test_file = Path(self.temp_dir) / 'test.csv'
        with open(test_file, 'w') as f:
            f.write('col1\tcol2\tcol3\n1\t2\t3\n4\t5\t6\n')
        
        result = self.handler._detect_csv_separator(test_file)
        self.assertEqual(result, '\t')
    
    def test_detect_default_separator_fallback(self):
        """Test fallback to comma when no separator detected"""
        test_file = Path(self.temp_dir) / 'test.csv'
        with open(test_file, 'w') as f:
            f.write('noseparatorhere\njusttext\n')
        
        result = self.handler._detect_csv_separator(test_file)
        self.assertEqual(result, ',')


class TestDataHandlerDataLoading(unittest.TestCase):
    """Test data loading functionality - UNIT-007-A"""
    
    def setUp(self):
        self.handler = DataHandler()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_load_csv_data_success(self):
        """Test successful CSV data loading"""
        # Create test CSV file
        test_file = Path(self.temp_dir) / 'test.csv'
        test_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10.1, 20.2, 30.3, 40.4, 50.5],
            'feature3': ['a', 'b', 'c', 'd', 'e']
        })
        test_data.to_csv(test_file, index=False)
        
        # Load data
        result = self.handler.load_data(test_file)
        
        # Verify loading success
        self.assertTrue(result)
        self.assertIsNotNone(self.handler.loaded_data)
        self.assertIsNotNone(self.handler.metadata)
        
        # Verify data content
        loaded_data = self.handler.get_data()
        pd.testing.assert_frame_equal(loaded_data, test_data)
        
        # Verify metadata
        metadata = self.handler.get_metadata()
        self.assertEqual(metadata.rows, 5)
        self.assertEqual(metadata.columns, 3)
        self.assertEqual(metadata.file_format, 'csv')
        self.assertEqual(set(metadata.features), {'feature1', 'feature2', 'feature3'})
    
    def test_load_json_data_success(self):
        """Test successful JSON data loading"""
        # Create test JSON file
        test_file = Path(self.temp_dir) / 'test.json'
        test_data = [
            {'id': 1, 'value': 10.5, 'label': 'A'},
            {'id': 2, 'value': 20.5, 'label': 'B'},
            {'id': 3, 'value': 30.5, 'label': 'C'}
        ]
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
        
        # Load data
        result = self.handler.load_data(test_file)
        
        # Verify loading success
        self.assertTrue(result)
        loaded_data = self.handler.get_data()
        
        # Verify data structure
        self.assertEqual(len(loaded_data), 3)
        self.assertEqual(set(loaded_data.columns), {'id', 'value', 'label'})
    
    def test_load_nonexistent_file(self):
        """Test loading nonexistent file fails gracefully"""
        nonexistent_file = Path(self.temp_dir) / 'nonexistent.csv'
        
        result = self.handler.load_data(nonexistent_file)
        
        self.assertFalse(result)
        self.assertIsNone(self.handler.loaded_data)
        self.assertTrue(len(self.handler.loading_errors) > 0)
        self.assertIn("not found", self.handler.loading_errors[0])
    
    def test_load_unsupported_format(self):
        """Test loading unsupported format fails gracefully"""
        # Create test file with unsupported extension
        test_file = Path(self.temp_dir) / 'test.xlsx'
        test_file.touch()
        
        result = self.handler.load_data(test_file)
        
        self.assertFalse(result)
        self.assertIsNone(self.handler.loaded_data)
        self.assertTrue(len(self.handler.loading_errors) > 0)
        self.assertIn("Unsupported file format", self.handler.loading_errors[0])
    
    def test_load_corrupted_csv_file(self):
        """Test loading corrupted CSV file fails gracefully"""
        # Create corrupted CSV file
        test_file = Path(self.temp_dir) / 'corrupted.csv'
        with open(test_file, 'w') as f:
            f.write('col1,col2\n1,2,3,4,5\n"unclosed quote\n')
        
        result = self.handler.load_data(test_file)
        
        self.assertFalse(result)
        self.assertTrue(len(self.handler.loading_errors) > 0)
    
    def test_load_data_with_custom_parameters(self):
        """Test loading data with custom parameters"""
        # Create CSV with semicolon separator
        test_file = Path(self.temp_dir) / 'test.csv'
        with open(test_file, 'w') as f:
            f.write('col1;col2;col3\n1;2;3\n4;5;6\n')
        
        # Load with custom separator
        result = self.handler.load_data(test_file, sep=';')
        
        self.assertTrue(result)
        loaded_data = self.handler.get_data()
        self.assertEqual(len(loaded_data.columns), 3)
        self.assertEqual(list(loaded_data.columns), ['col1', 'col2', 'col3'])


class TestDataHandlerDataValidation(unittest.TestCase):
    """Test data validation functionality - UNIT-007-B"""
    
    def setUp(self):
        self.handler = DataHandler()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def create_test_data(self, rows=100, columns=5, missing_pct=0, duplicates=0):
        """Helper to create test data with specific characteristics"""
        np.random.seed(42)
        data = pd.DataFrame({
            f'feature_{i}': np.random.randn(rows) for i in range(columns)
        })
        
        # Add missing values if requested
        if missing_pct > 0:
            # Create a flat mask for missing values to ensure exact percentage
            total_cells = rows * columns
            n_missing = int(total_cells * missing_pct / 100)
            
            # Create a flat boolean mask
            mask = np.zeros(total_cells, dtype=bool)
            mask[:n_missing] = True
            np.random.shuffle(mask)
            
            # Reshape and apply mask
            mask = mask.reshape(rows, columns)
            for i in range(rows):
                for j in range(columns):
                    if mask[i, j]:
                        data.iloc[i, j] = np.nan
        
        # Add duplicate rows if requested
        if duplicates > 0:
            duplicate_rows = data.head(duplicates).copy()
            data = pd.concat([data, duplicate_rows], ignore_index=True)
        
        return data
    
    def test_validate_data_passes(self):
        """Test data validation passes for valid data"""
        # Create valid test data
        test_data = self.create_test_data(rows=50, columns=3)
        test_file = Path(self.temp_dir) / 'valid.csv'
        test_data.to_csv(test_file, index=False)
        
        result = self.handler.load_data(test_file)
        
        self.assertTrue(result)
        metadata = self.handler.get_metadata()
        self.assertEqual(metadata.validation_status, 'valid')
        self.assertEqual(len(metadata.validation_errors), 0)
    
    def test_validate_insufficient_rows(self):
        """Test validation fails for insufficient rows"""
        # Create data with too few rows
        test_data = self.create_test_data(rows=5, columns=3)  # Default min_rows is 10
        test_file = Path(self.temp_dir) / 'few_rows.csv'
        test_data.to_csv(test_file, index=False)
        
        result = self.handler.load_data(test_file)
        
        self.assertTrue(result)  # Loading succeeds but validation fails
        metadata = self.handler.get_metadata()
        self.assertEqual(metadata.validation_status, 'invalid')
        self.assertTrue(any('Insufficient rows' in error for error in metadata.validation_errors))
    
    def test_validate_insufficient_columns(self):
        """Test validation fails for insufficient columns"""
        # Set validation rule requiring minimum columns
        self.handler.set_validation_rules({'min_columns': 3})
        
        # Create data with too few columns
        test_data = pd.DataFrame({'feature1': [1, 2, 3, 4, 5] * 5})  # Only 1 column
        test_file = Path(self.temp_dir) / 'few_cols.csv'
        test_data.to_csv(test_file, index=False)
        
        result = self.handler.load_data(test_file)
        
        self.assertTrue(result)
        metadata = self.handler.get_metadata()
        self.assertEqual(metadata.validation_status, 'invalid')
        self.assertTrue(any('Insufficient columns' in error for error in metadata.validation_errors))
    
    def test_validate_excessive_missing_data(self):
        """Test validation fails for excessive missing data"""
        # Create data with high percentage of missing values
        test_data = self.create_test_data(rows=50, columns=3, missing_pct=60)  # 60% missing
        test_file = Path(self.temp_dir) / 'missing.csv'
        test_data.to_csv(test_file, index=False)
        
        result = self.handler.load_data(test_file)
        
        self.assertTrue(result)
        metadata = self.handler.get_metadata()
        self.assertEqual(metadata.validation_status, 'invalid')
        self.assertTrue(any('Too much missing data' in error for error in metadata.validation_errors))
    
    def test_validate_excessive_duplicates(self):
        """Test validation identifies excessive duplicate rows"""
        # Create data with many duplicates
        test_data = self.create_test_data(rows=20, columns=3, duplicates=15)  # 75% duplicates
        test_file = Path(self.temp_dir) / 'duplicates.csv'
        test_data.to_csv(test_file, index=False)
        
        result = self.handler.load_data(test_file)
        
        self.assertTrue(result)
        metadata = self.handler.get_metadata()
        # Should have warning about excessive duplicates
        self.assertTrue(any('duplicate' in error.lower() for error in metadata.validation_errors))
    
    def test_validate_required_numeric_features(self):
        """Test validation for required numeric features"""
        # Set validation rule requiring numeric features
        self.handler.set_validation_rules({'required_numeric_features': 2})
        
        # Create data with only text features
        test_data = pd.DataFrame({
            'text1': ['a', 'b', 'c'] * 10,
            'text2': ['x', 'y', 'z'] * 10
        })
        test_file = Path(self.temp_dir) / 'no_numeric.csv'
        test_data.to_csv(test_file, index=False)
        
        result = self.handler.load_data(test_file)
        
        self.assertTrue(result)
        metadata = self.handler.get_metadata()
        self.assertEqual(metadata.validation_status, 'invalid')
        self.assertTrue(any('Insufficient numeric features' in error for error in metadata.validation_errors))


class TestDataHandlerMetadata(unittest.TestCase):
    """Test metadata management functionality - UNIT-007-D"""
    
    def setUp(self):
        self.handler = DataHandler()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_metadata_creation(self):
        """Test metadata is correctly created during data loading"""
        # Create test data
        test_data = pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'text_col': ['a', 'b', 'c', 'd', 'e']
        })
        test_file = Path(self.temp_dir) / 'test.csv'
        test_data.to_csv(test_file, index=False)
        
        self.handler.load_data(test_file)
        metadata = self.handler.get_metadata()
        
        # Verify basic metadata
        self.assertEqual(metadata.source_file, str(test_file))
        self.assertEqual(metadata.file_format, 'csv')
        self.assertEqual(metadata.rows, 5)
        self.assertEqual(metadata.columns, 3)
        self.assertEqual(set(metadata.features), {'numeric_col', 'float_col', 'text_col'})
        
        # Verify data types are recorded
        self.assertIn('numeric_col', metadata.data_types)
        self.assertIn('float_col', metadata.data_types)
        self.assertIn('text_col', metadata.data_types)
        
        # Verify timestamps and file info
        self.assertIsInstance(metadata.load_timestamp, datetime)
        self.assertGreater(metadata.file_size_bytes, 0)
        self.assertIsNotNone(metadata.checksum)
    
    def test_quality_metrics_calculation(self):
        """Test quality metrics are correctly calculated"""
        # Create data with known characteristics
        test_data = pd.DataFrame({
            'complete_col': [1, 2, 3, 4, 5],
            'missing_col': [1, np.nan, 3, np.nan, 5],
            'text_col': ['a', 'b', 'a', 'b', 'c']  # Some duplicates
        })
        # Add duplicate row
        test_data = pd.concat([test_data, test_data.iloc[[0]]], ignore_index=True)
        
        test_file = Path(self.temp_dir) / 'quality_test.csv'
        test_data.to_csv(test_file, index=False)
        
        self.handler.load_data(test_file)
        metadata = self.handler.get_metadata()
        
        quality_metrics = metadata.quality_metrics
        
        # Verify completeness metrics
        self.assertIn('completeness_percentage', quality_metrics)
        self.assertIn('missing_values_by_column', quality_metrics)
        
        # Verify duplicate detection
        self.assertIn('duplicate_rows', quality_metrics)
        self.assertIn('duplicate_percentage', quality_metrics)
        self.assertEqual(quality_metrics['duplicate_rows'], 1)
        
        # Verify data type distribution
        self.assertIn('data_type_distribution', quality_metrics)
        
        # Verify numeric statistics for numeric columns
        self.assertIn('numeric_statistics', quality_metrics)
        self.assertIn('complete_col', quality_metrics['numeric_statistics'])
        
        # Verify text statistics for text columns
        self.assertIn('text_statistics', quality_metrics)
        self.assertIn('text_col', quality_metrics['text_statistics'])
    
    def test_checksum_calculation(self):
        """Test data checksum calculation for integrity verification"""
        test_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        test_file = Path(self.temp_dir) / 'checksum_test.csv'
        test_data.to_csv(test_file, index=False)
        
        self.handler.load_data(test_file)
        original_checksum = self.handler.get_metadata().checksum
        
        # Verify checksum is consistent
        calculated_checksum = self.handler._calculate_data_checksum(self.handler.get_data())
        self.assertEqual(original_checksum, calculated_checksum)
        
        # Verify integrity validation
        self.assertTrue(self.handler.validate_data_integrity())
    
    def test_data_summary_generation(self):
        """Test comprehensive data summary generation"""
        test_data = pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5],
            'text': ['a', 'b', 'c', 'd', 'e']
        })
        test_file = Path(self.temp_dir) / 'summary_test.csv'
        test_data.to_csv(test_file, index=False)
        
        self.handler.load_data(test_file)
        summary = self.handler.get_data_summary()
        
        # Verify summary structure
        required_keys = {
            'source_file', 'file_format', 'shape', 'features',
            'data_types', 'validation_status', 'quality_metrics',
            'load_timestamp', 'file_size_mb', 'checksum'
        }
        self.assertTrue(required_keys.issubset(summary.keys()))
        
        # Verify feature breakdown
        self.assertEqual(summary['features']['total'], 2)
        self.assertEqual(summary['features']['numeric'], 1)
        self.assertEqual(summary['features']['categorical'], 1)


class TestDataHandlerFeatureAccess(unittest.TestCase):
    """Test feature access methods"""
    
    def setUp(self):
        self.handler = DataHandler()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_get_features(self):
        """Test getting all feature names"""
        test_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'feature3': ['a', 'b', 'c']
        })
        test_file = Path(self.temp_dir) / 'features_test.csv'
        test_data.to_csv(test_file, index=False)
        
        self.handler.load_data(test_file)
        features = self.handler.get_features()
        
        self.assertEqual(set(features), {'feature1', 'feature2', 'feature3'})
        self.assertEqual(len(features), 3)
    
    def test_get_numeric_features(self):
        """Test getting numeric feature names"""
        test_data = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'text_col': ['a', 'b', 'c'],
            'bool_col': [True, False, True]
        })
        test_file = Path(self.temp_dir) / 'numeric_test.csv'
        test_data.to_csv(test_file, index=False)
        
        self.handler.load_data(test_file)
        numeric_features = self.handler.get_numeric_features()
        
        # Should include int and float columns
        self.assertIn('int_col', numeric_features)
        self.assertIn('float_col', numeric_features)
        # Should not include text or bool columns
        self.assertNotIn('text_col', numeric_features)
    
    def test_get_categorical_features(self):
        """Test getting categorical feature names"""
        test_data = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'text_col': ['a', 'b', 'c'],
            'category_col': pd.Categorical(['x', 'y', 'z'])
        })
        test_file = Path(self.temp_dir) / 'categorical_test.csv'
        test_data.to_csv(test_file, index=False)
        
        self.handler.load_data(test_file)
        categorical_features = self.handler.get_categorical_features()
        
        # Should include text columns
        self.assertIn('text_col', categorical_features)
        # Should not include numeric columns
        self.assertNotIn('int_col', categorical_features)
        self.assertNotIn('float_col', categorical_features)
    
    def test_features_with_no_data(self):
        """Test feature access methods when no data is loaded"""
        # No data loaded
        self.assertEqual(self.handler.get_features(), [])
        self.assertEqual(self.handler.get_numeric_features(), [])
        self.assertEqual(self.handler.get_categorical_features(), [])


class TestDataHandlerDataExport(unittest.TestCase):
    """Test data export functionality"""
    
    def setUp(self):
        self.handler = DataHandler()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_export_csv_success(self):
        """Test successful CSV export"""
        # Load test data
        test_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        test_file = Path(self.temp_dir) / 'input.csv'
        test_data.to_csv(test_file, index=False)
        self.handler.load_data(test_file)
        
        # Export data
        output_file = Path(self.temp_dir) / 'output.csv'
        result = self.handler.export_data(output_file, format='csv')
        
        self.assertTrue(result)
        self.assertTrue(output_file.exists())
        
        # Verify exported data matches original
        exported_data = pd.read_csv(output_file)
        pd.testing.assert_frame_equal(exported_data, test_data)
    
    def test_export_json_success(self):
        """Test successful JSON export"""
        test_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        test_file = Path(self.temp_dir) / 'input.csv'
        test_data.to_csv(test_file, index=False)
        self.handler.load_data(test_file)
        
        # Export as JSON
        output_file = Path(self.temp_dir) / 'output.json'
        result = self.handler.export_data(output_file, format='json')
        
        self.assertTrue(result)
        self.assertTrue(output_file.exists())
        
        # Verify JSON content
        with open(output_file, 'r') as f:
            json_data = json.load(f)
        self.assertEqual(len(json_data), 3)  # 3 rows
    
    def test_export_with_no_data(self):
        """Test export fails when no data is loaded"""
        output_file = Path(self.temp_dir) / 'output.csv'
        result = self.handler.export_data(output_file, format='csv')
        
        self.assertFalse(result)
        self.assertFalse(output_file.exists())
    
    def test_export_unsupported_format(self):
        """Test export fails for unsupported format"""
        test_data = pd.DataFrame({'col1': [1, 2, 3]})
        test_file = Path(self.temp_dir) / 'input.csv'
        test_data.to_csv(test_file, index=False)
        self.handler.load_data(test_file)
        
        output_file = Path(self.temp_dir) / 'output.xlsx'
        result = self.handler.export_data(output_file, format='xlsx')
        
        self.assertFalse(result)
    
    def test_export_creates_directories(self):
        """Test export creates output directories if needed"""
        test_data = pd.DataFrame({'col1': [1, 2, 3]})
        test_file = Path(self.temp_dir) / 'input.csv'
        test_data.to_csv(test_file, index=False)
        self.handler.load_data(test_file)
        
        # Export to nested directory
        output_file = Path(self.temp_dir) / 'subdir' / 'nested' / 'output.csv'
        result = self.handler.export_data(output_file, format='csv')
        
        self.assertTrue(result)
        self.assertTrue(output_file.exists())
        self.assertTrue(output_file.parent.exists())


class TestDataHandlerUtilityMethods(unittest.TestCase):
    """Test utility and helper methods"""
    
    def setUp(self):
        self.handler = DataHandler()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_clear_data(self):
        """Test data clearing functionality"""
        # Load some data first
        test_data = pd.DataFrame({'col1': [1, 2, 3]})
        test_file = Path(self.temp_dir) / 'test.csv'
        test_data.to_csv(test_file, index=False)
        self.handler.load_data(test_file)
        
        # Verify data is loaded
        self.assertIsNotNone(self.handler.loaded_data)
        self.assertIsNotNone(self.handler.metadata)
        
        # Clear data
        self.handler.clear_data()
        
        # Verify data is cleared
        self.assertIsNone(self.handler.loaded_data)
        self.assertIsNone(self.handler.metadata)
        self.assertEqual(self.handler.loading_errors, [])
    
    def test_get_loading_errors(self):
        """Test loading error tracking"""
        # Attempt to load nonexistent file
        nonexistent_file = Path(self.temp_dir) / 'nonexistent.csv'
        self.handler.load_data(nonexistent_file)
        
        errors = self.handler.get_loading_errors()
        self.assertTrue(len(errors) > 0)
        self.assertIn("not found", errors[0])
        
        # Verify we get a copy, not the original list
        errors.clear()
        original_errors = self.handler.get_loading_errors()
        self.assertTrue(len(original_errors) > 0)
    
    def test_get_data_summary_no_data(self):
        """Test data summary when no data is loaded"""
        summary = self.handler.get_data_summary()
        self.assertEqual(summary, {"status": "no_data_loaded"})
    
    def test_validate_data_integrity_no_data(self):
        """Test data integrity validation when no data is loaded"""
        result = self.handler.validate_data_integrity()
        self.assertFalse(result)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
