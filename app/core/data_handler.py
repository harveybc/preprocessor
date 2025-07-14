"""Data Handler Unit

This module implements the DataHandler class that provides data loading,
validation, and format handling capabilities.

Behavioral Specification:
- Loads data from various file formats (CSV, Parquet, JSON)
- Validates data structure and quality
- Provides standardized data format for internal processing
- Manages data metadata and provenance information
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class DataMetadata:
    """Data metadata container"""
    source_file: str
    file_format: str
    rows: int
    columns: int
    features: List[str]
    data_types: Dict[str, str]
    load_timestamp: datetime
    file_size_bytes: int
    checksum: Optional[str] = None
    validation_status: str = "unknown"
    validation_errors: List[str] = field(default_factory=list)
    quality_metrics: Dict[str, Any] = field(default_factory=dict)


class DataHandler:
    """
    Manages data loading, validation, and format handling.
    
    Behavioral Contract:
    - MUST load data from supported file formats correctly
    - MUST validate data structure and quality
    - MUST provide standardized internal data format
    - MUST maintain data metadata throughout processing
    - MUST handle data loading errors gracefully
    """
    
    def __init__(self, supported_formats: List[str] = None):
        """
        Initialize data handler.
        
        Args:
            supported_formats: List of supported file formats
            
        Behaviors:
        - Sets up supported file format configurations
        - Initializes data validation rules
        - Prepares metadata tracking structures
        """
        self.logger = logging.getLogger(__name__)
        self.supported_formats = supported_formats or ['csv', 'parquet', 'json']
        self.loaded_data: Optional[pd.DataFrame] = None
        self.metadata: Optional[DataMetadata] = None
        self.validation_rules: Dict[str, Any] = {}
        self.loading_errors: List[str] = []
        
        # Set up default validation rules
        self._setup_default_validation_rules()
    
    def _setup_default_validation_rules(self) -> None:
        """
        Sets up default data validation rules.
        
        Behavior:
        - Defines minimum data quality requirements
        - Sets up data type validation rules
        - Configures completeness thresholds
        """
        self.validation_rules = {
            'min_rows': 10,
            'min_columns': 1,
            'max_missing_percentage': 50.0,
            'required_numeric_features': 0,
            'allowed_data_types': ['int64', 'float64', 'object', 'datetime64[ns]', 'bool'],
            'max_file_size_mb': 1000,
            'check_duplicates': True,
            'check_outliers': True
        }
    
    def set_validation_rules(self, rules: Dict[str, Any]) -> None:
        """
        Sets custom validation rules.
        
        Args:
            rules: Dictionary of validation rules
            
        Behavior:
        - Updates validation rules with provided values
        - Merges with existing rules
        - Validates rule format and values
        """
        for key, value in rules.items():
            if key in self.validation_rules:
                self.validation_rules[key] = value
                self.logger.debug(f"Updated validation rule {key} = {value}")
            else:
                self.logger.warning(f"Unknown validation rule: {key}")
    
    def load_data(self, file_path: Union[str, Path], **kwargs) -> bool:
        """
        Loads data from a file.
        
        Args:
            file_path: Path to data file
            **kwargs: Additional loading parameters
            
        Returns:
            True if loading successful, False otherwise
            
        Behavior:
        - Detects file format automatically
        - Loads data using appropriate method
        - Creates metadata for loaded data
        - Validates data structure and quality
        """
        try:
            file_path = Path(file_path)
            
            # Validate file exists
            if not file_path.exists():
                error_msg = f"Data file not found: {file_path}"
                self.loading_errors.append(error_msg)
                self.logger.error(error_msg)
                return False
            
            # Detect file format
            file_format = self._detect_file_format(file_path)
            if not file_format:
                error_msg = f"Unsupported file format for: {file_path}"
                self.loading_errors.append(error_msg)
                self.logger.error(error_msg)
                return False
            
            # Load data based on format
            data = self._load_data_by_format(file_path, file_format, **kwargs)
            if data is None:
                return False
            
            # Create metadata
            metadata = self._create_metadata(file_path, file_format, data)
            
            # Validate data
            validation_result = self._validate_data(data, metadata)
            metadata.validation_status = "valid" if validation_result else "invalid"
            
            # Store results
            self.loaded_data = data
            self.metadata = metadata
            
            self.logger.info(f"Successfully loaded data from {file_path}: {data.shape[0]} rows, {data.shape[1]} columns")
            return True
            
        except Exception as e:
            error_msg = f"Error loading data from {file_path}: {e}"
            self.loading_errors.append(error_msg)
            self.logger.error(error_msg)
            return False
    
    def _detect_file_format(self, file_path: Path) -> Optional[str]:
        """
        Detects file format based on extension and content.
        
        Args:
            file_path: Path to file
            
        Returns:
            File format string or None if unsupported
            
        Behavior:
        - Uses file extension as primary indicator
        - Falls back to content analysis if extension unclear
        - Returns format compatible with loading methods
        """
        suffix = file_path.suffix.lower()
        
        # Map file extensions to formats
        extension_map = {
            '.csv': 'csv',
            '.tsv': 'csv',
            '.txt': 'csv',
            '.parquet': 'parquet',
            '.pq': 'parquet',
            '.json': 'json',
            '.jsonl': 'json'
        }
        
        detected_format = extension_map.get(suffix)
        
        if detected_format and detected_format in self.supported_formats:
            return detected_format
        
        # Try content-based detection for ambiguous cases
        if suffix in ['.txt', '.dat']:
            return self._detect_format_by_content(file_path)
        
        return None
    
    def _detect_format_by_content(self, file_path: Path) -> Optional[str]:
        """
        Detects file format by examining file content.
        
        Args:
            file_path: Path to file
            
        Returns:
            Detected format or None
            
        Behavior:
        - Reads first few lines of file
        - Analyzes content patterns
        - Returns most likely format
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_lines = [f.readline().strip() for _ in range(5)]
            
            first_line = first_lines[0] if first_lines else ""
            
            # Check for JSON patterns first (more specific)
            if first_line.startswith('{') or first_line.startswith('['):
                return 'json'
            
            # Check for CSV patterns
            if ',' in first_line or '\t' in first_line or ';' in first_line:
                return 'csv'
            
            return None
            
        except Exception:
            return None
    
    def _load_data_by_format(self, file_path: Path, file_format: str, **kwargs) -> Optional[pd.DataFrame]:
        """
        Loads data using format-specific method.
        
        Args:
            file_path: Path to data file
            file_format: Detected file format
            **kwargs: Format-specific loading parameters
            
        Returns:
            Loaded DataFrame or None if failed
            
        Behavior:
        - Uses appropriate pandas loading method
        - Applies format-specific parameters
        - Handles loading errors gracefully
        """
        try:
            if file_format == 'csv':
                # Handle CSV loading with various delimiters
                csv_kwargs = {
                    'encoding': kwargs.get('encoding', 'utf-8'),
                    'sep': kwargs.get('sep', None),  # Auto-detect separator
                    'header': kwargs.get('header', 0),
                    'index_col': kwargs.get('index_col', None),
                    'parse_dates': kwargs.get('parse_dates', True)
                }
                
                # If separator not specified, try to detect it
                if csv_kwargs['sep'] is None:
                    csv_kwargs['sep'] = self._detect_csv_separator(file_path)
                
                data = pd.read_csv(file_path, **csv_kwargs)
                
            elif file_format == 'parquet':
                parquet_kwargs = {
                    'engine': kwargs.get('engine', 'auto')
                }
                data = pd.read_parquet(file_path, **parquet_kwargs)
                
            elif file_format == 'json':
                json_kwargs = {
                    'orient': kwargs.get('orient', 'records'),
                    'lines': kwargs.get('lines', False),
                    'encoding': kwargs.get('encoding', 'utf-8')
                }
                data = pd.read_json(file_path, **json_kwargs)
                
            else:
                error_msg = f"Loading method not implemented for format: {file_format}"
                self.loading_errors.append(error_msg)
                self.logger.error(error_msg)
                return None
            
            return data
            
        except Exception as e:
            error_msg = f"Error loading {file_format} file {file_path}: {e}"
            self.loading_errors.append(error_msg)
            self.logger.error(error_msg)
            return None
    
    def _detect_csv_separator(self, file_path: Path) -> str:
        """
        Detects CSV separator by analyzing file content.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Most likely separator character
            
        Behavior:
        - Reads sample of file content
        - Counts occurrence of potential separators
        - Returns separator with highest consistent count
        """
        separators = [',', ';', '\t', '|']
        separator_counts = {sep: 0 for sep in separators}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read first few lines to detect separator
                sample_lines = [f.readline().strip() for _ in range(min(10, sum(1 for _ in f)))]
                f.seek(0)  # Reset file pointer
                sample_lines = [f.readline().strip() for _ in range(min(10, len(sample_lines)))]
            
            # Count separators in each line
            for line in sample_lines:
                for sep in separators:
                    separator_counts[sep] += line.count(sep)
            
            # Return separator with highest count
            best_separator = max(separator_counts, key=separator_counts.get)
            return best_separator if separator_counts[best_separator] > 0 else ','
            
        except Exception:
            return ','  # Default to comma
    
    def _create_metadata(self, file_path: Path, file_format: str, data: pd.DataFrame) -> DataMetadata:
        """
        Creates metadata for loaded data.
        
        Args:
            file_path: Path to source file
            file_format: File format
            data: Loaded DataFrame
            
        Returns:
            DataMetadata object with complete information
            
        Behavior:
        - Extracts data characteristics and statistics
        - Records file information and loading context
        - Calculates quality metrics
        """
        # Get file statistics
        file_stat = file_path.stat()
        
        # Extract data characteristics
        features = data.columns.tolist()
        data_types = {col: str(dtype) for col, dtype in data.dtypes.items()}
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(data)
        
        # Create metadata object
        metadata = DataMetadata(
            source_file=str(file_path),
            file_format=file_format,
            rows=len(data),
            columns=len(data.columns),
            features=features,
            data_types=data_types,
            load_timestamp=datetime.now(),
            file_size_bytes=file_stat.st_size,
            quality_metrics=quality_metrics
        )
        
        # Calculate checksum for data integrity
        metadata.checksum = self._calculate_data_checksum(data)
        
        return metadata
    
    def _calculate_quality_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates data quality metrics.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Dictionary of quality metrics
            
        Behavior:
        - Calculates completeness, uniqueness, and consistency metrics
        - Identifies potential data quality issues
        - Provides statistical summaries
        """
        metrics = {}
        
        # Completeness metrics
        total_cells = data.size
        missing_cells = data.isnull().sum().sum()
        metrics['completeness_percentage'] = ((total_cells - missing_cells) / total_cells * 100) if total_cells > 0 else 0
        metrics['missing_values_by_column'] = data.isnull().sum().to_dict()
        
        # Uniqueness metrics
        metrics['duplicate_rows'] = data.duplicated().sum()
        metrics['duplicate_percentage'] = (metrics['duplicate_rows'] / len(data) * 100) if len(data) > 0 else 0
        
        # Data type distribution
        type_counts = data.dtypes.value_counts().to_dict()
        metrics['data_type_distribution'] = {str(k): v for k, v in type_counts.items()}
        
        # Numeric column statistics
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            numeric_stats = data[numeric_columns].describe().to_dict()
            metrics['numeric_statistics'] = numeric_stats
            
            # Detect potential outliers using IQR method
            outlier_counts = {}
            for col in numeric_columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = data[(data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))]
                outlier_counts[col] = len(outliers)
            metrics['outlier_counts'] = outlier_counts
        
        # Text column statistics
        text_columns = data.select_dtypes(include=['object']).columns
        if len(text_columns) > 0:
            text_stats = {}
            for col in text_columns:
                text_stats[col] = {
                    'unique_values': data[col].nunique(),
                    'most_common': data[col].mode().iloc[0] if not data[col].mode().empty else None,
                    'avg_length': data[col].astype(str).str.len().mean() if not data[col].empty else 0
                }
            metrics['text_statistics'] = text_stats
        
        return metrics
    
    def _calculate_data_checksum(self, data: pd.DataFrame) -> str:
        """
        Calculates checksum for data integrity verification.
        
        Args:
            data: DataFrame to checksum
            
        Returns:
            Hexadecimal checksum string
            
        Behavior:
        - Creates reproducible checksum of data content
        - Uses hash of data structure and values
        - Enables integrity verification
        """
        import hashlib
        
        # Create a string representation of the data
        data_str = data.to_csv(index=False).encode('utf-8')
        
        # Calculate MD5 hash
        checksum = hashlib.md5(data_str).hexdigest()
        
        return checksum
    
    def _validate_data(self, data: pd.DataFrame, metadata: DataMetadata) -> bool:
        """
        Validates data against configured rules.
        
        Args:
            data: DataFrame to validate
            metadata: Data metadata to update with validation results
            
        Returns:
            True if validation passes, False otherwise
            
        Behavior:
        - Applies all configured validation rules
        - Records specific validation errors
        - Updates metadata with validation status
        """
        validation_errors = []
        
        # Validate minimum rows
        if len(data) < self.validation_rules['min_rows']:
            validation_errors.append(f"Insufficient rows: {len(data)} < {self.validation_rules['min_rows']}")
        
        # Validate minimum columns
        if len(data.columns) < self.validation_rules['min_columns']:
            validation_errors.append(f"Insufficient columns: {len(data.columns)} < {self.validation_rules['min_columns']}")
        
        # Validate missing data percentage
        completeness_percentage = metadata.quality_metrics.get('completeness_percentage', 100)
        missing_percentage = 100 - completeness_percentage
        if missing_percentage > self.validation_rules['max_missing_percentage']:
            validation_errors.append(f"Too much missing data: {missing_percentage:.2f}% > {self.validation_rules['max_missing_percentage']}%")
        
        # Validate file size
        file_size_mb = metadata.file_size_bytes / (1024 * 1024)
        if file_size_mb > self.validation_rules['max_file_size_mb']:
            validation_errors.append(f"File too large: {file_size_mb:.2f}MB > {self.validation_rules['max_file_size_mb']}MB")
        
        # Validate data types
        allowed_types = self.validation_rules['allowed_data_types']
        for col, dtype in metadata.data_types.items():
            if not any(allowed_type in str(dtype) for allowed_type in allowed_types):
                validation_errors.append(f"Invalid data type for column {col}: {dtype}")
        
        # Check for required numeric features
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) < self.validation_rules['required_numeric_features']:
            validation_errors.append(f"Insufficient numeric features: {len(numeric_columns)} < {self.validation_rules['required_numeric_features']}")
        
        # Check for excessive duplicates if enabled
        if self.validation_rules['check_duplicates']:
            duplicate_percentage = metadata.quality_metrics.get('duplicate_percentage', 0)
            if duplicate_percentage > 25.0:  # More than 25% duplicates
                validation_errors.append(f"Excessive duplicate rows: {duplicate_percentage:.2f}%")
        
        # Update metadata with validation results
        metadata.validation_errors = validation_errors
        
        if validation_errors:
            for error in validation_errors:
                self.logger.warning(f"Data validation error: {error}")
            return False
        
        self.logger.info("Data validation passed")
        return True
    
    def get_data(self) -> Optional[pd.DataFrame]:
        """
        Returns loaded data.
        
        Returns:
            Loaded DataFrame or None if no data loaded
            
        Behavior:
        - Returns current loaded data
        - Provides access to validated data
        - Returns None if no data has been loaded
        """
        return self.loaded_data
    
    def get_metadata(self) -> Optional[DataMetadata]:
        """
        Returns data metadata.
        
        Returns:
            DataMetadata object or None if no data loaded
            
        Behavior:
        - Returns complete metadata for loaded data
        - Includes quality metrics and validation status
        - Returns None if no data has been loaded
        """
        return self.metadata
    
    def get_features(self) -> List[str]:
        """
        Returns list of feature names.
        
        Returns:
            List of column names or empty list if no data loaded
            
        Behavior:
        - Returns column names from loaded data
        - Maintains order from original data
        - Returns empty list if no data loaded
        """
        if self.loaded_data is not None:
            return self.loaded_data.columns.tolist()
        return []
    
    def get_numeric_features(self) -> List[str]:
        """
        Returns list of numeric feature names.
        
        Returns:
            List of numeric column names
            
        Behavior:
        - Filters features to include only numeric types
        - Returns column names suitable for mathematical operations
        - Returns empty list if no numeric features or no data loaded
        """
        if self.loaded_data is not None:
            numeric_columns = self.loaded_data.select_dtypes(include=[np.number]).columns
            return numeric_columns.tolist()
        return []
    
    def get_categorical_features(self) -> List[str]:
        """
        Returns list of categorical feature names.
        
        Returns:
            List of categorical column names
            
        Behavior:
        - Filters features to include only categorical types
        - Returns column names containing text or categorical data
        - Returns empty list if no categorical features or no data loaded
        """
        if self.loaded_data is not None:
            categorical_columns = self.loaded_data.select_dtypes(include=['object', 'category']).columns
            return categorical_columns.tolist()
        return []
    
    def validate_data_integrity(self) -> bool:
        """
        Validates integrity of loaded data against metadata.
        
        Returns:
            True if data integrity is maintained, False otherwise
            
        Behavior:
        - Compares current data checksum with metadata checksum
        - Verifies data structure matches metadata
        - Returns integrity status
        """
        if self.loaded_data is None or self.metadata is None:
            return False
        
        # Recalculate checksum and compare
        current_checksum = self._calculate_data_checksum(self.loaded_data)
        return current_checksum == self.metadata.checksum
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Returns comprehensive summary of loaded data.
        
        Returns:
            Dictionary containing data summary information
            
        Behavior:
        - Provides overview of data characteristics
        - Includes quality metrics and validation status
        - Useful for monitoring and reporting
        """
        if self.loaded_data is None or self.metadata is None:
            return {"status": "no_data_loaded"}
        
        return {
            "source_file": self.metadata.source_file,
            "file_format": self.metadata.file_format,
            "shape": {"rows": self.metadata.rows, "columns": self.metadata.columns},
            "features": {
                "total": len(self.metadata.features),
                "numeric": len(self.get_numeric_features()),
                "categorical": len(self.get_categorical_features())
            },
            "data_types": self.metadata.data_types,
            "validation_status": self.metadata.validation_status,
            "validation_errors": self.metadata.validation_errors,
            "quality_metrics": self.metadata.quality_metrics,
            "load_timestamp": self.metadata.load_timestamp.isoformat(),
            "file_size_mb": self.metadata.file_size_bytes / (1024 * 1024),
            "checksum": self.metadata.checksum
        }
    
    def export_data(self, output_path: Union[str, Path], format: str = 'csv', **kwargs) -> bool:
        """
        Exports loaded data to file.
        
        Args:
            output_path: Path for output file
            format: Output format ('csv', 'parquet', 'json')
            **kwargs: Format-specific export parameters
            
        Returns:
            True if export successful, False otherwise
            
        Behavior:
        - Exports data in specified format
        - Preserves data structure and types where possible
        - Handles export errors gracefully
        """
        if self.loaded_data is None:
            self.logger.error("No data loaded to export")
            return False
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == 'csv':
                csv_kwargs = {
                    'index': kwargs.get('index', False),
                    'encoding': kwargs.get('encoding', 'utf-8'),
                    'sep': kwargs.get('sep', ',')
                }
                self.loaded_data.to_csv(output_path, **csv_kwargs)
                
            elif format.lower() == 'parquet':
                parquet_kwargs = {
                    'engine': kwargs.get('engine', 'auto'),
                    'compression': kwargs.get('compression', 'snappy')
                }
                self.loaded_data.to_parquet(output_path, **parquet_kwargs)
                
            elif format.lower() == 'json':
                json_kwargs = {
                    'orient': kwargs.get('orient', 'records'),
                    'indent': kwargs.get('indent', 2)
                }
                self.loaded_data.to_json(output_path, **json_kwargs)
                
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return False
            
            self.logger.info(f"Data exported successfully to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting data to {output_path}: {e}")
            return False
    
    def clear_data(self) -> None:
        """
        Clears loaded data and metadata.
        
        Behavior:
        - Releases memory used by loaded data
        - Clears metadata and error tracking
        - Prepares handler for loading new data
        """
        self.loaded_data = None
        self.metadata = None
        self.loading_errors.clear()
        self.logger.debug("Data and metadata cleared")
    
    def get_loading_errors(self) -> List[str]:
        """
        Returns list of loading errors.
        
        Returns:
            List of error messages from loading operations
            
        Behavior:
        - Provides detailed error information for troubleshooting
        - Includes errors from all loading attempts
        - Clears on successful data loading
        """
        return self.loading_errors.copy()
    
    def update_data(self, data: pd.DataFrame, update_metadata: bool = True) -> bool:
        """
        Updates the loaded data with new data.
        
        Args:
            data: New DataFrame to replace current data
            update_metadata: Whether to update metadata for the new data
            
        Returns:
            True if update successful, False otherwise
            
        Behavior:
        - Replaces current loaded data with new data
        - Optionally updates metadata to reflect new data characteristics
        - Validates new data structure
        - Maintains data handler state consistency
        """
        try:
            if not isinstance(data, pd.DataFrame):
                self.logger.error("Update data must be a pandas DataFrame")
                return False
            
            # Store the new data
            self.loaded_data = data.copy()
            
            # Update metadata if requested
            if update_metadata and self.metadata:
                self.metadata.features = list(data.columns)
                self.metadata.data_types = {col: str(dtype) for col, dtype in data.dtypes.items()}
                # Update other metadata fields as needed
                
                # Re-run validation on new data
                validation_result = self._validate_data(data)
                self.metadata.validation_status = "valid" if validation_result[0] else "invalid"
                self.metadata.validation_errors = validation_result[1]
            
            self.logger.info(f"Data updated successfully. New shape: {data.shape}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update data: {e}")
            return False
