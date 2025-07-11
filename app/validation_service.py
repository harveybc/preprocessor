"""
Validation Service for Preprocessor System

This module provides comprehensive data validation, business rule
enforcement, and quality assurance for preprocessing operations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ValidationLevel(Enum):
    """Validation strictness levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]
    suggestions: List[str]


class ValidationService:
    """
    Comprehensive validation service for data quality assurance
    and business rule enforcement.
    
    Behavioral Requirements:
    - BR-VALID-001: Validate data completeness and consistency
    - BR-VALID-002: Enforce business rules and constraints
    - BR-VALID-003: Provide quality metrics and recommendations
    - BR-VALID-004: Support configurable validation levels
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.business_rules: Dict[str, callable] = {}
        self.quality_thresholds = {
            'missing_data_threshold': 0.1,  # 10% missing data threshold
            'outlier_threshold': 3.0,       # Z-score threshold for outliers
            'correlation_threshold': 0.95,   # High correlation threshold
            'variance_threshold': 0.01      # Low variance threshold
        }
    
    def validate_dataset(self, data: pd.DataFrame, config: Dict[str, Any]) -> ValidationResult:
        """
        Comprehensive dataset validation including structure,
        quality, and business rules.
        
        Args:
            data: Dataset to validate
            config: Validation configuration
            
        Returns:
            ValidationResult with comprehensive validation information
        """
        errors = []
        warnings = []
        metrics = {}
        suggestions = []
        
        # Basic structure validation
        structure_result = self._validate_structure(data, config)
        errors.extend(structure_result['errors'])
        warnings.extend(structure_result['warnings'])
        metrics.update(structure_result['metrics'])
        
        # Data quality validation
        quality_result = self._validate_quality(data, config)
        errors.extend(quality_result['errors'])
        warnings.extend(quality_result['warnings'])
        metrics.update(quality_result['metrics'])
        suggestions.extend(quality_result['suggestions'])
        
        # Business rules validation
        if self.validation_level in [ValidationLevel.STANDARD, ValidationLevel.STRICT]:
            business_result = self._validate_business_rules(data, config)
            errors.extend(business_result['errors'])
            warnings.extend(business_result['warnings'])
            metrics.update(business_result['metrics'])
        
        # Preprocessing-specific validation
        preprocessing_result = self._validate_preprocessing_requirements(data, config)
        errors.extend(preprocessing_result['errors'])
        warnings.extend(preprocessing_result['warnings'])
        metrics.update(preprocessing_result['metrics'])
        suggestions.extend(preprocessing_result['suggestions'])
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            suggestions=suggestions
        )
    
    def _validate_structure(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate dataset structure and format."""
        errors = []
        warnings = []
        metrics = {}
        
        # Check if data is empty
        if data.empty:
            errors.append("Dataset is empty")
            return {'errors': errors, 'warnings': warnings, 'metrics': metrics}
        
        # Record basic metrics
        metrics['total_rows'] = len(data)
        metrics['total_columns'] = len(data.columns)
        metrics['memory_usage_mb'] = data.memory_usage(deep=True).sum() / 1024 / 1024
        
        # Validate required columns
        required_columns = config.get('required_columns', [])
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {list(missing_columns)}")
        
        # Check for duplicate columns
        duplicate_columns = data.columns[data.columns.duplicated()].tolist()
        if duplicate_columns:
            warnings.append(f"Duplicate column names found: {duplicate_columns}")
        
        # Validate data types
        for column in data.columns:
            if column in config.get('numeric_columns', []):
                if not pd.api.types.is_numeric_dtype(data[column]):
                    errors.append(f"Column '{column}' should be numeric but has type {data[column].dtype}")
        
        return {'errors': errors, 'warnings': warnings, 'metrics': metrics}
    
    def _validate_quality(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data quality and completeness."""
        errors = []
        warnings = []
        metrics = {}
        suggestions = []
        
        # Missing data analysis
        missing_data = data.isnull().sum()
        missing_percentages = (missing_data / len(data)) * 100
        metrics['missing_data_by_column'] = missing_percentages.to_dict()
        
        # Check missing data thresholds
        high_missing_columns = missing_percentages[
            missing_percentages > self.quality_thresholds['missing_data_threshold'] * 100
        ]
        
        if not high_missing_columns.empty:
            if self.validation_level == ValidationLevel.STRICT:
                errors.append(f"Columns with high missing data: {high_missing_columns.to_dict()}")
            else:
                warnings.append(f"Columns with high missing data: {high_missing_columns.to_dict()}")
                suggestions.append("Consider imputation or removal of columns with high missing data")
        
        # Outlier detection for numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        outlier_metrics = {}
        
        for column in numeric_columns:
            if column in data.columns:
                z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
                outlier_count = (z_scores > self.quality_thresholds['outlier_threshold']).sum()
                outlier_percentage = (outlier_count / len(data)) * 100
                outlier_metrics[column] = {
                    'outlier_count': outlier_count,
                    'outlier_percentage': outlier_percentage
                }
                
                if outlier_percentage > 5:  # More than 5% outliers
                    warnings.append(f"Column '{column}' has {outlier_percentage:.2f}% outliers")
                    suggestions.append(f"Consider outlier treatment for column '{column}'")
        
        metrics['outlier_analysis'] = outlier_metrics
        
        # Variance analysis
        low_variance_columns = []
        for column in numeric_columns:
            if column in data.columns and data[column].var() < self.quality_thresholds['variance_threshold']:
                low_variance_columns.append(column)
        
        if low_variance_columns:
            warnings.append(f"Low variance columns detected: {low_variance_columns}")
            suggestions.append("Consider removing low variance columns")
        
        metrics['low_variance_columns'] = low_variance_columns
        
        return {'errors': errors, 'warnings': warnings, 'metrics': metrics, 'suggestions': suggestions}
    
    def _validate_business_rules(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate business-specific rules and constraints."""
        errors = []
        warnings = []
        metrics = {}
        
        # Apply registered business rules
        for rule_name, rule_function in self.business_rules.items():
            try:
                rule_result = rule_function(data, config)
                if not rule_result['passed']:
                    if rule_result['severity'] == 'error':
                        errors.append(f"Business rule '{rule_name}' failed: {rule_result['message']}")
                    else:
                        warnings.append(f"Business rule '{rule_name}' warning: {rule_result['message']}")
                
                metrics[f'business_rule_{rule_name}'] = rule_result
            except Exception as e:
                errors.append(f"Business rule '{rule_name}' execution failed: {str(e)}")
        
        return {'errors': errors, 'warnings': warnings, 'metrics': metrics}
    
    def _validate_preprocessing_requirements(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate preprocessing-specific requirements."""
        errors = []
        warnings = []
        metrics = {}
        suggestions = []
        
        # Validate minimum data requirements for splitting
        min_rows = config.get('min_rows_for_splitting', 100)
        if len(data) < min_rows:
            errors.append(f"Insufficient data for splitting: {len(data)} rows < {min_rows} required")
        
        # Validate split proportions
        split_proportions = [
            config.get('d1_proportion', 0.33),
            config.get('d2_proportion', 0.083),
            config.get('d3_proportion', 0.083),
            config.get('d4_proportion', 0.33),
            config.get('d5_proportion', 0.083),
            config.get('d6_proportion', 0.083)
        ]
        
        total_proportion = sum(split_proportions)
        if abs(total_proportion - 1.0) > 0.001:
            errors.append(f"Split proportions sum to {total_proportion}, should be 1.0")
        
        # Validate each split will have minimum data
        min_split_size = config.get('min_split_size', 10)
        for i, proportion in enumerate(split_proportions, 1):
            split_size = int(len(data) * proportion)
            if split_size < min_split_size:
                warnings.append(f"Split D{i} will have only {split_size} rows")
        
        metrics['split_validation'] = {
            'total_proportion': total_proportion,
            'split_sizes': [int(len(data) * p) for p in split_proportions]
        }
        
        # Validate normalization requirements
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_columns:
            warnings.append("No numeric columns found for normalization")
        else:
            metrics['numeric_columns_count'] = len(numeric_columns)
            
            # Check for columns with zero variance (will cause normalization issues)
            zero_variance_columns = []
            for column in numeric_columns:
                if data[column].var() == 0:
                    zero_variance_columns.append(column)
            
            if zero_variance_columns:
                warnings.append(f"Columns with zero variance (normalization will fail): {zero_variance_columns}")
                suggestions.append("Remove or handle zero variance columns before normalization")
        
        return {'errors': errors, 'warnings': warnings, 'metrics': metrics, 'suggestions': suggestions}
    
    def register_business_rule(self, name: str, rule_function: callable):
        """Register a custom business rule for validation."""
        self.business_rules[name] = rule_function
    
    def update_quality_thresholds(self, thresholds: Dict[str, float]):
        """Update quality validation thresholds."""
        self.quality_thresholds.update(thresholds)
    
    def validate_normalization_parameters(self, params: Dict[str, Dict[str, float]]) -> ValidationResult:
        """Validate normalization parameters for consistency."""
        errors = []
        warnings = []
        metrics = {}
        suggestions = []
        
        for column, param_dict in params.items():
            if 'min' not in param_dict or 'max' not in param_dict:
                errors.append(f"Missing min/max parameters for column '{column}'")
                continue
            
            min_val = param_dict['min']
            max_val = param_dict['max']
            
            if min_val >= max_val:
                errors.append(f"Invalid range for column '{column}': min ({min_val}) >= max ({max_val})")
            
            if abs(max_val - min_val) < 1e-10:
                warnings.append(f"Very small range for column '{column}': {max_val - min_val}")
                suggestions.append(f"Consider handling zero/near-zero variance for column '{column}'")
        
        metrics['validated_columns'] = len(params)
        metrics['parameter_consistency'] = len(errors) == 0
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            suggestions=suggestions
        )
