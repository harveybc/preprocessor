"""Normalization Handler Unit

This module implements the NormalizationHandler class that manages z-score normalization
with parameter persistence and dual JSON storage capabilities.

Behavioral Specification:
- Computes normalization parameters exclusively from training datasets
- Applies consistent normalization across all datasets using computed parameters
- Persists parameters to separate means.json and stds.json files
- Supports parameter loading and reuse for new data
- Provides denormalization capabilities with high accuracy
- Handles per-feature parameter management independently
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import hashlib


@dataclass
class NormalizationParameters:
    """Container for normalization parameters with metadata"""
    means: Dict[str, float]
    stds: Dict[str, float]
    features: List[str]
    computation_timestamp: datetime
    source_datasets: List[str]
    sample_count: int
    feature_count: int
    checksum: Optional[str] = None
    
    def __post_init__(self):
        """Validate parameters after initialization"""
        if set(self.means.keys()) != set(self.stds.keys()):
            raise ValueError("Means and standard deviations must have identical feature sets")
        
        if set(self.features) != set(self.means.keys()):
            raise ValueError("Feature list must match parameter keys")
        
        for feature, std in self.stds.items():
            if std <= 0:
                raise ValueError(f"Standard deviation for feature '{feature}' must be positive, got {std}")
        
        # Generate checksum for integrity verification
        if self.checksum is None:
            self.checksum = self._compute_checksum()
    
    def _compute_checksum(self) -> str:
        """Computes checksum for parameter integrity verification"""
        # Create reproducible string representation
        params_str = json.dumps({
            'means': self.means,
            'stds': self.stds,
            'features': sorted(self.features)
        }, sort_keys=True)
        
        return hashlib.md5(params_str.encode('utf-8')).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verifies parameter integrity using checksum"""
        return self.checksum == self._compute_checksum()


class NormalizationHandler:
    """
    Manages z-score normalization with parameter persistence.
    
    Behavioral Contract:
    - MUST compute parameters from training datasets only
    - MUST apply identical normalization to all datasets
    - MUST persist parameters to separate JSON files
    - MUST support parameter loading and reuse
    - MUST provide accurate denormalization capabilities
    - MUST handle feature mismatches gracefully
    """
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize normalization handler.
        
        Args:
            tolerance: Numerical tolerance for statistical computations
            
        Behaviors:
        - Sets up normalization infrastructure
        - Initializes parameter storage
        - Configures statistical computation tolerance
        """
        self.logger = logging.getLogger(__name__)
        self.tolerance = tolerance
        self.parameters: Optional[NormalizationParameters] = None
        self.normalization_history: List[Dict[str, Any]] = []
        self.feature_exclusions: List[str] = []
    
    def set_feature_exclusions(self, excluded_features: List[str]) -> None:
        """
        Sets features to exclude from normalization.
        
        Args:
            excluded_features: List of feature names to exclude
            
        Behavior:
        - Excludes specified features from parameter computation
        - Passes excluded features through unchanged during normalization
        - Useful for categorical or already-normalized features
        """
        self.feature_exclusions = excluded_features.copy()
        self.logger.info(f"Set feature exclusions: {self.feature_exclusions}")
    
    def compute_parameters(self, training_datasets: Dict[str, pd.DataFrame], 
                         config: Optional[Dict[str, Any]] = None) -> NormalizationParameters:
        """
        Computes normalization parameters from training datasets only.
        
        Args:
            training_datasets: Dictionary of training datasets (typically d1, d2)
            config: Configuration parameters for computation
            
        Returns:
            NormalizationParameters object with computed statistics
            
        Behavior:
        - Combines training datasets for parameter computation
        - Computes per-feature means and standard deviations
        - Excludes non-numeric and excluded features
        - Validates parameter quality and completeness
        """
        if not training_datasets:
            raise ValueError("Training datasets cannot be empty")
        
        config = config or {}
        exclude_features = config.get('exclude_features', []) + self.feature_exclusions
        
        self.logger.info(f"Computing normalization parameters from {len(training_datasets)} training datasets")
        
        # Combine training datasets
        combined_data = self._combine_training_data(training_datasets)
        
        # Select numeric features for normalization
        numeric_features = self._select_numeric_features(combined_data, exclude_features)
        
        if not numeric_features:
            raise ValueError("No numeric features available for normalization")
        
        # Compute parameters
        means = {}
        stds = {}
        
        for feature in numeric_features:
            feature_data = combined_data[feature].dropna()
            
            if len(feature_data) == 0:
                raise ValueError(f"Feature '{feature}' has no valid data for parameter computation")
            
            if feature_data.nunique() == 1:
                self.logger.warning(f"Feature '{feature}' has constant values (zero variance)")
                # Handle zero variance by setting std to 1.0 to avoid division by zero
                means[feature] = float(feature_data.iloc[0])
                stds[feature] = 1.0
            else:
                means[feature] = float(feature_data.mean())
                stds[feature] = float(feature_data.std(ddof=1))  # Sample standard deviation
                
                # Validate computed parameters
                if not np.isfinite(means[feature]):
                    raise ValueError(f"Invalid mean computed for feature '{feature}': {means[feature]}")
                if not np.isfinite(stds[feature]) or stds[feature] <= 0:
                    raise ValueError(f"Invalid standard deviation computed for feature '{feature}': {stds[feature]}")
        
        # Create parameters object
        parameters = NormalizationParameters(
            means=means,
            stds=stds,
            features=list(numeric_features),
            computation_timestamp=datetime.now(),
            source_datasets=list(training_datasets.keys()),
            sample_count=len(combined_data),
            feature_count=len(numeric_features)
        )
        
        # Store parameters
        self.parameters = parameters
        
        # Record computation in history
        self._record_computation_history(training_datasets, parameters)
        
        self.logger.info(f"Computed parameters for {len(numeric_features)} features from {len(combined_data)} samples")
        
        return parameters
    
    def _combine_training_data(self, training_datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combines training datasets into single DataFrame.
        
        Args:
            training_datasets: Dictionary of training datasets
            
        Returns:
            Combined DataFrame with all training data
            
        Behavior:
        - Concatenates datasets vertically
        - Maintains feature consistency across datasets
        - Handles empty datasets gracefully
        """
        dataset_list = []
        
        for name, dataset in training_datasets.items():
            if dataset.empty:
                self.logger.warning(f"Training dataset '{name}' is empty, skipping")
                continue
            
            dataset_list.append(dataset)
        
        if not dataset_list:
            raise ValueError("All training datasets are empty")
        
        # Combine datasets
        combined = pd.concat(dataset_list, ignore_index=True)
        
        # Verify feature consistency
        first_features = set(dataset_list[0].columns)
        for i, dataset in enumerate(dataset_list[1:], 1):
            dataset_features = set(dataset.columns)
            if dataset_features != first_features:
                missing_in_current = first_features - dataset_features
                extra_in_current = dataset_features - first_features
                self.logger.warning(
                    f"Feature mismatch in training dataset {i}: "
                    f"missing={missing_in_current}, extra={extra_in_current}"
                )
        
        return combined
    
    def _select_numeric_features(self, data: pd.DataFrame, exclude_features: List[str]) -> List[str]:
        """
        Selects numeric features for normalization.
        
        Args:
            data: Input DataFrame
            exclude_features: Features to exclude
            
        Returns:
            List of numeric feature names suitable for normalization
            
        Behavior:
        - Identifies numeric data types
        - Excludes specified features
        - Filters out features with insufficient variance
        """
        # Get numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove excluded features
        numeric_features = [col for col in numeric_columns if col not in exclude_features]
        
        # Check for sufficient variance
        valid_features = []
        for feature in numeric_features:
            feature_data = data[feature].dropna()
            if len(feature_data) < 2:
                self.logger.warning(f"Feature '{feature}' has insufficient data points, excluding")
                continue
            
            variance = feature_data.var()
            if variance < self.tolerance:
                self.logger.warning(f"Feature '{feature}' has very low variance ({variance}), may cause numerical issues")
            
            valid_features.append(feature)
        
        return valid_features
    
    def _record_computation_history(self, training_datasets: Dict[str, pd.DataFrame], 
                                  parameters: NormalizationParameters) -> None:
        """
        Records parameter computation in history.
        
        Args:
            training_datasets: Training datasets used
            parameters: Computed parameters
        """
        history_entry = {
            'timestamp': parameters.computation_timestamp.isoformat(),
            'training_datasets': list(training_datasets.keys()),
            'sample_count': parameters.sample_count,
            'feature_count': parameters.feature_count,
            'features': parameters.features.copy(),
            'checksum': parameters.checksum
        }
        
        self.normalization_history.append(history_entry)
    
    def persist_parameters(self, storage_config: Dict[str, Any]) -> bool:
        """
        Persists normalization parameters to JSON files.
        
        Args:
            storage_config: Configuration for parameter storage
            
        Returns:
            True if persistence successful, False otherwise
            
        Behavior:
        - Saves means and stds to separate JSON files
        - Uses atomic write operations to prevent corruption
        - Creates backup copies if specified
        - Validates file integrity after writing
        """
        if self.parameters is None:
            self.logger.error("No parameters computed for persistence")
            return False
        
        means_file = storage_config.get('means_file', 'means.json')
        stds_file = storage_config.get('stds_file', 'stds.json')
        backup_enabled = storage_config.get('backup', True)
        
        try:
            # Prepare data for persistence
            means_data = {
                'metadata': {
                    'computation_timestamp': self.parameters.computation_timestamp.isoformat(),
                    'source_datasets': self.parameters.source_datasets,
                    'feature_count': self.parameters.feature_count,
                    'sample_count': self.parameters.sample_count,
                    'checksum': self.parameters.checksum
                },
                'means': self.parameters.means
            }
            
            stds_data = {
                'metadata': {
                    'computation_timestamp': self.parameters.computation_timestamp.isoformat(),
                    'source_datasets': self.parameters.source_datasets,
                    'feature_count': self.parameters.feature_count,
                    'sample_count': self.parameters.sample_count,
                    'checksum': self.parameters.checksum
                },
                'stds': self.parameters.stds
            }
            
            # Write files atomically
            success = True
            success &= self._write_json_file(means_file, means_data, backup_enabled)
            success &= self._write_json_file(stds_file, stds_data, backup_enabled)
            
            if success:
                self.logger.info(f"Successfully persisted parameters to {means_file} and {stds_file}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error persisting parameters: {e}")
            return False
    
    def _write_json_file(self, file_path: str, data: Dict[str, Any], backup: bool) -> bool:
        """
        Writes JSON file using atomic operation.
        
        Args:
            file_path: Target file path
            data: Data to write
            backup: Whether to create backup
            
        Returns:
            True if write successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create backup if file exists and backup enabled
            if backup and file_path.exists():
                backup_path = file_path.with_suffix(f'.backup{file_path.suffix}')
                file_path.rename(backup_path)
                self.logger.debug(f"Created backup: {backup_path}")
            
            # Write to temporary file first
            temp_path = file_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            # Verify file integrity
            with open(temp_path, 'r') as f:
                loaded_data = json.load(f)
            
            # Atomic rename
            temp_path.rename(file_path)
            
            # Verify final file
            with open(file_path, 'r') as f:
                final_data = json.load(f)
                assert final_data == loaded_data
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing JSON file {file_path}: {e}")
            return False
    
    def load_parameters(self, storage_config: Dict[str, Any]) -> bool:
        """
        Loads normalization parameters from JSON files.
        
        Args:
            storage_config: Configuration for parameter loading
            
        Returns:
            True if loading successful, False otherwise
            
        Behavior:
        - Loads means and stds from separate JSON files
        - Validates parameter integrity and consistency
        - Reconstructs NormalizationParameters object
        - Verifies checksums for data integrity
        """
        means_file = storage_config.get('means_file', 'means.json')
        stds_file = storage_config.get('stds_file', 'stds.json')
        
        try:
            # Load JSON files
            with open(means_file, 'r') as f:
                means_data = json.load(f)
            
            with open(stds_file, 'r') as f:
                stds_data = json.load(f)
            
            # Validate file consistency
            means_metadata = means_data['metadata']
            stds_metadata = stds_data['metadata']
            
            if means_metadata['checksum'] != stds_metadata['checksum']:
                raise ValueError("Means and stds files have inconsistent checksums")
            
            if means_metadata['computation_timestamp'] != stds_metadata['computation_timestamp']:
                raise ValueError("Means and stds files have different computation timestamps")
            
            # Reconstruct parameters
            parameters = NormalizationParameters(
                means=means_data['means'],
                stds=stds_data['stds'],
                features=list(means_data['means'].keys()),
                computation_timestamp=datetime.fromisoformat(means_metadata['computation_timestamp']),
                source_datasets=means_metadata['source_datasets'],
                sample_count=means_metadata['sample_count'],
                feature_count=means_metadata['feature_count'],
                checksum=means_metadata['checksum']
            )
            
            # Verify integrity
            if not parameters.verify_integrity():
                raise ValueError("Loaded parameters failed integrity verification")
            
            # Store parameters
            self.parameters = parameters
            
            self.logger.info(f"Successfully loaded parameters from {means_file} and {stds_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading parameters: {e}")
            self.parameters = None  # Ensure parameters are cleared on error
            return False
    
    def apply_normalization(self, datasets: Dict[str, pd.DataFrame], 
                          parameters: Optional[NormalizationParameters] = None) -> Dict[str, pd.DataFrame]:
        """
        Applies normalization to datasets using computed parameters.
        
        Args:
            datasets: Dictionary of datasets to normalize
            parameters: Parameters to use (uses stored if None)
            
        Returns:
            Dictionary of normalized datasets
            
        Behavior:
        - Applies identical normalization to all datasets
        - Uses formula: (value - mean) / std for each feature
        - Preserves non-numeric and excluded features unchanged
        - Handles feature mismatches gracefully
        """
        if parameters is None:
            parameters = self.parameters
        
        if parameters is None:
            raise ValueError("No normalization parameters available. Compute or load parameters first.")
        
        normalized_datasets = {}
        
        for dataset_name, dataset in datasets.items():
            if dataset.empty:
                self.logger.warning(f"Dataset '{dataset_name}' is empty, skipping normalization")
                normalized_datasets[dataset_name] = dataset.copy()
                continue
            
            normalized_dataset = self._normalize_single_dataset(dataset, parameters, dataset_name)
            normalized_datasets[dataset_name] = normalized_dataset
        
        self.logger.info(f"Applied normalization to {len(datasets)} datasets using {len(parameters.features)} features")
        
        return normalized_datasets
    
    def _normalize_single_dataset(self, dataset: pd.DataFrame, 
                                parameters: NormalizationParameters, 
                                dataset_name: str) -> pd.DataFrame:
        """
        Normalizes a single dataset.
        
        Args:
            dataset: Dataset to normalize
            parameters: Normalization parameters
            dataset_name: Name for logging
            
        Returns:
            Normalized dataset
        """
        normalized = dataset.copy()
        
        # Track normalization statistics
        normalized_features = []
        missing_features = []
        extra_features = []
        
        # Apply normalization to each parameter feature
        for feature in parameters.features:
            if feature in normalized.columns:
                if pd.api.types.is_numeric_dtype(normalized[feature]):
                    mean = parameters.means[feature]
                    std = parameters.stds[feature]
                    
                    # Apply normalization: (x - mean) / std
                    normalized[feature] = (normalized[feature] - mean) / std
                    normalized_features.append(feature)
                else:
                    self.logger.warning(f"Feature '{feature}' in dataset '{dataset_name}' is not numeric, skipping")
            else:
                missing_features.append(feature)
        
        # Identify extra features
        dataset_features = set(normalized.select_dtypes(include=[np.number]).columns)
        parameter_features = set(parameters.features)
        extra_features = list(dataset_features - parameter_features - set(self.feature_exclusions))
        
        # Log feature handling results
        if missing_features:
            self.logger.warning(f"Dataset '{dataset_name}' missing features: {missing_features}")
        
        if extra_features:
            self.logger.info(f"Dataset '{dataset_name}' has extra numeric features (not normalized): {extra_features}")
        
        self.logger.debug(f"Normalized {len(normalized_features)} features in dataset '{dataset_name}'")
        
        return normalized
    
    def denormalize_data(self, normalized_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                        parameters: Optional[NormalizationParameters] = None) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Denormalizes data to recover original values.
        
        Args:
            normalized_data: Normalized data to denormalize
            parameters: Parameters to use (uses stored if None)
            
        Returns:
            Denormalized data in same format as input
            
        Behavior:
        - Reverses normalization using formula: (normalized * std) + mean
        - Handles both single DataFrames and dictionaries of DataFrames
        - Preserves non-normalized features unchanged
        - Achieves high accuracy in value recovery
        """
        if parameters is None:
            parameters = self.parameters
        
        if parameters is None:
            raise ValueError("No normalization parameters available for denormalization")
        
        if isinstance(normalized_data, dict):
            # Handle dictionary of datasets
            denormalized = {}
            for name, dataset in normalized_data.items():
                denormalized[name] = self._denormalize_single_dataset(dataset, parameters)
            return denormalized
        else:
            # Handle single dataset
            return self._denormalize_single_dataset(normalized_data, parameters)
    
    def _denormalize_single_dataset(self, dataset: pd.DataFrame, 
                                   parameters: NormalizationParameters) -> pd.DataFrame:
        """
        Denormalizes a single dataset.
        
        Args:
            dataset: Normalized dataset
            parameters: Normalization parameters
            
        Returns:
            Denormalized dataset
        """
        denormalized = dataset.copy()
        
        for feature in parameters.features:
            if feature in denormalized.columns:
                if pd.api.types.is_numeric_dtype(denormalized[feature]):
                    mean = parameters.means[feature]
                    std = parameters.stds[feature]
                    
                    # Apply denormalization: (normalized * std) + mean
                    denormalized[feature] = (denormalized[feature] * std) + mean
        
        return denormalized
    
    def validate_normalization_quality(self, normalized_datasets: Dict[str, pd.DataFrame], 
                                     training_keys: List[str] = None) -> Dict[str, Any]:
        """
        Validates normalization quality by checking statistical properties.
        
        Args:
            normalized_datasets: Normalized datasets to validate
            training_keys: Keys of training datasets (default: source_datasets)
            
        Returns:
            Dictionary containing validation metrics and results
            
        Behavior:
        - Verifies training data has mean ≈ 0.0 and std ≈ 1.0
        - Checks normalization consistency across datasets
        - Computes quality metrics and deviations
        - Reports validation status and issues
        """
        if self.parameters is None:
            raise ValueError("No parameters available for quality validation")
        
        if training_keys is None:
            training_keys = self.parameters.source_datasets
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'validation_passed': True,
            'training_validation': {},
            'consistency_validation': {},
            'issues_found': []
        }
        
        # Validate training dataset statistics
        for training_key in training_keys:
            if training_key in normalized_datasets:
                training_stats = self._validate_training_statistics(
                    normalized_datasets[training_key], training_key
                )
                validation_results['training_validation'][training_key] = training_stats
                
                if not training_stats['validation_passed']:
                    validation_results['validation_passed'] = False
                    validation_results['issues_found'].extend(training_stats['issues'])
        
        # Validate consistency across all datasets
        consistency_stats = self._validate_consistency(normalized_datasets)
        validation_results['consistency_validation'] = consistency_stats
        
        if not consistency_stats['validation_passed']:
            validation_results['validation_passed'] = False
            validation_results['issues_found'].extend(consistency_stats['issues'])
        
        return validation_results
    
    def _validate_training_statistics(self, training_dataset: pd.DataFrame, 
                                    dataset_name: str) -> Dict[str, Any]:
        """
        Validates that training dataset has expected normalized statistics.
        
        Args:
            training_dataset: Normalized training dataset
            dataset_name: Dataset name for reporting
            
        Returns:
            Validation results for training statistics
        """
        validation = {
            'dataset_name': dataset_name,
            'validation_passed': True,
            'feature_statistics': {},
            'issues': []
        }
        
        for feature in self.parameters.features:
            if feature in training_dataset.columns:
                feature_data = training_dataset[feature].dropna()
                
                if len(feature_data) > 0:
                    computed_mean = feature_data.mean()
                    computed_std = feature_data.std(ddof=1)
                    
                    # Check mean ≈ 0.0 (±0.1 for individual training datasets)
                    mean_ok = abs(computed_mean) <= 0.1
                    
                    # Check std ≈ 1.0 (±0.1 for individual training datasets)
                    std_ok = abs(computed_std - 1.0) <= 0.1
                    
                    validation['feature_statistics'][feature] = {
                        'mean': computed_mean,
                        'std': computed_std,
                        'mean_validation_passed': mean_ok,
                        'std_validation_passed': std_ok
                    }
                    
                    if not mean_ok:
                        validation['issues'].append(
                            f"Feature '{feature}' in dataset '{dataset_name}' has mean {computed_mean:.6f}, expected ≈ 0.0"
                        )
                        validation['validation_passed'] = False
                    
                    if not std_ok:
                        validation['issues'].append(
                            f"Feature '{feature}' in dataset '{dataset_name}' has std {computed_std:.6f}, expected ≈ 1.0"
                        )
                        validation['validation_passed'] = False
        
        return validation
    
    def _validate_consistency(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Validates normalization consistency across datasets.
        
        Args:
            datasets: Normalized datasets to check
            
        Returns:
            Consistency validation results
        """
        validation = {
            'validation_passed': True,
            'cross_dataset_consistency': {},
            'issues': []
        }
        
        # Check that all datasets have same normalized schema
        first_dataset_name = list(datasets.keys())[0]
        first_normalized_features = set(datasets[first_dataset_name].select_dtypes(include=[np.number]).columns)
        
        for dataset_name, dataset in datasets.items():
            dataset_features = set(dataset.select_dtypes(include=[np.number]).columns)
            
            if dataset_features != first_normalized_features:
                missing = first_normalized_features - dataset_features
                extra = dataset_features - first_normalized_features
                
                validation['issues'].append(
                    f"Dataset '{dataset_name}' feature mismatch: missing={missing}, extra={extra}"
                )
                validation['validation_passed'] = False
        
        validation['cross_dataset_consistency'] = {
            'feature_consistency': validation['validation_passed'],
            'expected_features': list(first_normalized_features),
            'feature_count': len(first_normalized_features)
        }
        
        return validation
    
    def get_parameters(self) -> Optional[NormalizationParameters]:
        """
        Returns current normalization parameters.
        
        Returns:
            Current parameters or None if not computed
            
        Behavior:
        - Provides access to computed parameters
        - Returns None if no parameters available
        - Includes complete parameter metadata
        """
        return self.parameters
    
    def get_normalization_history(self) -> List[Dict[str, Any]]:
        """
        Returns history of normalization computations.
        
        Returns:
            List of computation history entries
            
        Behavior:
        - Provides audit trail of parameter computations
        - Includes timestamps and dataset information
        - Useful for debugging and reproducibility
        """
        return self.normalization_history.copy()
    
    def clear_parameters(self) -> None:
        """
        Clears stored normalization parameters and history.
        
        Behavior:
        - Resets parameters to None
        - Clears computation history
        - Prepares for new parameter computation
        """
        self.parameters = None
        self.normalization_history.clear()
        self.logger.debug("Cleared normalization parameters and history")
