"""Data Processor Unit

This module implements the DataProcessor class that manages data processing pipelines,
six-dataset splitting, and data transformation coordination.

Behavioral Specification:
- Manages processing pipeline execution with proper state tracking
- Implements six-dataset splitting with configurable ratios
- Provides data transformation tracking and reversibility
- Coordinates data flow between processing stages
- Maintains data integrity throughout processing operations
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
class SplitConfiguration:
    """Configuration for six-dataset splitting"""
    ratios: Dict[str, float]
    random_seed: Optional[int] = None
    stratify_column: Optional[str] = None
    temporal_split: bool = False
    temporal_column: Optional[str] = None
    shuffle: bool = True
    
    def __post_init__(self):
        """Validate split configuration after initialization"""
        if not self.ratios or len(self.ratios) != 6:
            raise ValueError("Split configuration must specify exactly 6 dataset ratios")
        
        expected_keys = {'d1', 'd2', 'd3', 'd4', 'd5', 'd6'}
        if set(self.ratios.keys()) != expected_keys:
            raise ValueError(f"Split ratios must have keys {expected_keys}")
        
        ratio_sum = sum(self.ratios.values())
        if abs(ratio_sum - 1.0) > 0.001:
            raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")
        
        for key, value in self.ratios.items():
            if not 0 < value < 1:
                raise ValueError(f"Split ratio for {key} must be between 0 and 1, got {value}")


@dataclass
class SplitResult:
    """Result of dataset splitting operation"""
    datasets: Dict[str, pd.DataFrame]
    split_metadata: Dict[str, Any]
    execution_metrics: Dict[str, Any]
    temporal_boundaries: Optional[Dict[str, Any]] = None


@dataclass
class ProcessingState:
    """Tracks processing pipeline state"""
    stage: str
    input_shape: Tuple[int, int]
    output_shape: Tuple[int, int]
    transformation_applied: str
    timestamp: datetime
    processing_time_seconds: float
    memory_usage_mb: float
    
    
class DataProcessor:
    """
    Manages data processing pipelines and dataset operations.
    
    Behavioral Contract:
    - MUST execute processing steps in correct order
    - MUST maintain data integrity throughout processing
    - MUST provide accurate six-dataset splitting
    - MUST track all transformations for reversibility
    - MUST handle processing errors gracefully
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize data processor.
        
        Args:
            random_seed: Random seed for reproducible operations
            
        Behaviors:
        - Sets up processing pipeline infrastructure
        - Initializes transformation tracking
        - Prepares state management structures
        """
        self.logger = logging.getLogger(__name__)
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.processing_history: List[ProcessingState] = []
        self.current_data: Optional[pd.DataFrame] = None
        self.split_datasets: Optional[Dict[str, pd.DataFrame]] = None
        self.split_metadata: Optional[Dict[str, Any]] = None
        
        # Default split configuration
        self.default_split_config = SplitConfiguration(
            ratios={'d1': 0.5, 'd2': 0.1, 'd3': 0.1, 'd4': 0.1, 'd5': 0.1, 'd6': 0.1},
            random_seed=random_seed,
            shuffle=True
        )
    
    def set_data(self, data: pd.DataFrame) -> None:
        """
        Sets the input data for processing.
        
        Args:
            data: Input DataFrame for processing
            
        Behavior:
        - Validates input data structure and quality
        - Stores data for processing operations
        - Resets any existing processing state
        """
        if data is None or data.empty:
            raise ValueError("Input data cannot be None or empty")
        
        if len(data) < 60:
            self.logger.warning(f"Data has only {len(data)} samples, minimum recommended is 60")
        
        self.current_data = data.copy()
        self.split_datasets = None
        self.split_metadata = None
        self.processing_history.clear()
        
        self.logger.info(f"Set input data: {data.shape[0]} rows, {data.shape[1]} columns")
    
    def validate_split_configuration(self, config: SplitConfiguration) -> Tuple[bool, List[str]]:
        """
        Validates split configuration against current data.
        
        Args:
            config: Split configuration to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
            
        Behavior:
        - Checks configuration completeness and validity
        - Validates against current data characteristics
        - Returns specific validation errors if any
        """
        errors = []
        
        if self.current_data is None:
            errors.append("No data loaded for split validation")
            return False, errors
        
        # Check minimum sample requirements
        total_samples = len(self.current_data)
        min_samples_per_split = 10
        
        for dataset_key, ratio in config.ratios.items():
            expected_samples = int(total_samples * ratio)
            if expected_samples < min_samples_per_split:
                errors.append(
                    f"Dataset {dataset_key} would have {expected_samples} samples "
                    f"(ratio={ratio:.3f}), minimum required is {min_samples_per_split}"
                )
        
        # Validate temporal split requirements
        if config.temporal_split:
            if not config.temporal_column:
                errors.append("Temporal split requires temporal_column to be specified")
            elif config.temporal_column not in self.current_data.columns:
                errors.append(f"Temporal column '{config.temporal_column}' not found in data")
            else:
                # Check if temporal column is sortable
                try:
                    temp_sorted = self.current_data[config.temporal_column].sort_values()
                    if temp_sorted.isnull().any():
                        errors.append(f"Temporal column '{config.temporal_column}' contains null values")
                except Exception as e:
                    errors.append(f"Cannot sort temporal column '{config.temporal_column}': {e}")
        
        # Validate stratification requirements
        if config.stratify_column:
            if config.stratify_column not in self.current_data.columns:
                errors.append(f"Stratify column '{config.stratify_column}' not found in data")
            else:
                # Check if stratification is feasible
                unique_values = self.current_data[config.stratify_column].nunique()
                if unique_values > len(self.current_data) * 0.1:
                    errors.append(
                        f"Stratify column '{config.stratify_column}' has too many unique values "
                        f"({unique_values}) for effective stratification"
                    )
        
        is_valid = len(errors) == 0
        if is_valid:
            self.logger.info("Split configuration validation passed")
        else:
            self.logger.warning(f"Split configuration validation failed: {len(errors)} errors")
        
        return is_valid, errors
    
    def execute_split(self, config: Optional[SplitConfiguration] = None) -> SplitResult:
        """
        Executes six-dataset splitting operation.
        
        Args:
            config: Split configuration, uses default if None
            
        Returns:
            SplitResult containing six datasets and metadata
            
        Behavior:
        - Validates configuration before execution
        - Splits data according to specified ratios and method
        - Maintains data integrity and ordering requirements
        - Generates comprehensive split metadata
        """
        start_time = datetime.now()
        
        if self.current_data is None:
            raise ValueError("No data loaded for splitting operation")
        
        if config is None:
            config = self.default_split_config
        
        # Validate configuration
        is_valid, errors = self.validate_split_configuration(config)
        if not is_valid:
            raise ValueError(f"Invalid split configuration: {'; '.join(errors)}")
        
        # Prepare data for splitting
        data_to_split = self.current_data.copy()
        total_samples = len(data_to_split)
        
        # Handle temporal splitting
        if config.temporal_split and config.temporal_column:
            data_to_split = data_to_split.sort_values(config.temporal_column).reset_index(drop=True)
            temporal_boundaries = self._compute_temporal_boundaries(data_to_split, config)
        else:
            temporal_boundaries = None
            
        # Handle shuffling (only if not temporal split)
        if config.shuffle and not config.temporal_split:
            if config.random_seed is not None:
                data_to_split = data_to_split.sample(frac=1, random_state=config.random_seed).reset_index(drop=True)
            else:
                data_to_split = data_to_split.sample(frac=1).reset_index(drop=True)
        
        # Calculate split indices
        split_indices = self._calculate_split_indices(total_samples, config.ratios)
        
        # Execute the split
        datasets = {}
        current_idx = 0
        
        for dataset_key in ['d1', 'd2', 'd3', 'd4', 'd5', 'd6']:
            start_idx = current_idx
            end_idx = current_idx + split_indices[dataset_key]
            
            if end_idx > start_idx:
                datasets[dataset_key] = data_to_split.iloc[start_idx:end_idx].copy()
            else:
                # Handle edge case where ratio results in 0 samples
                datasets[dataset_key] = pd.DataFrame(columns=data_to_split.columns)
            
            current_idx = end_idx
        
        # Generate metadata
        execution_time = (datetime.now() - start_time).total_seconds()
        split_metadata = self._generate_split_metadata(datasets, config, execution_time)
        
        # Generate execution metrics
        execution_metrics = {
            'processing_time_seconds': execution_time,
            'total_samples_processed': total_samples,
            'split_configuration': {
                'ratios': config.ratios,
                'temporal_split': config.temporal_split,
                'shuffle': config.shuffle,
                'random_seed': config.random_seed
            },
            'memory_usage_estimate_mb': data_to_split.memory_usage(deep=True).sum() / (1024 * 1024)
        }
        
        # Store results
        self.split_datasets = datasets
        self.split_metadata = split_metadata
        
        # Record processing state
        self._record_processing_state(
            stage="split_execution",
            input_shape=self.current_data.shape,
            output_shape=(sum(len(ds) for ds in datasets.values()), self.current_data.shape[1]),
            transformation="six_dataset_split",
            processing_time=execution_time
        )
        
        self.logger.info(f"Successfully split data into 6 datasets in {execution_time:.3f} seconds")
        
        return SplitResult(
            datasets=datasets,
            split_metadata=split_metadata,
            execution_metrics=execution_metrics,
            temporal_boundaries=temporal_boundaries
        )
    
    def _calculate_split_indices(self, total_samples: int, ratios: Dict[str, float]) -> Dict[str, int]:
        """
        Calculates exact sample indices for each split.
        
        Args:
            total_samples: Total number of samples to split
            ratios: Ratios for each dataset
            
        Returns:
            Dictionary mapping dataset keys to sample counts
            
        Behavior:
        - Ensures exact sample allocation without rounding errors
        - Handles remainder samples by distributing to largest splits
        - Maintains total sample count preservation
        """
        # Calculate base sample counts
        sample_counts = {}
        total_allocated = 0
        
        for dataset_key in ['d1', 'd2', 'd3', 'd4', 'd5', 'd6']:
            count = int(total_samples * ratios[dataset_key])
            sample_counts[dataset_key] = count
            total_allocated += count
        
        # Distribute remainder samples to maintain exact count
        remainder = total_samples - total_allocated
        if remainder > 0:
            # Sort by original ratio (descending) to give remainder to largest splits
            sorted_keys = sorted(ratios.keys(), key=lambda k: ratios[k], reverse=True)
            for i in range(remainder):
                key = sorted_keys[i % 6]
                sample_counts[key] += 1
        
        # Verify total preservation
        assert sum(sample_counts.values()) == total_samples, "Sample count preservation failed"
        
        return sample_counts
    
    def _compute_temporal_boundaries(self, data: pd.DataFrame, config: SplitConfiguration) -> Dict[str, Any]:
        """
        Computes temporal boundaries for temporal splitting.
        
        Args:
            data: Data sorted by temporal column
            config: Split configuration with temporal settings
            
        Returns:
            Dictionary with temporal boundary information
        """
        temporal_col = config.temporal_column
        temporal_values = data[temporal_col]
        
        boundaries = {}
        total_samples = len(data)
        current_idx = 0
        
        for dataset_key in ['d1', 'd2', 'd3', 'd4', 'd5', 'd6']:
            ratio = config.ratios[dataset_key]
            end_idx = current_idx + int(total_samples * ratio)
            end_idx = min(end_idx, total_samples - 1)
            
            if current_idx < len(temporal_values):
                start_time = temporal_values.iloc[current_idx]
                end_time = temporal_values.iloc[min(end_idx, len(temporal_values) - 1)]
                
                boundaries[dataset_key] = {
                    'start_time': start_time,
                    'end_time': end_time,
                    'start_index': current_idx,
                    'end_index': end_idx,
                    'sample_count': end_idx - current_idx + 1
                }
            
            current_idx = end_idx + 1
        
        return boundaries
    
    def _generate_split_metadata(self, datasets: Dict[str, pd.DataFrame], 
                                config: SplitConfiguration, execution_time: float) -> Dict[str, Any]:
        """
        Generates comprehensive metadata for split operation.
        
        Args:
            datasets: Split datasets
            config: Split configuration used
            execution_time: Time taken for split operation
            
        Returns:
            Dictionary containing split metadata
        """
        metadata = {
            'split_timestamp': datetime.now().isoformat(),
            'execution_time_seconds': execution_time,
            'configuration': {
                'ratios': config.ratios,
                'random_seed': config.random_seed,
                'temporal_split': config.temporal_split,
                'temporal_column': config.temporal_column,
                'stratify_column': config.stratify_column,
                'shuffle': config.shuffle
            },
            'dataset_statistics': {},
            'integrity_verification': {}
        }
        
        # Generate statistics for each dataset
        total_original_samples = len(self.current_data)
        total_split_samples = sum(len(ds) for ds in datasets.values())
        
        for dataset_key, dataset in datasets.items():
            metadata['dataset_statistics'][dataset_key] = {
                'sample_count': len(dataset),
                'actual_ratio': len(dataset) / total_original_samples if total_original_samples > 0 else 0,
                'expected_ratio': config.ratios[dataset_key],
                'feature_count': len(dataset.columns),
                'memory_usage_mb': dataset.memory_usage(deep=True).sum() / (1024 * 1024),
                'data_types': {col: str(dtype) for col, dtype in dataset.dtypes.items()}
            }
        
        # Integrity verification
        metadata['integrity_verification'] = {
            'total_samples_preserved': total_split_samples == total_original_samples,
            'original_sample_count': total_original_samples,
            'split_sample_count': total_split_samples,
            'sample_count_difference': total_split_samples - total_original_samples,
            'all_features_preserved': all(
                set(ds.columns) == set(self.current_data.columns) 
                for ds in datasets.values() if not ds.empty
            )
        }
        
        return metadata
    
    def _record_processing_state(self, stage: str, input_shape: Tuple[int, int], 
                               output_shape: Tuple[int, int], transformation: str, 
                               processing_time: float) -> None:
        """
        Records processing state for transformation tracking.
        
        Args:
            stage: Processing stage identifier
            input_shape: Shape of input data
            output_shape: Shape of output data
            transformation: Description of transformation applied
            processing_time: Time taken for processing
        """
        import psutil
        import os
        
        # Get current memory usage
        process = psutil.Process(os.getpid())
        memory_usage_mb = process.memory_info().rss / (1024 * 1024)
        
        state = ProcessingState(
            stage=stage,
            input_shape=input_shape,
            output_shape=output_shape,
            transformation_applied=transformation,
            timestamp=datetime.now(),
            processing_time_seconds=processing_time,
            memory_usage_mb=memory_usage_mb
        )
        
        self.processing_history.append(state)
        self.logger.debug(f"Recorded processing state: {stage} - {transformation}")
    
    def get_split_datasets(self) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Returns split datasets if available.
        
        Returns:
            Dictionary of split datasets or None if no split has been performed
            
        Behavior:
        - Returns current split datasets
        - Returns None if no split operation has been executed
        - Provides access to all six datasets
        """
        return self.split_datasets
    
    def get_split_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Returns split metadata if available.
        
        Returns:
            Split metadata dictionary or None if no split has been performed
            
        Behavior:
        - Returns comprehensive split metadata
        - Includes configuration, statistics, and verification data
        - Returns None if no split operation has been executed
        """
        return self.split_metadata
    
    def get_processing_history(self) -> List[ProcessingState]:
        """
        Returns processing history for transformation tracking.
        
        Returns:
            List of ProcessingState objects representing processing pipeline
            
        Behavior:
        - Returns complete processing history
        - Enables transformation reversibility analysis
        - Useful for debugging and auditing
        """
        return self.processing_history.copy()
    
    def get_training_datasets(self, training_keys: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Returns datasets designated for training.
        
        Args:
            training_keys: List of dataset keys for training (default: ['d1', 'd2'])
            
        Returns:
            Dictionary of training datasets
            
        Behavior:
        - Returns subset of split datasets designated for training
        - Default training datasets are d1 and d2
        - Used for parameter computation in normalization
        """
        if self.split_datasets is None:
            raise ValueError("No split datasets available. Execute split operation first.")
        
        if training_keys is None:
            training_keys = ['d1', 'd2']
        
        training_datasets = {}
        for key in training_keys:
            if key in self.split_datasets:
                training_datasets[key] = self.split_datasets[key]
            else:
                self.logger.warning(f"Training dataset key '{key}' not found in split datasets")
        
        return training_datasets
    
    def verify_data_integrity(self) -> Dict[str, Any]:
        """
        Verifies data integrity after processing operations.
        
        Returns:
            Dictionary containing integrity verification results
            
        Behavior:
        - Checks sample count preservation across operations
        - Verifies feature consistency across datasets
        - Validates data type preservation
        - Reports any integrity issues found
        """
        integrity_report = {
            'timestamp': datetime.now().isoformat(),
            'checks_performed': [],
            'issues_found': [],
            'integrity_status': 'unknown'
        }
        
        if self.current_data is None:
            integrity_report['issues_found'].append("No original data available for comparison")
            integrity_report['integrity_status'] = 'cannot_verify'
            return integrity_report
        
        if self.split_datasets is None:
            integrity_report['issues_found'].append("No split datasets available for verification")
            integrity_report['integrity_status'] = 'cannot_verify'
            return integrity_report
        
        # Check sample count preservation
        original_count = len(self.current_data)
        split_count = sum(len(ds) for ds in self.split_datasets.values())
        
        integrity_report['checks_performed'].append('sample_count_preservation')
        if original_count != split_count:
            integrity_report['issues_found'].append(
                f"Sample count mismatch: original={original_count}, split_total={split_count}"
            )
        
        # Check feature consistency
        original_features = set(self.current_data.columns)
        integrity_report['checks_performed'].append('feature_consistency')
        
        for dataset_key, dataset in self.split_datasets.items():
            if not dataset.empty:
                dataset_features = set(dataset.columns)
                if dataset_features != original_features:
                    integrity_report['issues_found'].append(
                        f"Feature mismatch in dataset {dataset_key}: "
                        f"missing={original_features - dataset_features}, "
                        f"extra={dataset_features - original_features}"
                    )
        
        # Check data type preservation
        original_dtypes = self.current_data.dtypes.to_dict()
        integrity_report['checks_performed'].append('data_type_preservation')
        
        for dataset_key, dataset in self.split_datasets.items():
            if not dataset.empty:
                for col in dataset.columns:
                    if col in original_dtypes:
                        if str(dataset[col].dtype) != str(original_dtypes[col]):
                            integrity_report['issues_found'].append(
                                f"Data type mismatch in {dataset_key}.{col}: "
                                f"original={original_dtypes[col]}, current={dataset[col].dtype}"
                            )
        
        # Determine overall integrity status
        if len(integrity_report['issues_found']) == 0:
            integrity_report['integrity_status'] = 'verified'
        else:
            integrity_report['integrity_status'] = 'issues_found'
        
        return integrity_report
    
    def export_split_datasets(self, output_dir: Union[str, Path], 
                            format: str = 'csv', **kwargs) -> bool:
        """
        Exports split datasets to files.
        
        Args:
            output_dir: Directory to save datasets
            format: Export format ('csv', 'parquet', 'json')
            **kwargs: Format-specific export parameters
            
        Returns:
            True if export successful, False otherwise
            
        Behavior:
        - Exports all six datasets to separate files
        - Creates output directory if needed
        - Includes metadata file with split information
        - Handles export errors gracefully
        """
        if self.split_datasets is None:
            self.logger.error("No split datasets available for export")
            return False
        
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Export each dataset
            for dataset_key, dataset in self.split_datasets.items():
                if dataset.empty:
                    self.logger.warning(f"Dataset {dataset_key} is empty, skipping export")
                    continue
                
                filename = f"{dataset_key}.{format}"
                file_path = output_path / filename
                
                if format.lower() == 'csv':
                    csv_kwargs = {
                        'index': kwargs.get('index', False),
                        'encoding': kwargs.get('encoding', 'utf-8')
                    }
                    dataset.to_csv(file_path, **csv_kwargs)
                    
                elif format.lower() == 'parquet':
                    parquet_kwargs = {
                        'engine': kwargs.get('engine', 'auto'),
                        'compression': kwargs.get('compression', 'snappy')
                    }
                    dataset.to_parquet(file_path, **parquet_kwargs)
                    
                elif format.lower() == 'json':
                    json_kwargs = {
                        'orient': kwargs.get('orient', 'records'),
                        'indent': kwargs.get('indent', 2)
                    }
                    dataset.to_json(file_path, **json_kwargs)
                    
                else:
                    self.logger.error(f"Unsupported export format: {format}")
                    return False
            
            # Export metadata
            if self.split_metadata:
                metadata_path = output_path / "split_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(self.split_metadata, f, indent=2, default=str)
            
            self.logger.info(f"Successfully exported split datasets to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting split datasets: {e}")
            return False
    
    def clear_processing_state(self) -> None:
        """
        Clears all processing state and data.
        
        Behavior:
        - Releases memory used by current data and split datasets
        - Clears processing history
        - Prepares processor for new data
        """
        self.current_data = None
        self.split_datasets = None
        self.split_metadata = None
        self.processing_history.clear()
        
        self.logger.debug("Processing state cleared")
