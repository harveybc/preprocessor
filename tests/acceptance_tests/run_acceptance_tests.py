"""
Acceptance Test Suite Runner
===========================

Comprehensive test runner for all acceptance tests with performance monitoring,
reporting, and quality metrics validation.
"""

import pytest
import time
import sys
import json
import os
from pathlib import Path
from unittest import mock
import pandas as pd


def run_acceptance_tests():
    """
    Run all acceptance tests with comprehensive reporting.
    
    Returns:
        dict: Test results and metrics
    """
    print("=" * 60)
    print("PREPROCESSOR ACCEPTANCE TEST SUITE")
    print("=" * 60)
    
    # Test configuration
    test_config = {
        'verbose': True,
        'capture': 'no',  # Show print statements
        'tb': 'short',    # Short traceback format
        'exitfirst': False,  # Don't stop on first failure
        'markers': 'not slow',  # Skip slow tests by default
    }
    
    # Build pytest arguments
    pytest_args = [
        str(Path(__file__).parent),  # Test directory
        '-v',  # Verbose
        '--tb=short',  # Short traceback
        '--no-header',  # No pytest header
        '--durations=10',  # Show 10 slowest tests
    ]
    
    # Add markers for different test categories
    acceptance_test_categories = [
        'temporal_splitting',
        'normalization', 
        'plugin_integration',
        'configuration',
        'backward_compatibility'
    ]
    
    print(f"Running acceptance tests for categories: {', '.join(acceptance_test_categories)}")
    print()
    
    # Track test metrics
    start_time = time.time()
    
    # Run tests
    exit_code = pytest.main(pytest_args)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Generate test report
    report = {
        'status': 'PASSED' if exit_code == 0 else 'FAILED',
        'exit_code': exit_code,
        'total_time': total_time,
        'categories_tested': acceptance_test_categories,
        'test_environment': {
            'python_version': sys.version,
            'pytest_version': pytest.__version__,
            'working_directory': os.getcwd()
        }
    }
    
    print("=" * 60)
    print("ACCEPTANCE TEST RESULTS")
    print("=" * 60)
    print(f"Status: {report['status']}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Exit Code: {exit_code}")
    
    if exit_code == 0:
        print("\n✅ All acceptance tests PASSED!")
        print("The preprocessor system meets all business requirements.")
    else:
        print("\n❌ Some acceptance tests FAILED!")
        print("Review test output above for specific failure details.")
    
    print("=" * 60)
    
    return report


def run_performance_validation():
    """
    Run performance-specific acceptance tests.
    
    Returns:
        dict: Performance metrics
    """
    print("\nRunning performance validation tests...")
    
    # Performance-specific pytest arguments
    perf_args = [
        str(Path(__file__).parent / 'test_acceptance_core.py::TestSixDatasetTemporalSplitting::test_performance_requirements_large_dataset'),
        '-v',
        '--tb=short'
    ]
    
    start_time = time.time()
    exit_code = pytest.main(perf_args)
    end_time = time.time()
    
    perf_time = end_time - start_time
    
    perf_report = {
        'performance_test_status': 'PASSED' if exit_code == 0 else 'FAILED',
        'performance_test_time': perf_time
    }
    
    print(f"Performance test completed in {perf_time:.2f} seconds")
    return perf_report


def validate_test_coverage():
    """
    Validate that all acceptance criteria are covered by tests.
    
    Returns:
        dict: Coverage validation results
    """
    print("\nValidating test coverage against acceptance criteria...")
    
    # Expected test coverage mapping
    acceptance_criteria_coverage = {
        'AC1_SixDatasetSplitting': [
            'test_standard_temporal_split_with_configurable_ratios',
            'test_minimum_dataset_size_validation', 
            'test_custom_split_ratio_validation',
            'test_performance_requirements_large_dataset'
        ],
        'AC2_DualZScoreNormalization': [
            'test_parameter_computation_from_training_datasets',
            'test_consistent_normalization_across_datasets',
            'test_parameter_persistence_and_reusability',
            'test_denormalization_accuracy'
        ],
        'AC3_FeatureEngineeringPlugins': [
            'test_plugin_discovery_and_loading',
            'test_plugin_pipeline_execution_with_data_chaining',
            'test_plugin_failure_isolation_and_recovery',
            'test_plugin_configuration_and_parameterization'
        ],
        'AC4_PostprocessingPlugins': [
            'test_postprocessing_pipeline_execution_order',
            'test_conditional_postprocessing_based_on_data_characteristics',
            'test_data_integrity_preservation_throughout_postprocessing'
        ],
        'AC5_HierarchicalConfiguration': [
            'test_configuration_hierarchy_and_inheritance',
            'test_comprehensive_configuration_validation',
            'test_configuration_migration_and_backward_compatibility',
            'test_configuration_error_reporting_and_guidance'
        ],
        'AC6_BackwardCompatibility': [
            'test_legacy_workflow_preservation',
            'test_api_contract_preservation',
            'test_data_format_compatibility',
            'test_migration_tools_and_documentation'
        ]
    }
    
    # Check if test files exist and contain expected tests
    test_files = [
        'test_acceptance_core.py',
        'test_acceptance_plugins.py', 
        'test_acceptance_config.py'
    ]
    
    coverage_results = {
        'total_acceptance_criteria': len(acceptance_criteria_coverage),
        'total_expected_tests': sum(len(tests) for tests in acceptance_criteria_coverage.values()),
        'covered_criteria': [],
        'missing_tests': []
    }
    
    test_dir = Path(__file__).parent
    
    for ac_name, expected_tests in acceptance_criteria_coverage.items():
        ac_covered = False
        for test_file in test_files:
            test_file_path = test_dir / test_file
            if test_file_path.exists():
                content = test_file_path.read_text()
                if any(test_name in content for test_name in expected_tests):
                    ac_covered = True
                    break
        
        if ac_covered:
            coverage_results['covered_criteria'].append(ac_name)
        else:
            coverage_results['missing_tests'].extend(expected_tests)
    
    coverage_percentage = (len(coverage_results['covered_criteria']) / 
                          coverage_results['total_acceptance_criteria']) * 100
    
    print(f"Acceptance Criteria Coverage: {coverage_percentage:.1f}%")
    print(f"Covered: {len(coverage_results['covered_criteria'])}/{coverage_results['total_acceptance_criteria']}")
    
    if coverage_results['missing_tests']:
        print(f"Missing tests: {len(coverage_results['missing_tests'])}")
        for test in coverage_results['missing_tests'][:5]:  # Show first 5
            print(f"  - {test}")
    
    coverage_results['coverage_percentage'] = coverage_percentage
    return coverage_results


if __name__ == '__main__':
    """Run acceptance test suite when executed directly."""
    
    # Run main acceptance tests
    main_results = run_acceptance_tests()
    
    # Run performance validation
    perf_results = run_performance_validation()
    
    # Validate test coverage
    coverage_results = validate_test_coverage()
    
    # Combine all results
    final_results = {
        **main_results,
        **perf_results,
        **coverage_results
    }
    
    # Save results to file
    results_file = Path(__file__).parent / 'acceptance_test_results.json'
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Exit with appropriate code
    exit_code = main_results['exit_code']
    if perf_results['performance_test_status'] == 'FAILED':
        exit_code = 1
    
    sys.exit(exit_code)
