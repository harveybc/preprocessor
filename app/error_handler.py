"""
Error Handling Service for Preprocessor System

This module provides centralized error handling with categorization,
recovery strategies, and comprehensive logging.
"""

import logging
import traceback
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


class ErrorCategory(Enum):
    """Error categories for classification and handling."""
    CONFIGURATION = "configuration"
    DATA_VALIDATION = "data_validation"
    PLUGIN_EXECUTION = "plugin_execution"
    NORMALIZATION = "normalization"
    FILE_IO = "file_io"
    EXTERNAL_SERVICE = "external_service"
    SYSTEM = "system"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for errors."""
    component: str
    operation: str
    parameters: Dict[str, Any]
    timestamp: str
    traceback: str


class ProcessorError(Exception):
    """Base exception for preprocessor system."""
    
    def __init__(self, message: str, category: ErrorCategory, 
                 severity: ErrorSeverity, context: Optional[ErrorContext] = None):
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.context = context


class ErrorHandler:
    """
    Centralized error handling service providing categorization,
    logging, and recovery strategies.
    
    Behavioral Requirements:
    - BR-ERROR-001: Categorize errors by type and severity
    - BR-ERROR-002: Provide context-aware error messages
    - BR-ERROR-003: Implement recovery strategies
    - BR-ERROR-004: Maintain error audit trails
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_counts: Dict[ErrorCategory, int] = {}
        self.recovery_strategies: Dict[ErrorCategory, callable] = {}
        
    def handle_error(self, error: Exception, category: ErrorCategory, 
                    severity: ErrorSeverity, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle an error with categorization, logging, and recovery.
        
        Args:
            error: The exception that occurred
            category: Error category for classification
            severity: Error severity level
            context: Additional context information
            
        Returns:
            Dict containing error information and recovery actions
        """
        error_context = ErrorContext(
            component=context.get('component', 'unknown') if context else 'unknown',
            operation=context.get('operation', 'unknown') if context else 'unknown',
            parameters=context.get('parameters', {}) if context else {},
            timestamp=context.get('timestamp', '') if context else '',
            traceback=traceback.format_exc()
        )
        
        # Create structured error
        processor_error = ProcessorError(str(error), category, severity, error_context)
        
        # Update error counts
        self.error_counts[category] = self.error_counts.get(category, 0) + 1
        
        # Log error with appropriate level
        self._log_error(processor_error)
        
        # Attempt recovery
        recovery_result = self._attempt_recovery(processor_error)
        
        return {
            'error_id': id(processor_error),
            'category': category.value,
            'severity': severity.value,
            'message': str(error),
            'context': error_context,
            'recovery_attempted': recovery_result['attempted'],
            'recovery_successful': recovery_result['successful'],
            'recovery_actions': recovery_result['actions']
        }
    
    def _log_error(self, error: ProcessorError):
        """Log error with appropriate level based on severity."""
        log_message = f"[{error.category.value.upper()}] {error}"
        
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def _attempt_recovery(self, error: ProcessorError) -> Dict[str, Any]:
        """Attempt error recovery based on category."""
        recovery_result = {
            'attempted': False,
            'successful': False,
            'actions': []
        }
        
        if error.category in self.recovery_strategies:
            try:
                recovery_result['attempted'] = True
                strategy = self.recovery_strategies[error.category]
                actions = strategy(error)
                recovery_result['actions'] = actions
                recovery_result['successful'] = True
            except Exception as recovery_error:
                self.logger.error(f"Recovery failed: {recovery_error}")
        
        return recovery_result
    
    def register_recovery_strategy(self, category: ErrorCategory, strategy: callable):
        """Register a recovery strategy for an error category."""
        self.recovery_strategies[category] = strategy
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and trends."""
        return {
            'total_errors': sum(self.error_counts.values()),
            'errors_by_category': dict(self.error_counts),
            'categories_with_recovery': list(self.recovery_strategies.keys())
        }
