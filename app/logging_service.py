"""
Logging Service for Preprocessor System

This module provides structured logging with configurable levels,
formatters, and handlers for comprehensive system monitoring.
"""

import logging
import logging.handlers
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class LoggingService:
    """
    Centralized logging service providing structured logging with
    configurable levels, formatters, and handlers.
    
    Behavioral Requirements:
    - BR-LOG-001: Provide structured logging with consistent format
    - BR-LOG-002: Support multiple output destinations
    - BR-LOG-003: Enable log level configuration
    - BR-LOG-004: Maintain audit trails for critical operations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.loggers: Dict[str, logging.Logger] = {}
        self._setup_logging()
    
    def _setup_logging(self):
        """Initialize logging configuration."""
        # Create logs directory if it doesn't exist
        log_dir = Path(self.config.get('log_directory', 'logs'))
        log_dir.mkdir(exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger('preprocessor')
        root_logger.setLevel(getattr(logging, self.config.get('log_level', 'INFO')))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Add console handler
        if self.config.get('console_logging', True):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(self._get_formatter('console'))
            root_logger.addHandler(console_handler)
        
        # Add file handler
        if self.config.get('file_logging', True):
            log_file = log_dir / self.config.get('log_file', 'preprocessor.log')
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.config.get('max_log_size', 10 * 1024 * 1024),  # 10MB
                backupCount=self.config.get('backup_count', 5)
            )
            file_handler.setFormatter(self._get_formatter('file'))
            root_logger.addHandler(file_handler)
        
        # Add JSON handler for structured logging
        if self.config.get('structured_logging', False):
            json_file = log_dir / self.config.get('json_log_file', 'preprocessor.json')
            json_handler = logging.FileHandler(json_file)
            json_handler.setFormatter(self._get_json_formatter())
            root_logger.addHandler(json_handler)
        
        self.loggers['preprocessor'] = root_logger
    
    def _get_formatter(self, handler_type: str) -> logging.Formatter:
        """Get formatter based on handler type."""
        if handler_type == 'console':
            return logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        elif handler_type == 'file':
            return logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            return logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    def _get_json_formatter(self) -> logging.Formatter:
        """Get JSON formatter for structured logging."""
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno
                }
                
                # Add exception info if present
                if record.exc_info:
                    log_entry['exception'] = self.formatException(record.exc_info)
                
                # Add extra fields
                for key, value in record.__dict__.items():
                    if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                                  'filename', 'module', 'lineno', 'funcName', 'created',
                                  'msecs', 'relativeCreated', 'thread', 'threadName',
                                  'processName', 'process', 'getMessage', 'exc_info',
                                  'exc_text', 'stack_info']:
                        log_entry[key] = value
                
                return json.dumps(log_entry)
        
        return JsonFormatter()
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger with the specified name."""
        if name not in self.loggers:
            logger = logging.getLogger(f'preprocessor.{name}')
            logger.setLevel(getattr(logging, self.config.get('log_level', 'INFO')))
            self.loggers[name] = logger
        
        return self.loggers[name]
    
    def log_operation(self, operation: str, details: Dict[str, Any], 
                     level: str = 'INFO', logger_name: str = 'preprocessor'):
        """Log an operation with structured details."""
        logger = self.get_logger(logger_name)
        log_level = getattr(logging, level.upper())
        
        # Create structured log entry
        log_message = f"Operation: {operation}"
        
        # Log with extra context
        logger.log(log_level, log_message, extra={
            'operation': operation,
            'details': details,
            'component': logger_name
        })
    
    def log_performance(self, operation: str, duration: float, 
                       metadata: Optional[Dict[str, Any]] = None):
        """Log performance metrics for operations."""
        logger = self.get_logger('performance')
        
        performance_data = {
            'operation': operation,
            'duration_seconds': duration,
            'metadata': metadata or {}
        }
        
        logger.info(f"Performance: {operation} completed in {duration:.3f}s", 
                   extra=performance_data)
    
    def log_audit(self, action: str, user: str, resource: str, 
                 result: str, details: Optional[Dict[str, Any]] = None):
        """Log audit trail for security-sensitive operations."""
        logger = self.get_logger('audit')
        
        audit_data = {
            'action': action,
            'user': user,
            'resource': resource,
            'result': result,
            'details': details or {},
            'audit_timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Audit: {action} by {user} on {resource} - {result}", 
                   extra=audit_data)
    
    def configure_log_level(self, level: str, logger_name: Optional[str] = None):
        """Configure log level for a specific logger or all loggers."""
        log_level = getattr(logging, level.upper())
        
        if logger_name:
            if logger_name in self.loggers:
                self.loggers[logger_name].setLevel(log_level)
        else:
            for logger in self.loggers.values():
                logger.setLevel(log_level)
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get logging statistics and configuration."""
        return {
            'active_loggers': list(self.loggers.keys()),
            'log_level': self.config.get('log_level', 'INFO'),
            'console_logging': self.config.get('console_logging', True),
            'file_logging': self.config.get('file_logging', True),
            'structured_logging': self.config.get('structured_logging', False)
        }
