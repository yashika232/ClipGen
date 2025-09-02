#!/usr/bin/env python3
"""
Pipeline Logger - Centralized Logging Infrastructure
Provides structured logging for the video synthesis pipeline with session correlation,
performance tracking, and comprehensive error handling.
"""

import os
import sys
import json
import time
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import traceback
import psutil
import functools


class LogLevel(Enum):
    """Standard log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogComponent(Enum):
    """Pipeline components for logging."""
    API_SERVER = "api_server"
    GEMINI_GENERATOR = "gemini_generator"
    THUMBNAIL_GENERATOR = "thumbnail_generator"
    SESSION_MANAGER = "session_manager"
    PROGRESS_MANAGER = "progress_manager"
    FILE_VALIDATOR = "file_validator"
    PIPELINE_STAGE = "pipeline_stage"
    FRONTEND = "frontend"
    SYSTEM = "system"


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: str
    level: str
    session_id: Optional[str]
    component: str
    event: str
    message: str
    metadata: Dict[str, Any]
    execution_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    error_details: Optional[Dict[str, Any]] = None
    correlation_id: Optional[str] = None


class PipelineLogger:
    """Centralized logging system for the video synthesis pipeline."""
    
    def __init__(self, log_dir: str = "logs", max_file_size: int = 100*1024*1024):
        """Initialize the pipeline logger.
        
        Args:
            log_dir: Directory for log files
            max_file_size: Maximum size per log file in bytes (default: 100MB)
        """
        self.log_dir = Path(log_dir)
        self.max_file_size = max_file_size
        self.session_context = threading.local()
        self._setup_directories()
        self._setup_loggers()
        
    def _setup_directories(self):
        """Create log directory structure."""
        subdirs = ['pipeline', 'frontend', 'system', 'errors']
        for subdir in subdirs:
            (self.log_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    def _setup_loggers(self):
        """Setup logging handlers and formatters."""
        # Remove existing handlers
        logging.getLogger().handlers = []
        
        # Create formatters
        json_formatter = logging.Formatter('')
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup main logger
        self.logger = logging.getLogger('pipeline')
        self.logger.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handlers will be created dynamically
        self._file_handlers = {}
        
    def set_session_context(self, session_id: str, user_id: str = None, 
                           correlation_id: str = None):
        """Set session context for current thread."""
        self.session_context.session_id = session_id
        self.session_context.user_id = user_id
        self.session_context.correlation_id = correlation_id or str(uuid.uuid4())
        
    def get_session_context(self) -> Dict[str, Any]:
        """Get current session context."""
        return {
            'session_id': getattr(self.session_context, 'session_id', None),
            'user_id': getattr(self.session_context, 'user_id', None),
            'correlation_id': getattr(self.session_context, 'correlation_id', None)
        }
    
    def _get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics."""
        try:
            process = psutil.Process()
            return {
                'memory_usage_mb': process.memory_info().rss / 1024 / 1024,
                'cpu_percent': process.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent
            }
        except Exception:
            return {}
    
    def _get_file_handler(self, component: LogComponent) -> logging.FileHandler:
        """Get or create file handler for component."""
        date_str = datetime.now().strftime('%Y-%m-%d')
        
        # Determine log file path
        if component in [LogComponent.API_SERVER, LogComponent.SESSION_MANAGER, 
                        LogComponent.PROGRESS_MANAGER, LogComponent.FILE_VALIDATOR]:
            log_file = self.log_dir / 'pipeline' / f'{component.value}_{date_str}.log'
        elif component == LogComponent.FRONTEND:
            log_file = self.log_dir / 'frontend' / f'frontend_{date_str}.log'
        elif component == LogComponent.SYSTEM:
            log_file = self.log_dir / 'system' / f'system_{date_str}.log'
        else:
            log_file = self.log_dir / 'pipeline' / f'{component.value}_{date_str}.log'
        
        handler_key = str(log_file)
        
        if handler_key not in self._file_handlers:
            handler = logging.FileHandler(log_file)
            handler.setLevel(logging.DEBUG)
            self._file_handlers[handler_key] = handler
            
        return self._file_handlers[handler_key]
    
    def _filter_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter sensitive information from log data."""
        sensitive_keys = [
            'password', 'token', 'api_key', 'secret', 'private_key',
            'authorization', 'cookie', 'session_token'
        ]
        
        filtered = {}
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                filtered[key] = "***REDACTED***"
            elif isinstance(value, dict):
                filtered[key] = self._filter_sensitive_data(value)
            else:
                filtered[key] = value
                
        return filtered
    
    def log(self, level: LogLevel, component: LogComponent, event: str, 
            message: str, metadata: Dict[str, Any] = None, 
            execution_time_ms: float = None, error: Exception = None):
        """Log a structured entry."""
        
        # Get session context
        context = self.get_session_context()
        
        # Prepare metadata
        safe_metadata = self._filter_sensitive_data(metadata or {})
        
        # Get system metrics
        system_metrics = self._get_system_metrics()
        
        # Create log entry
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level.value,
            session_id=context['session_id'],
            component=component.value,
            event=event,
            message=message,
            metadata=safe_metadata,
            execution_time_ms=execution_time_ms,
            memory_usage_mb=system_metrics.get('memory_usage_mb'),
            correlation_id=context['correlation_id']
        )
        
        # Add error details if provided
        if error:
            entry.error_details = {
                'error_type': type(error).__name__,
                'error_message': str(error),
                'stack_trace': traceback.format_exc()
            }
        
        # Convert to JSON
        log_json = json.dumps(asdict(entry), indent=None)
        
        # Write to file
        try:
            file_handler = self._get_file_handler(component)
            file_handler.emit(logging.LogRecord(
                name=f'pipeline.{component.value}',
                level=getattr(logging, level.value),
                pathname='',
                lineno=0,
                msg=log_json,
                args=(),
                exc_info=None
            ))
        except Exception as e:
            # Fallback to console if file logging fails
            print(f"Failed to write log: {e}")
            print(log_json)
        
        # Also log to console for errors and above
        if level.value in ['ERROR', 'CRITICAL']:
            self.logger.error(f"[{component.value}] {event}: {message}")
            if error:
                self.logger.error(f"Error details: {error}")
    
    def debug(self, component: LogComponent, event: str, message: str, 
              metadata: Dict[str, Any] = None, execution_time_ms: float = None):
        """Log debug message."""
        self.log(LogLevel.DEBUG, component, event, message, metadata, execution_time_ms)
    
    def info(self, component: LogComponent, event: str, message: str, 
             metadata: Dict[str, Any] = None, execution_time_ms: float = None):
        """Log info message."""
        self.log(LogLevel.INFO, component, event, message, metadata, execution_time_ms)
    
    def warning(self, component: LogComponent, event: str, message: str, 
                metadata: Dict[str, Any] = None, execution_time_ms: float = None):
        """Log warning message."""
        self.log(LogLevel.WARNING, component, event, message, metadata, execution_time_ms)
    
    def error(self, component: LogComponent, event: str, message: str, 
              metadata: Dict[str, Any] = None, execution_time_ms: float = None, 
              error: Exception = None):
        """Log error message."""
        self.log(LogLevel.ERROR, component, event, message, metadata, execution_time_ms, error)
    
    def critical(self, component: LogComponent, event: str, message: str, 
                 metadata: Dict[str, Any] = None, execution_time_ms: float = None, 
                 error: Exception = None):
        """Log critical message."""
        self.log(LogLevel.CRITICAL, component, event, message, metadata, execution_time_ms, error)
    
    def log_api_request(self, method: str, endpoint: str, status_code: int, 
                       execution_time_ms: float, request_data: Dict[str, Any] = None):
        """Log API request."""
        self.info(
            LogComponent.API_SERVER,
            "api_request",
            f"{method} {endpoint} - {status_code}",
            metadata={
                'method': method,
                'endpoint': endpoint,
                'status_code': status_code,
                'request_data': request_data
            },
            execution_time_ms=execution_time_ms
        )
    
    def log_ai_api_call(self, service: str, operation: str, success: bool, 
                       execution_time_ms: float, cost: float = None, 
                       tokens_used: int = None, error: Exception = None):
        """Log AI API call."""
        level = LogLevel.INFO if success else LogLevel.ERROR
        self.log(
            level,
            LogComponent.GEMINI_GENERATOR if 'gemini' in service.lower() else LogComponent.THUMBNAIL_GENERATOR,
            "ai_api_call",
            f"{service} {operation} - {'Success' if success else 'Failed'}",
            metadata={
                'service': service,
                'operation': operation,
                'success': success,
                'cost': cost,
                'tokens_used': tokens_used
            },
            execution_time_ms=execution_time_ms,
            error=error
        )
    
    def log_pipeline_stage(self, stage_name: str, status: str, 
                          execution_time_ms: float = None, 
                          input_data: Dict[str, Any] = None, 
                          output_data: Dict[str, Any] = None, 
                          error: Exception = None):
        """Log pipeline stage execution."""
        level = LogLevel.INFO if status == 'completed' else LogLevel.ERROR
        self.log(
            level,
            LogComponent.PIPELINE_STAGE,
            "stage_execution",
            f"Stage {stage_name} - {status}",
            metadata={
                'stage_name': stage_name,
                'status': status,
                'input_data': input_data,
                'output_data': output_data
            },
            execution_time_ms=execution_time_ms,
            error=error
        )
    
    def log_session_event(self, event_type: str, details: Dict[str, Any] = None):
        """Log session-related events."""
        self.info(
            LogComponent.SESSION_MANAGER,
            "session_event",
            f"Session {event_type}",
            metadata={
                'event_type': event_type,
                'details': details
            }
        )
    
    def log_file_operation(self, operation: str, filename: str, file_size: int = None, 
                          success: bool = True, error: Exception = None):
        """Log file operations."""
        level = LogLevel.INFO if success else LogLevel.ERROR
        self.log(
            level,
            LogComponent.FILE_VALIDATOR,
            "file_operation",
            f"File {operation}: {filename}",
            metadata={
                'operation': operation,
                'filename': filename,
                'file_size': file_size,
                'success': success
            },
            error=error
        )
    
    def performance_monitor(self, component: LogComponent, operation: str):
        """Decorator for performance monitoring."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    execution_time = (time.time() - start_time) * 1000
                    
                    self.info(
                        component,
                        "performance_monitor",
                        f"{operation} completed successfully",
                        metadata={
                            'operation': operation,
                            'function': func.__name__,
                            'success': True
                        },
                        execution_time_ms=execution_time
                    )
                    return result
                    
                except Exception as e:
                    execution_time = (time.time() - start_time) * 1000
                    self.error(
                        component,
                        "performance_monitor",
                        f"{operation} failed",
                        metadata={
                            'operation': operation,
                            'function': func.__name__,
                            'success': False
                        },
                        execution_time_ms=execution_time,
                        error=e
                    )
                    raise
            return wrapper
        return decorator
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old log files."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        for log_file in self.log_dir.rglob('*.log'):
            if log_file.stat().st_mtime < cutoff_date.timestamp():
                try:
                    log_file.unlink()
                    self.info(
                        LogComponent.SYSTEM,
                        "log_cleanup",
                        f"Deleted old log file: {log_file.name}"
                    )
                except Exception as e:
                    self.error(
                        LogComponent.SYSTEM,
                        "log_cleanup",
                        f"Failed to delete log file: {log_file.name}",
                        error=e
                    )


# Global logger instance
pipeline_logger = PipelineLogger()


def get_logger() -> PipelineLogger:
    """Get the global pipeline logger instance."""
    return pipeline_logger


def set_session_context(session_id: str, user_id: str = None, correlation_id: str = None):
    """Set session context for the current thread."""
    pipeline_logger.set_session_context(session_id, user_id, correlation_id)


def log_api_request(method: str, endpoint: str, status_code: int, 
                   execution_time_ms: float, request_data: Dict[str, Any] = None):
    """Convenience function for API request logging."""
    pipeline_logger.log_api_request(method, endpoint, status_code, execution_time_ms, request_data)


def log_ai_api_call(service: str, operation: str, success: bool, 
                   execution_time_ms: float, cost: float = None, 
                   tokens_used: int = None, error: Exception = None):
    """Convenience function for AI API call logging."""
    pipeline_logger.log_ai_api_call(service, operation, success, execution_time_ms, cost, tokens_used, error)


def performance_monitor(component: LogComponent, operation: str):
    """Convenience decorator for performance monitoring."""
    return pipeline_logger.performance_monitor(component, operation)


if __name__ == "__main__":
    # Test the logger
    logger = get_logger()
    
    # Set session context
    set_session_context("test_session_123", "user_456")
    
    # Test different log levels
    logger.info(LogComponent.API_SERVER, "test_event", "Testing pipeline logger")
    logger.error(LogComponent.GEMINI_GENERATOR, "test_error", "Test error message", 
                error=Exception("Test exception"))
    
    # Test performance monitoring
    @performance_monitor(LogComponent.PIPELINE_STAGE, "test_operation")
    def test_function():
        time.sleep(0.1)
        return "success"
    
    result = test_function()
    print(f"Test completed: {result}")