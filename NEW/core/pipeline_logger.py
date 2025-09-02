#!/usr/bin/env python3
"""
Centralized Pipeline Logger - Comprehensive Logging System
Provides structured logging for the entire video synthesis pipeline
"""

import logging
import logging.handlers
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import traceback


class PipelineLogger:
    """Centralized logger for the video synthesis pipeline with structured output."""
    
    def __init__(self, base_dir: str = None, session_id: str = None):
        """Initialize the pipeline logger.
        
        Args:
            base_dir: Base directory for log files. Defaults to NEW/logs/
            session_id: Unique session identifier for this pipeline run
        """
        if base_dir is None:
            self.base_dir = Path(__file__).parent / "logs"
        else:
            self.base_dir = Path(base_dir)
        
        self.session_id = session_id or f"session_{int(time.time())}"
        self.start_time = time.time()
        
        # Create log directory structure
        self._setup_log_directories()
        
        # Initialize loggers
        self._setup_loggers()
        
        # Log session start
        self.info("STARTING Pipeline logging system initialized", {
            'session_id': self.session_id,
            'log_directory': str(self.base_dir),
            'start_time': datetime.now().isoformat()
        })
    
    def _setup_log_directories(self):
        """Create the organized log directory structure."""
        directories = [
            self.base_dir,
            self.base_dir / "pipeline",
            self.base_dir / "errors", 
            self.base_dir / "debug",
            self.base_dir / "performance",
            self.base_dir / "cli"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _setup_loggers(self):
        """Setup multiple specialized loggers."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Main pipeline logger
        self.main_logger = self._create_logger(
            'pipeline_main',
            self.base_dir / "pipeline" / f"main_pipeline_{timestamp}.log",
            level=logging.INFO
        )
        
        # CLI execution logger  
        self.cli_logger = self._create_logger(
            'pipeline_cli',
            self.base_dir / "cli" / f"cli_execution_{timestamp}.log",
            level=logging.DEBUG
        )
        
        # Error logger (all errors across pipeline)
        self.error_logger = self._create_logger(
            'pipeline_errors',
            self.base_dir / "errors" / f"pipeline_errors_{datetime.now().strftime('%Y%m%d')}.log",
            level=logging.WARNING
        )
        
        # Performance metrics logger
        self.perf_logger = self._create_logger(
            'pipeline_performance',
            self.base_dir / "performance" / f"stage_timings_{timestamp}.log",
            level=logging.INFO,
            format_string='%(asctime)s - PERFORMANCE - %(message)s'
        )
        
        # Debug loggers for specific components
        self.sadtalker_logger = self._create_logger(
            'sadtalker_debug',
            self.base_dir / "debug" / f"sadtalker_debug_{datetime.now().strftime('%Y%m%d')}.log",
            level=logging.DEBUG
        )
        
        self.enhancement_logger = self._create_logger(
            'enhancement_debug', 
            self.base_dir / "debug" / f"enhancement_debug_{datetime.now().strftime('%Y%m%d')}.log",
            level=logging.DEBUG
        )
        
        self.voice_logger = self._create_logger(
            'voice_debug',
            self.base_dir / "debug" / f"voice_processing_debug_{datetime.now().strftime('%Y%m%d')}.log",
            level=logging.DEBUG
        )
        
        # Critical failures logger
        self.critical_logger = self._create_logger(
            'critical_failures',
            self.base_dir / "errors" / f"critical_failures_{datetime.now().strftime('%Y%m%d')}.log",
            level=logging.CRITICAL
        )
    
    def _create_logger(self, name: str, log_file: Path, level: int = logging.INFO, 
                      format_string: str = None) -> logging.Logger:
        """Create a configured logger with file and console output."""
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Default format
        if format_string is None:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        formatter = logging.Formatter(format_string)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=50*1024*1024, backupCount=5  # 50MB files, keep 5 backups
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler for important messages
        if level <= logging.WARNING:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
            logger.addHandler(console_handler)
        
        return logger
    
    def _format_message(self, message: str, context: Dict[str, Any] = None) -> str:
        """Format log message with optional context data."""
        if context:
            return f"{message} | Context: {json.dumps(context, default=str)}"
        return message
    
    # Main logging methods
    def info(self, message: str, context: Dict[str, Any] = None):
        """Log informational message."""
        formatted = self._format_message(message, context)
        self.main_logger.info(formatted)
    
    def debug(self, message: str, context: Dict[str, Any] = None):
        """Log debug message."""
        formatted = self._format_message(message, context)
        self.main_logger.debug(formatted)
    
    def warning(self, message: str, context: Dict[str, Any] = None):
        """Log warning message."""
        formatted = self._format_message(message, context)
        self.main_logger.warning(formatted)
        self.error_logger.warning(formatted)
    
    def error(self, message: str, context: Dict[str, Any] = None, exception: Exception = None):
        """Log error message with optional exception details."""
        formatted = self._format_message(message, context)
        if exception:
            formatted += f" | Exception: {str(exception)} | Traceback: {traceback.format_exc()}"
        
        self.main_logger.error(formatted)
        self.error_logger.error(formatted)
    
    def critical(self, message: str, context: Dict[str, Any] = None, exception: Exception = None):
        """Log critical failure."""
        formatted = self._format_message(message, context)
        if exception:
            formatted += f" | Exception: {str(exception)} | Traceback: {traceback.format_exc()}"
        
        self.main_logger.critical(formatted)
        self.error_logger.critical(formatted)
        self.critical_logger.critical(formatted)
    
    # CLI-specific logging
    def cli_info(self, message: str, context: Dict[str, Any] = None):
        """Log CLI-specific information."""
        formatted = self._format_message(message, context)
        self.cli_logger.info(formatted)
        print(f"Step {message}")
    
    def cli_debug(self, message: str, context: Dict[str, Any] = None):
        """Log CLI debug information."""
        formatted = self._format_message(message, context)
        self.cli_logger.debug(formatted)
    
    def cli_error(self, message: str, context: Dict[str, Any] = None, exception: Exception = None):
        """Log CLI error."""
        formatted = self._format_message(message, context)
        if exception:
            formatted += f" | Exception: {str(exception)}"
        
        self.cli_logger.error(formatted)
        self.error_logger.error(f"CLI_ERROR: {formatted}")
        print(f"[ERROR] {message}")
    
    # Performance logging
    def log_stage_start(self, stage_name: str, context: Dict[str, Any] = None):
        """Log the start of a pipeline stage."""
        stage_context = {
            'stage': stage_name,
            'start_time': time.time(),
            'session_id': self.session_id
        }
        if context:
            stage_context.update(context)
        
        message = f"[EMOJI] Starting stage: {stage_name}"
        self.info(message, stage_context)
        self.perf_logger.info(self._format_message(message, stage_context))
        
        return stage_context['start_time']
    
    def log_stage_end(self, stage_name: str, start_time: float, success: bool = True, 
                     context: Dict[str, Any] = None):
        """Log the end of a pipeline stage with timing."""
        end_time = time.time()
        duration = end_time - start_time
        
        stage_context = {
            'stage': stage_name,
            'duration_seconds': round(duration, 2),
            'success': success,
            'session_id': self.session_id
        }
        if context:
            stage_context.update(context)
        
        status = "[SUCCESS] Completed" if success else "[ERROR] Failed"
        message = f"{status} stage: {stage_name} (Duration: {duration:.2f}s)"
        
        if success:
            self.info(message, stage_context)
        else:
            self.error(message, stage_context)
        
        self.perf_logger.info(self._format_message(message, stage_context))
        
        return duration
    
    # Component-specific logging
    def sadtalker_debug(self, message: str, context: Dict[str, Any] = None):
        """Log SadTalker-specific debug information."""
        formatted = self._format_message(f"SADTALKER: {message}", context)
        self.sadtalker_logger.debug(formatted)
        self.debug(f"Style: SadTalker: {message}", context)
    
    def sadtalker_error(self, message: str, context: Dict[str, Any] = None, exception: Exception = None):
        """Log SadTalker-specific errors."""
        formatted = self._format_message(f"SADTALKER_ERROR: {message}", context)
        if exception:
            formatted += f" | Exception: {str(exception)}"
        
        self.sadtalker_logger.error(formatted)
        self.error(f"Style: SadTalker Error: {message}", context, exception)
    
    def enhancement_debug(self, message: str, context: Dict[str, Any] = None):
        """Log Enhancement-specific debug information."""
        formatted = self._format_message(f"ENHANCEMENT: {message}", context)
        self.enhancement_logger.debug(formatted)
        self.debug(f"Enhanced Enhancement: {message}", context)
    
    def enhancement_error(self, message: str, context: Dict[str, Any] = None, exception: Exception = None):
        """Log Enhancement-specific errors."""
        formatted = self._format_message(f"ENHANCEMENT_ERROR: {message}", context)
        if exception:
            formatted += f" | Exception: {str(exception)}"
        
        self.enhancement_logger.error(formatted)
        self.error(f"Enhanced Enhancement Error: {message}", context, exception)
    
    def voice_debug(self, message: str, context: Dict[str, Any] = None):
        """Log Voice processing debug information."""
        formatted = self._format_message(f"VOICE: {message}", context)
        self.voice_logger.debug(formatted)
        self.debug(f"Recording Voice: {message}", context)
    
    def voice_error(self, message: str, context: Dict[str, Any] = None, exception: Exception = None):
        """Log Voice processing errors."""
        formatted = self._format_message(f"VOICE_ERROR: {message}", context)
        if exception:
            formatted += f" | Exception: {str(exception)}"
        
        self.voice_logger.error(formatted)
        self.error(f"Recording Voice Error: {message}", context, exception)
    
    # File operation logging
    def log_file_operation(self, operation: str, file_path: str, success: bool = True, 
                          context: Dict[str, Any] = None):
        """Log file operations with details."""
        file_context = {
            'operation': operation,
            'file_path': file_path,
            'success': success
        }
        
        # Add file size if it exists
        if Path(file_path).exists():
            file_context['file_size_bytes'] = Path(file_path).stat().st_size
        
        if context:
            file_context.update(context)
        
        message = f"Assets: File {operation}: {Path(file_path).name}"
        
        if success:
            self.info(message, file_context)
        else:
            self.error(message, file_context)
    
    # Subprocess logging
    def log_subprocess(self, command: list, success: bool, output: str = "", 
                      error: str = "", context: Dict[str, Any] = None):
        """Log subprocess execution details."""
        subprocess_context = {
            'command': ' '.join(command) if isinstance(command, list) else str(command),
            'success': success,
            'output_length': len(output),
            'error_length': len(error)
        }
        if context:
            subprocess_context.update(context)
        
        message = f"Configuration Subprocess: {command[0] if isinstance(command, list) else 'unknown'}"
        
        if success:
            self.info(message, subprocess_context)
        else:
            subprocess_context['stderr'] = error[:1000]  # First 1000 chars of error
            self.error(message, subprocess_context)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current logging session."""
        return {
            'session_id': self.session_id,
            'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
            'duration_seconds': round(time.time() - self.start_time, 2),
            'log_directory': str(self.base_dir),
            'log_files': [
                str(handler.baseFilename) for logger in [
                    self.main_logger, self.cli_logger, self.error_logger, 
                    self.perf_logger, self.sadtalker_logger, self.enhancement_logger,
                    self.voice_logger, self.critical_logger
                ] for handler in logger.handlers if hasattr(handler, 'baseFilename')
            ]
        }
    
    def close(self):
        """Close all logging handlers and log session summary."""
        summary = self.get_session_summary()
        self.info("[EMOJI] Pipeline logging session ended", summary)
        
        # Close all handlers
        for logger in [self.main_logger, self.cli_logger, self.error_logger, 
                      self.perf_logger, self.sadtalker_logger, self.enhancement_logger,
                      self.voice_logger, self.critical_logger]:
            for handler in logger.handlers:
                handler.close()


# Global pipeline logger instance
_pipeline_logger: Optional[PipelineLogger] = None

def get_pipeline_logger(base_dir: str = None, session_id: str = None) -> PipelineLogger:
    """Get or create the global pipeline logger instance."""
    global _pipeline_logger
    if _pipeline_logger is None:
        _pipeline_logger = PipelineLogger(base_dir, session_id)
    return _pipeline_logger

def close_pipeline_logger():
    """Close the global pipeline logger."""
    global _pipeline_logger
    if _pipeline_logger is not None:
        _pipeline_logger.close()
        _pipeline_logger = None


if __name__ == "__main__":
    # Test the logging system
    logger = PipelineLogger()
    
    logger.info("Testing pipeline logger")
    logger.cli_info("Testing CLI logging")
    logger.sadtalker_debug("Testing SadTalker debug", {'test': True})
    logger.enhancement_debug("Testing Enhancement debug", {'frames': 100})
    logger.voice_debug("Testing Voice debug", {'duration': 10.5})
    
    start_time = logger.log_stage_start("test_stage", {'input': 'test.mp4'})
    time.sleep(1)
    logger.log_stage_end("test_stage", start_time, True, {'output': 'test_out.mp4'})
    
    logger.log_file_operation("read", "/test/file.mp4", True, {'purpose': 'testing'})
    logger.log_subprocess(['ffmpeg', '-i', 'input.mp4'], True, "success", "")
    
    print("\nStatus: Session Summary:")
    print(json.dumps(logger.get_session_summary(), indent=2))
    
    logger.close()