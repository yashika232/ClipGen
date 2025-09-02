#!/usr/bin/env python3
"""
Pipeline Exception Hierarchy - Comprehensive Error Handling
Custom exceptions for structured error handling throughout the video synthesis pipeline
Designed for frontend integration with specific error codes and user-friendly messages
"""

from typing import Dict, Any, Optional


class PipelineException(Exception):
    """Base exception for all pipeline-related errors."""
    
    def __init__(self, message: str, error_code: str = "PIPELINE_ERROR", 
                 details: Optional[Dict[str, Any]] = None, user_message: Optional[str] = None):
        """Initialize pipeline exception.
        
        Args:
            message: Technical error message for logging
            error_code: Unique error code for frontend handling
            details: Additional error details and context
            user_message: User-friendly message for frontend display
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.user_message = user_message or message
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            'error': True,
            'error_code': self.error_code,
            'message': self.message,
            'user_message': self.user_message,
            'details': self.details
        }


# === Session Management Exceptions ===

class SessionException(PipelineException):
    """Base class for session-related errors."""
    pass


class SessionNotFoundError(SessionException):
    """Raised when a session ID cannot be found."""
    
    def __init__(self, session_id: str):
        super().__init__(
            message=f"Session {session_id} not found",
            error_code="SESSION_NOT_FOUND",
            details={'session_id': session_id},
            user_message="Session not found. Please start a new session."
        )


class SessionExpiredError(SessionException):
    """Raised when a session has expired."""
    
    def __init__(self, session_id: str, expiry_time: str):
        super().__init__(
            message=f"Session {session_id} expired at {expiry_time}",
            error_code="SESSION_EXPIRED",
            details={'session_id': session_id, 'expiry_time': expiry_time},
            user_message="Your session has expired. Please start a new session."
        )


class SessionConcurrencyError(SessionException):
    """Raised when session has concurrent access conflicts."""
    
    def __init__(self, session_id: str, operation: str):
        super().__init__(
            message=f"Session {session_id} is already processing operation: {operation}",
            error_code="SESSION_CONCURRENT_ACCESS",
            details={'session_id': session_id, 'operation': operation},
            user_message="Another operation is already in progress. Please wait and try again."
        )


class SessionStateError(SessionException):
    """Raised when session is in invalid state for requested operation."""
    
    def __init__(self, session_id: str, current_state: str, required_state: str):
        super().__init__(
            message=f"Session {session_id} is in state '{current_state}', required: '{required_state}'",
            error_code="INVALID_SESSION_STATE",
            details={'session_id': session_id, 'current_state': current_state, 'required_state': required_state},
            user_message=f"Operation not allowed in current session state. Current: {current_state}"
        )


# === User Input Validation Exceptions ===

class ValidationException(PipelineException):
    """Base class for input validation errors."""
    pass


class RequiredFieldError(ValidationException):
    """Raised when required fields are missing."""
    
    def __init__(self, missing_fields: list):
        super().__init__(
            message=f"Missing required fields: {missing_fields}",
            error_code="MISSING_REQUIRED_FIELDS",
            details={'missing_fields': missing_fields},
            user_message=f"Please provide the following required fields: {', '.join(missing_fields)}"
        )


class InvalidFieldValueError(ValidationException):
    """Raised when field values are invalid."""
    
    def __init__(self, field: str, value: Any, allowed_values: list = None):
        details = {'field': field, 'value': value}
        if allowed_values:
            details['allowed_values'] = allowed_values
            user_msg = f"Invalid value for {field}. Allowed values: {', '.join(allowed_values)}"
        else:
            user_msg = f"Invalid value for {field}: {value}"
            
        super().__init__(
            message=f"Invalid value for field '{field}': {value}",
            error_code="INVALID_FIELD_VALUE",
            details=details,
            user_message=user_msg
        )


class FileValidationError(ValidationException):
    """Raised when uploaded files fail validation."""
    
    def __init__(self, filename: str, reason: str, max_size: Optional[int] = None):
        details = {'filename': filename, 'reason': reason}
        if max_size:
            details['max_size_mb'] = max_size // (1024 * 1024)
            
        super().__init__(
            message=f"File validation failed for '{filename}': {reason}",
            error_code="FILE_VALIDATION_ERROR",
            details=details,
            user_message=f"File '{filename}' is invalid: {reason}"
        )


# === AI Processing Exceptions ===

class AIProcessingException(PipelineException):
    """Base class for AI processing errors."""
    pass


class GeminiAPIError(AIProcessingException):
    """Raised when Gemini API calls fail."""
    
    def __init__(self, operation: str, api_error: str):
        super().__init__(
            message=f"Gemini API error during {operation}: {api_error}",
            error_code="GEMINI_API_ERROR",
            details={'operation': operation, 'api_error': api_error},
            user_message="Content generation service is temporarily unavailable. Please try again."
        )


class ContentGenerationError(AIProcessingException):
    """Raised when content generation fails."""
    
    def __init__(self, content_type: str, reason: str):
        super().__init__(
            message=f"Content generation failed for {content_type}: {reason}",
            error_code="CONTENT_GENERATION_ERROR",
            details={'content_type': content_type, 'reason': reason},
            user_message=f"Failed to generate {content_type}. Please check your inputs and try again."
        )


# === Pipeline Stage Exceptions ===

class PipelineStageException(PipelineException):
    """Base class for pipeline stage processing errors."""
    pass


class VoiceCloningError(PipelineStageException):
    """Raised when voice cloning fails."""
    
    def __init__(self, reason: str, voice_file: Optional[str] = None):
        details = {'reason': reason}
        if voice_file:
            details['voice_file'] = voice_file
            
        super().__init__(
            message=f"Voice cloning failed: {reason}",
            error_code="VOICE_CLONING_ERROR",
            details=details,
            user_message="Voice synthesis failed. Please check your voice sample and try again."
        )


class FaceProcessingError(PipelineStageException):
    """Raised when face processing fails."""
    
    def __init__(self, reason: str, image_file: Optional[str] = None):
        details = {'reason': reason}
        if image_file:
            details['image_file'] = image_file
            
        super().__init__(
            message=f"Face processing failed: {reason}",
            error_code="FACE_PROCESSING_ERROR",
            details=details,
            user_message="Face processing failed. Please ensure your image contains a clear face."
        )


class VideoGenerationError(PipelineStageException):
    """Raised when video generation fails."""
    
    def __init__(self, reason: str, stage: str = "unknown"):
        super().__init__(
            message=f"Video generation failed at {stage}: {reason}",
            error_code="VIDEO_GENERATION_ERROR",
            details={'reason': reason, 'stage': stage},
            user_message="Video generation failed. Please try again or contact support."
        )


class VideoEnhancementError(PipelineStageException):
    """Raised when video enhancement fails."""
    
    def __init__(self, reason: str, enhancement_type: str = "unknown"):
        super().__init__(
            message=f"Video enhancement failed ({enhancement_type}): {reason}",
            error_code="VIDEO_ENHANCEMENT_ERROR",
            details={'reason': reason, 'enhancement_type': enhancement_type},
            user_message="Video enhancement failed. The original video will be used instead."
        )


class ManimeAnimationError(PipelineStageException):
    """Raised when Manim animation generation fails."""
    
    def __init__(self, reason: str, script_error: Optional[str] = None):
        details = {'reason': reason}
        if script_error:
            details['script_error'] = script_error
            
        super().__init__(
            message=f"Manim animation failed: {reason}",
            error_code="MANIM_ANIMATION_ERROR",
            details=details,
            user_message="Animation generation failed. The video will be created without animations."
        )


class FinalAssemblyError(PipelineStageException):
    """Raised when final video assembly fails."""
    
    def __init__(self, reason: str, missing_components: Optional[list] = None):
        details = {'reason': reason}
        if missing_components:
            details['missing_components'] = missing_components
            
        super().__init__(
            message=f"Final assembly failed: {reason}",
            error_code="FINAL_ASSEMBLY_ERROR",
            details=details,
            user_message="Video assembly failed. Please try processing again."
        )


# === System Resource Exceptions ===

class ResourceException(PipelineException):
    """Base class for system resource errors."""
    pass


class InsufficientStorageError(ResourceException):
    """Raised when system runs out of storage space."""
    
    def __init__(self, required_mb: int, available_mb: int):
        super().__init__(
            message=f"Insufficient storage: required {required_mb}MB, available {available_mb}MB",
            error_code="INSUFFICIENT_STORAGE",
            details={'required_mb': required_mb, 'available_mb': available_mb},
            user_message="Insufficient storage space. Please try again later or reduce file sizes."
        )


class InsufficientMemoryError(ResourceException):
    """Raised when system runs out of memory."""
    
    def __init__(self, operation: str, required_mb: Optional[int] = None):
        details = {'operation': operation}
        if required_mb:
            details['required_mb'] = required_mb
            
        super().__init__(
            message=f"Insufficient memory for operation: {operation}",
            error_code="INSUFFICIENT_MEMORY",
            details=details,
            user_message="System is currently overloaded. Please try again in a few minutes."
        )


class ProcessingTimeoutError(ResourceException):
    """Raised when processing exceeds maximum allowed time."""
    
    def __init__(self, operation: str, timeout_seconds: int):
        super().__init__(
            message=f"Processing timeout for {operation}: exceeded {timeout_seconds} seconds",
            error_code="PROCESSING_TIMEOUT",
            details={'operation': operation, 'timeout_seconds': timeout_seconds},
            user_message=f"Processing is taking longer than expected. Operation timed out after {timeout_seconds} seconds."
        )


# === Environment and Dependency Exceptions ===

class EnvironmentException(PipelineException):
    """Base class for environment and dependency errors."""
    pass


class CondaEnvironmentError(EnvironmentException):
    """Raised when conda environment is not available or corrupted."""
    
    def __init__(self, env_name: str, reason: str):
        super().__init__(
            message=f"Conda environment '{env_name}' error: {reason}",
            error_code="CONDA_ENVIRONMENT_ERROR",
            details={'env_name': env_name, 'reason': reason},
            user_message="System environment error. Please contact support."
        )


class DependencyError(EnvironmentException):
    """Raised when required dependencies are missing or incompatible."""
    
    def __init__(self, dependency: str, required_version: Optional[str] = None):
        details = {'dependency': dependency}
        if required_version:
            details['required_version'] = required_version
            
        super().__init__(
            message=f"Missing or incompatible dependency: {dependency}",
            error_code="DEPENDENCY_ERROR",
            details=details,
            user_message="System dependency error. Please contact support."
        )


# === Exception Utilities ===

class ExceptionHandler:
    """Utility class for handling pipeline exceptions."""
    
    @staticmethod
    def handle_exception(e: Exception) -> Dict[str, Any]:
        """Convert any exception to a standardized API response.
        
        Args:
            e: Exception to handle
            
        Returns:
            Standardized error response dictionary
        """
        if isinstance(e, PipelineException):
            return e.to_dict()
        else:
            # Handle unexpected exceptions
            return {
                'error': True,
                'error_code': 'UNEXPECTED_ERROR',
                'message': str(e),
                'user_message': 'An unexpected error occurred. Please try again or contact support.',
                'details': {'exception_type': type(e).__name__}
            }
    
    @staticmethod
    def is_retryable_error(error_code: str) -> bool:
        """Determine if an error is retryable.
        
        Args:
            error_code: Error code to check
            
        Returns:
            True if error is retryable, False otherwise
        """
        retryable_errors = {
            'GEMINI_API_ERROR',
            'PROCESSING_TIMEOUT',
            'INSUFFICIENT_MEMORY',
            'SESSION_CONCURRENT_ACCESS'
        }
        return error_code in retryable_errors
    
    @staticmethod
    def get_retry_delay(error_code: str) -> int:
        """Get recommended retry delay in seconds.
        
        Args:
            error_code: Error code
            
        Returns:
            Recommended delay in seconds
        """
        delay_mapping = {
            'GEMINI_API_ERROR': 30,
            'PROCESSING_TIMEOUT': 60,
            'INSUFFICIENT_MEMORY': 120,
            'SESSION_CONCURRENT_ACCESS': 5
        }
        return delay_mapping.get(error_code, 10)


# === Exception Factory ===

class PipelineExceptionFactory:
    """Factory for creating appropriate pipeline exceptions."""
    
    @staticmethod
    def create_validation_error(validation_result: Dict[str, Any]) -> ValidationException:
        """Create validation exception from validation result.
        
        Args:
            validation_result: Validation result dictionary
            
        Returns:
            Appropriate validation exception
        """
        if 'missing_fields' in validation_result:
            return RequiredFieldError(validation_result['missing_fields'])
        elif 'invalid_fields' in validation_result:
            # For multiple invalid fields, create a generic validation error
            return ValidationException(
                message=f"Validation failed: {validation_result['errors']}",
                error_code="VALIDATION_ERROR",
                details=validation_result,
                user_message="Please check your input values and try again."
            )
        else:
            return ValidationException(
                message="Unknown validation error",
                error_code="VALIDATION_ERROR",
                details=validation_result,
                user_message="Input validation failed. Please check your values."
            )