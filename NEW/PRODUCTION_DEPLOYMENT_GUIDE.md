# Production Deployment Guide

## Video Synthesis Pipeline - Production Ready System

This guide covers the deployment and integration of the enhanced production-ready video synthesis pipeline with all new core systems.

## 🔧 Core Systems Overview

The production system includes six critical components that work together to provide enterprise-grade video synthesis:

### 1. Production Mode Manager (`production_mode_manager.py`)
- **Purpose**: Distinguishes between simulation (testing) and production (actual video processing) modes
- **Key Features**:
  - Automatic conda environment detection
  - Stage capability assessment
  - Processing time estimation
  - Output expectations management

### 2. Exception Management (`pipeline_exceptions.py`)
- **Purpose**: Structured error handling with user-friendly messages and error codes
- **Key Features**:
  - Custom exception hierarchy with error codes
  - User-friendly error messages for frontend
  - Structured error details for debugging
  - Global exception handler

### 3. Concurrent Session Manager (`concurrent_session_manager.py`)
- **Purpose**: Multi-user session isolation and resource management
- **Key Features**:
  - Thread-safe session management
  - Resource allocation and limits
  - Session isolation directories
  - Automatic cleanup of expired sessions

### 4. Real-time Progress Manager (`realtime_progress_manager.py`)
- **Purpose**: Live progress updates for frontend integration
- **Key Features**:
  - WebSocket and SSE compatible
  - Stage-based progress tracking
  - Event subscription system
  - Progress estimation and completion tracking

### 5. Secure File Validator (`secure_file_validator.py`)
- **Purpose**: Comprehensive file upload security and validation
- **Key Features**:
  - Multi-layer security scanning
  - File signature validation
  - Virus scanning (ClamAV integration)
  - Safe file quarantine system

### 6. Memory Cleanup Manager (`memory_cleanup_manager.py`)
- **Purpose**: Automatic resource management and system health monitoring
- **Key Features**:
  - Real-time memory monitoring
  - Automatic cleanup scheduling
  - Emergency resource recovery
  - Background maintenance threads

## 🚀 Quick Start Production Setup

### Prerequisites

1. **Conda Environments**:
   ```bash
   # Required conda environments for full production mode
   conda create -n xtts_voice_cloning python=3.9
   conda create -n sadtalker python=3.8
   conda create -n realesrgan python=3.9
   conda create -n video-audio-processing python=3.9
   ```

2. **Environment Variables**:
   ```bash
   export GEMINI_API_KEY="your_gemini_api_key"
   export PRODUCTION_MODE="true"
   ```

3. **System Dependencies**:
   ```bash
   # Install ClamAV for virus scanning (optional but recommended)
   sudo apt-get install clamav clamav-daemon
   ```

### Basic Integration

```python
from NEW.core.production_mode_manager import ProductionModeManager
from NEW.core.concurrent_session_manager import ConcurrentSessionManager
from NEW.core.realtime_progress_manager import RealtimeProgressManager
from NEW.core.secure_file_validator import SecureFileValidator
from NEW.core.memory_cleanup_manager import MemoryCleanupManager

# Initialize core systems
production_manager = ProductionModeManager()
session_manager = ConcurrentSessionManager(max_concurrent_sessions=10)
progress_manager = RealtimeProgressManager()
file_validator = SecureFileValidator()
cleanup_manager = MemoryCleanupManager()

# Check production readiness
capabilities = production_manager.get_system_capabilities()
if capabilities['production_readiness']['production_ready']:
    production_manager.set_processing_mode('production')
else:
    production_manager.set_processing_mode('simulation')
```

## 📋 Processing Pipeline Integration

### 1. Session Creation and Management

```python
# Create user session
session_id = session_manager.create_session(user_id="user123")

# Get session isolation directory
session_path = session_manager.get_session_isolation_path(session_id)
```

### 2. File Upload and Validation

```python
# Validate uploaded files
from NEW.core.secure_file_validator import FileType

face_result = file_validator.validate_file(
    file_path="/path/to/face_image.jpg",
    expected_type=FileType.FACE_IMAGE
)

if face_result.is_valid:
    # Create safe copy in session directory
    safe_path = file_validator.create_safe_copy(
        file_path="/path/to/face_image.jpg",
        destination=str(session_path / "assets")
    )
else:
    # Handle validation errors
    for error in face_result.errors:
        print(f"Validation error: {error}")
```

### 3. Real-time Progress Tracking

```python
# Subscribe to progress updates
def progress_callback(event):
    print(f"Progress: {event.progress_percent}% - {event.message}")

subscription_id = progress_manager.subscribe_to_session(session_id, progress_callback)

# During processing, emit progress updates
progress_manager.start_stage(session_id, "voice_cloning", estimated_duration=30)
progress_manager.update_stage_progress(session_id, "voice_cloning", 50, "Processing voice...")
progress_manager.complete_stage(session_id, "voice_cloning", success=True)
```

### 4. Pipeline Stage Execution

```python
# Check if stage can run in production mode
if production_manager.can_run_stage_in_production("voice_cloning"):
    # Run actual voice cloning
    result = run_voice_cloning_production(session_path, audio_file)
else:
    # Run simulation
    result = run_voice_cloning_simulation(session_path)

# Handle stage completion
if result.success:
    progress_manager.complete_stage(session_id, "voice_cloning", True, "Voice cloning completed")
else:
    progress_manager.complete_stage(session_id, "voice_cloning", False, result.error_message)
```

## 🔧 Frontend Integration Guide

### WebSocket Integration for Real-time Updates

```javascript
// Connect to progress WebSocket
const ws = new WebSocket(`ws://localhost:8000/progress/${sessionId}`);

ws.onmessage = function(event) {
    const progressData = JSON.parse(event.data);
    
    // Update UI based on progress
    updateProgressBar(progressData.progress_percent);
    updateStatusMessage(progressData.message);
    
    if (progressData.event_type === 'pipeline_completed') {
        showCompletionDialog(progressData.details.final_output);
    }
};
```

### REST API Endpoints

```python
# Example Flask/FastAPI integration
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/start-processing', methods=['POST'])
def start_processing():
    try:
        # Create session
        session_id = session_manager.create_session(
            user_id=request.json.get('user_id')
        )
        
        # Validate uploaded files
        face_file = request.files['face_image']
        validation_result = file_validator.validate_file(
            face_file.filename, 
            FileType.FACE_IMAGE
        )
        
        if not validation_result.is_valid:
            return jsonify({
                'error': 'FILE_VALIDATION_FAILED',
                'details': validation_result.errors
            }), 400
        
        # Start processing pipeline
        start_pipeline_processing(session_id, face_file)
        
        return jsonify({
            'session_id': session_id,
            'status': 'processing_started'
        })
        
    except Exception as e:
        return handle_pipeline_exception(e)

@app.route('/api/session/<session_id>/status', methods=['GET'])
def get_session_status(session_id):
    try:
        progress = progress_manager.get_session_progress(session_id)
        return jsonify(progress)
    except SessionNotFoundError:
        return jsonify({'error': 'Session not found'}), 404
```

## 🔒 Security Configuration

### File Upload Security

```python
# Configure file validation limits
file_validator.max_file_sizes = {
    FileType.FACE_IMAGE: 10 * 1024 * 1024,  # 10MB
    FileType.VOICE_AUDIO: 50 * 1024 * 1024,  # 50MB
}

# Enable virus scanning (requires ClamAV)
# Files with threats are automatically quarantined
```

### Session Security

```python
# Configure session limits
session_manager = ConcurrentSessionManager(
    max_concurrent_sessions=10,  # Limit concurrent users
)

# Sessions automatically expire and clean up
# Isolated directories prevent cross-session access
```

## 📊 Monitoring and Health Checks

### System Health Monitoring

```python
# Get comprehensive system status
def check_system_health():
    # Memory and resource usage
    cleanup_status = cleanup_manager.check_resource_health()
    
    # Session management status
    session_status = session_manager.get_system_status()
    
    # Production readiness
    production_status = production_manager.get_system_capabilities()
    
    return {
        'memory_health': cleanup_status,
        'session_health': session_status,
        'production_capabilities': production_status,
        'overall_healthy': all([
            cleanup_status['overall_health'] != 'critical',
            session_status['system_healthy'],
            not production_status['production_readiness']['production_ready'] or 
            production_status['current_mode'] == 'production'
        ])
    }
```

### Emergency Procedures

```python
# Emergency cleanup when resources are low
if cleanup_manager.check_resource_health()['immediate_actions_needed']:
    emergency_results = cleanup_manager.emergency_cleanup()
    print(f"Emergency cleanup freed {emergency_results['space_freed_mb']}MB")

# Force cleanup of all expired sessions
session_manager._cleanup_expired_sessions()
```

## 🐛 Error Handling and Debugging

### Structured Exception Handling

```python
from NEW.core.pipeline_exceptions import ExceptionHandler

# Global exception handler setup
exception_handler = ExceptionHandler()

try:
    # Pipeline processing code
    process_video_pipeline(session_id, inputs)
except Exception as e:
    # Convert to user-friendly format
    user_error = exception_handler.handle_exception(e)
    
    # Log for developers
    logger.error(f"Pipeline error: {user_error['error_code']}")
    
    # Return to frontend
    return {
        'error': user_error['error_code'],
        'message': user_error['user_message'],
        'details': user_error['details'] if DEBUG else None
    }
```

### Common Issues and Solutions

1. **Environment Not Found**:
   ```bash
   # Check conda environments
   conda env list
   
   # Verify environment functionality
   python -c "from NEW.core.production_mode_manager import ProductionModeManager; print(ProductionModeManager().get_system_capabilities())"
   ```

2. **Memory Issues**:
   ```python
   # Check memory usage
   status = cleanup_manager.get_resource_usage()
   print(f"Memory usage: {status['system_stats']['memory']['used_percent']}%")
   
   # Force cleanup if needed
   cleanup_manager.force_cleanup()
   ```

3. **Session Conflicts**:
   ```python
   # List active sessions
   sessions = session_manager.list_active_sessions()
   
   # Clean up specific session
   session_manager.cleanup_session(session_id, force=True)
   ```

## 🚀 Production Deployment Checklist

### Pre-deployment

- [ ] All conda environments installed and tested
- [ ] GEMINI_API_KEY configured
- [ ] ClamAV installed and updated (optional)
- [ ] Sufficient disk space (minimum 10GB recommended)
- [ ] System memory adequate (minimum 16GB recommended)

### Deployment

- [ ] Copy NEW/core/ directory to production server
- [ ] Install required Python dependencies
- [ ] Run production capability check
- [ ] Configure session and memory limits
- [ ] Set up monitoring and logging
- [ ] Test with simulation mode first

### Post-deployment

- [ ] Monitor system health endpoints
- [ ] Set up log rotation for session data
- [ ] Configure automated backups
- [ ] Test emergency cleanup procedures
- [ ] Verify real-time progress updates working

## 📈 Performance Optimization

### Recommended Settings

```python
# Production optimized settings
session_manager = ConcurrentSessionManager(
    max_concurrent_sessions=5,  # Conservative for stability
)

cleanup_manager = MemoryCleanupManager()
cleanup_manager.memory_warning_threshold = 70  # Earlier warnings
cleanup_manager.storage_warning_threshold = 80

# Enable aggressive cleanup for high-load scenarios
cleanup_manager.cleanup_intervals[CleanupPriority.HIGH] = timedelta(minutes=5)
```

### Scaling Considerations

- **Horizontal Scaling**: Each instance manages its own sessions
- **Load Balancing**: Use session-sticky load balancing
- **Database Integration**: Session metadata can be stored in database
- **Caching**: Progress events can be cached for performance

## 📞 Support and Troubleshooting

### Logging Configuration

```python
import logging

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
```

### Health Check Endpoint

```python
@app.route('/health')
def health_check():
    try:
        health = check_system_health()
        status_code = 200 if health['overall_healthy'] else 503
        return jsonify(health), status_code
    except Exception as e:
        return jsonify({'error': 'Health check failed'}), 500
```

## 🎯 Next Steps for Frontend Development

With this production-ready backend, the frontend can now:

1. **Upload files securely** with comprehensive validation
2. **Track progress in real-time** via WebSocket/SSE
3. **Handle multiple concurrent users** with session isolation
4. **Display meaningful error messages** with structured exceptions
5. **Monitor system health** and resource usage
6. **Switch between production and simulation modes** based on capabilities

The system is now ready for production deployment and frontend integration!