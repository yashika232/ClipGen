#!/usr/bin/env python3
"""
Frontend API with WebSocket Support for Video Synthesis Pipeline
Enhanced Flask API with real-time WebSocket progress updates for local prototype
"""

import os
import sys
import json
import logging
import traceback
import time
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
from werkzeug.utils import secure_filename
import uuid
import threading

# Add path for AI generators
sys.path.insert(0, str(Path(__file__).parent))
from gemini_script_generator import GeminiScriptGenerator
from ai_thumbnail_generator import AIThumbnailGenerator

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "NEW" / "core"))

from production_mode_manager import ProductionModeManager
from concurrent_session_manager import ConcurrentSessionManager
from realtime_progress_manager import RealtimeProgressManager
from secure_file_validator import SecureFileValidator, FileType
from memory_cleanup_manager import MemoryCleanupManager
from pipeline_exceptions import ExceptionHandler
from production_video_synthesis_pipeline import ProductionVideoSynthesisPipeline

# Import logging system
from pipeline_logger import get_logger, set_session_context, LogComponent, performance_monitor
from logging_config import get_config, set_environment, Environment

# Setup logging
pipeline_logger = get_logger()
logger = logging.getLogger(__name__)

# Initialize Flask app with SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev_secret_key_local_only'
CORS(app, origins=["*"])  # Allow all origins for local development
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)

# Initialize core managers
production_manager = ProductionModeManager(quick_startup=True)
session_manager = ConcurrentSessionManager(max_concurrent_sessions=20)  # Increased limit for script generation
progress_manager = RealtimeProgressManager()
file_validator = SecureFileValidator()
cleanup_manager = MemoryCleanupManager()
exception_handler = ExceptionHandler()

# Load configuration
config_path = Path(__file__).parent / "config.json"
gemini_api_key = None
if config_path.exists():
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            gemini_api_key = config.get('gemini_api_key', '')
    except Exception as e:
        print(f"Warning: Could not load config.json: {e}")

# Initialize AI generators
gemini_script_generator = GeminiScriptGenerator(api_key=gemini_api_key)
ai_thumbnail_generator = AIThumbnailGenerator()

# Set production mode
production_manager.set_processing_mode('production')

# Configure upload settings
UPLOAD_FOLDER = Path(__file__).parent / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}

# Track active WebSocket connections
active_connections = {}

def allowed_file(filename, file_type='image'):
    """Check if file extension is allowed."""
    if '.' not in filename:
        return False
    
    ext = filename.rsplit('.', 1)[1].lower()
    
    if file_type == 'image':
        return ext in ALLOWED_IMAGE_EXTENSIONS
    elif file_type == 'audio':
        return ext in ALLOWED_AUDIO_EXTENSIONS
    
    return False

def handle_api_error(error):
    """Handle API errors consistently."""
    error_response = exception_handler.handle_exception(error)
    
    # Log the error
    pipeline_logger.error(
        LogComponent.API_SERVER,
        "api_error",
        f"API error: {error_response['user_message']}",
        metadata={
            'error_code': error_response['error_code'],
            'details': error_response.get('details', {})
        },
        error=error
    )
    
    return jsonify({
        'success': False,
        'error': error_response['error_code'],
        'message': error_response['user_message'],
        'details': error_response.get('details', {})
    }), 500

def log_api_request(f):
    """Decorator to log API requests."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            # Set session context if available
            session_id = request.form.get('session_id') or request.json.get('session_id') if request.is_json else None
            if session_id:
                set_session_context(session_id)
            
            # Log request start
            pipeline_logger.info(
                LogComponent.API_SERVER,
                "api_request_start",
                f"{request.method} {request.endpoint}",
                metadata={
                    'method': request.method,
                    'endpoint': request.endpoint,
                    'url': request.url,
                    'content_type': request.content_type,
                    'content_length': request.content_length,
                    'user_agent': request.headers.get('User-Agent'),
                    'remote_addr': request.remote_addr
                }
            )
            
            # Execute request
            response = f(*args, **kwargs)
            execution_time = (time.time() - start_time) * 1000
            
            # Determine status code
            status_code = response[1] if isinstance(response, tuple) else 200
            
            # Log successful response
            pipeline_logger.info(
                LogComponent.API_SERVER,
                "api_request_complete",
                f"{request.method} {request.endpoint} - {status_code}",
                metadata={
                    'method': request.method,
                    'endpoint': request.endpoint,
                    'status_code': status_code,
                    'response_size': len(str(response[0])) if isinstance(response, tuple) else len(str(response))
                },
                execution_time_ms=execution_time
            )
            
            return response
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            # Log error
            pipeline_logger.error(
                LogComponent.API_SERVER,
                "api_request_error",
                f"{request.method} {request.endpoint} - Error",
                metadata={
                    'method': request.method,
                    'endpoint': request.endpoint,
                    'error_type': type(e).__name__
                },
                execution_time_ms=execution_time,
                error=e
            )
            
            # Re-raise to be handled by error handler
            raise
            
    wrapper.__name__ = f.__name__
    return wrapper

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info(f"Client connected: {request.sid}")
    
    # Log WebSocket connection
    pipeline_logger.info(
        LogComponent.API_SERVER,
        "websocket_connect",
        f"Client connected via WebSocket",
        metadata={
            'client_id': request.sid,
            'remote_addr': request.remote_addr,
            'user_agent': request.headers.get('User-Agent')
        }
    )
    
    emit('connected', {'message': 'Connected to video synthesis pipeline'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info(f"Client disconnected: {request.sid}")
    
    # Log WebSocket disconnection
    pipeline_logger.info(
        LogComponent.API_SERVER,
        "websocket_disconnect",
        f"Client disconnected from WebSocket",
        metadata={
            'client_id': request.sid,
            'had_active_connection': request.sid in active_connections
        }
    )
    
    # Clean up any subscriptions for this client
    if request.sid in active_connections:
        session_id = active_connections[request.sid].get('session_id')
        subscription_id = active_connections[request.sid].get('subscription_id')
        
        if subscription_id:
            try:
                progress_manager.unsubscribe(subscription_id)
                logger.info(f"Unsubscribed progress updates for session {session_id}")
                
                # Log cleanup
                pipeline_logger.info(
                    LogComponent.PROGRESS_MANAGER,
                    "subscription_cleanup",
                    f"Cleaned up subscription for disconnected client",
                    metadata={
                        'client_id': request.sid,
                        'session_id': session_id,
                        'subscription_id': subscription_id
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to unsubscribe: {e}")
                pipeline_logger.warning(
                    LogComponent.PROGRESS_MANAGER,
                    "subscription_cleanup_failed",
                    f"Failed to clean up subscription for disconnected client",
                    metadata={
                        'client_id': request.sid,
                        'session_id': session_id,
                        'subscription_id': subscription_id
                    },
                    error=e
                )
        
        del active_connections[request.sid]

@socketio.on('subscribe_progress')
def handle_subscribe_progress(data):
    """Subscribe to progress updates for a session."""
    session_id = data.get('session_id')
    
    if not session_id:
        emit('error', {'message': 'Session ID required'})
        return
    
    try:
        # Join room for session-specific updates
        join_room(f"session_{session_id}")
        
        # Create progress callback that emits to the specific room
        def progress_callback(event):
            socketio.emit('progress_update', {
                'session_id': session_id,
                'event_type': event.event_type,
                'stage': event.stage,
                'progress_percent': event.progress_percent,
                'message': event.message,
                'timestamp': event.timestamp.isoformat(),
                'details': event.details
            }, room=f"session_{session_id}")
        
        # Subscribe to progress manager
        subscription_id = progress_manager.subscribe_to_session(session_id, progress_callback)
        
        # Track connection
        active_connections[request.sid] = {
            'session_id': session_id,
            'subscription_id': subscription_id,
            'joined_at': datetime.now().isoformat()
        }
        
        emit('subscribed', {
            'session_id': session_id,
            'subscription_id': subscription_id,
            'message': 'Subscribed to progress updates'
        })
        
        logger.info(f"Client {request.sid} subscribed to session {session_id}")
        
    except Exception as e:
        logger.error(f"Failed to subscribe to progress: {e}")
        emit('error', {'message': f'Failed to subscribe: {str(e)}'})

@socketio.on('unsubscribe_progress')
def handle_unsubscribe_progress():
    """Unsubscribe from progress updates."""
    if request.sid in active_connections:
        connection_info = active_connections[request.sid]
        session_id = connection_info.get('session_id')
        subscription_id = connection_info.get('subscription_id')
        
        try:
            if subscription_id:
                progress_manager.unsubscribe(subscription_id)
            
            if session_id:
                leave_room(f"session_{session_id}")
            
            del active_connections[request.sid]
            
            emit('unsubscribed', {'message': 'Unsubscribed from progress updates'})
            logger.info(f"Client {request.sid} unsubscribed from session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe: {e}")
            emit('error', {'message': f'Failed to unsubscribe: {str(e)}'})

# REST API endpoints (same as before but enhanced)
@app.errorhandler(413)
def too_large(e):
    """Handle file too large errors."""
    return jsonify({
        'success': False,
        'error': 'FILE_TOO_LARGE',
        'message': 'File is too large. Maximum size is 50MB.'
    }), 413

@app.route('/health', methods=['GET'])
@log_api_request
def health_check():
    """Health check endpoint for frontend."""
    try:
        # Get system capabilities
        capabilities = production_manager.get_system_capabilities()
        
        # Get memory status
        memory_status = cleanup_manager.check_resource_health()
        
        # Get session status
        session_status = session_manager.get_system_status()
        
        overall_healthy = (
            capabilities['production_readiness']['production_ready'] and
            memory_status['overall_health'] != 'critical' and
            session_status['system_healthy']
        )
        
        return jsonify({
            'success': True,
            'healthy': overall_healthy,
            'timestamp': datetime.now().isoformat(),
            'production_ready': capabilities['production_readiness']['production_ready'],
            'available_environments': capabilities['environment_status']['available_environments'],
            'total_environments': capabilities['environment_status']['total_environments'],
            'memory_usage_percent': session_status['memory_usage']['utilization_percent'],
            'active_sessions': session_status['active_sessions'],
            'max_sessions': session_status['max_sessions'],
            'websocket_connections': len(active_connections)
        })
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'success': False,
            'healthy': False,
            'error': str(e)
        }), 500

@app.route('/capabilities', methods=['GET'])
@log_api_request
def get_capabilities():
    """Get detailed system capabilities."""
    try:
        capabilities = production_manager.get_system_capabilities()
        validation = production_manager.validate_environment_for_production()
        
        # Convert any enum values to strings for JSON serialization
        def convert_enums_to_strings(obj):
            """Recursively convert enum objects to their string values for JSON serialization."""
            try:
                # Try to serialize directly first - if it works, return as-is
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                pass
            
            if hasattr(obj, '__dict__'):
                # Handle objects with attributes (like StageCapability)
                result = {}
                for key, value in obj.__dict__.items():
                    if not key.startswith('_'):  # Skip private attributes
                        result[key] = convert_enums_to_strings(value)
                return result
            elif isinstance(obj, dict):
                # Handle dictionaries
                return {key: convert_enums_to_strings(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                # Handle lists and tuples
                return [convert_enums_to_strings(item) for item in obj]
            elif hasattr(obj, 'value'):
                # Handle enum objects
                return obj.value
            elif hasattr(obj, '__str__'):
                # Handle any object with string representation
                return str(obj)
            else:
                # Handle primitive types or fallback to string
                try:
                    return obj
                except:
                    return str(obj)
        
        # Convert all enum values to strings
        safe_capabilities = convert_enums_to_strings(capabilities)
        safe_validation = convert_enums_to_strings(validation)
        
        return jsonify({
            'success': True,
            'capabilities': safe_capabilities,
            'validation': safe_validation,
            'estimated_times': {
                stage: {'estimated_seconds': production_manager.estimate_processing_time(stage, 1000)}
                for stage in ['script_generation', 'voice_cloning', 'video_generation', 
                             'video_enhancement', 'final_assembly']
            },
            'websocket_endpoint': 'ws://localhost:5002/socket.io',
            'supported_events': ['subscribe_progress', 'unsubscribe_progress', 'progress_update']
        })
        
    except Exception as e:
        return handle_api_error(e)

@app.route('/session/create', methods=['POST'])
@log_api_request
def create_session():
    """Create a new processing session."""
    try:
        data = request.get_json() or {}
        user_id = data.get('user_id', 'anonymous')
        
        session_id = session_manager.create_session(user_id=user_id)
        session_info = session_manager.get_session(session_id)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'session_info': {
                'user_id': session_info.user_id,
                'state': session_info.state.value,
                'created_at': session_info.created_at.isoformat(),
                'expires_at': session_info.expires_at.isoformat(),
                'isolation_path': session_info.isolation_directory
            },
            'websocket_room': f"session_{session_id}"
        })
        
    except Exception as e:
        return handle_api_error(e)

@app.route('/session/<session_id>/status', methods=['GET'])
def get_session_status(session_id):
    """Get session status and progress."""
    try:
        session_info = session_manager.get_session(session_id)
        progress_info = progress_manager.get_session_progress(session_id)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'state': session_info.state.value,
            'current_stage': session_info.current_stage,
            'progress': progress_info,
            'last_activity': session_info.last_activity.isoformat(),
            'websocket_room': f"session_{session_id}"
        })
        
    except Exception as e:
        return handle_api_error(e)

@app.route('/upload/face', methods=['POST'])
def upload_face_image():
    """Upload and validate a face image."""
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'NO_FILE',
                'message': 'No file provided'
            }), 400
        
        file = request.files['file']
        session_id = request.form.get('session_id')
        
        if not session_id:
            return jsonify({
                'success': False,
                'error': 'NO_SESSION',
                'message': 'Session ID required'
            }), 400
        
        # Validate session exists
        try:
            session_info = session_manager.get_session(session_id)
            if not session_info:
                return jsonify({
                    'success': False,
                    'error': 'SESSION_NOT_FOUND',
                    'message': 'Session not found. Please start a new session.'
                }), 404
        except Exception as e:
            return jsonify({
                'success': False,
                'error': 'SESSION_ERROR',
                'message': 'Session not found. Please start a new session.'
            }), 404
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'NO_FILENAME',
                'message': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename, 'image'):
            return jsonify({
                'success': False,
                'error': 'INVALID_FILE_TYPE',
                'message': f'Invalid file type. Allowed: {", ".join(ALLOWED_IMAGE_EXTENSIONS)}'
            }), 400
        
        # Emit upload start event
        socketio.emit('upload_start', {
            'session_id': session_id,
            'file_type': 'face_image',
            'filename': file.filename
        }, room=f"session_{session_id}")
        
        # Secure filename and save
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = Path(app.config['UPLOAD_FOLDER']) / unique_filename
        file.save(str(file_path))
        
        # Validate file
        validation_result = file_validator.validate_file(str(file_path), FileType.FACE_IMAGE)
        
        if not validation_result.is_valid:
            # Remove invalid file
            file_path.unlink(missing_ok=True)
            
            # Emit upload failure
            socketio.emit('upload_failed', {
                'session_id': session_id,
                'file_type': 'face_image',
                'errors': validation_result.errors
            }, room=f"session_{session_id}")
            
            return jsonify({
                'success': False,
                'error': 'FILE_VALIDATION_FAILED',
                'message': 'File validation failed',
                'errors': validation_result.errors
            }), 400
        
        # Copy to session directory with error handling
        try:
            session_path = session_manager.get_session_isolation_path(session_id, 'assets')
            safe_path = file_validator.create_safe_copy(str(file_path), str(session_path))
            
            # Remove temporary upload only after successful copy
            file_path.unlink(missing_ok=True)
            
            print(f"[SUCCESS] Face image successfully saved to session: {safe_path}")
            
        except Exception as copy_error:
            print(f"[ERROR] Failed to copy face image to session directory: {copy_error}")
            print(f"   Session path attempted: {session_path if 'session_path' in locals() else 'unknown'}")
            print(f"   Temp file remains at: {file_path}")
            
            # Don't remove temp file if copy failed, use temp path as fallback
            safe_path = str(file_path)
            print(f"   Using temp path as fallback: {safe_path}")
        
        # Emit upload success
        socketio.emit('upload_success', {
            'session_id': session_id,
            'file_type': 'face_image',
            'filename': filename,
            'file_size': Path(safe_path).stat().st_size
        }, room=f"session_{session_id}")
        
        return jsonify({
            'success': True,
            'message': 'Face image uploaded successfully',
            'file_info': {
                'original_name': filename,
                'safe_path': safe_path,
                'file_size': Path(safe_path).stat().st_size,
                'warnings': validation_result.warnings
            }
        })
        
    except Exception as e:
        # Emit upload error
        if 'session_id' in locals():
            socketio.emit('upload_error', {
                'session_id': session_id,
                'error': str(e)
            }, room=f"session_{session_id}")
        
        return handle_api_error(e)

@app.route('/upload/audio', methods=['POST'])
def upload_audio():
    """Upload and validate an audio file."""
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'NO_FILE',
                'message': 'No file provided'
            }), 400
        
        file = request.files['file']
        session_id = request.form.get('session_id')
        
        if not session_id:
            return jsonify({
                'success': False,
                'error': 'NO_SESSION',
                'message': 'Session ID required'
            }), 400
        
        # Validate session exists
        try:
            session_info = session_manager.get_session(session_id)
            if not session_info:
                return jsonify({
                    'success': False,
                    'error': 'SESSION_NOT_FOUND',
                    'message': 'Session not found. Please start a new session.'
                }), 404
        except Exception as e:
            return jsonify({
                'success': False,
                'error': 'SESSION_ERROR',
                'message': 'Session not found. Please start a new session.'
            }), 404
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'NO_FILENAME',
                'message': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename, 'audio'):
            return jsonify({
                'success': False,
                'error': 'INVALID_FILE_TYPE',
                'message': f'Invalid file type. Allowed: {", ".join(ALLOWED_AUDIO_EXTENSIONS)}'
            }), 400
        
        # Emit upload start event
        socketio.emit('upload_start', {
            'session_id': session_id,
            'file_type': 'audio',
            'filename': file.filename
        }, room=f"session_{session_id}")
        
        # Secure filename and save
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = Path(app.config['UPLOAD_FOLDER']) / unique_filename
        file.save(str(file_path))
        
        # Validate file
        validation_result = file_validator.validate_file(str(file_path), FileType.VOICE_AUDIO)
        
        if not validation_result.is_valid:
            # Remove invalid file
            file_path.unlink(missing_ok=True)
            
            # Emit upload failure
            socketio.emit('upload_failed', {
                'session_id': session_id,
                'file_type': 'audio',
                'errors': validation_result.errors
            }, room=f"session_{session_id}")
            
            return jsonify({
                'success': False,
                'error': 'FILE_VALIDATION_FAILED',
                'message': 'File validation failed',
                'errors': validation_result.errors
            }), 400
        
        # Copy to session directory with error handling
        try:
            session_path = session_manager.get_session_isolation_path(session_id, 'assets')
            safe_path = file_validator.create_safe_copy(str(file_path), str(session_path))
            
            # Remove temporary upload only after successful copy
            file_path.unlink(missing_ok=True)
            
            print(f"[SUCCESS] Audio file successfully saved to session: {safe_path}")
            
        except Exception as copy_error:
            print(f"[ERROR] Failed to copy audio file to session directory: {copy_error}")
            print(f"   Session path attempted: {session_path if 'session_path' in locals() else 'unknown'}")
            print(f"   Temp file remains at: {file_path}")
            
            # Don't remove temp file if copy failed, use temp path as fallback
            safe_path = str(file_path)
            print(f"   Using temp path as fallback: {safe_path}")
        
        # Emit upload success
        socketio.emit('upload_success', {
            'session_id': session_id,
            'file_type': 'audio',
            'filename': filename,
            'file_size': Path(safe_path).stat().st_size
        }, room=f"session_{session_id}")
        
        return jsonify({
            'success': True,
            'message': 'Audio file uploaded successfully',
            'file_info': {
                'original_name': filename,
                'safe_path': safe_path,
                'file_size': Path(safe_path).stat().st_size,
                'warnings': validation_result.warnings
            }
        })
        
    except Exception as e:
        # Emit upload error
        if 'session_id' in locals():
            socketio.emit('upload_error', {
                'session_id': session_id,
                'error': str(e)
            }, room=f"session_{session_id}")
        
        return handle_api_error(e)

@app.route('/process/start', methods=['POST'])
def start_processing():
    """Start video processing pipeline."""
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id')
        script_text = data.get('script_text', '')
        emotion = data.get('emotion', 'confident')
        tone = data.get('tone', 'professional')
        
        if not session_id:
            return jsonify({
                'success': False,
                'error': 'NO_SESSION',
                'message': 'Session ID required'
            }), 400
        
        if not script_text.strip():
            return jsonify({
                'success': False,
                'error': 'NO_SCRIPT',
                'message': 'Script text required'
            }), 400
        
        # Validate session exists
        session_info = session_manager.get_session(session_id)
        
        # Start processing with actual pipeline integration
        session_manager.start_session_processing(session_id, 'script_generation')
        
        # Emit processing start event
        socketio.emit('processing_started', {
            'session_id': session_id,
            'script_length': len(script_text),
            'emotion': emotion,
            'tone': tone,
            'timestamp': datetime.now().isoformat()
        }, room=f"session_{session_id}")
        
        # Get estimated total time
        total_estimated_time = sum([
            production_manager.estimate_processing_time(stage, len(script_text))
            for stage in ['script_generation', 'voice_cloning', 'video_generation', 'final_assembly']
        ])
        
        # Start actual video generation pipeline in background thread
        def run_video_pipeline():
            try:
                # Initialize production video synthesis pipeline
                pipeline = ProductionVideoSynthesisPipeline()
                
                # Get session info and asset paths
                session_info = session_manager.get_session(session_id)
                session_dir = Path(session_info.isolation_directory)
                assets_dir = session_dir / 'assets'
                
                # Find uploaded files - check session assets first, then uploads fallback
                voice_reference = None
                face_image = None
                
                print(f"Looking for user assets for session {session_id}")
                print(f"   Session assets dir: {assets_dir}")
                
                # Check session assets directory first
                if assets_dir.exists():
                    print(f"   Session assets directory exists, scanning...")
                    for file_path in assets_dir.iterdir():
                        if file_path.is_file():
                            print(f"   Found file: {file_path}")
                            if file_path.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
                                voice_reference = str(file_path)
                                print(f"   [FOUND] Using session voice: {voice_reference}")
                            elif file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                                face_image = str(file_path)
                                print(f"   [FOUND] Using session face: {face_image}")
                else:
                    print(f"   Session assets directory does not exist")
                
                # Fallback: Look in uploads directory if session files not found
                if not voice_reference or not face_image:
                    uploads_dir = Path(app.config['UPLOAD_FOLDER'])
                    print(f"   Checking uploads fallback directory: {uploads_dir}")
                    
                    if uploads_dir.exists():
                        for file_path in uploads_dir.iterdir():
                            if file_path.is_file():
                                if not voice_reference and file_path.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
                                    voice_reference = str(file_path)
                                    print(f"   [FALLBACK] Using uploads voice fallback: {voice_reference}")
                                elif not face_image and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                                    face_image = str(file_path)
                                    print(f"   [FALLBACK] Using uploads face fallback: {face_image}")
                
                # Final fallback: Use default assets if still none found
                if not voice_reference:
                    voice_reference = "/Users/aryanjain/Documents/video-synthesis-pipeline copy/NEW/user_assets/voices/sample_voice.wav"
                    print(f"   [WARNING] No voice found, using default: {voice_reference}")
                if not face_image:
                    face_image = "/Users/aryanjain/Documents/video-synthesis-pipeline copy/NEW/user_assets/faces/sample_face.jpg"
                    print(f"   [WARNING] No face found, using default: {face_image}")
                    
                print(f"Final asset selection:")
                print(f"   Voice: {voice_reference}")
                print(f"   Face: {face_image}")
                
                # Set up output path in session directory
                output_dir = session_dir / 'output'
                output_dir.mkdir(exist_ok=True)
                output_path = output_dir / f'video_{session_id}.mp4'
                
                # Progress tracking function
                def emit_progress(stage, progress_percent, message):
                    socketio.emit('progress_update', {
                        'session_id': session_id,
                        'stage': stage,
                        'progress_percent': progress_percent,
                        'message': message,
                        'timestamp': datetime.now().isoformat()
                    }, room=f"session_{session_id}")
                
                # Start processing stages
                emit_progress('initializing', 0, 'Initializing production pipeline...')
                
                # Run the complete production pipeline
                result = pipeline.run_complete_pipeline(
                    text=script_text,
                    voice_reference=voice_reference,
                    face_image=face_image,
                    output_path=str(output_path)
                )
                
                if result['success']:
                    # Final progress update
                    socketio.emit('progress_update', {
                        'session_id': session_id,
                        'stage': 'completed',
                        'progress_percent': 100,
                        'message': 'Video generation completed successfully!',
                        'timestamp': datetime.now().isoformat(),
                        'outputs': {
                            'final_video': result['output_video'],
                            'total_time': result['total_time_seconds'],
                            'chunks_processed': result['total_chunks']
                        }
                    }, room=f"session_{session_id}")
                    
                    # Update session to completed
                    session_manager.complete_session_processing(session_id, success=True)
                    
                else:
                    # Error in pipeline
                    socketio.emit('processing_error', {
                        'session_id': session_id,
                        'error': result.get('error', 'Unknown pipeline error'),
                        'message': 'Video generation failed. Please try again.',
                        'timestamp': datetime.now().isoformat()
                    }, room=f"session_{session_id}")
                    
                    # Mark session as failed
                    session_manager.complete_session_processing(session_id, success=False)
                    
            except Exception as e:
                logger.error(f"Pipeline processing error for session {session_id}: {e}")
                socketio.emit('processing_error', {
                    'session_id': session_id,
                    'error': str(e),
                    'message': 'An error occurred during video generation.',
                    'timestamp': datetime.now().isoformat()
                }, room=f"session_{session_id}")
                
                # Mark session as failed
                session_manager.complete_session_processing(session_id, success=False)
        
        # Start pipeline in background thread
        import threading
        pipeline_thread = threading.Thread(target=run_video_pipeline)
        pipeline_thread.daemon = True
        pipeline_thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Processing started',
            'session_id': session_id,
            'estimated_total_time_seconds': total_estimated_time,
            'estimated_total_time_minutes': round(total_estimated_time / 60, 1),
            'websocket_room': f"session_{session_id}",
            'note': 'Subscribe to WebSocket for real-time progress updates'
        })
        
    except Exception as e:
        # Emit processing error
        if 'session_id' in locals():
            socketio.emit('processing_error', {
                'session_id': session_id,
                'error': str(e)
            }, room=f"session_{session_id}")
        
        return handle_api_error(e)

@app.route('/outputs/<session_id>', methods=['GET'])
def get_outputs(session_id):
    """Get list of output files for a session."""
    try:
        session_info = session_manager.get_session(session_id)
        output_path = Path(session_info.isolation_directory) / 'output'
        
        if not output_path.exists():
            return jsonify({
                'success': True,
                'outputs': [],
                'message': 'No outputs yet'
            })
        
        outputs = []
        for file_path in output_path.iterdir():
            if file_path.is_file():
                outputs.append({
                    'filename': file_path.name,
                    'size': file_path.stat().st_size,
                    'created': datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
                    'download_url': f'/download/{session_id}/{file_path.name}'
                })
        
        return jsonify({
            'success': True,
            'outputs': outputs,
            'count': len(outputs)
        })
        
    except Exception as e:
        return handle_api_error(e)

@app.route('/download/<session_id>/<filename>', methods=['GET'])
def download_file(session_id, filename):
    """Download a file from session outputs."""
    try:
        session_info = session_manager.get_session(session_id)
        file_path = Path(session_info.isolation_directory) / 'output' / filename
        
        if not file_path.exists():
            return jsonify({
                'success': False,
                'error': 'FILE_NOT_FOUND',
                'message': 'File not found'
            }), 404
        
        return send_file(str(file_path), as_attachment=True, download_name=filename)
        
    except Exception as e:
        return handle_api_error(e)

@app.route('/cleanup', methods=['POST'])
def cleanup_resources():
    """Manually trigger cleanup (for testing)."""
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id')
        
        if session_id:
            # Clean specific session
            success = session_manager.cleanup_session(session_id, force=True)
            message = f"Session {session_id} cleanup: {'success' if success else 'failed'}"
            
            # Emit cleanup event
            socketio.emit('session_cleaned', {
                'session_id': session_id,
                'success': success
            }, room=f"session_{session_id}")
        else:
            # General cleanup
            cleanup_results = cleanup_manager.force_cleanup()
            message = f"Cleanup completed: {cleanup_results['tasks_successful']}/{cleanup_results['tasks_run']} successful"
        
        return jsonify({
            'success': True,
            'message': message
        })
        
    except Exception as e:
        return handle_api_error(e)

@app.route('/generate/script', methods=['POST'])
@log_api_request
def generate_script():
    """Generate script using Gemini API and store in metadata."""
    try:
        data = request.get_json() or {}
        session_id = data.get('sessionId') or data.get('session_id')
        
        # Required parameters
        topic = data.get('topic', '').strip()
        if not topic:
            return jsonify({
                'success': False,
                'error': 'MISSING_TOPIC',
                'message': 'Topic is required'
            }), 400
        
        # Optional parameters with defaults
        duration = data.get('duration', 5)
        tone = data.get('tone', 'professional')
        emotion = data.get('emotion', 'confident')
        audience = data.get('audience', 'general public')
        content_type = data.get('contentType', 'educational')
        
        # Validate session
        if session_id:
            session_info = session_manager.get_session(session_id)
        else:
            # Create new session if none provided
            session_id = session_manager.create_session(f"script_gen_{int(time.time())}")
        
        # Generate script using real Gemini API
        script_params = {
            'topic': topic,
            'duration': duration,
            'tone': tone,
            'emotion': emotion,
            'audience': audience,
            'contentType': content_type
        }
        
        # Log script generation start
        pipeline_logger.info(
            LogComponent.GEMINI_GENERATOR,
            "script_generation_start",
            f"Starting script generation for topic: {topic}",
            metadata={
                'topic': topic,
                'duration': duration,
                'tone': tone,
                'emotion': emotion,
                'audience': audience,
                'content_type': content_type
            }
        )
        
        script_result = gemini_script_generator.generate_script(script_params)
        
        if not script_result['success']:
            pipeline_logger.error(
                LogComponent.GEMINI_GENERATOR,
                "script_generation_failed",
                f"Failed to generate script for topic: {topic}",
                metadata=script_params
            )
            return jsonify({
                'success': False,
                'error': 'SCRIPT_GENERATION_FAILED',
                'message': 'Failed to generate script'
            }), 500
        
        script_content = script_result['script']
        estimated_duration = script_result.get('estimated_duration', duration * 60)
        
        # Log successful script generation
        pipeline_logger.info(
            LogComponent.GEMINI_GENERATOR,
            "script_generation_complete",
            f"Successfully generated script for topic: {topic}",
            metadata={
                'topic': topic,
                'word_count': script_result.get('word_count', 0),
                'estimated_duration': estimated_duration,
                'generation_method': script_result.get('generation_method', 'unknown'),
                'script_length': len(script_content)
            }
        )
        
        # Store in metadata system
        metadata_update = {
            'generated_content': {
                'script': script_content,
                'script_sections': script_result.get('sections', {}),
                'script_params': {
                    'topic': topic,
                    'duration': duration,
                    'tone': tone,
                    'emotion': emotion,
                    'audience': audience,
                    'content_type': content_type
                },
                'script_stats': {
                    'word_count': script_result.get('word_count', 0),
                    'estimated_duration': estimated_duration,
                    'generation_method': script_result.get('generation_method', 'unknown')
                },
                'script_generated_at': datetime.now().isoformat()
            },
            'target_duration': estimated_duration
        }
        
        # Update session metadata - store in session for pipeline use
        try:
            session_info = session_manager.get_session(session_id)
            if hasattr(session_info, 'metadata'):
                session_info.metadata.update(metadata_update)
            else:
                session_info.metadata = metadata_update
        except Exception as e:
            logger.warning(f"Failed to update session metadata: {e}")
        
        # Emit generation event
        socketio.emit('generation_completed', {
            'session_id': session_id,
            'type': 'script',
            'result': {
                'script': script_content,
                'content': script_content
            }
        }, room=f"session_{session_id}")
        
        # Extract structured sections from the script result
        script_sections = script_result.get('sections', {})
        
        # Format response to match app.py structure with clean content
        response_data = {
            'success': True,
            'script': {
                'hook': script_sections.get('hook', ''),
                'objectives': script_sections.get('objectives', []),
                'core_content': script_sections.get('core_content', script_content),
                'interactive': script_sections.get('interactive', ''),
                'summary': script_sections.get('summary', '')
            },
            'content': script_content,
            'session_id': session_id,
            'params': {
                'topic': topic,
                'duration': duration,
                'tone': tone,
                'emotion': emotion,
                'audience': audience,
                'content_type': content_type
            },
            'stats': {
                'word_count': script_result.get('word_count', 0),
                'estimated_duration': estimated_duration,
                'generation_method': script_result.get('generation_method', 'unknown')
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return handle_api_error(e)

@app.route('/generate/thumbnail', methods=['POST'])
@log_api_request
def generate_thumbnail():
    """Generate thumbnails using AI and store in metadata."""
    try:
        data = request.get_json() or {}
        session_id = data.get('sessionId') or data.get('session_id')
        
        # Required parameters
        prompt = data.get('prompt', '').strip()
        if not prompt:
            return jsonify({
                'success': False,
                'error': 'MISSING_PROMPT',
                'message': 'Prompt is required'
            }), 400
        
        # Optional parameters with defaults
        style = data.get('style', 'modern')
        quality = data.get('quality', 'high')
        count = data.get('count', 4)
        
        # Validate session
        if session_id:
            session_info = session_manager.get_session(session_id)
        else:
            # Create new session if none provided
            session_id = session_manager.create_session(f"thumb_gen_{int(time.time())}")
        
        # Generate thumbnails using real AI APIs
        thumbnail_params = {
            'prompt': prompt,
            'style': style,
            'quality': quality,
            'count': count
        }
        
        # Log thumbnail generation start
        pipeline_logger.info(
            LogComponent.THUMBNAIL_GENERATOR,
            "thumbnail_generation_start",
            f"Starting thumbnail generation with prompt: {prompt[:50]}...",
            metadata={
                'prompt': prompt,
                'style': style,
                'quality': quality,
                'count': count
            }
        )
        
        thumbnail_result = ai_thumbnail_generator.generate_thumbnails(thumbnail_params)
        
        if not thumbnail_result['success']:
            pipeline_logger.error(
                LogComponent.THUMBNAIL_GENERATOR,
                "thumbnail_generation_failed",
                f"Failed to generate thumbnails for prompt: {prompt[:50]}...",
                metadata=thumbnail_params
            )
            return jsonify({
                'success': False,
                'error': 'THUMBNAIL_GENERATION_FAILED',
                'message': 'Failed to generate thumbnails'
            }), 500
        
        thumbnails = []
        for i, thumbnail_data in enumerate(thumbnail_result['thumbnails']):
            thumbnail = {
                'url': thumbnail_data['url'],
                'style': style,
                'quality': quality,
                'sessionId': session_id,
                'filename': thumbnail_data['filename'],
                'prompt': prompt,
                'method': thumbnail_data.get('method', 'unknown'),
                'local_path': thumbnail_data.get('local_path')
            }
            thumbnails.append(thumbnail)
        
        # Log successful thumbnail generation
        pipeline_logger.info(
            LogComponent.THUMBNAIL_GENERATOR,
            "thumbnail_generation_complete",
            f"Successfully generated {len(thumbnails)} thumbnails",
            metadata={
                'prompt': prompt,
                'thumbnails_generated': len(thumbnails),
                'generation_methods': thumbnail_result.get('generation_methods', []),
                'enhanced_prompt': thumbnail_result.get('enhanced_prompt', prompt)
            }
        )
        
        # Store in metadata system
        metadata_update = {
            'generated_content': {
                'thumbnails': thumbnails,
                'thumbnail_params': {
                    'prompt': prompt,
                    'style': style,
                    'quality': quality,
                    'count': count
                },
                'thumbnail_stats': {
                    'generation_methods': thumbnail_result.get('generation_methods', []),
                    'enhanced_prompt': thumbnail_result.get('enhanced_prompt', prompt),
                    'total_generated': len(thumbnails)
                },
                'thumbnails_generated_at': datetime.now().isoformat()
            }
        }
        
        # Update session metadata - store in session for pipeline use
        try:
            session_info = session_manager.get_session(session_id)
            if hasattr(session_info, 'metadata'):
                session_info.metadata.update(metadata_update)
            else:
                session_info.metadata = metadata_update
        except Exception as e:
            logger.warning(f"Failed to update session metadata: {e}")
        
        # Emit generation event
        socketio.emit('generation_completed', {
            'session_id': session_id,
            'type': 'thumbnail',
            'result': {
                'thumbnails': thumbnails
            }
        }, room=f"session_{session_id}")
        
        return jsonify({
            'success': True,
            'thumbnails': thumbnails,
            'session_id': session_id,
            'params': {
                'prompt': prompt,
                'style': style,
                'quality': quality,
                'count': count
            }
        })
        
    except Exception as e:
        return handle_api_error(e)

if __name__ == '__main__':
    print("Starting Video Synthesis Pipeline API with WebSocket Support")
    print("=" * 70)
    print(f"Health check: http://localhost:5002/health")
    print(f"Capabilities: http://localhost:5002/capabilities")
    print(f"WebSocket URL: ws://localhost:5002/socket.io")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print("=" * 70)
    
    # Check production readiness
    capabilities = production_manager.get_system_capabilities()
    if capabilities['production_readiness']['production_ready']:
        print("[READY] System is production ready!")
    else:
        print("[WARNING] Some environments missing - running in hybrid mode")
    
    print(f"Available environments: {capabilities['environment_status']['available_environments']}/{capabilities['environment_status']['total_environments']}")
    
    print("\nServer WebSocket Events:")
    print("  Client -> Server: subscribe_progress, unsubscribe_progress")
    print("  Server -> Client: progress_update, upload_start, upload_success, processing_started")
    
    # Start Flask app with SocketIO
    socketio.run(app, host='0.0.0.0', port=5002, debug=True, allow_unsafe_werkzeug=True)