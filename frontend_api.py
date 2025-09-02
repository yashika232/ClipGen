#!/usr/bin/env python3
"""
Frontend API for Video Synthesis Pipeline
Simple Flask API for local prototype with frontend integration
"""

import os
import sys
import json
import logging
import traceback
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import uuid

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "NEW" / "core"))

from production_mode_manager import ProductionModeManager
from concurrent_session_manager import ConcurrentSessionManager
from realtime_progress_manager import RealtimeProgressManager
from secure_file_validator import SecureFileValidator, FileType
from memory_cleanup_manager import MemoryCleanupManager
from pipeline_exceptions import ExceptionHandler

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=["*"])  # Allow all origins for local development

# Initialize core managers
production_manager = ProductionModeManager()
session_manager = ConcurrentSessionManager(max_concurrent_sessions=5)  # Limit for local use
progress_manager = RealtimeProgressManager()
file_validator = SecureFileValidator()
cleanup_manager = MemoryCleanupManager()
exception_handler = ExceptionHandler()

# Set production mode
production_manager.set_processing_mode('production')

# Configure upload settings
UPLOAD_FOLDER = Path(__file__).parent / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}

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
    return jsonify({
        'success': False,
        'error': error_response['error_code'],
        'message': error_response['user_message'],
        'details': error_response.get('details', {})
    }), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large errors."""
    return jsonify({
        'success': False,
        'error': 'FILE_TOO_LARGE',
        'message': 'File is too large. Maximum size is 50MB.'
    }), 413

@app.route('/health', methods=['GET'])
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
            'memory_usage_percent': memory_status.get('memory_usage_percent', 0),
            'active_sessions': session_status['active_sessions'],
            'max_sessions': session_status['max_sessions']
        })
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'success': False,
            'healthy': False,
            'error': str(e)
        }), 500

@app.route('/capabilities', methods=['GET'])
def get_capabilities():
    """Get detailed system capabilities."""
    try:
        capabilities = production_manager.get_system_capabilities()
        validation = production_manager.validate_environment_for_production()
        
        return jsonify({
            'success': True,
            'capabilities': capabilities,
            'validation': validation,
            'estimated_times': {
                stage: production_manager.estimate_processing_time(stage, 1000)
                for stage in ['script_generation', 'voice_cloning', 'video_generation', 
                             'video_enhancement', 'final_assembly']
            }
        })
        
    except Exception as e:
        return handle_api_error(e)

@app.route('/session/create', methods=['POST'])
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
            }
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
            'last_activity': session_info.last_activity.isoformat()
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
            return jsonify({
                'success': False,
                'error': 'FILE_VALIDATION_FAILED',
                'message': 'File validation failed',
                'errors': validation_result.errors
            }), 400
        
        # Copy to session directory
        session_path = session_manager.get_session_isolation_path(session_id, 'assets')
        safe_path = file_validator.create_safe_copy(str(file_path), str(session_path))
        
        # Remove temporary upload
        file_path.unlink(missing_ok=True)
        
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
            return jsonify({
                'success': False,
                'error': 'FILE_VALIDATION_FAILED',
                'message': 'File validation failed',
                'errors': validation_result.errors
            }), 400
        
        # Copy to session directory
        session_path = session_manager.get_session_isolation_path(session_id, 'assets')
        safe_path = file_validator.create_safe_copy(str(file_path), str(session_path))
        
        # Remove temporary upload
        file_path.unlink(missing_ok=True)
        
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
        return handle_api_error(e)

@app.route('/process/start', methods=['POST'])
def start_processing():
    """Start video processing pipeline."""
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id')
        script_text = data.get('script_text', '')
        
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
        
        # Start processing (this would trigger the actual pipeline)
        session_manager.start_session_processing(session_id, 'script_generation')
        
        # Subscribe to progress updates (for logging/monitoring)
        def progress_callback(event):
            logger.info(f"Session {session_id}: {event.progress_percent}% - {event.message}")
        
        subscription_id = progress_manager.subscribe_to_session(session_id, progress_callback)
        
        # Get estimated total time
        total_estimated_time = sum([
            production_manager.estimate_processing_time(stage, len(script_text))['estimated_seconds']
            for stage in ['script_generation', 'voice_cloning', 'video_generation', 'final_assembly']
        ])
        
        return jsonify({
            'success': True,
            'message': 'Processing started',
            'session_id': session_id,
            'estimated_total_time_seconds': total_estimated_time,
            'estimated_total_time_minutes': round(total_estimated_time / 60, 1),
            'subscription_id': subscription_id,
            'progress_endpoint': f'/session/{session_id}/status',
            'websocket_endpoint': f'ws://localhost:5000/ws/progress/{session_id}'
        })
        
    except Exception as e:
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

if __name__ == '__main__':
    print("STARTING Starting Video Synthesis Pipeline API")
    print("=" * 50)
    print(f"Health check: http://localhost:5000/health")
    print(f"Capabilities: http://localhost:5000/capabilities")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print("=" * 50)
    
    # Check production readiness
    capabilities = production_manager.get_system_capabilities()
    if capabilities['production_readiness']['production_ready']:
        print("[SUCCESS] System is production ready!")
    else:
        print("[WARNING] Some environments missing - running in hybrid mode")
    
    print(f"Available environments: {capabilities['environment_status']['available_environments']}/{capabilities['environment_status']['total_environments']}")
    
    # Start Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)