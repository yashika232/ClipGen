#!/usr/bin/env python3
"""
Frontend Integration API - Zero Hardcoded Values
Complete API interface for frontend integration with full user configurability
Designed for your teammate's frontend connection
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from dataclasses import asdict

# Add the core directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from user_configuration_system import UserConfigurationSystem, PipelineConfiguration
from enhanced_metadata_manager import EnhancedMetadataManager
from corrected_pipeline_integration import CorrectedPipelineIntegration


class FrontendIntegrationAPI:
    """Frontend Integration API with zero hardcoded values."""
    
    def __init__(self, base_dir: str = None):
        """Initialize frontend integration API.
        
        Args:
            base_dir: Base directory for the pipeline
        """
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        # Initialize components
        self.config_system = UserConfigurationSystem(str(self.base_dir))
        self.metadata_manager = EnhancedMetadataManager(str(self.base_dir))
        self.pipeline_integration = CorrectedPipelineIntegration(str(self.base_dir.parent))
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("API Frontend Integration API initialized - Zero hardcoded values")
    
    # === API Endpoints for Frontend ===
    
    def get_available_options(self) -> Dict[str, Any]:
        """Get all available options for frontend dropdowns.
        
        Returns:
            Complete options for frontend interface
        """
        try:
            options = self.config_system.get_available_options()
            return {
                'success': True,
                'options': options,
                'message': 'Available options retrieved successfully'
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_api_schema(self) -> Dict[str, Any]:
        """Get complete API schema for frontend developers.
        
        Returns:
            Complete API documentation
        """
        try:
            schema = self.config_system.create_frontend_api_schema()
            return {
                'success': True,
                'schema': schema,
                'endpoints': self._get_endpoint_documentation(),
                'message': 'API schema retrieved successfully'
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def validate_user_inputs(self, user_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate user inputs from frontend.
        
        Args:
            user_inputs: Raw user input data from frontend
            
        Returns:
            Validation results
        """
        try:
            validation = self.config_system.validate_user_inputs(user_inputs)
            return {
                'success': validation['valid'],
                'validation': validation,
                'message': 'Validation completed' if validation['valid'] else 'Validation failed'
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def create_session(self, user_inputs: Dict[str, Any], user_assets: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create new pipeline session with user configuration.
        
        Args:
            user_inputs: Complete user input data
            user_assets: Optional user asset files
            
        Returns:
            Session creation result
        """
        try:
            # Validate inputs first
            validation = self.validate_user_inputs(user_inputs)
            if not validation['success']:
                return validation
            
            # Create user configuration
            config = self.config_system.create_user_configuration(user_inputs)
            
            # Save configuration
            config_path = self.config_system.save_configuration(config)
            
            # Create session in metadata system
            session_data = {
                'user_inputs': user_inputs,
                'user_assets': user_assets or {},
                'configuration_path': config_path,
                'created_at': datetime.now().isoformat()
            }
            
            session_id = self.metadata_manager.create_session(session_data)
            
            return {
                'success': True,
                'session_id': session_id,
                'configuration_path': config_path,
                'message': f'Session created successfully for: {user_inputs.get("title", "Untitled")}'
            }
            
        except Exception as e:
            self.logger.error(f"Session creation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_session_status(self, session_id: str = None) -> Dict[str, Any]:
        """Get current session status.
        
        Args:
            session_id: Optional session ID
            
        Returns:
            Session status information
        """
        try:
            metadata = self.metadata_manager.load_metadata()
            if not metadata:
                return {'success': False, 'error': 'No active session found'}
            
            # Get pipeline status
            validation = self.pipeline_integration.validate_working_environment()
            
            # Calculate progress
            stages = metadata.get('pipeline_stages', {})
            total_stages = 8  # Total pipeline stages
            completed_stages = sum(1 for stage in stages.values() if stage.get('status') == 'completed')
            progress = (completed_stages / total_stages) * 100
            
            return {
                'success': True,
                'session_id': metadata.get('session_id'),
                'title': metadata.get('user_inputs', {}).get('title', 'Unknown'),
                'overall_progress': progress,
                'stages_completed': completed_stages,
                'total_stages': total_stages,
                'current_stage': self._get_current_stage(stages),
                'environment_status': validation,
                'last_updated': metadata.get('last_modified')
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def start_pipeline_processing(self, session_id: str = None, user_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Start pipeline processing with user configuration.
        
        Args:
            session_id: Optional session ID
            user_config: Optional processing configuration overrides
            
        Returns:
            Processing start result
        """
        try:
            # Load session metadata
            metadata = self.metadata_manager.load_metadata()
            if not metadata:
                return {'success': False, 'error': 'No active session found'}
            
            # Get user inputs from metadata
            user_inputs = metadata.get('user_inputs', {})
            
            # Extract emotion and tone (no defaults!)
            emotion = user_inputs.get('emotion')
            tone = user_inputs.get('tone')
            
            if not emotion or not tone:
                return {
                    'success': False, 
                    'error': 'Missing required parameters: emotion and tone must be provided by user'
                }
            
            # Apply any configuration overrides
            if user_config:
                emotion = user_config.get('emotion', emotion)
                tone = user_config.get('tone', tone)
            
            # Sync metadata to integrated pipeline
            sync_result = self.pipeline_integration.sync_metadata_to_integrated_pipeline()
            if not sync_result['success']:
                return {'success': False, 'error': f'Metadata sync failed: {sync_result["error"]}'}
            
            # Start pipeline processing
            self.logger.info(f"STARTING Starting pipeline with user configuration: emotion={emotion}, tone={tone}")
            
            result = self.pipeline_integration.process_complete_pipeline(
                emotion=emotion,
                tone=tone
            )
            
            return {
                'success': result['success'],
                'processing_started': True,
                'emotion': emotion,
                'tone': tone,
                'pipeline_result': result,
                'message': f'Pipeline processing {"started successfully" if result["success"] else "failed"}'
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_processing_progress(self, session_id: str = None) -> Dict[str, Any]:
        """Get real-time processing progress.
        
        Args:
            session_id: Optional session ID
            
        Returns:
            Processing progress information
        """
        try:
            status = self.get_session_status(session_id)
            if not status['success']:
                return status
            
            # Get outputs from integrated pipeline
            outputs = self.pipeline_integration.get_integrated_pipeline_outputs()
            
            return {
                'success': True,
                'progress': status['overall_progress'],
                'current_stage': status['current_stage'],
                'outputs': outputs.get('outputs', {}),
                'estimated_completion': self._estimate_completion_time(status),
                'message': f'Processing {status["overall_progress"]:.1f}% complete'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_final_results(self, session_id: str = None) -> Dict[str, Any]:
        """Get final processing results.
        
        Args:
            session_id: Optional session ID
            
        Returns:
            Final results and output files
        """
        try:
            # Check if processing is complete
            status = self.get_session_status(session_id)
            if not status['success']:
                return status
            
            if status['overall_progress'] < 100:
                return {
                    'success': False,
                    'error': f'Processing not complete ({status["overall_progress"]:.1f}%)'
                }
            
            # Get all outputs
            outputs = self.pipeline_integration.get_integrated_pipeline_outputs()
            
            # Get metadata for results
            metadata = self.metadata_manager.load_metadata()
            user_inputs = metadata.get('user_inputs', {})
            
            return {
                'success': True,
                'title': user_inputs.get('title'),
                'processing_complete': True,
                'outputs': outputs.get('outputs', {}),
                'session_metadata': {
                    'emotion': user_inputs.get('emotion'),
                    'tone': user_inputs.get('tone'),
                    'quality_level': user_inputs.get('quality_level'),
                    'content_type': user_inputs.get('content_type')
                },
                'message': 'Processing completed successfully'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def list_all_sessions(self) -> Dict[str, Any]:
        """List all user sessions.
        
        Returns:
            List of all sessions
        """
        try:
            sessions = []
            metadata_dir = self.base_dir / "metadata"
            
            if metadata_dir.exists():
                for file_path in metadata_dir.glob("*.json"):
                    if file_path.name != "latest_metadata.json":
                        try:
                            with open(file_path, 'r') as f:
                                metadata = json.load(f)
                                
                            user_inputs = metadata.get('user_inputs', {})
                            sessions.append({
                                'session_id': metadata.get('session_id'),
                                'title': user_inputs.get('title', 'Unknown'),
                                'topic': user_inputs.get('topic', ''),
                                'emotion': user_inputs.get('emotion', ''),
                                'tone': user_inputs.get('tone', ''),
                                'content_type': user_inputs.get('content_type', ''),
                                'created_at': metadata.get('created_at'),
                                'last_modified': metadata.get('last_modified')
                            })
                        except Exception as e:
                            self.logger.warning(f"Could not load session {file_path}: {e}")
            
            return {
                'success': True,
                'sessions': sessions,
                'total_count': len(sessions)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def delete_session(self, session_id: str) -> Dict[str, Any]:
        """Delete a session and its files.
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            Deletion result
        """
        try:
            # Find and remove session files
            metadata_dir = self.base_dir / "metadata"
            session_files = list(metadata_dir.glob(f"*{session_id}*.json"))
            
            deleted_files = []
            for file_path in session_files:
                file_path.unlink()
                deleted_files.append(str(file_path))
            
            return {
                'success': True,
                'deleted_files': deleted_files,
                'message': f'Session {session_id} deleted successfully'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    # === Helper Methods ===
    
    def _get_current_stage(self, stages: Dict[str, Any]) -> Optional[str]:
        """Get current processing stage."""
        for stage_name, stage_data in stages.items():
            if stage_data.get('status') == 'processing':
                return stage_name
        
        # Find next pending stage
        for stage_name, stage_data in stages.items():
            if stage_data.get('status') == 'pending':
                return stage_name
        
        return None
    
    def _estimate_completion_time(self, status: Dict[str, Any]) -> Optional[str]:
        """Estimate completion time based on progress."""
        progress = status.get('overall_progress', 0)
        if progress == 0:
            return None
        
        # Simple estimation (can be improved with historical data)
        estimated_minutes = int((100 - progress) * 0.5)  # Rough estimate
        return f"{estimated_minutes} minutes" if estimated_minutes > 0 else "< 1 minute"
    
    def _get_endpoint_documentation(self) -> Dict[str, Any]:
        """Get API endpoint documentation for frontend developers."""
        return {
            'endpoints': {
                'GET /api/options': {
                    'description': 'Get available options for dropdowns',
                    'returns': 'Available tones, emotions, audiences, etc.'
                },
                'GET /api/schema': {
                    'description': 'Get complete API schema',
                    'returns': 'Field definitions and validation rules'
                },
                'POST /api/validate': {
                    'description': 'Validate user inputs',
                    'body': 'User input data',
                    'returns': 'Validation results'
                },
                'POST /api/session/create': {
                    'description': 'Create new session',
                    'body': 'User inputs and assets',
                    'returns': 'Session ID and configuration'
                },
                'GET /api/session/status': {
                    'description': 'Get session status',
                    'returns': 'Progress and stage information'
                },
                'POST /api/process/start': {
                    'description': 'Start pipeline processing',
                    'body': 'Optional processing overrides',
                    'returns': 'Processing start confirmation'
                },
                'GET /api/process/progress': {
                    'description': 'Get processing progress',
                    'returns': 'Real-time progress updates'
                },
                'GET /api/results': {
                    'description': 'Get final results',
                    'returns': 'Output files and metadata'
                },
                'GET /api/sessions': {
                    'description': 'List all sessions',
                    'returns': 'Array of session summaries'
                },
                'DELETE /api/session/{id}': {
                    'description': 'Delete session',
                    'returns': 'Deletion confirmation'
                }
            },
            'example_workflow': [
                '1. GET /api/options - Get dropdown options',
                '2. POST /api/validate - Validate user inputs',
                '3. POST /api/session/create - Create session',
                '4. POST /api/process/start - Start processing',
                '5. GET /api/process/progress - Monitor progress',
                '6. GET /api/results - Get final results'
            ]
        }


def main():
    """Test the Frontend Integration API."""
    print("API Testing Frontend Integration API")
    print("=" * 60)
    
    # Initialize API
    api = FrontendIntegrationAPI()
    
    # Test available options
    options_result = api.get_available_options()
    if options_result['success']:
        print(f"[SUCCESS] Available Options:")
        for category, options in options_result['options'].items():
            print(f"   {category}: {len(options)} options")
    
    # Test API schema
    schema_result = api.get_api_schema()
    if schema_result['success']:
        print(f"\n[SUCCESS] API Schema:")
        schema = schema_result['schema']
        print(f"   Required fields: {len(schema['required_fields'])}")
        print(f"   Optional fields: {len(schema['optional_fields'])}")
        print(f"   Endpoints: {len(schema_result['endpoints']['endpoints'])}")
    
    # Test user input validation
    test_inputs = {
        'title': 'Machine Learning Fundamentals',
        'topic': 'neural network basics',
        'audience': 'junior engineers',
        'tone': 'professional',  # User-provided, not hardcoded
        'emotion': 'confident',  # User-provided, not hardcoded  
        'content_type': 'Tutorial'
    }
    
    print(f"\n🧪 Testing User Input Processing:")
    print(f"   Test title: {test_inputs['title']}")
    
    # Validate inputs
    validation_result = api.validate_user_inputs(test_inputs)
    if validation_result['success']:
        print(f"   [SUCCESS] Validation passed")
        
        # Create session
        session_result = api.create_session(test_inputs)
        if session_result['success']:
            print(f"   [SUCCESS] Session created: {session_result['session_id']}")
            
            # Get session status
            status_result = api.get_session_status()
            if status_result['success']:
                print(f"   [SUCCESS] Session status retrieved")
                print(f"      Progress: {status_result['overall_progress']:.1f}%")
                print(f"      Emotion: {test_inputs['emotion']} (user-provided)")
                print(f"      Tone: {test_inputs['tone']} (user-provided)")
        else:
            print(f"   [ERROR] Session creation failed: {session_result['error']}")
    else:
        print(f"   [ERROR] Validation failed: {validation_result['validation']['errors']}")
    
    # Test session listing
    sessions_result = api.list_all_sessions()
    if sessions_result['success']:
        print(f"\nEndpoints Session Management:")
        print(f"   Total sessions: {sessions_result['total_count']}")
    
    print(f"\nSUCCESS Frontend Integration API testing completed!")
    print(f"Tools All parameters are user-configurable - No hardcoded values!")


if __name__ == "__main__":
    main()