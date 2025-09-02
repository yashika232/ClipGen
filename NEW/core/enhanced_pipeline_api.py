#!/usr/bin/env python3
"""
Enhanced Pipeline API for Video Synthesis Pipeline
Provides comprehensive frontend integration interface for teammate development
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

# Add the core directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import available enhanced stages
from enhanced_metadata_manager import EnhancedMetadataManager
from enhanced_voice_cloning_stage import EnhancedVoiceCloningStage
from enhanced_face_processing_stage import EnhancedFaceProcessingStage
from enhanced_sadtalker_stage import EnhancedSadTalkerStage
from enhanced_video_enhancement_stage import EnhancedVideoEnhancementStage
from enhanced_manim_stage import EnhancedManimStage
from enhanced_final_assembly_stage import EnhancedFinalAssemblyStage

# Import other available modules
from enhanced_gemini_integration import EnhancedGeminiIntegration
from unified_input_handler import UnifiedInputHandler


class PipelineStage(Enum):
    """Enumeration of all pipeline stages."""
    INPUT_PROCESSING = "input_processing"
    CONTENT_GENERATION = "content_generation"
    VOICE_CLONING = "voice_cloning"
    FACE_PROCESSING = "face_processing"
    VIDEO_GENERATION = "video_generation"
    VIDEO_ENHANCEMENT = "video_enhancement"
    BACKGROUND_ANIMATION = "background_animation"
    FINAL_ASSEMBLY = "final_assembly"


class StageStatus(Enum):
    """Enumeration of stage statuses."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class UserInputs:
    """User input data structure."""
    title: str
    topic: str
    audience: str
    tone: str
    emotion: str
    content_type: str
    additional_context: Optional[str] = None


@dataclass
class UserAssets:
    """User assets data structure."""
    face_image: Optional[str] = None
    voice_sample: Optional[str] = None
    document_path: Optional[str] = None


@dataclass
class PipelineStatus:
    """Pipeline status data structure."""
    session_id: str
    current_stage: Optional[str]
    overall_progress: float
    stages: Dict[str, Dict[str, Any]]
    errors: List[str]
    warnings: List[str]


@dataclass
class ProcessingResult:
    """Processing result data structure."""
    success: bool
    stage: str
    output_path: Optional[str] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None


class EnhancedPipelineAPI:
    """Enhanced Pipeline API for frontend integration."""
    
    def __init__(self, base_dir: str = None):
        """Initialize the enhanced pipeline API.
        
        Args:
            base_dir: Base directory for the pipeline. Defaults to NEW directory.
        """
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        # Initialize metadata manager
        self.metadata_manager = EnhancedMetadataManager(str(self.base_dir))
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize all pipeline stages
        self._initialize_stages()
        
        self.logger.info("STARTING Enhanced Pipeline API initialized")
    
    def _initialize_stages(self):
        """Initialize available pipeline stages."""
        try:
            # Initialize available enhanced stages
            self.input_handler = UnifiedInputHandler(str(self.base_dir))
            self.content_generator = EnhancedGeminiIntegration(str(self.base_dir))
            self.voice_cloning_stage = EnhancedVoiceCloningStage(str(self.base_dir))
            self.face_processing_stage = EnhancedFaceProcessingStage(str(self.base_dir))
            self.sadtalker_stage = EnhancedSadTalkerStage(str(self.base_dir))
            self.video_enhancement_stage = EnhancedVideoEnhancementStage(str(self.base_dir))
            self.manim_stage = EnhancedManimStage(str(self.base_dir))
            self.final_assembly_stage = EnhancedFinalAssemblyStage(str(self.base_dir))
            
            self.logger.info("[SUCCESS] Available pipeline stages initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline stages: {e}")
            raise
    
    # === Session Management ===
    
    def create_session(self, user_inputs: UserInputs, user_assets: UserAssets = None) -> Dict[str, Any]:
        """Create a new pipeline session.
        
        Args:
            user_inputs: User input parameters
            user_assets: User asset files (optional)
            
        Returns:
            Session creation result with session_id
        """
        try:
            # Convert dataclasses to dictionaries
            inputs_dict = asdict(user_inputs)
            assets_dict = asdict(user_assets) if user_assets else {}
            
            # Create new session using input handler
            session_result = self.input_handler.create_session_with_assets(
                inputs_dict, assets_dict
            )
            
            if session_result['success']:
                return {
                    'success': True,
                    'session_id': session_result['session_id'],
                    'created_at': datetime.now().isoformat(),
                    'message': 'Session created successfully'
                }
            else:
                return {
                    'success': False,
                    'error': session_result.get('error', 'Session creation failed')
                }
                
        except Exception as e:
            self.logger.error(f"Session creation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_session_status(self, session_id: str = None) -> PipelineStatus:
        """Get current session status.
        
        Args:
            session_id: Optional session ID (uses current if not provided)
            
        Returns:
            Pipeline status information
        """
        try:
            # Load metadata
            metadata = self.metadata_manager.load_metadata()
            if not metadata:
                raise ValueError("No active session found")
            
            # Calculate overall progress
            total_stages = len(PipelineStage)
            completed_stages = 0
            current_stage = None
            errors = []
            warnings = []
            
            stages_info = {}
            
            for stage in PipelineStage:
                stage_name = stage.value
                stage_status = self.metadata_manager.get_stage_status(stage_name)
                
                if stage_status:
                    status = stage_status.get('status', 'pending')
                    stages_info[stage_name] = {
                        'status': status,
                        'timestamps': stage_status.get('timestamps', {}),
                        'processing_data': stage_status.get('processing_data', {}),
                        'input_paths': stage_status.get('input_paths', {}),
                        'output_paths': stage_status.get('output_paths', {})
                    }
                    
                    if status == 'completed':
                        completed_stages += 1
                    elif status == 'processing':
                        current_stage = stage_name
                    elif status == 'failed':
                        error_info = stage_status.get('processing_data', {})
                        error_detail = error_info.get('error_details', f"{stage_name} failed")
                        errors.append(error_detail)
                else:
                    stages_info[stage_name] = {
                        'status': 'pending',
                        'timestamps': {},
                        'processing_data': {},
                        'input_paths': {},
                        'output_paths': {}
                    }
            
            overall_progress = (completed_stages / total_stages) * 100
            
            return PipelineStatus(
                session_id=metadata['session_id'],
                current_stage=current_stage,
                overall_progress=overall_progress,
                stages=stages_info,
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get session status: {e}")
            return PipelineStatus(
                session_id="unknown",
                current_stage=None,
                overall_progress=0.0,
                stages={},
                errors=[str(e)],
                warnings=[]
            )
    
    def list_sessions(self) -> Dict[str, Any]:
        """List all available sessions.
        
        Returns:
            Dictionary with session information
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
                                sessions.append({
                                    'session_id': metadata.get('session_id'),
                                    'created_at': metadata.get('created_at'),
                                    'last_modified': metadata.get('last_modified'),
                                    'title': metadata.get('user_inputs', {}).get('title', 'Unknown'),
                                    'status': self._get_session_summary_status(metadata)
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
    
    def _get_session_summary_status(self, metadata: Dict[str, Any]) -> str:
        """Get summary status for a session."""
        stages = metadata.get('pipeline_stages', {})
        if any(stage.get('status') == 'failed' for stage in stages.values()):
            return 'failed'
        elif any(stage.get('status') == 'processing' for stage in stages.values()):
            return 'processing'
        elif all(stage.get('status') == 'completed' for stage in stages.values()):
            return 'completed'
        else:
            return 'pending'
    
    # === Individual Stage Processing ===
    
    def process_content_generation(self) -> ProcessingResult:
        """Process content generation stage."""
        try:
            start_time = time.time()
            result = self.content_generator.process_enhanced_content_generation()
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                success=result['success'],
                stage='content_generation',
                processing_time=processing_time,
                error=result.get('error'),
                additional_data={
                    'generated_script': result.get('generated_script'),
                    'manim_script': result.get('manim_script'),
                    'script_length': result.get('script_length', 0)
                }
            )
            
        except Exception as e:
            return ProcessingResult(success=False, stage='content_generation', error=str(e))
    
    def process_voice_cloning(self) -> ProcessingResult:
        """Process voice cloning stage."""
        try:
            start_time = time.time()
            result = self.voice_cloning_stage.process_voice_cloning()
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                success=result['success'],
                stage='voice_cloning',
                output_path=result.get('output_path'),
                processing_time=processing_time,
                error=result.get('error'),
                additional_data={
                    'chunks': result.get('chunks', []),
                    'voice_duration': result.get('voice_duration', 0)
                }
            )
            
        except Exception as e:
            return ProcessingResult(success=False, stage='voice_cloning', error=str(e))
    
    def process_face_processing(self) -> ProcessingResult:
        """Process face processing stage."""
        try:
            start_time = time.time()
            result = self.face_processing_stage.process_face_processing()
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                success=result['success'],
                stage='face_processing',
                output_path=result.get('output_path'),
                processing_time=processing_time,
                error=result.get('error'),
                additional_data={
                    'face_quality': result.get('face_quality', {}),
                    'processing_method': result.get('processing_method')
                }
            )
            
        except Exception as e:
            return ProcessingResult(success=False, stage='face_processing', error=str(e))
    
    def process_video_generation(self) -> ProcessingResult:
        """Process video generation stage."""
        try:
            start_time = time.time()
            result = self.sadtalker_stage.process_video_generation()
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                success=result['success'],
                stage='video_generation',
                output_path=result.get('output_path'),
                processing_time=processing_time,
                error=result.get('error'),
                additional_data={
                    'chunks': result.get('chunks', []),
                    'synthesis_method': result.get('synthesis_method')
                }
            )
            
        except Exception as e:
            return ProcessingResult(success=False, stage='video_generation', error=str(e))
    
    def process_video_enhancement(self) -> ProcessingResult:
        """Process video enhancement stage."""
        try:
            start_time = time.time()
            result = self.video_enhancement_stage.process_video_enhancement()
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                success=result['success'],
                stage='video_enhancement',
                output_path=result.get('output_path'),
                processing_time=processing_time,
                error=result.get('error'),
                additional_data={
                    'enhancement_method': result.get('enhancement_method'),
                    'enhancement_stages': result.get('enhancement_stages', [])
                }
            )
            
        except Exception as e:
            return ProcessingResult(success=False, stage='video_enhancement', error=str(e))
    
    def process_background_animation(self) -> ProcessingResult:
        """Process background animation stage."""
        try:
            start_time = time.time()
            result = self.manim_stage.process_background_animation()
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                success=result['success'],
                stage='background_animation',
                output_path=result.get('output_path'),
                processing_time=processing_time,
                error=result.get('error'),
                additional_data={
                    'animation_duration': result.get('animation_duration', 0),
                    'animation_method': result.get('animation_method')
                }
            )
            
        except Exception as e:
            return ProcessingResult(success=False, stage='background_animation', error=str(e))
    
    def process_final_assembly(self) -> ProcessingResult:
        """Process final assembly stage."""
        try:
            start_time = time.time()
            result = self.final_assembly_stage.process_final_assembly()
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                success=result['success'],
                stage='final_assembly',
                output_path=result.get('output_path'),
                processing_time=processing_time,
                error=result.get('error'),
                additional_data={
                    'output_formats': result.get('output_formats', []),
                    'assembly_method': result.get('assembly_method')
                }
            )
            
        except Exception as e:
            return ProcessingResult(success=False, stage='final_assembly', error=str(e))
    
    # === Full Pipeline Processing ===
    
    def process_full_pipeline(self, skip_failed: bool = False) -> Dict[str, Any]:
        """Process the complete pipeline from start to finish.
        
        Args:
            skip_failed: Whether to skip failed stages and continue
            
        Returns:
            Complete pipeline processing results
        """
        try:
            pipeline_start = time.time()
            results = {}
            
            # Define pipeline order
            pipeline_stages = [
                ('content_generation', self.process_content_generation),
                ('voice_cloning', self.process_voice_cloning),
                ('face_processing', self.process_face_processing),
                ('video_generation', self.process_video_generation),
                ('video_enhancement', self.process_video_enhancement),
                ('background_animation', self.process_background_animation),
                ('final_assembly', self.process_final_assembly)
            ]
            
            for stage_name, stage_method in pipeline_stages:
                self.logger.info(f"Target: Processing {stage_name}...")
                
                result = stage_method()
                results[stage_name] = asdict(result)
                
                if not result.success:
                    self.logger.error(f"[ERROR] Stage {stage_name} failed: {result.error}")
                    if not skip_failed:
                        break
                else:
                    self.logger.info(f"[SUCCESS] Stage {stage_name} completed successfully")
            
            total_time = time.time() - pipeline_start
            
            # Calculate success metrics
            total_stages = len(pipeline_stages)
            successful_stages = sum(1 for result in results.values() if result['success'])
            success_rate = (successful_stages / total_stages) * 100
            
            return {
                'success': success_rate == 100,
                'total_processing_time': total_time,
                'stages_processed': len(results),
                'stages_successful': successful_stages,
                'success_rate': success_rate,
                'results': results,
                'final_output': results.get('final_assembly', {}).get('output_path')
            }
            
        except Exception as e:
            self.logger.error(f"Full pipeline processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'results': results if 'results' in locals() else {}
            }
    
    # === Utility Methods ===
    
    def validate_stage_prerequisites(self, stage: PipelineStage) -> Dict[str, Any]:
        """Validate prerequisites for a specific stage.
        
        Args:
            stage: Pipeline stage to validate
            
        Returns:
            Validation results
        """
        try:
            stage_name = stage.value
            
            if stage == PipelineStage.CONTENT_GENERATION:
                return self.content_generator.validate_enhanced_content_generation_prerequisites()
            elif stage == PipelineStage.VOICE_CLONING:
                return self.voice_cloning_stage.validate_voice_cloning_prerequisites()
            elif stage == PipelineStage.FACE_PROCESSING:
                return self.face_processing_stage.validate_face_processing_prerequisites()
            elif stage == PipelineStage.VIDEO_GENERATION:
                return self.sadtalker_stage.validate_video_generation_prerequisites()
            elif stage == PipelineStage.VIDEO_ENHANCEMENT:
                return self.video_enhancement_stage.validate_video_enhancement_prerequisites()
            elif stage == PipelineStage.BACKGROUND_ANIMATION:
                return self.manim_stage.validate_background_animation_prerequisites()
            elif stage == PipelineStage.FINAL_ASSEMBLY:
                return self.final_assembly_stage.validate_final_assembly_prerequisites()
            else:
                return {'valid': False, 'errors': [f'Unknown stage: {stage_name}']}
                
        except Exception as e:
            return {'valid': False, 'errors': [str(e)]}
    
    def get_stage_outputs(self, stage: PipelineStage) -> Dict[str, Any]:
        """Get outputs from a specific completed stage.
        
        Args:
            stage: Pipeline stage to get outputs from
            
        Returns:
            Stage output information
        """
        try:
            stage_name = stage.value
            stage_status = self.metadata_manager.get_stage_status(stage_name)
            
            if not stage_status:
                return {'success': False, 'error': f'Stage {stage_name} not found'}
            
            if stage_status.get('status') != 'completed':
                return {'success': False, 'error': f'Stage {stage_name} not completed'}
            
            return {
                'success': True,
                'stage': stage_name,
                'output_paths': stage_status.get('output_paths', {}),
                'processing_data': stage_status.get('processing_data', {}),
                'timestamps': stage_status.get('timestamps', {})
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def cleanup_session(self, session_id: str = None) -> Dict[str, Any]:
        """Clean up session temporary files.
        
        Args:
            session_id: Session ID to clean up (uses current if not provided)
            
        Returns:
            Cleanup results
        """
        try:
            # This would implement cleanup logic for temporary files
            # For now, just return success
            return {
                'success': True,
                'message': 'Session cleanup completed',
                'cleaned_items': []
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def export_session_data(self, session_id: str = None, format: str = 'json') -> Dict[str, Any]:
        """Export session data for external use.
        
        Args:
            session_id: Session ID to export (uses current if not provided)
            format: Export format ('json', 'summary')
            
        Returns:
            Exported session data
        """
        try:
            metadata = self.metadata_manager.load_metadata()
            if not metadata:
                return {'success': False, 'error': 'No session data found'}
            
            if format == 'json':
                return {
                    'success': True,
                    'format': 'json',
                    'data': metadata
                }
            elif format == 'summary':
                # Create a summarized version
                summary = {
                    'session_id': metadata.get('session_id'),
                    'created_at': metadata.get('created_at'),
                    'user_inputs': metadata.get('user_inputs', {}),
                    'pipeline_summary': {}
                }
                
                stages = metadata.get('pipeline_stages', {})
                for stage_name, stage_data in stages.items():
                    summary['pipeline_summary'][stage_name] = {
                        'status': stage_data.get('status'),
                        'processing_time': stage_data.get('timestamps', {}).get('processing_time', 0)
                    }
                
                return {
                    'success': True,
                    'format': 'summary',
                    'data': summary
                }
            else:
                return {'success': False, 'error': f'Unsupported format: {format}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}


def main():
    """Test the Enhanced Pipeline API."""
    print("🧪 Testing Enhanced Pipeline API")
    print("=" * 50)
    
    # Initialize API
    api = EnhancedPipelineAPI()
    
    # Test session status
    status = api.get_session_status()
    print(f"Status: Session Status:")
    print(f"   Session ID: {status.session_id}")
    print(f"   Overall Progress: {status.overall_progress:.1f}%")
    print(f"   Current Stage: {status.current_stage}")
    print(f"   Errors: {len(status.errors)}")
    print(f"   Warnings: {len(status.warnings)}")
    
    # Test stage validation
    print(f"\nSearch Testing Stage Prerequisites:")
    for stage in PipelineStage:
        validation = api.validate_stage_prerequisites(stage)
        status_icon = "[SUCCESS]" if validation['valid'] else "[ERROR]"
        error_count = len(validation.get('errors', []))
        status_text = 'Valid' if validation['valid'] else f'Invalid - {error_count} errors'
        print(f"   {status_icon} {stage.value}: {status_text}")
    
    print(f"\nSUCCESS Enhanced Pipeline API testing completed!")


if __name__ == "__main__":
    main()