#!/usr/bin/env python3
"""
Enhanced SadTalker Stage for Video Synthesis Pipeline
Integrates SadTalker with metadata-driven architecture for talking head video generation
"""

import os
import sys
import subprocess
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import json

# Add the core directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_metadata_manager import EnhancedMetadataManager

# Try to import existing SadTalker components
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "INTEGRATED_PIPELINE" / "src"))
    from sadtalker_stage import SadTalkerStage
    SADTALKER_AVAILABLE = True
except ImportError:
    SADTALKER_AVAILABLE = False

# Try to import chunking utilities
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "useless_files" / "experimental_scripts"))
    from xtts_clean_chunker import XTTSCleanChunker
    from sync_video_concatenator import SyncVideoConcatenator
    CHUNKING_AVAILABLE = True
except ImportError:
    CHUNKING_AVAILABLE = False


class EnhancedSadTalkerStage:
    """Enhanced SadTalker stage with metadata-driven architecture."""
    
    def __init__(self, base_dir: str = None):
        """Initialize enhanced SadTalker stage.
        
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
        
        # Output directories
        self.video_output_dir = self.base_dir / "processed" / "video_chunks"
        self.video_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.chunks_dir = self.video_output_dir / "chunks"
        self.chunks_dir.mkdir(exist_ok=True)
        
        # Try to initialize SadTalker stage
        self.sadtalker_stage = None
        if SADTALKER_AVAILABLE:
            try:
                self.sadtalker_stage = SadTalkerStage()
                self.logger.info("[SUCCESS] SadTalker stage initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize SadTalker stage: {e}")
                self.sadtalker_stage = None
        
        # Try to initialize chunking utilities
        self.audio_chunker = None
        self.video_concatenator = None
        if CHUNKING_AVAILABLE:
            try:
                self.audio_chunker = XTTSCleanChunker()
                self.video_concatenator = SyncVideoConcatenator()
                self.logger.info("[SUCCESS] Chunking utilities initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize chunking utilities: {e}")
        
        # SadTalker configuration
        self.sadtalker_config = {
            'chunk_duration_threshold': 60,  # seconds
            'max_chunk_duration': 15,        # seconds per chunk
            'expression_scale_base': 1.0,
            'animation_strength_base': 1.0,
            'quality_preset': 'high',
            'fps': 25
        }
        
        self.logger.info("STARTING Enhanced SadTalker Stage initialized")
    
    def process_video_generation(self) -> Dict[str, Any]:
        """Process video generation using metadata-driven approach.
        
        Returns:
            Dictionary with processing results
        """
        try:
            # Update stage status to processing
            self.metadata_manager.update_stage_status(
                "video_generation", 
                "processing",
                {"prerequisites": "voice_cloning+face_processing"}
            )
            
            # Load current session metadata
            metadata = self.metadata_manager.load_metadata()
            if not metadata:
                raise ValueError("No active session found")
            
            # Validate prerequisites
            validation_result = self._validate_prerequisites(metadata)
            if not validation_result['valid']:
                raise ValueError(f"Prerequisites not met: {validation_result['errors']}")
            
            # Get processed inputs from previous stages
            voice_cloning_stage = self.metadata_manager.get_stage_status("voice_cloning")
            face_processing_stage = self.metadata_manager.get_stage_status("face_processing")
            
            voice_output_path = voice_cloning_stage['output_paths']['synthesized_voice']
            face_output_path = face_processing_stage['output_paths']['processed_face']
            
            # Convert to absolute paths
            voice_full_path = self.base_dir / voice_output_path
            face_full_path = self.base_dir / face_output_path
            
            # Get user preferences
            user_inputs = metadata.get('user_inputs', {})
            tone = user_inputs.get('tone', 'professional')
            emotion = user_inputs.get('emotion', 'confident')
            content_type = user_inputs.get('content_type', 'Short-Form Video Reel')
            
            self.logger.info(f"Target: Processing video generation:")
            self.logger.info(f"   Voice: {voice_output_path}")
            self.logger.info(f"   Face: {face_output_path}")
            self.logger.info(f"   Tone: {tone}, Emotion: {emotion}")
            self.logger.info(f"   Content type: {content_type}")
            
            processing_start = time.time()
            
            # Determine processing strategy based on audio duration
            pipeline_config = metadata.get('pipeline_config', {})
            auto_chunking = pipeline_config.get('auto_chunking', True)
            
            if auto_chunking and self._should_use_chunking(str(voice_full_path)):
                # Process with chunking for longer content
                result = self._process_chunked_video_generation(
                    str(face_full_path), str(voice_full_path), tone, emotion, content_type
                )
            else:
                # Process as single file for shorter content
                result = self._process_single_video_generation(
                    str(face_full_path), str(voice_full_path), tone, emotion, content_type
                )
            
            processing_time = time.time() - processing_start
            
            if result['success']:
                # Get relative path for metadata
                output_path = Path(result['output_path'])
                if output_path.is_absolute():
                    try:
                        relative_path = output_path.relative_to(self.base_dir)
                        result['output_path'] = str(relative_path)
                    except ValueError:
                        pass  # Keep absolute if outside base directory
                
                # Prepare comprehensive processing data
                processing_data = {
                    "generated_video": result['output_path'],
                    "chunks": result.get('chunks', []),
                    "processing_duration": processing_time,
                    "synthesis_method": result.get('synthesis_method', 'single'),
                    "sadtalker_params": result.get('sadtalker_params', {}),
                    "video_config": {
                        "tone": tone,
                        "emotion": emotion,
                        "content_type": content_type,
                        "fps": self.sadtalker_config['fps']
                    }
                }
                
                # Update metadata with successful results
                self.metadata_manager.update_stage_status(
                    "video_generation",
                    "completed",
                    input_paths={
                        "input_voice": voice_output_path,
                        "input_face": face_output_path
                    },
                    output_paths={"generated_video": result['output_path']},
                    processing_data=processing_data
                )
                
                self.logger.info(f"[SUCCESS] Video generation completed successfully in {processing_time:.1f}s")
                self.logger.info(f"   Output: {result['output_path']}")
                
                return {
                    'success': True,
                    'output_path': result['output_path'],
                    'chunks': result.get('chunks', []),
                    'processing_time': processing_time,
                    'stage_updated': True
                }
            else:
                # Update metadata with failure
                error_data = {
                    "processing_duration": processing_time,
                    "error_details": result.get('error', 'Unknown error')
                }
                
                self.metadata_manager.update_stage_status(
                    "video_generation",
                    "failed",
                    input_paths={
                        "input_voice": voice_output_path,
                        "input_face": face_output_path
                    },
                    processing_data=error_data,
                    error_info={"error": result.get('error', 'Unknown error')}
                )
                
                self.logger.error(f"[ERROR] Video generation failed: {result.get('error')}")
                return result
        
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"[ERROR] Video generation error: {error_msg}")
            
            # Update metadata with error
            self.metadata_manager.update_stage_status(
                "video_generation",
                "failed",
                error_info={"error": error_msg}
            )
            
            return {
                'success': False,
                'error': error_msg,
                'stage_updated': True
            }
    
    def _validate_prerequisites(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that video generation prerequisites are met.
        
        Args:
            metadata: Session metadata
            
        Returns:
            Validation results
        """
        errors = []
        warnings = []
        
        # Check voice cloning completion
        voice_stage = self.metadata_manager.get_stage_status("voice_cloning")
        if not voice_stage or voice_stage.get("status") != "completed":
            errors.append("Voice cloning stage not completed")
        else:
            voice_output = voice_stage.get("output_paths", {}).get("synthesized_voice")
            if not voice_output:
                errors.append("No voice output from voice cloning stage")
            else:
                voice_path = self.base_dir / voice_output
                if not voice_path.exists():
                    errors.append(f"Voice file not found: {voice_output}")
        
        # Check face processing completion
        face_stage = self.metadata_manager.get_stage_status("face_processing")
        if not face_stage or face_stage.get("status") != "completed":
            errors.append("Face processing stage not completed")
        else:
            face_output = face_stage.get("output_paths", {}).get("processed_face")
            if not face_output:
                errors.append("No face output from face processing stage")
            else:
                face_path = self.base_dir / face_output
                if not face_path.exists():
                    errors.append(f"Face file not found: {face_output}")
        
        # Check SadTalker availability
        if not self.sadtalker_stage:
            warnings.append("SadTalker not available - will use fallback method")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def _should_use_chunking(self, audio_path: str) -> bool:
        """Determine if chunking should be used based on audio duration.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            True if chunking is recommended
        """
        try:
            # Get audio duration using ffprobe
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                duration = float(result.stdout.strip())
                self.logger.info(f"   Audio duration: {duration:.1f} seconds")
                return duration > self.sadtalker_config['chunk_duration_threshold']
            else:
                self.logger.warning("Could not determine audio duration, defaulting to chunking")
                return True
                
        except Exception as e:
            self.logger.warning(f"Error checking audio duration: {e}, defaulting to chunking")
            return True
    
    def _process_single_video_generation(self, face_path: str, voice_path: str,
                                       tone: str, emotion: str, content_type: str) -> Dict[str, Any]:
        """Process video generation as a single file.
        
        Args:
            face_path: Path to processed face image
            voice_path: Path to synthesized voice
            tone: Voice tone parameter
            emotion: Voice emotion parameter
            content_type: Content type
            
        Returns:
            Processing results
        """
        try:
            timestamp = int(datetime.now().timestamp())
            output_filename = f"video_single_{timestamp}.mp4"
            output_path = self.video_output_dir / output_filename
            
            self.logger.info("VIDEO PIPELINE Generating single talking head video...")
            
            # Get SadTalker parameters
            sadtalker_params = self._get_sadtalker_parameters(tone, emotion, content_type)
            
            # Generate video
            generation_result = self._generate_video_with_sadtalker(
                face_path, voice_path, str(output_path), sadtalker_params
            )
            
            if generation_result['success']:
                return {
                    'success': True,
                    'output_path': str(output_path),
                    'synthesis_method': 'single',
                    'sadtalker_params': sadtalker_params
                }
            else:
                return generation_result
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _process_chunked_video_generation(self, face_path: str, voice_path: str,
                                        tone: str, emotion: str, content_type: str) -> Dict[str, Any]:
        """Process video generation with audio chunking.
        
        Args:
            face_path: Path to processed face image
            voice_path: Path to synthesized voice
            tone: Voice tone parameter
            emotion: Voice emotion parameter
            content_type: Content type
            
        Returns:
            Processing results
        """
        try:
            # Chunk audio first
            audio_chunks = self._chunk_audio_for_processing(voice_path)
            if not audio_chunks:
                return {'success': False, 'error': 'Failed to create audio chunks'}
            
            timestamp = int(datetime.now().timestamp())
            video_chunks = []
            
            self.logger.info(f"VIDEO PIPELINE Generating {len(audio_chunks)} video chunks...")
            
            # Get SadTalker parameters
            sadtalker_params = self._get_sadtalker_parameters(tone, emotion, content_type)
            
            # Process each chunk
            for i, audio_chunk in enumerate(audio_chunks):
                chunk_filename = f"video_chunk_{timestamp}_{i:03d}.mp4"
                chunk_output = self.chunks_dir / chunk_filename
                
                self.logger.info(f"   Processing chunk {i+1}/{len(audio_chunks)}")
                
                generation_result = self._generate_video_with_sadtalker(
                    face_path, audio_chunk, str(chunk_output), sadtalker_params
                )
                
                if generation_result['success']:
                    video_chunks.append(str(chunk_output))
                else:
                    return {
                        'success': False,
                        'error': f"Chunk {i+1} generation failed: {generation_result.get('error')}"
                    }
            
            # Concatenate video chunks
            final_filename = f"video_combined_{timestamp}.mp4"
            final_output = self.video_output_dir / final_filename
            
            concat_result = self._concatenate_video_chunks(video_chunks, final_output)
            
            if concat_result['success']:
                return {
                    'success': True,
                    'output_path': str(final_output),
                    'chunks': [str(Path(chunk).relative_to(self.base_dir)) for chunk in video_chunks],
                    'synthesis_method': 'chunked',
                    'sadtalker_params': sadtalker_params
                }
            else:
                return concat_result
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _chunk_audio_for_processing(self, audio_path: str) -> List[str]:
        """Chunk audio file for video processing.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of audio chunk paths
        """
        try:
            if self.audio_chunker:
                # Use advanced chunker if available
                chunks_audio_dir = self.chunks_dir / "audio"
                chunks_audio_dir.mkdir(exist_ok=True)
                
                chunk_paths = self.audio_chunker.create_clean_chunks(
                    audio_path, str(chunks_audio_dir)
                )
                
                self.logger.info(f"   Created {len(chunk_paths)} clean audio chunks")
                return chunk_paths
            else:
                # Use basic FFmpeg chunking
                return self._basic_audio_chunking(audio_path)
                
        except Exception as e:
            self.logger.error(f"Audio chunking failed: {e}")
            return []
    
    def _basic_audio_chunking(self, audio_path: str) -> List[str]:
        """Basic audio chunking using FFmpeg.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of audio chunk paths
        """
        try:
            chunks_audio_dir = self.chunks_dir / "audio"
            chunks_audio_dir.mkdir(exist_ok=True)
            
            # Get total duration
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                raise ValueError("Could not get audio duration")
            
            total_duration = float(result.stdout.strip())
            chunk_duration = self.sadtalker_config['max_chunk_duration']
            num_chunks = int(total_duration / chunk_duration) + 1
            
            chunk_paths = []
            timestamp = int(datetime.now().timestamp())
            
            for i in range(num_chunks):
                start_time = i * chunk_duration
                if start_time >= total_duration:
                    break
                
                chunk_filename = f"audio_chunk_{timestamp}_{i:03d}.wav"
                chunk_path = chunks_audio_dir / chunk_filename
                
                cmd = [
                    'ffmpeg', '-y', '-i', audio_path,
                    '-ss', str(start_time), '-t', str(chunk_duration),
                    '-c', 'copy', str(chunk_path)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0 and chunk_path.exists():
                    chunk_paths.append(str(chunk_path))
                else:
                    self.logger.warning(f"Failed to create audio chunk {i}")
            
            self.logger.info(f"   Created {len(chunk_paths)} basic audio chunks")
            return chunk_paths
            
        except Exception as e:
            self.logger.error(f"Basic audio chunking failed: {e}")
            return []
    
    def _get_sadtalker_parameters(self, tone: str, emotion: str, content_type: str) -> Dict[str, Any]:
        """Get SadTalker parameters based on tone, emotion, and content type.
        
        Args:
            tone: Voice tone
            emotion: Voice emotion
            content_type: Content type
            
        Returns:
            SadTalker parameters
        """
        # Base parameters
        params = {
            'expression_scale': self.sadtalker_config['expression_scale_base'],
            'face_animation_strength': self.sadtalker_config['animation_strength_base'],
            'still_mode': False,
            'preprocess': 'crop',
            'quality_preset': self.sadtalker_config['quality_preset'],
            'fps': self.sadtalker_config['fps']
        }
        
        # Emotion-based adjustments
        emotion_adjustments = {
            'inspired': {'expression_scale': 1.2, 'face_animation_strength': 1.1},
            'confident': {'expression_scale': 1.3, 'face_animation_strength': 1.2},
            'curious': {'expression_scale': 0.9, 'face_animation_strength': 0.8},
            'excited': {'expression_scale': 1.4, 'face_animation_strength': 1.3},
            'calm': {'expression_scale': 0.8, 'face_animation_strength': 0.7}
        }
        
        # Tone-based adjustments
        tone_adjustments = {
            'professional': {'expression_scale': 1.0, 'face_animation_strength': 0.9},
            'friendly': {'expression_scale': 1.1, 'face_animation_strength': 1.0},
            'motivational': {'expression_scale': 1.3, 'face_animation_strength': 1.2},
            'casual': {'expression_scale': 1.2, 'face_animation_strength': 1.1}
        }
        
        # Content type adjustments
        content_adjustments = {
            'Short-Form Video Reel': {'expression_scale': 1.3, 'face_animation_strength': 1.2},
            'Full Training Module': {'expression_scale': 1.0, 'face_animation_strength': 0.9},
            'Quick Tutorial': {'expression_scale': 1.1, 'face_animation_strength': 1.0},
            'Presentation': {'expression_scale': 0.9, 'face_animation_strength': 0.8}
        }
        
        # Apply adjustments
        if emotion in emotion_adjustments:
            for key, value in emotion_adjustments[emotion].items():
                params[key] *= value
        
        if tone in tone_adjustments:
            for key, value in tone_adjustments[tone].items():
                params[key] = (params[key] + value) / 2  # Average with emotion adjustment
        
        if content_type in content_adjustments:
            for key, value in content_adjustments[content_type].items():
                params[key] = (params[key] + value) / 2  # Average with previous adjustments
        
        # Ensure values stay within reasonable bounds
        params['expression_scale'] = max(0.5, min(2.0, params['expression_scale']))
        params['face_animation_strength'] = max(0.5, min(2.0, params['face_animation_strength']))
        
        return params
    
    def _generate_video_with_sadtalker(self, face_path: str, voice_path: str,
                                     output_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate video using SadTalker with specified parameters.
        
        Args:
            face_path: Path to face image
            voice_path: Path to voice audio
            output_path: Output video path
            params: SadTalker parameters
            
        Returns:
            Generation results
        """
        try:
            if self.sadtalker_stage:
                # Use SadTalker stage
                result = self.sadtalker_stage.generate_talking_head_video(
                    face_image=face_path,
                    audio_file=voice_path,
                    output_path=output_path,
                    expression_scale=params.get('expression_scale', 1.0),
                    face_animation_strength=params.get('face_animation_strength', 1.0),
                    still_mode=params.get('still_mode', False),
                    preprocess=params.get('preprocess', 'crop')
                )
                return result
            else:
                # Use fallback subprocess method
                return self._generate_with_subprocess(face_path, voice_path, output_path, params)
                
        except Exception as e:
            return {'success': False, 'error': f"SadTalker generation failed: {str(e)}"}
    
    def _generate_with_subprocess(self, face_path: str, voice_path: str,
                                output_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback video generation using subprocess.
        
        Args:
            face_path: Path to face image
            voice_path: Path to voice audio
            output_path: Output video path
            params: SadTalker parameters
            
        Returns:
            Generation results
        """
        try:
            # Use SadTalker stage via subprocess
            sadtalker_script = self.base_dir.parent.parent / "INTEGRATED_PIPELINE" / "src" / "sadtalker_stage.py"
            
            if not sadtalker_script.exists():
                return {'success': False, 'error': 'SadTalker stage script not found'}
            
            # Build command
            cmd = [
                sys.executable, str(sadtalker_script),
                '--face-image', face_path,
                '--audio-file', voice_path,
                '--output', output_path
            ]
            
            # Add parameters
            if params.get('expression_scale', 1.0) != 1.0:
                cmd.extend(['--expression-scale', str(params['expression_scale'])])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0 and Path(output_path).exists():
                return {'success': True, 'output_path': output_path}
            else:
                return {'success': False, 'error': result.stderr or 'Subprocess failed'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _concatenate_video_chunks(self, video_chunks: List[str], output_path: Path) -> Dict[str, Any]:
        """Concatenate video chunks using available methods.
        
        Args:
            video_chunks: List of video chunk paths
            output_path: Output path for concatenated video
            
        Returns:
            Concatenation results
        """
        try:
            if self.video_concatenator:
                # Use advanced concatenator if available
                success = self.video_concatenator.concatenate_videos(
                    video_chunks, str(output_path)
                )
                
                if success:
                    self.logger.info("[SUCCESS] Video chunks concatenated with sync concatenator")
                    return {'success': True}
                else:
                    # Fallback to basic concatenation
                    return self._basic_video_concatenation(video_chunks, output_path)
            else:
                # Use basic FFmpeg concatenation
                return self._basic_video_concatenation(video_chunks, output_path)
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _basic_video_concatenation(self, video_chunks: List[str], output_path: Path) -> Dict[str, Any]:
        """Basic video concatenation using FFmpeg.
        
        Args:
            video_chunks: List of video chunk paths
            output_path: Output path for concatenated video
            
        Returns:
            Concatenation results
        """
        try:
            # Create concat file
            concat_file = output_path.with_suffix('.txt')
            with open(concat_file, 'w') as f:
                for chunk_path in video_chunks:
                    f.write(f"file '{chunk_path}'\n")
            
            # Run FFmpeg with sync-aware settings
            cmd = [
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                '-i', str(concat_file),
                '-c:v', 'libx264', '-c:a', 'aac',
                '-b:a', '128k', '-af', 'aresample=async=1000:first_pts=0',
                '-fflags', '+shortest', '-avoid_negative_ts', 'make_zero',
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Cleanup
            concat_file.unlink(missing_ok=True)
            
            if result.returncode == 0:
                self.logger.info("[SUCCESS] Video chunks concatenated with FFmpeg")
                return {'success': True}
            else:
                return {'success': False, 'error': f"FFmpeg concatenation failed: {result.stderr}"}
                
        except Exception as e:
            return {'success': False, 'error': f"Basic video concatenation failed: {str(e)}"}
    
    def get_video_generation_status(self) -> Dict[str, Any]:
        """Get current video generation stage status.
        
        Returns:
            Status information
        """
        try:
            stage_status = self.metadata_manager.get_stage_status("video_generation")
            if stage_status:
                return {
                    'success': True,
                    'stage_status': stage_status
                }
            else:
                return {
                    'success': False,
                    'error': 'No video generation stage found in metadata'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def validate_video_generation_prerequisites(self) -> Dict[str, Any]:
        """Validate that video generation can be started.
        
        Returns:
            Validation results
        """
        try:
            # Check metadata
            metadata = self.metadata_manager.load_metadata()
            if not metadata:
                return {'valid': False, 'errors': ['No active session found']}
            
            return self._validate_prerequisites(metadata)
            
        except Exception as e:
            return {
                'valid': False,
                'errors': [str(e)]
            }


def main():
    """Test the enhanced SadTalker stage."""
    print("🧪 Testing Enhanced SadTalker Stage")
    print("=" * 50)
    
    # Initialize stage
    stage = EnhancedSadTalkerStage()
    
    # Check prerequisites
    prereq_result = stage.validate_video_generation_prerequisites()
    print(f"[SUCCESS] Prerequisites check:")
    print(f"   Valid: {prereq_result['valid']}")
    if prereq_result.get('errors'):
        print(f"   Errors: {prereq_result['errors']}")
    if prereq_result.get('warnings'):
        print(f"   Warnings: {prereq_result['warnings']}")
    
    # Check current status
    status_result = stage.get_video_generation_status()
    if status_result['success']:
        stage_info = status_result['stage_status']
        print(f"[SUCCESS] Current status: {stage_info.get('status', 'unknown')}")
    else:
        print(f"[ERROR] Status check failed: {status_result['error']}")
    
    # Process video generation if prerequisites are met
    if prereq_result['valid']:
        print("\nTarget: Starting video generation...")
        result = stage.process_video_generation()
        
        if result['success']:
            print("[SUCCESS] Video generation completed successfully!")
            print(f"   Output: {result['output_path']}")
            print(f"   Processing time: {result['processing_time']:.1f}s")
            if result.get('chunks'):
                print(f"   Chunks: {len(result['chunks'])}")
        else:
            print(f"[ERROR] Video generation failed: {result['error']}")
    else:
        print("\n[ERROR] Prerequisites not met - skipping processing")
    
    print("\nSUCCESS Enhanced SadTalker Stage testing completed!")


if __name__ == "__main__":
    main()