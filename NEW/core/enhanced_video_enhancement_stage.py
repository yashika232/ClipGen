#!/usr/bin/env python3
"""
Enhanced Video Enhancement Stage for Video Synthesis Pipeline
Integrates Real-ESRGAN and CodeFormer with metadata-driven architecture for video quality enhancement
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

# Add the core directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_metadata_manager import EnhancedMetadataManager

# Try to import existing enhancement components
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "INTEGRATED_PIPELINE" / "src"))
    from realesrgan_stage import RealESRGANStage
    from codeformer_stage import CodeFormerStage
    ENHANCEMENT_AVAILABLE = True
except ImportError:
    ENHANCEMENT_AVAILABLE = False

# Try to import chunking utilities
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "useless_files" / "experimental_scripts"))
    from chunked_enhancement_processor import ChunkedEnhancementProcessor
    from final_enhanced_concatenator import FinalEnhancedConcatenator
    CHUNKING_AVAILABLE = True
except ImportError:
    CHUNKING_AVAILABLE = False


class EnhancedVideoEnhancementStage:
    """Enhanced video enhancement stage with metadata-driven architecture."""
    
    def __init__(self, base_dir: str = None):
        """Initialize enhanced video enhancement stage.
        
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
        self.enhancement_output_dir = self.base_dir / "enhanced" / "video_chunks"
        self.enhancement_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.realesrgan_dir = self.enhancement_output_dir / "realesrgan"
        self.realesrgan_dir.mkdir(exist_ok=True)
        
        self.codeformer_dir = self.enhancement_output_dir / "codeformer"
        self.codeformer_dir.mkdir(exist_ok=True)
        
        self.chunks_dir = self.enhancement_output_dir / "chunks"
        self.chunks_dir.mkdir(exist_ok=True)
        
        # Try to initialize enhancement stages
        self.realesrgan_stage = None
        self.codeformer_stage = None
        
        if ENHANCEMENT_AVAILABLE:
            try:
                self.realesrgan_stage = RealESRGANStage()
                self.logger.info("[SUCCESS] Real-ESRGAN stage initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Real-ESRGAN stage: {e}")
            
            try:
                self.codeformer_stage = CodeFormerStage()
                self.logger.info("[SUCCESS] CodeFormer stage initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize CodeFormer stage: {e}")
        
        # Try to initialize chunking utilities
        self.chunked_processor = None
        self.final_concatenator = None
        if CHUNKING_AVAILABLE:
            try:
                self.chunked_processor = ChunkedEnhancementProcessor()
                self.final_concatenator = FinalEnhancedConcatenator()
                self.logger.info("[SUCCESS] Enhancement chunking utilities initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize chunking utilities: {e}")
        
        # Enhancement configuration
        self.enhancement_config = {
            'chunk_duration_threshold': 30,  # seconds
            'max_chunk_duration': 15,        # seconds per chunk
            'realesrgan_scale_base': 2,      # base upscaling factor
            'codeformer_fidelity_base': 0.7, # base fidelity
            'quality_preset': 'high',
            'max_file_size_mb': 100          # chunk large videos above this size
        }
        
        self.logger.info("STARTING Enhanced Video Enhancement Stage initialized")
    
    def process_video_enhancement(self) -> Dict[str, Any]:
        """Process video enhancement using metadata-driven approach.
        
        Returns:
            Dictionary with processing results
        """
        try:
            # Update stage status to processing
            self.metadata_manager.update_stage_status(
                "video_enhancement", 
                "processing",
                {"prerequisites": "video_generation"}
            )
            
            # Load current session metadata
            metadata = self.metadata_manager.load_metadata()
            if not metadata:
                raise ValueError("No active session found")
            
            # Validate prerequisites
            validation_result = self._validate_prerequisites(metadata)
            if not validation_result['valid']:
                raise ValueError(f"Prerequisites not met: {validation_result['errors']}")
            
            # Get processed video from video generation stage
            video_generation_stage = self.metadata_manager.get_stage_status("video_generation")
            video_output_path = video_generation_stage['output_paths']['generated_video']
            
            # Convert to absolute path
            video_full_path = self.base_dir / video_output_path
            
            # Get user preferences
            user_inputs = metadata.get('user_inputs', {})
            tone = user_inputs.get('tone', 'professional')
            emotion = user_inputs.get('emotion', 'confident')
            content_type = user_inputs.get('content_type', 'Short-Form Video Reel')
            
            self.logger.info(f"Target: Processing video enhancement:")
            self.logger.info(f"   Input video: {video_output_path}")
            self.logger.info(f"   Tone: {tone}, Emotion: {emotion}")
            self.logger.info(f"   Content type: {content_type}")
            
            processing_start = time.time()
            
            # Determine processing strategy based on video characteristics
            pipeline_config = metadata.get('pipeline_config', {})
            auto_chunking = pipeline_config.get('auto_chunking', True)
            
            if auto_chunking and self._should_use_chunking(str(video_full_path)):
                # Process with chunking for large/long videos
                result = self._process_chunked_video_enhancement(
                    str(video_full_path), tone, emotion, content_type
                )
            else:
                # Process as single file for smaller videos
                result = self._process_single_video_enhancement(
                    str(video_full_path), tone, emotion, content_type
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
                    "enhanced_video": result['output_path'],
                    "chunks": result.get('chunks', []),
                    "processing_duration": processing_time,
                    "enhancement_method": result.get('enhancement_method', 'single'),
                    "enhancement_params": result.get('enhancement_params', {}),
                    "enhancement_stages": result.get('enhancement_stages', []),
                    "video_config": {
                        "tone": tone,
                        "emotion": emotion,
                        "content_type": content_type,
                        "quality_preset": self.enhancement_config['quality_preset']
                    }
                }
                
                # Update metadata with successful results
                self.metadata_manager.update_stage_status(
                    "video_enhancement",
                    "completed",
                    input_paths={"input_video": video_output_path},
                    output_paths={"enhanced_video": result['output_path']},
                    processing_data=processing_data
                )
                
                self.logger.info(f"[SUCCESS] Video enhancement completed successfully in {processing_time:.1f}s")
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
                    "video_enhancement",
                    "failed",
                    input_paths={"input_video": video_output_path},
                    processing_data=error_data,
                    error_info={"error": result.get('error', 'Unknown error')}
                )
                
                self.logger.error(f"[ERROR] Video enhancement failed: {result.get('error')}")
                return result
        
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"[ERROR] Video enhancement error: {error_msg}")
            
            # Update metadata with error
            self.metadata_manager.update_stage_status(
                "video_enhancement",
                "failed",
                error_info={"error": error_msg}
            )
            
            return {
                'success': False,
                'error': error_msg,
                'stage_updated': True
            }
    
    def _validate_prerequisites(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that video enhancement prerequisites are met.
        
        Args:
            metadata: Session metadata
            
        Returns:
            Validation results
        """
        errors = []
        warnings = []
        
        # Check video generation completion
        video_stage = self.metadata_manager.get_stage_status("video_generation")
        if not video_stage or video_stage.get("status") != "completed":
            errors.append("Video generation stage not completed")
        else:
            video_output = video_stage.get("output_paths", {}).get("generated_video")
            if not video_output:
                errors.append("No video output from video generation stage")
            else:
                video_path = self.base_dir / video_output
                if not video_path.exists():
                    errors.append(f"Video file not found: {video_output}")
        
        # Check enhancement components availability
        if not self.realesrgan_stage:
            warnings.append("Real-ESRGAN not available - will use fallback enhancement")
        
        if not self.codeformer_stage:
            warnings.append("CodeFormer not available - will use fallback enhancement")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def _should_use_chunking(self, video_path: str) -> bool:
        """Determine if chunking should be used based on video characteristics.
        
        Args:
            video_path: Path to video file
            
        Returns:
            True if chunking is recommended
        """
        try:
            # Get video duration and file size
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration,size',
                '-of', 'default=noprint_wrappers=1', video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                duration = 0
                size = 0
                
                for line in lines:
                    if line.startswith('duration='):
                        duration = float(line.split('=')[1])
                    elif line.startswith('size='):
                        size = int(line.split('=')[1])
                
                size_mb = size / (1024 * 1024)
                self.logger.info(f"   Video duration: {duration:.1f}s, size: {size_mb:.1f}MB")
                
                # Use chunking for long videos or large files
                return (duration > self.enhancement_config['chunk_duration_threshold'] or 
                        size_mb > self.enhancement_config['max_file_size_mb'])
            else:
                self.logger.warning("Could not determine video properties, defaulting to chunking")
                return True
                
        except Exception as e:
            self.logger.warning(f"Error checking video properties: {e}, defaulting to chunking")
            return True
    
    def _process_single_video_enhancement(self, video_path: str, tone: str, emotion: str, 
                                        content_type: str) -> Dict[str, Any]:
        """Process video enhancement as a single file.
        
        Args:
            video_path: Path to input video
            tone: Voice tone parameter
            emotion: Voice emotion parameter
            content_type: Content type
            
        Returns:
            Processing results
        """
        try:
            timestamp = int(datetime.now().timestamp())
            
            self.logger.info("Frontend Enhancing single video file...")
            
            # Get enhancement parameters
            enhancement_params = self._get_enhancement_parameters(tone, emotion, content_type)
            
            # Step 1: Real-ESRGAN enhancement
            realesrgan_output = self.realesrgan_dir / f"realesrgan_{timestamp}.mp4"
            realesrgan_result = self._enhance_with_realesrgan(
                video_path, str(realesrgan_output), enhancement_params
            )
            
            if not realesrgan_result['success']:
                return realesrgan_result
            
            self.logger.info("[SUCCESS] Real-ESRGAN enhancement completed")
            
            # Step 2: CodeFormer enhancement
            final_output = self.codeformer_dir / f"enhanced_{timestamp}.mp4"
            codeformer_result = self._enhance_with_codeformer(
                str(realesrgan_output), str(final_output), enhancement_params
            )
            
            if not codeformer_result['success']:
                return codeformer_result
            
            self.logger.info("[SUCCESS] CodeFormer enhancement completed")
            
            return {
                'success': True,
                'output_path': str(final_output),
                'enhancement_method': 'single',
                'enhancement_params': enhancement_params,
                'enhancement_stages': ['realesrgan', 'codeformer'],
                'intermediate_files': [str(realesrgan_output)]
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _process_chunked_video_enhancement(self, video_path: str, tone: str, emotion: str,
                                         content_type: str) -> Dict[str, Any]:
        """Process video enhancement with chunking.
        
        Args:
            video_path: Path to input video
            tone: Voice tone parameter
            emotion: Voice emotion parameter
            content_type: Content type
            
        Returns:
            Processing results
        """
        try:
            if self.chunked_processor and self.final_concatenator:
                # Use advanced chunked processor
                return self._process_with_advanced_chunking(video_path, tone, emotion, content_type)
            else:
                # Use manual chunking
                return self._process_with_manual_chunking(video_path, tone, emotion, content_type)
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _process_with_advanced_chunking(self, video_path: str, tone: str, emotion: str,
                                      content_type: str) -> Dict[str, Any]:
        """Process using advanced chunked enhancement processor.
        
        Args:
            video_path: Path to input video
            tone: Voice tone parameter
            emotion: Voice emotion parameter
            content_type: Content type
            
        Returns:
            Processing results
        """
        try:
            self.logger.info("Frontend Using advanced chunked enhancement processor...")
            
            # Process all chunks with advanced processor
            result = self.chunked_processor.process_all_chunks(
                [video_path], emotion=emotion, tone=tone
            )
            
            if result['success']:
                # Create final enhanced video
                final_result = self.final_concatenator.create_final_enhanced_video()
                
                if final_result['success']:
                    return {
                        'success': True,
                        'output_path': final_result['final_video'],
                        'chunks': result.get('enhanced_chunks', []),
                        'enhancement_method': 'advanced_chunked',
                        'enhancement_stages': ['realesrgan', 'codeformer']
                    }
                else:
                    return final_result
            else:
                return result
                
        except Exception as e:
            return {'success': False, 'error': f"Advanced chunking failed: {str(e)}"}
    
    def _process_with_manual_chunking(self, video_path: str, tone: str, emotion: str,
                                    content_type: str) -> Dict[str, Any]:
        """Process using manual chunking approach.
        
        Args:
            video_path: Path to input video
            tone: Voice tone parameter
            emotion: Voice emotion parameter
            content_type: Content type
            
        Returns:
            Processing results
        """
        try:
            # Chunk video first
            video_chunks = self._chunk_video_for_enhancement(video_path)
            if not video_chunks:
                return {'success': False, 'error': 'Failed to create video chunks'}
            
            timestamp = int(datetime.now().timestamp())
            enhanced_chunks = []
            
            self.logger.info(f"Frontend Enhancing {len(video_chunks)} video chunks...")
            
            # Get enhancement parameters
            enhancement_params = self._get_enhancement_parameters(tone, emotion, content_type)
            
            # Process each chunk
            for i, chunk_path in enumerate(video_chunks):
                self.logger.info(f"   Processing chunk {i+1}/{len(video_chunks)}")
                
                # Real-ESRGAN enhancement
                realesrgan_output = self.realesrgan_dir / f"chunk_realesrgan_{timestamp}_{i:03d}.mp4"
                realesrgan_result = self._enhance_with_realesrgan(
                    chunk_path, str(realesrgan_output), enhancement_params
                )
                
                if not realesrgan_result['success']:
                    return {
                        'success': False,
                        'error': f"Chunk {i+1} Real-ESRGAN failed: {realesrgan_result.get('error')}"
                    }
                
                # CodeFormer enhancement
                codeformer_output = self.codeformer_dir / f"chunk_enhanced_{timestamp}_{i:03d}.mp4"
                codeformer_result = self._enhance_with_codeformer(
                    str(realesrgan_output), str(codeformer_output), enhancement_params
                )
                
                if not codeformer_result['success']:
                    return {
                        'success': False,
                        'error': f"Chunk {i+1} CodeFormer failed: {codeformer_result.get('error')}"
                    }
                
                enhanced_chunks.append(str(codeformer_output))
            
            # Concatenate enhanced chunks
            final_filename = f"enhanced_combined_{timestamp}.mp4"
            final_output = self.enhancement_output_dir / final_filename
            
            concat_result = self._concatenate_enhanced_chunks(enhanced_chunks, final_output)
            
            if concat_result['success']:
                return {
                    'success': True,
                    'output_path': str(final_output),
                    'chunks': [str(Path(chunk).relative_to(self.base_dir)) for chunk in enhanced_chunks],
                    'enhancement_method': 'manual_chunked',
                    'enhancement_params': enhancement_params,
                    'enhancement_stages': ['realesrgan', 'codeformer']
                }
            else:
                return concat_result
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _chunk_video_for_enhancement(self, video_path: str) -> List[str]:
        """Chunk video file for enhancement processing.
        
        Args:
            video_path: Path to input video
            
        Returns:
            List of video chunk paths
        """
        try:
            chunks_video_dir = self.chunks_dir / "video"
            chunks_video_dir.mkdir(exist_ok=True)
            
            # Get total duration
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                raise ValueError("Could not get video duration")
            
            total_duration = float(result.stdout.strip())
            chunk_duration = self.enhancement_config['max_chunk_duration']
            num_chunks = int(total_duration / chunk_duration) + 1
            
            chunk_paths = []
            timestamp = int(datetime.now().timestamp())
            
            for i in range(num_chunks):
                start_time = i * chunk_duration
                if start_time >= total_duration:
                    break
                
                chunk_filename = f"video_chunk_{timestamp}_{i:03d}.mp4"
                chunk_path = chunks_video_dir / chunk_filename
                
                cmd = [
                    'ffmpeg', '-y', '-i', video_path,
                    '-ss', str(start_time), '-t', str(chunk_duration),
                    '-c', 'copy', str(chunk_path)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0 and chunk_path.exists():
                    chunk_paths.append(str(chunk_path))
                else:
                    self.logger.warning(f"Failed to create video chunk {i}")
            
            self.logger.info(f"   Created {len(chunk_paths)} video chunks")
            return chunk_paths
            
        except Exception as e:
            self.logger.error(f"Video chunking failed: {e}")
            return []
    
    def _get_enhancement_parameters(self, tone: str, emotion: str, content_type: str) -> Dict[str, Any]:
        """Get enhancement parameters based on tone, emotion, and content type.
        
        Args:
            tone: Voice tone
            emotion: Voice emotion
            content_type: Content type
            
        Returns:
            Enhancement parameters
        """
        # Base parameters
        params = {
            'realesrgan_scale': self.enhancement_config['realesrgan_scale_base'],
            'realesrgan_model': 'RealESRGAN_x2plus',
            'codeformer_fidelity': self.enhancement_config['codeformer_fidelity_base']
        }
        
        # Emotion-based adjustments
        emotion_adjustments = {
            'inspired': {'codeformer_fidelity': 0.8},
            'confident': {'codeformer_fidelity': 0.9, 'realesrgan_scale': 2},
            'curious': {'codeformer_fidelity': 0.6},
            'excited': {'codeformer_fidelity': 0.8, 'realesrgan_scale': 2},
            'calm': {'codeformer_fidelity': 0.7}
        }
        
        # Tone-based adjustments
        tone_adjustments = {
            'professional': {'realesrgan_scale': 2, 'codeformer_fidelity': 0.9},
            'friendly': {'realesrgan_scale': 2, 'codeformer_fidelity': 0.7},
            'motivational': {'realesrgan_scale': 2, 'codeformer_fidelity': 0.8},
            'casual': {'realesrgan_scale': 2, 'codeformer_fidelity': 0.6}
        }
        
        # Content type adjustments
        content_adjustments = {
            'Short-Form Video Reel': {'realesrgan_scale': 2, 'codeformer_fidelity': 0.8},
            'Full Training Module': {'realesrgan_scale': 2, 'codeformer_fidelity': 0.9},
            'Quick Tutorial': {'realesrgan_scale': 2, 'codeformer_fidelity': 0.7},
            'Presentation': {'realesrgan_scale': 2, 'codeformer_fidelity': 0.9}
        }
        
        # Apply adjustments
        if emotion in emotion_adjustments:
            params.update(emotion_adjustments[emotion])
        
        if tone in tone_adjustments:
            for key, value in tone_adjustments[tone].items():
                if key in params:
                    params[key] = (params[key] + value) / 2  # Average with emotion adjustment
                else:
                    params[key] = value
        
        if content_type in content_adjustments:
            for key, value in content_adjustments[content_type].items():
                if key in params:
                    params[key] = (params[key] + value) / 2  # Average with previous adjustments
                else:
                    params[key] = value
        
        return params
    
    def _enhance_with_realesrgan(self, input_path: str, output_path: str, 
                               params: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance video using Real-ESRGAN.
        
        Args:
            input_path: Path to input video
            output_path: Path for enhanced output
            params: Enhancement parameters
            
        Returns:
            Enhancement results
        """
        try:
            if self.realesrgan_stage:
                # Use Real-ESRGAN stage
                result = self.realesrgan_stage.upscale_video(
                    input_video=input_path,
                    output_video=output_path,
                    scale=params.get('realesrgan_scale', 2),
                    model_name=params.get('realesrgan_model', 'RealESRGAN_x2plus')
                )
                return result
            else:
                # Use fallback enhancement
                return self._enhance_with_fallback_upscaling(input_path, output_path, params)
                
        except Exception as e:
            return {'success': False, 'error': f"Real-ESRGAN enhancement failed: {str(e)}"}
    
    def _enhance_with_codeformer(self, input_path: str, output_path: str,
                               params: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance video using CodeFormer.
        
        Args:
            input_path: Path to input video
            output_path: Path for enhanced output
            params: Enhancement parameters
            
        Returns:
            Enhancement results
        """
        try:
            if self.codeformer_stage:
                # Use CodeFormer stage
                result = self.codeformer_stage.enhance_video(
                    input_video=input_path,
                    output_video=output_path,
                    fidelity_weight=params.get('codeformer_fidelity', 0.7)
                )
                return result
            else:
                # Use fallback enhancement
                return self._enhance_with_fallback_face_enhancement(input_path, output_path)
                
        except Exception as e:
            return {'success': False, 'error': f"CodeFormer enhancement failed: {str(e)}"}
    
    def _enhance_with_fallback_upscaling(self, input_path: str, output_path: str,
                                       params: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback video upscaling using FFmpeg.
        
        Args:
            input_path: Path to input video
            output_path: Path for upscaled output
            params: Enhancement parameters
            
        Returns:
            Enhancement results
        """
        try:
            scale = params.get('realesrgan_scale', 2)
            self.logger.info(f"Using FFmpeg fallback upscaling (scale: {scale}x)")
            
            # Get original dimensions
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'stream=width,height',
                '-of', 'csv=s=x:p=0', input_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                dimensions = result.stdout.strip().split('x')
                if len(dimensions) == 2:
                    width = int(dimensions[0]) * scale
                    height = int(dimensions[1]) * scale
                else:
                    width, height = 1920, 1080  # Default
            else:
                width, height = 1920, 1080  # Default
            
            # Use high-quality FFmpeg scaling
            cmd = [
                'ffmpeg', '-y', '-i', input_path,
                '-vf', f'scale={width}:{height}:flags=lanczos',
                '-c:v', 'libx264', '-preset', 'slow', '-crf', '18',
                '-c:a', 'copy', output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            
            if result.returncode == 0 and Path(output_path).exists():
                return {'success': True, 'output_path': output_path, 'method': 'ffmpeg_upscale'}
            else:
                return {'success': False, 'error': f"FFmpeg upscaling failed: {result.stderr}"}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _enhance_with_fallback_face_enhancement(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """Fallback face enhancement using FFmpeg filters.
        
        Args:
            input_path: Path to input video
            output_path: Path for enhanced output
            
        Returns:
            Enhancement results
        """
        try:
            self.logger.info("Using FFmpeg fallback face enhancement")
            
            # Use basic enhancement filters
            cmd = [
                'ffmpeg', '-y', '-i', input_path,
                '-vf', 'unsharp=5:5:1.0:5:5:0.0,eq=contrast=1.1:brightness=0.02:saturation=1.1',
                '-c:v', 'libx264', '-preset', 'slow', '-crf', '18',
                '-c:a', 'copy', output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            
            if result.returncode == 0 and Path(output_path).exists():
                return {'success': True, 'output_path': output_path, 'method': 'ffmpeg_enhance'}
            else:
                # Final fallback: copy original
                self.logger.warning("Enhancement failed, copying original file")
                shutil.copy2(input_path, output_path)
                return {'success': True, 'output_path': output_path, 'method': 'copy_original'}
                
        except Exception as e:
            try:
                # Final fallback: copy original
                shutil.copy2(input_path, output_path)
                return {'success': True, 'output_path': output_path, 'method': 'copy_original'}
            except Exception as copy_error:
                return {'success': False, 'error': f"All enhancement methods failed: {str(e)}"}
    
    def _concatenate_enhanced_chunks(self, enhanced_chunks: List[str], output_path: Path) -> Dict[str, Any]:
        """Concatenate enhanced video chunks.
        
        Args:
            enhanced_chunks: List of enhanced chunk paths
            output_path: Output path for final video
            
        Returns:
            Concatenation results
        """
        try:
            # Create concat file
            concat_file = output_path.with_suffix('.txt')
            with open(concat_file, 'w') as f:
                for chunk_path in enhanced_chunks:
                    f.write(f"file '{chunk_path}'\n")
            
            # Run FFmpeg with professional settings
            cmd = [
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                '-i', str(concat_file),
                '-c:v', 'libx264', '-preset', 'slow', '-crf', '18',
                '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k',
                '-ar', '48000', '-af', 'aresample=async=1000:first_pts=0',
                '-movflags', '+faststart', str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            # Cleanup
            concat_file.unlink(missing_ok=True)
            
            if result.returncode == 0:
                self.logger.info("[SUCCESS] Enhanced chunks concatenated successfully")
                return {'success': True}
            else:
                return {'success': False, 'error': f"FFmpeg concatenation failed: {result.stderr}"}
                
        except Exception as e:
            return {'success': False, 'error': f"Enhanced chunk concatenation failed: {str(e)}"}
    
    def get_video_enhancement_status(self) -> Dict[str, Any]:
        """Get current video enhancement stage status.
        
        Returns:
            Status information
        """
        try:
            stage_status = self.metadata_manager.get_stage_status("video_enhancement")
            if stage_status:
                return {
                    'success': True,
                    'stage_status': stage_status
                }
            else:
                return {
                    'success': False,
                    'error': 'No video enhancement stage found in metadata'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def validate_video_enhancement_prerequisites(self) -> Dict[str, Any]:
        """Validate that video enhancement can be started.
        
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
    """Test the enhanced video enhancement stage."""
    print("🧪 Testing Enhanced Video Enhancement Stage")
    print("=" * 50)
    
    # Initialize stage
    stage = EnhancedVideoEnhancementStage()
    
    # Check prerequisites
    prereq_result = stage.validate_video_enhancement_prerequisites()
    print(f"[SUCCESS] Prerequisites check:")
    print(f"   Valid: {prereq_result['valid']}")
    if prereq_result.get('errors'):
        print(f"   Errors: {prereq_result['errors']}")
    if prereq_result.get('warnings'):
        print(f"   Warnings: {prereq_result['warnings']}")
    
    # Check current status
    status_result = stage.get_video_enhancement_status()
    if status_result['success']:
        stage_info = status_result['stage_status']
        print(f"[SUCCESS] Current status: {stage_info.get('status', 'unknown')}")
    else:
        print(f"[ERROR] Status check failed: {status_result['error']}")
    
    # Process video enhancement if prerequisites are met
    if prereq_result['valid']:
        print("\nTarget: Starting video enhancement...")
        result = stage.process_video_enhancement()
        
        if result['success']:
            print("[SUCCESS] Video enhancement completed successfully!")
            print(f"   Output: {result['output_path']}")
            print(f"   Processing time: {result['processing_time']:.1f}s")
            if result.get('chunks'):
                print(f"   Chunks: {len(result['chunks'])}")
        else:
            print(f"[ERROR] Video enhancement failed: {result['error']}")
    else:
        print("\n[ERROR] Prerequisites not met - skipping processing")
    
    print("\nSUCCESS Enhanced Video Enhancement Stage testing completed!")


if __name__ == "__main__":
    main()