#!/usr/bin/env python3
"""
Enhanced Final Assembly Stage for Video Synthesis Pipeline
Combines all processed components into final video output with metadata-driven architecture
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

# Try to import existing assembly components
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "INTEGRATED_PIPELINE" / "src"))
    from final_assembly_stage import FinalAssemblyStage
    ASSEMBLY_AVAILABLE = True
except ImportError:
    ASSEMBLY_AVAILABLE = False

# Try to import video processing utilities
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "useless_files" / "experimental_scripts"))
    from video_compositor import VideoCompositor
    from audio_synchronizer import AudioSynchronizer
    from quality_enhancer import QualityEnhancer
    COMPOSITOR_AVAILABLE = True
except ImportError:
    COMPOSITOR_AVAILABLE = False


class EnhancedFinalAssemblyStage:
    """Enhanced final assembly stage with metadata-driven architecture."""
    
    def __init__(self, base_dir: str = None):
        """Initialize enhanced final assembly stage.
        
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
        self.final_output_dir = self.base_dir / "final_output"
        self.final_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.temp_assembly_dir = self.final_output_dir / "temp"
        self.temp_assembly_dir.mkdir(exist_ok=True)
        
        # Try to initialize assembly stage
        self.assembly_stage = None
        if ASSEMBLY_AVAILABLE:
            try:
                self.assembly_stage = FinalAssemblyStage()
                self.logger.info("[SUCCESS] Final assembly stage initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize assembly stage: {e}")
                self.assembly_stage = None
        
        # Try to initialize compositor utilities
        self.video_compositor = None
        self.audio_synchronizer = None
        self.quality_enhancer = None
        if COMPOSITOR_AVAILABLE:
            try:
                self.video_compositor = VideoCompositor()
                self.audio_synchronizer = AudioSynchronizer()
                self.quality_enhancer = QualityEnhancer()
                self.logger.info("[SUCCESS] Video compositor utilities initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize compositor utilities: {e}")
        
        # Assembly configuration
        self.assembly_config = {
            'output_formats': ['mp4'],  # Can add 'mov', 'avi', 'webm'
            'video_codec': 'libx264',
            'audio_codec': 'aac',
            'video_bitrate': '8000k',
            'audio_bitrate': '192k',
            'fps': 30,
            'resolution': '1920x1080',
            'quality_preset': 'high',
            'color_correction': True,
            'audio_normalization': True
        }
        
        self.logger.info("STARTING Enhanced Final Assembly Stage initialized")
    
    def process_final_assembly(self) -> Dict[str, Any]:
        """Process final assembly using metadata-driven approach.
        
        Returns:
            Dictionary with processing results
        """
        try:
            # Update stage status to processing
            self.metadata_manager.update_stage_status(
                "final_assembly", 
                "processing",
                {"prerequisites": "all_stages_completed"}
            )
            
            # Load current session metadata
            metadata = self.metadata_manager.load_metadata()
            if not metadata:
                raise ValueError("No active session found")
            
            # Validate prerequisites
            validation_result = self._validate_prerequisites(metadata)
            if not validation_result['valid']:
                raise ValueError(f"Prerequisites not met: {validation_result['errors']}")
            
            # Get processed components from previous stages
            components = self._gather_processed_components(metadata)
            
            # Get user preferences for final assembly
            user_inputs = metadata.get('user_inputs', {})
            tone = user_inputs.get('tone', 'professional')
            emotion = user_inputs.get('emotion', 'confident')
            content_type = user_inputs.get('content_type', 'Short-Form Video Reel')
            title = user_inputs.get('title', 'Video Synthesis Output')
            
            self.logger.info(f"VIDEO PIPELINE Processing final assembly:")
            self.logger.info(f"   Title: {title}")
            self.logger.info(f"   Components: {list(components.keys())}")
            self.logger.info(f"   Tone: {tone}, Emotion: {emotion}")
            
            processing_start = time.time()
            
            # Process final assembly
            result = self._process_video_assembly(
                components, title, tone, emotion, content_type
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
                    "final_video": result['output_path'],
                    "assembly_components": list(components.keys()),
                    "processing_duration": processing_time,
                    "output_formats": result.get('output_formats', ['mp4']),
                    "video_specs": result.get('video_specs', {}),
                    "assembly_method": result.get('assembly_method', 'ffmpeg'),
                    "final_config": {
                        "title": title,
                        "tone": tone,
                        "emotion": emotion,
                        "content_type": content_type,
                        "quality_preset": self.assembly_config['quality_preset']
                    }
                }
                
                # Update metadata with successful results
                self.metadata_manager.update_stage_status(
                    "final_assembly",
                    "completed",
                    processing_data
                )
                
                self.logger.info(f"[SUCCESS] Final assembly completed successfully in {processing_time:.1f}s")
                self.logger.info(f"   Output: {result['output_path']}")
                
                return {
                    'success': True,
                    'output_path': result['output_path'],
                    'output_formats': result.get('output_formats', ['mp4']),
                    'processing_time': processing_time,
                    'stage_updated': True
                }
            else:
                # Update metadata with failure
                error_data = {
                    "processing_duration": processing_time,
                    "error_details": result.get('error', 'Unknown error'),
                    "assembly_components": list(components.keys())
                }
                
                self.metadata_manager.update_stage_status(
                    "final_assembly",
                    "failed",
                    error_data
                )
                
                self.logger.error(f"[ERROR] Final assembly failed: {result.get('error')}")
                return result
        
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"[ERROR] Final assembly error: {error_msg}")
            
            # Update metadata with error
            self.metadata_manager.update_stage_status(
                "final_assembly",
                "failed",
                {"error": error_msg}
            )
            
            return {
                'success': False,
                'error': error_msg,
                'stage_updated': True
            }
    
    def _validate_prerequisites(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that final assembly prerequisites are met.
        
        Args:
            metadata: Session metadata
            
        Returns:
            Validation results
        """
        errors = []
        warnings = []
        required_stages = ["voice_cloning", "face_processing", "video_generation", "video_enhancement"]
        optional_stages = ["background_animation"]
        
        # Check required stages
        for stage_name in required_stages:
            stage = self.metadata_manager.get_stage_status(stage_name)
            if not stage or stage.get("status") != "completed":
                errors.append(f"{stage_name.replace('_', ' ').title()} stage not completed")
        
        # Check optional stages
        for stage_name in optional_stages:
            stage = self.metadata_manager.get_stage_status(stage_name)
            if not stage or stage.get("status") != "completed":
                warnings.append(f"{stage_name.replace('_', ' ').title()} stage not completed - will skip")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def _gather_processed_components(self, metadata: Dict[str, Any]) -> Dict[str, str]:
        """Gather all processed components from previous stages.
        
        Args:
            metadata: Session metadata
            
        Returns:
            Dictionary of component types to file paths
        """
        components = {}
        
        try:
            # Enhanced video (from video enhancement stage)
            video_enhancement_stage = self.metadata_manager.get_stage_status("video_enhancement")
            if video_enhancement_stage and video_enhancement_stage.get("status") == "completed":
                processing_data = video_enhancement_stage.get("processing_data", {})
                enhanced_video = processing_data.get("enhanced_video")
                if enhanced_video:
                    components['enhanced_video'] = str(self.base_dir / enhanced_video)
            
            # Check for enhanced video chunks with audio (SadTalker multi-chunk output)
            enhanced_chunks = self._find_enhanced_video_chunks()
            if enhanced_chunks:
                components['enhanced_video_chunks'] = enhanced_chunks
                self.logger.info(f"   Found {len(enhanced_chunks)} enhanced video chunks with audio")
            
            # Background animation (from Manim stage)
            background_stage = self.metadata_manager.get_stage_status("background_animation")
            if background_stage and background_stage.get("status") == "completed":
                processing_data = background_stage.get("processing_data", {})
                background_animation = processing_data.get("background_animation")
                if background_animation:
                    components['background_animation'] = str(self.base_dir / background_animation)
            
            # Fallback: get video from video generation if enhancement not available
            if 'enhanced_video' not in components and 'enhanced_video_chunks' not in components:
                video_gen_stage = self.metadata_manager.get_stage_status("video_generation")
                if video_gen_stage and video_gen_stage.get("status") == "completed":
                    processing_data = video_gen_stage.get("processing_data", {})
                    generated_video = processing_data.get("generated_video")
                    if generated_video:
                        components['generated_video'] = str(self.base_dir / generated_video)
            
            # Voice audio (for audio track verification)
            voice_stage = self.metadata_manager.get_stage_status("voice_cloning")
            if voice_stage and voice_stage.get("status") == "completed":
                processing_data = voice_stage.get("processing_data", {})
                synthesized_voice = processing_data.get("synthesized_voice")
                if synthesized_voice:
                    components['voice_audio'] = str(self.base_dir / synthesized_voice)
            
            self.logger.info(f"   Gathered components: {list(components.keys())}")
            return components
            
        except Exception as e:
            self.logger.error(f"Error gathering components: {e}")
            return components
    
    def _find_enhanced_video_chunks(self) -> List[str]:
        """Find enhanced video chunks with audio from SadTalker processing.
        
        Returns:
            List of enhanced video chunk paths with audio
        """
        chunk_paths = []
        
        try:
            # Look for enhanced_with_audio_chunk*.mp4 files in multiple possible locations
            possible_dirs = [
                self.base_dir / "processed" / "enhancement",  # Expected location
                self.base_dir / "NEW" / "processed" / "enhancement",  # Actual location
                Path.cwd() / "NEW" / "processed" / "enhancement"  # Alternative location
            ]
            
            for enhancement_dir in possible_dirs:
                if enhancement_dir.exists():
                    chunk_files = sorted(enhancement_dir.glob("enhanced_with_audio_chunk*.mp4"))
                    for chunk_file in chunk_files:
                        if chunk_file.exists() and chunk_file.stat().st_size > 1000:  # At least 1KB
                            chunk_paths.append(str(chunk_file))
                    
                    if chunk_paths:
                        self.logger.info(f"   Found {len(chunk_paths)} enhanced video chunks in {enhancement_dir}")
                        break
            
            return chunk_paths
            
        except Exception as e:
            self.logger.warning(f"Error finding enhanced video chunks: {e}")
            return []
    
    def _process_video_assembly(self, components: Dict[str, str], title: str,
                              tone: str, emotion: str, content_type: str) -> Dict[str, Any]:
        """Process video assembly with available components.
        
        Args:
            components: Dictionary of component types to file paths
            title: Video title
            tone: Voice tone parameter
            emotion: Voice emotion parameter
            content_type: Content type
            
        Returns:
            Processing results
        """
        try:
            timestamp = int(datetime.now().timestamp())
            
            # Determine assembly strategy based on available components (NEW PRIORITY ORDER)
            if 'enhanced_video_chunks' in components and 'background_animation' in components:
                # NEW: Composite enhanced video chunks with Manim background animation
                result = self._composite_enhanced_chunks_with_background(
                    components['enhanced_video_chunks'], components['background_animation'], 
                    components, title, timestamp, tone, emotion, content_type
                )
            elif 'enhanced_video_chunks' in components:
                # Concatenate enhanced video chunks with audio (SadTalker multi-chunk output)
                result = self._concatenate_enhanced_chunks(
                    components['enhanced_video_chunks'], components, title, timestamp, tone, emotion, content_type
                )
            elif 'enhanced_video' in components and 'background_animation' in components:
                # Full assembly with background compositing
                result = self._composite_video_with_background(
                    components, title, timestamp, tone, emotion, content_type
                )
            elif 'enhanced_video' in components or 'generated_video' in components:
                # Video-only assembly with audio enhancement
                video_path = components.get('enhanced_video') or components.get('generated_video')
                result = self._enhance_video_only(
                    video_path, components.get('voice_audio'), title, timestamp, tone, emotion
                )
            else:
                return {'success': False, 'error': 'No video components available for assembly'}
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _concatenate_enhanced_chunks(self, enhanced_chunks: List[str], components: Dict[str, str], 
                                   title: str, timestamp: int, tone: str, emotion: str, content_type: str) -> Dict[str, Any]:
        """Concatenate enhanced video chunks with audio into final video.
        
        Args:
            enhanced_chunks: List of enhanced video chunk paths with audio
            components: Available components dictionary
            title: Video title
            timestamp: Timestamp for output naming
            tone: Voice tone
            emotion: Voice emotion
            content_type: Content type
            
        Returns:
            Concatenation results
        """
        try:
            self.logger.info("[EMOJI] Concatenating enhanced video chunks with audio...")
            
            # Create output filename
            output_filename = f"final_video_{timestamp}.mp4"
            output_path = self.final_output_dir / output_filename
            
            # Create file list for FFmpeg concatenation
            temp_dir = self.final_output_dir / "temp"
            temp_dir.mkdir(exist_ok=True)
            
            filelist_path = temp_dir / "enhanced_chunks_filelist.txt"
            with open(filelist_path, 'w') as f:
                for chunk in enhanced_chunks:
                    f.write(f"file '{chunk}'\n")
            
            # Use FFmpeg to concatenate all chunks
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(filelist_path),
                '-c', 'copy',  # Copy streams without re-encoding for speed and quality
                str(output_path)
            ]
            
            self.logger.info(f"   Concatenating {len(enhanced_chunks)} enhanced chunks...")
            self.logger.info(f"   Command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            # Clean up temp file
            if filelist_path.exists():
                filelist_path.unlink()
            
            if result.returncode == 0 and output_path.exists():
                self.logger.info("[SUCCESS] Enhanced chunks concatenated successfully")
                
                # Apply final enhancements and metadata
                return self._apply_final_enhancements(
                    str(output_path), title, timestamp, tone, emotion
                )
            else:
                return {'success': False, 'error': f"FFmpeg concatenation failed: {result.stderr}"}
                
        except Exception as e:
            return {'success': False, 'error': f"Enhanced chunks concatenation failed: {str(e)}"}
    
    def _composite_enhanced_chunks_with_background(self, enhanced_chunks: List[str], background_animation: str,
                                                 components: Dict[str, str], title: str, timestamp: int, 
                                                 tone: str, emotion: str, content_type: str) -> Dict[str, Any]:
        """Composite enhanced video chunks with Manim background animation.
        
        This is the key missing integration: combine multiple enhanced video chunks with audio
        AND overlay them on a Manim background animation.
        
        Args:
            enhanced_chunks: List of enhanced video chunk paths with audio
            background_animation: Path to Manim background animation
            components: Available components dictionary
            title: Video title
            timestamp: Timestamp for output naming
            tone: Voice tone
            emotion: Voice emotion
            content_type: Content type
            
        Returns:
            Compositing results with both video and audio
        """
        try:
            self.logger.info("Frontend Compositing enhanced video chunks with Manim background animation...")
            
            # Step 1: First concatenate all enhanced chunks into a single temp video (preserving audio)
            temp_dir = self.final_output_dir / "temp"
            temp_dir.mkdir(exist_ok=True)
            
            temp_concatenated = temp_dir / f"temp_concatenated_{timestamp}.mp4"
            
            # Create file list for FFmpeg concatenation
            filelist_path = temp_dir / "enhanced_chunks_filelist.txt"
            with open(filelist_path, 'w') as f:
                for chunk in enhanced_chunks:
                    f.write(f"file '{chunk}'\n")
            
            # Concatenate enhanced chunks preserving audio
            concat_cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0', 
                '-i', str(filelist_path),
                '-c', 'copy',  # Preserve quality and audio
                str(temp_concatenated)
            ]
            
            self.logger.info(f"   Step 1: Concatenating {len(enhanced_chunks)} enhanced chunks...")
            self.logger.info(f"   Command: {' '.join(concat_cmd)}")
            
            concat_result = subprocess.run(concat_cmd, capture_output=True, text=True, timeout=600)
            
            if concat_result.returncode != 0 or not temp_concatenated.exists():
                return {'success': False, 'error': f"Chunk concatenation failed: {concat_result.stderr}"}
            
            # Step 2: Composite the concatenated talking head video with Manim background
            output_filename = f"final_video_with_manim_{timestamp}.mp4"
            output_path = self.final_output_dir / output_filename
            
            # Use FFmpeg to composite talking head over Manim background (preserving talking head audio)
            composite_cmd = [
                'ffmpeg', '-y',
                '-i', str(background_animation),  # Input 0: Manim background video
                '-i', str(temp_concatenated),     # Input 1: Concatenated talking head with audio
                '-filter_complex', 
                # Scale talking head to 512x768 and overlay on center of 1920x1080 background
                '[1:v]scale=512:768[face];[0:v][face]overlay=(W-w)/2:(H-h)/2[v]',
                '-map', '[v]',         # Use composited video
                '-map', '1:a',         # Use audio from talking head (input 1)
                '-c:v', 'libx264',     # Re-encode video for compatibility
                '-c:a', 'aac',         # Re-encode audio for compatibility
                '-b:v', '8000k',       # High quality video bitrate
                '-b:a', '192k',        # High quality audio bitrate
                '-shortest',           # Match shortest input duration
                str(output_path)
            ]
            
            self.logger.info(f"   Step 2: Compositing with Manim background...")
            self.logger.info(f"   Background: {background_animation}")
            self.logger.info(f"   Command: {' '.join(composite_cmd)}")
            
            composite_result = subprocess.run(composite_cmd, capture_output=True, text=True, timeout=1200)
            
            # Clean up temp files
            if filelist_path.exists():
                filelist_path.unlink()
            if temp_concatenated.exists():
                temp_concatenated.unlink()
            
            if composite_result.returncode == 0 and output_path.exists():
                self.logger.info("[SUCCESS] Enhanced chunks with Manim background composited successfully")
                
                # Apply final enhancements and metadata
                return self._apply_final_enhancements(
                    str(output_path), title, timestamp, tone, emotion
                )
            else:
                return {'success': False, 'error': f"Manim background compositing failed: {composite_result.stderr}"}
                
        except Exception as e:
            return {'success': False, 'error': f"Enhanced chunks + Manim background compositing failed: {str(e)}"}
    
    def _composite_video_with_background(self, components: Dict[str, str], title: str,
                                       timestamp: int, tone: str, emotion: str,
                                       content_type: str) -> Dict[str, Any]:
        """Composite talking head video with background animation.
        
        Args:
            components: Available components
            title: Video title
            timestamp: Timestamp for output naming
            tone: Voice tone
            emotion: Voice emotion
            content_type: Content type
            
        Returns:
            Compositing results
        """
        try:
            enhanced_video = components['enhanced_video']
            background_animation = components['background_animation']
            
            # Output file
            output_filename = f"final_video_{timestamp}.mp4"
            output_path = self.final_output_dir / output_filename
            
            self.logger.info("Frontend Compositing video with background animation...")
            
            if self.video_compositor:
                # Use advanced compositor
                result = self.video_compositor.composite_video_with_background(
                    foreground_video=enhanced_video,
                    background_video=background_animation,
                    output_path=str(output_path),
                    composition_mode='overlay'
                )
                
                if result['success']:
                    # Apply final enhancements
                    return self._apply_final_enhancements(
                        str(output_path), title, timestamp, tone, emotion
                    )
                else:
                    # Fallback to basic compositing
                    return self._basic_video_compositing(
                        enhanced_video, background_animation, str(output_path), title, timestamp
                    )
            else:
                # Use basic FFmpeg compositing
                return self._basic_video_compositing(
                    enhanced_video, background_animation, str(output_path), title, timestamp
                )
                
        except Exception as e:
            return {'success': False, 'error': f"Video compositing failed: {str(e)}"}
    
    def _basic_video_compositing(self, foreground_video: str, background_video: str,
                               output_path: str, title: str, timestamp: int) -> Dict[str, Any]:
        """Basic video compositing using FFmpeg.
        
        Args:
            foreground_video: Path to foreground (talking head) video
            background_video: Path to background animation
            output_path: Output path for composited video
            title: Video title
            timestamp: Timestamp for naming
            
        Returns:
            Compositing results
        """
        try:
            self.logger.info("Using FFmpeg for basic video compositing...")
            
            # Create composited video with overlay
            cmd = [
                'ffmpeg', '-y',
                '-i', background_video,    # Background
                '-i', foreground_video,    # Foreground
                '-filter_complex', 
                '[0:v]scale=1920:1080[bg];[1:v]scale=1280:720[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2[v]',
                '-map', '[v]',
                '-map', '1:a',  # Use audio from foreground video
                '-c:v', self.assembly_config['video_codec'],
                '-c:a', self.assembly_config['audio_codec'],
                '-b:v', self.assembly_config['video_bitrate'],
                '-b:a', self.assembly_config['audio_bitrate'],
                '-r', str(self.assembly_config['fps']),
                '-preset', 'slow',
                '-crf', '18',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            
            if result.returncode == 0 and Path(output_path).exists():
                self.logger.info("[SUCCESS] Basic video compositing completed")
                
                # Apply final enhancements
                return self._apply_final_enhancements(
                    output_path, title, timestamp, 'professional', 'confident'
                )
            else:
                return {'success': False, 'error': f"FFmpeg compositing failed: {result.stderr}"}
                
        except Exception as e:
            return {'success': False, 'error': f"Basic compositing failed: {str(e)}"}
    
    def _enhance_video_only(self, video_path: str, audio_path: Optional[str], title: str,
                          timestamp: int, tone: str, emotion: str) -> Dict[str, Any]:
        """Enhance video-only assembly without background.
        
        Args:
            video_path: Path to main video
            audio_path: Optional audio path for enhancement
            title: Video title
            timestamp: Timestamp for output naming
            tone: Voice tone
            emotion: Voice emotion
            
        Returns:
            Enhancement results
        """
        try:
            output_filename = f"final_video_{timestamp}.mp4"
            output_path = self.final_output_dir / output_filename
            
            self.logger.info("VIDEO PIPELINE Processing video-only assembly...")
            
            if audio_path and Path(audio_path).exists():
                # Enhance with high-quality audio processing
                cmd = [
                    'ffmpeg', '-y',
                    '-i', video_path,
                    '-i', audio_path,
                    '-c:v', self.assembly_config['video_codec'],
                    '-c:a', self.assembly_config['audio_codec'],
                    '-b:v', self.assembly_config['video_bitrate'],
                    '-b:a', self.assembly_config['audio_bitrate'],
                    '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11:measured_I=-20:measured_LRA=7:measured_TP=-2',
                    '-preset', 'slow',
                    '-crf', '18',
                    str(output_path)
                ]
            else:
                # Copy with re-encoding for quality
                cmd = [
                    'ffmpeg', '-y',
                    '-i', video_path,
                    '-c:v', self.assembly_config['video_codec'],
                    '-c:a', self.assembly_config['audio_codec'],
                    '-b:v', self.assembly_config['video_bitrate'],
                    '-b:a', self.assembly_config['audio_bitrate'],
                    '-preset', 'slow',
                    '-crf', '18',
                    str(output_path)
                ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            
            if result.returncode == 0 and output_path.exists():
                self.logger.info("[SUCCESS] Video-only assembly completed")
                
                # Apply final enhancements
                return self._apply_final_enhancements(
                    str(output_path), title, timestamp, tone, emotion
                )
            else:
                return {'success': False, 'error': f"Video processing failed: {result.stderr}"}
                
        except Exception as e:
            return {'success': False, 'error': f"Video-only enhancement failed: {str(e)}"}
    
    def _apply_final_enhancements(self, video_path: str, title: str, timestamp: int,
                                tone: str, emotion: str) -> Dict[str, Any]:
        """Apply final enhancements and create multiple output formats.
        
        Args:
            video_path: Path to processed video
            title: Video title
            timestamp: Timestamp for naming
            tone: Voice tone
            emotion: Voice emotion
            
        Returns:
            Final enhancement results
        """
        try:
            final_outputs = []
            video_specs = {}
            
            # Get video information
            video_info = self._get_video_info(video_path)
            if video_info:
                video_specs = video_info
            
            # Create final output with title metadata
            for format_ext in self.assembly_config['output_formats']:
                final_filename = f"{title.replace(' ', '_').lower()}_{timestamp}.{format_ext}"
                final_output = self.final_output_dir / final_filename
                
                # Add metadata and final quality adjustments
                cmd = [
                    'ffmpeg', '-y',
                    '-i', video_path,
                    '-metadata', f'title={title}',
                    '-metadata', f'artist=Video Synthesis Pipeline',
                    '-metadata', f'comment=Generated with tone: {tone}, emotion: {emotion}',
                    '-movflags', '+faststart',  # Optimize for streaming
                    str(final_output)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                
                if result.returncode == 0 and final_output.exists():
                    final_outputs.append(str(final_output.relative_to(self.base_dir)))
                    self.logger.info(f"[SUCCESS] Created final output: {final_output.name}")
                else:
                    self.logger.warning(f"Failed to create {format_ext} output: {result.stderr}")
            
            if final_outputs:
                return {
                    'success': True,
                    'output_path': final_outputs[0],  # Primary output
                    'output_formats': final_outputs,
                    'video_specs': video_specs,
                    'assembly_method': 'ffmpeg_enhanced'
                }
            else:
                return {'success': False, 'error': 'No final outputs created'}
                
        except Exception as e:
            return {'success': False, 'error': f"Final enhancement failed: {str(e)}"}
    
    def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video information using ffprobe.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Video information dictionary
        """
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                info = json.loads(result.stdout)
                
                # Extract relevant information
                video_info = {}
                if 'format' in info:
                    format_info = info['format']
                    video_info['duration'] = float(format_info.get('duration', 0))
                    video_info['size'] = int(format_info.get('size', 0))
                    video_info['bit_rate'] = int(format_info.get('bit_rate', 0))
                
                for stream in info.get('streams', []):
                    if stream.get('codec_type') == 'video':
                        video_info['width'] = stream.get('width')
                        video_info['height'] = stream.get('height')
                        video_info['fps'] = eval(stream.get('r_frame_rate', '30/1'))
                        video_info['codec'] = stream.get('codec_name')
                
                return video_info
            else:
                return {}
                
        except Exception as e:
            self.logger.warning(f"Could not get video info: {e}")
            return {}
    
    def get_final_assembly_status(self) -> Dict[str, Any]:
        """Get current final assembly stage status.
        
        Returns:
            Status information
        """
        try:
            stage_status = self.metadata_manager.get_stage_status("final_assembly")
            if stage_status:
                return {
                    'success': True,
                    'stage_status': stage_status
                }
            else:
                return {
                    'success': False,
                    'error': 'No final assembly stage found in metadata'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def validate_final_assembly_prerequisites(self) -> Dict[str, Any]:
        """Validate that final assembly can be started.
        
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
    """Test the enhanced final assembly stage."""
    print("🧪 Testing Enhanced Final Assembly Stage")
    print("=" * 50)
    
    # Initialize stage
    stage = EnhancedFinalAssemblyStage()
    
    # Check prerequisites
    prereq_result = stage.validate_final_assembly_prerequisites()
    print(f"[SUCCESS] Prerequisites check:")
    print(f"   Valid: {prereq_result['valid']}")
    if prereq_result.get('errors'):
        print(f"   Errors: {prereq_result['errors']}")
    if prereq_result.get('warnings'):
        print(f"   Warnings: {prereq_result['warnings']}")
    
    # Check current status
    status_result = stage.get_final_assembly_status()
    if status_result['success']:
        stage_info = status_result['stage_status']
        print(f"[SUCCESS] Current status: {stage_info.get('status', 'unknown')}")
    else:
        print(f"[ERROR] Status check failed: {status_result['error']}")
    
    # Process final assembly if prerequisites are met
    if prereq_result['valid']:
        print("\nVIDEO PIPELINE Starting final assembly...")
        result = stage.process_final_assembly()
        
        if result['success']:
            print("[SUCCESS] Final assembly completed successfully!")
            print(f"   Output: {result['output_path']}")
            print(f"   Formats: {result.get('output_formats', [])}")
            print(f"   Processing time: {result['processing_time']:.1f}s")
        else:
            print(f"[ERROR] Final assembly failed: {result['error']}")
    else:
        print("\n[ERROR] Prerequisites not met - skipping processing")
    
    print("\nSUCCESS Enhanced Final Assembly Stage testing completed!")


if __name__ == "__main__":
    main()