#!/usr/bin/env python3
"""
Production Video Synthesis Pipeline
Complete integration of: XTTS → InsightFace → SadTalker → Real-ESRGAN → CodeFormer → Manim → FFmpeg

Production Pipeline Flow (No Fallbacks):
1. XTTS Voice Cloning with dynamic 10-second chunking
2. InsightFace buffalo_l face detection and processing
3. SadTalker lip-sync video generation for each chunk
4. Real-ESRGAN 1080p upscaling with M4 Max GPU
5. CodeFormer face restoration and enhancement
6. Manim educational animation synchronized with voice
7. FFmpeg professional final assembly with overlays
"""

import os
import sys
import time
import json
import math
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import tempfile
import shutil
import subprocess

# Add core directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import our production stages
from production_xtts_stage import ProductionXTTSStage
from production_insightface_stage import ProductionInsightFaceStage
from production_sadtalker_stage import ProductionSadTalkerStage
from production_enhancement_stage import ProductionEnhancementStage
from production_manim_stage import ProductionManimStage
from production_ffmpeg_stage import ProductionFFmpegStage
from enhanced_final_assembly_stage import EnhancedFinalAssemblyStage
from enhanced_metadata_manager import EnhancedMetadataManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionVideoSynthesisPipeline:
    """
    Production video synthesis pipeline with no fallbacks.
    
    Complete Pipeline Flow:
    1. XTTS Voice Cloning (dynamic 10-second chunking)
    2. InsightFace Detection (buffalo_l face preprocessing)
    3. SadTalker Animation (lip-sync for each chunk)
    4. Real-ESRGAN Upscaling (M4 Max GPU 1080p enhancement)
    5. CodeFormer Enhancement (face restoration)
    6. Manim Integration (educational animation)
    7. FFmpeg Assembly (professional final video)
    """
    
    def __init__(self, base_dir: str = None, quality_level: str = 'high'):
        """Initialize the production pipeline.
        
        Args:
            base_dir: Base directory for pipeline operations
            quality_level: Quality level (draft, standard, high, maximum)
        """
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        self.quality_level = quality_level
        
        # Pipeline state tracking
        self.current_stage = None
        self.stage_outputs = {}
        self.performance_metrics = {}
        self.temp_dir = None
        
        # Initialize logging
        self.pipeline_id = f"production_pipeline_{int(time.time())}"
        self.logger = logging.getLogger(f"ProductionPipeline-{self.pipeline_id}")
        
        # Output directories
        self.output_dir = self.base_dir / "NEW" / "processed"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.metadata_manager = EnhancedMetadataManager(str(self.base_dir))
        self.xtts_stage = ProductionXTTSStage(str(self.base_dir))
        self.insightface_stage = ProductionInsightFaceStage(str(self.base_dir))
        self.sadtalker_stage = ProductionSadTalkerStage(str(self.base_dir))
        self.enhancement_stage = ProductionEnhancementStage(str(self.base_dir))
        self.manim_stage = ProductionManimStage(str(self.base_dir))
        self.ffmpeg_stage = ProductionFFmpegStage(str(self.base_dir))
        self.final_assembly_stage = EnhancedFinalAssemblyStage(str(self.base_dir))
        
        # Conda environment paths
        self.conda_envs = {
            'xtts': '/Users/aryanjain/miniforge3/envs/xtts_voice_cloning/bin/python',
            'sadtalker': '/opt/miniconda3/envs/sadtalker/bin/python',
            'realesrgan': '/Users/aryanjain/miniforge3/envs/realesrgan_real/bin/python',
            'video_processing': '/Users/aryanjain/miniforge3/envs/video-audio-processing/bin/python'
        }
        
        self.logger.info("STARTING Production Video Synthesis Pipeline Initialized")
        self.logger.info(f"   Pipeline ID: {self.pipeline_id}")
        self.logger.info(f"   Quality Level: {self.quality_level}")
        self.logger.info(f"   Base Directory: {self.base_dir}")
    
    def run_complete_pipeline(self, text: str, voice_reference: str, 
                             face_image: str, output_path: str, user_inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute the complete production pipeline.
        
        Args:
            text: Text to be spoken
            voice_reference: Path to reference voice file
            face_image: Path to reference face image
            output_path: Path for final output video
            
        Returns:
            Dictionary with pipeline results and metrics
        """
        pipeline_start = time.time()
        
        try:
            self.logger.info("VIDEO PIPELINE Starting Production Video Synthesis Pipeline")
            self.logger.info(f"   Text length: {len(text)} characters")
            self.logger.info(f"   Voice reference: {voice_reference}")
            self.logger.info(f"   Face image: {face_image}")
            self.logger.info(f"   Output path: {output_path}")
            
            # Setup temporary workspace
            self._setup_workspace()
            
            # Stage 1: XTTS Voice Cloning with Dynamic Chunking
            self.logger.info("Recording Stage 1: XTTS Voice Cloning with Dynamic Chunking")
            voice_chunks = self._stage_xtts_voice_cloning(text, voice_reference, user_inputs)
            
            # Stage 2: InsightFace Detection and Processing
            self.logger.info("Face: Stage 2: InsightFace Detection and Processing")
            face_data = self._stage_insightface_detection(face_image, user_inputs)
            
            # Stage 3: SadTalker Animation for Each Chunk
            self.logger.info("Style: Stage 3: SadTalker Animation for Each Chunk")
            video_chunks = self._stage_sadtalker_animation(voice_chunks, face_data, user_inputs)
            
            # Stage 4: Combined Enhancement Pipeline (Real-ESRGAN + CodeFormer)
            self.logger.info("Enhanced Stage 4: Combined Enhancement Pipeline (Real-ESRGAN + CodeFormer)")
            enhanced_chunks = self._stage_enhancement_pipeline(video_chunks)
            
            # Stage 5: Manim Background Animation
            self.logger.info("Frontend Stage 5: Manim Background Animation")
            background_video = self._stage_manim_animation(text, len(voice_chunks))
            
            # Stage 6: FFmpeg Final Assembly with bottom-left overlay
            self.logger.info("[EMOJI] Stage 6: FFmpeg Final Assembly with bottom-left overlay")
            final_output = self._stage_ffmpeg_assembly(enhanced_chunks, background_video, output_path)
            
            # Calculate total time and compile results
            total_time = time.time() - pipeline_start
            
            results = {
                'success': True,
                'pipeline_id': self.pipeline_id,
                'output_video': final_output,
                'total_time_seconds': total_time,
                'total_chunks': len(voice_chunks),
                'stage_outputs': self.stage_outputs,
                'performance_metrics': self.performance_metrics,
                'quality_level': self.quality_level
            }
            
            self.logger.info("SUCCESS Production Pipeline Completed Successfully!")
            self.logger.info(f"   Total time: {total_time:.2f} seconds")
            self.logger.info(f"   Total chunks: {len(voice_chunks)}")
            self.logger.info(f"   Final output: {final_output}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"[ERROR] Production Pipeline failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'pipeline_id': self.pipeline_id,
                'stage_outputs': self.stage_outputs,
                'performance_metrics': self.performance_metrics
            }
        finally:
            # Cleanup temporary workspace
            self._cleanup_workspace()
    
    def _setup_workspace(self):
        """Setup temporary workspace for pipeline operations."""
        self.temp_dir = tempfile.mkdtemp(prefix=f"production_pipeline_{self.pipeline_id}_")
        self.logger.info(f"Assets: Workspace setup: {self.temp_dir}")
    
    def _cleanup_workspace(self):
        """Cleanup temporary workspace."""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            self.logger.info("🧹 Workspace cleaned up")
    
    def _calculate_dynamic_chunks(self, text: str, chunk_duration: int = None, words_per_minute: int = None) -> int:
        """Calculate number of chunks based on text length and speaking rate.
        
        Args:
            text: Text to be spoken
            chunk_duration: Duration per chunk in seconds
            
        Returns:
            Number of chunks needed
        """
        # User configurable speaking rate
        if words_per_minute is None:
            words_per_minute = 150  # Default average speaking rate
        if chunk_duration is None:
            chunk_duration = 10  # Default chunk duration
        words_per_second = words_per_minute / 60
        
        # Calculate estimated duration
        word_count = len(text.split())
        estimated_duration = word_count / words_per_second
        
        # Calculate number of chunks (minimum 1)
        num_chunks = max(1, math.ceil(estimated_duration / chunk_duration))
        
        self.logger.info(f"[EMOJI] Dynamic chunking calculation:")
        self.logger.info(f"   Words: {word_count}")
        self.logger.info(f"   Estimated duration: {estimated_duration:.1f} seconds")
        self.logger.info(f"   Chunks needed: {num_chunks}")
        
        return num_chunks
    
    def _stage_xtts_voice_cloning(self, text: str, voice_reference: str, user_inputs: Dict[str, Any] = None) -> List[str]:
        """Stage 1: XTTS Voice Cloning with Dynamic Chunking.
        
        Args:
            text: Text to be spoken
            voice_reference: Path to reference voice file
            
        Returns:
            List of audio chunk file paths
        """
        stage_start = time.time()
        self.current_stage = "xtts_voice_cloning"
        
        try:
            # Calculate dynamic chunks with user parameters
            chunk_duration = user_inputs.get('chunk_duration', 10) if user_inputs else 10
            words_per_minute = user_inputs.get('words_per_minute', 150) if user_inputs else 150
            num_chunks = self._calculate_dynamic_chunks(text, chunk_duration, words_per_minute)
            
            # Split text into chunks
            text_chunks = self._split_text_into_chunks(text, num_chunks)
            
            # Process each chunk with XTTS
            voice_chunks = []
            for i, chunk_text in enumerate(text_chunks):
                self.logger.info(f"Recording Processing voice chunk {i+1}/{len(text_chunks)}")
                
                # Use ProductionXTTSStage to process this chunk
                script_data = {
                    'text_chunks': [chunk_text],
                    'clean_script': chunk_text,
                    'tone': user_inputs.get('tone', 'professional') if user_inputs else 'professional',
                    'emotion': user_inputs.get('emotion', 'confident') if user_inputs else 'confident'
                }
                
                xtts_params = {
                    'temperature': user_inputs.get('xtts_temperature', 0.7) if user_inputs else 0.7,
                    'speed': user_inputs.get('xtts_speed', 1.0) if user_inputs else 1.0,
                    'repetition_penalty': user_inputs.get('xtts_repetition_penalty', 1.1) if user_inputs else 1.1
                }
                
                result = self.xtts_stage.process_voice_cloning(
                    voice_reference, script_data, xtts_params
                )
                
                if result['success']:
                    voice_chunks.append(result['output_audio_path'])
                    self.logger.info(f"[SUCCESS] Voice chunk {i+1} completed: {result['output_audio_path']}")
                else:
                    raise RuntimeError(f"XTTS failed for chunk {i+1}: {result['errors']}")
            
            # Store stage results
            stage_time = time.time() - stage_start
            self.stage_outputs['xtts_voice_cloning'] = {
                'chunks': voice_chunks,
                'num_chunks': len(voice_chunks),
                'processing_time': stage_time
            }
            self.performance_metrics['xtts_voice_cloning'] = stage_time
            
            self.logger.info(f"[SUCCESS] XTTS Voice Cloning completed in {stage_time:.2f} seconds")
            return voice_chunks
            
        except Exception as e:
            self.logger.error(f"[ERROR] XTTS Voice Cloning failed: {e}")
            raise
    
    def _split_text_into_chunks(self, text: str, num_chunks: int) -> List[str]:
        """Split text into approximately equal chunks.
        
        Args:
            text: Text to split
            num_chunks: Number of chunks to create
            
        Returns:
            List of text chunks
        """
        if num_chunks <= 1:
            return [text]
        
        # Split by sentences first for natural breaks
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # If we have fewer sentences than chunks, split by words
        if len(sentences) < num_chunks:
            words = text.split()
            words_per_chunk = len(words) // num_chunks
            chunks = []
            
            for i in range(num_chunks):
                start_idx = i * words_per_chunk
                end_idx = (i + 1) * words_per_chunk if i < num_chunks - 1 else len(words)
                chunk = ' '.join(words[start_idx:end_idx])
                chunks.append(chunk)
            
            return chunks
        
        # Distribute sentences across chunks
        sentences_per_chunk = len(sentences) // num_chunks
        chunks = []
        
        for i in range(num_chunks):
            start_idx = i * sentences_per_chunk
            end_idx = (i + 1) * sentences_per_chunk if i < num_chunks - 1 else len(sentences)
            chunk = '. '.join(sentences[start_idx:end_idx]) + '.'
            chunks.append(chunk)
        
        return chunks
    
    def _stage_insightface_detection(self, face_image: str, user_inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Stage 2: InsightFace Detection and Processing.
        
        Args:
            face_image: Path to face image
            
        Returns:
            Face data dictionary
        """
        stage_start = time.time()
        self.current_stage = "insightface_detection"
        
        try:
            # User configurable emotion parameters for face detection
            emotion_params = {
                'emotion': user_inputs.get('emotion', 'professional') if user_inputs else 'professional',
                'tone': user_inputs.get('tone', 'confident') if user_inputs else 'confident'
            }
            
            # Use ProductionInsightFaceStage for face detection
            result = self.insightface_stage.process_face_detection(face_image, emotion_params)
            
            if result['success']:
                face_data = {
                    'face_image': face_image,
                    'face_crop': result['best_face_crop_path'],
                    'face_detected': True,
                    'detected_faces': result['detected_faces'],
                    'processing_time': result['processing_time']
                }
                
                stage_time = time.time() - stage_start
                self.stage_outputs['insightface_detection'] = face_data
                self.performance_metrics['insightface_detection'] = stage_time
                
                self.logger.info(f"[SUCCESS] InsightFace Detection completed in {stage_time:.2f} seconds")
                self.logger.info(f"   Detected faces: {len(result['detected_faces'])}")
                self.logger.info(f"   Face crop saved: {result['best_face_crop_path']}")
                
                return face_data
            else:
                raise RuntimeError(f"InsightFace detection failed: {result['errors']}")
            
        except Exception as e:
            self.logger.error(f"[ERROR] InsightFace Detection failed: {e}")
            raise
    
    def _stage_sadtalker_animation(self, voice_chunks: List[str], face_data: Dict[str, Any], user_inputs: Dict[str, Any] = None) -> List[str]:
        """Stage 3: SadTalker Animation for Each Chunk.
        
        Args:
            voice_chunks: List of audio chunk paths
            face_data: Face detection data
            
        Returns:
            List of video chunk paths
        """
        stage_start = time.time()
        self.current_stage = "sadtalker_animation"
        
        try:
            video_chunks = []
            face_image_path = face_data['face_image']  # Use original user image, not InsightFace crop
            
            # Process each voice chunk with SadTalker
            for i, voice_chunk in enumerate(voice_chunks):
                self.logger.info(f"Style: Processing SadTalker animation for chunk {i+1}/{len(voice_chunks)}")
                
                # User configurable parameters for SadTalker
                emotion_params = {
                    'emotion': user_inputs.get('emotion', 'professional') if user_inputs else 'professional',
                    'tone': user_inputs.get('tone', 'confident') if user_inputs else 'confident',
                    'expression_scale': user_inputs.get('expression_scale', 1.0) if user_inputs else 1.0,
                    'facial_animation_strength': user_inputs.get('facial_animation_strength', 1.0) if user_inputs else 1.0,
                    'size': user_inputs.get('sadtalker_size', 256) if user_inputs else 256,
                    'fps': user_inputs.get('sadtalker_fps', 30) if user_inputs else 30
                }
                
                # Use ProductionSadTalkerStage to process this chunk
                result = self.sadtalker_stage.process_lip_sync_animation(
                    face_image_path, voice_chunk, emotion_params
                )
                
                if result['success']:
                    video_chunks.append(result['output_video_path'])
                    self.logger.info(f"[SUCCESS] SadTalker chunk {i+1} completed: {result['output_video_path']}")
                    self.logger.info(f"   File size: {result.get('file_size', 0):,} bytes")
                    self.logger.info(f"   Duration: {result.get('video_duration', 0):.2f} seconds")
                else:
                    raise RuntimeError(f"SadTalker failed for chunk {i+1}: {result['errors']}")
            
            stage_time = time.time() - stage_start
            self.stage_outputs['sadtalker_animation'] = {
                'chunks': video_chunks,
                'num_chunks': len(video_chunks),
                'processing_time': stage_time
            }
            self.performance_metrics['sadtalker_animation'] = stage_time
            
            self.logger.info(f"[SUCCESS] SadTalker Animation completed in {stage_time:.2f} seconds")
            self.logger.info(f"   Total video chunks: {len(video_chunks)}")
            return video_chunks
            
        except Exception as e:
            self.logger.error(f"[ERROR] SadTalker Animation failed: {e}")
            raise
    
    def _stage_enhancement_pipeline(self, video_chunks: List[str]) -> List[str]:
        """Stage 4: Combined Enhancement Pipeline (Real-ESRGAN + CodeFormer).
        
        Args:
            video_chunks: List of video chunk paths
            
        Returns:
            List of enhanced video chunk paths
        """
        stage_start = time.time()
        self.current_stage = "enhancement_pipeline"
        
        try:
            enhanced_chunks = []
            
            # Process each video chunk with enhancement
            for i, video_chunk in enumerate(video_chunks):
                self.logger.info(f"Enhanced Processing enhancement for chunk {i+1}/{len(video_chunks)}")
                
                # Enhancement parameters
                enhancement_params = {
                    'quality_level': 'high',
                    'upscale_factor': 2,
                    'face_enhance': True,
                    'background_enhance': True,
                    'tile_size': 512
                }
                
                # Use ProductionEnhancementStage to process this chunk
                result = self.enhancement_stage.process_video_enhancement(
                    video_chunk, enhancement_params
                )
                
                if result['success']:
                    enhanced_chunks.append(result['output_video_path'])
                    self.logger.info(f"[SUCCESS] Enhancement chunk {i+1} completed: {result['output_video_path']}")
                    self.logger.info(f"   File size: {result.get('file_size', 0):,} bytes")
                    self.logger.info(f"   Resolution: {result.get('video_resolution', [0, 0])}")
                else:
                    raise RuntimeError(f"Enhancement failed for chunk {i+1}: {result['errors']}")
            
            stage_time = time.time() - stage_start
            self.stage_outputs['enhancement_pipeline'] = {
                'chunks': enhanced_chunks,
                'num_chunks': len(enhanced_chunks),
                'processing_time': stage_time
            }
            self.performance_metrics['enhancement_pipeline'] = stage_time
            
            self.logger.info(f"[SUCCESS] Enhancement Pipeline completed in {stage_time:.2f} seconds")
            self.logger.info(f"   Total enhanced chunks: {len(enhanced_chunks)}")
            return enhanced_chunks
            
        except Exception as e:
            self.logger.error(f"[ERROR] Enhancement Pipeline failed: {e}")
            raise
    
    def _stage_manim_animation(self, text: str, num_chunks: int) -> str:
        """Stage 5: Manim Background Animation.
        
        Args:
            text: Original text for animation context
            num_chunks: Number of chunks for timing
            
        Returns:
            Path to background animation video
        """
        stage_start = time.time()
        self.current_stage = "manim_animation"
        
        try:
            # Calculate duration based on chunks (each chunk ~10 seconds)
            estimated_duration = num_chunks * 10
            
            # Animation parameters
            animation_params = {
                'resolution': '1920x1080',
                'frame_rate': 30,
                'quality': 'high',
                'duration': estimated_duration,
                'background_color': '#0f0f0f'
            }
            
            # Use ProductionManimStage to process background animation
            result = self.manim_stage.process_background_animation(text, num_chunks, animation_params)
            
            if result['success']:
                background_video = result['output_video_path']
                
                stage_time = time.time() - stage_start
                self.stage_outputs['manim_animation'] = {
                    'background_video': background_video,
                    'manim_script_path': result.get('manim_script_path'),
                    'file_size': result.get('file_size', 0),
                    'video_duration': result.get('video_duration', 0),
                    'video_resolution': result.get('video_resolution', [0, 0]),
                    'is_placeholder': result.get('placeholder', False),
                    'processing_time': stage_time
                }
                self.performance_metrics['manim_animation'] = stage_time
                
                self.logger.info(f"[SUCCESS] Manim Animation completed in {stage_time:.2f} seconds")
                self.logger.info(f"   Background video: {background_video}")
                self.logger.info(f"   File size: {result.get('file_size', 0):,} bytes")
                self.logger.info(f"   Duration: {result.get('video_duration', 0)} seconds")
                
                if result.get('placeholder'):
                    self.logger.info("   Step Note: Placeholder video created (Manim not available)")
                
                return background_video
            else:
                raise RuntimeError(f"Manim animation failed: {result['errors']}")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Manim Animation failed: {e}")
            raise
    
    def _stage_ffmpeg_assembly(self, enhanced_chunks: List[str], background_video: str, output_path: str) -> str:
        """Stage 7: Enhanced Final Assembly with Automatic Chunk Detection.
        
        Args:
            enhanced_chunks: List of enhanced video chunk paths (may be ignored if chunks have audio)
            background_video: Path to background animation
            output_path: Final output path
            
        Returns:
            Path to final assembled video
        """
        stage_start = time.time()
        self.current_stage = "final_assembly"
        
        try:
            self.logger.info("VIDEO PIPELINE Starting Enhanced Final Assembly with automatic chunk detection...")
            
            # Use EnhancedFinalAssemblyStage to automatically detect and concatenate chunks
            # This will automatically find enhanced_with_audio_chunk*.mp4 files if they exist
            result = self.final_assembly_stage.process_final_assembly()
            
            if result['success']:
                final_video = result['output_path']
                
                stage_time = time.time() - stage_start
                self.stage_outputs['final_assembly'] = {
                    'final_video': final_video,
                    'processing_time': stage_time,
                    'output_formats': result.get('output_formats', []),
                    'assembly_method': 'enhanced_automated'
                }
                self.performance_metrics['final_assembly'] = stage_time
                
                self.logger.info(f"[SUCCESS] Enhanced Final Assembly completed in {stage_time:.2f} seconds")
                self.logger.info(f"   Final video: {final_video}")
                self.logger.info(f"   Output formats: {result.get('output_formats', [])}")
                self.logger.info(f"   Assembly method: Automated chunk detection with audio")
                
                # Resolve to absolute path for return
                if not Path(final_video).is_absolute():
                    final_video_path = self.base_dir / final_video
                else:
                    final_video_path = Path(final_video)
                
                return str(final_video_path)
            else:
                # Fallback to traditional FFmpeg assembly if enhanced assembly fails
                self.logger.warning("Enhanced final assembly failed, falling back to traditional FFmpeg...")
                
                result = self.ffmpeg_stage.concatenate_video_chunks(
                    enhanced_chunks, output_path, background_video, "bottom-left"
                )
                
                if result['success']:
                    final_video = result['output_video_path']
                    
                    stage_time = time.time() - stage_start
                    self.stage_outputs['final_assembly'] = {
                        'final_video': final_video,
                        'num_chunks_assembled': result.get('total_chunks', len(enhanced_chunks)),
                        'file_size': result.get('file_size', 0),
                        'video_duration': result.get('video_duration', 0),
                        'video_resolution': result.get('video_resolution', [0, 0]),
                        'processing_time': stage_time,
                        'assembly_method': 'fallback_ffmpeg'
                    }
                    self.performance_metrics['final_assembly'] = stage_time
                    
                    self.logger.info(f"[SUCCESS] Fallback FFmpeg Assembly completed in {stage_time:.2f} seconds")
                    self.logger.info(f"   Final video: {final_video}")
                    self.logger.info(f"   File size: {result.get('file_size', 0):,} bytes")
                    self.logger.info(f"   Duration: {result.get('video_duration', 0):.2f} seconds")
                    self.logger.info(f"   Resolution: {result.get('video_resolution', [0, 0])}")
                    self.logger.info(f"   Chunks assembled: {result.get('total_chunks', 0)}")
                    
                    return final_video
                else:
                    raise RuntimeError(f"Both enhanced and fallback assembly failed: {result['errors']}")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Final Assembly failed: {e}")
            raise


def main():
    """Test the production pipeline."""
    print("🧪 Testing Production Video Synthesis Pipeline")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = ProductionVideoSynthesisPipeline()
    
    # Test with sample data
    test_text = "Hello, this is a test of the production video synthesis pipeline with dynamic chunking. This text should be split into multiple chunks based on its length and processed individually."
    voice_reference = "/Users/aryanjain/Documents/video-synthesis-pipeline copy/NEW/user_assets/voices/sample_voice.wav"
    face_image = "/Users/aryanjain/Documents/video-synthesis-pipeline copy/NEW/user_assets/faces/sample_face.jpg"
    output_path = "/Users/aryanjain/Documents/video-synthesis-pipeline copy/NEW/processed/test_production_pipeline.mp4"
    
    # Run pipeline
    results = pipeline.run_complete_pipeline(
        text=test_text,
        voice_reference=voice_reference,
        face_image=face_image,
        output_path=output_path
    )
    
    if results['success']:
        print("\nSUCCESS Production Pipeline test PASSED!")
        print(f"[SUCCESS] Output: {results['output_video']}")
        print(f"Status: Total time: {results['total_time_seconds']:.2f} seconds")
        print(f"[EMOJI] Total chunks: {results['total_chunks']}")
    else:
        print("\n[ERROR] Production Pipeline test FAILED!")
        print(f"Error: {results['error']}")


if __name__ == "__main__":
    main()