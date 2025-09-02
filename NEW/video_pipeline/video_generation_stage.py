#!/usr/bin/env python3
"""
Video Generation Stage for NEW Video Pipeline
Integrates SadTalker with chunking support and NEW metadata system
"""

import os
import sys
import subprocess
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

# Add paths for existing SadTalker integration
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "INTEGRATED_PIPELINE" / "src"))

from metadata_manager import MetadataManager

# Import existing chunking and processing utilities
try:
    from sadtalker_stage import SadTalkerStage
except ImportError:
    SadTalkerStage = None

try:
    # Import existing chunked processing utilities
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "useless_files" / "experimental_scripts"))
    from xtts_clean_chunker import XTTSCleanChunker
    from sync_video_concatenator import SyncVideoConcatenator
except ImportError:
    XTTSCleanChunker = None
    SyncVideoConcatenator = None


class VideoGenerationStage:
    """Video generation stage that integrates with NEW metadata system."""
    
    def __init__(self, metadata_manager: MetadataManager = None):
        """Initialize video generation stage.
        
        Args:
            metadata_manager: Metadata manager instance
        """
        self.metadata_manager = metadata_manager or MetadataManager()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Output directory
        self.output_dir = self.metadata_manager.output_dir / "video_generation"
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "chunks").mkdir(exist_ok=True)
        
        # Try to import existing SadTalker stage
        self.sadtalker_stage = None
        if SadTalkerStage:
            try:
                self.sadtalker_stage = SadTalkerStage()
                self.logger.info("[SUCCESS] SadTalker stage imported successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize SadTalker stage: {e}")
        
        # Initialize chunking utilities
        self.audio_chunker = XTTSCleanChunker() if XTTSCleanChunker else None
        self.video_concatenator = SyncVideoConcatenator() if SyncVideoConcatenator else None
    
    def validate_inputs(self) -> Dict[str, Any]:
        """Validate inputs for video generation.
        
        Returns:
            Validation results dictionary
        """
        validation = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            metadata = self.metadata_manager.load_metadata()
            if metadata is None:
                validation["valid"] = False
                validation["errors"].append("No metadata available")
                return validation
            
            # Check script content
            script_content = metadata.get("script_generated", {}).get("core_content")
            if not script_content:
                validation["valid"] = False
                validation["errors"].append("No script content available")
            elif len(script_content.strip()) < 10:
                validation["warnings"].append("Script content is very short")
            
            # Check voice cloning output
            voice_stage = self.metadata_manager.get_stage_status("voice_cloning")
            if not voice_stage or voice_stage.get("status") != "completed":
                validation["valid"] = False
                validation["errors"].append("Voice cloning stage not completed")
            else:
                voice_output = voice_stage.get("output_voice")
                if not voice_output:
                    validation["valid"] = False
                    validation["errors"].append("No voice output from voice cloning stage")
                else:
                    voice_path = self.metadata_manager.new_dir / voice_output
                    if not voice_path.exists():
                        validation["valid"] = False
                        validation["errors"].append(f"Voice file not found: {voice_output}")
            
            # Check face processing output
            face_stage = self.metadata_manager.get_stage_status("face_processing")
            if not face_stage or face_stage.get("status") != "completed":
                validation["valid"] = False
                validation["errors"].append("Face processing stage not completed")
            else:
                face_output = face_stage.get("face_crop")
                if not face_output:
                    validation["valid"] = False
                    validation["errors"].append("No face output from face processing stage")
                else:
                    face_path = self.metadata_manager.new_dir / face_output
                    if not face_path.exists():
                        validation["valid"] = False
                        validation["errors"].append(f"Face file not found: {face_output}")
            
            # Check emotion and tone parameters
            emotion = metadata.get("emotion")
            tone = metadata.get("tone")
            
            if not emotion:
                validation["warnings"].append("No emotion specified, will use default")
            if not tone:
                validation["warnings"].append("No tone specified, will use default")
            
        except Exception as e:
            validation["valid"] = False
            validation["errors"].append(f"Validation error: {str(e)}")
        
        return validation
    
    def should_use_chunking(self, voice_file_path: str) -> bool:
        """Determine if chunking should be used based on audio duration.
        
        Args:
            voice_file_path: Path to voice audio file
            
        Returns:
            True if chunking should be used, False otherwise
        """
        try:
            # Get audio duration using ffprobe
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', voice_file_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                duration = float(result.stdout.strip())
                self.logger.info(f"Audio duration: {duration:.1f} seconds")
                
                # Use chunking for audio longer than 60 seconds
                return duration > 60.0
            else:
                self.logger.warning(f"Could not determine audio duration, defaulting to chunking")
                return True
        
        except Exception as e:
            self.logger.warning(f"Error checking audio duration: {e}, defaulting to chunking")
            return True
    
    def chunk_audio_for_processing(self, voice_file_path: str, chunk_duration: int = 10) -> List[str]:
        """Chunk audio file for processing.
        
        Args:
            voice_file_path: Path to voice audio file
            chunk_duration: Duration per chunk in seconds
            
        Returns:
            List of chunk file paths
        """
        try:
            if self.audio_chunker:
                # Use existing clean chunker
                chunks_dir = self.output_dir / "chunks" / "audio"
                chunks_dir.mkdir(parents=True, exist_ok=True)
                
                chunk_paths = self.audio_chunker.create_clean_chunks(
                    voice_file_path, str(chunks_dir)
                )
                
                self.logger.info(f"[SUCCESS] Audio chunked into {len(chunk_paths)} segments")
                return chunk_paths
            else:
                # Fallback to basic FFmpeg chunking
                return self._basic_audio_chunking(voice_file_path, chunk_duration)
        
        except Exception as e:
            self.logger.error(f"Audio chunking failed: {e}")
            return []
    
    def _basic_audio_chunking(self, voice_file_path: str, chunk_duration: int) -> List[str]:
        """Basic audio chunking using FFmpeg.
        
        Args:
            voice_file_path: Path to voice audio file
            chunk_duration: Duration per chunk in seconds
            
        Returns:
            List of chunk file paths
        """
        try:
            chunks_dir = self.output_dir / "chunks" / "audio"
            chunks_dir.mkdir(parents=True, exist_ok=True)
            
            # Get total duration
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', voice_file_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise ValueError("Could not get audio duration")
            
            total_duration = float(result.stdout.strip())
            num_chunks = int(total_duration / chunk_duration) + 1
            
            chunk_paths = []
            timestamp = int(time.time())
            
            for i in range(num_chunks):
                start_time = i * chunk_duration
                if start_time >= total_duration:
                    break
                
                chunk_path = chunks_dir / f"audio_chunk_{timestamp}_{i:03d}.wav"
                
                cmd = [
                    'ffmpeg', '-y',
                    '-i', voice_file_path,
                    '-ss', str(start_time),
                    '-t', str(chunk_duration),
                    '-c', 'copy',
                    str(chunk_path)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0 and chunk_path.exists():
                    chunk_paths.append(str(chunk_path))
                else:
                    self.logger.warning(f"Failed to create chunk {i}")
            
            self.logger.info(f"[SUCCESS] Created {len(chunk_paths)} audio chunks")
            return chunk_paths
        
        except Exception as e:
            self.logger.error(f"Basic audio chunking failed: {e}")
            return []
    
    def generate_video_with_sadtalker(self, face_image_path: str, voice_file_path: str,
                                    output_path: str, emotion: str = "inspired",
                                    tone: str = "professional") -> Dict[str, Any]:
        """Generate video using SadTalker.
        
        Args:
            face_image_path: Path to face image
            voice_file_path: Path to voice audio
            output_path: Output path for generated video
            emotion: Emotion parameter
            tone: Tone parameter
            
        Returns:
            Video generation results dictionary
        """
        try:
            if self.sadtalker_stage:
                # Use existing SadTalker stage
                result = self.sadtalker_stage.generate_talking_head_video(
                    face_image=face_image_path,
                    audio_file=voice_file_path,
                    output_path=output_path,
                    emotion=emotion,
                    tone=tone
                )
                return result
            else:
                # Fallback to subprocess call
                return self._generate_with_subprocess(face_image_path, voice_file_path, output_path)
        
        except Exception as e:
            self.logger.error(f"SadTalker generation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_with_subprocess(self, face_image_path: str, voice_file_path: str,
                                output_path: str) -> Dict[str, Any]:
        """Fallback video generation using subprocess.
        
        Args:
            face_image_path: Path to face image
            voice_file_path: Path to voice audio
            output_path: Output path for generated video
            
        Returns:
            Video generation results dictionary
        """
        try:
            # Use existing SadTalker stage via subprocess
            sadtalker_script = Path(__file__).parent.parent.parent / "INTEGRATED_PIPELINE" / "src" / "sadtalker_stage.py"
            
            if not sadtalker_script.exists():
                return {"success": False, "error": "SadTalker stage script not found"}
            
            # Run SadTalker via subprocess
            cmd = [
                sys.executable, str(sadtalker_script),
                "--face-image", face_image_path,
                "--audio-file", voice_file_path,
                "--output", output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0 and Path(output_path).exists():
                return {"success": True, "output_path": output_path}
            else:
                return {"success": False, "error": result.stderr}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def process_chunked_video_generation(self, face_image_path: str, audio_chunks: List[str],
                                       emotion: str, tone: str) -> Dict[str, Any]:
        """Process video generation with audio chunks.
        
        Args:
            face_image_path: Path to face image
            audio_chunks: List of audio chunk paths
            emotion: Emotion parameter
            tone: Tone parameter
            
        Returns:
            Processing results dictionary
        """
        video_chunks = []
        chunks_dir = self.output_dir / "chunks" / "video"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        
        # Process each audio chunk
        for i, audio_chunk in enumerate(audio_chunks):
            chunk_output = chunks_dir / f"video_chunk_{timestamp}_{i:03d}.mp4"
            
            self.logger.info(f"Processing video chunk {i+1}/{len(audio_chunks)}")
            
            result = self.generate_video_with_sadtalker(
                face_image_path, audio_chunk, str(chunk_output), emotion, tone
            )
            
            if result["success"]:
                video_chunks.append(str(chunk_output))
            else:
                self.logger.error(f"Failed to process video chunk {i+1}: {result.get('error')}")
                return {
                    "success": False,
                    "error": f"Video chunk {i+1} failed: {result.get('error')}"
                }
        
        # Concatenate video chunks
        final_output = self.output_dir / f"video_combined_{timestamp}.mp4"
        concat_result = self._concatenate_video_chunks(video_chunks, final_output)
        
        if concat_result["success"]:
            return {
                "success": True,
                "output_path": str(final_output),
                "video_chunks": video_chunks,
                "chunk_count": len(video_chunks)
            }
        else:
            return concat_result
    
    def _concatenate_video_chunks(self, video_chunks: List[str], output_path: Path) -> Dict[str, Any]:
        """Concatenate video chunks using existing logic.
        
        Args:
            video_chunks: List of video chunk paths
            output_path: Output path for concatenated video
            
        Returns:
            Concatenation results dictionary
        """
        try:
            if self.video_concatenator:
                # Use existing sync video concatenator
                success = self.video_concatenator.concatenate_videos(
                    video_chunks, str(output_path)
                )
                
                if success:
                    self.logger.info(f"[SUCCESS] Video chunks concatenated: {output_path}")
                    return {"success": True}
                else:
                    return {"success": False, "error": "Video concatenation failed"}
            else:
                # Fallback to basic FFmpeg concatenation
                return self._basic_video_concatenation(video_chunks, output_path)
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _basic_video_concatenation(self, video_chunks: List[str], output_path: Path) -> Dict[str, Any]:
        """Basic video concatenation using FFmpeg.
        
        Args:
            video_chunks: List of video chunk paths
            output_path: Output path for concatenated video
            
        Returns:
            Concatenation results dictionary
        """
        try:
            # Create concat file for FFmpeg
            concat_file = output_path.with_suffix('.txt')
            with open(concat_file, 'w') as f:
                for chunk_path in video_chunks:
                    f.write(f"file '{chunk_path}'\\n")
            
            # Run FFmpeg concatenation with sync-aware settings
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_file),
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-af', 'aresample=async=1000:first_pts=0',
                '-fflags', '+shortest',
                '-avoid_negative_ts', 'make_zero',
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Cleanup concat file
            concat_file.unlink()
            
            if result.returncode == 0:
                self.logger.info(f"[SUCCESS] Video chunks concatenated: {output_path}")
                return {"success": True}
            else:
                return {"success": False, "error": result.stderr}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_emotion_parameters(self, emotion: str, tone: str) -> Dict[str, Any]:
        """Get SadTalker parameters based on emotion and tone.
        
        Args:
            emotion: Emotion parameter
            tone: Tone parameter
            
        Returns:
            SadTalker parameters dictionary
        """
        # Base parameters
        params = {
            "expression_scale": 1.0,
            "face_animation_strength": 1.0,
            "still_mode": False,
            "preprocess": "crop"
        }
        
        # Emotion-based adjustments
        emotion_adjustments = {
            "inspired": {"expression_scale": 1.2, "face_animation_strength": 1.1},
            "confident": {"expression_scale": 1.3, "face_animation_strength": 1.2},
            "curious": {"expression_scale": 0.9, "face_animation_strength": 0.8},
            "excited": {"expression_scale": 1.4, "face_animation_strength": 1.3},
            "calm": {"expression_scale": 0.8, "face_animation_strength": 0.7}
        }
        
        # Tone-based adjustments
        tone_adjustments = {
            "professional": {"expression_scale": 1.0, "face_animation_strength": 0.9},
            "friendly": {"expression_scale": 1.1, "face_animation_strength": 1.0},
            "motivational": {"expression_scale": 1.3, "face_animation_strength": 1.2},
            "casual": {"expression_scale": 1.2, "face_animation_strength": 1.1}
        }
        
        # Apply adjustments
        if emotion in emotion_adjustments:
            params.update(emotion_adjustments[emotion])
        
        if tone in tone_adjustments:
            for key, value in tone_adjustments[tone].items():
                params[key] = (params[key] + value) / 2  # Average with emotion adjustment
        
        return params
    
    def process_video_generation(self) -> Dict[str, Any]:
        """Process video generation stage using metadata.
        
        Returns:
            Processing results dictionary
        """
        start_time = time.time()
        
        # Update status to processing
        self.metadata_manager.update_stage_status("video_generation", "processing")
        
        try:
            # Load metadata
            metadata = self.metadata_manager.load_metadata()
            if metadata is None:
                raise ValueError("Failed to load metadata")
            
            # Get input paths from metadata
            voice_file_path = self.metadata_manager.get_stage_input("video_generation", "input_voice")
            face_image_path = self.metadata_manager.get_stage_input("video_generation", "input_face")
            
            if voice_file_path is None:
                raise ValueError("Voice file not found in metadata")
            
            if face_image_path is None:
                raise ValueError("Face image not found in metadata")
            
            # Get emotion and tone parameters
            emotion = metadata.get("emotion", "inspired")
            tone = metadata.get("tone", "professional")
            
            self.logger.info(f"Processing video generation:")
            self.logger.info(f"  Voice file: {voice_file_path}")
            self.logger.info(f"  Face image: {face_image_path}")
            self.logger.info(f"  Emotion: {emotion}, Tone: {tone}")
            
            # Check if chunking is needed
            pipeline_config = metadata.get("pipeline_config", {})
            auto_chunking = pipeline_config.get("auto_chunking", True)
            chunk_duration = pipeline_config.get("chunk_duration", 10)
            
            if auto_chunking and self.should_use_chunking(voice_file_path):
                # Process with chunking
                self.logger.info("[EMOJI] Using chunked processing for long audio")
                
                # Chunk audio
                audio_chunks = self.chunk_audio_for_processing(voice_file_path, chunk_duration)
                
                if not audio_chunks:
                    raise ValueError("Failed to create audio chunks")
                
                # Process chunked video generation
                result = self.process_chunked_video_generation(
                    face_image_path, audio_chunks, emotion, tone
                )
            else:
                # Process as single file
                self.logger.info("[EMOJI] Using single file processing")
                
                timestamp = int(time.time())
                output_path = self.output_dir / f"video_single_{timestamp}.mp4"
                
                result = self.generate_video_with_sadtalker(
                    face_image_path, voice_file_path, str(output_path), emotion, tone
                )
                
                if result["success"]:
                    result.update({
                        "video_chunks": [],
                        "chunk_count": 1
                    })
            
            processing_time = time.time() - start_time
            
            if result["success"]:
                # Convert to relative path
                output_path = Path(result["output_path"])
                if output_path.is_absolute():
                    try:
                        relative_path = output_path.relative_to(self.metadata_manager.new_dir)
                        result["output_path"] = str(relative_path)
                    except ValueError:
                        pass  # Keep absolute path if outside NEW directory
                
                # Update metadata with success
                update_data = {
                    "combined_video": result["output_path"],
                    "video_chunks": result.get("video_chunks", []),
                    "chunk_count": result.get("chunk_count", 0),
                    "processing_time": processing_time,
                    "error": None
                }
                
                self.metadata_manager.update_stage_status("video_generation", "completed", update_data)
                
                self.logger.info(f"[SUCCESS] Video generation completed in {processing_time:.1f}s")
                return result
            else:
                # Update metadata with failure
                update_data = {
                    "processing_time": processing_time,
                    "error": result.get("error")
                }
                
                self.metadata_manager.update_stage_status("video_generation", "failed", update_data)
                
                self.logger.error(f"[ERROR] Video generation failed: {result.get('error')}")
                return result
        
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            # Update metadata with error
            update_data = {
                "processing_time": processing_time,
                "error": error_msg
            }
            
            self.metadata_manager.update_stage_status("video_generation", "failed", update_data)
            
            self.logger.error(f"[ERROR] Video generation stage failed: {error_msg}")
            return {"success": False, "error": error_msg}


def main():
    """Test video generation stage."""
    stage = VideoGenerationStage()
    
    # Check metadata status
    metadata = stage.metadata_manager.load_metadata()
    if metadata:
        print("[SUCCESS] Metadata loaded successfully")
        
        # Check if video generation is ready
        voice_file = stage.metadata_manager.get_stage_input("video_generation", "input_voice")
        face_image = stage.metadata_manager.get_stage_input("video_generation", "input_face")
        
        if voice_file and face_image:
            print(f"Recording Voice file found: {voice_file}")
            print(f"[EMOJI] Face image found: {face_image}")
            
            # Process video generation
            result = stage.process_video_generation()
            
            if result["success"]:
                print("[SUCCESS] Video generation completed successfully")
                print(f"Output: {result.get('output_path')}")
                print(f"Chunks: {result.get('chunk_count', 0)}")
            else:
                print(f"[ERROR] Video generation failed: {result.get('error')}")
        else:
            print("[ERROR] Required inputs not found in metadata")
            print(f"  Voice file: {voice_file}")
            print(f"  Face image: {face_image}")
    else:
        print("[ERROR] Failed to load metadata")


if __name__ == "__main__":
    main()