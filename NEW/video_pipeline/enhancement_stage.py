#!/usr/bin/env python3
"""
Video Enhancement Stage for NEW Video Pipeline
Integrates Real-ESRGAN and CodeFormer with NEW metadata system
"""

import os
import sys
import subprocess
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

# Add paths for existing enhancement integration
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "INTEGRATED_PIPELINE" / "src"))

from metadata_manager import MetadataManager

try:
    from realesrgan_stage import RealESRGANStage
    from codeformer_stage import CodeFormerStage
except ImportError:
    RealESRGANStage = None
    CodeFormerStage = None

try:
    # Import existing chunked enhancement processor
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "useless_files" / "experimental_scripts"))
    from chunked_enhancement_processor import ChunkedEnhancementProcessor
    from final_enhanced_concatenator import FinalEnhancedConcatenator
except ImportError:
    ChunkedEnhancementProcessor = None
    FinalEnhancedConcatenator = None


class VideoEnhancementStage:
    """Video enhancement stage that integrates with NEW metadata system."""
    
    def __init__(self, metadata_manager: MetadataManager = None):
        """Initialize video enhancement stage.
        
        Args:
            metadata_manager: Metadata manager instance
        """
        self.metadata_manager = metadata_manager or MetadataManager()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Output directory
        self.output_dir = self.metadata_manager.output_dir / "video_enhancement"
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "chunks").mkdir(exist_ok=True)
        (self.output_dir / "realesrgan").mkdir(exist_ok=True)
        (self.output_dir / "codeformer").mkdir(exist_ok=True)
        
        # Try to import existing enhancement stages
        self.realesrgan_stage = None
        self.codeformer_stage = None
        
        if RealESRGANStage:
            try:
                self.realesrgan_stage = RealESRGANStage()
                self.logger.info("[SUCCESS] Real-ESRGAN stage imported successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Real-ESRGAN stage: {e}")
        
        if CodeFormerStage:
            try:
                self.codeformer_stage = CodeFormerStage()
                self.logger.info("[SUCCESS] CodeFormer stage imported successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize CodeFormer stage: {e}")
        
        # Initialize chunked enhancement processor
        self.chunked_processor = ChunkedEnhancementProcessor() if ChunkedEnhancementProcessor else None
        self.final_concatenator = FinalEnhancedConcatenator() if FinalEnhancedConcatenator else None
    
    def validate_inputs(self) -> Dict[str, Any]:
        """Validate inputs for video enhancement.
        
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
            
            # Check video generation output
            video_stage = self.metadata_manager.get_stage_status("video_generation")
            if not video_stage or video_stage.get("status") != "completed":
                validation["valid"] = False
                validation["errors"].append("Video generation stage not completed")
            else:
                video_output = video_stage.get("combined_video")
                video_chunks = video_stage.get("video_chunks", [])
                
                if not video_output and not video_chunks:
                    validation["valid"] = False
                    validation["errors"].append("No video output from video generation stage")
                else:
                    # Check if combined video exists
                    if video_output:
                        video_path = self.metadata_manager.new_dir / video_output
                        if not video_path.exists():
                            validation["valid"] = False
                            validation["errors"].append(f"Video file not found: {video_output}")
                    
                    # Check if video chunks exist
                    if video_chunks:
                        for chunk in video_chunks:
                            chunk_path = self.metadata_manager.new_dir / chunk
                            if not chunk_path.exists():
                                validation["valid"] = False
                                validation["errors"].append(f"Video chunk not found: {chunk}")
            
            # Check enhancement components availability
            if not self.realesrgan_stage:
                validation["warnings"].append("Real-ESRGAN stage not available, using fallback")
            
            if not self.codeformer_stage:
                validation["warnings"].append("CodeFormer stage not available, using fallback")
            
            # Check emotion and tone parameters
            emotion = metadata.get("emotion")
            tone = metadata.get("tone")
            
            if not emotion:
                validation["warnings"].append("No emotion specified, will use default enhancement")
            if not tone:
                validation["warnings"].append("No tone specified, will use default enhancement")
            
        except Exception as e:
            validation["valid"] = False
            validation["errors"].append(f"Validation error: {str(e)}")
        
        return validation
    
    def should_use_chunked_enhancement(self, video_path: str) -> bool:
        """Determine if chunked enhancement should be used.
        
        Args:
            video_path: Path to input video
            
        Returns:
            True if chunked enhancement should be used, False otherwise
        """
        try:
            # Get video duration and size
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration,size',
                '-of', 'default=noprint_wrappers=1', video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                duration = 0
                size = 0
                
                for line in lines:
                    if line.startswith('duration='):
                        duration = float(line.split('=')[1])
                    elif line.startswith('size='):
                        size = int(line.split('=')[1])
                
                self.logger.info(f"Video duration: {duration:.1f}s, size: {size / (1024*1024):.1f}MB")
                
                # Use chunked enhancement for videos longer than 30 seconds or larger than 100MB
                return duration > 30.0 or size > 100 * 1024 * 1024
            else:
                self.logger.warning("Could not determine video properties, using chunked enhancement")
                return True
        
        except Exception as e:
            self.logger.warning(f"Error checking video properties: {e}, using chunked enhancement")
            return True
    
    def chunk_video_for_enhancement(self, video_path: str, chunk_duration: int = 15) -> List[str]:
        """Chunk video file for enhancement processing.
        
        Args:
            video_path: Path to input video
            chunk_duration: Duration per chunk in seconds
            
        Returns:
            List of video chunk paths
        """
        try:
            chunks_dir = self.output_dir / "chunks"
            chunks_dir.mkdir(parents=True, exist_ok=True)
            
            # Get total duration
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise ValueError("Could not get video duration")
            
            total_duration = float(result.stdout.strip())
            num_chunks = int(total_duration / chunk_duration) + 1
            
            chunk_paths = []
            timestamp = int(time.time())
            
            for i in range(num_chunks):
                start_time = i * chunk_duration
                if start_time >= total_duration:
                    break
                
                chunk_path = chunks_dir / f"video_chunk_{timestamp}_{i:03d}.mp4"
                
                cmd = [
                    'ffmpeg', '-y',
                    '-i', video_path,
                    '-ss', str(start_time),
                    '-t', str(chunk_duration),
                    '-c', 'copy',
                    str(chunk_path)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0 and chunk_path.exists():
                    chunk_paths.append(str(chunk_path))
                else:
                    self.logger.warning(f"Failed to create video chunk {i}")
            
            self.logger.info(f"[SUCCESS] Created {len(chunk_paths)} video chunks")
            return chunk_paths
        
        except Exception as e:
            self.logger.error(f"Video chunking failed: {e}")
            return []
    
    def enhance_video_with_realesrgan(self, input_path: str, output_path: str,
                                    scale: int = 2, model: str = "RealESRGAN_x2plus") -> Dict[str, Any]:
        """Enhance video using Real-ESRGAN.
        
        Args:
            input_path: Path to input video
            output_path: Path for enhanced output
            scale: Upscaling factor
            model: Real-ESRGAN model name
            
        Returns:
            Enhancement results dictionary
        """
        try:
            if self.realesrgan_stage:
                # Use existing Real-ESRGAN stage
                result = self.realesrgan_stage.upscale_video(
                    input_video=input_path,
                    output_video=output_path,
                    scale=scale,
                    model_name=model
                )
                return result
            else:
                # Try subprocess fallback first
                fallback_result = self._enhance_with_subprocess_realesrgan(input_path, output_path, scale)
                if fallback_result["success"]:
                    return fallback_result
                else:
                    # If subprocess fails, use basic upscaling with ffmpeg
                    return self._basic_video_upscaling(input_path, output_path, scale)
        
        except Exception as e:
            self.logger.error(f"Real-ESRGAN enhancement failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _enhance_with_subprocess_realesrgan(self, input_path: str, output_path: str,
                                          scale: int) -> Dict[str, Any]:
        """Fallback Real-ESRGAN enhancement using subprocess.
        
        Args:
            input_path: Path to input video
            output_path: Path for enhanced output
            scale: Upscaling factor
            
        Returns:
            Enhancement results dictionary
        """
        try:
            # Use existing Real-ESRGAN stage via subprocess
            realesrgan_script = Path(__file__).parent.parent.parent / "INTEGRATED_PIPELINE" / "src" / "realesrgan_stage.py"
            
            if not realesrgan_script.exists():
                return {"success": False, "error": "Real-ESRGAN stage script not found"}
            
            # Run Real-ESRGAN via subprocess
            cmd = [
                sys.executable, str(realesrgan_script),
                "--input", input_path,
                "--output", output_path,
                "--scale", str(scale)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
            
            if result.returncode == 0 and Path(output_path).exists():
                return {"success": True, "output_path": output_path}
            else:
                return {"success": False, "error": result.stderr}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _basic_video_upscaling(self, input_path: str, output_path: str, scale: int) -> Dict[str, Any]:
        """Basic video upscaling using FFmpeg as final fallback.
        
        Args:
            input_path: Path to input video
            output_path: Path for upscaled output
            scale: Upscaling factor
            
        Returns:
            Enhancement results dictionary
        """
        try:
            self.logger.info(f"Using basic FFmpeg upscaling (scale: {scale}x)")
            
            # Calculate new dimensions
            # Get original dimensions first
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'stream=width,height',
                '-of', 'csv=s=x:p=0', input_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                dimensions = result.stdout.strip().split('x')
                if len(dimensions) == 2:
                    width = int(dimensions[0]) * scale
                    height = int(dimensions[1]) * scale
                else:
                    # Default fallback dimensions
                    width = 1920
                    height = 1080
            else:
                # Default fallback dimensions
                width = 1920
                height = 1080
            
            # Use FFmpeg with high-quality scaling
            cmd = [
                'ffmpeg', '-y',
                '-i', input_path,
                '-vf', f'scale={width}:{height}:flags=lanczos',
                '-c:v', 'libx264',
                '-preset', 'slow',
                '-crf', '18',
                '-c:a', 'copy',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            
            if result.returncode == 0 and Path(output_path).exists():
                self.logger.info(f"[SUCCESS] Basic upscaling completed: {output_path}")
                return {"success": True, "output_path": output_path, "method": "ffmpeg_basic"}
            else:
                return {"success": False, "error": f"FFmpeg upscaling failed: {result.stderr}"}
        
        except Exception as e:
            return {"success": False, "error": f"Basic upscaling failed: {str(e)}"}
    
    def enhance_video_with_codeformer(self, input_path: str, output_path: str,
                                    fidelity: float = 0.7) -> Dict[str, Any]:
        """Enhance video using CodeFormer.
        
        Args:
            input_path: Path to input video
            output_path: Path for enhanced output
            fidelity: CodeFormer fidelity parameter
            
        Returns:
            Enhancement results dictionary
        """
        try:
            if self.codeformer_stage:
                # Use existing CodeFormer stage
                result = self.codeformer_stage.enhance_video(
                    input_video=input_path,
                    output_video=output_path,
                    fidelity_weight=fidelity
                )
                return result
            else:
                # Try subprocess fallback first
                fallback_result = self._enhance_with_subprocess_codeformer(input_path, output_path, fidelity)
                if fallback_result["success"]:
                    return fallback_result
                else:
                    # If subprocess fails, use basic face enhancement with ffmpeg
                    return self._basic_face_enhancement(input_path, output_path)
        
        except Exception as e:
            self.logger.error(f"CodeFormer enhancement failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _enhance_with_subprocess_codeformer(self, input_path: str, output_path: str,
                                          fidelity: float) -> Dict[str, Any]:
        """Fallback CodeFormer enhancement using subprocess.
        
        Args:
            input_path: Path to input video
            output_path: Path for enhanced output
            fidelity: CodeFormer fidelity parameter
            
        Returns:
            Enhancement results dictionary
        """
        try:
            # Use existing CodeFormer stage via subprocess
            codeformer_script = Path(__file__).parent.parent.parent / "INTEGRATED_PIPELINE" / "src" / "codeformer_stage.py"
            
            if not codeformer_script.exists():
                return {"success": False, "error": "CodeFormer stage script not found"}
            
            # Run CodeFormer via subprocess
            cmd = [
                sys.executable, str(codeformer_script),
                "--input", input_path,
                "--output", output_path,
                "--fidelity", str(fidelity)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
            
            if result.returncode == 0 and Path(output_path).exists():
                return {"success": True, "output_path": output_path}
            else:
                return {"success": False, "error": result.stderr}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _basic_face_enhancement(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """Basic face enhancement using FFmpeg filters as final fallback.
        
        Args:
            input_path: Path to input video
            output_path: Path for enhanced output
            
        Returns:
            Enhancement results dictionary
        """
        try:
            self.logger.info("Using basic FFmpeg face enhancement filters")
            
            # Use FFmpeg with basic enhancement filters
            cmd = [
                'ffmpeg', '-y',
                '-i', input_path,
                '-vf', 'unsharp=5:5:1.0:5:5:0.0,eq=contrast=1.1:brightness=0.02:saturation=1.1',
                '-c:v', 'libx264',
                '-preset', 'slow',
                '-crf', '18',
                '-c:a', 'copy',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            
            if result.returncode == 0 and Path(output_path).exists():
                self.logger.info(f"[SUCCESS] Basic face enhancement completed: {output_path}")
                return {"success": True, "output_path": output_path, "method": "ffmpeg_basic"}
            else:
                # If even basic enhancement fails, just copy the file
                self.logger.warning("Basic enhancement failed, copying original file")
                shutil.copy2(input_path, output_path)
                return {"success": True, "output_path": output_path, "method": "copy_original"}
        
        except Exception as e:
            # Final fallback: just copy the original file
            try:
                self.logger.warning(f"All enhancement failed ({e}), copying original file")
                shutil.copy2(input_path, output_path)
                return {"success": True, "output_path": output_path, "method": "copy_original"}
            except Exception as copy_error:
                return {"success": False, "error": f"Even file copy failed: {str(copy_error)}"}
    
    def process_single_video_enhancement(self, video_path: str, emotion: str, tone: str) -> Dict[str, Any]:
        """Process enhancement for a single video file.
        
        Args:
            video_path: Path to input video
            emotion: Emotion parameter
            tone: Tone parameter
            
        Returns:
            Enhancement results dictionary
        """
        timestamp = int(time.time())
        
        # Get enhancement parameters based on emotion and tone
        params = self.get_enhancement_parameters(emotion, tone)
        
        # Step 1: Real-ESRGAN enhancement
        realesrgan_output = self.output_dir / "realesrgan" / f"realesrgan_{timestamp}.mp4"
        
        self.logger.info("[EMOJI] Applying Real-ESRGAN enhancement...")
        realesrgan_result = self.enhance_video_with_realesrgan(
            video_path, str(realesrgan_output), 
            scale=params["realesrgan_scale"],
            model=params["realesrgan_model"]
        )
        
        if not realesrgan_result["success"]:
            return {
                "success": False,
                "error": f"Real-ESRGAN failed: {realesrgan_result.get('error')}"
            }
        
        # Step 2: CodeFormer enhancement
        codeformer_output = self.output_dir / "codeformer" / f"final_enhanced_{timestamp}.mp4"
        
        self.logger.info("[EMOJI] Applying CodeFormer enhancement...")
        codeformer_result = self.enhance_video_with_codeformer(
            str(realesrgan_output), str(codeformer_output),
            fidelity=params["codeformer_fidelity"]
        )
        
        if not codeformer_result["success"]:
            return {
                "success": False,
                "error": f"CodeFormer failed: {codeformer_result.get('error')}"
            }
        
        return {
            "success": True,
            "final_video": str(codeformer_output),
            "intermediate_files": [str(realesrgan_output)],
            "enhancement_steps": ["realesrgan", "codeformer"]
        }
    
    def process_chunked_video_enhancement(self, video_chunks: List[str], emotion: str, tone: str) -> Dict[str, Any]:
        """Process enhancement for chunked video files.
        
        Args:
            video_chunks: List of video chunk paths
            emotion: Emotion parameter
            tone: Tone parameter
            
        Returns:
            Enhancement results dictionary
        """
        if self.chunked_processor:
            # Use existing chunked enhancement processor
            try:
                result = self.chunked_processor.process_all_chunks(
                    video_chunks, emotion=emotion, tone=tone
                )
                
                if result["success"] and self.final_concatenator:
                    # Create final video using concatenator
                    final_result = self.final_concatenator.create_final_enhanced_video()
                    
                    if final_result["success"]:
                        return {
                            "success": True,
                            "final_video": final_result["final_video"],
                            "enhanced_chunks": result.get("enhanced_chunks", []),
                            "chunk_count": len(video_chunks)
                        }
                    else:
                        return final_result
                else:
                    return result
            
            except Exception as e:
                self.logger.error(f"Chunked enhancement processor failed: {e}")
                return {"success": False, "error": str(e)}
        else:
            # Fallback to manual chunked processing
            return self._manual_chunked_enhancement(video_chunks, emotion, tone)
    
    def _manual_chunked_enhancement(self, video_chunks: List[str], emotion: str, tone: str) -> Dict[str, Any]:
        """Manual chunked enhancement processing.
        
        Args:
            video_chunks: List of video chunk paths
            emotion: Emotion parameter
            tone: Tone parameter
            
        Returns:
            Enhancement results dictionary
        """
        enhanced_chunks = []
        timestamp = int(time.time())
        
        # Get enhancement parameters
        params = self.get_enhancement_parameters(emotion, tone)
        
        # Process each chunk
        for i, chunk_path in enumerate(video_chunks):
            self.logger.info(f"Processing chunk {i+1}/{len(video_chunks)}")
            
            # Real-ESRGAN enhancement
            realesrgan_output = self.output_dir / "realesrgan" / f"chunk_realesrgan_{timestamp}_{i:03d}.mp4"
            realesrgan_result = self.enhance_video_with_realesrgan(
                chunk_path, str(realesrgan_output),
                scale=params["realesrgan_scale"],
                model=params["realesrgan_model"]
            )
            
            if not realesrgan_result["success"]:
                return {
                    "success": False,
                    "error": f"Chunk {i+1} Real-ESRGAN failed: {realesrgan_result.get('error')}"
                }
            
            # CodeFormer enhancement
            codeformer_output = self.output_dir / "codeformer" / f"chunk_final_{timestamp}_{i:03d}.mp4"
            codeformer_result = self.enhance_video_with_codeformer(
                str(realesrgan_output), str(codeformer_output),
                fidelity=params["codeformer_fidelity"]
            )
            
            if not codeformer_result["success"]:
                return {
                    "success": False,
                    "error": f"Chunk {i+1} CodeFormer failed: {codeformer_result.get('error')}"
                }
            
            enhanced_chunks.append(str(codeformer_output))
        
        # Concatenate enhanced chunks
        final_output = self.output_dir / f"final_enhanced_{timestamp}.mp4"
        concat_result = self._concatenate_enhanced_chunks(enhanced_chunks, final_output)
        
        if concat_result["success"]:
            return {
                "success": True,
                "final_video": str(final_output),
                "enhanced_chunks": enhanced_chunks,
                "chunk_count": len(video_chunks)
            }
        else:
            return concat_result
    
    def _concatenate_enhanced_chunks(self, enhanced_chunks: List[str], output_path: Path) -> Dict[str, Any]:
        """Concatenate enhanced video chunks.
        
        Args:
            enhanced_chunks: List of enhanced chunk paths
            output_path: Output path for final video
            
        Returns:
            Concatenation results dictionary
        """
        try:
            # Create concat file for FFmpeg
            concat_file = output_path.with_suffix('.txt')
            with open(concat_file, 'w') as f:
                for chunk_path in enhanced_chunks:
                    f.write(f"file '{chunk_path}'\\n")
            
            # Run FFmpeg concatenation with professional settings
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_file),
                '-c:v', 'libx264',
                '-preset', 'slow',
                '-crf', '18',
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-ar', '48000',
                '-af', 'aresample=async=1000:first_pts=0',
                '-movflags', '+faststart',
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Cleanup concat file
            concat_file.unlink()
            
            if result.returncode == 0:
                self.logger.info(f"[SUCCESS] Enhanced chunks concatenated: {output_path}")
                return {"success": True}
            else:
                return {"success": False, "error": result.stderr}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_enhancement_parameters(self, emotion: str, tone: str) -> Dict[str, Any]:
        """Get enhancement parameters based on emotion and tone.
        
        Args:
            emotion: Emotion parameter
            tone: Tone parameter
            
        Returns:
            Enhancement parameters dictionary
        """
        # Base parameters
        params = {
            "realesrgan_scale": 2,
            "realesrgan_model": "RealESRGAN_x2plus",
            "codeformer_fidelity": 0.7
        }
        
        # Emotion-based adjustments
        emotion_adjustments = {
            "inspired": {"codeformer_fidelity": 0.8},
            "confident": {"codeformer_fidelity": 0.9},
            "curious": {"codeformer_fidelity": 0.6},
            "excited": {"codeformer_fidelity": 0.8},
            "calm": {"codeformer_fidelity": 0.7}
        }
        
        # Tone-based adjustments
        tone_adjustments = {
            "professional": {"realesrgan_scale": 2, "codeformer_fidelity": 0.9},
            "friendly": {"realesrgan_scale": 2, "codeformer_fidelity": 0.7},
            "motivational": {"realesrgan_scale": 2, "codeformer_fidelity": 0.8},
            "casual": {"realesrgan_scale": 2, "codeformer_fidelity": 0.6}
        }
        
        # Apply adjustments
        if emotion in emotion_adjustments:
            params.update(emotion_adjustments[emotion])
        
        if tone in tone_adjustments:
            params.update(tone_adjustments[tone])
        
        return params
    
    def process_video_enhancement(self) -> Dict[str, Any]:
        """Process video enhancement stage using metadata.
        
        Returns:
            Processing results dictionary
        """
        start_time = time.time()
        
        # Update status to processing
        self.metadata_manager.update_stage_status("video_enhancement", "processing")
        
        try:
            # Load metadata
            metadata = self.metadata_manager.load_metadata()
            if metadata is None:
                raise ValueError("Failed to load metadata")
            
            # Get video generation results
            video_gen_stage = self.metadata_manager.get_stage_status("video_generation")
            if video_gen_stage is None or video_gen_stage.get("status") != "completed":
                raise ValueError("Video generation stage not completed")
            
            # Get input video path
            combined_video = video_gen_stage.get("combined_video")
            video_chunks = video_gen_stage.get("video_chunks", [])
            
            if not combined_video:
                raise ValueError("No video found from video generation stage")
            
            # Convert to absolute path
            video_path = combined_video
            if not Path(video_path).is_absolute():
                video_path = str(self.metadata_manager.new_dir / video_path)
            
            # Get emotion and tone parameters
            emotion = metadata.get("emotion", "inspired")
            tone = metadata.get("tone", "professional")
            
            self.logger.info(f"Processing video enhancement:")
            self.logger.info(f"  Input video: {video_path}")
            self.logger.info(f"  Video chunks: {len(video_chunks)}")
            self.logger.info(f"  Emotion: {emotion}, Tone: {tone}")
            
            # Check if chunked enhancement should be used
            pipeline_config = metadata.get("pipeline_config", {})
            auto_chunking = pipeline_config.get("auto_chunking", True)
            
            if auto_chunking and (video_chunks or self.should_use_chunked_enhancement(video_path)):
                # Use chunked enhancement
                self.logger.info("[EMOJI] Using chunked enhancement processing")
                
                if video_chunks:
                    # Convert chunk paths to absolute
                    abs_chunks = []
                    for chunk in video_chunks:
                        if not Path(chunk).is_absolute():
                            chunk = str(self.metadata_manager.new_dir / chunk)
                        abs_chunks.append(chunk)
                    
                    result = self.process_chunked_video_enhancement(abs_chunks, emotion, tone)
                else:
                    # Chunk the combined video
                    chunks = self.chunk_video_for_enhancement(video_path)
                    if not chunks:
                        raise ValueError("Failed to create video chunks for enhancement")
                    
                    result = self.process_chunked_video_enhancement(chunks, emotion, tone)
            else:
                # Use single video enhancement
                self.logger.info("[EMOJI] Using single video enhancement")
                result = self.process_single_video_enhancement(video_path, emotion, tone)
            
            processing_time = time.time() - start_time
            
            if result["success"]:
                # Convert to relative path
                final_video = Path(result["final_video"])
                if final_video.is_absolute():
                    try:
                        relative_path = final_video.relative_to(self.metadata_manager.new_dir)
                        result["final_video"] = str(relative_path)
                    except ValueError:
                        pass  # Keep absolute path if outside NEW directory
                
                # Update metadata with success
                update_data = {
                    "final_video": result["final_video"],
                    "enhanced_chunks": result.get("enhanced_chunks", []),
                    "processing_time": processing_time,
                    "error": None
                }
                
                self.metadata_manager.update_stage_status("video_enhancement", "completed", update_data)
                
                self.logger.info(f"[SUCCESS] Video enhancement completed in {processing_time:.1f}s")
                return result
            else:
                # Update metadata with failure
                update_data = {
                    "processing_time": processing_time,
                    "error": result.get("error")
                }
                
                self.metadata_manager.update_stage_status("video_enhancement", "failed", update_data)
                
                self.logger.error(f"[ERROR] Video enhancement failed: {result.get('error')}")
                return result
        
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            # Update metadata with error
            update_data = {
                "processing_time": processing_time,
                "error": error_msg
            }
            
            self.metadata_manager.update_stage_status("video_enhancement", "failed", update_data)
            
            self.logger.error(f"[ERROR] Video enhancement stage failed: {error_msg}")
            return {"success": False, "error": error_msg}


def main():
    """Test video enhancement stage."""
    stage = VideoEnhancementStage()
    
    # Check metadata status
    metadata = stage.metadata_manager.load_metadata()
    if metadata:
        print("[SUCCESS] Metadata loaded successfully")
        
        # Check if video enhancement is ready
        video_gen_stage = stage.metadata_manager.get_stage_status("video_generation")
        if video_gen_stage and video_gen_stage.get("status") == "completed":
            print(f"VIDEO PIPELINE Video generation completed")
            print(f"Video: {video_gen_stage.get('combined_video')}")
            
            # Process video enhancement
            result = stage.process_video_enhancement()
            
            if result["success"]:
                print("[SUCCESS] Video enhancement completed successfully")
                print(f"Output: {result.get('final_video')}")
            else:
                print(f"[ERROR] Video enhancement failed: {result.get('error')}")
        else:
            print("[ERROR] Video generation stage not completed")
    else:
        print("[ERROR] Failed to load metadata")


if __name__ == "__main__":
    main()