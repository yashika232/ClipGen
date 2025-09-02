#!/usr/bin/env python3
"""
Stage 4: Video Generation - Simple SadTalker Integration
Generates talking head video from face image and audio
"""

import os
import sys
import json
import time
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleVideoGeneration:
    """Simple video generation using SadTalker."""
    
    def __init__(self):
        """Initialize video generation."""
        self.project_root = PROJECT_ROOT
        self.outputs_dir = self.project_root / "outputs"
        self.outputs_dir.mkdir(exist_ok=True)
        
        # Load session data
        self.session_data = self._load_session_data()
        
        logger.info("VIDEO PIPELINE Simple Video Generation initialized")
    
    def _load_session_data(self) -> Dict[str, Any]:
        """Load session data from JSON file."""
        session_file = self.outputs_dir / "session_data.json"
        if session_file.exists():
            with open(session_file, 'r') as f:
                return json.load(f)
        return {}
    
    def generate_video(self) -> bool:
        """Generate talking head video using SadTalker."""
        try:
            logger.info("Target: Starting video generation...")
            
            # Check required inputs
            face_crop_path = self.outputs_dir / "face_crop.jpg"
            audio_path = self.outputs_dir / "synthesized_speech.wav"
            
            if not face_crop_path.exists():
                raise FileNotFoundError(f"Face crop not found: {face_crop_path}")
            
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Try to use existing SadTalker implementation
            success = self._generate_with_sadtalker(face_crop_path, audio_path)
            
            if not success:
                # Fallback to simple video creation
                logger.warning("Using fallback video generation...")
                success = self._generate_with_fallback(face_crop_path, audio_path)
            
            return success
            
        except Exception as e:
            logger.error(f"[ERROR] Video generation failed: {str(e)}")
            return False
    
    def _generate_with_sadtalker(self, face_path: Path, audio_path: Path) -> bool:
        """Generate video using SadTalker."""
        try:
            # Check if SadTalker is available
            sadtalker_path = self.project_root / "models" / "SadTalker"
            if not sadtalker_path.exists():
                sadtalker_path = self.project_root / "models" / "SadTalker_PIRender"
            
            if not sadtalker_path.exists():
                logger.warning("SadTalker model not found, using fallback")
                return False
            
            # Create SadTalker output directory
            sadtalker_output = self.outputs_dir / "sadtalker_output"
            sadtalker_output.mkdir(exist_ok=True)
            
            # Build SadTalker command
            inference_script = sadtalker_path / "inference.py"
            if not inference_script.exists():
                inference_script = sadtalker_path / "src" / "inference.py"
            
            if not inference_script.exists():
                logger.warning("SadTalker inference script not found")
                return False
            
            cmd = [
                "python", str(inference_script),
                "--driven_audio", str(audio_path),
                "--source_image", str(face_path),
                "--result_dir", str(sadtalker_output),
                "--still", "--preprocess", "full"
            ]
            
            logger.info(f"Running SadTalker: {' '.join(cmd)}")
            
            # Run SadTalker
            result = subprocess.run(
                cmd,
                cwd=str(sadtalker_path),
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                # Find generated video
                generated_videos = list(sadtalker_output.glob("**/*.mp4"))
                if generated_videos:
                    generated_video = generated_videos[0]
                    
                    # Copy to standard output location
                    output_path = self.outputs_dir / "talking_head.mp4"
                    import shutil
                    shutil.copy2(str(generated_video), str(output_path))
                    
                    logger.info(f"[SUCCESS] Video generated successfully with SadTalker")
                    logger.info(f"   Output: {output_path}")
                    
                    # Save video generation metadata
                    video_metadata = {
                        "generated_at": time.time(),
                        "face_image_path": str(face_path),
                        "audio_path": str(audio_path),
                        "output_video_path": str(output_path),
                        "generation_method": "sadtalker",
                        "generation_success": True
                    }
                    
                    metadata_file = self.outputs_dir / "video_generation_metadata.json"
                    with open(metadata_file, 'w') as f:
                        json.dump(video_metadata, f, indent=2)
                    
                    return True
                else:
                    logger.error("SadTalker completed but no video found")
                    return False
            else:
                logger.error(f"SadTalker failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"SadTalker video generation failed: {str(e)}")
            return False
    
    def _generate_with_fallback(self, face_path: Path, audio_path: Path) -> bool:
        """Fallback video generation using FFmpeg."""
        try:
            # Create a simple video from static image and audio
            output_path = self.outputs_dir / "talking_head.mp4"
            
            # Use FFmpeg to create video from image and audio
            cmd = [
                "ffmpeg", "-y",
                "-loop", "1",
                "-i", str(face_path),
                "-i", str(audio_path),
                "-c:v", "libx264",
                "-c:a", "aac",
                "-b:a", "192k",
                "-r", "25",
                "-shortest",
                str(output_path)
            ]
            
            logger.info(f"Running FFmpeg fallback: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            
            if result.returncode == 0 and output_path.exists():
                logger.info(f"[SUCCESS] Video generated successfully with FFmpeg fallback")
                logger.info(f"   Output: {output_path}")
                
                # Save video generation metadata
                video_metadata = {
                    "generated_at": time.time(),
                    "face_image_path": str(face_path),
                    "audio_path": str(audio_path),
                    "output_video_path": str(output_path),
                    "generation_method": "ffmpeg_fallback",
                    "generation_success": True
                }
                
                metadata_file = self.outputs_dir / "video_generation_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(video_metadata, f, indent=2)
                
                return True
            else:
                logger.error(f"FFmpeg fallback failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Fallback video generation failed: {str(e)}")
            return False


def main():
    """Main entry point."""
    try:
        generator = SimpleVideoGeneration()
        success = generator.generate_video()
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Video generation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())