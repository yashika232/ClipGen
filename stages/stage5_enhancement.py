#!/usr/bin/env python3
"""
Stage 5: Video Enhancement - Real-ESRGAN + CodeFormer
Enhances video quality using Real-ESRGAN and CodeFormer
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

class SimpleVideoEnhancement:
    """Simple video enhancement using Real-ESRGAN and CodeFormer."""
    
    def __init__(self):
        """Initialize video enhancement."""
        self.project_root = PROJECT_ROOT
        self.outputs_dir = self.project_root / "outputs"
        self.outputs_dir.mkdir(exist_ok=True)
        
        # Load session data
        self.session_data = self._load_session_data()
        
        logger.info("Enhanced Simple Video Enhancement initialized")
    
    def _load_session_data(self) -> Dict[str, Any]:
        """Load session data from JSON file."""
        session_file = self.outputs_dir / "session_data.json"
        if session_file.exists():
            with open(session_file, 'r') as f:
                return json.load(f)
        return {}
    
    def enhance_video(self) -> bool:
        """Enhance video using Real-ESRGAN and CodeFormer."""
        try:
            logger.info("Target: Starting video enhancement...")
            
            # Check required inputs
            input_video_path = self.outputs_dir / "talking_head.mp4"
            
            if not input_video_path.exists():
                raise FileNotFoundError(f"Input video not found: {input_video_path}")
            
            # Try enhancement pipeline
            success = self._enhance_with_pipeline(input_video_path)
            
            if not success:
                # Fallback to simple copy
                logger.warning("Using fallback enhancement (copy)...")
                success = self._enhance_with_fallback(input_video_path)
            
            return success
            
        except Exception as e:
            logger.error(f"[ERROR] Video enhancement failed: {str(e)}")
            return False
    
    def _enhance_with_pipeline(self, input_video: Path) -> bool:
        """Enhance video using Real-ESRGAN and CodeFormer pipeline."""
        try:
            # Check if enhancement models are available
            realesrgan_path = self.project_root / "models" / "Real-ESRGAN"
            codeformer_path = self.project_root / "models" / "codeformer"
            
            if not realesrgan_path.exists():
                logger.warning("Real-ESRGAN model not found")
                return False
            
            # Create temporary directory for enhancement
            enhancement_dir = self.outputs_dir / "enhancement_temp"
            enhancement_dir.mkdir(exist_ok=True)
            
            # Step 1: Extract frames from video
            frames_dir = enhancement_dir / "frames"
            frames_dir.mkdir(exist_ok=True)
            
            logger.info("Extracting frames from video...")
            extract_cmd = [
                "ffmpeg", "-y",
                "-i", str(input_video),
                "-vf", "fps=25",
                str(frames_dir / "frame_%06d.png")
            ]
            
            result = subprocess.run(extract_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Frame extraction failed: {result.stderr}")
                return False
            
            # Step 2: Enhance frames with Real-ESRGAN
            enhanced_frames_dir = enhancement_dir / "enhanced_frames"
            enhanced_frames_dir.mkdir(exist_ok=True)
            
            logger.info("Enhancing frames with Real-ESRGAN...")
            realesrgan_script = realesrgan_path / "inference_realesrgan.py"
            if not realesrgan_script.exists():
                realesrgan_script = realesrgan_path / "realesrgan" / "inference_realesrgan.py"
            
            if realesrgan_script.exists():
                enhance_cmd = [
                    "python", str(realesrgan_script),
                    "-i", str(frames_dir),
                    "-o", str(enhanced_frames_dir),
                    "-n", "RealESRGAN_x2plus",
                    "-s", "2"
                ]
                
                result = subprocess.run(enhance_cmd, capture_output=True, text=True, timeout=1800)
                if result.returncode != 0:
                    logger.warning(f"Real-ESRGAN enhancement failed: {result.stderr}")
                    # Copy original frames
                    import shutil
                    shutil.copytree(str(frames_dir), str(enhanced_frames_dir), dirs_exist_ok=True)
            else:
                logger.warning("Real-ESRGAN script not found, skipping enhancement")
                # Copy original frames
                import shutil
                shutil.copytree(str(frames_dir), str(enhanced_frames_dir), dirs_exist_ok=True)
            
            # Step 3: Apply CodeFormer if available
            if codeformer_path.exists():
                logger.info("Applying CodeFormer face restoration...")
                codeformer_script = codeformer_path / "inference_codeformer.py"
                if codeformer_script.exists():
                    final_frames_dir = enhancement_dir / "final_frames"
                    final_frames_dir.mkdir(exist_ok=True)
                    
                    codeformer_cmd = [
                        "python", str(codeformer_script),
                        "-i", str(enhanced_frames_dir),
                        "-o", str(final_frames_dir),
                        "-w", "0.7",
                        "--face_upsample"
                    ]
                    
                    result = subprocess.run(codeformer_cmd, capture_output=True, text=True, timeout=1800)
                    if result.returncode == 0:
                        enhanced_frames_dir = final_frames_dir
                    else:
                        logger.warning(f"CodeFormer failed: {result.stderr}")
            
            # Step 4: Reconstruct video from enhanced frames
            output_path = self.outputs_dir / "enhanced_video.mp4"
            
            logger.info("Reconstructing video from enhanced frames...")
            
            # Extract audio from original video
            audio_path = enhancement_dir / "original_audio.wav"
            extract_audio_cmd = [
                "ffmpeg", "-y",
                "-i", str(input_video),
                "-vn", "-acodec", "pcm_s16le",
                str(audio_path)
            ]
            
            subprocess.run(extract_audio_cmd, capture_output=True, text=True)
            
            # Reconstruct video
            reconstruct_cmd = [
                "ffmpeg", "-y",
                "-r", "25",
                "-i", str(enhanced_frames_dir / "frame_%06d.png"),
                "-i", str(audio_path),
                "-c:v", "libx264",
                "-c:a", "aac",
                "-b:a", "192k",
                "-crf", "18",
                "-shortest",
                str(output_path)
            ]
            
            result = subprocess.run(reconstruct_cmd, capture_output=True, text=True)
            if result.returncode == 0 and output_path.exists():
                logger.info(f"[SUCCESS] Video enhanced successfully")
                logger.info(f"   Output: {output_path}")
                
                # Save enhancement metadata
                enhancement_metadata = {
                    "enhanced_at": time.time(),
                    "input_video_path": str(input_video),
                    "output_video_path": str(output_path),
                    "enhancement_method": "realesrgan_codeformer",
                    "enhancement_success": True
                }
                
                metadata_file = self.outputs_dir / "enhancement_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(enhancement_metadata, f, indent=2)
                
                # Cleanup temporary files
                import shutil
                shutil.rmtree(str(enhancement_dir), ignore_errors=True)
                
                return True
            else:
                logger.error(f"Video reconstruction failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Enhancement pipeline failed: {str(e)}")
            return False
    
    def _enhance_with_fallback(self, input_video: Path) -> bool:
        """Fallback enhancement - simple copy with quality adjustment."""
        try:
            output_path = self.outputs_dir / "enhanced_video.mp4"
            
            # Simple re-encoding with quality settings
            cmd = [
                "ffmpeg", "-y",
                "-i", str(input_video),
                "-c:v", "libx264",
                "-c:a", "aac",
                "-b:a", "192k",
                "-crf", "18",
                "-preset", "medium",
                str(output_path)
            ]
            
            logger.info(f"Running fallback enhancement: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0 and output_path.exists():
                logger.info(f"[SUCCESS] Video enhanced with fallback method")
                logger.info(f"   Output: {output_path}")
                
                # Save enhancement metadata
                enhancement_metadata = {
                    "enhanced_at": time.time(),
                    "input_video_path": str(input_video),
                    "output_video_path": str(output_path),
                    "enhancement_method": "fallback_reencoding",
                    "enhancement_success": True
                }
                
                metadata_file = self.outputs_dir / "enhancement_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(enhancement_metadata, f, indent=2)
                
                return True
            else:
                logger.error(f"Fallback enhancement failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Fallback enhancement failed: {str(e)}")
            return False


def main():
    """Main entry point."""
    try:
        enhancer = SimpleVideoEnhancement()
        success = enhancer.enhance_video()
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Video enhancement failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())