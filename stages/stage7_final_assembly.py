#!/usr/bin/env python3
"""
Stage 7: Final Assembly - Simple Video Composition
Combines all components into final video using FFmpeg
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

class SimpleFinalAssembly:
    """Simple final assembly using FFmpeg."""
    
    def __init__(self):
        """Initialize final assembly."""
        self.project_root = PROJECT_ROOT
        self.outputs_dir = self.project_root / "outputs"
        self.outputs_dir.mkdir(exist_ok=True)
        
        # Load session data
        self.session_data = self._load_session_data()
        
        logger.info("VIDEO PIPELINE Simple Final Assembly initialized")
    
    def _load_session_data(self) -> Dict[str, Any]:
        """Load session data from JSON file."""
        session_file = self.outputs_dir / "session_data.json"
        if session_file.exists():
            with open(session_file, 'r') as f:
                return json.load(f)
        return {}
    
    def assemble_final_video(self) -> bool:
        """Assemble final video from all components."""
        try:
            logger.info("Target: Starting final video assembly...")
            
            # Check what components are available
            components = self._check_available_components()
            
            if not components:
                raise ValueError("No video components available for assembly")
            
            # Assemble based on available components
            if components["has_background"] and components["has_talking_head"]:
                success = self._assemble_with_background_and_talking_head()
            elif components["has_talking_head"]:
                success = self._assemble_talking_head_only()
            else:
                success = self._assemble_fallback()
            
            return success
            
        except Exception as e:
            logger.error(f"[ERROR] Final assembly failed: {str(e)}")
            return False
    
    def _check_available_components(self) -> Dict[str, Any]:
        """Check which components are available for assembly."""
        components = {
            "has_talking_head": False,
            "has_background": False,
            "has_enhanced": False,
            "talking_head_path": None,
            "background_path": None,
            "enhanced_path": None
        }
        
        # Check for enhanced video (preferred)
        enhanced_video = self.outputs_dir / "enhanced_video.mp4"
        if enhanced_video.exists():
            components["has_enhanced"] = True
            components["enhanced_path"] = enhanced_video
            components["has_talking_head"] = True
            components["talking_head_path"] = enhanced_video
        
        # Check for talking head video
        talking_head = self.outputs_dir / "talking_head.mp4"
        if talking_head.exists() and not components["has_talking_head"]:
            components["has_talking_head"] = True
            components["talking_head_path"] = talking_head
        
        # Check for background animation
        background = self.outputs_dir / "background_animation.mp4"
        if background.exists():
            components["has_background"] = True
            components["background_path"] = background
        
        logger.info(f"Available components:")
        logger.info(f"  Talking Head: {components['has_talking_head']}")
        logger.info(f"  Background: {components['has_background']}")
        logger.info(f"  Enhanced: {components['has_enhanced']}")
        
        return components
    
    def _assemble_with_background_and_talking_head(self) -> bool:
        """Assemble video with both background animation and talking head."""
        try:
            components = self._check_available_components()
            
            talking_head_path = components["talking_head_path"]
            background_path = components["background_path"]
            
            if not talking_head_path or not background_path:
                return False
            
            # Final output path
            output_path = self.outputs_dir / "final_video.mp4"
            
            # Create composite video with face overlay
            # Based on legacy approach: 300x300 face in bottom-right corner
            cmd = [
                "ffmpeg", "-y",
                "-i", str(background_path),
                "-i", str(talking_head_path),
                "-filter_complex", (
                    "[0:v]scale=1920:1080[bg];"
                    "[1:v]scale=300:300[face];"
                    "[bg][face]overlay=1580:740[v]"
                ),
                "-map", "[v]",
                "-map", "1:a",  # Use talking head audio
                "-c:v", "libx264",
                "-c:a", "aac",
                "-b:a", "192k",
                "-crf", "18",
                "-preset", "medium",
                str(output_path)
            ]
            
            logger.info("Creating composite video with background and talking head...")
            logger.info(f"Command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            
            if result.returncode == 0 and output_path.exists():
                logger.info(f"[SUCCESS] Final video assembled successfully")
                logger.info(f"   Output: {output_path}")
                
                # Save assembly metadata
                assembly_metadata = {
                    "assembled_at": time.time(),
                    "components_used": {
                        "background": str(background_path),
                        "talking_head": str(talking_head_path)
                    },
                    "output_video_path": str(output_path),
                    "assembly_method": "background_with_talking_head",
                    "assembly_success": True
                }
                
                metadata_file = self.outputs_dir / "assembly_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(assembly_metadata, f, indent=2)
                
                return True
            else:
                logger.error(f"Video assembly failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Background + talking head assembly failed: {str(e)}")
            return False
    
    def _assemble_talking_head_only(self) -> bool:
        """Assemble video with talking head only."""
        try:
            components = self._check_available_components()
            talking_head_path = components["talking_head_path"]
            
            if not talking_head_path:
                return False
            
            # Final output path
            output_path = self.outputs_dir / "final_video.mp4"
            
            # Simple copy/re-encode of talking head video
            cmd = [
                "ffmpeg", "-y",
                "-i", str(talking_head_path),
                "-c:v", "libx264",
                "-c:a", "aac",
                "-b:a", "192k",
                "-crf", "18",
                "-preset", "medium",
                str(output_path)
            ]
            
            logger.info("Creating final video from talking head only...")
            logger.info(f"Command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0 and output_path.exists():
                logger.info(f"[SUCCESS] Final video assembled successfully")
                logger.info(f"   Output: {output_path}")
                
                # Save assembly metadata
                assembly_metadata = {
                    "assembled_at": time.time(),
                    "components_used": {
                        "talking_head": str(talking_head_path)
                    },
                    "output_video_path": str(output_path),
                    "assembly_method": "talking_head_only",
                    "assembly_success": True
                }
                
                metadata_file = self.outputs_dir / "assembly_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(assembly_metadata, f, indent=2)
                
                return True
            else:
                logger.error(f"Video assembly failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Talking head only assembly failed: {str(e)}")
            return False
    
    def _assemble_fallback(self) -> bool:
        """Fallback assembly method."""
        try:
            # Create a simple placeholder video
            output_path = self.outputs_dir / "final_video.mp4"
            
            # Get user inputs for basic content
            user_inputs = self.session_data.get("user_inputs", {})
            title = user_inputs.get("title", "Video Title")
            
            # Create simple text video
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"color=c=0x0F1419:size=1920x1080:duration=10",
                "-vf", f"drawtext=text='{title}':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=(h-text_h)/2",
                "-c:v", "libx264",
                "-r", "25",
                str(output_path)
            ]
            
            logger.info("Creating fallback video...")
            logger.info(f"Command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and output_path.exists():
                logger.info(f"[SUCCESS] Fallback video created successfully")
                logger.info(f"   Output: {output_path}")
                
                # Save assembly metadata
                assembly_metadata = {
                    "assembled_at": time.time(),
                    "components_used": {
                        "title": title
                    },
                    "output_video_path": str(output_path),
                    "assembly_method": "fallback_text_video",
                    "assembly_success": True
                }
                
                metadata_file = self.outputs_dir / "assembly_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(assembly_metadata, f, indent=2)
                
                return True
            else:
                logger.error(f"Fallback video creation failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Fallback assembly failed: {str(e)}")
            return False


def main():
    """Main entry point."""
    try:
        assembler = SimpleFinalAssembly()
        success = assembler.assemble_final_video()
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Final assembly failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())