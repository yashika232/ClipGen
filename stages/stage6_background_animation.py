#!/usr/bin/env python3
"""
Stage 6: Background Animation - Simple Manim Integration
Creates background animation using Manim from generated Python code
This is the second place where we use metadata (reads Python code from generated_script.json)
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

class SimpleBackgroundAnimation:
    """Simple background animation using Manim."""
    
    def __init__(self):
        """Initialize background animation."""
        self.project_root = PROJECT_ROOT
        self.outputs_dir = self.project_root / "outputs"
        self.outputs_dir.mkdir(exist_ok=True)
        
        # Load session data
        self.session_data = self._load_session_data()
        
        logger.info("Frontend Simple Background Animation initialized")
    
    def _load_session_data(self) -> Dict[str, Any]:
        """Load session data from JSON file."""
        session_file = self.outputs_dir / "session_data.json"
        if session_file.exists():
            with open(session_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_generated_script(self) -> Dict[str, Any]:
        """Load generated script from JSON file."""
        script_file = self.outputs_dir / "generated_script.json"
        if script_file.exists():
            with open(script_file, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError("Generated script not found. Run stage 1 first.")
    
    def create_animation(self) -> bool:
        """Create background animation using Manim."""
        try:
            logger.info("Target: Starting background animation creation...")
            
            # Load generated script data
            script_data = self._load_generated_script()
            manim_code = script_data.get("manim_code", "")
            
            if not manim_code:
                logger.warning("No Manim code found, creating default animation")
                manim_code = self._create_default_animation()
            
            # Try to create animation with Manim
            success = self._create_with_manim(manim_code)
            
            if not success:
                # Fallback to simple animation
                logger.warning("Using fallback animation creation...")
                success = self._create_with_fallback()
            
            return success
            
        except Exception as e:
            logger.error(f"[ERROR] Background animation creation failed: {str(e)}")
            return False
    
    def _create_with_manim(self, manim_code: str) -> bool:
        """Create animation using Manim."""
        try:
            # Save Manim code to file
            animation_script = self.outputs_dir / "animation_script.py"
            with open(animation_script, 'w') as f:
                f.write(manim_code)
            
            # Create Manim output directory
            manim_output_dir = self.outputs_dir / "manim_output"
            manim_output_dir.mkdir(exist_ok=True)
            
            # Run Manim
            cmd = [
                "manim", "-ql",  # Low quality for faster processing
                "--media_dir", str(manim_output_dir),
                str(animation_script),
                "VideoAnimation"
            ]
            
            logger.info(f"Running Manim: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1200  # 20 minutes timeout
            )
            
            if result.returncode == 0:
                # Find generated video
                generated_videos = list(manim_output_dir.glob("**/*.mp4"))
                if generated_videos:
                    generated_video = generated_videos[0]
                    
                    # Copy to standard output location
                    output_path = self.outputs_dir / "background_animation.mp4"
                    import shutil
                    shutil.copy2(str(generated_video), str(output_path))
                    
                    logger.info(f"[SUCCESS] Background animation created successfully with Manim")
                    logger.info(f"   Output: {output_path}")
                    
                    # Save animation metadata
                    animation_metadata = {
                        "created_at": time.time(),
                        "manim_code_length": len(manim_code),
                        "output_animation_path": str(output_path),
                        "animation_method": "manim",
                        "animation_success": True
                    }
                    
                    metadata_file = self.outputs_dir / "animation_metadata.json"
                    with open(metadata_file, 'w') as f:
                        json.dump(animation_metadata, f, indent=2)
                    
                    return True
                else:
                    logger.error("Manim completed but no video found")
                    return False
            else:
                logger.error(f"Manim failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Manim animation creation failed: {str(e)}")
            return False
    
    def _create_with_fallback(self) -> bool:
        """Create simple fallback animation using FFmpeg."""
        try:
            output_path = self.outputs_dir / "background_animation.mp4"
            
            # Create a simple colored background video
            # Duration based on enhanced video if available
            enhanced_video = self.outputs_dir / "enhanced_video.mp4"
            talking_head = self.outputs_dir / "talking_head.mp4"
            
            # Get duration from existing video
            duration = 10  # Default duration
            
            if enhanced_video.exists():
                duration_cmd = [
                    "ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "csv=p=0", str(enhanced_video)
                ]
                result = subprocess.run(duration_cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    try:
                        duration = float(result.stdout.strip())
                    except:
                        pass
            elif talking_head.exists():
                duration_cmd = [
                    "ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "csv=p=0", str(talking_head)
                ]
                result = subprocess.run(duration_cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    try:
                        duration = float(result.stdout.strip())
                    except:
                        pass
            
            # Create simple colored background
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"color=c=0x0F1419:size=1920x1080:duration={duration}",
                "-c:v", "libx264",
                "-r", "25",
                str(output_path)
            ]
            
            logger.info(f"Creating fallback animation: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and output_path.exists():
                logger.info(f"[SUCCESS] Background animation created with fallback method")
                logger.info(f"   Output: {output_path}")
                
                # Save animation metadata
                animation_metadata = {
                    "created_at": time.time(),
                    "duration": duration,
                    "output_animation_path": str(output_path),
                    "animation_method": "fallback_colored_background",
                    "animation_success": True
                }
                
                metadata_file = self.outputs_dir / "animation_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(animation_metadata, f, indent=2)
                
                return True
            else:
                logger.error(f"Fallback animation creation failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Fallback animation creation failed: {str(e)}")
            return False
    
    def _create_default_animation(self) -> str:
        """Create default Manim animation code."""
        user_inputs = self.session_data.get("user_inputs", {})
        title = user_inputs.get("title", "Video Title")
        topic = user_inputs.get("topic", "Topic")
        
        return f'''from manim import *
import time

class VideoAnimation(Scene):
    def construct(self):
        # Setup
        self.camera.background_color = "#0F1419"
        
        # Title
        title = Text("{title}", font_size=48, color="#3498DB")
        title.to_edge(UP)
        
        # Topic
        topic_text = Text("{topic}", font_size=36, color="#27AE60")
        topic_text.next_to(title, DOWN, buff=0.5)
        
        # Animate title and topic
        self.play(FadeIn(title))
        self.wait(1)
        self.play(FadeIn(topic_text))
        self.wait(2)
        
        # Main content area
        content_box = Rectangle(width=10, height=4, color="#9B59B6")
        content_box.set_fill(color="#9B59B6", opacity=0.1)
        content_box.move_to(ORIGIN)
        
        self.play(Create(content_box))
        self.wait(1)
        
        # Content text
        content_text = Text(
            "Educational Content",
            font_size=24,
            color="#F39C12"
        )
        content_text.move_to(content_box.get_center())
        
        self.play(Write(content_text))
        self.wait(3)
        
        # Fade out
        self.play(FadeOut(VGroup(title, topic_text, content_box, content_text)))
        self.wait(1)
'''


def main():
    """Main entry point."""
    try:
        animator = SimpleBackgroundAnimation()
        success = animator.create_animation()
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Background animation creation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())