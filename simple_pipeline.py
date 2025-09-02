#!/usr/bin/env python3
"""
Simple Video Synthesis Pipeline
Main controller that orchestrates all stages sequentially
Based on legacy working approach - simplified and straightforward
"""

import os
import sys
import json
import time
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)

class SimpleVideoPipeline:
    """Simple video synthesis pipeline controller."""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the pipeline with configuration."""
        self.logger = logging.getLogger(__name__)
        self.project_root = Path(__file__).parent
        self.config = self._load_config(config_path)
        self.outputs_dir = self.project_root / "outputs"
        self.outputs_dir.mkdir(exist_ok=True)
        
        # Current session data
        self.session_data = {}
        
        self.logger.info("STARTING Simple Video Synthesis Pipeline initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        config_file = self.project_root / config_path
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
        else:
            # Default configuration
            config = {
                "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
                "conda_environments": {
                    "script_generation": "base",
                    "voice_synthesis": "xtts",
                    "face_processing": "base", 
                    "video_generation": "sadtalker",
                    "enhancement": "enhancement",
                    "background_animation": "manim",
                    "final_assembly": "video-audio-processing"
                },
                "stage_timeouts": {
                    "script_generation": 300,    # 5 minutes
                    "voice_synthesis": 1800,     # 30 minutes
                    "face_processing": 600,      # 10 minutes
                    "video_generation": 3600,    # 1 hour
                    "enhancement": 1800,         # 30 minutes
                    "background_animation": 1200, # 20 minutes
                    "final_assembly": 600        # 10 minutes
                }
            }
            
            # Save default config
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
        
        return config
    
    def _run_stage(self, stage_name: str, script_path: str, args: list = None) -> bool:
        """Run a pipeline stage in isolated conda environment."""
        self.logger.info(f"Target: Starting stage: {stage_name}")
        
        # Get conda environment for this stage
        env_name = self.config["conda_environments"].get(stage_name, "base")
        timeout = self.config["stage_timeouts"].get(stage_name, 1800)
        
        # Build command
        cmd = ["conda", "run", "-n", env_name, "python", script_path]
        if args:
            cmd.extend(args)
        
        self.logger.info(f"Running: {' '.join(cmd)}")
        
        try:
            # Run stage with timeout
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                self.logger.info(f"[SUCCESS] Stage {stage_name} completed successfully")
                if result.stdout:
                    self.logger.info(f"Output: {result.stdout}")
                return True
            else:
                self.logger.error(f"[ERROR] Stage {stage_name} failed with exit code {result.returncode}")
                if result.stderr:
                    self.logger.error(f"Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"[ERROR] Stage {stage_name} timed out after {timeout} seconds")
            return False
        except Exception as e:
            self.logger.error(f"[ERROR] Stage {stage_name} failed with exception: {str(e)}")
            return False
    
    def _save_session_data(self, data: Dict[str, Any]):
        """Save session data to JSON file."""
        session_file = self.outputs_dir / "session_data.json"
        with open(session_file, 'w') as f:
            json.dump(data, f, indent=2)
        self.session_data = data
    
    def _load_session_data(self) -> Dict[str, Any]:
        """Load session data from JSON file."""
        session_file = self.outputs_dir / "session_data.json"
        if session_file.exists():
            with open(session_file, 'r') as f:
                self.session_data = json.load(f)
        return self.session_data
    
    def run_pipeline(self, user_inputs: Dict[str, Any], face_image_path: str, 
                    voice_audio_path: str = None) -> bool:
        """Run the complete video synthesis pipeline."""
        
        start_time = time.time()
        
        self.logger.info("="*60)
        self.logger.info("VIDEO PIPELINE STARTING SIMPLE VIDEO SYNTHESIS PIPELINE")
        self.logger.info("="*60)
        
        # Initialize session data
        session_data = {
            "user_inputs": user_inputs,
            "face_image_path": face_image_path,
            "voice_audio_path": voice_audio_path,
            "stage_results": {},
            "pipeline_start_time": start_time
        }
        self._save_session_data(session_data)
        
        # Stage 1: Script Generation
        if not self._run_stage_1_script_generation():
            return False
        
        # Stage 2: Voice Synthesis
        if not self._run_stage_2_voice_synthesis():
            return False
        
        # Stage 3: Face Processing
        if not self._run_stage_3_face_processing():
            return False
        
        # Stage 4: Video Generation
        if not self._run_stage_4_video_generation():
            return False
        
        # Stage 5: Enhancement
        if not self._run_stage_5_enhancement():
            return False
        
        # Stage 6: Background Animation (if needed)
        if user_inputs.get("include_animation", False):
            if not self._run_stage_6_background_animation():
                return False
        
        # Stage 7: Final Assembly
        if not self._run_stage_7_final_assembly():
            return False
        
        # Pipeline completed successfully
        total_time = time.time() - start_time
        self.logger.info("="*60)
        self.logger.info("SUCCESS PIPELINE COMPLETED SUCCESSFULLY!")
        self.logger.info(f"Total processing time: {total_time:.2f} seconds")
        self.logger.info("="*60)
        
        return True
    
    def _run_stage_1_script_generation(self) -> bool:
        """Stage 1: Generate script using Gemini API."""
        return self._run_stage(
            "script_generation",
            "stages/stage1_script_generation.py"
        )
    
    def _run_stage_2_voice_synthesis(self) -> bool:
        """Stage 2: Synthesize voice using XTTS."""
        return self._run_stage(
            "voice_synthesis", 
            "stages/stage2_voice_synthesis.py"
        )
    
    def _run_stage_3_face_processing(self) -> bool:
        """Stage 3: Process face using InsightFace."""
        return self._run_stage(
            "face_processing",
            "stages/stage3_face_processing.py"
        )
    
    def _run_stage_4_video_generation(self) -> bool:
        """Stage 4: Generate video using SadTalker."""
        return self._run_stage(
            "video_generation",
            "stages/stage4_video_generation.py"
        )
    
    def _run_stage_5_enhancement(self) -> bool:
        """Stage 5: Enhance video using Real-ESRGAN + CodeFormer."""
        return self._run_stage(
            "enhancement",
            "stages/stage5_enhancement.py"
        )
    
    def _run_stage_6_background_animation(self) -> bool:
        """Stage 6: Create background animation using Manim."""
        return self._run_stage(
            "background_animation",
            "stages/stage6_background_animation.py"
        )
    
    def _run_stage_7_final_assembly(self) -> bool:
        """Stage 7: Final assembly using FFmpeg."""
        return self._run_stage(
            "final_assembly",
            "stages/stage7_final_assembly.py"
        )


def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Simple Video Synthesis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive input collection (recommended)
  python get_user_input.py

  # Direct command line usage
  python simple_pipeline.py --title "ML Introduction" --topic "Machine Learning Basics" --face-image face.jpg --voice-audio voice.wav

  # With background animation
  python simple_pipeline.py --title "ML Introduction" --topic "Machine Learning Basics" --face-image face.jpg --voice-audio voice.wav --include-animation

  # Load from session file
  python simple_pipeline.py --session-file user_assets/session_12345/user_inputs.json
        """
    )
    
    # Input methods (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--session-file", help="Path to session input file (from get_user_input.py)")
    input_group.add_argument("--title", help="Video title (for direct input)")
    
    # Direct input arguments (required when not using session file)
    parser.add_argument("--topic", help="Video topic")
    parser.add_argument("--face-image", help="Path to face image")
    
    # Optional arguments
    parser.add_argument("--voice-audio", help="Path to voice audio sample")
    parser.add_argument("--audience", default="general", help="Target audience")
    parser.add_argument("--tone", default="professional", help="Video tone")
    parser.add_argument("--emotion", default="confident", help="Video emotion")
    parser.add_argument("--content-type", default="Short-Form Video Reel", help="Content type")
    parser.add_argument("--include-animation", action="store_true", help="Include background animation")
    parser.add_argument("--config", default="config.json", help="Configuration file")
    
    args = parser.parse_args()
    
    # Load inputs from session file or command line
    if args.session_file:
        # Load from session file
        session_file = Path(args.session_file)
        if not session_file.exists():
            print(f"Error: Session file not found: {args.session_file}")
            return 1
        
        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            # Extract inputs from session data
            user_inputs = session_data["user_inputs"]
            face_image_path = session_data["files"]["face_image"]
            voice_audio_path = session_data["files"]["voice_sample"]
            
            print(f"[SUCCESS] Loaded session: {session_data.get('session_id', 'unknown')}")
            print(f"   Title: {user_inputs['title']}")
            print(f"   Topic: {user_inputs['topic']}")
            
        except Exception as e:
            print(f"Error: Could not load session file: {e}")
            return 1
    else:
        # Validate direct input arguments
        if not args.title:
            print("Error: --title is required when not using --session-file")
            return 1
        if not args.topic:
            print("Error: --topic is required when not using --session-file")
            return 1
        if not args.face_image:
            print("Error: --face-image is required when not using --session-file")
            return 1
        
        # Validate file paths
        if not Path(args.face_image).exists():
            print(f"Error: Face image not found: {args.face_image}")
            return 1
        
        if args.voice_audio and not Path(args.voice_audio).exists():
            print(f"Error: Voice audio not found: {args.voice_audio}")
            return 1
        
        # Prepare user inputs from command line
        user_inputs = {
            "title": args.title,
            "topic": args.topic,
            "audience": args.audience,
            "tone": args.tone,
            "emotion": args.emotion,
            "content_type": args.content_type,
            "include_animation": args.include_animation
        }
        
        face_image_path = args.face_image
        voice_audio_path = args.voice_audio
    
    try:
        # Initialize and run pipeline
        pipeline = SimpleVideoPipeline(args.config)
        success = pipeline.run_pipeline(
            user_inputs=user_inputs,
            face_image_path=face_image_path,
            voice_audio_path=voice_audio_path
        )
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())