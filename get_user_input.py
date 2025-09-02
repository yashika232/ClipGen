#!/usr/bin/env python3
"""
Interactive User Input Script for Simple Video Pipeline
Collects user inputs interactively and validates files
"""

import os
import sys
import json
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import uuid
from datetime import datetime

class UserInputCollector:
    """Interactive user input collector with file validation."""
    
    def __init__(self):
        """Initialize user input collector."""
        self.project_root = Path(__file__).parent
        self.user_assets_dir = self.project_root / "user_assets"
        self.user_assets_dir.mkdir(exist_ok=True)
        
        # Create session ID
        self.session_id = f"session_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        self.session_dir = self.user_assets_dir / self.session_id
        self.session_dir.mkdir(exist_ok=True)
        
        print(f"VIDEO PIPELINE Video Synthesis Pipeline - Interactive Input")
        print(f"Session ID: {self.session_id}")
        print("=" * 60)
    
    def validate_image_file(self, file_path: str) -> Tuple[bool, str]:
        """Validate image file format and properties."""
        try:
            file_path = Path(file_path)
            
            # Check if file exists
            if not file_path.exists():
                return False, f"File not found: {file_path}"
            
            # Check file extension
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            if file_path.suffix.lower() not in valid_extensions:
                return False, f"Invalid image format. Supported: {', '.join(valid_extensions)}"
            
            # Check file size (max 50MB)
            file_size = file_path.stat().st_size
            max_size = 50 * 1024 * 1024  # 50MB
            if file_size > max_size:
                return False, f"File too large: {file_size / 1024 / 1024:.1f}MB (max 50MB)"
            
            # Try to validate with PIL if available
            try:
                from PIL import Image
                with Image.open(file_path) as img:
                    # Check image dimensions
                    width, height = img.size
                    if width < 100 or height < 100:
                        return False, f"Image too small: {width}x{height} (minimum 100x100)"
                    if width > 4096 or height > 4096:
                        return False, f"Image too large: {width}x{height} (maximum 4096x4096)"
                    
                    print(f"  [SUCCESS] Image validated: {width}x{height}, {file_size / 1024:.1f}KB")
            except ImportError:
                print(f"  [WARNING]  PIL not available, basic validation only")
            except Exception as e:
                return False, f"Invalid image file: {str(e)}"
            
            return True, "Valid image file"
            
        except Exception as e:
            return False, f"Error validating image: {str(e)}"
    
    def validate_audio_file(self, file_path: str) -> Tuple[bool, str]:
        """Validate audio file format and properties."""
        try:
            file_path = Path(file_path)
            
            # Check if file exists
            if not file_path.exists():
                return False, f"File not found: {file_path}"
            
            # Check file extension
            valid_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}
            if file_path.suffix.lower() not in valid_extensions:
                return False, f"Invalid audio format. Supported: {', '.join(valid_extensions)}"
            
            # Check file size (max 100MB)
            file_size = file_path.stat().st_size
            max_size = 100 * 1024 * 1024  # 100MB
            if file_size > max_size:
                return False, f"File too large: {file_size / 1024 / 1024:.1f}MB (max 100MB)"
            
            # Try to validate with librosa if available
            try:
                import librosa
                y, sr = librosa.load(file_path, duration=1.0)  # Load first second
                duration_cmd = f"ffprobe -v quiet -show_entries format=duration -of csv=p=0 '{file_path}'"
                import subprocess
                result = subprocess.run(duration_cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    duration = float(result.stdout.strip())
                    if duration < 1.0:
                        return False, f"Audio too short: {duration:.1f}s (minimum 1 second)"
                    if duration > 300:
                        return False, f"Audio too long: {duration:.1f}s (maximum 5 minutes)"
                    print(f"  [SUCCESS] Audio validated: {duration:.1f}s, {sr}Hz, {file_size / 1024:.1f}KB")
                else:
                    print(f"  [WARNING]  Could not get audio duration, basic validation only")
            except ImportError:
                print(f"  [WARNING]  librosa not available, basic validation only")
            except Exception as e:
                return False, f"Invalid audio file: {str(e)}"
            
            return True, "Valid audio file"
            
        except Exception as e:
            return False, f"Error validating audio: {str(e)}"
    
    def get_content_description(self) -> Dict[str, str]:
        """Get content description from user."""
        print("\nStep Content Description")
        print("-" * 30)
        
        # Get title
        while True:
            title = input("[EMOJI] Video Title: ").strip()
            if title:
                break
            print("   [ERROR] Title cannot be empty. Please enter a title.")
        
        # Get topic/subject
        while True:
            topic = input("Target: Main Topic/Subject: ").strip()
            if topic:
                break
            print("   [ERROR] Topic cannot be empty. Please enter a topic.")
        
        # Get detailed description (optional)
        print("Endpoints Detailed Description (optional):")
        print("   (Press Enter twice to finish, or type single Enter for empty)")
        description_lines = []
        empty_count = 0
        while empty_count < 2:
            line = input("   ")
            if line.strip() == "":
                empty_count += 1
            else:
                empty_count = 0
                description_lines.append(line)
        
        description = "\n".join(description_lines).strip()
        
        return {
            "title": title,
            "topic": topic,
            "description": description if description else f"A video about {topic}"
        }
    
    def get_file_input(self, file_type: str, validator_func, required: bool = True) -> Optional[str]:
        """Get file input from user with validation."""
        print(f"\nAssets: {file_type} Upload")
        print("-" * 30)
        
        while True:
            # Get file path
            file_path = input(f"[EMOJI] Path to {file_type.lower()} file: ").strip()
            
            # Handle empty input
            if not file_path:
                if not required:
                    return None
                print(f"   [ERROR] {file_type} file is required. Please provide a file path.")
                continue
            
            # Remove quotes if present
            file_path = file_path.strip('"\'')
            
            # Validate file
            print(f"   Search Validating {file_type.lower()}...")
            is_valid, message = validator_func(file_path)
            
            if is_valid:
                # Copy file to session directory
                source_path = Path(file_path)
                if file_type == "Face Image":
                    dest_filename = f"face_image{source_path.suffix}"
                else:  # Voice Sample
                    dest_filename = f"voice_sample{source_path.suffix}"
                
                dest_path = self.session_dir / dest_filename
                
                try:
                    shutil.copy2(source_path, dest_path)
                    print(f"   [SUCCESS] {file_type} copied to: {dest_path}")
                    return str(dest_path)
                except Exception as e:
                    print(f"   [ERROR] Error copying file: {e}")
                    continue
            else:
                print(f"   [ERROR] {message}")
                
                # Ask if user wants to try again
                retry = input("   [EMOJI] Try again? (y/n): ").strip().lower()
                if retry not in ['y', 'yes']:
                    if not required:
                        return None
                    print(f"   [WARNING]  {file_type} is required for video generation.")
    
    def get_preferences(self) -> Dict[str, str]:
        """Get user preferences for video generation."""
        print("\nConfiguration  Video Preferences (Optional)")
        print("-" * 40)
        
        # Audience
        print("Multi-user Target Audience:")
        print("   1. General audience")
        print("   2. Students")
        print("   3. Professionals")
        print("   4. Experts")
        
        audience_choice = input("   Choose (1-4, or press Enter for general): ").strip()
        audience_map = {
            "1": "general", "2": "students", "3": "professionals", "4": "experts"
        }
        audience = audience_map.get(audience_choice, "general")
        
        # Tone
        print("\nStyle: Video Tone:")
        print("   1. Professional")
        print("   2. Friendly")
        print("   3. Motivational")
        print("   4. Casual")
        
        tone_choice = input("   Choose (1-4, or press Enter for professional): ").strip()
        tone_map = {
            "1": "professional", "2": "friendly", "3": "motivational", "4": "casual"
        }
        tone = tone_map.get(tone_choice, "professional")
        
        # Emotion
        print("\n[EMOJI] Video Emotion:")
        print("   1. Confident")
        print("   2. Inspired")
        print("   3. Curious")
        print("   4. Excited")
        print("   5. Calm")
        
        emotion_choice = input("   Choose (1-5, or press Enter for confident): ").strip()
        emotion_map = {
            "1": "confident", "2": "inspired", "3": "curious", "4": "excited", "5": "calm"
        }
        emotion = emotion_map.get(emotion_choice, "confident")
        
        # Content type
        print("\nVIDEO PIPELINE Content Type:")
        print("   1. Short-Form Video Reel")
        print("   2. Educational Tutorial")
        print("   3. Presentation")
        print("   4. Explainer Video")
        
        content_choice = input("   Choose (1-4, or press Enter for reel): ").strip()
        content_map = {
            "1": "Short-Form Video Reel",
            "2": "Educational Tutorial", 
            "3": "Presentation",
            "4": "Explainer Video"
        }
        content_type = content_map.get(content_choice, "Short-Form Video Reel")
        
        # Background animation
        animation_choice = input("\nFrontend Include background animation? (y/n, default: n): ").strip().lower()
        include_animation = animation_choice in ['y', 'yes']
        
        return {
            "audience": audience,
            "tone": tone,
            "emotion": emotion,
            "content_type": content_type,
            "include_animation": include_animation
        }
    
    def confirm_inputs(self, content: Dict[str, str], face_image_path: str, 
                      voice_sample_path: Optional[str], preferences: Dict[str, str]) -> bool:
        """Show input summary and get confirmation."""
        print("\n" + "=" * 60)
        print("Endpoints INPUT SUMMARY")
        print("=" * 60)
        
        print(f"[EMOJI] Title: {content['title']}")
        print(f"Target: Topic: {content['topic']}")
        print(f"Endpoints Description: {content['description']}")
        print(f"[EMOJI] Face Image: {Path(face_image_path).name}")
        
        if voice_sample_path:
            print(f"Recording Voice Sample: {Path(voice_sample_path).name}")
        else:
            print(f"Recording Voice Sample: None (will use default)")
        
        print(f"Multi-user Audience: {preferences['audience']}")
        print(f"Style: Tone: {preferences['tone']}")
        print(f"[EMOJI] Emotion: {preferences['emotion']}")
        print(f"VIDEO PIPELINE Content Type: {preferences['content_type']}")
        print(f"Frontend Background Animation: {'Yes' if preferences['include_animation'] else 'No'}")
        
        print("\n" + "=" * 60)
        
        while True:
            confirm = input("[SUCCESS] Proceed with video generation? (y/n): ").strip().lower()
            if confirm in ['y', 'yes']:
                return True
            elif confirm in ['n', 'no']:
                return False
            else:
                print("   Please enter 'y' for yes or 'n' for no.")
    
    def save_inputs(self, content: Dict[str, str], face_image_path: str,
                   voice_sample_path: Optional[str], preferences: Dict[str, str]) -> str:
        """Save all inputs to JSON file."""
        inputs_data = {
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
            "content": content,
            "files": {
                "face_image": face_image_path,
                "voice_sample": voice_sample_path
            },
            "preferences": preferences,
            "user_inputs": {
                "title": content["title"],
                "topic": content["topic"],
                "audience": preferences["audience"],
                "tone": preferences["tone"],
                "emotion": preferences["emotion"],
                "content_type": preferences["content_type"],
                "include_animation": preferences["include_animation"],
                "description": content["description"]
            }
        }
        
        # Save to session directory
        inputs_file = self.session_dir / "user_inputs.json"
        with open(inputs_file, 'w') as f:
            json.dump(inputs_data, f, indent=2)
        
        print(f"Storage Inputs saved to: {inputs_file}")
        return str(inputs_file)
    
    def collect_all_inputs(self) -> Optional[Dict[str, Any]]:
        """Main function to collect all user inputs."""
        try:
            print("Welcome! Let's create your video step by step.\n")
            
            # Step 1: Get content description
            content = self.get_content_description()
            
            # Step 2: Get face image
            face_image_path = self.get_file_input("Face Image", self.validate_image_file, required=True)
            if not face_image_path:
                print("[ERROR] Face image is required. Exiting.")
                return None
            
            # Step 3: Get voice sample (optional)
            print("\nVoice sample is optional but recommended for voice cloning.")
            voice_sample_path = self.get_file_input("Voice Sample", self.validate_audio_file, required=False)
            
            # Step 4: Get preferences
            preferences = self.get_preferences()
            
            # Step 5: Confirm inputs
            if not self.confirm_inputs(content, face_image_path, voice_sample_path, preferences):
                print("[ERROR] Video generation cancelled.")
                return None
            
            # Step 6: Save inputs
            inputs_file = self.save_inputs(content, face_image_path, voice_sample_path, preferences)
            
            return {
                "session_id": self.session_id,
                "session_dir": str(self.session_dir),
                "inputs_file": inputs_file,
                "face_image_path": face_image_path,
                "voice_sample_path": voice_sample_path,
                "user_inputs": {
                    "title": content["title"],
                    "topic": content["topic"],
                    "audience": preferences["audience"],
                    "tone": preferences["tone"],
                    "emotion": preferences["emotion"],
                    "content_type": preferences["content_type"],
                    "include_animation": preferences["include_animation"],
                    "description": content["description"]
                }
            }
            
        except KeyboardInterrupt:
            print("\n\n[ERROR] Input collection cancelled by user.")
            return None
        except Exception as e:
            print(f"\n[ERROR] Error during input collection: {e}")
            return None


def main():
    """Main function."""
    print("VIDEO PIPELINE Simple Video Pipeline - Interactive Input Collection")
    print("=" * 60)
    
    collector = UserInputCollector()
    result = collector.collect_all_inputs()
    
    if result:
        print("\nSUCCESS Input collection completed successfully!")
        print(f"Session ID: {result['session_id']}")
        print(f"Session Directory: {result['session_dir']}")
        
        # Ask if user wants to start video generation
        print("\n" + "=" * 60)
        start_generation = input("STARTING Start video generation now? (y/n): ").strip().lower()
        
        if start_generation in ['y', 'yes']:
            print("\nVIDEO PIPELINE Starting video generation...")
            
            # Import and run the simple pipeline
            try:
                from simple_pipeline import SimpleVideoPipeline
                
                pipeline = SimpleVideoPipeline()
                success = pipeline.run_pipeline(
                    user_inputs=result["user_inputs"],
                    face_image_path=result["face_image_path"],
                    voice_audio_path=result["voice_sample_path"]
                )
                
                if success:
                    print("\nSUCCESS Video generation completed successfully!")
                else:
                    print("\n[ERROR] Video generation failed. Check logs for details.")
                    
            except Exception as e:
                print(f"\n[ERROR] Error starting video generation: {e}")
                print("You can run the pipeline manually later using:")
                print(f"python simple_pipeline.py --title \"{result['user_inputs']['title']}\" --topic \"{result['user_inputs']['topic']}\" --face-image \"{result['face_image_path']}\" --voice-audio \"{result['voice_sample_path'] or ''}\"")
        else:
            print("\nStep You can start video generation later using:")
            print(f"python simple_pipeline.py --title \"{result['user_inputs']['title']}\" --topic \"{result['user_inputs']['topic']}\" --face-image \"{result['face_image_path']}\"", end="")
            if result['voice_sample_path']:
                print(f" --voice-audio \"{result['voice_sample_path']}\"", end="")
            if result['user_inputs']['include_animation']:
                print(" --include-animation", end="")
            print()
    else:
        print("\n[ERROR] Input collection failed or was cancelled.")


if __name__ == "__main__":
    main()