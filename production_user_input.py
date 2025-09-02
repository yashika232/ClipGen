#!/usr/bin/env python3
"""
Production User Input Collector for Video Synthesis Pipeline
Integrates with the production API endpoints and WebSocket system
Works with the full production architecture including session management
"""

import os
import sys
import json
import time
import requests
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import uuid

# Optional WebSocket support
try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

class ProductionUserInputCollector:
    """User input collector that integrates with production API."""
    
    def __init__(self, api_base_url: str = "http://localhost:5002", 
                 websocket_url: str = "ws://localhost:5002/socket.io"):
        """Initialize production user input collector.
        
        Args:
            api_base_url: Base URL for the production API
            websocket_url: WebSocket URL for real-time updates
        """
        self.api_base_url = api_base_url.rstrip('/')
        self.websocket_url = websocket_url
        self.session_id = None
        self.session_info = None
        self.ws = None
        
        print(f"VIDEO PIPELINE Production Video Synthesis Pipeline - Interactive Input")
        print(f"API Server: {self.api_base_url}")
        print("=" * 60)
    
    def check_api_health(self) -> bool:
        """Check if the production API is available."""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print(f"[SUCCESS] API Server Status: {health_data.get('status', 'OK')}")
                return True
            else:
                print(f"[ERROR] API Server unhealthy: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Cannot connect to API server: {e}")
            print(f"INFO: Make sure to start the server with: python start_pipeline.py")
            return False
    
    def create_session(self, user_id: str = "interactive_user") -> bool:
        """Create a new session via the production API."""
        try:
            response = requests.post(
                f"{self.api_base_url}/session/create",
                json={"user_id": user_id},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    self.session_id = data['session_id']
                    self.session_info = data['session_info']
                    print(f"[SUCCESS] Session created: {self.session_id}")
                    print(f"   User ID: {self.session_info['user_id']}")
                    print(f"   State: {self.session_info['state']}")
                    return True
            
            print(f"[ERROR] Failed to create session: {response.text}")
            return False
            
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Error creating session: {e}")
            return False
    
    def setup_websocket(self):
        """Setup WebSocket connection for real-time updates."""
        try:
            if not WEBSOCKET_AVAILABLE:
                print("Ports WebSocket not available (python-socketio not installed)")
                print("   Continuing with API polling for progress updates...")
                return False
            
            # Simple WebSocket connection (production has socket.io but this is basic)
            print("Ports Setting up real-time progress monitoring...")
            print(f"   WebSocket: {self.websocket_url}")
            # Note: Full socket.io client would require python-socketio package
            # For now, we'll rely on API polling for simplicity
            return True
        except Exception as e:
            print(f"[WARNING]  WebSocket setup failed: {e}")
            print("   Continuing without real-time updates...")
            return False
    
    def get_content_description(self) -> Dict[str, str]:
        """Get content description from user (same as before)."""
        print("\\nStep Content Description")
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
        
        description = "\\n".join(description_lines).strip()
        
        return {
            "title": title,
            "topic": topic,
            "description": description if description else f"A video about {topic}"
        }
    
    def upload_file_to_api(self, file_path: str, file_type: str) -> Tuple[bool, Optional[str]]:
        """Upload file via production API.
        
        Args:
            file_path: Path to the file to upload
            file_type: 'face' or 'audio'
            
        Returns:
            Tuple of (success, uploaded_filename)
        """
        try:
            endpoint = f"/upload/{file_type}"
            
            with open(file_path, 'rb') as f:
                files = {'file': (Path(file_path).name, f)}
                data = {'session_id': self.session_id}
                
                print(f"   [EMOJI] Uploading to API: {endpoint}")
                response = requests.post(
                    f"{self.api_base_url}{endpoint}",
                    files=files,
                    data=data,
                    timeout=30
                )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    filename = result.get('filename', Path(file_path).name)
                    print(f"   [SUCCESS] Upload successful: {filename}")
                    return True, filename
                else:
                    print(f"   [ERROR] Upload failed: {result.get('message', 'Unknown error')}")
                    return False, None
            else:
                print(f"   [ERROR] Upload failed: HTTP {response.status_code}")
                if response.headers.get('content-type', '').startswith('application/json'):
                    error_data = response.json()
                    print(f"      Error: {error_data.get('message', 'Unknown error')}")
                return False, None
                
        except requests.exceptions.RequestException as e:
            print(f"   [ERROR] Upload error: {e}")
            return False, None
        except Exception as e:
            print(f"   [ERROR] Unexpected error: {e}")
            return False, None
    
    def get_file_input(self, file_type: str, display_name: str, required: bool = True) -> Optional[str]:
        """Get file input from user and upload via API.
        
        Args:
            file_type: 'face' or 'audio' for API endpoint
            display_name: Human-readable name for prompts
            required: Whether the file is required
            
        Returns:
            Uploaded filename or None
        """
        print(f"\\nAssets: {display_name} Upload")
        print("-" * 30)
        
        while True:
            # Get file path
            file_path = input(f"[EMOJI] Path to {display_name.lower()} file: ").strip()
            
            # Handle empty input
            if not file_path:
                if not required:
                    return None
                print(f"   [ERROR] {display_name} file is required. Please provide a file path.")
                continue
            
            # Remove quotes if present
            file_path = file_path.strip('\'"')
            
            # Check if file exists
            if not Path(file_path).exists():
                print(f"   [ERROR] File not found: {file_path}")
                retry = input("   [EMOJI] Try again? (y/n): ").strip().lower()
                if retry not in ['y', 'yes']:
                    if not required:
                        return None
                    print(f"   [WARNING]  {display_name} is required for video generation.")
                continue
            
            # Upload via API
            print(f"   Search Uploading and validating {display_name.lower()}...")
            success, uploaded_filename = self.upload_file_to_api(file_path, file_type)
            
            if success:
                return uploaded_filename
            else:
                # Ask if user wants to try again
                retry = input("   [EMOJI] Try again with different file? (y/n): ").strip().lower()
                if retry not in ['y', 'yes']:
                    if not required:
                        return None
                    print(f"   [WARNING]  {display_name} is required for video generation.")
    
    def get_preferences(self) -> Dict[str, str]:
        """Get user preferences (same as before)."""
        print("\\nConfiguration  Video Preferences (Optional)")
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
        print("\\nStyle: Video Tone:")
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
        print("\\n[EMOJI] Video Emotion:")
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
        print("\\nVIDEO PIPELINE Content Type:")
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
        animation_choice = input("\\nFrontend Include background animation? (y/n, default: n): ").strip().lower()
        include_animation = animation_choice in ['y', 'yes']
        
        return {
            "audience": audience,
            "tone": tone,
            "emotion": emotion,
            "content_type": content_type,
            "include_animation": include_animation
        }
    
    def confirm_inputs(self, content: Dict[str, str], face_filename: str, 
                      voice_filename: Optional[str], preferences: Dict[str, str]) -> bool:
        """Show input summary and get confirmation."""
        print("\\n" + "=" * 60)
        print("Endpoints INPUT SUMMARY")
        print("=" * 60)
        
        print(f"[EMOJI] Session ID: {self.session_id}")
        print(f"[EMOJI] Title: {content['title']}")
        print(f"Target: Topic: {content['topic']}")
        print(f"Endpoints Description: {content['description']}")
        print(f"[EMOJI] Face Image: {face_filename}")
        
        if voice_filename:
            print(f"Recording Voice Sample: {voice_filename}")
        else:
            print(f"Recording Voice Sample: None (will use default)")
        
        print(f"Multi-user Audience: {preferences['audience']}")
        print(f"Style: Tone: {preferences['tone']}")
        print(f"[EMOJI] Emotion: {preferences['emotion']}")
        print(f"VIDEO PIPELINE Content Type: {preferences['content_type']}")
        print(f"Frontend Background Animation: {'Yes' if preferences['include_animation'] else 'No'}")
        
        print("\\n" + "=" * 60)
        
        while True:
            confirm = input("[SUCCESS] Proceed with video generation? (y/n): ").strip().lower()
            if confirm in ['y', 'yes']:
                return True
            elif confirm in ['n', 'no']:
                return False
            else:
                print("   Please enter 'y' for yes or 'n' for no.")
    
    def start_video_generation(self, content: Dict[str, str], preferences: Dict[str, str]) -> bool:
        """Start video generation via production API."""
        try:
            # Prepare the request data
            process_data = {
                "session_id": self.session_id,
                "config": {
                    "title": content["title"],
                    "topic": content["topic"],
                    "description": content["description"],
                    "audience": preferences["audience"],
                    "tone": preferences["tone"],
                    "emotion": preferences["emotion"],
                    "content_type": preferences["content_type"],
                    "include_animation": preferences["include_animation"]
                }
            }
            
            print(f"\\nSTARTING Starting video generation...")
            print(f"   Session: {self.session_id}")
            
            response = requests.post(
                f"{self.api_base_url}/process/start",
                json=process_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print(f"[SUCCESS] Video generation started successfully!")
                    print(f"   Processing ID: {result.get('processing_id', 'N/A')}")
                    return True
                else:
                    print(f"[ERROR] Failed to start generation: {result.get('message', 'Unknown error')}")
                    return False
            else:
                print(f"[ERROR] API request failed: HTTP {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Error starting generation: {e}")
            return False
    
    def monitor_progress(self):
        """Monitor video generation progress."""
        print(f"\\nMonitoring Monitoring progress... (Check the web interface for real-time updates)")
        print(f"   Session status: {self.api_base_url}/session/{self.session_id}/status")
        print(f"   Web interface: http://localhost:8080")
        print(f"\\nINFO: You can also check progress via the React frontend")
    
    def collect_all_inputs(self) -> Optional[Dict[str, Any]]:
        """Main function to collect all user inputs and start generation."""
        try:
            # Check API health
            if not self.check_api_health():
                return None
            
            # Create session
            if not self.create_session():
                return None
            
            # Setup WebSocket (optional)
            self.setup_websocket()
            
            print("\\nWelcome! Let's create your video step by step.\\n")
            
            # Step 1: Get content description
            content = self.get_content_description()
            
            # Step 2: Get face image (required)
            face_filename = self.get_file_input("face", "Face Image", required=True)
            if not face_filename:
                print("[ERROR] Face image is required. Exiting.")
                return None
            
            # Step 3: Get voice sample (optional)
            print("\\nVoice sample is optional but recommended for voice cloning.")
            voice_filename = self.get_file_input("audio", "Voice Sample", required=False)
            
            # Step 4: Get preferences
            preferences = self.get_preferences()
            
            # Step 5: Confirm inputs
            if not self.confirm_inputs(content, face_filename, voice_filename, preferences):
                print("[ERROR] Video generation cancelled.")
                return None
            
            # Step 6: Start video generation
            if not self.start_video_generation(content, preferences):
                print("[ERROR] Failed to start video generation.")
                return None
            
            # Step 7: Monitor progress
            self.monitor_progress()
            
            return {
                "session_id": self.session_id,
                "session_info": self.session_info,
                "content": content,
                "face_filename": face_filename,
                "voice_filename": voice_filename,
                "preferences": preferences
            }
            
        except KeyboardInterrupt:
            print("\\n\\n[ERROR] Input collection cancelled by user.")
            return None
        except Exception as e:
            print(f"\\n[ERROR] Error during input collection: {e}")
            return None


def main():
    """Main function."""
    print("VIDEO PIPELINE Production Video Pipeline - Interactive Input Collection")
    print("=" * 60)
    
    collector = ProductionUserInputCollector()
    result = collector.collect_all_inputs()
    
    if result:
        print("\\nSUCCESS Video generation started successfully!")
        print(f"Session ID: {result['session_id']}")
        print(f"\\nAPI Monitor progress at:")
        print(f"   • Web Interface: http://localhost:8080")
        print(f"   • API Status: http://localhost:5002/session/{result['session_id']}/status")
        print(f"\\nDownload When complete, download from:")
        print(f"   • API Downloads: http://localhost:5002/outputs/{result['session_id']}")
    else:
        print("\\n[ERROR] Video generation setup failed or was cancelled.")


if __name__ == "__main__":
    main()