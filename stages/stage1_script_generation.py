#!/usr/bin/env python3
"""
Stage 1: Script Generation
Simple script generation using Gemini API
Outputs script content to JSON file for XTTS and Manim to use
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleScriptGenerator:
    """Simple script generator using Gemini API."""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize script generator."""
        self.project_root = PROJECT_ROOT
        self.config = self._load_config(config_path)
        self.outputs_dir = self.project_root / "outputs"
        self.outputs_dir.mkdir(exist_ok=True)
        
        # Load user inputs from session data
        self.session_data = self._load_session_data()
        
        logger.info("Step Simple Script Generator initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration."""
        config_file = self.project_root / config_path
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_session_data(self) -> Dict[str, Any]:
        """Load session data from JSON file."""
        session_file = self.outputs_dir / "session_data.json"
        if session_file.exists():
            with open(session_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _call_gemini_api(self, prompt: str) -> str:
        """Call Gemini API to generate script."""
        try:
            # Import Google Generative AI
            import google.generativeai as genai
            
            # Configure API key
            api_key = self.config.get("gemini_api_key") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Gemini API key not found in config or environment")
            
            genai.configure(api_key=api_key)
            
            # Initialize model
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Generate content
            logger.info("AI Calling Gemini API for script generation...")
            response = model.generate_content(prompt)
            
            if response.text:
                logger.info("[SUCCESS] Script generated successfully")
                return response.text
            else:
                raise ValueError("Empty response from Gemini API")
                
        except Exception as e:
            logger.error(f"[ERROR] Gemini API call failed: {str(e)}")
            raise
    
    def _build_prompt(self, user_inputs: Dict[str, Any]) -> str:
        """Build prompt for Gemini API."""
        
        title = user_inputs.get("title", "")
        topic = user_inputs.get("topic", "")
        audience = user_inputs.get("audience", "general")
        tone = user_inputs.get("tone", "professional")
        emotion = user_inputs.get("emotion", "confident")
        content_type = user_inputs.get("content_type", "Short-Form Video Reel")
        
        prompt = f"""Create a comprehensive teaching script for a video about:

Title: {title}
Topic: {topic}
Audience: {audience}
Tone: {tone}
Emotion: {emotion}
Content Type: {content_type}

Requirements:
1. Create an engaging script suitable for a talking head video
2. Include clear explanations with examples
3. Write in a {tone} tone with {emotion} emotion
4. Target audience: {audience}
5. Keep it concise but informative
6. Use natural speech patterns (avoid markdown formatting)
7. Include practical examples where relevant

Structure the script with:
- Hook (opening that grabs attention)
- Main content (core explanations with examples)
- Summary (key takeaways)

Important: Write the script in plain text suitable for text-to-speech synthesis. 
Avoid markdown formatting, bullet points, or complex formatting.
Write as if you're speaking directly to the audience."""

        return prompt
    
    def _clean_script_for_speech(self, script: str) -> str:
        """Clean script text for speech synthesis."""
        import re
        
        # Remove markdown headers
        script = re.sub(r'^#{1,6}\s+', '', script, flags=re.MULTILINE)
        
        # Remove markdown formatting
        script = re.sub(r'\*\*(.*?)\*\*', r'\1', script)  # Bold
        script = re.sub(r'\*(.*?)\*', r'\1', script)      # Italic
        script = re.sub(r'`(.*?)`', r'\1', script)        # Code
        
        # Remove lists and bullet points
        script = re.sub(r'^\s*[-*+]\s+', '', script, flags=re.MULTILINE)
        script = re.sub(r'^\s*\d+\.\s+', '', script, flags=re.MULTILINE)
        
        # Remove multiple newlines
        script = re.sub(r'\n\s*\n', '\n\n', script)
        
        # Remove extra whitespace
        script = re.sub(r'\s+', ' ', script)
        
        # Clean up sentences
        script = script.strip()
        
        return script
    
    def _generate_manim_code(self, script: str, user_inputs: Dict[str, Any]) -> str:
        """Generate basic Manim animation code based on script."""
        
        title = user_inputs.get("title", "Video Title")
        topic = user_inputs.get("topic", "Topic")
        
        # Basic Manim code template
        manim_code = f'''from manim import *
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
        
        return manim_code
    
    def generate_script(self) -> bool:
        """Generate script and save to outputs."""
        try:
            logger.info("VIDEO PIPELINE Starting script generation...")
            
            # Get user inputs
            user_inputs = self.session_data.get("user_inputs", {})
            if not user_inputs:
                raise ValueError("No user inputs found in session data")
            
            # Build prompt
            prompt = self._build_prompt(user_inputs)
            
            # Generate script using Gemini API
            raw_script = self._call_gemini_api(prompt)
            
            # Clean script for speech synthesis
            clean_script = self._clean_script_for_speech(raw_script)
            
            # Generate Manim code
            manim_code = self._generate_manim_code(clean_script, user_inputs)
            
            # Save generated script data
            script_data = {
                "generated_at": time.time(),
                "user_inputs": user_inputs,
                "raw_script": raw_script,
                "clean_script": clean_script,
                "manim_code": manim_code,
                "script_length": len(clean_script),
                "word_count": len(clean_script.split())
            }
            
            # Save to outputs directory
            script_file = self.outputs_dir / "generated_script.json"
            with open(script_file, 'w') as f:
                json.dump(script_data, f, indent=2)
            
            # Also save clean script as text file for easy reading
            text_file = self.outputs_dir / "clean_script.txt"
            with open(text_file, 'w') as f:
                f.write(clean_script)
            
            # Save Manim code as Python file
            manim_file = self.outputs_dir / "animation_script.py"
            with open(manim_file, 'w') as f:
                f.write(manim_code)
            
            logger.info(f"[SUCCESS] Script generated successfully:")
            logger.info(f"   Length: {len(clean_script)} characters")
            logger.info(f"   Words: {len(clean_script.split())}")
            logger.info(f"   Files: {script_file.name}, {text_file.name}, {manim_file.name}")
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Script generation failed: {str(e)}")
            return False


def main():
    """Main entry point."""
    try:
        generator = SimpleScriptGenerator()
        success = generator.generate_script()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Script generation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())