#!/usr/bin/env python3
"""
Enhanced Gemini API Integration for Video Synthesis Pipeline
Integrates with metadata-driven architecture for script and animation generation
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import google.generativeai as genai

# Add the core directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_metadata_manager import EnhancedMetadataManager

# Add the project root to path for working GeminiScriptGenerator
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

from gemini_script_generator import GeminiScriptGenerator


class EnhancedGeminiIntegration:
    """Enhanced Gemini API integration with metadata-driven architecture."""
    
    def __init__(self, base_dir: str = None, api_key: str = None):
        """Initialize the enhanced Gemini integration.
        
        Args:
            base_dir: Base directory for the pipeline. Defaults to NEW directory.
            api_key: Gemini API key. If None, tries to get from environment.
        """
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        # Initialize metadata manager
        self.metadata_manager = EnhancedMetadataManager(str(self.base_dir))
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Configure Gemini API
        if api_key is None:
            api_key = os.getenv('GEMINI_API_KEY', 'AIzaSyAmNvdcDuP1X_v2VkWREZr4zYrMTrS6zhY')
        
        # Initialize the working GeminiScriptGenerator 
        self.gemini_script_generator = GeminiScriptGenerator(api_key)
        
        # Updated model fallbacks (using non-deprecated models from working version)
        self.model_fallbacks = [
            "gemini-2.5-flash",
            "gemini-2.5-pro", 
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite"
        ]
        
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_fallbacks[0])  # Use primary model
            self.logger.info("STARTING Enhanced Gemini Integration initialized")
            self.logger.info(f"   Primary model: {self.model_fallbacks[0]}")
            self.logger.info(f"   Fallback models: {len(self.model_fallbacks)} available")
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to initialize Gemini API: {e}")
            self.model = None
        
        # Script generation templates
        self.script_templates = {
            'professional': {
                'intro_style': 'clear and authoritative',
                'tone_indicators': 'confident, measured pace',
                'structure': 'problem → solution → implementation'
            },
            'friendly': {
                'intro_style': 'warm and approachable',
                'tone_indicators': 'conversational, enthusiastic',
                'structure': 'story → insight → application'
            },
            'motivational': {
                'intro_style': 'inspiring and energetic',
                'tone_indicators': 'uplifting, dynamic pace',
                'structure': 'challenge → transformation → success'
            },
            'casual': {
                'intro_style': 'relaxed and informal',
                'tone_indicators': 'natural, easy-going',
                'structure': 'context → exploration → takeaway'
            }
        }
        
        # Emotion-based modifiers
        self.emotion_modifiers = {
            'inspired': 'with inspiring examples and forward-looking perspectives',
            'confident': 'with strong assertions and clear directional guidance',
            'curious': 'with thought-provoking questions and exploratory language',
            'excited': 'with energetic language and enthusiastic discoveries',
            'calm': 'with steady, reassuring language and measured explanations'
        }
        
        # Content type specifications
        self.content_specifications = {
            'Short-Form Video Reel': {
                'duration': '60-90 seconds',
                'structure': 'hook → key point → call to action',
                'pace': 'fast, engaging',
                'focus': 'single core concept'
            },
            'Full Training Module': {
                'duration': '5-8 minutes',
                'structure': 'intro → concepts → examples → practice → summary',
                'pace': 'comprehensive, detailed',
                'focus': 'complete understanding'
            },
            'Quick Tutorial': {
                'duration': '2-3 minutes',
                'structure': 'problem → step-by-step solution → verification',
                'pace': 'direct, actionable',
                'focus': 'immediate implementation'
            },
            'Presentation': {
                'duration': '10-15 minutes',
                'structure': 'agenda → deep dive → implications → Q&A prep',
                'pace': 'thorough, professional',
                'focus': 'comprehensive coverage'
            }
        }
    
    def generate_comprehensive_content(self) -> Dict[str, Any]:
        """Generate all content types based on current session metadata.
        
        Returns:
            Dictionary with generation results
        """
        try:
            # Load current session metadata
            metadata = self.metadata_manager.load_metadata()
            if not metadata:
                return {
                    'success': False,
                    'error': 'No active session found',
                    'generated_content': {}
                }
            
            user_inputs = metadata.get('user_inputs', {})
            if not user_inputs:
                return {
                    'success': False,
                    'error': 'No user inputs found in session',
                    'generated_content': {}
                }
            
            # Generate clean script for voice cloning
            self.logger.info("Target: Generating clean script for voice cloning...")
            clean_script_result = self._generate_clean_script(user_inputs)
            
            # Generate Manim Python script for background animations
            self.logger.info("Frontend Generating Manim script for background animations...")
            manim_script_result = self._generate_manim_script(user_inputs)
            
            # Prepare results
            generated_content = {}
            generation_success = True
            errors = []
            
            if clean_script_result['success']:
                generated_content['clean_script'] = clean_script_result['script']
                self.logger.info("[SUCCESS] Clean script generated successfully")
            else:
                generation_success = False
                errors.append(f"Clean script generation failed: {clean_script_result['error']}")
            
            if manim_script_result['success']:
                generated_content['manim_script'] = manim_script_result['script']
                self.logger.info("[SUCCESS] Manim script generated successfully")
            else:
                generation_success = False
                errors.append(f"Manim script generation failed: {manim_script_result['error']}")
            
            # Update metadata with generated content
            if generation_success:
                success = self.metadata_manager.update_generated_content(
                    clean_script=generated_content.get('clean_script'),
                    manim_script=generated_content.get('manim_script'),
                    thumbnail_prompts=metadata.get('generated_content', {}).get('thumbnail_prompts', [])
                )
                
                if success:
                    self.logger.info("[SUCCESS] Metadata updated with generated content")
                else:
                    errors.append("Failed to update metadata with generated content")
            
            return {
                'success': generation_success and len(errors) == 0,
                'generated_content': generated_content,
                'errors': errors,
                'metadata_updated': generation_success and len(errors) == 0
            }
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error in comprehensive content generation: {e}")
            return {
                'success': False,
                'error': str(e),
                'generated_content': {}
            }
    
    def _generate_clean_script(self, user_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate clean script optimized for voice cloning.
        
        Args:
            user_inputs: User inputs from metadata
            
        Returns:
            Dictionary with script generation results
        """
        try:
            # Extract user inputs
            title = user_inputs.get('title', '')
            topic = user_inputs.get('topic', '')
            audience = user_inputs.get('audience', 'professionals')
            tone = user_inputs.get('tone', 'professional')
            emotion = user_inputs.get('emotion', 'confident')
            content_type = user_inputs.get('content_type', 'Short-Form Video Reel')
            additional_context = user_inputs.get('additional_context', '')
            
            # Get templates and specifications
            tone_template = self.script_templates.get(tone, self.script_templates['professional'])
            emotion_modifier = self.emotion_modifiers.get(emotion, '')
            content_spec = self.content_specifications.get(content_type, self.content_specifications['Short-Form Video Reel'])
            
            # Build conversational and natural script generation prompt
            conversational_starters = [
                f"Let's dive into {topic} and discover what makes it fascinating",
                f"Have you ever wondered about {topic}? Let me show you", 
                f"Picture this scenario involving {topic}",
                f"Here's something interesting about {topic} that you might not know"
            ]
            
            natural_transitions = [
                "But here's where it gets interesting",
                "This brings us to an important point",
                "Let me break this down for you",
                "Now, think about it this way",
                "Here's what this really means"
            ]
            
            conversational_closers = [
                f"Understanding {topic} opens up new possibilities",
                f"This knowledge about {topic} can transform how you approach",
                f"Now that you know about {topic}, you can",
                f"The key takeaway about {topic} is this"
            ]

            prompt = f"""Create a natural, conversational voice narration script for a {content_type.lower()} about "{title}" focusing on "{topic}".

TARGET AUDIENCE: {audience}
TONE: {tone} ({tone_template['tone_indicators']})
EMOTION: {emotion} {emotion_modifier}
DURATION: {content_spec['duration']}
STRUCTURE: {content_spec['structure']}

NATURAL CONVERSATION REQUIREMENTS:
1. Sound like a knowledgeable friend explaining the concept
2. Use smooth, natural speech flow without artificial breaks
3. Start with engaging conversational openers like "{conversational_starters[0]}"
4. Use natural transitions like "{natural_transitions[0]}"
5. Explain technical concepts in simple terms first, then add detail
6. AVOID: excessive ellipses (...), forced ALL CAPS emphasis, robotic phrases
7. Write numbers as words when spoken (e.g., "twenty twenty-four" not "2024")
8. End with meaningful, actionable insights

CONVERSATIONAL STYLE GUIDE:
- Instead of "Here's the thing..." use "What's fascinating is..."
- Instead of "Now, our focus shifts to..." use "This brings us to..."
- Instead of "So..." use "Here's why this matters..."
- Use natural emphasis through word choice, not CAPITALIZATION
- Let ideas flow smoothly from one to the next

CONTENT FOCUS: {content_spec['focus']}
{f"ADDITIONAL CONTEXT: {additional_context}" if additional_context else ""}

SCRIPT STRUCTURE:
- Hook: Start with {conversational_starters[1]} style opener
- Development: Build understanding through {tone_template['structure']} progression  
- Conclusion: End with {conversational_closers[0]} style closer
- Duration: Target {content_spec['duration']} of smooth, natural speech

Generate ONLY the conversational script text. Make it sound like natural human speech that flows smoothly when spoken aloud. No stage directions, no metadata, just engaging narration that feels like a real conversation."""

            # Generate content using Gemini with fallback logic
            if self.model is None:
                return {
                    'success': False,
                    'error': 'Gemini API not initialized',
                    'script': None
                }
            
            # Try each model with retry logic
            clean_script = None
            last_error = None
            
            for model_name in self.model_fallbacks:
                try:
                    self.logger.info(f"AI Trying script generation with {model_name}")
                    current_model = genai.GenerativeModel(model_name)
                    
                    # Try this model with limited retries
                    for attempt in range(2):  # 2 attempts per model
                        try:
                            response = current_model.generate_content(prompt)
                            clean_script = response.text.strip()
                            
                            if clean_script:
                                self.logger.info(f"[SUCCESS] Script generated successfully with {model_name}")
                                break
                            else:
                                raise ValueError("Empty response from Gemini")
                                
                        except Exception as e:
                            self.logger.warning(f"[WARNING] {model_name} attempt {attempt + 1} failed: {e}")
                            if attempt == 1:  # Last attempt for this model
                                raise e
                            time.sleep(2)  # Wait before retry
                    
                    if clean_script:
                        break
                        
                except Exception as e:
                    last_error = e
                    self.logger.warning(f"[ERROR] {model_name} failed completely: {e}")
                    continue
            
            if not clean_script:
                error_msg = self._get_user_friendly_error_message(last_error)
                return {
                    'success': False,
                    'error': error_msg,
                    'script': None
                }
            
            # Post-process the script for voice optimization and natural flow
            clean_script = self._optimize_script_for_voice(clean_script)
            clean_script = self._validate_and_improve_script_quality(clean_script)
            
            return {
                'success': True,
                'script': clean_script,
                'metadata': {
                    'generated_for': content_type,
                    'target_duration': content_spec['duration'],
                    'tone': tone,
                    'emotion': emotion,
                    'audience': audience
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'script': None
            }
    
    def _generate_manim_script(self, user_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Manim Python script for background animations.
        
        Args:
            user_inputs: User inputs from metadata
            
        Returns:
            Dictionary with script generation results
        """
        try:
            # Extract user inputs
            title = user_inputs.get('title', '')
            topic = user_inputs.get('topic', '')
            content_type = user_inputs.get('content_type', 'Short-Form Video Reel')
            tone = user_inputs.get('tone', 'professional')
            emotion = user_inputs.get('emotion', 'confident')
            
            # Get clean_script from metadata as educational context
            generated_content = self.metadata_manager.get_generated_content()
            clean_script = ''
            if generated_content:
                clean_script = generated_content.get('clean_script', '')
            
            if not clean_script:
                self.logger.warning("[WARNING] No clean_script found in metadata - Manim generation may be less effective")
                educational_context = f"Educational content about {topic}"
            else:
                educational_context = clean_script[:500] + "..." if len(clean_script) > 500 else clean_script
                self.logger.info(f"[SUCCESS] Using clean_script as educational context ({len(clean_script)} chars)")
            
            # Determine animation style based on content type and tone
            if content_type == 'Short-Form Video Reel':
                animation_style = 'dynamic, fast-paced with smooth transitions'
                complexity = 'moderate'
            elif content_type == 'Full Training Module':
                animation_style = 'detailed, educational with clear diagrams'
                complexity = 'high'
            elif content_type == 'Quick Tutorial':
                animation_style = 'step-by-step, practical demonstrations'
                complexity = 'moderate'
            else:  # Presentation
                animation_style = 'professional, clean with emphasis on data'
                complexity = 'high'
            
            # Analyze content to determine appropriate visual concepts
            visual_concepts = self._analyze_content_for_visuals(topic, educational_context)
            animation_patterns = self._get_animation_patterns_for_concepts(visual_concepts)
            
            # Build concept-aware 3Blue1Brown-style prompt with meaningful animations
            prompt = f"""Create a MEANINGFUL, visually engaging 3Blue1Brown-style Manim Python script for background animations about "{title}" focusing on "{topic}".

EDUCATIONAL CONTEXT (from the script narration):
{educational_context}

VISUAL CONCEPTS TO REPRESENT:
{', '.join(visual_concepts)}

ANIMATION PATTERNS TO USE:
{animation_patterns}

MEANINGFUL VISUAL STORYTELLING:
- Create animations that DIRECTLY illustrate the concepts being explained
- Use visual metaphors that help viewers understand the topic
- Build a coherent visual narrative across the three sequences
- Each animation should reinforce what's being said in the narration

3BLUE1BROWN COLOR PALETTE (use these exact colors):
primary_blue = "#1f77b4"
accent_yellow = "#ff7f0e" 
soft_green = "#2ca02c"
subtle_gray = "#7f7f7f"

CONCEPT-SPECIFIC MANIM EXAMPLES (adapt these patterns for your topic):

# Example 1: Data Flow Animation (for pipeline/process topics)
source = Circle(radius=0.8, color=primary_blue, fill_opacity=0.3).shift(LEFT * 3)
target = Square(side_length=1.5, color=soft_green, fill_opacity=0.3).shift(RIGHT * 3)
flow_line = Arrow(source.get_right(), target.get_left(), color=accent_yellow, stroke_width=3)
data_dot = Dot(color=accent_yellow).move_to(source.get_center())
self.play(Create(source), Create(target), run_time=1)
self.play(Create(flow_line), run_time=1)
self.play(MoveAlongPath(data_dot, flow_line), run_time=2)
self.wait(1)

# Example 2: Network/Connection Animation (for network/integration topics)
nodes = [Circle(radius=0.4, color=primary_blue, fill_opacity=0.3) for _ in range(4)]
positions = [LEFT*2+UP*1, RIGHT*2+UP*1, LEFT*2+DOWN*1, RIGHT*2+DOWN*1]
for i, node in enumerate(nodes):
    node.shift(positions[i])
self.play(*[Create(node) for node in nodes], run_time=2)
connections = [Line(nodes[i].get_center(), nodes[j].get_center(), color=soft_green, stroke_width=2) 
               for i in range(len(nodes)) for j in range(i+1, len(nodes))]
self.play(*[Create(conn) for conn in connections], run_time=2)
self.wait(1)

# Example 3: Progress/Improvement Animation (for optimization/learning topics)
progress_bar_bg = Rectangle(width=4, height=0.5, color=subtle_gray, fill_opacity=0.3)
progress_bar_fill = Rectangle(width=0.1, height=0.5, color=accent_yellow, fill_opacity=0.8)
progress_bar_fill.align_to(progress_bar_bg, LEFT)
self.play(Create(progress_bar_bg), run_time=1)
self.play(progress_bar_fill.animate.scale_to_fit_width(4), run_time=3)
checkmark = Text("[EMOJI]", color=soft_green, font_size=36).next_to(progress_bar_bg, RIGHT)
self.play(Write(checkmark), run_time=1)
self.wait(1)

# Example 4: Synchronization/Alignment Animation (for timing/validation topics)
wave1 = FunctionGraph(lambda x: np.sin(x), x_range=[-3, 3], color=primary_blue)
wave2 = FunctionGraph(lambda x: np.sin(x + PI/2), x_range=[-3, 3], color=accent_yellow).shift(DOWN*0.5)
self.play(Create(wave1), Create(wave2), run_time=2)
self.play(wave2.animate.shift(UP*0.5), run_time=2)  # Align waves
sync_text = Text("SYNCHRONIZED", color=soft_green, font_size=24).next_to(wave1, UP)
self.play(Write(sync_text), run_time=1)
self.wait(1)

PROHIBITED PATTERNS (NEVER use these):
- VGroup().shuffle() (returns None)
- Complex list comprehensions in self.play()
- Helper methods or functions
- .animate.shift() with complex calculations
- opacity parameter (use stroke_opacity/fill_opacity)
- Complex loops inside animations
- Random number generation in animations
- get_top_edge(), get_bottom_edge(), get_left_edge(), get_right_edge() (deprecated)
- ALWAYS use get_top(), get_bottom(), get_left(), get_right() instead

REQUIRED CONCEPT-DRIVEN SCRIPT STRUCTURE:
```python
from manim import *
import numpy as np

config.frame_rate = 30
config.pixel_height = 1080
config.pixel_width = 1920

class BackgroundAnimation(Scene):
    def construct(self):
        # Define colors
        primary_blue = "#1f77b4"
        accent_yellow = "#ff7f0e"
        soft_green = "#2ca02c"
        subtle_gray = "#7f7f7f"
        
        # Sequence 1: Introduce the main concept visually (8-12 lines)
        # Create visual elements that represent the core topic
        # Use meaningful shapes, text, or diagrams that relate to the subject
        # AVOID generic circles/squares - use concept-specific visuals
        
        # Sequence 2: Demonstrate the concept in action (10-15 lines)
        # Show the process, workflow, or transformation being explained
        # Use animations that mirror what's being said in the narration
        # Include movement, transformation, or interaction that teaches
        
        # Sequence 3: Reinforce key insights and conclusion (8-12 lines)
        # Highlight the benefits, results, or key takeaways visually
        # Use completion animations, success indicators, or summary visuals
        # End with a strong visual that reinforces the main message
        
        # Each sequence: setup → meaningful animation → emphasis → transition
```

CRITICAL API REQUIREMENTS (follow exactly):
- Circle(radius=1, color=color, fill_opacity=0.3, stroke_width=2)
- Square(side_length=2, color=color, fill_opacity=0.3, stroke_width=2)  
- Text("content", color=color, font_size=48)
- Line(start_point, end_point, stroke_width=2, stroke_color=color)
- self.play(Create(object), run_time=2)
- self.play(Write(text), run_time=2) 
- self.play(FadeOut(object), run_time=1)
- self.wait(1)

IMPORTANT: DEPRECATED METHODS TO AVOID:
- get_top_edge() → use get_top() instead
- get_bottom_edge() → use get_bottom() instead  
- get_left_edge() → use get_left() instead
- get_right_edge() → use get_right() instead
- opacity parameter → use fill_opacity and stroke_opacity instead

CORRECT POSITION METHODS:
- object.get_top() - get top edge position
- object.get_bottom() - get bottom edge position
- object.get_left() - get left edge position  
- object.get_right() - get right edge position
- object.get_center() - get center position

CONCEPT-DRIVEN ANIMATION FLOW:
1. Analyze the topic and create visuals that directly represent the concepts
2. Use the identified animation patterns to show meaningful transformations  
3. Build a coherent visual story across the three sequences
4. Ensure animations reinforce and illustrate what's being narrated
5. Total: 30-40 lines of meaningful, concept-specific code

Generate a MEANINGFUL, concept-driven Manim script that directly illustrates "{topic}" using the specified visual concepts and animation patterns. The animations should teach and reinforce the subject matter, not just be decorative. Make every visual element purposeful and educational."""

            # Generate content using the enhanced prompt directly with Gemini API
            try:
                # Use direct Gemini API call with the comprehensive 3Blue1Brown prompt
                # instead of the basic GeminiScriptGenerator to ensure full prompt control
                if self.model is None:
                    return {
                        'success': False,
                        'error': 'Gemini API not initialized',
                        'script': None
                    }
                
                # Try each model with retry logic using the enhanced 3Blue1Brown prompt
                manim_script = None
                last_error = None
                
                for model_name in self.model_fallbacks:
                    try:
                        self.logger.info(f"Frontend Generating 3Blue1Brown-style Manim script with {model_name}")
                        current_model = genai.GenerativeModel(model_name)
                        
                        # Try this model with limited retries
                        for attempt in range(2):  # 2 attempts per model
                            try:
                                response = current_model.generate_content(prompt)
                                manim_script = response.text.strip()
                                
                                if manim_script:
                                    self.logger.info(f"[SUCCESS] 3Blue1Brown Manim script generated successfully with {model_name}")
                                    break
                                else:
                                    raise ValueError("Empty response from Gemini")
                                    
                            except Exception as e:
                                self.logger.warning(f"[WARNING] {model_name} attempt {attempt + 1} failed: {e}")
                                if attempt == 1:  # Last attempt for this model
                                    raise e
                                time.sleep(2)  # Wait before retry
                        
                        if manim_script:
                            break
                            
                    except Exception as e:
                        last_error = e
                        self.logger.warning(f"[ERROR] {model_name} failed completely: {e}")
                        continue
                
                if not manim_script:
                    error_msg = self._get_user_friendly_error_message(last_error)
                    return {
                        'success': False,
                        'error': f"3Blue1Brown Manim script generation failed: {error_msg}",
                        'script': None
                    }
                    
            except Exception as e:
                return {
                    'success': False,
                    'error': f"Failed to generate Manim script: {str(e)}",
                    'script': None
                }
            
            # Post-process the script to ensure proper Manim syntax
            manim_script = self._optimize_manim_script(manim_script)
            
            return {
                'success': True,
                'script': manim_script,
                'metadata': {
                    'animation_style': animation_style,
                    'complexity': complexity,
                    'content_type': content_type,
                    'target_resolution': '1920x1080',
                    'frame_rate': 30
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'script': None
            }
    
    def _optimize_script_for_voice(self, script: str) -> str:
        """Optimize script for better voice synthesis.
        
        Args:
            script: Raw script text
            
        Returns:
            Optimized script text
        """
        # Remove any stage directions or metadata
        lines = script.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines, stage directions, or metadata
            if (line and 
                not line.startswith('[') and 
                not line.startswith('(') and
                not line.startswith('Stage:') and
                not line.startswith('Note:') and
                not line.startswith('Duration:') and
                not line.startswith('Tone:')):
                clean_lines.append(line)
        
        # Join and clean up
        optimized_script = '\n\n'.join(clean_lines)
        
        # Ensure proper punctuation for speech
        optimized_script = optimized_script.replace('..', '...')  # Normalize pauses
        optimized_script = optimized_script.replace(' ,', ',')   # Fix spacing
        optimized_script = optimized_script.replace(' .', '.')   # Fix spacing
        
        return optimized_script.strip()
    
    def _validate_and_improve_script_quality(self, script: str) -> str:
        """Validate and improve script quality for natural conversational flow.
        
        Args:
            script: Script text to validate and improve
            
        Returns:
            Improved script with natural conversational flow
        """
        import re
        
        # Remove excessive ellipses that break natural flow
        script = re.sub(r'\.{4,}', '.', script)  # Remove 4+ dots
        script = re.sub(r'\.{3}\.+', '...', script)  # Normalize ellipses to max 3 dots
        script = script.replace('....', '.').replace('...', '.')  # Remove most ellipses for smoother flow
        
        # Fix robotic phrases with more natural alternatives
        robotic_replacements = {
            r'\bHere\'s the thing[.,]*': 'What\'s fascinating is',
            r'\bNow, our focus shifts to\b': 'This brings us to',
            r'\bSo,?\s*': 'Here\'s why this matters: ',
            r'\bIt\'s a fundamental challenge\b': 'This is where things get interesting',
            r'\bThis isn\'t merely about\b': 'This goes beyond just',
            r'\bSystematically ensuring\b': 'making sure we get',
            r'\bPrecise.*?articulation\b': 'clear speech that looks natural',
            r'\bRigorously compared\b': 'carefully matched',
            r'\bQuantify.*?accuracy\b': 'measure how well things sync',
            r'\bDefinitive step towards\b': 'key to achieving',
            r'\brigorous\b': 'thorough',
            r'\brigorously\b': 'carefully',
            r'\bRigorous\b': 'Thorough',
            r'\bcomprehensive testing framework\b': 'thorough testing approach',
            r'\bvalidation framework\b': 'checking process',
            r'\bsystematic validation\b': 'careful checking',
            r'\bmethodology\b': 'approach',
            r'\bfundamental\b': 'important',
            r'\boptimization methodologies\b': 'ways to improve things'
        }
        
        for pattern, replacement in robotic_replacements.items():
            script = re.sub(pattern, replacement, script, flags=re.IGNORECASE)
        
        # Reduce excessive ALL CAPS emphasis to natural emphasis
        script = re.sub(r'\b[A-Z]{4,}\b', lambda m: m.group().lower(), script)  # Convert long ALL CAPS to lowercase
        script = re.sub(r'\b[A-Z]{2,3}(?=[.,!?\s])', lambda m: m.group().lower(), script)  # Convert short ALL CAPS followed by punctuation
        
        # Improve sentence flow by connecting short, choppy sentences
        sentences = script.split('. ')
        improved_sentences = []
        
        for i, sentence in enumerate(sentences):
            if len(sentence.strip().split()) < 4 and i > 0 and i < len(sentences) - 1:
                # Short sentence - try to connect with previous or next
                if len(improved_sentences) > 0:
                    # Connect with previous sentence using natural connectors
                    connectors = [', which means ', ', and this ', ', so ']
                    connector = connectors[i % len(connectors)]
                    improved_sentences[-1] = improved_sentences[-1] + connector + sentence.strip().lower()
                    continue
            
            improved_sentences.append(sentence.strip())
        
        script = '. '.join(improved_sentences)
        
        # Ensure natural breathing pauses with commas instead of artificial breaks  
        script = re.sub(r'([a-z]),\s*\.{3}', r'\1.', script)  # Remove ellipses after commas
        script = re.sub(r'([A-Z][a-z]+)\.\s*([A-Z][a-z]+)\.\s*([A-Z])', r'\1, \2. \3', script)  # Connect short statements
        
        # Improve technical explanations with simpler language first
        technical_improvements = {
            r'synchronization accuracy': 'how well the lips match the speech',
            r'comprehensive testing framework': 'thorough testing process',
            r'systematic.*?validation': 'careful checking',
            r'ground truth audio': 'original audio',
            r'quantify.*?performance': 'measure how well it works',
            r'production-ready': 'professional-quality'
        }
        
        for pattern, replacement in technical_improvements.items():
            script = re.sub(pattern, replacement, script, flags=re.IGNORECASE)
        
        # Ensure proper sentence ending punctuation
        sentences = [s.strip() for s in script.split('.') if s.strip()]
        final_sentences = []
        
        for sentence in sentences:
            if sentence and not sentence.endswith(('!', '?', '.')):
                sentence += '.'
            final_sentences.append(sentence)
        
        # Join with proper spacing and ensure smooth transitions
        improved_script = ' '.join(final_sentences).strip()
        
        # Final cleanup: ensure natural paragraph breaks
        improved_script = re.sub(r'\.(\s+)([A-Z])', r'. \2', improved_script)  # Proper spacing after periods
        improved_script = re.sub(r'\s+', ' ', improved_script)  # Remove multiple spaces
        
        return improved_script
    
    def _analyze_content_for_visuals(self, topic: str, context: str) -> List[str]:
        """Analyze topic and context to identify key visual concepts to represent.
        
        Args:
            topic: The main topic of the video
            context: Educational context from the script
            
        Returns:
            List of visual concepts that should be animated
        """
        # Combine topic and context for analysis
        full_content = f"{topic} {context}".lower()
        
        # Define concept mappings based on common educational topics
        concept_mappings = {
            # Technology & Programming
            'neural network': ['network_nodes', 'data_flow', 'learning_process'],
            'machine learning': ['data_transformation', 'model_training', 'prediction_accuracy'],
            'pipeline': ['data_flow', 'process_stages', 'transformation_steps'],
            'algorithm': ['step_by_step_process', 'decision_trees', 'optimization'],
            'database': ['data_storage', 'connections', 'query_flow'],
            'api': ['request_response', 'data_exchange', 'system_integration'],
            
            # Science & Math
            'synchronization': ['wave_alignment', 'timing_coordination', 'phase_matching'],
            'testing': ['validation_checkmarks', 'quality_gates', 'verification_process'],
            'validation': ['comparison_checks', 'accuracy_measurement', 'quality_standards'],
            'optimization': ['improvement_curves', 'efficiency_gains', 'goal_achievement'],
            'analysis': ['data_breakdown', 'pattern_recognition', 'insight_extraction'],
            
            # Business & Process
            'workflow': ['process_flow', 'task_sequence', 'efficiency_improvement'],
            'automation': ['manual_to_automatic', 'process_optimization', 'time_savings'],
            'integration': ['system_connection', 'unified_platform', 'seamless_flow'],
            'scalability': ['growth_patterns', 'expansion_visualization', 'capacity_increase'],
            
            # General concepts  
            'improvement': ['before_after_comparison', 'progress_indication', 'quality_enhancement'],
            'connection': ['network_links', 'relationship_mapping', 'information_flow'],
            'transformation': ['change_visualization', 'evolution_process', 'state_transition'],
            'comparison': ['side_by_side_analysis', 'difference_highlighting', 'evaluation_metrics']
        }
        
        # Find relevant concepts
        identified_concepts = []
        for keyword, concepts in concept_mappings.items():
            if keyword in full_content:
                identified_concepts.extend(concepts)
        
        # If no specific concepts found, use general educational concepts
        if not identified_concepts:
            if any(word in full_content for word in ['learn', 'teach', 'explain', 'understand']):
                identified_concepts = ['knowledge_transfer', 'concept_illustration', 'step_by_step_learning']
            else:
                identified_concepts = ['information_flow', 'concept_development', 'visual_explanation']
        
        # Remove duplicates and limit to 3 main concepts
        unique_concepts = list(dict.fromkeys(identified_concepts))[:3]
        
        return unique_concepts if unique_concepts else ['concept_illustration', 'information_flow', 'visual_explanation']
    
    def _get_animation_patterns_for_concepts(self, visual_concepts: List[str]) -> str:
        """Generate specific animation patterns for the identified visual concepts.
        
        Args:
            visual_concepts: List of visual concepts to animate
            
        Returns:
            String describing specific animation patterns to use
        """
        # Define animation patterns for different concept types
        pattern_mappings = {
            'network_nodes': 'Create circles connected by lines, animate connections growing',
            'data_flow': 'Show arrows moving along paths, data packets traveling through systems',
            'learning_process': 'Transform simple shapes into complex ones, show gradual improvement',
            'data_transformation': 'Morph shapes from one form to another, show input becoming output',
            'model_training': 'Show iterative improvements, curves rising to show progress',
            'prediction_accuracy': 'Show target vs achieved results with alignment animations',
            'process_stages': 'Show sequential steps with checkpoints and progress indicators',
            'transformation_steps': 'Multi-stage morphing animations showing each phase',
            'step_by_step_process': 'Sequential reveals with clear numbered stages',
            'decision_trees': 'Branching animations showing different paths and outcomes',
            'optimization': 'Show curves improving, peaks being reached, efficiency gains',
            'data_storage': 'Show organized containers filling up, categorized information',
            'connections': 'Lines growing between elements, network formation',
            'query_flow': 'Show search paths, information retrieval, results highlighting',
            'request_response': 'Back-and-forth animations showing communication flow',
            'data_exchange': 'Bidirectional arrows, information passing between systems',
            'system_integration': 'Separate elements coming together into unified whole',
            'wave_alignment': 'Synchronized wave patterns, phase matching animations',
            'timing_coordination': 'Multiple elements moving in perfect synchronization',
            'phase_matching': 'Wave patterns aligning, frequency synchronization',
            'validation_checkmarks': 'Progressive validation with checkmarks appearing',
            'quality_gates': 'Approval processes, gates opening after validation',
            'verification_process': 'Comparison animations, accuracy checking',
            'comparison_checks': 'Side-by-side elements showing differences and matches',
            'accuracy_measurement': 'Precision indicators, measurement visualizations',
            'quality_standards': 'Benchmark comparisons, standard achievement indicators',
            'improvement_curves': 'Rising curves, progress indicators, enhancement visualization',
            'efficiency_gains': 'Streamlined processes, reduced complexity animations',
            'goal_achievement': 'Target reaching, milestone celebrations, success indicators',
            'data_breakdown': 'Complex information splitting into understandable parts',
            'pattern_recognition': 'Highlighting similar elements, pattern emergence',
            'insight_extraction': 'Key information highlighting, revelation moments',
            'process_flow': 'Sequential workflow with clear directional movement',
            'task_sequence': 'Ordered operations with completion indicators',
            'efficiency_improvement': 'Before/after comparisons showing streamlined processes',
            'manual_to_automatic': 'Hand operations transitioning to automated systems',
            'process_optimization': 'Workflow refinement, unnecessary steps elimination',
            'time_savings': 'Clock animations showing reduced time, speed improvements',
            'system_connection': 'Integration animations, unified system formation',
            'unified_platform': 'Multiple elements converging into single interface',
            'seamless_flow': 'Smooth transitions, uninterrupted movement patterns',
            'growth_patterns': 'Expanding networks, scaling visualizations',
            'expansion_visualization': 'Size increases, capability growth animations',
            'capacity_increase': 'Container expansions, capability scaling',
            'before_after_comparison': 'Split-screen improvements, transformation reveals',
            'progress_indication': 'Progress bars, step completion, advancement tracking',
            'quality_enhancement': 'Refinement processes, clarity improvements',
            'network_links': 'Connection establishment, link strengthening animations',
            'relationship_mapping': 'Connection diagrams, interdependency visualization',
            'information_flow': 'Data streams, knowledge transfer pathways',
            'change_visualization': 'Transformation sequences, evolution animations',
            'evolution_process': 'Gradual development, staged improvements',
            'state_transition': 'Clear before/after states with smooth transitions',
            'side_by_side_analysis': 'Comparative layouts, difference highlighting',
            'difference_highlighting': 'Contrast emphasis, distinction clarification',
            'evaluation_metrics': 'Scoring systems, assessment visualizations',
            'knowledge_transfer': 'Information moving from source to recipient',
            'concept_illustration': 'Abstract ideas becoming concrete visuals',
            'step_by_step_learning': 'Progressive understanding, building knowledge',
            'visual_explanation': 'Complex concepts broken into simple visual elements'
        }
        
        # Generate specific patterns for the identified concepts
        patterns = []
        for concept in visual_concepts:
            if concept in pattern_mappings:
                patterns.append(f"- {concept.replace('_', ' ').title()}: {pattern_mappings[concept]}")
            else:
                patterns.append(f"- {concept.replace('_', ' ').title()}: Visual representation of this concept")
        
        return '\n'.join(patterns)
    
    def _optimize_manim_script(self, script: str) -> str:
        """Optimize Manim script for proper syntax and execution.
        
        Args:
            script: Raw Manim script
            
        Returns:
            Optimized Manim script
        """
        # Ensure proper imports at the beginning
        imports = """from manim import *
import numpy as np

config.frame_rate = 30
config.pixel_height = 1080
config.pixel_width = 1920

"""
        
        # Remove markdown code block markers if present
        script = script.replace('```python', '').replace('```', '')
        
        # Extract only valid Python code lines while preserving indentation
        lines = script.split('\n')
        python_lines = []
        in_python_section = False
        
        for line in lines:
            original_line = line  # Keep original line with indentation
            stripped_line = line.strip()
            
            # Skip empty lines initially
            if not stripped_line:
                if in_python_section:
                    python_lines.append('')
                continue
            
            # Check if this is a Python code line
            if (stripped_line.startswith('from ') or 
                stripped_line.startswith('import ') or 
                stripped_line.startswith('config.') or
                stripped_line.startswith('class ') or
                stripped_line.startswith('def ') or
                original_line.startswith('    ') or  # Indented code (check original line)
                original_line.startswith('\t') or    # Tab-indented code
                stripped_line.startswith('#') or    # Comments
                stripped_line.endswith(':') or     # Python statements ending with colon
                in_python_section):
                
                # Skip explanatory text mixed with code
                if ('Here is' in stripped_line or 
                    'script for' in stripped_line or 
                    '**' in stripped_line or
                    'Visual:' in stripped_line or
                    'By the end' in stripped_line or
                    'At its core' in stripped_line or
                    'Perhaps the most' in stripped_line):
                    continue
                
                python_lines.append(original_line)  # Preserve original indentation
                in_python_section = True
            else:
                # Reset if we hit non-Python content
                if in_python_section and stripped_line.startswith('class '):
                    python_lines.append(original_line)
                elif not ('**' in stripped_line or 'Visual:' in stripped_line or 'By the end' in stripped_line):
                    in_python_section = False
        
        # Join the extracted Python code
        clean_script = '\n'.join(python_lines)
        
        # Ensure the script starts with imports
        if 'from manim import' not in clean_script:
            clean_script = imports + clean_script
        
        # Ensure proper class structure - NO FALLBACKS, fail if Gemini doesn't generate proper code
        if 'class BackgroundAnimation' not in clean_script and 'class ' not in clean_script:
            # No fallback templates - return the script as-is and let it fail if invalid
            # This forces improvement of the Gemini prompt instead of relying on templates
            pass
        
        return clean_script.strip()
    
    def _get_user_friendly_error_message(self, error: Exception) -> str:
        """Convert technical API errors into user-friendly messages."""
        error_str = str(error).lower()
        
        if "503" in error_str or "service unavailable" in error_str or "overloaded" in error_str:
            return "[EMOJI] The AI script generation service is temporarily overloaded. Please try again in a few minutes."
        elif "429" in error_str or "quota" in error_str or "rate limit" in error_str:
            return "Duration: Too many requests. Please wait a moment and try again."
        elif "401" in error_str or "unauthorized" in error_str or "api key" in error_str:
            return "API Keys API authentication failed. Please check your API key configuration."
        elif "400" in error_str or "bad request" in error_str:
            return "Step Invalid request format. Please check your input parameters."
        elif "404" in error_str or "not found" in error_str:
            return "Search AI model not found. This may be a temporary issue."
        elif "timeout" in error_str or "connection" in error_str:
            return "API Connection timeout. Please check your internet connection and try again."
        else:
            return "[WARNING] AI script generation is temporarily unavailable. Please try again later."
    
    def regenerate_content(self, content_type: str) -> Dict[str, Any]:
        """Regenerate specific content type.
        
        Args:
            content_type: Type of content to regenerate ('clean_script' or 'manim_script')
            
        Returns:
            Dictionary with regeneration results
        """
        try:
            metadata = self.metadata_manager.load_metadata()
            if not metadata:
                return {'success': False, 'error': 'No active session found'}
            
            user_inputs = metadata.get('user_inputs', {})
            if not user_inputs:
                return {'success': False, 'error': 'No user inputs found'}
            
            if content_type == 'clean_script':
                result = self._generate_clean_script(user_inputs)
                if result['success']:
                    # Update metadata
                    self.metadata_manager.update_generated_content(
                        clean_script=result['script'],
                        manim_script=metadata.get('generated_content', {}).get('manim_script'),
                        thumbnail_prompts=metadata.get('generated_content', {}).get('thumbnail_prompts', [])
                    )
                return result
            
            elif content_type == 'manim_script':
                result = self._generate_manim_script(user_inputs)
                if result['success']:
                    # Update metadata
                    self.metadata_manager.update_generated_content(
                        clean_script=metadata.get('generated_content', {}).get('clean_script'),
                        manim_script=result['script'],
                        thumbnail_prompts=metadata.get('generated_content', {}).get('thumbnail_prompts', [])
                    )
                return result
            
            else:
                return {'success': False, 'error': f'Unknown content type: {content_type}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_generation_status(self) -> Dict[str, Any]:
        """Get current generation status from metadata.
        
        Returns:
            Dictionary with generation status
        """
        try:
            metadata = self.metadata_manager.load_metadata()
            if not metadata:
                return {'success': False, 'error': 'No active session found'}
            
            generated_content = metadata.get('generated_content', {})
            
            status = {
                'clean_script_generated': bool(generated_content.get('clean_script')),
                'manim_script_generated': bool(generated_content.get('manim_script')),
                'thumbnail_prompts_generated': len(generated_content.get('thumbnail_prompts', [])) > 0,
                'generation_timestamp': generated_content.get('generated_at'),
                'content_preview': {}
            }
            
            # Add content previews
            if status['clean_script_generated']:
                clean_script = generated_content['clean_script']
                status['content_preview']['clean_script'] = clean_script[:100] + "..." if len(clean_script) > 100 else clean_script
            
            if status['manim_script_generated']:
                manim_script = generated_content['manim_script']
                lines = manim_script.split('\n')
                status['content_preview']['manim_script'] = f"{len(lines)} lines of Manim code"
            
            return {'success': True, 'status': status}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}


def main():
    """Test the enhanced Gemini integration."""
    print("🧪 Testing Enhanced Gemini Integration")
    print("=" * 50)
    
    # Initialize integration
    integration = EnhancedGeminiIntegration()
    
    # Check if Gemini API is available
    if integration.model is None:
        print("[ERROR] Gemini API not available - skipping generation tests")
        return
    
    # Test generation status
    status_result = integration.get_generation_status()
    if status_result['success']:
        status = status_result['status']
        print(f"[SUCCESS] Current generation status:")
        print(f"   Clean script: {'[SUCCESS]' if status['clean_script_generated'] else '[ERROR]'}")
        print(f"   Manim script: {'[SUCCESS]' if status['manim_script_generated'] else '[ERROR]'}")
        print(f"   Thumbnail prompts: {'[SUCCESS]' if status['thumbnail_prompts_generated'] else '[ERROR]'}")
    else:
        print(f"[ERROR] Status check failed: {status_result['error']}")
        return
    
    # Test comprehensive content generation if not already generated
    if not (status['clean_script_generated'] and status['manim_script_generated']):
        print("\nTarget: Testing comprehensive content generation...")
        generation_result = integration.generate_comprehensive_content()
        
        if generation_result['success']:
            print("[SUCCESS] Content generation completed successfully")
            generated = generation_result['generated_content']
            if 'clean_script' in generated:
                print(f"   Clean script: {len(generated['clean_script'])} characters")
            if 'manim_script' in generated:
                print(f"   Manim script: {len(generated['manim_script'])} characters")
        else:
            print(f"[ERROR] Content generation failed: {generation_result['errors']}")
    else:
        print("\n[SUCCESS] Content already generated - skipping generation test")
    
    print("\nSUCCESS Enhanced Gemini Integration testing completed!")


if __name__ == "__main__":
    main()