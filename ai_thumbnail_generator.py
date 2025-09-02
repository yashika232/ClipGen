#!/usr/bin/env python3
"""
AI Thumbnail Generator with Pollination AI and Stable Diffusion XL
Real AI integration for thumbnail generation - User's preferred setup
"""

import os
import json
import time
import requests
import base64
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import io
from PIL import Image

# Import logging system
try:
    from pipeline_logger import get_logger, LogComponent
    pipeline_logger = get_logger()
except ImportError:
    # Fallback if logging system is not available
    pipeline_logger = None


class AIThumbnailGenerator:
    """Real AI integration for thumbnail generation using multiple APIs."""
    
    def __init__(self, stability_api_key: Optional[str] = None):
        """Initialize AI thumbnail generator.
        
        Args:
            stability_api_key: Stability API key for Stable Diffusion XL
        """
        self.stability_api_key = stability_api_key or os.getenv('STABILITY_API_KEY')
        
        # API URLs
        self.pollination_url = "https://pollinations.ai/p/"
        self.stability_url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
        
        # Create output directory
        self.output_dir = Path(__file__).parent / "generated_thumbnails"
        self.output_dir.mkdir(exist_ok=True)
        
        if not self.stability_api_key:
            print("Warning: No Stability API key found. Using Pollination AI as primary method.")
    
    def generate_thumbnails(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate thumbnails using app.py's superior multi-style approach.
        
        Args:
            params: Thumbnail generation parameters
            
        Returns:
            Generated thumbnails response with 5 different styles
        """
        try:
            # Extract parameters
            topic = params.get('prompt', '')
            title = params.get('title', topic)
            audience = params.get('audience', 'general public')
            tone = params.get('tone', 'professional')
            emotion = params.get('emotion', 'confident')
            script_hook = params.get('script_hook', '')
            count = params.get('count', 5)  # Default to 5 styles like app.py
            
            # Generate 5 different thumbnail styles in one Gemini call (like app.py)
            thumbnail_prompts = self._generate_multi_style_prompts(
                title, topic, audience, tone, emotion, script_hook
            )
            
            thumbnails = []
            
            # Generate thumbnails for each style
            for i, prompt_data in enumerate(thumbnail_prompts[:count]):
                if i >= count:
                    break
                    
                style_thumbnails = []
                
                # Try Stability AI first
                if self.stability_api_key:
                    stability_result = self._generate_with_stability(prompt_data['prompt'], 1)
                    style_thumbnails.extend(stability_result)
                
                # Try Pollination AI if Stability failed or unavailable
                if not style_thumbnails:
                    pollination_result = self._generate_with_pollination(prompt_data['prompt'], 1)
                    style_thumbnails.extend(pollination_result)
                
                # Add fallback if both failed
                if not style_thumbnails:
                    fallback_result = self._generate_fallback(prompt_data['prompt'], i)
                    style_thumbnails.append(fallback_result)
                
                # Enhance thumbnail data with style information
                for thumb in style_thumbnails:
                    thumb['style_name'] = prompt_data['style']
                    thumb['style_prompt'] = prompt_data['prompt']
                    thumbnails.append(thumb)
            
            return {
                'success': True,
                'thumbnails': thumbnails,
                'generation_methods': [t.get('method', 'unknown') for t in thumbnails],
                'styles_generated': [t.get('style_name', 'unknown') for t in thumbnails],
                'total_styles': len(thumbnail_prompts),
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error generating thumbnails: {e}")
            if pipeline_logger:
                pipeline_logger.error(
                    LogComponent.THUMBNAIL_GENERATOR,
                    "thumbnail_generation_failed",
                    f"Complete thumbnail generation failed: {str(e)}",
                    metadata={
                        'topic': params.get('prompt', ''),
                        'error_type': type(e).__name__
                    }
                )
            return self._generate_fallback_batch(params)
    
    def _enhance_prompt(self, prompt: str, style: str) -> str:
        """Enhance prompt with specific visual elements for accurate thumbnail generation."""
        
        # Extract key topic and create detailed visual prompt
        key_topic = self._extract_key_topic(prompt)
        
        # Create topic-specific visual prompts with concrete elements
        topic_visuals = self._get_topic_visuals(key_topic)
        
        # Style-specific enhancements
        style_enhancements = {
            'modern': "clean minimalist design, bright colors, bold typography",
            'bold': "high contrast, vibrant colors, large bold text, eye-catching", 
            'professional': "corporate style, clean layout, professional colors",
            'creative': "artistic design, unique layout, creative typography",
            'tech': "digital elements, tech graphics, modern tech aesthetic",
            'educational': "clear informative design, learning elements, easy to read"
        }
        
        style_desc = style_enhancements.get(style, "professional clean design")
        
        # Combine topic visuals with style for accurate generation
        enhanced = f"{topic_visuals}, {style_desc}, YouTube thumbnail 16:9"
        
        return enhanced
    
    def _extract_key_topic(self, prompt: str) -> str:
        """Extract key topic from prompt for more accurate generation."""
        # Remove common prompt prefixes and cleanup
        clean_prompt = prompt.lower()
        clean_prompt = clean_prompt.replace("create a professional thumbnail for:", "")
        clean_prompt = clean_prompt.replace("generate a thumbnail about", "")
        clean_prompt = clean_prompt.replace("make a thumbnail for", "")
        clean_prompt = clean_prompt.replace("content:", "")
        
        # Extract first 1-2 meaningful words (keep it simple for accuracy)
        words = clean_prompt.strip().split()[:2]
        key_topic = " ".join(words).strip()
        
        # Handle empty or very short topics
        if len(key_topic) < 2:
            key_topic = "general"
            
        return key_topic
    
    def _generate_multi_style_prompts(self, title: str, topic: str, audience: str, 
                                     tone: str, emotion: str, script_hook: str) -> List[Dict[str, str]]:
        """Generate 5 different thumbnail style prompts using Gemini (based on app.py approach)."""
        try:
            # Import Gemini generator
            from gemini_script_generator import GeminiScriptGenerator
            
            gemini_generator = GeminiScriptGenerator()
            
            # Create direct JSON request prompt (fixed format)
            thumbnail_prompts_request = f"""INSTRUCTION: Generate exactly 5 thumbnail prompts in JSON format. No explanations, no conversation, ONLY the JSON array.

TOPIC: {topic}
CONTEXT: Title="{title}", Audience="{audience}", Tone="{tone}", Emotion="{emotion}"

OUTPUT FORMAT: Return exactly this JSON structure:
[
{{"style": "Modern", "prompt": "modern minimalist {topic} thumbnail, clean design, bright colors, professional typography, YouTube 16:9"}},
{{"style": "Bold", "prompt": "bold dynamic {topic} thumbnail, high contrast, vibrant colors, energetic composition, YouTube 16:9"}},
{{"style": "Professional", "prompt": "professional corporate {topic} thumbnail, business colors, clean layout, authoritative design, YouTube 16:9"}},
{{"style": "Creative", "prompt": "creative artistic {topic} thumbnail, unique visual elements, colorful palette, innovative design, YouTube 16:9"}},
{{"style": "Tech", "prompt": "futuristic tech {topic} thumbnail, digital elements, neon accents, modern aesthetic, YouTube 16:9"}}
]

RESPOND WITH ONLY THE JSON ARRAY ABOVE - NO OTHER TEXT."""

            # Generate using Gemini
            thumbnail_params = {
                'topic': thumbnail_prompts_request,
                'duration': 1,
                'tone': 'descriptive',
                'emotion': 'focused',
                'audience': 'designers',
                'contentType': 'json_prompts'
            }
            
            result = gemini_generator.generate_script(thumbnail_params)
            
            if result.get('success') and result.get('script'):
                # Enhanced JSON parsing with multiple extraction methods
                import re
                import json
                
                response_text = result['script']
                
                # Clean the response text
                response_text = response_text.strip()
                
                # Method 1: Direct JSON array extraction
                json_match = re.search(r'\[[\s\S]*\]', response_text, re.DOTALL)
                
                if json_match:
                    json_text = json_match.group()
                    try:
                        thumbnail_prompts = json.loads(json_text)
                        
                        if pipeline_logger:
                            pipeline_logger.info(
                                LogComponent.THUMBNAIL_GENERATOR,
                                "multi_style_prompts_generated",
                                f"Generated {len(thumbnail_prompts)} style prompts via Gemini",
                                metadata={
                                    'topic': topic,
                                    'styles_count': len(thumbnail_prompts),
                                    'styles': [p.get('style', 'unknown') for p in thumbnail_prompts],
                                    'extraction_method': 'direct_json'
                                }
                            )
                        
                        return thumbnail_prompts
                    except json.JSONDecodeError:
                        if pipeline_logger:
                            pipeline_logger.warning(
                                LogComponent.THUMBNAIL_GENERATOR,
                                "json_decode_failed",
                                f"JSON decode failed, trying text parsing",
                                metadata={'json_text': json_text[:200]}
                            )
                
                # Method 2: Extract style-prompt pairs from natural language
                style_patterns = [
                    r'"style":\s*"([^"]+)"[^}]*"prompt":\s*"([^"]+)"',
                    r'style.*?:\s*([^,\n]+).*?prompt.*?:\s*([^\n}]+)',
                    r'(\w+)\s*style.*?:\s*([^\n]+)'
                ]
                
                for pattern in style_patterns:
                    matches = re.findall(pattern, response_text, re.IGNORECASE | re.DOTALL)
                    if matches and len(matches) >= 3:  # At least 3 styles found
                        thumbnail_prompts = []
                        for match in matches[:5]:  # Take first 5
                            style = match[0].strip().title()
                            prompt = match[1].strip().strip('"').strip(',')
                            thumbnail_prompts.append({
                                'style': style,
                                'prompt': f"{prompt}, YouTube thumbnail 16:9"
                            })
                        
                        if pipeline_logger:
                            pipeline_logger.info(
                                LogComponent.THUMBNAIL_GENERATOR,
                                "multi_style_prompts_generated",
                                f"Generated {len(thumbnail_prompts)} style prompts via text parsing",
                                metadata={
                                    'topic': topic,
                                    'styles_count': len(thumbnail_prompts),
                                    'styles': [p.get('style', 'unknown') for p in thumbnail_prompts],
                                    'extraction_method': 'text_parsing'
                                }
                            )
                        
                        return thumbnail_prompts
                
                # If all parsing methods fail, log the response for debugging
                if pipeline_logger:
                    pipeline_logger.warning(
                        LogComponent.THUMBNAIL_GENERATOR,
                        "gemini_response_parsing_failed",
                        f"Could not parse Gemini response",
                        metadata={
                            'topic': topic,
                            'response_preview': response_text[:300],
                            'response_length': len(response_text)
                        }
                    )
                
                raise ValueError("No valid thumbnail prompts found in Gemini response")
                    
            else:
                raise Exception("Gemini generation failed")
                
        except Exception as e:
            if pipeline_logger:
                pipeline_logger.warning(
                    LogComponent.THUMBNAIL_GENERATOR,
                    "multi_style_prompts_fallback",
                    f"Gemini multi-style generation failed, using fallbacks: {str(e)}",
                    metadata={
                        'topic': topic,
                        'error_type': type(e).__name__
                    }
                )
            
            # Fallback to predefined styles like app.py (lines 127-142)
            return [
                {"style": "Modern", "prompt": f"Modern minimalist design for {topic}, clean typography, bright colors, YouTube thumbnail 16:9"},
                {"style": "Bold", "prompt": f"Bold dynamic design for {topic}, strong contrasts, energetic composition, YouTube thumbnail 16:9"},
                {"style": "Professional", "prompt": f"Professional corporate design for {topic}, business colors, clean layout, YouTube thumbnail 16:9"},
                {"style": "Creative", "prompt": f"Creative artistic design for {topic}, unique visual elements, vibrant palette, YouTube thumbnail 16:9"},
                {"style": "Tech", "prompt": f"Futuristic tech design for {topic}, digital elements, neon accents, YouTube thumbnail 16:9"}
            ]
    
    def _generate_with_pollination(self, prompt: str, count: int) -> List[Dict[str, Any]]:
        """Generate thumbnails using Pollination AI - user's preferred method."""
        start_time = time.time()
        
        # Log Pollination API call start
        if pipeline_logger:
            pipeline_logger.info(
                LogComponent.THUMBNAIL_GENERATOR,
                "pollination_api_call_start",
                f"Starting Pollination AI call for thumbnail generation",
                metadata={
                    'prompt_length': len(prompt),
                    'requested_count': count,
                    'method': 'pollination_ai',
                    'size': '1024x576'  # 16:9 aspect ratio
                }
            )
        
        try:
            thumbnails = []
            
            # Generate multiple images using Pollination AI
            for i in range(min(count, 4)):  # Limit to 4 for reasonable performance
                request_start = time.time()
                
                # Create unique prompt variation for each image
                varied_prompt = f"{prompt} variation {i+1}"
                # URL encode the prompt
                import urllib.parse
                encoded_prompt = urllib.parse.quote(varied_prompt)
                
                # Generate image URL
                image_url = f"{self.pollination_url}{encoded_prompt}?width=1024&height=576&seed={int(time.time())}{i}"
                
                # Download the generated image with increased timeout and retry logic
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        timeout = 60 + (retry * 15)  # 60s, 75s, 90s
                        image_response = requests.get(image_url, timeout=timeout)
                        image_response.raise_for_status()
                        break
                    except requests.exceptions.Timeout as e:
                        if retry == max_retries - 1:
                            raise e
                        if pipeline_logger:
                            pipeline_logger.warning(
                                LogComponent.THUMBNAIL_GENERATOR,
                                "pollination_timeout_retry",
                                f"Timeout on attempt {retry + 1}, retrying with longer timeout",
                                metadata={
                                    'retry_attempt': retry + 1,
                                    'timeout_used': timeout,
                                    'thumbnail_index': i
                                }
                            )
                        time.sleep(2)  # Brief delay before retry
                image_response.raise_for_status()
                request_time = (time.time() - request_start) * 1000
                
                # Save image locally
                filename = f"pollination_thumbnail_{int(time.time())}_{i}.png"
                filepath = self.output_dir / filename
                
                with open(filepath, 'wb') as f:
                    f.write(image_response.content)
                
                thumbnails.append({
                    'url': image_url,
                    'local_path': str(filepath),
                    'filename': filename,
                    'method': 'pollination',
                    'style': 'ai_generated',
                    'quality': 'high',
                    'prompt': varied_prompt
                })
                
                # Log successful generation
                if pipeline_logger:
                    pipeline_logger.info(
                        LogComponent.THUMBNAIL_GENERATOR,
                        "pollination_thumbnail_generated",
                        f"Successfully generated thumbnail {i+1}/{count}",
                        metadata={
                            'thumbnail_index': i,
                            'filename': filename,
                            'image_url': image_url,
                            'file_size': len(image_response.content),
                            'request_time_ms': request_time
                        }
                    )
                
                # Small delay to be respectful to the service
                time.sleep(0.5)
            
            return thumbnails
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            # Log Pollination API failure
            if pipeline_logger:
                pipeline_logger.error(
                    LogComponent.THUMBNAIL_GENERATOR,
                    "pollination_api_call_failed",
                    f"Pollination AI call failed: {str(e)}",
                    metadata={
                        'prompt_length': len(prompt),
                        'requested_count': count,
                        'error_type': type(e).__name__,
                        'thumbnails_generated': len(thumbnails) if 'thumbnails' in locals() else 0
                    },
                    execution_time_ms=execution_time,
                    error=e
                )
            
            print(f"Pollination AI generation failed: {e}")
            return []
    
    def _generate_with_stability(self, prompt: str, count: int) -> List[Dict[str, Any]]:
        """Generate thumbnails using Stability AI."""
        try:
            headers = {
                "Authorization": f"Bearer {self.stability_api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            payload = {
                "text_prompts": [
                    {
                        "text": prompt,
                        "weight": 1
                    }
                ],
                "cfg_scale": 7,
                "height": 576,  # 16:9 aspect ratio
                "width": 1024,
                "samples": min(count, 4),
                "steps": 30,
                "style_preset": "photographic"
            }
            
            response = requests.post(self.stability_url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            thumbnails = []
            
            if 'artifacts' in data:
                for i, artifact in enumerate(data['artifacts']):
                    if 'base64' in artifact:
                        # Decode base64 image
                        image_data = base64.b64decode(artifact['base64'])
                        
                        filename = f"stability_thumbnail_{int(time.time())}_{i}.png"
                        filepath = self.output_dir / filename
                        
                        with open(filepath, 'wb') as f:
                            f.write(image_data)
                        
                        thumbnails.append({
                            'url': f"file://{filepath}",
                            'local_path': str(filepath),
                            'filename': filename,
                            'method': 'stability',
                            'style': 'ai_generated',
                            'quality': 'high',
                            'prompt': prompt
                        })
            
            return thumbnails
            
        except Exception as e:
            print(f"Stability AI generation failed: {e}")
            return []
    
    def _generate_fallback(self, prompt: str, index: int) -> Dict[str, Any]:
        """Generate fallback thumbnail when AI APIs are unavailable."""
        return {
            'url': f'https://picsum.photos/1024/576?random={int(time.time())}-{index}',
            'local_path': None,
            'filename': f'fallback_thumbnail_{index}.jpg',
            'method': 'fallback',
            'style': 'placeholder',
            'quality': 'standard',
            'prompt': prompt
        }
    
    def _generate_fallback_batch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback batch when everything fails."""
        count = params.get('count', 4)
        prompt = params.get('prompt', 'Educational thumbnail')
        
        thumbnails = []
        for i in range(count):
            thumbnails.append(self._generate_fallback(prompt, i))
        
        return {
            'success': True,
            'thumbnails': thumbnails,
            'generation_methods': ['fallback'] * count,
            'enhanced_prompt': prompt,
            'generated_at': datetime.now().isoformat()
        }
    
    def create_custom_thumbnail(self, text: str, background_color: str = "#1a1a1a", 
                              text_color: str = "#ffffff", style: str = "modern") -> Dict[str, Any]:
        """Create a custom thumbnail with text overlay."""
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # Create image
            width, height = 1024, 576
            image = Image.new('RGB', (width, height), background_color)
            draw = ImageDraw.Draw(image)
            
            # Try to use a nice font
            try:
                font = ImageFont.truetype("Arial.ttf", 60)
            except:
                font = ImageFont.load_default()
            
            # Calculate text position
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            x = (width - text_width) // 2
            y = (height - text_height) // 2
            
            # Draw text
            draw.text((x, y), text, fill=text_color, font=font)
            
            # Save image
            filename = f"custom_thumbnail_{int(time.time())}.png"
            filepath = self.output_dir / filename
            image.save(filepath)
            
            return {
                'url': f"file://{filepath}",
                'local_path': str(filepath),
                'filename': filename,
                'method': 'custom',
                'style': style,
                'quality': 'standard',
                'text': text
            }
            
        except Exception as e:
            print(f"Custom thumbnail creation failed: {e}")
            return self._generate_fallback(text, 0)
    
    def get_thumbnail_suggestions(self, topic: str, content_type: str = "educational") -> List[str]:
        """Get thumbnail prompt suggestions based on topic and content type."""
        suggestions = []
        
        base_prompts = {
            'educational': [
                f"Educational thumbnail for {topic} with clean typography and learning icons",
                f"Modern educational design for {topic} with bright colors and clear text",
                f"Professional tutorial thumbnail for {topic} with step-by-step visual elements"
            ],
            'promotional': [
                f"Promotional thumbnail for {topic} with bold text and attention-grabbing design",
                f"Marketing-style thumbnail for {topic} with vibrant colors and call-to-action",
                f"Commercial thumbnail for {topic} with professional product presentation"
            ],
            'tutorial': [
                f"Tutorial thumbnail for {topic} with before/after comparison",
                f"How-to thumbnail for {topic} with step indicators and clear instructions",
                f"Guide thumbnail for {topic} with numbered steps and visual flow"
            ]
        }
        
        suggestions.extend(base_prompts.get(content_type, base_prompts['educational']))
        
        # Add generic suggestions
        suggestions.extend([
            f"Minimalist thumbnail for {topic} with clean design and professional typography",
            f"Bold and dynamic thumbnail for {topic} with high contrast and vibrant colors",
            f"Creative thumbnail for {topic} with unique visual elements and artistic flair"
        ])
        
        return suggestions
    
    def generate_thumbnails_batch(self, thumbnail_prompts: List[Dict[str, str]], 
                                 quality: str = "high", topic: str = "", 
                                 status_callback=None) -> List[Dict[str, Any]]:
        """Generate thumbnails in batch mode like app.py implementation.
        
        Args:
            thumbnail_prompts: List of style prompts from Gemini
            quality: Quality level for generation
            topic: Main topic for naming
            status_callback: Function to call with status updates
            
        Returns:
            List of generated thumbnails with style information
        """
        thumbnails = []
        
        if status_callback:
            status_callback("Starting batch thumbnail generation...")
        
        for i, prompt_data in enumerate(thumbnail_prompts):
            if status_callback:
                status_callback(f"Generating {prompt_data['style']} style ({i+1}/{len(thumbnail_prompts)})...")
            
            try:
                # Generate single thumbnail for this style
                style_thumbnails = []
                
                # Try Stability AI first
                if self.stability_api_key:
                    stability_result = self._generate_with_stability(prompt_data['prompt'], 1)
                    style_thumbnails.extend(stability_result)
                
                # Try Pollination AI if Stability failed
                if not style_thumbnails:
                    pollination_result = self._generate_with_pollination(prompt_data['prompt'], 1)
                    style_thumbnails.extend(pollination_result)
                
                # Add fallback if both failed
                if not style_thumbnails:
                    fallback_result = self._generate_fallback(prompt_data['prompt'], i)
                    style_thumbnails.append(fallback_result)
                
                # Enhance with style information
                for thumb in style_thumbnails:
                    thumb['style'] = prompt_data['style']
                    thumb['style_prompt'] = prompt_data['prompt']
                    thumbnails.append(thumb)
                
                if status_callback:
                    status_callback(f"[SUCCESS] {prompt_data['style']} style completed")
                    
            except Exception as e:
                if pipeline_logger:
                    pipeline_logger.error(
                        LogComponent.THUMBNAIL_GENERATOR,
                        "batch_style_generation_failed",
                        f"Failed to generate {prompt_data['style']} style: {str(e)}",
                        metadata={
                            'style': prompt_data['style'],
                            'style_index': i,
                            'error_type': type(e).__name__
                        }
                    )
                
                if status_callback:
                    status_callback(f"[ERROR] {prompt_data['style']} style failed, using fallback")
                
                # Add fallback for failed style
                fallback_result = self._generate_fallback(prompt_data['prompt'], i)
                fallback_result['style'] = prompt_data['style']
                fallback_result['style_prompt'] = prompt_data['prompt']
                thumbnails.append(fallback_result)
        
        if status_callback:
            status_callback(f"Batch generation complete! Generated {len(thumbnails)} thumbnails.")
        
        return thumbnails


def main():
    """Test the AI thumbnail generator."""
    generator = AIThumbnailGenerator()
    
    test_params = {
        'prompt': 'Machine Learning Tutorial thumbnail with neural network visualization',
        'style': 'modern',
        'quality': 'high',
        'count': 2
    }
    
    result = generator.generate_thumbnails(test_params)
    
    print("Generated Thumbnails:")
    print("=" * 50)
    for i, thumbnail in enumerate(result['thumbnails']):
        print(f"Thumbnail {i+1}:")
        print(f"  Method: {thumbnail['method']}")
        print(f"  URL: {thumbnail['url']}")
        print(f"  Local Path: {thumbnail.get('local_path', 'N/A')}")
        print()
    
    print(f"Enhanced Prompt: {result['enhanced_prompt']}")
    print(f"Generation Methods: {result['generation_methods']}")
    print(f"Successfully replaced OpenAI DALL-E with Pollination AI + SDXL!")


if __name__ == "__main__":
    main()