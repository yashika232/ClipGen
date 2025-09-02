#!/usr/bin/env python3
"""
Real Gemini API Integration for Script Generation
Replaces mock script generation with actual Google Gemini API calls
"""

import os
import json
import time
import requests
from typing import Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import logging system
try:
    from pipeline_logger import get_logger, LogComponent
    pipeline_logger = get_logger()
except ImportError:
    # Fallback if logging system is not available
    pipeline_logger = None


class GeminiScriptGenerator:
    """Real Gemini API integration for script generation."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini script generator.
        
        Args:
            api_key: Gemini API key (optional, can use environment variable)
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        
        # List of models to try in order of preference (updated for 2025 - non-deprecated models)
        self.model_fallbacks = [
            "gemini-2.5-flash",
            "gemini-2.5-pro", 
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite"
        ]
        self.base_url_template = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        
        if not self.api_key:
            print("Warning: No Gemini API key found. Script generation will fail without API key.")
    
    def generate_script(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate script using Gemini API.
        
        Args:
            params: Script generation parameters
            
        Returns:
            Generated script response
        """
        if not self.api_key:
            raise ValueError("Gemini API key is required for script generation. Please set GEMINI_API_KEY environment variable.")
        
        try:
            return self._generate_with_gemini(params)
        except Exception as e:
            print(f"Error generating script: {e}")
            raise e
    
    def _generate_with_gemini(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate script using real Gemini API with retry logic and model fallbacks."""
        prompt = self._create_script_prompt(params)
        
        # Try each model in the fallback list
        last_error = None
        for model_name in self.model_fallbacks:
            try:
                return self._try_model(model_name, prompt, params)
            except Exception as e:
                last_error = e
                if pipeline_logger:
                    pipeline_logger.warning(
                        LogComponent.GEMINI_GENERATOR,
                        "model_fallback",
                        f"Model {model_name} failed, trying next: {str(e)}",
                        metadata={
                            'model': model_name,
                            'error_type': type(e).__name__,
                            'topic': params.get('topic')
                        }
                    )
                continue
        
        # If all models failed, raise a user-friendly error
        user_friendly_message = self._get_user_friendly_error_message(last_error)
        raise Exception(user_friendly_message)
    
    def _try_model(self, model_name: str, prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Try generating script with a specific model."""
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 2048,
            }
        }
        
        headers = {
            "Content-Type": "application/json",
        }
        
        url = f"{self.base_url_template.format(model=model_name)}?key={self.api_key}"
        start_time = time.time()
        
        # Log API call start
        if pipeline_logger:
            pipeline_logger.info(
                LogComponent.GEMINI_GENERATOR,
                "gemini_api_call_start",
                f"Starting Gemini API call with {model_name} for topic: {params.get('topic', 'unknown')}",
                metadata={
                    'model': model_name,
                    'prompt_length': len(prompt),
                    'temperature': 0.7,
                    'max_tokens': 2048,
                    'topic': params.get('topic')
                }
            )
        
        # Try this specific model with limited retries (only for transient errors)
        max_retries = 1  # Reduced retries since we have model fallbacks
        base_delay = 2  # seconds
        
        for attempt in range(max_retries + 1):
            try:
                # Increase timeout for each retry
                timeout = 30 + (attempt * 15)  # 30s, 45s, 60s, 75s
                
                response = requests.post(url, json=payload, headers=headers, timeout=timeout)
                response.raise_for_status()
                execution_time = (time.time() - start_time) * 1000
                
                data = response.json()
                
                if 'candidates' in data and len(data['candidates']) > 0:
                    candidate = data['candidates'][0]
                    if 'content' not in candidate:
                        raise ValueError(f"No content in candidate: {candidate}")
                    
                    content = candidate['content']
                    if 'parts' not in content:
                        raise ValueError(f"No parts in content: {content}")
                    
                    parts = content['parts']
                    if not parts or 'text' not in parts[0]:
                        raise ValueError(f"No text in parts: {parts}")
                    
                    generated_text = parts[0]['text']
                    
                    # Clean the generated text - simple cleaning for natural flow
                    cleaned_text = self._clean_natural_script(generated_text)
                    
                    # Log successful API call
                    if pipeline_logger:
                        pipeline_logger.info(
                            LogComponent.GEMINI_GENERATOR,
                            "gemini_api_call_success",
                            f"Successfully generated script via Gemini API (attempt {attempt + 1})",
                            metadata={
                                'topic': params.get('topic'),
                                'generated_length': len(generated_text),
                                'cleaned_length': len(cleaned_text),
                                'word_count': len(cleaned_text.split()),
                                'response_size': len(str(data)),
                                'generation_approach': 'natural_flow',
                                'attempt': attempt + 1
                            },
                            execution_time_ms=execution_time
                        )
                    
                    return {
                        'success': True,
                        'script': cleaned_text,
                        'content': cleaned_text,
                        'generation_method': 'gemini_api_natural',
                        'estimated_duration': self._estimate_duration(cleaned_text, params),
                        'word_count': len(cleaned_text.split()),
                        'generated_at': datetime.now().isoformat()
                    }
                else:
                    raise ValueError("No content generated from Gemini API")
                    
            except (requests.exceptions.RequestException, ValueError) as e:
                execution_time = (time.time() - start_time) * 1000
                
                # Check if this is a non-retryable error
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_data = e.response.json()
                        error_code = error_data.get('error', {}).get('code')
                        error_message = error_data.get('error', {}).get('message', str(e))
                        
                        # Don't retry on certain errors - try next model instead
                        if error_code in [404, 400, 429]:  # Not found, bad request, quota exceeded
                            raise Exception(f"Model {model_name} error {error_code}: {error_message}")
                    except (ValueError, KeyError):
                        pass
                
                # Log retry attempt
                if pipeline_logger:
                    pipeline_logger.warning(
                        LogComponent.GEMINI_GENERATOR,
                        "gemini_api_retry",
                        f"Gemini API attempt {attempt + 1} failed: {str(e)}",
                        metadata={
                            'topic': params.get('topic'),
                            'error_type': type(e).__name__,
                            'status_code': getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None,
                            'attempt': attempt + 1,
                            'max_retries': max_retries,
                            'error_message': str(e)
                        },
                        execution_time_ms=execution_time
                    )
                
                # If this is the last attempt, raise the error
                if attempt == max_retries:
                    if pipeline_logger:
                        pipeline_logger.error(
                            LogComponent.GEMINI_GENERATOR,
                            "gemini_api_failed_all_attempts",
                            f"All Gemini API attempts failed after {max_retries + 1} tries",
                            metadata={
                                'topic': params.get('topic'),
                                'total_attempts': max_retries + 1,
                                'final_error': str(e)
                            }
                        )
                    
                    # Raise the error - no fallback
                    raise e
                
                # Wait before retrying (exponential backoff)
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)  # 1s, 2s, 4s
                    time.sleep(delay)
                    
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                
                # Log general API error
                if pipeline_logger:
                    pipeline_logger.error(
                        LogComponent.GEMINI_GENERATOR,
                        "gemini_api_error",
                        f"Gemini API error on attempt {attempt + 1}: {str(e)}",
                        metadata={
                            'topic': params.get('topic'),
                            'error_type': type(e).__name__,
                            'attempt': attempt + 1,
                            'error_message': str(e)
                        },
                        execution_time_ms=execution_time
                    )
                
                # If this is the last attempt, raise the error
                if attempt == max_retries:
                    raise e
                
                # Wait before retrying
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
    
    def _create_script_prompt(self, params: Dict[str, Any]) -> str:
        """Create natural, conversational script prompt - no artificial sections."""
        topic = params.get('topic', 'Educational content')
        tone = params.get('tone', 'professional')
        emotion = params.get('emotion', 'confident')
        audience = params.get('audience', 'general public')
        content_type = params.get('contentType', 'educational')
        duration = params.get('duration', 2)
        
        # Calculate target word count: 150-160 words per minute for natural speaking
        target_words = duration * 155
        
        # Create tone descriptions for more natural results
        tone_style = {
            'professional': 'clear and authoritative',
            'friendly': 'warm and approachable',
            'casual': 'relaxed and conversational',
            'enthusiastic': 'energetic and exciting',
            'educational': 'informative and engaging'
        }.get(tone, tone)
        
        emotion_style = {
            'confident': 'assured and knowledgeable',
            'excited': 'passionate and energetic',
            'calm': 'steady and reassuring',
            'curious': 'inquisitive and exploratory',
            'inspired': 'motivational and uplifting'
        }.get(emotion, emotion)
        
        prompt = f"""Write a natural, flowing {content_type} script about "{topic}" for {audience}.

STYLE REQUIREMENTS:
- Tone: {tone_style}
- Emotion: {emotion_style}
- Duration: {duration} minutes (~{target_words} words)
- Audience: {audience}

CONTENT GUIDELINES:
- Start with an engaging hook that draws viewers in immediately
- Present information in a logical, easy-to-follow flow
- Use conversational language - write as if speaking directly to the viewer
- Include specific examples, practical tips, or real-world applications
- End with a clear takeaway or call to action
- NO artificial sections or numbered lists - just natural conversation

WRITING STYLE:
- Use "you" to directly address the viewer
- Include natural transitions between ideas
- Add rhetorical questions to maintain engagement
- Use simple, clear language appropriate for {audience}
- Make it sound like a real person talking, not a corporate presentation

Write the complete script as one flowing conversation. Make it exactly {target_words} words to fit the {duration}-minute timeframe."""

        return prompt
    
    def _clean_natural_script(self, script_text: str) -> str:
        """Clean generated script for natural flow - remove artificial formatting."""
        if not script_text or not isinstance(script_text, str):
            return ""
        
        # Start with the original text
        cleaned = script_text.strip()
        
        # Remove common AI-generated artifacts
        cleaned = cleaned.replace("**", "")  # Remove markdown bold
        cleaned = cleaned.replace("##", "")  # Remove markdown headers
        cleaned = cleaned.replace("###", "")  # Remove markdown subheaders
        
        # Remove section numbers and artificial structure
        import re
        
        # Remove patterns like "1. Introduction:", "2. Main Content:", etc.
        cleaned = re.sub(r'^\d+\.\s*[A-Z][^:]*:\s*', '', cleaned, flags=re.MULTILINE)
        
        # Remove patterns like "(Introduction)", "[Hook]", etc.
        cleaned = re.sub(r'[\(\[].*?[\)\]]', '', cleaned)
        
        # Remove bullet points and convert to natural flow
        cleaned = re.sub(r'^\s*[-•*]\s*', '', cleaned, flags=re.MULTILINE)
        
        # Clean up multiple newlines - preserve natural paragraph breaks
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
        
        # Remove excessive whitespace
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        
        # Ensure proper sentence spacing
        cleaned = re.sub(r'\.\s*([A-Z])', r'. \1', cleaned)
        
        # Final cleanup
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _estimate_duration(self, script_text: str, params: Dict[str, Any]) -> float:
        """Estimate actual duration of generated script."""
        words = len(script_text.split())
        
        # Adjust WPM based on tone and emotion
        base_wpm = 160
        tone = params.get('tone', 'professional')
        emotion = params.get('emotion', 'confident')
        
        if tone in ['professional', 'calm']:
            wpm = base_wpm - 10
        elif tone in ['enthusiastic', 'excited']:
            wpm = base_wpm + 20
        else:
            wpm = base_wpm
            
        if emotion in ['excited', 'passionate']:
            wpm += 10
        elif emotion in ['calm', 'confident']:
            wpm -= 5
        
        # Calculate duration
        speaking_time = (words / wpm) * 60
        
        # Add time for pauses and transitions
        pause_time = script_text.count('.') * 0.5
        transition_time = script_text.count('\n') * 0.3
        
        total_duration = speaking_time + pause_time + transition_time
        
        return round(total_duration, 2)
    
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


def main():
    """Test the Gemini script generator."""
    generator = GeminiScriptGenerator()
    
    test_params = {
        'topic': 'Machine Learning Basics',
        'duration': 5,
        'tone': 'professional',
        'emotion': 'confident',
        'audience': 'junior engineers',
        'contentType': 'educational'
    }
    
    result = generator.generate_script(test_params)
    
    print("Generated Script:")
    print("=" * 50)
    print(result['script'])
    print("=" * 50)
    print(f"Method: {result['generation_method']}")
    print(f"Estimated Duration: {result['estimated_duration']} seconds")
    print(f"Word Count: {result['word_count']}")


if __name__ == "__main__":
    main()