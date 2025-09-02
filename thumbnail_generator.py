#!/usr/bin/env python3
"""
Legacy Thumbnail Generator Wrapper
Provides backward compatibility for app.py while using the new AI thumbnail generator
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from ai_thumbnail_generator import AIThumbnailGenerator

class ThumbnailGenerator:
    """Legacy thumbnail generator interface for backward compatibility."""
    
    def __init__(self):
        """Initialize the thumbnail generator."""
        self.ai_generator = AIThumbnailGenerator()
        self.presets = {
            "high_quality": {"width": 1024, "height": 576, "count": 4, "style": "professional"},
            "quick_draft": {"width": 1024, "height": 576, "count": 2, "style": "modern"}
        }
    
    def generate_thumbnail_local(self, prompt: str, preset: str = "quick_draft", 
                               topic_title: Optional[str] = None, 
                               progress_callback: Optional[Callable] = None) -> Optional[Dict[str, Any]]:
        """Generate thumbnail using local SDXL if available - legacy interface."""
        try:
            if progress_callback:
                progress_callback("Starting AI thumbnail generation...")
            
            config = self.presets.get(preset, self.presets["quick_draft"])
            
            # Use the new AI thumbnail generator
            params = {
                'prompt': prompt,
                'style': config.get('style', 'modern'),
                'quality': 'high' if preset == "high_quality" else 'standard',
                'count': config.get('count', 2)
            }
            
            if progress_callback:
                progress_callback("Generating with Pollination AI and SDXL...")
            
            result = self.ai_generator.generate_thumbnails(params)
            
            if result['success'] and result['thumbnails']:
                # Return first thumbnail in legacy format
                thumbnail = result['thumbnails'][0]
                
                if progress_callback:
                    progress_callback(f"Generated thumbnail using {thumbnail['method']}")
                
                return {
                    'success': True,
                    'image_path': thumbnail.get('local_path'),
                    'image_url': thumbnail.get('url'),
                    'method': thumbnail.get('method'),
                    'style': thumbnail.get('style'),
                    'all_thumbnails': result['thumbnails'],  # Include all generated thumbnails
                    'enhanced_prompt': result.get('enhanced_prompt', prompt)
                }
            else:
                if progress_callback:
                    progress_callback("Thumbnail generation failed")
                return None
                
        except Exception as e:
            print(f"Error in legacy thumbnail generation: {e}")
            if progress_callback:
                progress_callback(f"Error: {str(e)}")
            return None
    
    def generate_thumbnail_api(self, prompt: str, preset: str = "quick_draft", 
                             topic_title: Optional[str] = None,
                             progress_callback: Optional[Callable] = None) -> Optional[Dict[str, Any]]:
        """Generate thumbnail using API - legacy interface."""
        # Redirect to the new AI generator
        return self.generate_thumbnail_local(prompt, preset, topic_title, progress_callback)
    
    def generate_thumbnails(self, prompt: str, count: int = 2, style: str = "modern") -> Dict[str, Any]:
        """Generate multiple thumbnails - simplified interface."""
        params = {
            'prompt': prompt,
            'style': style,
            'quality': 'high',
            'count': count
        }
        
        return self.ai_generator.generate_thumbnails(params)
    
    def get_thumbnail_suggestions(self, topic: str, content_type: str = "educational") -> list:
        """Get thumbnail prompt suggestions."""
        return self.ai_generator.get_thumbnail_suggestions(topic, content_type)


def main():
    """Test the legacy thumbnail generator wrapper."""
    generator = ThumbnailGenerator()
    
    # Test legacy interface
    result = generator.generate_thumbnail_local(
        prompt="Machine Learning Tutorial",
        preset="high_quality",
        progress_callback=lambda msg: print(f"Progress: {msg}")
    )
    
    if result:
        print("Legacy Interface Test - Success!")
        print(f"Method: {result['method']}")
        print(f"Path: {result.get('image_path', 'N/A')}")
        print(f"URL: {result.get('image_url', 'N/A')}")
    else:
        print("Legacy Interface Test - Failed!")
    
    # Test new interface
    result2 = generator.generate_thumbnails("AI Technology", count=2, style="tech")
    print(f"\nNew Interface Test: {len(result2['thumbnails'])} thumbnails generated")
    print(f"Methods: {result2['generation_methods']}")


if __name__ == "__main__":
    main()