#!/usr/bin/env python3
"""
Production Manim Stage - Background Animation Generation
Generate background animations using Manim based on Gemini-generated scripts
NO FALLBACK MECHANISMS - Production mode only
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import tempfile

# Add core directory to path
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_gemini_integration import EnhancedGeminiIntegration

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionManimStage:
    """Production Manim Stage - Background animation generation."""
    
    def __init__(self, project_root: str = None):
        """Initialize Production Manim stage.
        
        Args:
            project_root: Path to project root directory
        """
        if project_root is None:
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = Path(project_root)
            
        # Output directory
        self.output_dir = self.project_root / "NEW" / "processed" / "manim"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Conda environment for Manim (using video-audio-processing environment)
        self.conda_python = Path("/Users/aryanjain/miniforge3/envs/video-audio-processing/bin/python")
        
        # Initialize Gemini integration for script generation
        self.gemini_integration = EnhancedGeminiIntegration(str(self.project_root))
        
        # Manim configuration
        self.default_params = {
            'resolution': '1920x1080',
            'frame_rate': 30,
            'quality': 'high',
            'background_color': '#0f0f0f',
            'duration': 60  # seconds
        }
        
        # Verify environment
        self.available = self._verify_environment()
        
        if not self.available:
            logger.warning("[WARNING] Manim production environment not fully available - will use script generation only")
        
        logger.info(f"Production Manim Stage initialized")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Conda environment: {self.conda_python}")
        logger.info(f"  Environment available: {self.available}")
    
    def _verify_environment(self) -> bool:
        """Verify Manim conda environment is available."""
        try:
            # Check if conda python exists
            if not self.conda_python.exists():
                logger.warning(f"Conda python not found: {self.conda_python}")
                return False
            
            # Test basic dependencies
            test_script = '''
import sys
try:
    import manim
    print("SUCCESS: Manim dependencies available")
    sys.exit(0)
except ImportError as e:
    print(f"WARNING: Manim not available: {e}")
    sys.exit(1)
'''
            
            result = subprocess.run(
                [str(self.conda_python), '-c', test_script],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                logger.info("[SUCCESS] Manim production environment verified")
                return True
            else:
                logger.warning(f"[WARNING] Manim environment test failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.warning(f"[WARNING] Environment verification failed: {e}")
            return False
    
    def process_background_animation(self, text: str, num_chunks: int, 
                                   animation_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process background animation generation.
        
        Args:
            text: Original text for animation context
            num_chunks: Number of chunks for timing
            animation_params: Animation parameters (optional)
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        
        result = {
            'stage': 'production_manim',
            'timestamp': time.time(),
            'success': False,
            'text': text,
            'num_chunks': num_chunks,
            'animation_params': animation_params or {},
            'output_video_path': None,
            'manim_script_path': None,
            'processing_time': 0,
            'errors': []
        }
        
        try:
            # Generate output filename
            timestamp = int(time.time())
            output_filename = f"background_animation_{timestamp}.mp4"
            output_path = self.output_dir / output_filename
            script_filename = f"manim_script_{timestamp}.py"
            script_path = self.output_dir / script_filename
            
            result['output_video_path'] = str(output_path)
            result['manim_script_path'] = str(script_path)
            
            # Prepare animation parameters
            params = self.default_params.copy()
            if animation_params:
                params.update(animation_params)
            
            # Step 1: Generate Manim script using Gemini
            logger.info("Frontend Generating Manim script using Gemini...")
            script_result = self._generate_manim_script(text, params)
            
            if not script_result['success']:
                result['errors'].extend(script_result['errors'])
                return result
            
            manim_script = script_result['script']
            
            # Step 2: Save the generated script
            logger.info(f"Storage Saving Manim script to {script_path}")
            with open(script_path, 'w') as f:
                f.write(manim_script)
            
            # Step 3: Generate video using Manim (if environment available)
            if self.available:
                logger.info("VIDEO PIPELINE Rendering animation with Manim...")
                video_result = self._render_manim_video(script_path, output_path, params)
                
                if video_result['success']:
                    result.update(video_result)
                    result['success'] = True
                    logger.info("[SUCCESS] Background animation rendered successfully!")
                else:
                    result['errors'].extend(video_result['errors'])
                    logger.warning("[WARNING] Animation rendering failed, but script generated successfully")
            else:
                # NO FALLBACKS - If Manim not available, the entire stage fails
                logger.error("Manim environment not available - cannot create background animation")
                result['errors'].append("Manim environment not available - production pipeline requires working Manim")
                result['success'] = False
                    
        except Exception as e:
            result['errors'].append(f"Production Manim processing failed: {str(e)}")
            logger.error(f"[ERROR] Production Manim Stage error: {e}")
            
        finally:
            result['processing_time'] = time.time() - start_time
            
            # Save stage results
            results_file = self.output_dir / f"production_manim_results_{int(time.time())}.json"
            with open(results_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"File: Production Manim results saved: {results_file}")
            logger.info(f"Duration:  Processing time: {result['processing_time']:.2f} seconds")
        
        return result
    
    def _generate_manim_script(self, text: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Manim script using Gemini integration."""
        try:
            # Prepare user inputs for Gemini
            user_inputs = {
                'title': 'Background Animation',
                'topic': text[:100] + "..." if len(text) > 100 else text,
                'content_type': 'Educational Animation',
                'tone': 'professional',     # USER CONFIGURABLE - can be overridden in animation_params
                'emotion': 'confident',     # USER CONFIGURABLE - can be overridden in animation_params
                'additional_context': f"Create background animation for {params.get('duration', 60)} seconds"
            }
            
            # Override with animation_params if provided
            if 'tone' in params:
                user_inputs['tone'] = params['tone']
            if 'emotion' in params:
                user_inputs['emotion'] = params['emotion']
            
            # Generate Manim script
            script_result = self.gemini_integration._generate_manim_script(user_inputs)
            
            if script_result['success']:
                return {
                    'success': True,
                    'script': script_result['script'],
                    'metadata': script_result['metadata'],
                    'errors': []
                }
            else:
                # NO FALLBACKS - If Gemini fails, the entire stage fails
                logger.error(f"Gemini script generation failed: {script_result['error']}")
                return {
                    'success': False,
                    'script': None,
                    'metadata': {},
                    'errors': [f"Gemini script generation failed: {script_result['error']}"]
                }
                
        except Exception as e:
            logger.error(f"Failed to generate Manim script: {str(e)}")
            # NO FALLBACKS - If any exception occurs, the entire stage fails
            return {
                'success': False,
                'script': None,
                'metadata': {},
                'errors': [f"Failed to generate Manim script: {str(e)}"]
            }
    
    
    def _render_manim_video(self, script_path: str, output_path: str, 
                          params: Dict[str, Any]) -> Dict[str, Any]:
        """Render video using Manim."""
        try:
            # Manim command
            resolution = params.get('resolution', '1920x1080')
            frame_rate = params.get('frame_rate', 30)
            quality = params.get('quality', 'high')
            
            # Quality mapping
            quality_flags = {
                'draft': ['-ql'],
                'standard': ['-qm'],
                'high': ['-qh'],
                'maximum': ['-qk']
            }
            
            # Convert resolution format (1920x1080 -> 1920,1080)
            resolution_formatted = resolution.replace('x', ',')
            
            cmd = [
                str(self.conda_python), '-m', 'manim',
                str(script_path),
                'BackgroundAnimation',
                '-r', resolution_formatted,
                '--frame_rate', str(frame_rate),
                '--output_file', str(output_path)
            ]
            
            # Add quality flags
            cmd.extend(quality_flags.get(quality, ['-qh']))
            
            logger.info(f"VIDEO PIPELINE Running Manim command: {' '.join(cmd)}")
            
            # Execute Manim
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            
            if result.returncode == 0:
                if Path(output_path).exists():
                    file_size = Path(output_path).stat().st_size
                    
                    return {
                        'success': True,
                        'file_size': file_size,
                        'video_duration': params.get('duration', 60),
                        'video_resolution': resolution.split('x'),
                        'frame_rate': frame_rate,
                        'errors': []
                    }
                else:
                    return {'success': False, 'errors': ['Output file not created']}
            else:
                logger.error(f"Manim rendering failed: {result.stderr}")
                return {'success': False, 'errors': [f'Manim failed: {result.stderr}']}
                
        except subprocess.TimeoutExpired:
            return {'success': False, 'errors': ['Manim rendering timed out']}
        except Exception as e:
            return {'success': False, 'errors': [f'Manim rendering error: {str(e)}']}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the production model."""
        return {
            "model_name": "Production Manim (Background Animation)",
            "conda_environment": str(self.conda_python),
            "gemini_integration": "enhanced_gemini_integration",
            "default_params": self.default_params,
            "available": self.available,
            "supports_script_generation": True,
            "supports_video_rendering": self.available,
            "supports_placeholder_creation": False,
            "fallback_enabled": False,
            "architecture": "production_only_no_fallbacks"
        }


def main():
    """Test Production Manim stage."""
    print("Target: Production Manim Stage Test")
    print("=" * 50)
    
    # Initialize stage
    stage = ProductionManimStage()
    
    # Test with sample text
    test_text = "Welcome to Machine Learning! Today we'll explore the basics of artificial intelligence and how computers learn from data."
    
    # Sample animation parameters
    animation_params = {
        'resolution': '1920x1080',
        'frame_rate': 30,
        'quality': 'high',
        'duration': 30  # 30 seconds for testing
    }
    
    # Run processing
    results = stage.process_background_animation(test_text, 3, animation_params)
    
    if results['success']:
        print("\\nSUCCESS Production Manim Stage test PASSED!")
        print(f"[SUCCESS] Output video: {results['output_video_path']}")
        print(f"[SUCCESS] Script file: {results['manim_script_path']}")
        print(f"[SUCCESS] File size: {results.get('file_size', 0):,} bytes")
        print(f"[SUCCESS] Duration: {results.get('video_duration', 0)} seconds")
        print(f"[SUCCESS] Resolution: {results.get('video_resolution', [0, 0])}")
        print(f"Duration:  Processing time: {results['processing_time']:.2f} seconds")
        
        if results.get('placeholder'):
            print("Step Note: Placeholder video created (Manim not available)")
    else:
        print("\\n[ERROR] Production Manim Stage test FAILED!")
        print("Tools Errors:")
        for error in results['errors']:
            print(f"   - {error}")
    
    print(f"\\nStatus: Model info: {stage.get_model_info()}")


if __name__ == "__main__":
    main()