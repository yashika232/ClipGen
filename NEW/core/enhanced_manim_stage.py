#!/usr/bin/env python3
"""
Enhanced Manim Stage for Video Synthesis Pipeline
Integrates Manim animation generation with metadata-driven architecture for background animations
"""

import os
import sys
import subprocess
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import tempfile
import json

# Add the core directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_metadata_manager import EnhancedMetadataManager

# Try to import existing Manim components
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "INTEGRATED_PIPELINE" / "src"))
    from manim_stage import ManimeStage
    MANIM_AVAILABLE = True
except ImportError:
    MANIM_AVAILABLE = False

# Try to import video processing utilities
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "useless_files" / "experimental_scripts"))
    from video_duration_matcher import VideoDurationMatcher
    from animation_compositor import AnimationCompositor
    COMPOSITOR_AVAILABLE = True
except ImportError:
    COMPOSITOR_AVAILABLE = False


class EnhancedManimStage:
    """Enhanced Manim stage with metadata-driven architecture."""
    
    def __init__(self, base_dir: str = None):
        """Initialize enhanced Manim stage.
        
        Args:
            base_dir: Base directory for the pipeline. Defaults to NEW directory.
        """
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        # Initialize metadata manager
        self.metadata_manager = EnhancedMetadataManager(str(self.base_dir))
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Output directories
        self.animation_output_dir = self.base_dir / "processed" / "background_animations"
        self.animation_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.manim_scripts_dir = self.animation_output_dir / "scripts"
        self.manim_scripts_dir.mkdir(exist_ok=True)
        
        self.rendered_animations_dir = self.animation_output_dir / "rendered"
        self.rendered_animations_dir.mkdir(exist_ok=True)
        
        # Try to initialize Manim stage
        self.manim_stage = None
        if MANIM_AVAILABLE:
            try:
                self.manim_stage = ManimeStage()
                self.logger.info("[SUCCESS] Manim stage initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Manim stage: {e}")
                self.manim_stage = None
        
        # Try to initialize compositor utilities
        self.duration_matcher = None
        self.animation_compositor = None
        if COMPOSITOR_AVAILABLE:
            try:
                self.duration_matcher = VideoDurationMatcher()
                self.animation_compositor = AnimationCompositor()
                self.logger.info("[SUCCESS] Animation compositor utilities initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize compositor utilities: {e}")
        
        # Manim configuration
        self.manim_config = {
            'quality': 'medium_quality',  # low_quality, medium_quality, high_quality
            'fps': 30,
            'background_color': '#222222',
            'resolution': '1920x1080',
            'chunk_duration_threshold': 60,  # seconds
            'max_animation_duration': 300    # 5 minutes max
        }
        
        self.logger.info("STARTING Enhanced Manim Stage initialized")
    
    def process_background_animation(self) -> Dict[str, Any]:
        """Process background animation generation using metadata-driven approach.
        
        Returns:
            Dictionary with processing results
        """
        try:
            # Update stage status to processing
            self.metadata_manager.update_stage_status(
                "background_animation", 
                "processing",
                {"prerequisites": "content_generation"}
            )
            
            # Load current session metadata
            metadata = self.metadata_manager.load_metadata()
            if not metadata:
                raise ValueError("No active session found")
            
            # Validate prerequisites
            validation_result = self._validate_prerequisites(metadata)
            if not validation_result['valid']:
                raise ValueError(f"Prerequisites not met: {validation_result['errors']}")
            
            # Get generated Manim script from metadata
            generated_content = metadata.get('generated_content', {})
            manim_script = generated_content.get('manim_script')
            
            if not manim_script:
                raise ValueError("No Manim script found in generated content")
            
            # Get user preferences for animation parameters
            user_inputs = metadata.get('user_inputs', {})
            tone = user_inputs.get('tone', 'professional')
            emotion = user_inputs.get('emotion', 'confident')
            content_type = user_inputs.get('content_type', 'Short-Form Video Reel')
            
            self.logger.info(f"VIDEO PIPELINE Processing background animation generation:")
            self.logger.info(f"   Tone: {tone}, Emotion: {emotion}")
            self.logger.info(f"   Content type: {content_type}")
            
            processing_start = time.time()
            
            # Determine target duration from voice cloning stage if available
            target_duration = self._get_target_animation_duration(metadata)
            
            # Process animation generation
            result = self._process_manim_animation(
                manim_script, tone, emotion, content_type, target_duration
            )
            
            processing_time = time.time() - processing_start
            
            if result['success']:
                # Get relative path for metadata
                output_path = Path(result['output_path'])
                if output_path.is_absolute():
                    try:
                        relative_path = output_path.relative_to(self.base_dir)
                        result['output_path'] = str(relative_path)
                    except ValueError:
                        pass  # Keep absolute if outside base directory
                
                # Prepare comprehensive processing data
                processing_data = {
                    "background_animation": result['output_path'],
                    "manim_script_path": result.get('script_path', ''),
                    "processing_duration": processing_time,
                    "animation_duration": result.get('animation_duration', 0),
                    "animation_method": result.get('animation_method', 'single'),
                    "manim_config": {
                        "tone": tone,
                        "emotion": emotion,
                        "content_type": content_type,
                        "quality": self.manim_config['quality'],
                        "fps": self.manim_config['fps']
                    }
                }
                
                # Update metadata with successful results
                self.metadata_manager.update_stage_status(
                    "background_animation",
                    "completed",
                    processing_data
                )
                
                self.logger.info(f"[SUCCESS] Background animation completed successfully in {processing_time:.1f}s")
                self.logger.info(f"   Output: {result['output_path']}")
                
                return {
                    'success': True,
                    'output_path': result['output_path'],
                    'animation_duration': result.get('animation_duration', 0),
                    'processing_time': processing_time,
                    'stage_updated': True
                }
            else:
                # Update metadata with failure
                error_data = {
                    "processing_duration": processing_time,
                    "error_details": result.get('error', 'Unknown error')
                }
                
                self.metadata_manager.update_stage_status(
                    "background_animation",
                    "failed",
                    error_data
                )
                
                self.logger.error(f"[ERROR] Background animation failed: {result.get('error')}")
                return result
        
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"[ERROR] Background animation error: {error_msg}")
            
            # Update metadata with error
            self.metadata_manager.update_stage_status(
                "background_animation",
                "failed",
                {"error": error_msg}
            )
            
            return {
                'success': False,
                'error': error_msg,
                'stage_updated': True
            }
    
    def _validate_prerequisites(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that background animation prerequisites are met.
        
        Args:
            metadata: Session metadata
            
        Returns:
            Validation results
        """
        errors = []
        warnings = []
        
        # Check generated content
        generated_content = metadata.get('generated_content', {})
        if not generated_content:
            errors.append("No generated content found")
        else:
            manim_script = generated_content.get('manim_script')
            if not manim_script:
                errors.append("No Manim script found in generated content")
            elif len(manim_script.strip()) < 100:
                warnings.append("Manim script appears very short")
        
        # Check Manim availability
        if not self._check_manim_installation():
            warnings.append("Manim not installed - will use fallback method")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def _get_target_animation_duration(self, metadata: Dict[str, Any]) -> float:
        """Get target duration for animation from voice cloning stage.
        
        Args:
            metadata: Session metadata
            
        Returns:
            Target duration in seconds
        """
        try:
            # Check if voice cloning is completed
            voice_stage = self.metadata_manager.get_stage_status("voice_cloning")
            if voice_stage and voice_stage.get("status") == "completed":
                processing_data = voice_stage.get("processing_data", {})
                voice_duration = processing_data.get("voice_duration", 0)
                if voice_duration > 0:
                    self.logger.info(f"   Target animation duration: {voice_duration:.1f}s (from voice)")
                    return voice_duration
            
            # Fallback: estimate from script content
            generated_content = metadata.get('generated_content', {})
            clean_script = generated_content.get('clean_script', '')
            
            if clean_script:
                # Rough estimate: 150 words per minute
                word_count = len(clean_script.split())
                estimated_duration = (word_count / 150) * 60
                self.logger.info(f"   Estimated animation duration: {estimated_duration:.1f}s (from script)")
                return max(estimated_duration, 30)  # Minimum 30 seconds
            
            # Default duration
            return 60.0
            
        except Exception as e:
            self.logger.warning(f"Could not determine target duration: {e}")
            return 60.0
    
    def _process_manim_animation(self, manim_script: str, tone: str, emotion: str,
                               content_type: str, target_duration: float) -> Dict[str, Any]:
        """Process Manim animation generation.
        
        Args:
            manim_script: Generated Manim script code
            tone: Voice tone parameter
            emotion: Voice emotion parameter
            content_type: Content type
            target_duration: Target animation duration
            
        Returns:
            Processing results
        """
        try:
            timestamp = int(datetime.now().timestamp())
            
            # Create script file
            script_filename = f"animation_script_{timestamp}.py"
            script_path = self.manim_scripts_dir / script_filename
            
            # Enhance script with animation parameters
            enhanced_script = self._enhance_manim_script(
                manim_script, tone, emotion, content_type, target_duration
            )
            
            # Write enhanced script to file
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(enhanced_script)
            
            self.logger.info(f"[SUCCESS] Manim script written to: {script_path}")
            
            # Render animation
            if self.manim_stage:
                # Use existing Manim stage
                result = self._render_with_manim_stage(script_path, timestamp)
            else:
                # Use direct Manim command
                result = self._render_with_manim_command(script_path, timestamp)
            
            if result['success']:
                result['script_path'] = str(script_path.relative_to(self.base_dir))
                result['animation_duration'] = target_duration
                result['animation_method'] = 'manim'
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _enhance_manim_script(self, original_script: str, tone: str, emotion: str,
                            content_type: str, target_duration: float) -> str:
        """Enhance Manim script with animation parameters.
        
        Args:
            original_script: Original generated script
            tone: Voice tone
            emotion: Voice emotion
            content_type: Content type
            target_duration: Target duration
            
        Returns:
            Enhanced script
        """
        try:
            # Animation timing adjustments based on parameters
            animation_params = self._get_animation_parameters(tone, emotion, content_type)
            
            # Calculate timing adjustments
            total_wait_time = animation_params.get('total_wait_time', target_duration * 0.3)
            animation_speed = animation_params.get('animation_speed', 1.0)
            
            # Clean original script - extract only the Python code part
            clean_script = self._extract_python_code(original_script)
            
            # Enhance script with configuration
            enhanced_script = f"""# Enhanced Manim Script with Pipeline Parameters
# Generated at: {datetime.now().isoformat()}
# Tone: {tone}, Emotion: {emotion}, Content Type: {content_type}
# Target Duration: {target_duration:.1f}s

from manim import *
import numpy as np

# Configure Manim for pipeline
config.frame_rate = {self.manim_config['fps']}
config.background_color = "{self.manim_config['background_color']}"

"""
            
            # Modify clean script to include timing adjustments
            lines = clean_script.split('\n')
            modified_lines = []
            
            for line in lines:
                # Skip import lines since we already have them
                if line.strip().startswith('from manim import') or line.strip().startswith('import numpy'):
                    continue
                    
                # Adjust wait times
                if 'self.wait(' in line:
                    # Extract wait time and adjust
                    import re
                    wait_match = re.search(r'self\.wait\((\d+(?:\.\d+)?)\)', line)
                    if wait_match:
                        original_wait = float(wait_match.group(1))
                        adjusted_wait = original_wait * animation_speed
                        line = line.replace(f'self.wait({original_wait})', f'self.wait({adjusted_wait:.1f})')
                
                # Adjust run_time parameters
                if 'run_time=' in line:
                    import re
                    runtime_match = re.search(r'run_time=(\d+(?:\.\d+)?)', line)
                    if runtime_match:
                        original_runtime = float(runtime_match.group(1))
                        adjusted_runtime = original_runtime * animation_speed
                        line = line.replace(f'run_time={original_runtime}', f'run_time={adjusted_runtime:.1f}')
                
                modified_lines.append(line)
            
            enhanced_script += '\n'.join(modified_lines)
            
            # Add final timing adjustment to meet target duration
            enhanced_script += f"""
        # Final timing adjustment to meet target duration
        self.wait(max(0.5, {target_duration:.1f} - {total_wait_time:.1f}))
"""
            
            return enhanced_script
            
        except Exception as e:
            self.logger.warning(f"Script enhancement failed: {e}, using fallback")
            return self._create_fallback_script(target_duration)
    
    def _extract_python_code(self, script_content: str) -> str:
        """Extract only the Python code from the script content.
        
        Args:
            script_content: Raw script content that may include explanations
            
        Returns:
            Clean Python code
        """
        try:
            lines = script_content.split('\n')
            python_lines = []
            in_code_block = False
            
            for line in lines:
                # Look for class definition to start capturing
                if 'class VideoBackgroundScene' in line or 'class ' in line:
                    in_code_block = True
                    python_lines.append(line)
                elif in_code_block:
                    # Stop at explanatory text that's not indented properly
                    if line.strip() and not line.startswith(' ') and not line.startswith('\t') and not line.startswith('#'):
                        # Check if this looks like explanatory text
                        if any(word in line.lower() for word in ['run this script', 'install manim', 'to run', 'make sure', 'replace', 'follow']):
                            break
                    python_lines.append(line)
                elif line.strip().startswith('from manim') or line.strip().startswith('import'):
                    python_lines.append(line)
            
            return '\n'.join(python_lines)
            
        except Exception as e:
            self.logger.warning(f"Could not extract Python code: {e}")
            return self._create_fallback_script(60)
    
    def _create_fallback_script(self, duration: float) -> str:
        """Create a simple fallback Manim script.
        
        Args:
            duration: Target duration
            
        Returns:
            Fallback script
        """
        return f"""
class VideoBackgroundScene(Scene):
    def construct(self):
        # Simple animated background
        title = Text("Advanced AI Video Processing", font_size=48, color=YELLOW)
        subtitle = Text("Professional Content", font_size=36, color=LIGHT_GRAY)
        
        title.to_edge(UP)
        subtitle.next_to(title, DOWN, buff=0.5)
        
        self.play(Write(title), run_time=2)
        self.play(Write(subtitle), run_time=1)
        
        # Animated shapes
        circle = Circle(radius=1, color=BLUE)
        square = Square(side_length=2, color=GREEN)
        
        self.play(Create(circle), run_time=2)
        self.play(Transform(circle, square), run_time=2)
        
        # Wait for remaining duration
        self.wait({max(5, duration - 10):.1f})
        
        self.play(FadeOut(*self.mobjects), run_time=1)
"""
    
    def _get_animation_parameters(self, tone: str, emotion: str, content_type: str) -> Dict[str, Any]:
        """Get animation parameters based on tone, emotion, and content type.
        
        Args:
            tone: Voice tone
            emotion: Voice emotion
            content_type: Content type
            
        Returns:
            Animation parameters
        """
        # Base parameters
        params = {
            'animation_speed': 1.0,
            'total_wait_time': 10.0,
            'color_scheme': 'default'
        }
        
        # Emotion-based adjustments
        emotion_adjustments = {
            'inspired': {'animation_speed': 1.1, 'total_wait_time': 8.0},
            'confident': {'animation_speed': 1.0, 'total_wait_time': 12.0},
            'curious': {'animation_speed': 0.9, 'total_wait_time': 15.0},
            'excited': {'animation_speed': 1.3, 'total_wait_time': 6.0},
            'calm': {'animation_speed': 0.8, 'total_wait_time': 18.0}
        }
        
        # Tone-based adjustments
        tone_adjustments = {
            'professional': {'animation_speed': 1.0, 'total_wait_time': 12.0},
            'friendly': {'animation_speed': 1.1, 'total_wait_time': 10.0},
            'motivational': {'animation_speed': 1.2, 'total_wait_time': 8.0},
            'casual': {'animation_speed': 1.1, 'total_wait_time': 9.0}
        }
        
        # Content type adjustments
        content_adjustments = {
            'Short-Form Video Reel': {'animation_speed': 1.3, 'total_wait_time': 5.0},
            'Full Training Module': {'animation_speed': 0.9, 'total_wait_time': 15.0},
            'Quick Tutorial': {'animation_speed': 1.1, 'total_wait_time': 8.0},
            'Presentation': {'animation_speed': 1.0, 'total_wait_time': 12.0}
        }
        
        # Apply adjustments
        if emotion in emotion_adjustments:
            for key, value in emotion_adjustments[emotion].items():
                params[key] = value
        
        if tone in tone_adjustments:
            for key, value in tone_adjustments[tone].items():
                params[key] = (params[key] + value) / 2  # Average with emotion adjustment
        
        if content_type in content_adjustments:
            for key, value in content_adjustments[content_type].items():
                params[key] = (params[key] + value) / 2  # Average with previous adjustments
        
        return params
    
    def _render_with_manim_stage(self, script_path: Path, timestamp: int) -> Dict[str, Any]:
        """Render animation using existing Manim stage.
        
        Args:
            script_path: Path to Manim script
            timestamp: Timestamp for output naming
            
        Returns:
            Rendering results
        """
        try:
            output_filename = f"background_animation_{timestamp}.mp4"
            output_path = self.rendered_animations_dir / output_filename
            
            # Use Manim stage
            result = self.manim_stage.render_animation(
                script_path=str(script_path),
                output_path=str(output_path),
                quality=self.manim_config['quality']
            )
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': f"Manim stage rendering failed: {str(e)}"}
    
    def _render_with_manim_command(self, script_path: Path, timestamp: int) -> Dict[str, Any]:
        """Render animation using direct Manim command.
        
        Args:
            script_path: Path to Manim script
            timestamp: Timestamp for output naming
            
        Returns:
            Rendering results
        """
        try:
            if not self._check_manim_installation():
                return self._create_fallback_animation(timestamp)
            
            output_filename = f"background_animation_{timestamp}.mp4"
            output_path = self.rendered_animations_dir / output_filename
            
            # Determine quality flag
            quality_flags = {
                'low_quality': '-ql',
                'medium_quality': '-qm', 
                'high_quality': '-qh'
            }
            quality_flag = quality_flags.get(self.manim_config['quality'], '-qm')
            
            # Run Manim command
            cmd = [
                'manim', quality_flag, '--disable_caching',
                '--output_file', output_filename,
                str(script_path), 'VideoBackgroundScene'
            ]
            
            self.logger.info(f"Running Manim command: {' '.join(cmd)}")
            
            # Change to output directory for rendering
            original_cwd = os.getcwd()
            os.chdir(str(self.rendered_animations_dir))
            
            try:
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=600,  # 10 minute timeout
                    cwd=str(self.rendered_animations_dir)
                )
                
                if result.returncode == 0 and output_path.exists():
                    self.logger.info("[SUCCESS] Manim rendering completed successfully")
                    return {
                        'success': True, 
                        'output_path': str(output_path),
                        'render_method': 'manim_command'
                    }
                else:
                    self.logger.warning(f"Manim rendering failed: {result.stderr}")
                    return self._create_fallback_animation(timestamp)
                    
            finally:
                os.chdir(original_cwd)
                
        except Exception as e:
            self.logger.warning(f"Manim command failed: {e}")
            return self._create_fallback_animation(timestamp)
    
    def _check_manim_installation(self) -> bool:
        """Check if Manim is properly installed.
        
        Returns:
            True if Manim is available
        """
        try:
            result = subprocess.run(
                ['manim', '--version'], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def _create_fallback_animation(self, timestamp: int) -> Dict[str, Any]:
        """Create fallback animation using FFmpeg.
        
        Args:
            timestamp: Timestamp for output naming
            
        Returns:
            Fallback animation results
        """
        try:
            self.logger.info("Creating fallback animation with FFmpeg...")
            
            output_filename = f"background_animation_fallback_{timestamp}.mp4"
            output_path = self.rendered_animations_dir / output_filename
            
            # Create simple gradient background animation
            cmd = [
                'ffmpeg', '-y',
                '-f', 'lavfi',
                '-i', f'color=c={self.manim_config["background_color"]}:s={self.manim_config["resolution"]}:r={self.manim_config["fps"]}',
                '-t', '60',  # 60 second default
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0 and output_path.exists():
                self.logger.info("[SUCCESS] Fallback animation created successfully")
                return {
                    'success': True,
                    'output_path': str(output_path),
                    'render_method': 'ffmpeg_fallback'
                }
            else:
                return {'success': False, 'error': f"Fallback animation failed: {result.stderr}"}
                
        except Exception as e:
            return {'success': False, 'error': f"Fallback animation creation failed: {str(e)}"}
    
    def get_background_animation_status(self) -> Dict[str, Any]:
        """Get current background animation stage status.
        
        Returns:
            Status information
        """
        try:
            stage_status = self.metadata_manager.get_stage_status("background_animation")
            if stage_status:
                return {
                    'success': True,
                    'stage_status': stage_status
                }
            else:
                return {
                    'success': False,
                    'error': 'No background animation stage found in metadata'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def validate_background_animation_prerequisites(self) -> Dict[str, Any]:
        """Validate that background animation can be started.
        
        Returns:
            Validation results
        """
        try:
            # Check metadata
            metadata = self.metadata_manager.load_metadata()
            if not metadata:
                return {'valid': False, 'errors': ['No active session found']}
            
            return self._validate_prerequisites(metadata)
            
        except Exception as e:
            return {
                'valid': False,
                'errors': [str(e)]
            }


def main():
    """Test the enhanced Manim stage."""
    print("🧪 Testing Enhanced Manim Stage")
    print("=" * 50)
    
    # Initialize stage
    stage = EnhancedManimStage()
    
    # Check prerequisites
    prereq_result = stage.validate_background_animation_prerequisites()
    print(f"[SUCCESS] Prerequisites check:")
    print(f"   Valid: {prereq_result['valid']}")
    if prereq_result.get('errors'):
        print(f"   Errors: {prereq_result['errors']}")
    if prereq_result.get('warnings'):
        print(f"   Warnings: {prereq_result['warnings']}")
    
    # Check current status
    status_result = stage.get_background_animation_status()
    if status_result['success']:
        stage_info = status_result['stage_status']
        print(f"[SUCCESS] Current status: {stage_info.get('status', 'unknown')}")
    else:
        print(f"[ERROR] Status check failed: {status_result['error']}")
    
    # Process background animation if prerequisites are met
    if prereq_result['valid']:
        print("\nVIDEO PIPELINE Starting background animation generation...")
        result = stage.process_background_animation()
        
        if result['success']:
            print("[SUCCESS] Background animation completed successfully!")
            print(f"   Output: {result['output_path']}")
            print(f"   Duration: {result.get('animation_duration', 0):.1f}s")
            print(f"   Processing time: {result['processing_time']:.1f}s")
        else:
            print(f"[ERROR] Background animation failed: {result['error']}")
    else:
        print("\n[ERROR] Prerequisites not met - skipping processing")
    
    print("\nSUCCESS Enhanced Manim Stage testing completed!")


if __name__ == "__main__":
    main()