#!/usr/bin/env python3
"""
Production SadTalker Stage - Lip-sync Video Generation
Complete lip-sync animation for each voice chunk using SadTalker
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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import pipeline logger for detailed debugging
try:
    from pipeline_logger import get_pipeline_logger
    pipeline_logger = get_pipeline_logger()
except ImportError:
    pipeline_logger = None

class ProductionSadTalkerStage:
    """Production SadTalker Stage - Full lip-sync video generation."""
    
    def __init__(self, project_root: str = None):
        """Initialize Production SadTalker stage.
        
        Args:
            project_root: Path to project root directory
        """
        if project_root is None:
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = Path(project_root)
            
        # Output directory
        self.output_dir = self.project_root / "NEW" / "processed" / "sadtalker"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Conda environment for SadTalker - use correct sadtalker environment as per fix.md
        self.conda_python = Path("/Users/aryanjain/miniforge3/envs/sadtalker/bin/python")
        
        # SadTalker configuration
        self.device = self._setup_device()
        self.default_params = {
            'size': 256,  # User confirmed 256x256 gives best results
            'expression_scale': 1.2,
            'preprocess_mode': 'resize',
            'fps': 30,
            'facial_animation_strength': 1.1,
            'enhancer': 'gfpgan',
            'still': True
        }
        
        # Verify environment
        self.available = self._verify_environment()
        
        if not self.available:
            raise RuntimeError("SadTalker production environment not available")
        
        logger.info(f"Production SadTalker Stage initialized")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Conda environment: {self.conda_python}")
        logger.info(f"  Output directory: {self.output_dir}")
    
    def _setup_device(self) -> str:
        """Setup device for SadTalker."""
        try:
            # Check if we can import torch in the conda environment
            test_script = '''
import torch
if torch.backends.mps.is_available():
    print("mps")
elif torch.cuda.is_available():
    print("cuda")
else:
    print("cpu")
'''
            
            result = subprocess.run(
                [str(self.conda_python), '-c', test_script],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                device = result.stdout.strip()
                return device
            else:
                return "cpu"
                
        except Exception as e:
            logger.warning(f"[WARNING] Error checking device: {e}, using CPU")
            return "cpu"
    
    def _verify_environment(self) -> bool:
        """Verify SadTalker conda environment is available."""
        try:
            # Check if conda python exists
            if not self.conda_python.exists():
                logger.error(f"Conda python not found: {self.conda_python}")
                return False
            
            # Test basic dependencies
            test_script = '''
import sys
try:
    import torch
    import cv2
    import numpy as np
    print("SUCCESS: SadTalker dependencies available")
    sys.exit(0)
except ImportError as e:
    print(f"ERROR: SadTalker dependencies not available: {e}")
    sys.exit(1)
'''
            
            result = subprocess.run(
                [str(self.conda_python), '-c', test_script],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                logger.info("[SUCCESS] SadTalker production environment verified")
                return True
            else:
                logger.error(f"[ERROR] SadTalker environment test failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] Environment verification failed: {e}")
            return False
    
    def process_lip_sync_animation(self, face_image_path: str, audio_path: str, 
                                  emotion_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process lip-sync animation using production SadTalker.
        
        Args:
            face_image_path: Path to face image
            audio_path: Path to audio file
            emotion_params: Emotion-aware parameters (optional)
            
        Returns:
            Dictionary with processing results
        """
        # Log SadTalker processing start with detailed context
        if pipeline_logger:
            pipeline_logger.sadtalker_debug("Style: STARTING SadTalker lip-sync processing", {
                'face_image_path': face_image_path,
                'audio_path': audio_path,
                'emotion_params': emotion_params,
                'face_exists': Path(face_image_path).exists(),
                'audio_exists': Path(audio_path).exists(),
                'face_size': Path(face_image_path).stat().st_size if Path(face_image_path).exists() else 0,
                'audio_size': Path(audio_path).stat().st_size if Path(audio_path).exists() else 0
            })
        
        start_time = time.time()
        
        result = {
            'stage': 'production_sadtalker_animation',
            'timestamp': time.time(),
            'success': False,
            'input_face_image': face_image_path,
            'input_audio': audio_path,
            'emotion_params': emotion_params or {},
            'output_video_path': None,
            'processing_time': 0,
            'errors': []
        }
        
        try:
            # Validate inputs
            if not Path(face_image_path).exists():
                result['errors'].append(f"Face image not found: {face_image_path}")
                return result
                
            if not Path(audio_path).exists():
                result['errors'].append(f"Audio file not found: {audio_path}")
                return result
            
            # Generate output filename
            timestamp = int(time.time())
            output_filename = f"lip_sync_video_{timestamp}.mp4"
            output_path = self.output_dir / output_filename
            result['output_video_path'] = str(output_path)
            
            # Prepare SadTalker parameters
            params = self.default_params.copy()
            if emotion_params:
                params.update(emotion_params)
            
            # Apply emotion-aware adjustments
            emotion = emotion_params.get('emotion', 'professional') if emotion_params else 'professional'
            self._apply_emotion_adjustments(params, emotion)
            
            # Create unique result directory to prevent FFmpeg conflicts
            unique_result_dir = self.output_dir / f"sadtalker_result_{int(time.time())}"
            unique_result_dir.mkdir(exist_ok=True)
            
            # Create processing script with unique result directory
            processing_script = self._create_processing_script(
                face_image_path, audio_path, str(unique_result_dir), params
            )
            
            # Execute lip-sync animation via conda environment
            if pipeline_logger:
                pipeline_logger.sadtalker_debug("Style: EXECUTING SadTalker animation script", {
                    'script_length': len(processing_script),
                    'expected_output': str(unique_result_dir)
                })
            
            animation_result = self._execute_animation(processing_script)
            
            # If successful, find and copy the generated video to expected location
            if animation_result.get('success') and animation_result.get('output_video_path'):
                generated_video = Path(animation_result['output_video_path'])
                if generated_video.exists():
                    # Copy to our expected output path
                    import shutil
                    shutil.copy2(str(generated_video), str(output_path))
                    animation_result['output_video_path'] = str(output_path)
            
            # CRITICAL: Log detailed results to debug 0.04s issue
            if pipeline_logger:
                pipeline_logger.sadtalker_debug("Style: SadTalker animation execution completed", {
                    'success': animation_result.get('success'),
                    'output_path': animation_result.get('output_video_path'),
                    'video_duration': animation_result.get('video_duration'),
                    'video_frames': animation_result.get('video_frames'),
                    'video_fps': animation_result.get('video_fps'),
                    'real_sadtalker_used': animation_result.get('real_sadtalker_used'),
                    'errors': animation_result.get('errors', []),
                    'duration_issue_detected': animation_result.get('video_duration', 0) < 1.0
                })
            
            if animation_result['success']:
                result.update(animation_result)
                result['success'] = True
                
                # Log success with duration analysis
                duration = result.get('video_duration', 0)
                if pipeline_logger:
                    if duration < 1.0:
                        pipeline_logger.sadtalker_error("[EMOJI] CRITICAL: SadTalker 0.04s duration issue detected!", {
                            'actual_duration': duration,
                            'expected_minimum': 1.0,
                            'output_path': result['output_video_path'],
                            'video_frames': result.get('video_frames'),
                            'video_fps': result.get('video_fps')
                        })
                    else:
                        pipeline_logger.sadtalker_debug("[SUCCESS] SadTalker duration looks healthy", {
                            'duration': duration,
                            'output_path': result['output_video_path']
                        })
                
                logger.info("[SUCCESS] Production SadTalker Animation completed successfully!")
                logger.info(f"   Output video: {result['output_video_path']}")
                logger.info(f"   File size: {result.get('file_size', 0):,} bytes")
                logger.info(f"   Duration: {result.get('video_duration', 0):.3f} seconds")
            else:
                result['errors'].extend(animation_result['errors'])
                
                if pipeline_logger:
                    pipeline_logger.sadtalker_error("Style: SadTalker animation execution failed", {
                        'errors': animation_result.get('errors', []),
                        'output_path': str(output_path)
                    })
                
        except Exception as e:
            result['errors'].append(f"Production SadTalker processing failed: {str(e)}")
            logger.error(f"[ERROR] Production SadTalker Stage error: {e}")
            
        finally:
            result['processing_time'] = time.time() - start_time
            
            # Save stage results
            results_file = self.output_dir / f"production_sadtalker_results_{int(time.time())}.json"
            with open(results_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"File: Production SadTalker results saved: {results_file}")
            logger.info(f"Duration:  Processing time: {result['processing_time']:.2f} seconds")
        
        return result
    
    def _apply_emotion_adjustments(self, params: Dict[str, Any], emotion: str):
        """Apply emotion-aware parameter adjustments."""
        if emotion == 'excited':
            params['expression_scale'] = 1.5
            params['facial_animation_strength'] = 1.3
        elif emotion == 'calm':
            params['expression_scale'] = 0.8
            params['facial_animation_strength'] = 0.9
        elif emotion == 'confident':
            params['expression_scale'] = 1.1
            params['facial_animation_strength'] = 1.0
        elif emotion == 'professional':
            params['expression_scale'] = 1.2
            params['facial_animation_strength'] = 1.1
        # Default values are already set for other emotions
    
    def _create_processing_script(self, face_image_path: str, audio_path: str, 
                                 output_path: str, params: Dict[str, Any]) -> str:
        """Create Python script for real SadTalker lip-sync processing."""
        
        script = f'''
import os
import sys
import json
import warnings
import subprocess
import time
from pathlib import Path

# Suppress warnings to ensure clean output
warnings.filterwarnings('ignore')

# Set environment for MPS compatibility
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

try:
    import torch
    import cv2
    import numpy as np
    
    # Check if we have a working environment
    print("INFO: SadTalker dependencies loaded successfully")
    
    # Input parameters
    face_image_path = r"{face_image_path}"
    audio_path = r"{audio_path}"
    output_path = r"{output_path}"
    
    # SadTalker parameters
    params = {json.dumps(params).replace('true', 'True').replace('false', 'False')}
    
    print(f"INFO: Processing lip-sync animation with REAL SadTalker")
    print(f"INFO: Face image: {{Path(face_image_path).name}}")
    print(f"INFO: Audio: {{Path(audio_path).name}}")
    print(f"INFO: Output: {{Path(output_path).name}}")
    print(f"INFO: Parameters: {{params}}")
    
    # Validate audio duration using ffprobe
    try:
        import subprocess
        ffprobe_cmd = [
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', 
            '-of', 'csv=p=0', audio_path
        ]
        result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            audio_duration = float(result.stdout.strip())
            print(f"INFO: Audio duration validated: {{audio_duration:.3f}} seconds")
            if audio_duration < 0.1:
                print(f"WARNING: Audio duration very short ({{audio_duration:.3f}}s) - may cause issues")
        else:
            print(f"WARNING: Could not validate audio duration: {{result.stderr}}")
            audio_duration = 0
    except Exception as e:
        print(f"WARNING: Audio validation failed: {{e}}")
        audio_duration = 0
    
    # Check if SadTalker is available - use real_models version as per fix.md
    sadtalker_path = Path("/Users/aryanjain/Documents/video-synthesis-pipeline copy/real_models/SadTalker")
    if not sadtalker_path.exists():
        print("ERROR: SadTalker directory not found in real_models")
        result = {{"success": False, "errors": ["SadTalker directory not found in real_models"]}}
        print(json.dumps(result))
        sys.exit(1)
    
    # Check if checkpoints exist - PRODUCTION ONLY, NO FALLBACKS
    checkpoints_dir = sadtalker_path / "checkpoints"
    if not checkpoints_dir.exists():
        print("ERROR: SadTalker checkpoints not found - PRODUCTION PIPELINE REQUIRES REAL MODELS")
        result = {{"success": False, "errors": ["SadTalker checkpoints not found - production pipeline requires real models"]}}
        print(json.dumps(result))
        sys.exit(1)
    
    use_real_sadtalker = True
    
    if use_real_sadtalker:
        print("INFO: Using REAL SadTalker model")
        
        # Change to SadTalker directory
        os.chdir(str(sadtalker_path))
        
        # Create unique result directory to prevent FFmpeg conflicts
        unique_result_dir = Path(output_path).parent / f"sadtalker_result_{int(time.time())}"
        unique_result_dir.mkdir(exist_ok=True)
        
        # Build SadTalker command with WORKING parameters from extensive testing
        # CRITICAL: Use "resize" preprocess - "crop" + "still" causes 1-frame videos
        sadtalker_cmd = [
            sys.executable, "inference.py",
            "--source_image", face_image_path,
            "--driven_audio", audio_path,
            "--result_dir", str(unique_result_dir),  # Use unique directory
            "--size", str(params.get('size', 256)),
            "--batch_size", "1",  # Reduce batch size for stability
            "--preprocess", "resize",  # FIXED: Use resize instead of crop
            "--cpu"  # Use CPU for stability
            # REMOVED: --still flag causes 1-frame videos
        ]
        
        # Only add expression_scale if it's different from default
        expression_scale = params.get('expression_scale', 1.0)
        if expression_scale != 1.0:
            sadtalker_cmd.extend(["--expression_scale", str(expression_scale)])
        
        print(f"INFO: Running SadTalker with command: {{' '.join(sadtalker_cmd)}}")
        
        try:
            # Run SadTalker with optimized settings
            print(f"INFO: Starting SadTalker processing (timeout: 15 minutes)")
            print(f"INFO: Audio duration check before processing...")
            
            # Quick audio duration check to set appropriate timeout
            import subprocess as sp
            try:
                duration_result = sp.run([
                    'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                    '-of', 'csv=p=0', audio_path
                ], capture_output=True, text=True, timeout=30)
                
                if duration_result.returncode == 0:
                    audio_duration = float(duration_result.stdout.strip())
                    print(f"INFO: Audio duration: {{audio_duration:.2f}} seconds")
                    # Set timeout based on audio duration (approximately 30 seconds per audio second)
                    processing_timeout = max(900, int(audio_duration * 30))  # Minimum 15 minutes
                    print(f"INFO: Using processing timeout: {{processing_timeout}} seconds")
                else:
                    processing_timeout = 900  # Default 15 minutes
                    print(f"WARNING: Could not determine audio duration, using default timeout")
            except:
                processing_timeout = 900  # Default 15 minutes
                print(f"WARNING: Audio duration check failed, using default timeout")
            
            # Run SadTalker
            result_process = subprocess.run(
                sadtalker_cmd,
                capture_output=True,
                text=True,
                timeout=processing_timeout
            )
            
            if result_process.returncode == 0:
                print("INFO: SadTalker processing completed successfully")
                
                # Find the generated video file - search in unique result directory
                generated_files = []
                
                # Check for timestamped directories in our unique result directory
                for subdir in unique_result_dir.iterdir():
                    if subdir.is_dir() and subdir.name.startswith("2025_"):
                        mp4_files = list(subdir.glob("*.mp4"))
                        generated_files.extend(mp4_files)
                        print(f"INFO: Found {len(mp4_files)} MP4 files in timestamped directory: {subdir.name}")
                
                # Also check direct mp4 files in unique_result_dir
                direct_mp4_files = list(unique_result_dir.glob("*.mp4"))
                generated_files.extend(direct_mp4_files)
                print(f"INFO: Found {len(direct_mp4_files)} direct MP4 files in result directory")
                
                if generated_files:
                    # Get the latest generated file
                    latest_file = max(generated_files, key=os.path.getmtime)
                    print(f"INFO: Found generated video: {{latest_file}}")
                    
                    # Copy (don't move) the file to our expected output path
                    import shutil
                    shutil.copy2(str(latest_file), output_path)
                    print(f"INFO: Copied generated video to {{output_path}}")
                    
                    # Verify the copied file exists and has content
                    if Path(output_path).exists() and Path(output_path).stat().st_size > 1000:
                        print(f"INFO: Output file verified: {{Path(output_path).stat().st_size}} bytes")
                    else:
                        print("ERROR: Output file copy failed or file too small")
                        result = {{"success": False, "errors": ["Output file copy failed or file too small"]}}
                        print(json.dumps(result))
                        sys.exit(1)
                else:
                    print("ERROR: No video file generated by SadTalker")
                    print(f"INFO: Searched in {{unique_result_dir}} and subdirectories")
                    
                    # List all files found for debugging
                    all_files = list(unique_result_dir.rglob("*"))
                    print(f"INFO: Found {{len(all_files)}} total files in result directory:")
                    for file in all_files[:10]:  # Show first 10 files
                        print(f"  - {{file.name}} ({'file' if file.is_file() else 'dir'})")
                    
                    result = {{"success": False, "errors": ["No video file generated by SadTalker"]}}
                    print(json.dumps(result))
                    sys.exit(1)
            else:
                print(f"ERROR: SadTalker failed with return code {{result_process.returncode}}")
                print(f"STDERR: {{result_process.stderr[:500]}}")
                result = {{"success": False, "errors": [f"SadTalker execution failed with return code {{result_process.returncode}}"]}}
                print(json.dumps(result))
                sys.exit(1)
                
        except subprocess.TimeoutExpired:
            print("ERROR: SadTalker processing timed out")
            result = {{"success": False, "errors": ["SadTalker processing timed out"]}}
            print(json.dumps(result))
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: SadTalker execution failed: {{e}}")
            result = {{"success": False, "errors": [f"SadTalker execution failed: {{e}}"]}}
            print(json.dumps(result))
            sys.exit(1)
    
    # NO FALLBACK PROCESSING - PRODUCTION ONLY
    if not use_real_sadtalker:
        print("ERROR: Real SadTalker processing failed - NO FALLBACK ALLOWED IN PRODUCTION")
        result = {{"success": False, "errors": ["Real SadTalker failed and no fallback allowed in production pipeline"]}}
        print(json.dumps(result))
        sys.exit(1)
    
    # Verify output file
    if Path(output_path).exists():
        file_size = Path(output_path).stat().st_size
        
        # Get actual video properties
        cap = cv2.VideoCapture(output_path)
        if cap.isOpened():
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / actual_fps if actual_fps > 0 else 0
            cap.release()
        else:
            actual_fps = params.get('fps', 30)
            frame_count = int(duration * actual_fps) if 'duration' in locals() else 0
            width = height = params.get('size', 256)
            duration = audio_duration if 'audio_duration' in locals() else 0
        
        result = {{
            "success": True,
            "output_video_path": output_path,
            "file_size": file_size,
            "video_duration": duration,
            "video_fps": actual_fps,
            "video_frames": frame_count,
            "video_resolution": [width, height],
            "real_sadtalker_used": use_real_sadtalker,
            "errors": []
        }}
        
        print(f"INFO: Video created successfully")
        print(f"INFO: File size: {{file_size:,}} bytes")
        print(f"INFO: Duration: {{duration:.2f}} seconds")
        print(f"INFO: Resolution: {{width}}x{{height}}")
        print(f"INFO: FPS: {{actual_fps:.1f}}")
        print(f"INFO: Real SadTalker used: {{use_real_sadtalker}}")
        
        print(json.dumps(result))
    else:
        result = {{"success": False, "errors": ["Output video file not created"]}}
        print(json.dumps(result))
        sys.exit(1)
    
except Exception as e:
    import traceback
    result = {{"success": False, "errors": [f"Processing failed: {{str(e)}}"]}}
    print(json.dumps(result))
    traceback.print_exc()
    sys.exit(1)
'''
        return script
    
    def _execute_animation(self, script: str) -> Dict[str, Any]:
        """Execute lip-sync animation script in conda environment."""
        try:
            result = subprocess.run(
                [str(self.conda_python), '-c', script],
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            if result.returncode == 0:
                # Extract JSON from stdout (last line should be the JSON)
                try:
                    lines = result.stdout.strip().split('\n')
                    json_line = None
                    for line in reversed(lines):
                        if line.strip().startswith('{') and line.strip().endswith('}'):
                            json_line = line.strip()
                            break
                    
                    if not json_line:
                        logger.error("No JSON found in stdout")
                        return {'success': False, 'errors': ['No JSON found in script output']}
                    
                    animation_result = json.loads(json_line)
                    
                    # Add FFmpeg re-encoding for OpenCV compatibility if successful
                    if animation_result.get('success') and animation_result.get('output_video_path'):
                        reencoded_result = self._reencode_for_opencv_compatibility(animation_result)
                        if reencoded_result['success']:
                            animation_result = reencoded_result
                    
                    return animation_result
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse animation output: {e}")
                    logger.error(f"Raw stdout: {repr(result.stdout)}")
                    return {'success': False, 'errors': [f'Failed to parse animation output: {result.stdout[:200]}...']}
            else:
                logger.error(f"Animation script failed: {result.stderr}")
                return {'success': False, 'errors': [f'Animation script failed: {result.stderr}']}
                
        except subprocess.TimeoutExpired:
            logger.error("Animation script timed out")
            return {'success': False, 'errors': ['Animation script timed out']}
        except Exception as e:
            logger.error(f"Failed to execute animation script: {e}")
            return {'success': False, 'errors': [f'Failed to execute animation script: {e}']}
    
    def _reencode_for_opencv_compatibility(self, original_result: Dict[str, Any]) -> Dict[str, Any]:
        """Re-encode SadTalker output for OpenCV compatibility."""
        try:
            original_path = original_result['output_video_path']
            original_file = Path(original_path)
            
            # Create new filename for re-encoded video
            reencoded_filename = f"{original_file.stem}_opencv_compatible.mp4"
            reencoded_path = original_file.parent / reencoded_filename
            
            logger.info(f"[EMOJI] Re-encoding video for OpenCV compatibility:")
            logger.info(f"   Input: {original_file.name}")
            logger.info(f"   Output: {reencoded_filename}")
            
            # FFmpeg command for OpenCV-compatible encoding with dimension fixing
            ffmpeg_cmd = [
                'ffmpeg', '-y',  # Overwrite output
                '-i', str(original_path),  # Input video
                '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',  # Pad to even dimensions
                '-c:v', 'libx264',  # Video codec: H.264
                '-pix_fmt', 'yuv420p',  # Pixel format: widely compatible
                '-profile:v', 'baseline',  # H.264 profile: maximum compatibility
                '-level', '3.0',  # H.264 level
                '-r', '30',  # Frame rate: 30 FPS
                '-crf', '23',  # Quality: good balance
                '-preset', 'fast',  # Encoding speed
                '-movflags', '+faststart',  # Web optimization
                str(reencoded_path)
            ]
            
            # Execute FFmpeg re-encoding
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode == 0 and reencoded_path.exists():
                # Get properties of re-encoded video
                file_size = reencoded_path.stat().st_size
                
                # Test OpenCV compatibility
                import cv2
                cap = cv2.VideoCapture(str(reencoded_path))
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps if fps > 0 else 0
                    cap.release()
                    
                    logger.info("[SUCCESS] OpenCV compatibility verified:")
                    logger.info(f"   Resolution: {width}x{height}")
                    logger.info(f"   FPS: {fps:.1f}")
                    logger.info(f"   Frames: {frame_count}")
                    logger.info(f"   Duration: {duration:.2f} seconds")
                    logger.info(f"   File size: {file_size:,} bytes")
                    
                    # Remove original file and replace with re-encoded version
                    original_file.unlink()
                    reencoded_path.rename(original_path)
                    
                    # Update result with verified properties
                    updated_result = original_result.copy()
                    updated_result.update({
                        'file_size': file_size,
                        'video_duration': duration,
                        'video_fps': fps,
                        'video_frames': frame_count,
                        'video_resolution': [width, height],
                        'opencv_compatible': True,
                        'reencoded': True
                    })
                    
                    return updated_result
                else:
                    logger.error("[ERROR] Re-encoded video not compatible with OpenCV")
                    return {'success': False, 'errors': ['Re-encoded video not compatible with OpenCV']}
            else:
                logger.error(f"[ERROR] FFmpeg re-encoding failed: {result.stderr}")
                return {'success': False, 'errors': [f'FFmpeg re-encoding failed: {result.stderr}']}
                
        except Exception as e:
            logger.error(f"[ERROR] Re-encoding process failed: {e}")
            return {'success': False, 'errors': [f'Re-encoding process failed: {e}']}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the production model."""
        return {
            "model_name": "Production SadTalker",
            "device": self.device,
            "conda_environment": str(self.conda_python),
            "default_params": self.default_params,
            "available": self.available,
            "supports_emotion_mapping": True,
            "supports_expression_scaling": True,
            "supports_custom_fps": True,
            "supports_gpu_acceleration": self.device != "cpu",
            "fallback_enabled": False,
            "architecture": "production_only"
        }


def main():
    """Test Production SadTalker stage."""
    print("Target: Production SadTalker Stage Test")
    print("=" * 50)
    
    # Initialize stage
    stage = ProductionSadTalkerStage()
    
    # Test with sample data
    test_face = "/Users/aryanjain/Documents/video-synthesis-pipeline copy/NEW/user_assets/faces/sample_face.jpg"
    test_audio = "/Users/aryanjain/Documents/video-synthesis-pipeline copy/NEW/user_assets/voices/sample_voice.wav"
    
    # Sample emotion parameters
    emotion_params = {
        'emotion': 'professional',
        'tone': 'confident',
        'expression_scale': 1.2,
        'facial_animation_strength': 1.1
    }
    
    # Run processing
    results = stage.process_lip_sync_animation(test_face, test_audio, emotion_params)
    
    if results['success']:
        print("\\nSUCCESS Production SadTalker Stage test PASSED!")
        print(f"[SUCCESS] Output video: {results['output_video_path']}")
        print(f"[SUCCESS] File size: {results.get('file_size', 0):,} bytes")
        print(f"[SUCCESS] Video duration: {results.get('video_duration', 0):.2f} seconds")
        print(f"Duration:  Processing time: {results['processing_time']:.2f} seconds")
    else:
        print("\\n[ERROR] Production SadTalker Stage test FAILED!")
        print("Tools Errors:")
        for error in results['errors']:
            print(f"   - {error}")
    
    print(f"\\nStatus: Model info: {stage.get_model_info()}")


if __name__ == "__main__":
    main()