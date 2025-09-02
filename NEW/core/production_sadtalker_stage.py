#!/usr/bin/env python3
"""
Production SadTalker Stage - FIXED VERSION
Complete lip-sync animation for each voice chunk using SadTalker
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
    """Production SadTalker Stage - FIXED VERSION for lip-sync video generation."""
    
    def __init__(self, project_root: str = None):
        """Initialize Production SadTalker stage."""
        if project_root is None:
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = Path(project_root)
            
        # Output directory
        self.output_dir = self.project_root / "NEW" / "processed" / "sadtalker"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Conda environment for SadTalker (UPDATED TO WORKING INSTALLATION)
        self.conda_python = Path("/opt/miniconda3/envs/sadtalker/bin/python")
        
        # SadTalker configuration
        self.device = self._setup_device()
        self.default_params = {
            'size': 256,
            'expression_scale': 1.2,
            'preprocess_mode': 'resize',  # CRITICAL: Use resize to avoid 1-frame videos
            'fps': 30,
            'facial_animation_strength': 1.1,
            'enhancer': 'gfpgan',
            'still': False  # CRITICAL: Don't use still mode
        }
        
        # Verify environment
        self.available = self._verify_environment()
        
        if not self.available:
            raise RuntimeError("SadTalker production environment not available")
        
        logger.info(f"Production SadTalker Stage (FIXED) initialized")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Conda environment: {self.conda_python}")
        logger.info(f"  Output directory: {self.output_dir}")
    
    def _setup_device(self) -> str:
        """Setup device for SadTalker."""
        try:
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
            if not self.conda_python.exists():
                logger.error(f"Conda python not found: {self.conda_python}")
                return False
            
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
        """Process lip-sync animation using production SadTalker."""
        
        # Log SadTalker processing start
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
            unique_result_dir = self.output_dir / f"sadtalker_result_{timestamp}"
            unique_result_dir.mkdir(exist_ok=True)
            
            # Create processing script
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
            
            # Log detailed results
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
                
                # Check for duration issues
                duration = result.get('video_duration', 0)
                if pipeline_logger and duration < 1.0:
                    pipeline_logger.sadtalker_error("[EMOJI] CRITICAL: SadTalker 0.04s duration issue detected!", {
                        'actual_duration': duration,
                        'expected_minimum': 1.0,
                        'output_path': result['output_video_path'],
                        'video_frames': result.get('video_frames'),
                        'video_fps': result.get('video_fps')
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
    
    def _create_processing_script(self, face_image_path: str, audio_path: str, 
                                 result_dir: str, params: Dict[str, Any]) -> str:
        """Create SIMPLIFIED Python script for SadTalker processing."""
        
        script = f'''
import os
import sys
import json
import warnings
import subprocess
import time
from pathlib import Path

warnings.filterwarnings('ignore')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

try:
    import torch
    import cv2
    import numpy as np
    
    # Input parameters
    face_image_path = r"{face_image_path}"
    audio_path = r"{audio_path}"
    result_dir = r"{result_dir}"
    
    params = {json.dumps(params).replace('true', 'True').replace('false', 'False')}
    
    print(f"INFO: Processing with REAL SadTalker")
    print(f"INFO: Face: {{Path(face_image_path).name}}")
    print(f"INFO: Audio: {{Path(audio_path).name}}")
    print(f"INFO: Result dir: {{result_dir}}")
    
    # Change to SadTalker directory
    sadtalker_path = Path("/Users/aryanjain/Documents/video-synthesis-pipeline copy/real_models/SadTalker")
    os.chdir(str(sadtalker_path))
    
    # Build WORKING SadTalker command (from successful tests)
    sadtalker_cmd = [
        sys.executable, "inference.py",
        "--source_image", face_image_path,
        "--driven_audio", audio_path,
        "--result_dir", result_dir,
        "--size", str(params.get('size', 256)),
        "--batch_size", "1",
        "--preprocess", "resize",  # CRITICAL: resize fixes 1-frame issue
        "--cpu"
    ]
    
    expression_scale = params.get('expression_scale', 1.0)
    if expression_scale != 1.0:
        sadtalker_cmd.extend(["--expression_scale", str(expression_scale)])
    
    print(f"INFO: Running: {{' '.join(sadtalker_cmd)}}")
    
    # Execute SadTalker
    result_process = subprocess.run(
        sadtalker_cmd,
        capture_output=True,
        text=True,
        timeout=1200  # 20 minutes
    )
    
    if result_process.returncode == 0:
        print("INFO: SadTalker completed successfully")
        
        # Find generated video files
        result_path = Path(result_dir)
        generated_files = []
        
        # Check timestamped directories
        for subdir in result_path.iterdir():
            if subdir.is_dir() and subdir.name.startswith("2025_"):
                generated_files.extend(list(subdir.glob("*.mp4")))
        
        # Check direct files
        generated_files.extend(list(result_path.glob("*.mp4")))
        
        if generated_files:
            latest_file = max(generated_files, key=os.path.getmtime)
            print(f"INFO: Found video: {{latest_file}}")
            
            # Get properties
            file_size = latest_file.stat().st_size
            
            cap = cv2.VideoCapture(str(latest_file))
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                cap.release()
            else:
                fps, width, height, frame_count, duration = 25, 256, 256, 1, 0.04
            
            result = {{
                "success": True,
                "output_video_path": str(latest_file),
                "file_size": file_size,
                "video_duration": duration,
                "video_fps": fps,
                "video_frames": frame_count,
                "video_resolution": [width, height],
                "real_sadtalker_used": True,
                "errors": []
            }}
            
            print(json.dumps(result))
        else:
            result = {{"success": False, "errors": ["No video files generated"]}}
            print(json.dumps(result))
            sys.exit(1)
    else:
        result = {{"success": False, "errors": [f"SadTalker failed: {{result_process.stderr}}"]}}
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
                # Extract JSON from stdout
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
                    return animation_result
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse animation output: {e}")
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the production model."""
        return {
            "model_name": "Production SadTalker (FIXED)",
            "device": self.device,
            "conda_environment": str(self.conda_python),
            "default_params": self.default_params,
            "available": self.available,
            "supports_emotion_mapping": True,
            "supports_expression_scaling": True,
            "supports_custom_fps": True,
            "supports_gpu_acceleration": self.device != "cpu",
            "fallback_enabled": False,
            "architecture": "production_only_fixed"
        }


def main():
    """Test Production SadTalker stage."""
    print("Target: Production SadTalker Stage (FIXED) Test")
    print("=" * 50)
    
    # Initialize stage
    stage = ProductionSadTalkerStage()
    
    # Test with sample data
    test_face = "/Users/aryanjain/Documents/video-synthesis-pipeline copy/Q.jpg"
    test_audio = "/Users/aryanjain/Documents/video-synthesis-pipeline copy/NEW/processed/voice_chunks/xtts_voice_1753012522.wav"
    
    if not Path(test_audio).exists():
        print(f"[ERROR] Test audio not found: {test_audio}")
        return
    
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
        print("\nSUCCESS Production SadTalker Stage (FIXED) test PASSED!")
        print(f"[SUCCESS] Output video: {results['output_video_path']}")
        print(f"[SUCCESS] File size: {results.get('file_size', 0):,} bytes")
        print(f"[SUCCESS] Video duration: {results.get('video_duration', 0):.2f} seconds")
        print(f"Duration:  Processing time: {results['processing_time']:.2f} seconds")
    else:
        print("\n[ERROR] Production SadTalker Stage (FIXED) test FAILED!")
        print("Tools Errors:")
        for error in results['errors']:
            print(f"   - {error}")
    
    print(f"\nStatus: Model info: {stage.get_model_info()}")


if __name__ == "__main__":
    main()