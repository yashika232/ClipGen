#!/usr/bin/env python3
"""
Production InsightFace Stage - Buffalo_l Model Only
Enhanced face detection and processing for video synthesis pipeline
NO FALLBACK MECHANISMS - Production mode only
"""

import os
import sys
import cv2
import numpy as np
import time
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
import logging
import subprocess

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionInsightFaceStage:
    """Production InsightFace Stage - Buffalo_l Model Only."""
    
    def __init__(self, project_root: str = None):
        """Initialize Production InsightFace stage.
        
        Args:
            project_root: Path to project root directory
        """
        if project_root is None:
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = Path(project_root)
            
        # Output directory
        self.output_dir = self.project_root / "NEW" / "processed" / "insightface"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Conda environment for InsightFace (using sadtalker env)
        self.conda_python = Path("/opt/miniconda3/envs/sadtalker/bin/python")
        
        # InsightFace configuration
        self.model_name = "buffalo_l"  # Production model only
        self.detection_threshold = 0.3
        self.det_size = (320, 320)
        self.max_faces = 1
        
        # Verify environment
        self.available = self._verify_environment()
        
        if not self.available:
            raise RuntimeError("InsightFace production environment not available")
        
        logger.info(f"Production InsightFace Stage initialized")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Conda environment: {self.conda_python}")
        logger.info(f"  Output directory: {self.output_dir}")
    
    def _verify_environment(self) -> bool:
        """Verify InsightFace conda environment is available."""
        try:
            # Check if conda python exists
            if not self.conda_python.exists():
                logger.error(f"Conda python not found: {self.conda_python}")
                return False
            
            # Test InsightFace import
            test_script = '''
import sys
try:
    import insightface
    from insightface.app import FaceAnalysis
    print("SUCCESS: InsightFace available")
    sys.exit(0)
except ImportError as e:
    print(f"ERROR: InsightFace not available: {e}")
    sys.exit(1)
'''
            
            result = subprocess.run(
                [str(self.conda_python), '-c', test_script],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                logger.info("[SUCCESS] InsightFace production environment verified")
                return True
            else:
                logger.error(f"[ERROR] InsightFace environment test failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] Environment verification failed: {e}")
            return False
    
    def process_face_detection(self, image_path: str, emotion_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process face detection using production InsightFace.
        
        Args:
            image_path: Path to input image
            emotion_params: Emotion-aware parameters (optional)
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        
        result = {
            'stage': 'production_insightface_detection',
            'timestamp': time.time(),
            'success': False,
            'input_image_path': image_path,
            'emotion_params': emotion_params or {},
            'detected_faces': [],
            'best_face_crop_path': None,
            'processing_time': 0,
            'errors': []
        }
        
        try:
            # Validate input
            if not Path(image_path).exists():
                result['errors'].append(f"Input image not found: {image_path}")
                return result
            
            # Create processing script
            processing_script = self._create_processing_script(image_path, emotion_params)
            
            # Execute face detection via conda environment
            detection_result = self._execute_detection(processing_script)
            
            if detection_result['success']:
                result.update(detection_result)
                result['success'] = True
                
                logger.info("[SUCCESS] Production InsightFace Detection completed successfully!")
                logger.info(f"   Detected faces: {len(result['detected_faces'])}")
                logger.info(f"   Best face crop: {result['best_face_crop_path']}")
            else:
                result['errors'].extend(detection_result['errors'])
                
        except Exception as e:
            result['errors'].append(f"Production InsightFace processing failed: {str(e)}")
            logger.error(f"[ERROR] Production InsightFace Stage error: {e}")
            
        finally:
            result['processing_time'] = time.time() - start_time
            
            # Save stage results
            results_file = self.output_dir / f"production_insightface_results_{int(time.time())}.json"
            with open(results_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"File: Production InsightFace results saved: {results_file}")
            logger.info(f"Duration:  Processing time: {result['processing_time']:.2f} seconds")
        
        return result
    
    def _create_processing_script(self, image_path: str, emotion_params: Dict[str, Any] = None) -> str:
        """Create Python script for InsightFace processing."""
        # Adjust detection threshold based on emotion
        det_thresh = self.detection_threshold
        if emotion_params:
            if emotion_params.get('emotion') == 'excited':
                det_thresh = 0.25  # More sensitive for excited expressions
            elif emotion_params.get('emotion') == 'calm':
                det_thresh = 0.4   # More conservative for calm expressions
        
        timestamp = int(time.time())
        crop_filename = f"face_crop_{timestamp}.jpg"
        crop_path = self.output_dir / crop_filename
        
        script = f'''
import cv2
import numpy as np
import json
import sys
import warnings
from pathlib import Path

# Suppress warnings to ensure clean JSON output
warnings.filterwarnings('ignore')

try:
    import insightface
    from insightface.app import FaceAnalysis
    
    # Initialize InsightFace
    app = FaceAnalysis(name='{self.model_name}', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size={self.det_size})
    
    # Load image
    image = cv2.imread(r'{image_path}')
    if image is None:
        result = {{"success": False, "errors": ["Failed to load image"]}}
        print(json.dumps(result))
        sys.exit(1)
    
    # Detect faces
    faces = app.get(image, max_num={self.max_faces})
    
    if not faces:
        result = {{"success": False, "errors": ["No faces detected"]}}
        print(json.dumps(result))
        sys.exit(1)
    
    # Process detected faces
    detected_faces = []
    for face in faces:
        if face.det_score >= {det_thresh}:
            # Convert numpy arrays to Python lists with native types
            bbox_list = [int(x) for x in face.bbox.tolist()]
            landmarks_list = [[int(x), int(y)] for x, y in face.kps.tolist()]
            
            face_data = {{
                "bbox": bbox_list,
                "confidence": float(face.det_score),
                "landmarks_5": landmarks_list,
                "age": int(getattr(face, 'age', 0)) if hasattr(face, 'age') and face.age is not None else None,
                "gender": int(getattr(face, 'gender', 0)) if hasattr(face, 'gender') and face.gender is not None else None
            }}
            detected_faces.append(face_data)
    
    if not detected_faces:
        result = {{"success": False, "errors": ["No faces above confidence threshold"]}}
        print(json.dumps(result))
        sys.exit(1)
    
    # Extract best face crop
    best_face = detected_faces[0]  # Highest confidence
    bbox = best_face["bbox"]
    x1, y1, x2, y2 = bbox
    
    # Add margin
    margin = 0.2
    width = x2 - x1
    height = y2 - y1
    margin_x = int(width * margin)
    margin_y = int(height * margin)
    
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(image.shape[1], x2 + margin_x)
    y2 = min(image.shape[0], y2 + margin_y)
    
    # Extract and resize face crop
    face_crop = image[y1:y2, x1:x2]
    face_crop = cv2.resize(face_crop, (512, 512), interpolation=cv2.INTER_LANCZOS4)
    
    # Save face crop
    cv2.imwrite(r'{crop_path}', face_crop)
    
    result = {{
        "success": True,
        "detected_faces": detected_faces,
        "best_face_crop_path": r"{crop_path}",
        "errors": []
    }}
    
    print(json.dumps(result))
    
except Exception as e:
    import traceback
    result = {{"success": False, "errors": [f"Processing failed: {{str(e)}}"]}}
    print(json.dumps(result))
    traceback.print_exc()
    sys.exit(1)
'''
        return script
    
    def _execute_detection(self, script: str) -> Dict[str, Any]:
        """Execute face detection script in conda environment."""
        try:
            result = subprocess.run(
                [str(self.conda_python), '-c', script],
                capture_output=True,
                text=True,
                timeout=120  # 2 minutes timeout
            )
            
            if result.returncode == 0:
                # Parse JSON output
                try:
                    stdout_clean = result.stdout.strip()
                    if not stdout_clean:
                        logger.error("Empty stdout from detection script")
                        return {'success': False, 'errors': ['Empty output from detection script']}
                    
                    # Extract JSON from stdout (last line should be the JSON)
                    lines = stdout_clean.split('\n')
                    json_line = None
                    for line in reversed(lines):
                        if line.strip().startswith('{') and line.strip().endswith('}'):
                            json_line = line.strip()
                            break
                    
                    if not json_line:
                        logger.error("No JSON found in stdout")
                        return {'success': False, 'errors': ['No JSON found in script output']}
                    
                    detection_result = json.loads(json_line)
                    return detection_result
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse detection output: {e}")
                    logger.error(f"Raw stdout: {repr(result.stdout)}")
                    logger.error(f"Raw stderr: {repr(result.stderr)}")
                    return {'success': False, 'errors': [f'Failed to parse detection output. Raw output: {result.stdout[:200]}...']}
            else:
                logger.error(f"Detection script failed: {result.stderr}")
                return {'success': False, 'errors': [f'Detection script failed: {result.stderr}']}
                
        except subprocess.TimeoutExpired:
            logger.error("Detection script timed out")
            return {'success': False, 'errors': ['Detection script timed out']}
        except Exception as e:
            logger.error(f"Failed to execute detection script: {e}")
            return {'success': False, 'errors': [f'Failed to execute detection script: {e}']}
    
    def get_detector_info(self) -> Dict[str, Any]:
        """Get information about the production detector."""
        return {
            "detector_name": "Production InsightFace",
            "model_name": self.model_name,
            "conda_environment": str(self.conda_python),
            "detection_threshold": self.detection_threshold,
            "det_size": self.det_size,
            "max_faces": self.max_faces,
            "available": self.available,
            "supports_landmarks": True,
            "supports_attributes": True,
            "fallback_enabled": False,
            "architecture": "production_only"
        }


def main():
    """Test Production InsightFace stage."""
    print("Target: Production InsightFace Stage Test")
    print("=" * 50)
    
    # Initialize stage
    stage = ProductionInsightFaceStage()
    
    # Test image path
    test_image = "/Users/aryanjain/Documents/video-synthesis-pipeline copy/NEW/user_assets/faces/sample_face.jpg"
    
    # Sample emotion parameters (USER CONFIGURABLE)
    emotion_params = {
        'emotion': 'professional',  # USER CONFIGURABLE
        'tone': 'confident'         # USER CONFIGURABLE
    }
    
    # Run processing
    results = stage.process_face_detection(test_image, emotion_params)
    
    if results['success']:
        print("\\nSUCCESS Production InsightFace Stage test PASSED!")
        print(f"[SUCCESS] Detected faces: {len(results['detected_faces'])}")
        print(f"[SUCCESS] Face crop saved: {results['best_face_crop_path']}")
        print(f"Duration:  Processing time: {results['processing_time']:.2f} seconds")
    else:
        print("\\n[ERROR] Production InsightFace Stage test FAILED!")
        print("Tools Errors:")
        for error in results['errors']:
            print(f"   - {error}")
    
    print(f"\\nStatus: Detector info: {stage.get_detector_info()}")


if __name__ == "__main__":
    main()