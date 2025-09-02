#!/usr/bin/env python3
"""
Stage 3: Face Processing - Simple Face Detection and Cropping
Processes face image for use in video generation
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

class SimpleFaceProcessing:
    """Simple face processing using MediaPipe."""
    
    def __init__(self):
        """Initialize face processing."""
        self.project_root = PROJECT_ROOT
        self.outputs_dir = self.project_root / "outputs"
        self.outputs_dir.mkdir(exist_ok=True)
        
        # Load session data
        self.session_data = self._load_session_data()
        
        logger.info("[EMOJI] Simple Face Processing initialized")
    
    def _load_session_data(self) -> Dict[str, Any]:
        """Load session data from JSON file."""
        session_file = self.outputs_dir / "session_data.json"
        if session_file.exists():
            with open(session_file, 'r') as f:
                return json.load(f)
        return {}
    
    def process_face(self) -> bool:
        """Process face image and create face crop."""
        try:
            logger.info("Target: Starting face processing...")
            
            # Get face image path
            face_image_path = self.session_data.get("face_image_path")
            if not face_image_path:
                raise ValueError("No face image path found in session data")
            
            face_image_path = Path(face_image_path)
            if not face_image_path.exists():
                raise FileNotFoundError(f"Face image not found: {face_image_path}")
            
            # Try to use existing face processing implementation
            success = self._process_with_existing_implementation(face_image_path)
            
            if not success:
                # Fallback to simple copy
                logger.warning("Using fallback face processing...")
                success = self._process_with_fallback(face_image_path)
            
            return success
            
        except Exception as e:
            logger.error(f"[ERROR] Face processing failed: {str(e)}")
            return False
    
    def _process_with_existing_implementation(self, face_image_path: Path) -> bool:
        """Process face using existing MediaPipe implementation."""
        try:
            import cv2
            import mediapipe as mp
            
            # Read image
            image = cv2.imread(str(face_image_path))
            if image is None:
                raise ValueError(f"Could not read image: {face_image_path}")
            
            # Initialize MediaPipe Face Detection
            mp_face_detection = mp.solutions.face_detection
            face_detection = mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5
            )
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process image
            results = face_detection.process(rgb_image)
            
            if results.detections:
                # Get first detection
                detection = results.detections[0]
                
                # Get bounding box
                h, w, _ = image.shape
                bbox = detection.location_data.relative_bounding_box
                
                # Convert to pixel coordinates
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Add padding
                padding = 50
                x = max(0, x - padding)
                y = max(0, y - padding)
                width = min(w - x, width + 2 * padding)
                height = min(h - y, height + 2 * padding)
                
                # Crop face
                face_crop = image[y:y+height, x:x+width]
                
                # Resize to standard size
                face_crop = cv2.resize(face_crop, (512, 512))
                
                # Save cropped face
                output_path = self.outputs_dir / "face_crop.jpg"
                cv2.imwrite(str(output_path), face_crop)
                
                logger.info(f"[SUCCESS] Face detected and cropped successfully")
                logger.info(f"   Face confidence: {detection.score[0]:.2f}")
                logger.info(f"   Face crop: {output_path}")
                
                # Save face processing metadata
                face_metadata = {
                    "processed_at": time.time(),
                    "input_image_path": str(face_image_path),
                    "output_crop_path": str(output_path),
                    "face_detected": True,
                    "face_confidence": float(detection.score[0]),
                    "crop_size": [512, 512],
                    "processing_success": True
                }
                
                metadata_file = self.outputs_dir / "face_processing_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(face_metadata, f, indent=2)
                
                return True
            else:
                logger.warning("No face detected in image")
                return False
                
        except Exception as e:
            logger.error(f"Face processing with MediaPipe failed: {str(e)}")
            return False
    
    def _process_with_fallback(self, face_image_path: Path) -> bool:
        """Fallback face processing - simple resize and copy."""
        try:
            import cv2
            
            # Read image
            image = cv2.imread(str(face_image_path))
            if image is None:
                raise ValueError(f"Could not read image: {face_image_path}")
            
            # Simple resize to standard size
            resized_image = cv2.resize(image, (512, 512))
            
            # Save as face crop
            output_path = self.outputs_dir / "face_crop.jpg"
            cv2.imwrite(str(output_path), resized_image)
            
            logger.info(f"[SUCCESS] Face processing completed with fallback method")
            logger.info(f"   Face crop: {output_path}")
            
            # Save face processing metadata
            face_metadata = {
                "processed_at": time.time(),
                "input_image_path": str(face_image_path),
                "output_crop_path": str(output_path),
                "face_detected": False,
                "face_confidence": 0.0,
                "crop_size": [512, 512],
                "processing_success": True,
                "processing_method": "fallback_resize"
            }
            
            metadata_file = self.outputs_dir / "face_processing_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(face_metadata, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Fallback face processing failed: {str(e)}")
            return False


def main():
    """Main entry point."""
    try:
        processor = SimpleFaceProcessing()
        success = processor.process_face()
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Face processing failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())