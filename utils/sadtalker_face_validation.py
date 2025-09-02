#!/usr/bin/env python3
"""
SadTalker Face Detection Validation Pipeline
Tests face detection capabilities before running full SadTalker inference
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import json
import traceback

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

def validate_face_detection(image_path: str, confidence_threshold: float = 0.5):
    """
    Validate that SadTalker can detect faces in the given image.
    
    Args:
        image_path: Path to the face image
        confidence_threshold: Minimum confidence for face detection
    
    Returns:
        dict: Validation results including detected faces and confidence scores
    """
    try:
        # Import SadTalker modules
        sadtalker_path = PROJECT_ROOT / "models" / "SadTalker"
        sys.path.append(str(sadtalker_path))
        
        from src.face3d.models import networks
        from src.utils.face_detection import FaceAlignment, LandmarksType
        
        # Initialize face detection components
        device = 'cpu'  # Use CPU for validation to avoid memory issues
        
        # Create face detector (using the same setup as SadTalker)
        fa = FaceAlignment(LandmarksType._2D, flip_input=False, device=device)
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        
        print(f"Search Validating face detection for: {Path(image_path).name}")
        print(f"Status: Image size: {image.size}")
        
        # Test face detection with different confidence thresholds
        validation_results = {
            "image_path": str(image_path),
            "image_size": image.size,
            "faces_detected": 0,
            "detection_results": [],
            "validation_status": "UNKNOWN",
            "recommendations": []
        }
        
        # Try face detection
        try:
            # Use face_alignment library for initial detection
            landmarks = fa.get_landmarks(image_array)
            
            if landmarks and len(landmarks) > 0:
                validation_results["faces_detected"] = len(landmarks)
                validation_results["validation_status"] = "PASS"
                
                for i, landmark in enumerate(landmarks):
                    face_info = {
                        "face_id": i + 1,
                        "landmark_points": len(landmark),
                        "confidence": "high",  # face_alignment doesn't return confidence
                        "bbox_estimated": estimate_bbox_from_landmarks(landmark)
                    }
                    validation_results["detection_results"].append(face_info)
                
                print(f"[SUCCESS] Detected {len(landmarks)} face(s) with landmarks")
                
            else:
                validation_results["validation_status"] = "FAIL"
                validation_results["recommendations"].append("No faces detected - try a clearer, well-lit face image")
                print("[ERROR] No faces detected")
                
        except Exception as e:
            validation_results["validation_status"] = "ERROR"
            validation_results["error"] = str(e)
            print(f"[ERROR] Face detection error: {e}")
        
        # Additional validation checks
        if validation_results["faces_detected"] > 1:
            validation_results["recommendations"].append("Multiple faces detected - SadTalker works best with single face images")
        
        if image.size[0] < 256 or image.size[1] < 256:
            validation_results["recommendations"].append("Image resolution is low - higher resolution may improve detection")
        
        return validation_results
        
    except ImportError as e:
        return {
            "validation_status": "ERROR",
            "error": f"Could not import SadTalker modules: {e}",
            "recommendations": ["Ensure SadTalker environment is properly set up"]
        }
    except Exception as e:
        return {
            "validation_status": "ERROR",
            "error": str(e),
            "recommendations": ["Check image file and SadTalker installation"]
        }

def estimate_bbox_from_landmarks(landmarks):
    """Estimate bounding box from facial landmarks."""
    if landmarks is None or len(landmarks) == 0:
        return None
    
    x_coords = landmarks[:, 0]
    y_coords = landmarks[:, 1]
    
    return {
        "x_min": float(np.min(x_coords)),
        "y_min": float(np.min(y_coords)),
        "x_max": float(np.max(x_coords)),
        "y_max": float(np.max(y_coords)),
        "width": float(np.max(x_coords) - np.min(x_coords)),
        "height": float(np.max(y_coords) - np.min(y_coords))
    }

def test_sadtalker_face_detector(image_path: str):
    """
    Test SadTalker's specific face detection method.
    This simulates the exact detection process used in SadTalker inference.
    """
    try:
        # Import SadTalker specific detection
        sadtalker_path = PROJECT_ROOT / "models" / "SadTalker"
        sys.path.append(str(sadtalker_path))
        
        from src.face3d.extract_kp_videos_safe import KeypointExtractor
        from src.utils.face_detection import FaceAlignment, LandmarksType
        
        # Initialize keypoint extractor (this is what SadTalker actually uses)
        device = 'cpu'
        kp_extractor = KeypointExtractor(device=device)
        
        # Load image as PIL Image
        image = Image.open(image_path).convert('RGB')
        
        print(f"🧪 Testing SadTalker's face detector on: {Path(image_path).name}")
        
        # Test the exact method that was failing
        try:
            # This is the method that was causing the IndexError
            current_kp = kp_extractor.extract_keypoint(image)
            
            return {
                "test_status": "PASS",
                "keypoints_extracted": True,
                "keypoint_shape": current_kp.shape if hasattr(current_kp, 'shape') else None,
                "message": "SadTalker face detection successful"
            }
            
        except IndexError as e:
            if "index 0 is out of bounds" in str(e):
                return {
                    "test_status": "FAIL",
                    "keypoints_extracted": False,
                    "error": "No faces detected by SadTalker (bbox array empty)",
                    "message": "Face detection confidence too low or no clear face in image"
                }
            else:
                raise e
                
        except Exception as e:
            return {
                "test_status": "ERROR",
                "keypoints_extracted": False,
                "error": str(e),
                "message": "Unexpected error in SadTalker face detection"
            }
            
    except Exception as e:
        return {
            "test_status": "ERROR",
            "error": str(e),
            "message": "Could not initialize SadTalker face detector"
        }

def validate_all_test_faces(test_dir: str):
    """Validate all face images in the test directory."""
    test_path = Path(test_dir)
    
    if not test_path.exists():
        print(f"[ERROR] Test directory not found: {test_dir}")
        return
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    face_images = []
    
    for ext in image_extensions:
        face_images.extend(test_path.glob(f"*{ext}"))
        face_images.extend(test_path.glob(f"*{ext.upper()}"))
    
    if not face_images:
        print(f"[ERROR] No face images found in {test_dir}")
        return
    
    print(f"Search Validating {len(face_images)} face images...")
    
    validation_report = {
        "test_directory": str(test_dir),
        "total_images": len(face_images),
        "validation_results": [],
        "summary": {
            "passed": 0,
            "failed": 0,
            "errors": 0
        }
    }
    
    for image_path in face_images:
        print(f"\n{'='*50}")
        
        # General face detection validation
        general_result = validate_face_detection(str(image_path))
        
        # SadTalker specific test
        sadtalker_result = test_sadtalker_face_detector(str(image_path))
        
        # Combine results
        combined_result = {
            "image_name": image_path.name,
            "image_path": str(image_path),
            "general_validation": general_result,
            "sadtalker_test": sadtalker_result
        }
        
        validation_report["validation_results"].append(combined_result)
        
        # Update summary
        if (general_result.get("validation_status") == "PASS" and 
            sadtalker_result.get("test_status") == "PASS"):
            validation_report["summary"]["passed"] += 1
            print(f"[SUCCESS] {image_path.name}: PASSED")
        elif (general_result.get("validation_status") == "FAIL" or 
              sadtalker_result.get("test_status") == "FAIL"):
            validation_report["summary"]["failed"] += 1
            print(f"[ERROR] {image_path.name}: FAILED")
        else:
            validation_report["summary"]["errors"] += 1
            print(f"[WARNING] {image_path.name}: ERROR")
    
    # Save validation report
    report_path = test_path / "face_validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    print(f"\n{'='*50}")
    print("Status: VALIDATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total images tested: {validation_report['summary']['passed'] + validation_report['summary']['failed'] + validation_report['summary']['errors']}")
    print(f"[SUCCESS] Passed: {validation_report['summary']['passed']}")
    print(f"[ERROR] Failed: {validation_report['summary']['failed']}")
    print(f"[WARNING] Errors: {validation_report['summary']['errors']}")
    print(f"File: Report saved: {report_path}")
    
    return validation_report

def main():
    """Main validation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate face detection for SadTalker")
    parser.add_argument("--test-dir", required=True, help="Directory containing test face images")
    parser.add_argument("--single-image", help="Test a single image file")
    
    args = parser.parse_args()
    
    if args.single_image:
        print("Search Testing single image...")
        general_result = validate_face_detection(args.single_image)
        sadtalker_result = test_sadtalker_face_detector(args.single_image)
        
        print("\nStatus: Results:")
        print(f"General validation: {general_result.get('validation_status')}")
        print(f"SadTalker test: {sadtalker_result.get('test_status')}")
        
    else:
        print("Search Testing all images in directory...")
        validate_all_test_faces(args.test_dir)

if __name__ == "__main__":
    main()