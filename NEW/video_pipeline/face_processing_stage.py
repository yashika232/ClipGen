#!/usr/bin/env python3
"""
Face Processing Stage for NEW Video Pipeline
Integrates InsightFace processing with NEW metadata system
"""

import os
import sys
import subprocess
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

# Add paths for existing InsightFace integration
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "INTEGRATED_PIPELINE" / "src"))

from metadata_manager import MetadataManager

try:
    from insightface_stage import InsightFaceStage
except ImportError:
    # Fallback if direct import fails
    InsightFaceStage = None


class FaceProcessingStage:
    """Face processing stage that integrates with NEW metadata system."""
    
    def __init__(self, metadata_manager: MetadataManager = None):
        """Initialize face processing stage.
        
        Args:
            metadata_manager: Metadata manager instance
        """
        self.metadata_manager = metadata_manager or MetadataManager()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Output directory
        self.output_dir = self.metadata_manager.output_dir / "face_processing"
        self.output_dir.mkdir(exist_ok=True)
        
        # Try to import existing InsightFace stage
        self.insightface_stage = None
        if InsightFaceStage:
            try:
                self.insightface_stage = InsightFaceStage()
                self.logger.info("[SUCCESS] InsightFace stage imported successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize InsightFace stage: {e}")
    
    def validate_face_image(self, image_path: str) -> Dict[str, Any]:
        """Validate face image quality and detectability.
        
        Args:
            image_path: Path to face image
            
        Returns:
            Validation results dictionary
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {"valid": False, "error": "Cannot load image"}
            
            height, width = image.shape[:2]
            
            # Check image dimensions
            if width < 256 or height < 256:
                return {"valid": False, "error": "Image too small (minimum 256x256)"}
            
            # Check if image is too large
            if width > 2048 or height > 2048:
                return {"valid": False, "error": "Image too large (maximum 2048x2048)"}
            
            # Basic face detection using OpenCV as fallback
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return {"valid": False, "error": "No face detected in image"}
            
            if len(faces) > 1:
                return {"valid": False, "error": "Multiple faces detected, please use single face image"}
            
            # Get face region
            x, y, w, h = faces[0]
            face_area_ratio = (w * h) / (width * height)
            
            if face_area_ratio < 0.1:
                return {"valid": False, "error": "Face too small in image"}
            
            return {
                "valid": True,
                "image_size": (width, height),
                "face_region": (x, y, w, h),
                "face_area_ratio": face_area_ratio
            }
        
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def crop_and_align_face(self, image_path: str, output_path: str) -> Dict[str, Any]:
        """Crop and align face using InsightFace or fallback method.
        
        Args:
            image_path: Path to input image
            output_path: Path for cropped face output
            
        Returns:
            Processing results dictionary
        """
        try:
            if self.insightface_stage:
                # Use existing InsightFace stage
                result = self.insightface_stage.process_face_image(
                    image_path=image_path,
                    output_path=output_path
                )
                return result
            else:
                # Fallback to basic face cropping
                return self._basic_face_crop(image_path, output_path)
        
        except Exception as e:
            self.logger.error(f"Face processing failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _basic_face_crop(self, image_path: str, output_path: str) -> Dict[str, Any]:
        """Basic face cropping using OpenCV.
        
        Args:
            image_path: Path to input image
            output_path: Path for cropped face output
            
        Returns:
            Processing results dictionary
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {"success": False, "error": "Cannot load image"}
            
            # Detect face
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return {"success": False, "error": "No face detected"}
            
            # Use the largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            
            # Expand crop region slightly
            margin = int(min(w, h) * 0.3)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(image.shape[1], x + w + margin)
            y2 = min(image.shape[0], y + h + margin)
            
            # Crop face
            face_crop = image[y1:y2, x1:x2]
            
            # Resize to square
            target_size = 512
            face_resized = cv2.resize(face_crop, (target_size, target_size))
            
            # Save cropped face
            success = cv2.imwrite(output_path, face_resized)
            
            if success:
                return {
                    "success": True,
                    "face_region": (x, y, w, h),
                    "crop_region": (x1, y1, x2, y2),
                    "output_size": (target_size, target_size)
                }
            else:
                return {"success": False, "error": "Failed to save cropped face"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def enhance_face_quality(self, face_image_path: str, output_path: str) -> Dict[str, Any]:
        """Enhance face image quality for better synthesis results.
        
        Args:
            face_image_path: Path to face image
            output_path: Path for enhanced output
            
        Returns:
            Enhancement results dictionary
        """
        try:
            # Load image
            image = cv2.imread(face_image_path)
            if image is None:
                return {"success": False, "error": "Cannot load image"}
            
            # Apply basic enhancement
            enhanced = image.copy()
            
            # Improve contrast
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Reduce noise
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # Save enhanced image
            success = cv2.imwrite(output_path, enhanced)
            
            if success:
                return {"success": True, "enhanced": True}
            else:
                return {"success": False, "error": "Failed to save enhanced image"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def process_face_processing(self) -> Dict[str, Any]:
        """Process face processing stage using metadata.
        
        Returns:
            Processing results dictionary
        """
        start_time = time.time()
        
        # Update status to processing
        self.metadata_manager.update_stage_status("face_processing", "processing")
        
        try:
            # Load metadata
            metadata = self.metadata_manager.load_metadata()
            if metadata is None:
                raise ValueError("Failed to load metadata")
            
            # Get face image path
            face_image_path = self.metadata_manager.get_stage_input("face_processing", "input_image")
            if face_image_path is None:
                raise ValueError("Face image not found in metadata")
            
            self.logger.info(f"Processing face image: {face_image_path}")
            
            # Validate face image
            validation = self.validate_face_image(face_image_path)
            if not validation["valid"]:
                raise ValueError(f"Face validation failed: {validation['error']}")
            
            self.logger.info(f"[SUCCESS] Face validation passed")
            self.logger.info(f"  Image size: {validation['image_size']}")
            self.logger.info(f"  Face area ratio: {validation['face_area_ratio']:.2%}")
            
            # Generate output paths
            timestamp = int(time.time())
            face_crop_path = self.output_dir / f"face_crop_{timestamp}.jpg"
            face_enhanced_path = self.output_dir / f"face_enhanced_{timestamp}.jpg"
            
            # Crop and align face
            crop_result = self.crop_and_align_face(face_image_path, str(face_crop_path))
            
            if not crop_result["success"]:
                raise ValueError(f"Face cropping failed: {crop_result['error']}")
            
            self.logger.info(f"[SUCCESS] Face cropped successfully")
            
            # Enhance face quality
            enhance_result = self.enhance_face_quality(str(face_crop_path), str(face_enhanced_path))
            
            # Use enhanced version if successful, otherwise use crop
            if enhance_result["success"]:
                final_face_path = face_enhanced_path
                self.logger.info(f"[SUCCESS] Face enhanced successfully")
            else:
                final_face_path = face_crop_path
                self.logger.warning(f"Face enhancement failed, using crop: {enhance_result.get('error')}")
            
            processing_time = time.time() - start_time
            
            # Convert to relative path
            relative_path = final_face_path.relative_to(self.metadata_manager.new_dir)
            
            # Prepare face data
            face_data = {
                "validation": validation,
                "crop_result": crop_result,
                "enhanced": enhance_result.get("success", False)
            }
            
            # Update metadata with success
            update_data = {
                "face_crop": str(relative_path),
                "face_data": face_data,
                "processing_time": processing_time,
                "error": None
            }
            
            self.metadata_manager.update_stage_status("face_processing", "completed", update_data)
            
            # Update video generation inputs
            self.metadata_manager.update_stage_status("video_generation", None, {
                "input_face": str(relative_path)
            })
            
            self.logger.info(f"[SUCCESS] Face processing completed in {processing_time:.1f}s")
            
            return {
                "success": True,
                "face_crop_path": str(relative_path),
                "face_data": face_data,
                "processing_time": processing_time
            }
        
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            # Update metadata with error
            update_data = {
                "processing_time": processing_time,
                "error": error_msg
            }
            
            self.metadata_manager.update_stage_status("face_processing", "failed", update_data)
            
            self.logger.error(f"[ERROR] Face processing stage failed: {error_msg}")
            return {"success": False, "error": error_msg}
    
    def get_emotion_parameters(self, emotion: str, tone: str) -> Dict[str, Any]:
        """Get face processing parameters based on emotion and tone.
        
        Args:
            emotion: Emotion parameter (inspired, confident, etc.)
            tone: Tone parameter (professional, friendly, etc.)
            
        Returns:
            Face processing parameters
        """
        # Base parameters
        params = {
            "detection_threshold": 0.6,
            "alignment_strength": 1.0,
            "enhancement_level": 0.7
        }
        
        # Emotion-based adjustments
        emotion_adjustments = {
            "inspired": {"detection_threshold": 0.7, "enhancement_level": 0.8},
            "confident": {"detection_threshold": 0.8, "enhancement_level": 0.9},
            "curious": {"detection_threshold": 0.6, "enhancement_level": 0.6},
            "excited": {"detection_threshold": 0.7, "enhancement_level": 0.8},
            "calm": {"detection_threshold": 0.6, "enhancement_level": 0.7}
        }
        
        # Tone-based adjustments
        tone_adjustments = {
            "professional": {"detection_threshold": 0.8, "enhancement_level": 0.9},
            "friendly": {"detection_threshold": 0.7, "enhancement_level": 0.7},
            "motivational": {"detection_threshold": 0.8, "enhancement_level": 0.8},
            "casual": {"detection_threshold": 0.6, "enhancement_level": 0.6}
        }
        
        # Apply adjustments
        if emotion in emotion_adjustments:
            params.update(emotion_adjustments[emotion])
        
        if tone in tone_adjustments:
            params.update(tone_adjustments[tone])
        
        return params


def main():
    """Test face processing stage."""
    stage = FaceProcessingStage()
    
    # Check metadata status
    metadata = stage.metadata_manager.load_metadata()
    if metadata:
        print("[SUCCESS] Metadata loaded successfully")
        
        # Check if face processing is ready
        face_image = stage.metadata_manager.get_stage_input("face_processing", "input_image")
        if face_image:
            print(f"[EMOJI] Face image found: {face_image}")
            
            # Process face processing
            result = stage.process_face_processing()
            
            if result["success"]:
                print("[SUCCESS] Face processing completed successfully")
                print(f"Output: {result.get('face_crop_path')}")
            else:
                print(f"[ERROR] Face processing failed: {result.get('error')}")
        else:
            print("[ERROR] No face image found in metadata")
    else:
        print("[ERROR] Failed to load metadata")


if __name__ == "__main__":
    main()