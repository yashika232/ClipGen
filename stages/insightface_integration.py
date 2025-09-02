#!/usr/bin/env python3
"""
InsightFace Integration Module
Enhanced face detection and analysis using InsightFace for superior accuracy and robustness.
Replaces default face detection with state-of-the-art models optimized for Apple Silicon.
"""

import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
import logging
import tempfile
import subprocess

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

from utils.insightface_compatibility_fix import apply_insightface_compatibility_fixes, create_insightface_wrapper

try:
    import insightface
    from insightface.app import FaceAnalysis
    from insightface.data import get_image as ins_get_image
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    FaceAnalysis = None

class InsightFaceDetector:
    """
    Advanced face detection and analysis using InsightFace.
    Provides superior accuracy, multi-face support, and age/gender analysis.
    """
    
    def __init__(self, 
                 device: str = "auto",
                 model_name: str = "buffalo_l",
                 det_size: Tuple[int, int] = (640, 640),
                 detection_threshold: float = 0.6):
        """
        Initialize InsightFace detector.
        
        Args:
            device: Device to use ('cpu', 'cuda', 'mps', 'auto')
            model_name: InsightFace model name ('buffalo_l', 'buffalo_m', 'buffalo_s')
            det_size: Detection input size
            detection_threshold: Minimum confidence for face detection
        """
        self.logger = logging.getLogger("InsightFaceDetector")
        self.device = self._setup_device(device)
        self.model_name = model_name
        self.det_size = det_size
        self.detection_threshold = detection_threshold
        
        # Initialize face analyzer
        self.app = None
        self.available = self._initialize_insightface()
        
        if not self.available:
            self.logger.warning("InsightFace not available. Will use fallback detection.")
    
    def _setup_device(self, device: str) -> str:
        """Setup and validate device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "cpu"  # InsightFace doesn't support MPS directly
            else:
                return "cpu"
        return device
    
    def _initialize_insightface(self) -> bool:
        """Initialize InsightFace models with compatibility fixes."""
        if not INSIGHTFACE_AVAILABLE:
            self.logger.error("InsightFace not installed. Please install with: pip install insightface")
            return False
        
        try:
            # Apply compatibility fixes
            apply_insightface_compatibility_fixes()
            
            # Use compatibility wrapper
            InsightFaceWrapper = create_insightface_wrapper()
            
            # Set provider based on device
            if self.device == "cuda":
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
            
            # Initialize with wrapper
            self.app = InsightFaceWrapper(
                model_name=self.model_name,
                providers=providers,
                device=self.device
            )
            
            if self.app.available:
                self.logger.info(f"InsightFace initialized successfully with compatibility fixes")
                return True
            else:
                self.logger.error("InsightFace wrapper initialization failed")
                return False
            
        except Exception as e:
            self.logger.error(f"Failed to initialize InsightFace: {e}")
            return False
    
    def detect_faces(self, 
                    image: np.ndarray, 
                    max_faces: int = 5) -> List[Dict]:
        """
        Detect faces in image with enhanced analysis.
        
        Args:
            image: Input image (BGR format)
            max_faces: Maximum number of faces to detect
            
        Returns:
            List of face detection results with bounding boxes, landmarks, and attributes
        """
        if not self.available:
            return self._fallback_detect_faces(image, max_faces)
        
        try:
            # Use the wrapper's detect_faces method
            faces = self.app.detect_faces(image, max_faces=max_faces)
            
            # Process results to match expected format
            results = []
            for idx, face in enumerate(faces):
                face_info = {
                    "face_id": idx,
                    "bbox": face.get("bbox", []),
                    "confidence": face.get("confidence", 0.0),
                    "landmarks": face.get("landmarks", []),
                    "landmarks_68": None,  # Will be computed if needed
                    "area": 0,
                    "age": face.get("age"),
                    "gender": face.get("gender"),
                    "embedding": None
                }
                
                # Calculate area from bbox
                bbox = face_info["bbox"]
                if len(bbox) >= 4:
                    face_info["area"] = int((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                
                results.append(face_info)
            
            self.logger.debug(f"Detected {len(results)} faces with InsightFace compatibility wrapper")
            return results
            
        except Exception as e:
            self.logger.error(f"InsightFace detection failed: {e}")
            return self._fallback_detect_faces(image, max_faces)
    
    def _compute_68_landmarks(self, image: np.ndarray, face) -> Optional[List[List[int]]]:
        """
        Compute 68-point landmarks from InsightFace 5-point landmarks.
        Uses face alignment model for detailed landmark detection.
        """
        try:
            # Extract face region
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            # Add padding
            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding) 
            x2 = min(image.shape[1], x2 + padding)
            y2 = min(image.shape[0], y2 + padding)
            
            face_crop = image[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                return None
            
            # Use face alignment to get 68 landmarks
            # This is a simplified approach - in production, you'd use a dedicated 68-point detector
            landmarks_68 = self._estimate_68_from_5_points(face.kps, bbox)
            
            return landmarks_68.astype(int).tolist()
            
        except Exception as e:
            self.logger.warning(f"68-point landmark computation failed: {e}")
            return None
    
    def _estimate_68_from_5_points(self, kps_5: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Estimate 68 landmarks from 5 key points using facial geometry.
        This is a simplified mapping - for production use, integrate with FAN or similar.
        """
        # Extract 5 key points
        left_eye = kps_5[0]
        right_eye = kps_5[1] 
        nose = kps_5[2]
        left_mouth = kps_5[3]
        right_mouth = kps_5[4]
        
        # Create a basic 68-point estimation
        landmarks_68 = np.zeros((68, 2))
        
        # Face outline (0-16) - estimated from bbox and eye positions
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]
        
        # Jaw line approximation
        for i in range(17):
            t = i / 16.0
            landmarks_68[i] = [
                bbox[0] + t * face_width,
                bbox[1] + 0.7 * face_height + 0.3 * face_height * np.sin(t * np.pi)
            ]
        
        # Eyebrows (17-26) - estimated above eyes
        eye_y_offset = -15
        landmarks_68[17:22, 0] = np.linspace(left_eye[0] - 20, left_eye[0] + 20, 5)
        landmarks_68[17:22, 1] = left_eye[1] + eye_y_offset
        landmarks_68[22:27, 0] = np.linspace(right_eye[0] - 20, right_eye[0] + 20, 5) 
        landmarks_68[22:27, 1] = right_eye[1] + eye_y_offset
        
        # Nose bridge and tip (27-35)
        landmarks_68[27:31, 0] = nose[0]
        landmarks_68[27:31, 1] = np.linspace(left_eye[1] + 10, nose[1] - 10, 4)
        landmarks_68[31:36, 0] = np.linspace(nose[0] - 8, nose[0] + 8, 5)
        landmarks_68[31:36, 1] = nose[1]
        
        # Eyes (36-47)
        landmarks_68[36:42] = self._create_eye_landmarks(left_eye)
        landmarks_68[42:48] = self._create_eye_landmarks(right_eye)
        
        # Mouth (48-67)
        landmarks_68[48:68] = self._create_mouth_landmarks(left_mouth, right_mouth, nose)
        
        return landmarks_68
    
    def _create_eye_landmarks(self, eye_center: np.ndarray) -> np.ndarray:
        """Create 6 eye landmarks around eye center."""
        eye_landmarks = np.zeros((6, 2))
        eye_landmarks[0] = eye_center + [-15, 0]   # Left corner
        eye_landmarks[1] = eye_center + [-8, -5]   # Top left
        eye_landmarks[2] = eye_center + [0, -8]    # Top center
        eye_landmarks[3] = eye_center + [15, 0]    # Right corner
        eye_landmarks[4] = eye_center + [8, 5]     # Bottom right
        eye_landmarks[5] = eye_center + [0, 8]     # Bottom center
        return eye_landmarks
    
    def _create_mouth_landmarks(self, left_mouth: np.ndarray, 
                              right_mouth: np.ndarray, 
                              nose: np.ndarray) -> np.ndarray:
        """Create 20 mouth landmarks."""
        mouth_landmarks = np.zeros((20, 2))
        mouth_center = (left_mouth + right_mouth) / 2
        mouth_width = np.linalg.norm(right_mouth - left_mouth)
        
        # Outer mouth contour (12 points)
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        for i, angle in enumerate(angles):
            mouth_landmarks[i] = mouth_center + [
                mouth_width * 0.6 * np.cos(angle),
                mouth_width * 0.3 * np.sin(angle)
            ]
        
        # Inner mouth contour (8 points)
        inner_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        for i, angle in enumerate(inner_angles):
            mouth_landmarks[12 + i] = mouth_center + [
                mouth_width * 0.3 * np.cos(angle),
                mouth_width * 0.15 * np.sin(angle)
            ]
        
        return mouth_landmarks
    
    def _fallback_detect_faces(self, image: np.ndarray, max_faces: int) -> List[Dict]:
        """
        Fallback face detection using OpenCV Haar cascades.
        """
        try:
            self.logger.info("Using OpenCV fallback face detection")
            
            # Load OpenCV face detector
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Limit number of faces
            if len(faces) > max_faces:
                # Sort by area and take largest faces
                areas = [(w * h, i) for i, (x, y, w, h) in enumerate(faces)]
                areas.sort(reverse=True)
                faces = [faces[areas[i][1]] for i in range(max_faces)]
            
            # Convert to standard format
            results = []
            for idx, (x, y, w, h) in enumerate(faces):
                results.append({
                    "face_id": idx,
                    "bbox": [x, y, x + w, y + h],
                    "confidence": 0.8,  # Default confidence for OpenCV
                    "landmarks": None,
                    "landmarks_68": None,
                    "area": w * h,
                    "age": None,
                    "gender": None,
                    "embedding": None
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Fallback face detection failed: {e}")
            return []
    
    def get_best_face(self, 
                     image: np.ndarray,
                     prefer_center: bool = True,
                     min_size: int = 100) -> Optional[Dict]:
        """
        Get the best face from image based on size, position, and quality.
        
        Args:
            image: Input image
            prefer_center: Prefer faces closer to image center
            min_size: Minimum face size in pixels
            
        Returns:
            Best face detection result or None
        """
        faces = self.detect_faces(image, max_faces=10)
        
        if not faces:
            return None
        
        # Filter by minimum size
        valid_faces = [f for f in faces if f["area"] >= min_size * min_size]
        
        if not valid_faces:
            return None
        
        # Score faces based on criteria
        image_center = np.array([image.shape[1] / 2, image.shape[0] / 2])
        
        best_face = None
        best_score = -1
        
        for face in valid_faces:
            bbox = face["bbox"]
            face_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            
            # Calculate score
            size_score = min(face["area"] / (image.shape[0] * image.shape[1]), 0.5)  # Prefer larger faces, cap at 50%
            
            center_distance = np.linalg.norm(face_center - image_center)
            center_score = 1.0 / (1.0 + center_distance / max(image.shape)) if prefer_center else 0.5
            
            confidence_score = face["confidence"]
            
            total_score = size_score * 0.4 + center_score * 0.3 + confidence_score * 0.3
            
            if total_score > best_score:
                best_score = total_score
                best_face = face
        
        return best_face
    
    def enhance_face_crop(self, 
                         image: np.ndarray, 
                         face: Dict, 
                         crop_size: Tuple[int, int] = (512, 512),
                         margin: float = 0.3) -> Optional[np.ndarray]:
        """
        Extract and enhance face crop with proper alignment.
        
        Args:
            image: Source image
            face: Face detection result
            crop_size: Output crop size
            margin: Additional margin around face bbox
            
        Returns:
            Cropped and aligned face image
        """
        try:
            bbox = face["bbox"]
            x1, y1, x2, y2 = bbox
            
            # Add margin
            width = x2 - x1
            height = y2 - y1
            margin_x = int(width * margin)
            margin_y = int(height * margin)
            
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(image.shape[1], x2 + margin_x)
            y2 = min(image.shape[0], y2 + margin_y)
            
            # Extract face crop
            face_crop = image[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                return None
            
            # Resize to target size
            face_crop = cv2.resize(face_crop, crop_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Apply basic enhancement
            face_crop = self._enhance_face_quality(face_crop)
            
            return face_crop
            
        except Exception as e:
            self.logger.error(f"Face crop extraction failed: {e}")
            return None
    
    def _enhance_face_quality(self, face_crop: np.ndarray) -> np.ndarray:
        """Apply basic face enhancement."""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge and convert back
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Slight sharpening
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * 0.1
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            enhanced = cv2.addWeighted(face_crop, 0.8, enhanced, 0.2, 0)
            
            return enhanced
            
        except Exception:
            return face_crop
    
    def get_detector_info(self) -> Dict[str, Any]:
        """Get information about the face detector."""
        return {
            "detector_name": "InsightFace" if self.available else "OpenCV_Fallback",
            "model_name": self.model_name if self.available else "haarcascade",
            "device": self.device,
            "detection_threshold": self.detection_threshold,
            "det_size": self.det_size,
            "available": self.available,
            "supports_landmarks": True,
            "supports_attributes": self.available,
            "supports_embeddings": self.available
        }


def install_insightface():
    """Install InsightFace if not available."""
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "insightface", "onnxruntime", "onnxruntime-gpu"
        ], check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    """Test InsightFace integration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test InsightFace Integration")
    parser.add_argument("--image", required=True, help="Test image path")
    parser.add_argument("--device", default="auto", help="Device (cpu/cuda/mps/auto)")
    parser.add_argument("--install", action="store_true", help="Install InsightFace if missing")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Install if requested
    if args.install and not INSIGHTFACE_AVAILABLE:
        print("Installing InsightFace...")
        if install_insightface():
            print("[SUCCESS] InsightFace installed successfully")
        else:
            print("[ERROR] Failed to install InsightFace")
            return 1
    
    # Initialize detector
    detector = InsightFaceDetector(device=args.device)
    
    # Print detector info
    info = detector.get_detector_info()
    print("\\nSearch Face Detector Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Load and test image
    image = cv2.imread(args.image)
    if image is None:
        print(f"[ERROR] Could not load image: {args.image}")
        return 1
    
    print(f"\\n[EMOJI] Testing with image: {args.image}")
    print(f"   Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Detect faces
    faces = detector.detect_faces(image)
    print(f"\\nTarget: Detected {len(faces)} faces:")
    
    for i, face in enumerate(faces):
        bbox = face["bbox"]
        print(f"  Face {i+1}:")
        print(f"    Bbox: {bbox}")
        print(f"    Confidence: {face['confidence']:.3f}")
        print(f"    Area: {face['area']} pixels")
        if face['age'] is not None:
            print(f"    Age: {face['age']}")
        if face['gender'] is not None:
            print(f"    Gender: {face['gender']}")
    
    # Get best face
    best_face = detector.get_best_face(image)
    if best_face:
        print(f"\\n[EMOJI] Best face: Face {best_face['face_id'] + 1}")
        
        # Extract face crop
        face_crop = detector.enhance_face_crop(image, best_face)
        if face_crop is not None:
            output_path = "best_face_crop.jpg"
            cv2.imwrite(output_path, face_crop)
            print(f"Storage Face crop saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())