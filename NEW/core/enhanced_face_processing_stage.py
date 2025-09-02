#!/usr/bin/env python3
"""
Enhanced Face Processing Stage for Video Synthesis Pipeline
Integrates InsightFace with metadata-driven architecture for face preparation
"""

import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging
from datetime import datetime
from PIL import Image, ImageEnhance

# Add the core directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_metadata_manager import EnhancedMetadataManager

# Try to import existing InsightFace components
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "INTEGRATED_PIPELINE" / "src"))
    from insightface_stage import InsightFaceStage
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

# Optional advanced face processing libraries
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False


class EnhancedFaceProcessingStage:
    """Enhanced face processing stage with metadata-driven architecture."""
    
    def __init__(self, base_dir: str = None):
        """Initialize enhanced face processing stage.
        
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
        self.face_output_dir = self.base_dir / "processed" / "face_crops"
        self.face_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to initialize InsightFace stage
        self.insightface_stage = None
        if INSIGHTFACE_AVAILABLE:
            try:
                self.insightface_stage = InsightFaceStage()
                self.logger.info("[SUCCESS] InsightFace stage initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize InsightFace stage: {e}")
                self.insightface_stage = None
        
        # Face processing configuration
        self.face_config = {
            'min_face_size': 256,
            'max_face_size': 2048,
            'target_size': 512,
            'quality_threshold': 0.7,
            'detection_confidence': 0.8,
            'alignment_threshold': 0.9,
            'enhancement_strength': 0.8
        }
        
        # Load face detection models
        self._load_face_detection_models()
        
        self.logger.info("STARTING Enhanced Face Processing Stage initialized")
    
    def _load_face_detection_models(self):
        """Load face detection models (OpenCV, dlib if available)."""
        try:
            # Load OpenCV face cascade
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            # Load dlib face detector if available
            if DLIB_AVAILABLE:
                try:
                    self.dlib_detector = dlib.get_frontal_face_detector()
                    self.dlib_predictor = None  # Would need shape_predictor_68_face_landmarks.dat
                    self.logger.info("[SUCCESS] Dlib face detector loaded")
                except Exception as e:
                    self.logger.warning(f"Dlib face detector failed: {e}")
                    self.dlib_detector = None
            else:
                self.dlib_detector = None
            
        except Exception as e:
            self.logger.error(f"Failed to load face detection models: {e}")
    
    def process_face_processing(self) -> Dict[str, Any]:
        """Process face processing using metadata-driven approach.
        
        Returns:
            Dictionary with processing results
        """
        try:
            # Update stage status to processing
            self.metadata_manager.update_stage_status(
                "face_processing", 
                "processing",
                {"input_image": "user_assets/faces/"}
            )
            
            # Load current session metadata
            metadata = self.metadata_manager.load_metadata()
            if not metadata:
                raise ValueError("No active session found")
            
            # Get face image from user assets
            user_assets = metadata.get('user_assets', {})
            face_image_path = user_assets.get('face_image')
            
            if not face_image_path:
                raise ValueError("No face image found in user assets")
            
            # Convert to absolute path
            face_image_full_path = self.base_dir / face_image_path
            if not face_image_full_path.exists():
                raise ValueError(f"Face image file not found: {face_image_full_path}")
            
            # Get user preferences for processing
            user_inputs = metadata.get('user_inputs', {})
            tone = user_inputs.get('tone', 'professional')
            emotion = user_inputs.get('emotion', 'confident')
            content_type = user_inputs.get('content_type', 'Short-Form Video Reel')
            
            self.logger.info(f"Target: Processing face image:")
            self.logger.info(f"   Face image: {face_image_path}")
            self.logger.info(f"   Tone: {tone}, Emotion: {emotion}")
            self.logger.info(f"   Content type: {content_type}")
            
            processing_start = time.time()
            
            # Step 1: Validate face image
            validation_result = self._validate_face_image(str(face_image_full_path))
            if not validation_result['valid']:
                raise ValueError(f"Face validation failed: {validation_result['error']}")
            
            self.logger.info("[SUCCESS] Face validation passed")
            
            # Step 2: Extract and align face
            extraction_result = self._extract_and_align_face(
                str(face_image_full_path), tone, emotion
            )
            if not extraction_result['success']:
                raise ValueError(f"Face extraction failed: {extraction_result['error']}")
            
            self.logger.info("[SUCCESS] Face extraction completed")
            
            # Step 3: Enhance face quality
            enhancement_result = self._enhance_face_quality(
                extraction_result['processed_path'], tone, emotion
            )
            
            # Use enhanced version if successful
            if enhancement_result['success']:
                final_face_path = enhancement_result['enhanced_path']
                self.logger.info("[SUCCESS] Face enhancement completed")
            else:
                final_face_path = extraction_result['processed_path']
                self.logger.warning(f"Face enhancement failed: {enhancement_result.get('error')}")
            
            processing_time = time.time() - processing_start
            
            # Get relative path for metadata
            relative_path = Path(final_face_path).relative_to(self.base_dir)
            
            # Prepare comprehensive face processing data
            processing_data = {
                "processed_face": str(relative_path),
                "validation_results": validation_result,
                "extraction_results": extraction_result,
                "enhancement_results": enhancement_result,
                "processing_duration": processing_time,
                "face_config": {
                    "tone": tone,
                    "emotion": emotion,
                    "content_type": content_type,
                    "target_size": self.face_config['target_size']
                }
            }
            
            # Update metadata with successful results
            self.metadata_manager.update_stage_status(
                "face_processing",
                "completed",
                input_paths={"input_image": face_image_path},
                output_paths={"processed_face": str(relative_path)},
                processing_data=processing_data
            )
            
            self.logger.info(f"[SUCCESS] Face processing completed successfully in {processing_time:.1f}s")
            self.logger.info(f"   Output: {relative_path}")
            
            return {
                'success': True,
                'output_path': str(relative_path),
                'processing_time': processing_time,
                'validation_results': validation_result,
                'stage_updated': True
            }
        
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"[ERROR] Face processing error: {error_msg}")
            
            # Update metadata with error
            self.metadata_manager.update_stage_status(
                "face_processing",
                "failed",
                error_info={"error": error_msg}
            )
            
            return {
                'success': False,
                'error': error_msg,
                'stage_updated': True
            }
    
    def _validate_face_image(self, image_path: str) -> Dict[str, Any]:
        """Comprehensive face image validation.
        
        Args:
            image_path: Path to face image file
            
        Returns:
            Validation results
        """
        try:
            # Load image with OpenCV
            image = cv2.imread(image_path)
            if image is None:
                return {'valid': False, 'error': 'Cannot load image file'}
            
            height, width = image.shape[:2]
            
            # Check dimensions
            if width < self.face_config['min_face_size'] or height < self.face_config['min_face_size']:
                return {
                    'valid': False, 
                    'error': f"Image too small (minimum {self.face_config['min_face_size']}x{self.face_config['min_face_size']})"
                }
            
            if width > self.face_config['max_face_size'] or height > self.face_config['max_face_size']:
                return {
                    'valid': False, 
                    'error': f"Image too large (maximum {self.face_config['max_face_size']}x{self.face_config['max_face_size']})"
                }
            
            # Face detection with multiple methods
            face_detection_results = self._detect_faces_multi_method(image)
            
            if not face_detection_results['faces_found']:
                return {'valid': False, 'error': 'No face detected in image'}
            
            if face_detection_results['face_count'] > 1:
                return {'valid': False, 'error': 'Multiple faces detected - please use single face image'}
            
            # Quality assessment
            quality_score = self._assess_image_quality(image)
            
            if quality_score < self.face_config['quality_threshold']:
                return {
                    'valid': False, 
                    'error': f"Image quality too low (score: {quality_score:.2f}, minimum: {self.face_config['quality_threshold']})"
                }
            
            return {
                'valid': True,
                'image_size': (width, height),
                'face_detection': face_detection_results,
                'quality_score': quality_score,
                'file_size_mb': Path(image_path).stat().st_size / (1024 * 1024)
            }
            
        except Exception as e:
            return {'valid': False, 'error': f"Validation error: {str(e)}"}
    
    def _detect_faces_multi_method(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect faces using multiple methods for better accuracy.
        
        Args:
            image: Input image array
            
        Returns:
            Face detection results
        """
        faces_opencv = []
        faces_dlib = []
        
        # OpenCV detection
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces_opencv = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
        except Exception as e:
            self.logger.warning(f"OpenCV face detection failed: {e}")
        
        # Dlib detection (if available)
        if self.dlib_detector:
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces_dlib = self.dlib_detector(gray)
                faces_dlib = [(d.left(), d.top(), d.width(), d.height()) for d in faces_dlib]
            except Exception as e:
                self.logger.warning(f"Dlib face detection failed: {e}")
        
        # Combine results
        all_faces = list(faces_opencv)
        if faces_dlib:
            all_faces.extend(faces_dlib)
        
        # Remove duplicates (faces that are too close to each other)
        unique_faces = self._remove_duplicate_faces(all_faces)
        
        return {
            'faces_found': len(unique_faces) > 0,
            'face_count': len(unique_faces),
            'faces': unique_faces,
            'opencv_count': len(faces_opencv),
            'dlib_count': len(faces_dlib)
        }
    
    def _remove_duplicate_faces(self, faces: List[Tuple], overlap_threshold: float = 0.5) -> List[Tuple]:
        """Remove duplicate face detections.
        
        Args:
            faces: List of face bounding boxes (x, y, w, h)
            overlap_threshold: Overlap threshold for considering faces as duplicates
            
        Returns:
            List of unique faces
        """
        if len(faces) <= 1:
            return faces
        
        # Calculate areas and remove duplicates
        unique_faces = []
        
        for face in faces:
            x, y, w, h = face
            is_duplicate = False
            
            for unique_face in unique_faces:
                ux, uy, uw, uh = unique_face
                
                # Calculate overlap
                overlap_x = max(0, min(x + w, ux + uw) - max(x, ux))
                overlap_y = max(0, min(y + h, uy + uh) - max(y, uy))
                overlap_area = overlap_x * overlap_y
                
                face_area = w * h
                unique_area = uw * uh
                
                if overlap_area / min(face_area, unique_area) > overlap_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_faces.append(face)
        
        return unique_faces
    
    def _assess_image_quality(self, image: np.ndarray) -> float:
        """Assess image quality using multiple metrics.
        
        Args:
            image: Input image array
            
        Returns:
            Quality score (0-1)
        """
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Sharpness (Laplacian variance)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(1.0, sharpness / 1000.0)
            
            # Contrast (standard deviation)
            contrast = np.std(gray)
            contrast_score = min(1.0, contrast / 128.0)
            
            # Brightness (avoid over/under exposure)
            brightness = np.mean(gray)
            brightness_score = 1.0 - abs(brightness - 128) / 128.0
            
            # Overall quality score (weighted average)
            quality_score = (
                sharpness_score * 0.4 +
                contrast_score * 0.3 +
                brightness_score * 0.3
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            self.logger.warning(f"Quality assessment failed: {e}")
            return 0.5  # Default moderate quality
    
    def _extract_and_align_face(self, image_path: str, tone: str, emotion: str) -> Dict[str, Any]:
        """Extract and align face for optimal processing.
        
        Args:
            image_path: Path to input image
            tone: Voice tone parameter
            emotion: Voice emotion parameter
            
        Returns:
            Extraction results
        """
        try:
            if self.insightface_stage:
                # Use InsightFace for professional face extraction
                return self._extract_with_insightface(image_path, tone, emotion)
            else:
                # Use enhanced OpenCV method
                return self._extract_with_opencv(image_path, tone, emotion)
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _extract_with_insightface(self, image_path: str, tone: str, emotion: str) -> Dict[str, Any]:
        """Extract face using InsightFace for best quality.
        
        Args:
            image_path: Path to input image
            tone: Voice tone parameter
            emotion: Voice emotion parameter
            
        Returns:
            Extraction results
        """
        try:
            timestamp = int(datetime.now().timestamp())
            output_filename = f"face_insightface_{timestamp}.jpg"
            output_path = self.face_output_dir / output_filename
            
            # Use InsightFace stage
            result = self.insightface_stage.process_face_image(
                image_path=image_path,
                output_path=str(output_path)
            )
            
            if result.get('success', False):
                return {
                    'success': True,
                    'processed_path': str(output_path),
                    'method': 'insightface',
                    'face_data': result
                }
            else:
                # Fallback to OpenCV if InsightFace fails
                return self._extract_with_opencv(image_path, tone, emotion)
                
        except Exception as e:
            # Fallback to OpenCV if InsightFace fails
            self.logger.warning(f"InsightFace extraction failed: {e}, falling back to OpenCV")
            return self._extract_with_opencv(image_path, tone, emotion)
    
    def _extract_with_opencv(self, image_path: str, tone: str, emotion: str) -> Dict[str, Any]:
        """Extract face using enhanced OpenCV method.
        
        Args:
            image_path: Path to input image
            tone: Voice tone parameter
            emotion: Voice emotion parameter
            
        Returns:
            Extraction results
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {'success': False, 'error': 'Cannot load image'}
            
            # Detect faces
            face_detection = self._detect_faces_multi_method(image)
            if not face_detection['faces_found']:
                return {'success': False, 'error': 'No face detected'}
            
            # Use the largest face
            faces = face_detection['faces']
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            
            # Calculate crop region with emotion-based margins
            margin_factor = self._get_margin_factor(tone, emotion)
            margin = int(min(w, h) * margin_factor)
            
            # Expand crop region
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(image.shape[1], x + w + margin)
            y2 = min(image.shape[0], y + h + margin)
            
            # Crop face
            face_crop = image[y1:y2, x1:x2]
            
            # Resize to target size
            target_size = self.face_config['target_size']
            face_resized = cv2.resize(face_crop, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
            
            # Save processed face
            timestamp = int(datetime.now().timestamp())
            output_filename = f"face_opencv_{timestamp}.jpg"
            output_path = self.face_output_dir / output_filename
            
            success = cv2.imwrite(str(output_path), face_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            if success:
                return {
                    'success': True,
                    'processed_path': str(output_path),
                    'method': 'opencv',
                    'face_region': (x, y, w, h),
                    'crop_region': (x1, y1, x2, y2),
                    'output_size': (target_size, target_size),
                    'margin_factor': margin_factor
                }
            else:
                return {'success': False, 'error': 'Failed to save processed face'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _get_margin_factor(self, tone: str, emotion: str) -> float:
        """Get crop margin factor based on tone and emotion.
        
        Args:
            tone: Voice tone
            emotion: Voice emotion
            
        Returns:
            Margin factor for face cropping
        """
        # Base margin
        margin = 0.3
        
        # Adjust based on tone
        tone_adjustments = {
            'professional': 0.25,  # Tighter crop for professional look
            'friendly': 0.35,      # More context for friendly appearance
            'motivational': 0.3,   # Standard crop
            'casual': 0.4          # Wider crop for casual feel
        }
        
        # Adjust based on emotion
        emotion_adjustments = {
            'inspired': 0.35,    # More context for inspired expression
            'confident': 0.25,   # Tighter crop for confident look
            'curious': 0.3,      # Standard crop
            'excited': 0.35,     # More context for excited expression
            'calm': 0.25         # Tighter crop for calm look
        }
        
        # Apply adjustments
        if tone in tone_adjustments:
            margin = tone_adjustments[tone]
        
        if emotion in emotion_adjustments:
            margin = (margin + emotion_adjustments[emotion]) / 2
        
        return margin
    
    def _enhance_face_quality(self, face_image_path: str, tone: str, emotion: str) -> Dict[str, Any]:
        """Enhance face image quality for better synthesis results.
        
        Args:
            face_image_path: Path to processed face image
            tone: Voice tone parameter
            emotion: Voice emotion parameter
            
        Returns:
            Enhancement results
        """
        try:
            # Load image
            image = cv2.imread(face_image_path)
            if image is None:
                return {'success': False, 'error': 'Cannot load processed face image'}
            
            enhanced = image.copy()
            
            # Get enhancement parameters based on tone and emotion
            enhancement_params = self._get_enhancement_parameters(tone, emotion)
            
            # Apply enhancements
            enhanced = self._apply_face_enhancements(enhanced, enhancement_params)
            
            # Save enhanced image
            timestamp = int(datetime.now().timestamp())
            output_filename = f"face_enhanced_{timestamp}.jpg"
            output_path = self.face_output_dir / output_filename
            
            success = cv2.imwrite(str(output_path), enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            if success:
                return {
                    'success': True,
                    'enhanced_path': str(output_path),
                    'enhancement_params': enhancement_params
                }
            else:
                return {'success': False, 'error': 'Failed to save enhanced face'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _get_enhancement_parameters(self, tone: str, emotion: str) -> Dict[str, float]:
        """Get enhancement parameters based on tone and emotion.
        
        Args:
            tone: Voice tone
            emotion: Voice emotion
            
        Returns:
            Enhancement parameters
        """
        # Base parameters
        params = {
            'contrast_enhancement': 1.1,
            'brightness_adjustment': 1.0,
            'saturation_boost': 1.05,
            'sharpening_strength': 0.5,
            'noise_reduction': 0.7
        }
        
        # Tone-based adjustments
        tone_adjustments = {
            'professional': {
                'contrast_enhancement': 1.15,
                'brightness_adjustment': 1.05,
                'saturation_boost': 1.0,
                'sharpening_strength': 0.6
            },
            'friendly': {
                'contrast_enhancement': 1.1,
                'brightness_adjustment': 1.1,
                'saturation_boost': 1.1,
                'sharpening_strength': 0.4
            },
            'motivational': {
                'contrast_enhancement': 1.2,
                'brightness_adjustment': 1.1,
                'saturation_boost': 1.15,
                'sharpening_strength': 0.7
            },
            'casual': {
                'contrast_enhancement': 1.05,
                'brightness_adjustment': 1.0,
                'saturation_boost': 1.05,
                'sharpening_strength': 0.3
            }
        }
        
        # Apply adjustments
        if tone in tone_adjustments:
            params.update(tone_adjustments[tone])
        
        return params
    
    def _apply_face_enhancements(self, image: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Apply face enhancements based on parameters.
        
        Args:
            image: Input face image
            params: Enhancement parameters
            
        Returns:
            Enhanced image
        """
        enhanced = image.copy()
        
        try:
            # Convert to PIL for easier enhancement operations
            pil_image = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
            
            # Contrast enhancement
            if params.get('contrast_enhancement', 1.0) != 1.0:
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(params['contrast_enhancement'])
            
            # Brightness adjustment
            if params.get('brightness_adjustment', 1.0) != 1.0:
                enhancer = ImageEnhance.Brightness(pil_image)
                pil_image = enhancer.enhance(params['brightness_adjustment'])
            
            # Saturation boost
            if params.get('saturation_boost', 1.0) != 1.0:
                enhancer = ImageEnhance.Color(pil_image)
                pil_image = enhancer.enhance(params['saturation_boost'])
            
            # Convert back to OpenCV format
            enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Apply sharpening if needed
            sharpening_strength = params.get('sharpening_strength', 0.0)
            if sharpening_strength > 0:
                enhanced = self._apply_unsharp_mask(enhanced, sharpening_strength)
            
            # Apply noise reduction if needed
            noise_reduction = params.get('noise_reduction', 0.0)
            if noise_reduction > 0:
                enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            return enhanced
            
        except Exception as e:
            self.logger.warning(f"Enhancement failed: {e}")
            return image  # Return original if enhancement fails
    
    def _apply_unsharp_mask(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Apply unsharp mask for sharpening.
        
        Args:
            image: Input image
            strength: Sharpening strength (0-1)
            
        Returns:
            Sharpened image
        """
        try:
            # Create Gaussian blur
            blurred = cv2.GaussianBlur(image, (0, 0), 2.0)
            
            # Create unsharp mask
            unsharp_mask = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
            
            return unsharp_mask
            
        except Exception as e:
            self.logger.warning(f"Sharpening failed: {e}")
            return image
    
    def get_face_processing_status(self) -> Dict[str, Any]:
        """Get current face processing stage status.
        
        Returns:
            Status information
        """
        try:
            stage_status = self.metadata_manager.get_stage_status("face_processing")
            if stage_status:
                return {
                    'success': True,
                    'stage_status': stage_status
                }
            else:
                return {
                    'success': False,
                    'error': 'No face processing stage found in metadata'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def validate_face_processing_prerequisites(self) -> Dict[str, Any]:
        """Validate that face processing can be started.
        
        Returns:
            Validation results
        """
        try:
            # Check metadata
            metadata = self.metadata_manager.load_metadata()
            if not metadata:
                return {'valid': False, 'errors': ['No active session found']}
            
            errors = []
            warnings = []
            
            # Check for face image
            user_assets = metadata.get('user_assets', {})
            face_image = user_assets.get('face_image')
            if not face_image:
                errors.append('No face image uploaded')
            elif face_image:
                face_path = self.base_dir / face_image
                if not face_path.exists():
                    errors.append(f'Face image file not found: {face_image}')
            
            # Check InsightFace availability
            if not self.insightface_stage:
                warnings.append('InsightFace not available - will use OpenCV fallback')
            
            # Check dlib availability
            if not DLIB_AVAILABLE:
                warnings.append('Dlib not available - using OpenCV only for face detection')
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings,
                'ready_for_processing': len(errors) == 0
            }
            
        except Exception as e:
            return {
                'valid': False,
                'errors': [str(e)]
            }


def main():
    """Test the enhanced face processing stage."""
    print("🧪 Testing Enhanced Face Processing Stage")
    print("=" * 50)
    
    # Initialize stage
    stage = EnhancedFaceProcessingStage()
    
    # Check prerequisites
    prereq_result = stage.validate_face_processing_prerequisites()
    print(f"[SUCCESS] Prerequisites check:")
    print(f"   Valid: {prereq_result['valid']}")
    if prereq_result.get('errors'):
        print(f"   Errors: {prereq_result['errors']}")
    if prereq_result.get('warnings'):
        print(f"   Warnings: {prereq_result['warnings']}")
    
    # Check current status
    status_result = stage.get_face_processing_status()
    if status_result['success']:
        stage_info = status_result['stage_status']
        print(f"[SUCCESS] Current status: {stage_info.get('status', 'unknown')}")
    else:
        print(f"[ERROR] Status check failed: {status_result['error']}")
    
    # Process face processing if prerequisites are met
    if prereq_result['valid']:
        print("\nTarget: Starting face processing...")
        result = stage.process_face_processing()
        
        if result['success']:
            print("[SUCCESS] Face processing completed successfully!")
            print(f"   Output: {result['output_path']}")
            print(f"   Processing time: {result['processing_time']:.1f}s")
        else:
            print(f"[ERROR] Face processing failed: {result['error']}")
    else:
        print("\n[ERROR] Prerequisites not met - skipping processing")
    
    print("\nSUCCESS Enhanced Face Processing Stage testing completed!")


if __name__ == "__main__":
    main()