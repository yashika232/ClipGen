#!/usr/bin/env python3
"""
Stage 2: Face Analysis
Analyzes video frames for face landmarks and 3D face coefficients.
"""

import argparse
import logging
import sys
import json
import time
from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp
from typing import List, Dict, Any, Tuple, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

# Import only what we need to avoid dependency issues
from utils.logger import setup_logger, TimedLogger
from utils.file_utils import ensure_directory, save_json

class FaceAnalyzer:
    """Face analysis pipeline for landmarks and 3D coefficients."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the face analyzer.
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = setup_logger("stage2_face")
        
        # Load configuration
        if config_path and Path(config_path).exists():
            from utils.file_utils import load_json
            self.config = load_json(config_path)
        else:
            self.config = self._get_default_config()
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Create face mesh detector
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=self.config["mediapipe"]["max_num_faces"],
            refine_landmarks=self.config["mediapipe"]["refine_landmarks"],
            min_detection_confidence=self.config["mediapipe"]["min_detection_confidence"],
            min_tracking_confidence=self.config["mediapipe"]["min_tracking_confidence"]
        )
        
        # Initialize face detection for backup
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 for short-range (< 2 meters), 1 for full-range
            min_detection_confidence=0.5
        )
        
        self.logger.info("Face analyzer initialized successfully")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "mediapipe": {
                "max_num_faces": 1,
                "refine_landmarks": True,
                "min_detection_confidence": 0.5,
                "min_tracking_confidence": 0.5,
                "static_image_mode": False
            },
            "face_analysis": {
                "target_landmarks": 468,  # MediaPipe face mesh landmarks
                "enable_3d_coefficients": True,
                "normalize_coordinates": True,
                "smooth_landmarks": True
            },
            "processing": {
                "batch_size": 32,
                "progress_interval": 30,
                "skip_frames_without_face": False
            },
            "output": {
                "save_metadata": True,
                "save_debug_images": False
            }
        }
    
    def load_video(self, video_path: str) -> cv2.VideoCapture:
        """
        Load video file and return capture object.
        
        Args:
            video_path: Path to video file
            
        Returns:
            OpenCV VideoCapture object
            
        Raises:
            ValueError: If video cannot be opened
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.logger.info(f"Loading video from: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.logger.info(f"Video loaded: {frame_count} frames at {fps:.2f} FPS ({width}x{height})")
        
        return cap
    
    def extract_face_landmarks(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Extract face landmarks from a single frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dictionary containing landmark data
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe Face Mesh
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]  # Take first face
            
            # Extract landmark coordinates
            landmarks = []
            for landmark in face_landmarks.landmark:
                # Convert normalized coordinates to pixel coordinates
                h, w, _ = frame.shape
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                z = landmark.z  # Relative depth
                
                landmarks.append({
                    'x': landmark.x,  # Normalized (0-1)
                    'y': landmark.y,  # Normalized (0-1)
                    'z': landmark.z,  # Relative depth
                    'x_px': x,        # Pixel coordinates
                    'y_px': y         # Pixel coordinates
                })
            
            # Try to detect face bounding box as well
            face_bbox = None
            detection_results = self.face_detection.process(rgb_frame)
            if detection_results.detections:
                detection = detection_results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                face_bbox = {
                    'x': int(bbox.xmin * w),
                    'y': int(bbox.ymin * h),
                    'width': int(bbox.width * w),
                    'height': int(bbox.height * h),
                    'confidence': detection.score[0]
                }
            
            return {
                'detected': True,
                'landmarks': landmarks,
                'num_landmarks': len(landmarks),
                'face_bbox': face_bbox,
                'confidence': face_bbox['confidence'] if face_bbox else 1.0
            }
        else:
            # Try fallback face detection
            detection_results = self.face_detection.process(rgb_frame)
            face_bbox = None
            
            if detection_results.detections:
                detection = detection_results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                face_bbox = {
                    'x': int(bbox.xmin * w),
                    'y': int(bbox.ymin * h),
                    'width': int(bbox.width * w),
                    'height': int(bbox.height * h),
                    'confidence': detection.score[0]
                }
            
            return {
                'detected': False,
                'landmarks': [],
                'num_landmarks': 0,
                'face_bbox': face_bbox,
                'confidence': face_bbox['confidence'] if face_bbox else 0.0
            }
    
    def compute_3d_coefficients(self, landmarks: List[Dict]) -> np.ndarray:
        """
        Compute 3D face coefficients from landmarks.
        
        Args:
            landmarks: List of landmark dictionaries
            
        Returns:
            3D coefficients array
        """
        if not landmarks:
            return np.zeros((468, 3))  # MediaPipe face mesh has 468 landmarks
        
        # Convert landmarks to numpy array
        points = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks])
        
        # Ensure we have exactly 468 landmarks (MediaPipe standard)
        target_landmarks = self.config["face_analysis"]["target_landmarks"]
        
        if len(points) > target_landmarks:
            points = points[:target_landmarks]
        elif len(points) < target_landmarks:
            # Pad with zeros if fewer landmarks
            padding = np.zeros((target_landmarks - len(points), 3))
            points = np.vstack([points, padding])
        
        return points
    
    def smooth_landmarks(self, landmarks_sequence: List[np.ndarray],
                        alpha: float = 0.7) -> List[np.ndarray]:
        """
        Apply temporal smoothing to landmark sequences.
        
        Args:
            landmarks_sequence: List of landmark arrays
            alpha: Smoothing factor (0-1, higher = more smoothing)
            
        Returns:
            Smoothed landmark sequence
        """
        if len(landmarks_sequence) < 2:
            return landmarks_sequence
        
        smoothed = [landmarks_sequence[0]]  # First frame unchanged
        
        for i in range(1, len(landmarks_sequence)):
            # Exponential moving average
            current = landmarks_sequence[i]
            previous = smoothed[i-1]
            
            # Only smooth if both frames have valid landmarks
            if current.size > 0 and previous.size > 0:
                smoothed_frame = alpha * previous + (1 - alpha) * current
            else:
                smoothed_frame = current
            
            smoothed.append(smoothed_frame)
        
        return smoothed
    
    def save_debug_frame(self, frame: np.ndarray, landmarks_data: Dict[str, Any],
                        frame_idx: int, output_dir: Path) -> None:
        """Save debug frame with landmarks visualization."""
        if not self.config["output"]["save_debug_images"]:
            return
        
        debug_frame = frame.copy()
        
        # Draw face bounding box if available
        if landmarks_data.get('face_bbox'):
            bbox = landmarks_data['face_bbox']
            cv2.rectangle(debug_frame,
                         (bbox['x'], bbox['y']),
                         (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']),
                         (0, 255, 0), 2)
            
            # Add confidence text
            conf_text = f"Conf: {bbox['confidence']:.2f}"
            cv2.putText(debug_frame, conf_text,
                       (bbox['x'], bbox['y'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw landmarks if detected
        if landmarks_data['detected'] and landmarks_data['landmarks']:
            for landmark in landmarks_data['landmarks']:
                x, y = landmark['x_px'], landmark['y_px']
                cv2.circle(debug_frame, (x, y), 1, (0, 0, 255), -1)
        
        # Save debug frame
        debug_dir = ensure_directory(output_dir / "debug_frames")
        debug_path = debug_dir / f"frame_{frame_idx:06d}_debug.jpg"
        cv2.imwrite(str(debug_path), debug_frame)
    
    def process_video(self, input_path: str, output_dir: str) -> bool:
        """
        Main video processing function.
        
        Args:
            input_path: Path to input video
            output_dir: Output directory for results
            
        Returns:
            True if processing successful, False otherwise
        """
        try:
            self.logger.info("="*50)
            self.logger.info("Starting Face Analysis Pipeline")
            self.logger.info("="*50)
            
            start_time = time.time()
            output_path = ensure_directory(output_dir)
            
            # Load video
            cap = self.load_video(input_path)
            
            frame_landmarks = []
            frame_coefficients = []
            frame_count = 0
            faces_detected = 0
            
            with TimedLogger(self.logger, "video frame processing"):
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Extract landmarks for this frame
                    landmarks_data = self.extract_face_landmarks(frame)
                    frame_landmarks.append(landmarks_data)
                    
                    # Count successful detections
                    if landmarks_data['detected']:
                        faces_detected += 1
                    
                    # Compute 3D coefficients
                    coeffs = self.compute_3d_coefficients(landmarks_data['landmarks'])
                    frame_coefficients.append(coeffs)
                    
                    # Save debug frame if enabled
                    self.save_debug_frame(frame, landmarks_data, frame_count, output_path)
                    
                    frame_count += 1
                    
                    # Log progress
                    if frame_count % self.config["processing"]["progress_interval"] == 0:
                        detection_rate = faces_detected / frame_count * 100
                        self.logger.info(f"Processed {frame_count} frames... "
                                       f"(Detection rate: {detection_rate:.1f}%)")
            
            cap.release()
            
            self.logger.info(f"Processed {frame_count} frames total")
            self.logger.info(f"Faces detected in {faces_detected}/{frame_count} frames "
                           f"({faces_detected/frame_count*100:.1f}%)")
            
            # Apply smoothing if enabled
            if self.config["face_analysis"]["smooth_landmarks"] and frame_coefficients:
                with TimedLogger(self.logger, "landmark smoothing"):
                    frame_coefficients = self.smooth_landmarks(frame_coefficients)
            
            # Save outputs
            self.save_outputs(output_path, frame_landmarks, frame_coefficients,
                            input_path, frame_count, faces_detected)
            
            # Final summary
            total_time = time.time() - start_time
            self.logger.info("="*50)
            self.logger.info("Face Analysis Pipeline Completed Successfully")
            self.logger.info(f"Total processing time: {total_time:.2f}s")
            self.logger.info(f"Average FPS: {frame_count/total_time:.2f}")
            self.logger.info("="*50)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Face analysis failed: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def save_outputs(self, output_path: Path, frame_landmarks: List[Dict],
                    frame_coefficients: List[np.ndarray], input_path: str,
                    frame_count: int, faces_detected: int) -> Dict[str, str]:
        """Save all outputs to specified directory."""
        
        # Save landmarks
        landmarks_path = output_path / "landmarks.json"
        save_json(frame_landmarks, landmarks_path)
        
        # Save 3D coefficients
        coeffs_array = np.array(frame_coefficients)
        coeffs_path = output_path / "coeffs.npy"
        np.save(coeffs_path, coeffs_array)
        
        # Save metadata
        metadata = {
            "input_video_path": str(input_path),
            "processing_timestamp": time.time(),
            "total_frames": frame_count,
            "faces_detected": faces_detected,
            "detection_rate": faces_detected / frame_count if frame_count > 0 else 0,
            "coefficients_shape": list(coeffs_array.shape),
            "landmarks_count": len(frame_landmarks),
            "config": self.config
        }
        
        if self.config["output"]["save_metadata"]:
            metadata_path = output_path / "face_metadata.json"
            save_json(metadata, metadata_path)
        
        output_files = {
            "landmarks": str(landmarks_path),
            "coefficients": str(coeffs_path),
            "metadata": str(metadata_path) if self.config["output"]["save_metadata"] else None
        }
        
        self.logger.info(f"Outputs saved to: {output_path}")
        for key, path in output_files.items():
            if path:
                file_size = Path(path).stat().st_size
                self.logger.info(f"  {key}: {Path(path).name} ({file_size:,} bytes)")
        
        return output_files


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Stage 2: Face Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stage2_face.py --input-video input.mp4
  python stage2_face.py --input-video input.mp4 --output-dir /path/to/output
  python stage2_face.py --input-video input.mp4 --debug-images
        """
    )
    
    parser.add_argument(
        "--input-video",
        type=str,
        required=True,
        help="Path to input video file (MP4, AVI, etc.)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "stage2"),
        help="Output directory (default: outputs/stage2/)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "face_config.json"),
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--debug-images",
        action="store_true",
        help="Save debug images with landmark visualization"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input file
    if not Path(args.input_video).exists():
        print(f"Error: Input video file not found: {args.input_video}")
        return 1
    
    # Update config for debug images
    config_path = args.config if Path(args.config).exists() else None
    
    try:
        analyzer = FaceAnalyzer(config_path=config_path)
        
        # Enable debug images if requested
        if args.debug_images:
            analyzer.config["output"]["save_debug_images"] = True
        
        success = analyzer.process_video(args.input_video, args.output_dir)
        return 0 if success else 1
        
    except Exception as e:
        print(f"Failed to create face analyzer: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
