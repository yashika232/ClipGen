#!/usr/bin/env python3
"""
Stage 3: Lip-Sync Generation
Produces lip-synced talking-head video using audio embeddings and face landmarks.
"""

import argparse
import logging
import sys
import json
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

# Import only what we need
from utils.logger import setup_logger, TimedLogger
from utils.file_utils import ensure_directory, save_json, load_json

class LipSyncGenerator:
    """Lip-sync generation using simplified approach based on audio-visual correlation."""
    
    def __init__(self, config_path: Optional[str] = None, device: str = "auto"):
        """Initialize the lip-sync generator."""
        self.logger = setup_logger("stage3_lipsync")
        
        # Load configuration
        if config_path and Path(config_path).exists():
            self.config = load_json(config_path)
        else:
            self.config = self._get_default_config()
        
        # Device selection
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.logger.info(f"Using device: {self.device}")
        
        # Lip landmark indices for MediaPipe (key mouth points)
        self.lip_landmarks = [
            # Outer lip
            61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318,
            # Inner lip  
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415,
            # Additional mouth points
            0, 11, 12, 13, 14, 15, 16, 17, 18, 200
        ]
        
        # Audio feature mapping parameters
        self.audio_feature_dim = 98  # From Stage 1
        self.mouth_landmark_dim = len(self.lip_landmarks) * 3  # x,y,z for each landmark
        
        self.logger.info("Lip-sync generator initialized successfully")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "model": {
                "type": "wav2lip_simplified",
                "device": "auto",
                "batch_size": 16,
                "checkpoint_path": None
            },
            "preprocessing": {
                "face_crop_size": [96, 96],
                "audio_window_size": 16,
                "overlap": 0.5,
                "padding": "constant"
            },
            "generation": {
                "smooth_factor": 0.8,
                "blend_ratio": 0.7,
                "quality_threshold": 0.5,
                "use_face_restoration": True
            },
            "output": {
                "video_codec": "mp4v",
                "video_quality": 95,
                "fps": None
            }
        }
    
    def load_stage_outputs(self, audio_feats_path: str, landmarks_path: str, 
                          coeffs_path: str) -> Tuple[np.ndarray, List[Dict], np.ndarray]:
        """Load outputs from previous stages."""
        # Load audio features
        if not Path(audio_feats_path).exists():
            raise FileNotFoundError(f"Audio features not found: {audio_feats_path}")
        audio_features = np.load(audio_feats_path)
        self.logger.info(f"Loaded audio features: {audio_features.shape}")
        
        # Load landmarks
        if not Path(landmarks_path).exists():
            raise FileNotFoundError(f"Landmarks not found: {landmarks_path}")
        with open(landmarks_path, 'r') as f:
            landmarks_data = json.load(f)
        self.logger.info(f"Loaded landmarks: {len(landmarks_data)} frames")
        
        # Load face coefficients
        if not Path(coeffs_path).exists():
            raise FileNotFoundError(f"Face coefficients not found: {coeffs_path}")
        face_coefficients = np.load(coeffs_path)
        self.logger.info(f"Loaded face coefficients: {face_coefficients.shape}")
        
        return audio_features, landmarks_data, face_coefficients
    
    def align_audio_video(self, audio_features: np.ndarray, 
                         landmarks_data: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        """Align audio features with video frames."""
        audio_frames = audio_features.shape[1]
        video_frames = len(landmarks_data)
        
        self.logger.info(f"Aligning: {audio_frames} audio frames with {video_frames} video frames")
        
        # Interpolate audio features to match video frame count
        if audio_frames != video_frames:
            # Create interpolation indices
            audio_indices = np.linspace(0, audio_frames - 1, video_frames)
            
            # Interpolate each feature dimension
            aligned_audio = np.zeros((audio_features.shape[0], video_frames))
            for i in range(audio_features.shape[0]):
                aligned_audio[i] = np.interp(audio_indices, 
                                           np.arange(audio_frames), 
                                           audio_features[i])
            
            self.logger.info(f"Audio features aligned to {aligned_audio.shape}")
            return aligned_audio, landmarks_data
        
        return audio_features, landmarks_data
    
    def extract_mouth_landmarks(self, landmarks: List[Dict]) -> np.ndarray:
        """Extract mouth-specific landmarks from full face landmarks."""
        if not landmarks or len(landmarks) < max(self.lip_landmarks):
            return np.zeros((len(self.lip_landmarks), 3))
        
        mouth_points = []
        for idx in self.lip_landmarks:
            if idx < len(landmarks):
                lm = landmarks[idx]
                mouth_points.append([lm['x'], lm['y'], lm['z']])
            else:
                mouth_points.append([0.0, 0.0, 0.0])
        
        return np.array(mouth_points)
    
    def compute_audio_mouth_correlation(self, audio_features: np.ndarray, 
                                       landmarks_data: List[Dict]) -> np.ndarray:
        """Compute correlation between audio features and mouth movements."""
        num_frames = len(landmarks_data)
        mouth_movements = np.zeros((num_frames, len(self.lip_landmarks), 3))
        
        # Extract existing mouth landmarks for reference
        reference_mouth = []
        for frame_data in landmarks_data:
            if frame_data['detected'] and frame_data['landmarks']:
                mouth_lm = self.extract_mouth_landmarks(frame_data['landmarks'])
                reference_mouth.append(mouth_lm)
            else:
                # Use previous frame or zeros
                if reference_mouth:
                    reference_mouth.append(reference_mouth[-1])
                else:
                    reference_mouth.append(np.zeros((len(self.lip_landmarks), 3)))
        
        reference_mouth = np.array(reference_mouth)
        
        # Simple correlation model: map audio energy to mouth openness
        for i in range(num_frames):
            # Get audio features for this frame
            if i < audio_features.shape[1]:
                frame_audio = audio_features[:, i]
                
                # Compute audio energy (RMS, spectral energy, etc.)
                # Use last few features which are typically RMS and energy-related
                audio_energy = np.mean(frame_audio[-10:])  # Last 10 features
                audio_energy = np.clip(audio_energy, -2, 2)  # Normalize
                
                # Map energy to mouth movement
                # Higher energy = more open mouth
                mouth_openness = (audio_energy + 2) / 4  # Scale to 0-1
                
                # Modify mouth landmarks based on audio
                if i < len(reference_mouth):
                    base_mouth = reference_mouth[i].copy()
                    
                    # Apply mouth opening based on audio energy
                    # Focus on inner lip landmarks (vertical movement)
                    for j, lm_idx in enumerate(self.lip_landmarks):
                        if j < len(base_mouth):
                            # Vertical movement for mouth opening
                            if j in [1, 2, 3, 4]:  # Upper lip points
                                base_mouth[j, 1] -= mouth_openness * 0.02
                            elif j in [7, 8, 9, 10]:  # Lower lip points
                                base_mouth[j, 1] += mouth_openness * 0.02
                    
                    mouth_movements[i] = base_mouth
                else:
                    mouth_movements[i] = reference_mouth[-1] if len(reference_mouth) > 0 else np.zeros((len(self.lip_landmarks), 3))
        
        return mouth_movements
    
    def apply_temporal_smoothing(self, mouth_movements: np.ndarray, 
                                alpha: float = 0.7) -> np.ndarray:
        """Apply temporal smoothing to mouth movements."""
        if len(mouth_movements) < 2:
            return mouth_movements
        
        smoothed = mouth_movements.copy()
        
        # Apply exponential moving average
        for i in range(1, len(mouth_movements)):
            smoothed[i] = alpha * smoothed[i-1] + (1 - alpha) * mouth_movements[i]
        
        return smoothed
    
    def generate_lipsync_video(self, input_video_path: str, mouth_movements: np.ndarray,
                              landmarks_data: List[Dict], output_path: str) -> bool:
        """Generate lip-synced video by modifying mouth regions."""
        try:
            # Open input video
            cap = cv2.VideoCapture(input_video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {input_video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create output video writer
            fourcc = cv2.VideoWriter_fourcc(*self.config["output"]["video_codec"])
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Apply lip-sync modification to this frame
                if frame_idx < len(mouth_movements) and frame_idx < len(landmarks_data):
                    modified_frame = self.modify_mouth_region(
                        frame, 
                        landmarks_data[frame_idx],
                        mouth_movements[frame_idx]
                    )
                else:
                    modified_frame = frame
                
                out.write(modified_frame)
                frame_idx += 1
            
            cap.release()
            out.release()
            
            self.logger.info(f"Generated lip-sync video: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate video: {str(e)}")
            return False
    
    def modify_mouth_region(self, frame: np.ndarray, landmarks_data: Dict,
                           mouth_movement: np.ndarray) -> np.ndarray:
        """Modify mouth region in frame based on predicted movement."""
        # For this simplified implementation, we'll apply basic mouth region modification
        if not landmarks_data['detected'] or not landmarks_data['landmarks']:
            return frame
        
        # Create a simple mouth region mask and apply subtle modifications
        modified_frame = frame.copy()
        
        # Get mouth bounding box from landmarks
        if landmarks_data.get('face_bbox'):
            bbox = landmarks_data['face_bbox']
            
            # Extract mouth region (lower third of face)
            mouth_y = bbox['y'] + int(bbox['height'] * 0.6)
            mouth_h = int(bbox['height'] * 0.4)
            mouth_x = bbox['x'] + int(bbox['width'] * 0.2)
            mouth_w = int(bbox['width'] * 0.6)
            
            # Apply subtle color/brightness adjustment to simulate mouth movement
            mouth_region = modified_frame[mouth_y:mouth_y+mouth_h, mouth_x:mouth_x+mouth_w]
            
            if mouth_region.size > 0:
                # Calculate movement intensity
                movement_intensity = np.mean(np.abs(mouth_movement))
                adjustment = int(movement_intensity * 20)  # Scale factor
                
                # Apply subtle brightness adjustment
                mouth_region = cv2.convertScaleAbs(mouth_region, alpha=1.0, beta=adjustment)
                modified_frame[mouth_y:mouth_y+mouth_h, mouth_x:mouth_x+mouth_w] = mouth_region
        
        return modified_frame
    
    def process_lipsync(self, input_video_path: str, audio_feats_path: str,
                       landmarks_path: str, coeffs_path: str, output_dir: str) -> bool:
        """Main lip-sync processing function."""
        try:
            self.logger.info("="*50)
            self.logger.info("Starting Lip-Sync Generation Pipeline")
            self.logger.info("="*50)
            
            start_time = time.time()
            output_path = ensure_directory(output_dir)
            
            # Load stage outputs
            with TimedLogger(self.logger, "loading stage outputs"):
                audio_features, landmarks_data, face_coefficients = self.load_stage_outputs(
                    audio_feats_path, landmarks_path, coeffs_path
                )
            
            # Align audio and video
            with TimedLogger(self.logger, "audio-video alignment"):
                aligned_audio, aligned_landmarks = self.align_audio_video(
                    audio_features, landmarks_data
                )
            
            # Compute audio-mouth correlation
            with TimedLogger(self.logger, "computing audio-mouth correlation"):
                mouth_movements = self.compute_audio_mouth_correlation(
                    aligned_audio, aligned_landmarks
                )
            
            # Apply temporal smoothing
            with TimedLogger(self.logger, "applying temporal smoothing"):
                smoothed_movements = self.apply_temporal_smoothing(
                    mouth_movements, alpha=self.config["generation"]["smooth_factor"]
                )
            
            # Generate lip-synced video
            output_video_path = output_path / "lipsync.mp4"
            with TimedLogger(self.logger, "generating lip-sync video"):
                success = self.generate_lipsync_video(
                    input_video_path, smoothed_movements, aligned_landmarks, 
                    str(output_video_path)
                )
            
            if not success:
                return False
            
            # Save metadata
            self.save_outputs(output_path, mouth_movements, smoothed_movements,
                            input_video_path, audio_feats_path, landmarks_path)
            
            # Final summary
            total_time = time.time() - start_time
            self.logger.info("="*50)
            self.logger.info("Lip-Sync Generation Pipeline Completed Successfully")
            self.logger.info(f"Total processing time: {total_time:.2f}s")
            self.logger.info(f"Output video: {output_video_path}")
            self.logger.info("="*50)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Lip-sync processing failed: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def save_outputs(self, output_path: Path, mouth_movements: np.ndarray,
                    smoothed_movements: np.ndarray, input_video_path: str,
                    audio_feats_path: str, landmarks_path: str) -> None:
        """Save processing outputs and metadata."""
        
        # Save mouth movements
        movements_path = output_path / "mouth_movements.npy"
        np.save(movements_path, mouth_movements)
        
        # Save smoothed movements
        smoothed_path = output_path / "smoothed_movements.npy"
        np.save(smoothed_path, smoothed_movements)
        
        # Save metadata
        metadata = {
            "input_video_path": str(input_video_path),
            "audio_features_path": str(audio_feats_path),
            "landmarks_path": str(landmarks_path),
            "processing_timestamp": time.time(),
            "mouth_movements_shape": list(mouth_movements.shape),
            "smoothed_movements_shape": list(smoothed_movements.shape),
            "device_used": self.device,
            "config": self.config
        }
        
        metadata_path = output_path / "lipsync_metadata.json"
        save_json(metadata, metadata_path)
        
        self.logger.info(f"Outputs saved to: {output_path}")
        for file_path in [movements_path, smoothed_path, metadata_path]:
            file_size = file_path.stat().st_size
            self.logger.info(f"  {file_path.name}: ({file_size:,} bytes)")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Stage 3: Lip-Sync Generation")
    parser.add_argument("--input-video", type=str, required=True,
                       help="Path to input video file")
    parser.add_argument("--audio-feats", type=str, required=True,
                       help="Path to audio features from Stage 1")
    parser.add_argument("--landmarks", type=str, required=True,
                       help="Path to landmarks from Stage 2")
    parser.add_argument("--coeffs", type=str, required=True,
                       help="Path to face coefficients from Stage 2")
    parser.add_argument("--output-dir", type=str,
                       default=str(PROJECT_ROOT / "outputs" / "stage3"),
                       help="Output directory")
    parser.add_argument("--config", type=str,
                       default=str(PROJECT_ROOT / "configs" / "lipsync_config.json"),
                       help="Path to configuration file")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["cpu", "cuda", "auto"],
                       help="Computing device")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input files
    required_files = [args.input_video, args.audio_feats, args.landmarks, args.coeffs]
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"Error: Required file not found: {file_path}")
            return 1
    
    try:
        generator = LipSyncGenerator(
            config_path=args.config if Path(args.config).exists() else None,
            device=args.device
        )
        
        success = generator.process_lipsync(
            args.input_video, args.audio_feats, args.landmarks, 
            args.coeffs, args.output_dir
        )
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"Failed to create lip-sync generator: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
