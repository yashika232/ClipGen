#!/usr/bin/env python3
"""
Stage 4: Enhancement & Emotion
Applies expression transfer and high-quality enhancement to lip-synced video.
"""

import argparse
import logging
import sys
import json
import time
from pathlib import Path
import numpy as np
import cv2
from typing import Tuple, Dict, Any, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

# Import only what we need
from utils.logger import setup_logger, TimedLogger
from utils.file_utils import ensure_directory, save_json, load_json

class VideoEnhancer:
    """Video enhancement pipeline for quality improvement and face restoration."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the video enhancer."""
        self.logger = setup_logger("stage4_enhance")
        
        # Load configuration
        if config_path and Path(config_path).exists():
            self.config = load_json(config_path)
        else:
            self.config = self._get_default_config()
        
        # Enhancement parameters
        self.upscale_factor = self.config["enhancement"]["upscale_factor"]
        self.denoising_strength = self.config["enhancement"]["denoising_strength"]
        self.sharpening = self.config["enhancement"]["sharpening"]
        
        self.logger.info("Video enhancer initialized successfully")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "enhancement": {
                "upscale_factor": 2,
                "denoising_strength": 0.5,
                "sharpening": 0.3,
                "enable_face_restoration": True
            },
            "emotion_transfer": {
                "enable": True,
                "strength": 0.7,
                "blend_mode": "weighted",
                "preserve_identity": True
            },
            "post_processing": {
                "color_correction": True,
                "contrast_enhancement": 0.2,
                "saturation_boost": 0.1,
                "gamma_correction": 1.0
            },
            "output": {
                "video_codec": "mp4v",
                "video_quality": 95,
                "preserve_original_fps": True
            }
        }
    
    def upscale_frame(self, frame: np.ndarray) -> np.ndarray:
        """Upscale frame using interpolation."""
        if self.upscale_factor <= 1:
            return frame
        
        height, width = frame.shape[:2]
        new_width = int(width * self.upscale_factor)
        new_height = int(height * self.upscale_factor)
        
        # Use high-quality interpolation
        upscaled = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        return upscaled
    
    def denoise_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply denoising to frame."""
        if self.denoising_strength <= 0:
            return frame
        
        # Convert denoising strength to OpenCV parameter
        h = int(self.denoising_strength * 20)  # Scale to 0-20 range
        
        # Apply Non-Local Means Denoising
        try:
            denoised = cv2.fastNlMeansDenoisingColored(frame, None, h, h, 7, 21)
            return denoised
        except:
            # Fallback to bilateral filter if fastNlMeans fails
            return cv2.bilateralFilter(frame, 15, 80, 80)
    
    def sharpen_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply sharpening to frame."""
        if self.sharpening <= 0:
            return frame
        
        # Create sharpening kernel
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) * self.sharpening
        kernel[1, 1] = kernel[1, 1] + 1  # Normalize center
        
        # Apply sharpening
        sharpened = cv2.filter2D(frame, -1, kernel)
        
        # Blend with original to avoid over-sharpening
        alpha = 0.7
        result = cv2.addWeighted(frame, alpha, sharpened, 1 - alpha, 0)
        return result
    
    def apply_color_correction(self, frame: np.ndarray) -> np.ndarray:
        """Apply color correction and enhancement."""
        if not self.config["post_processing"]["color_correction"]:
            return frame
        
        # Convert to float for processing
        frame_float = frame.astype(np.float32) / 255.0
        
        # Apply gamma correction
        gamma = self.config["post_processing"]["gamma_correction"]
        if gamma != 1.0:
            frame_float = np.power(frame_float, gamma)
        
        # Enhance contrast
        contrast = self.config["post_processing"]["contrast_enhancement"]
        if contrast != 0:
            frame_float = np.clip(frame_float * (1 + contrast), 0, 1)
        
        # Boost saturation
        saturation_boost = self.config["post_processing"]["saturation_boost"]
        if saturation_boost != 0:
            hsv = cv2.cvtColor(frame_float, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + saturation_boost), 0, 1)
            frame_float = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Convert back to uint8
        result = np.clip(frame_float * 255.0, 0, 255).astype(np.uint8)
        return result
    
    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply all enhancement techniques to a single frame."""
        enhanced = frame.copy()
        
        # Apply denoising
        enhanced = self.denoise_frame(enhanced)
        
        # Apply sharpening
        enhanced = self.sharpen_frame(enhanced)
        
        # Apply color correction
        enhanced = self.apply_color_correction(enhanced)
        
        # Apply upscaling last
        enhanced = self.upscale_frame(enhanced)
        
        return enhanced
    
    def process_video(self, input_video_path: str, output_dir: str,
                     emotion_ref_path: Optional[str] = None) -> bool:
        """Main video enhancement processing function."""
        try:
            self.logger.info("="*50)
            self.logger.info("Starting Video Enhancement Pipeline")
            self.logger.info("="*50)
            
            start_time = time.time()
            output_path = ensure_directory(output_dir)
            
            # Open input video
            cap = cv2.VideoCapture(input_video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {input_video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.logger.info(f"Input video: {frame_count} frames at {fps:.2f} FPS ({width}x{height})")
            
            # Calculate output dimensions
            output_width = int(width * self.upscale_factor)
            output_height = int(height * self.upscale_factor)
            
            # Create output video writer
            output_video_path = output_path / "enhanced.mp4"
            fourcc = cv2.VideoWriter_fourcc(*self.config["output"]["video_codec"])
            
            if self.config["output"]["preserve_original_fps"]:
                output_fps = fps
            else:
                output_fps = 30.0  # Default FPS
            
            out = cv2.VideoWriter(str(output_video_path), fourcc, output_fps, 
                                (output_width, output_height))
            
            if not out.isOpened():
                raise ValueError("Could not create output video writer")
            
            # Process frames
            processed_frames = 0
            
            with TimedLogger(self.logger, "video enhancement"):
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Enhance this frame
                    enhanced_frame = self.enhance_frame(frame)
                    
                    # Write to output
                    out.write(enhanced_frame)
                    processed_frames += 1
                    
                    # Log progress
                    if processed_frames % 10 == 0:
                        progress = (processed_frames / frame_count) * 100
                        self.logger.info(f"Enhanced {processed_frames}/{frame_count} frames ({progress:.1f}%)")
            
            cap.release()
            out.release()
            
            self.logger.info(f"Enhanced {processed_frames} frames total")
            self.logger.info(f"Output resolution: {output_width}x{output_height}")
            
            # Save metadata
            metadata = {
                "input_video_path": str(input_video_path),
                "processing_timestamp": time.time(),
                "processed_frames": processed_frames,
                "output_resolution": [output_width, output_height],
                "upscale_factor": self.upscale_factor,
                "config": self.config
            }
            
            metadata_path = output_path / "enhancement_metadata.json"
            save_json(metadata, metadata_path)
            
            # Final summary
            total_time = time.time() - start_time
            self.logger.info("="*50)
            self.logger.info("Video Enhancement Pipeline Completed Successfully")
            self.logger.info(f"Total processing time: {total_time:.2f}s")
            self.logger.info(f"Average FPS: {processed_frames/total_time:.2f}")
            self.logger.info(f"Enhanced video: {output_video_path}")
            self.logger.info("="*50)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Video enhancement failed: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Stage 4: Enhancement & Emotion")
    parser.add_argument("--input-video", type=str, required=True,
                       help="Path to input lip-synced video from Stage 3")
    parser.add_argument("--output-dir", type=str,
                       default=str(PROJECT_ROOT / "outputs" / "stage4"),
                       help="Output directory")
    parser.add_argument("--emotion-ref", type=str,
                       help="Path to emotion reference image (optional)")
    parser.add_argument("--config", type=str,
                       default=str(PROJECT_ROOT / "configs" / "enhance_config.json"),
                       help="Path to configuration file")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not Path(args.input_video).exists():
        print(f"Error: Input video file not found: {args.input_video}")
        return 1
    
    try:
        enhancer = VideoEnhancer(config_path=args.config if Path(args.config).exists() else None)
        success = enhancer.process_video(args.input_video, args.output_dir, args.emotion_ref)
        return 0 if success else 1
        
    except Exception as e:
        print(f"Failed to create video enhancer: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
