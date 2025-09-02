#!/usr/bin/env python3
"""
Real-ESRGAN Integration Module
High-quality upscaling using Real-ESRGAN for video enhancement.
This module provides MPS-optimized Real-ESRGAN functionality for 1080p upscaling.
"""

import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import tempfile
import subprocess
import logging
from PIL import Image

class RealESRGANUpscaler:
    """
    Real-ESRGAN-based upscaling wrapper with MPS optimization.
    Provides high-quality 2x upscaling for 1080p output.
    """
    
    def __init__(self, 
                 device: str = "auto",
                 model_name: str = "RealESRGAN_x2plus",
                 upscale: int = 2):
        """
        Initialize Real-ESRGAN upscaler.
        
        Args:
            device: Device to use ('cpu', 'cuda', 'mps', 'auto')
            model_name: Real-ESRGAN model name
            upscale: Upscale factor (2 for 1080p)
        """
        self.logger = logging.getLogger("RealESRGANUpscaler")
        self.device = self._setup_device(device)
        self.model_name = model_name
        self.upscale = upscale
        
        # Initialize Real-ESRGAN
        self.upsampler = None
        self.available = self._initialize_realesrgan()
        
        if not self.available:
            self.logger.warning("Real-ESRGAN not available. Will use fallback upscaling.")
    
    def _setup_device(self, device: str) -> str:
        """Setup and validate device with Apple Silicon MPS support."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _initialize_realesrgan(self) -> bool:
        """Initialize Real-ESRGAN model using conda environment."""
        try:
            # Use conda environment for Real-ESRGAN to avoid dependency conflicts
            conda_env_python = "/Users/aryanjain/miniforge3/envs/realesrgan/bin/python"
            
            # Test conda environment availability
            test_cmd = [conda_env_python, "-c", "import torch; from realesrgan import RealESRGANer; print('OK')"]
            result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                self.logger.error(f"Real-ESRGAN conda environment test failed: {result.stderr}")
                return False
            
            # Real-ESRGAN is available in conda environment
            self.conda_python = conda_env_python
            self.logger.info(f"Real-ESRGAN conda environment ready: {conda_env_python}")
            return True
            
        except Exception as e:
            self.logger.error(f"Real-ESRGAN conda environment check failed: {e}")
            return False
    
    def _download_model_weights(self, model_filename: str) -> Optional[str]:
        """Download Real-ESRGAN model weights if needed."""
        try:
            # Try to use existing weights from various locations
            possible_paths = [
                f"weights/{model_filename}",
                f"models/weights/{model_filename}",
                f"~/.cache/realesrgan/{model_filename}",
                f"/tmp/realesrgan/{model_filename}"
            ]
            
            for path in possible_paths:
                full_path = Path(path).expanduser()
                if full_path.exists():
                    self.logger.info(f"Found existing model weights: {full_path}")
                    return str(full_path)
            
            # Download weights using Real-ESRGAN's built-in mechanism
            from realesrgan.utils import RealESRGANer
            
            # Create cache directory
            cache_dir = Path.home() / ".cache" / "realesrgan"
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = cache_dir / model_filename
            
            if not model_path.exists():
                self.logger.info(f"Downloading {model_filename}...")
                # The RealESRGANer will handle automatic download
                return None  # Let RealESRGANer handle the download
            
            return str(model_path)
            
        except Exception as e:
            self.logger.warning(f"Model weight download/detection failed: {e}")
            return None
    
    def upscale_image(self, image: np.ndarray) -> np.ndarray:
        """
        Upscale a single image using conda environment.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Upscaled image as numpy array
        """
        if not self.available:
            return self._fallback_upscale_image(image)
        
        try:
            # Save input image to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_input:
                cv2.imwrite(tmp_input.name, image)
                input_path = tmp_input.name
            
            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_output:
                output_path = tmp_output.name
            
            # Run Real-ESRGAN in conda environment  
            model_path = "/Users/aryanjain/Documents/video-synthesis-pipeline copy/weights/RealESRGAN_x2plus.pth"
            upscale_cmd = [
                self.conda_python, "-c", f"""
import cv2
import numpy as np
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# Initialize model
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale={self.upscale})
upsampler = RealESRGANer(
    scale={self.upscale},
    model_path='{model_path}',
    model=model,
    tile=512,
    tile_pad=10,
    pre_pad=0,
    half=False,
    device='cpu'
)

# Process image
input_img = cv2.imread('{input_path}')
if input_img is None:
    raise ValueError(f'Could not read image: {input_path}')
output_img, _ = upsampler.enhance(input_img, outscale={self.upscale})
cv2.imwrite('{output_path}', output_img)
print(f"Upscaled {{input_img.shape}} to {{output_img.shape}}")
"""
            ]
            
            result = subprocess.run(upscale_cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                # Load upscaled image
                upscaled_image = cv2.imread(output_path)
                
                # Cleanup
                os.unlink(input_path)
                os.unlink(output_path)
                
                if upscaled_image is not None:
                    self.logger.debug(f"Real-ESRGAN upscaled image from {image.shape} to {upscaled_image.shape}")
                    return upscaled_image
                else:
                    self.logger.warning("Failed to load upscaled image")
                    return self._fallback_upscale_image(image)
            else:
                self.logger.warning(f"Real-ESRGAN subprocess failed: {result.stderr}")
                # Cleanup
                if os.path.exists(input_path):
                    os.unlink(input_path)
                if os.path.exists(output_path):
                    os.unlink(output_path)
                return self._fallback_upscale_image(image)
            
        except Exception as e:
            self.logger.warning(f"Real-ESRGAN upscaling failed: {e}")
            return self._fallback_upscale_image(image)
    
    def _fallback_upscale_image(self, image: np.ndarray) -> np.ndarray:
        """Fallback upscaling using OpenCV."""
        try:
            height, width = image.shape[:2]
            new_height, new_width = height * self.upscale, width * self.upscale
            
            # Use LANCZOS for high-quality upscaling
            upscaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            
            # Apply sharpening to improve quality
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1], 
                              [-1, -1, -1]])
            sharpened = cv2.filter2D(upscaled, -1, kernel * 0.1)
            result = cv2.addWeighted(upscaled, 0.8, sharpened, 0.2, 0)
            
            self.logger.debug(f"Fallback upscaled image from {image.shape} to {result.shape}")
            return result
            
        except Exception as e:
            self.logger.error(f"Fallback upscaling failed: {e}")
            return image
    
    def upscale_video(self, input_video_path: str, output_video_path: str, 
                     enable_temporal_consistency: bool = True) -> bool:
        """
        Upscale video frame by frame to 1080p with temporal consistency.
        
        Args:
            input_video_path: Path to input video
            output_video_path: Path to save upscaled video
            enable_temporal_consistency: Enable temporal consistency algorithms
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Upscaling video: {input_video_path}")
            
            # Open input video
            cap = cv2.VideoCapture(input_video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {input_video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate target dimensions for 1080p
            target_height = 1080
            target_width = int((target_height / original_height) * original_width)
            
            # Ensure even dimensions for H.264 compatibility
            if target_width % 2 != 0:
                target_width += 1
            
            self.logger.info(f"Upscaling from {original_width}x{original_height} to {target_width}x{target_height}")
            
            # Setup output video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (target_width, target_height))
            
            if not out.isOpened():
                raise ValueError(f"Could not create output video: {output_video_path}")
            
            # Initialize temporal consistency tracking
            previous_frame = None
            frame_buffer = []
            buffer_size = 3 if enable_temporal_consistency else 1
            
            # Process frames
            processed_frames = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Upscale frame
                upscaled_frame = self.upscale_image(frame)
                
                # Resize to exact target dimensions if needed
                if upscaled_frame.shape[:2] != (target_height, target_width):
                    upscaled_frame = cv2.resize(upscaled_frame, (target_width, target_height), 
                                              interpolation=cv2.INTER_LANCZOS4)
                
                # Apply temporal consistency if enabled
                if enable_temporal_consistency:
                    upscaled_frame = self._apply_temporal_consistency(
                        upscaled_frame, previous_frame, frame_buffer, processed_frames
                    )
                    previous_frame = upscaled_frame.copy()
                
                # Write frame
                out.write(upscaled_frame)
                processed_frames += 1
                
                if processed_frames % 10 == 0:
                    self.logger.info(f"Processed {processed_frames}/{frame_count} frames")
            
            # Cleanup
            cap.release()
            out.release()
            
            self.logger.info(f"Video upscaling completed: {processed_frames} frames processed")
            return True
            
        except Exception as e:
            self.logger.error(f"Video upscaling failed: {str(e)}")
            return False
    
    def _apply_temporal_consistency(self, current_frame: np.ndarray, 
                                  previous_frame: Optional[np.ndarray],
                                  frame_buffer: List[np.ndarray],
                                  frame_index: int) -> np.ndarray:
        """
        Apply temporal consistency to reduce flickering between frames.
        
        Args:
            current_frame: Current upscaled frame
            previous_frame: Previous processed frame
            frame_buffer: Buffer of recent frames
            frame_index: Current frame index
            
        Returns:
            Temporally consistent frame
        """
        if previous_frame is None:
            # First frame - just add to buffer
            frame_buffer.append(current_frame.copy())
            return current_frame
        
        try:
            # Apply temporal smoothing
            smoothed_frame = self._temporal_smoothing(current_frame, previous_frame)
            
            # Apply motion-compensated filtering
            consistent_frame = self._motion_compensated_filtering(
                smoothed_frame, previous_frame, frame_buffer
            )
            
            # Update frame buffer
            frame_buffer.append(consistent_frame.copy())
            if len(frame_buffer) > 5:  # Keep last 5 frames
                frame_buffer.pop(0)
            
            return consistent_frame
            
        except Exception as e:
            self.logger.warning(f"Temporal consistency failed: {e}")
            return current_frame
    
    def _temporal_smoothing(self, current_frame: np.ndarray, 
                          previous_frame: np.ndarray,
                          alpha: float = 0.2) -> np.ndarray:
        """
        Apply temporal smoothing between consecutive frames.
        
        Args:
            current_frame: Current frame
            previous_frame: Previous frame
            alpha: Smoothing factor (0.0 = no smoothing, 1.0 = full smoothing)
            
        Returns:
            Smoothed frame
        """
        try:
            # Ensure frames have same dimensions
            if current_frame.shape != previous_frame.shape:
                previous_frame = cv2.resize(previous_frame, 
                                          (current_frame.shape[1], current_frame.shape[0]))
            
            # Apply exponential moving average
            smoothed = cv2.addWeighted(current_frame, 1.0 - alpha, previous_frame, alpha, 0)
            
            return smoothed
            
        except Exception as e:
            self.logger.warning(f"Temporal smoothing failed: {e}")
            return current_frame
    
    def _motion_compensated_filtering(self, current_frame: np.ndarray,
                                    previous_frame: np.ndarray,
                                    frame_buffer: List[np.ndarray]) -> np.ndarray:
        """
        Apply motion-compensated filtering to reduce temporal artifacts.
        
        Args:
            current_frame: Current frame
            previous_frame: Previous frame
            frame_buffer: Buffer of recent frames
            
        Returns:
            Motion-compensated frame
        """
        try:
            # Convert to grayscale for motion estimation
            gray_curr = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            gray_prev = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowPyrLK(
                gray_prev, gray_curr, None, None,
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            
            # Apply motion-based adaptive filtering
            motion_magnitude = np.mean(np.abs(flow[0])) if flow[0] is not None else 0
            
            # Adjust filtering strength based on motion
            if motion_magnitude < 2.0:  # Low motion - stronger filtering
                filter_strength = 0.3
            elif motion_magnitude < 5.0:  # Medium motion
                filter_strength = 0.15
            else:  # High motion - minimal filtering
                filter_strength = 0.05
            
            # Apply adaptive bilateral filter
            filtered_frame = cv2.bilateralFilter(
                current_frame, 9, 
                sigma_color=75 * filter_strength,
                sigma_space=75 * filter_strength
            )
            
            # Blend with original based on motion
            result = cv2.addWeighted(current_frame, 1.0 - filter_strength, 
                                   filtered_frame, filter_strength, 0)
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Motion compensation failed: {e}")
            return current_frame
    
    def upscale_video_with_batching(self, input_video_path: str, 
                                  output_video_path: str,
                                  batch_size: int = 4,
                                  enable_temporal_consistency: bool = True) -> bool:
        """
        Upscale video using batch processing for improved efficiency.
        
        Args:
            input_video_path: Path to input video
            output_video_path: Path to save upscaled video
            batch_size: Number of frames to process in each batch
            enable_temporal_consistency: Enable temporal consistency
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Batch upscaling video: {input_video_path} (batch_size={batch_size})")
            
            # Open input video
            cap = cv2.VideoCapture(input_video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {input_video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate target dimensions
            target_height = 1080
            target_width = int((target_height / original_height) * original_width)
            if target_width % 2 != 0:
                target_width += 1
            
            # Setup output video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (target_width, target_height))
            
            if not out.isOpened():
                raise ValueError(f"Could not create output video: {output_video_path}")
            
            # Process in batches
            processed_frames = 0
            frame_buffer = []
            previous_frame = None
            
            while True:
                # Read batch of frames
                batch_frames = []
                for _ in range(batch_size):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    batch_frames.append(frame)
                
                if not batch_frames:
                    break
                
                # Process batch
                upscaled_batch = self._process_frame_batch(
                    batch_frames, target_width, target_height
                )
                
                # Apply temporal consistency to batch
                if enable_temporal_consistency:
                    upscaled_batch = self._apply_batch_temporal_consistency(
                        upscaled_batch, previous_frame, frame_buffer
                    )
                    if upscaled_batch:
                        previous_frame = upscaled_batch[-1].copy()
                
                # Write batch
                for frame in upscaled_batch:
                    out.write(frame)
                    processed_frames += 1
                
                if processed_frames % 20 == 0:
                    self.logger.info(f"Batch processed {processed_frames}/{frame_count} frames")
            
            # Cleanup
            cap.release()
            out.release()
            
            self.logger.info(f"Batch video upscaling completed: {processed_frames} frames processed")
            return True
            
        except Exception as e:
            self.logger.error(f"Batch video upscaling failed: {str(e)}")
            return False
    
    def _process_frame_batch(self, frames: List[np.ndarray], 
                           target_width: int, target_height: int) -> List[np.ndarray]:
        """
        Process a batch of frames efficiently.
        
        Args:
            frames: List of input frames
            target_width: Target width
            target_height: Target height
            
        Returns:
            List of upscaled frames
        """
        upscaled_frames = []
        
        for frame in frames:
            # Upscale frame
            upscaled = self.upscale_image(frame)
            
            # Resize to target dimensions
            if upscaled.shape[:2] != (target_height, target_width):
                upscaled = cv2.resize(upscaled, (target_width, target_height),
                                    interpolation=cv2.INTER_LANCZOS4)
            
            upscaled_frames.append(upscaled)
        
        return upscaled_frames
    
    def _apply_batch_temporal_consistency(self, frames: List[np.ndarray],
                                        previous_frame: Optional[np.ndarray],
                                        frame_buffer: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply temporal consistency to a batch of frames.
        
        Args:
            frames: List of frames to process
            previous_frame: Previous frame from last batch
            frame_buffer: Frame buffer for temporal consistency
            
        Returns:
            List of temporally consistent frames
        """
        if not frames:
            return frames
        
        consistent_frames = []
        current_previous = previous_frame
        
        for i, frame in enumerate(frames):
            if current_previous is not None:
                # Apply temporal consistency
                consistent_frame = self._apply_temporal_consistency(
                    frame, current_previous, frame_buffer, i
                )
            else:
                consistent_frame = frame
            
            consistent_frames.append(consistent_frame)
            current_previous = consistent_frame
        
        return consistent_frames
    
    def get_upscaler_info(self) -> Dict[str, Any]:
        """Get information about the upscaler capabilities."""
        return {
            "realesrgan_available": self.available,
            "model_name": self.model_name,
            "device": self.device,
            "upscale_factor": self.upscale,
            "target_resolution": "1080p",
            "fallback_method": "OpenCV LANCZOS with sharpening"
        }


def main():
    """Test the Real-ESRGAN upscaler."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Real-ESRGAN Integration")
    parser.add_argument("--input", required=True, help="Input image/video path")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("--model", default="RealESRGAN_x2plus", help="Model name")
    parser.add_argument("--device", default="auto", help="Device (cpu/cuda/mps/auto)")
    parser.add_argument("--upscale", type=int, default=2, help="Upscale factor")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create upscaler
    upscaler = RealESRGANUpscaler(
        device=args.device,
        model_name=args.model,
        upscale=args.upscale
    )
    
    # Print info
    info = upscaler.get_upscaler_info()
    print("Real-ESRGAN Upscaler Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Determine if input is image or video
    input_path = Path(args.input)
    if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        # Upscale image
        image = cv2.imread(args.input)
        if image is None:
            print(f"Could not read image: {args.input}")
            sys.exit(1)
        
        upscaled = upscaler.upscale_image(image)
        success = cv2.imwrite(args.output, upscaled)
        
        if success:
            print(f"Image upscaling completed: {args.output}")
        else:
            print("Image upscaling failed")
            sys.exit(1)
            
    elif input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        # Upscale video
        success = upscaler.upscale_video(args.input, args.output)
        
        if success:
            print(f"Video upscaling completed: {args.output}")
        else:
            print("Video upscaling failed")
            sys.exit(1)
    else:
        print(f"Unsupported file format: {input_path.suffix}")
        sys.exit(1)


if __name__ == "__main__":
    main()