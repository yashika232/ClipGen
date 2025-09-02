#!/usr/bin/env python3
"""
CodeFormer Integration Module
Advanced face restoration using CodeFormer for high-quality face enhancement.
This module provides a wrapper for CodeFormer functionality.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import tempfile
import subprocess
import logging

# Apply TorchVision compatibility fixes for BasicSR/CodeFormer
def apply_torchvision_compatibility_fixes():
    """Apply TorchVision compatibility fixes for BasicSR and CodeFormer."""
    try:
        # Add utils directory to path
        project_root = Path(__file__).parent.parent.absolute()
        utils_path = project_root / "utils"
        if str(utils_path) not in sys.path:
            sys.path.insert(0, str(utils_path))
        
        from torchvision_compatibility_fix import apply_torchvision_compatibility_fixes as apply_fixes
        return apply_fixes()
    except Exception as e:
        logging.warning(f"[WARNING] Could not apply TorchVision compatibility fixes: {e}")
        return False

# Apply fixes at module import time
apply_torchvision_compatibility_fixes()

class CodeFormerEnhancer:
    """
    CodeFormer-based face enhancement wrapper.
    Provides high-quality face restoration capabilities.
    """
    
    def __init__(self, 
                 codeformer_path: Optional[str] = None,
                 device: str = "auto",
                 background_enhance: bool = True,
                 face_upsample: bool = True,
                 upscale: int = 2):
        """
        Initialize CodeFormer enhancer.
        
        Args:
            codeformer_path: Path to CodeFormer repository/executable
            device: Device to use ('cpu', 'cuda', 'auto')
            background_enhance: Whether to enhance background
            face_upsample: Whether to upsample faces
            upscale: Upscale factor (1, 2, 4)
        """
        self.logger = logging.getLogger("CodeFormerEnhancer")
        self.device = self._setup_device(device)
        self.background_enhance = background_enhance
        self.face_upsample = face_upsample
        self.upscale = upscale
        
        # Try to locate CodeFormer
        self.codeformer_path = self._find_codeformer(codeformer_path)
        self.available = self.codeformer_path is not None
        
        if not self.available:
            self.logger.warning("CodeFormer not found. Face enhancement will use fallback methods.")
    
    def _setup_device(self, device: str) -> str:
        """Setup and validate device with Apple Silicon MPS support."""
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                elif torch.backends.mps.is_available():
                    return "mps"
                else:
                    return "cpu"
            except ImportError:
                return "cpu"
        return device
    
    def _find_codeformer(self, custom_path: Optional[str] = None) -> Optional[Path]:
        """
        Find CodeFormer installation.
        
        Args:
            custom_path: Custom path to CodeFormer
            
        Returns:
            Path to CodeFormer or None if not found
        """
        if custom_path and Path(custom_path).exists():
            return Path(custom_path)
        
        # Common locations to check
        project_root = Path(__file__).parent.parent.absolute()
        possible_paths = [
            project_root / "models/codeformer",
            project_root / "models/CodeFormer", 
            Path("models/CodeFormer"),
            Path("../CodeFormer"),
            Path("../../CodeFormer"),
            Path.home() / "CodeFormer",
            Path("/opt/CodeFormer"),
        ]
        
        for path in possible_paths:
            if path.exists() and (path / "inference_codeformer.py").exists():
                self.logger.info(f"Found CodeFormer at: {path}")
                return path
        
        # Check if codeformer command is available
        try:
            result = subprocess.run(["which", "codeformer"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return Path(result.stdout.strip()).parent
        except Exception:
            pass
        
        return None
    
    def enhance_image(self, 
                     input_path: str, 
                     output_path: str,
                     fidelity_weight: float = 0.7) -> bool:
        """
        Enhance a single image using CodeFormer optimized for 1080p output.
        
        Args:
            input_path: Path to input image
            output_path: Path to save enhanced image
            fidelity_weight: Weight for fidelity vs quality (0-1, higher for better quality)
            
        Returns:
            True if enhancement successful, False otherwise
        """
        if not self.available:
            return self._fallback_enhance_image(input_path, output_path)
        
        try:
            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_input = Path(temp_dir) / "input.jpg"
                temp_output = Path(temp_dir) / "output"
                
                # Copy input to temp location
                import shutil
                shutil.copy2(input_path, temp_input)
                
                # Build CodeFormer command optimized for 1080p and MPS
                cmd = [
                    sys.executable,
                    str(self.codeformer_path / "inference_codeformer.py"),
                    "-i", str(temp_input),
                    "-o", str(temp_output),
                    "--fidelity_weight", str(fidelity_weight),
                    "--upscale", str(self.upscale),
                    "--has_aligned",  # Skip face detection for faster processing
                    "--only_center_face",  # Focus on main face for efficiency
                ]
                
                # Use Real-ESRGAN for background enhancement if available
                if self.background_enhance:
                    cmd.extend(["--bg_upsampler", "realesrgan"])
                
                if self.face_upsample:
                    cmd.append("--face_upsample")
                
                # Force MPS device for Apple Silicon or use specified device
                if self.device == "mps":
                    cmd.extend(["--device", "mps"])
                elif self.device != "auto":
                    cmd.extend(["--device", self.device])
                
                # Execute CodeFormer
                self.logger.info(f"Running CodeFormer: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    self.logger.error(f"CodeFormer failed: {result.stderr}")
                    return self._fallback_enhance_image(input_path, output_path)
                
                # Find and copy output
                output_files = list(Path(temp_output).rglob("*.jpg"))
                output_files.extend(list(Path(temp_output).rglob("*.png")))
                
                if output_files:
                    shutil.copy2(output_files[0], output_path)
                    self.logger.info(f"Enhanced image saved to: {output_path}")
                    return True
                else:
                    self.logger.error("No output file generated by CodeFormer")
                    return self._fallback_enhance_image(input_path, output_path)
                    
        except Exception as e:
            self.logger.error(f"CodeFormer enhancement failed: {str(e)}")
            return self._fallback_enhance_image(input_path, output_path)
    
    def enhance_video_frames(self, 
                           input_video_path: str, 
                           output_video_path: str,
                           fidelity_weight: float = 0.7) -> bool:
        """
        Enhance video frames using CodeFormer with optimized batch processing for 1080p.
        
        Args:
            input_video_path: Path to input video
            output_video_path: Path to save enhanced video
            fidelity_weight: Weight for fidelity vs quality (0-1)
            
        Returns:
            True if enhancement successful, False otherwise
        """
        try:
            self.logger.info(f"Enhancing video frames: {input_video_path}")
            
            # Open input video
            cap = cv2.VideoCapture(input_video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {input_video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.logger.info(f"Video properties: {original_width}x{original_height}, {fps}fps, {frame_count} frames")
            
            # Setup output video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (original_width, original_height))
            
            if not out.isOpened():
                raise ValueError(f"Could not create output video: {output_video_path}")
            
            # Process frames in batches for efficiency
            batch_size = 10  # Process 10 frames at a time
            processed_frames = 0
            
            while True:
                batch_frames = []
                batch_indices = []
                
                # Read batch of frames
                for _ in range(batch_size):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    batch_frames.append(frame)
                    batch_indices.append(processed_frames)
                    processed_frames += 1
                
                if not batch_frames:
                    break
                
                # Enhance batch of frames
                enhanced_frames = self._enhance_frame_batch(batch_frames, fidelity_weight)
                
                # Write enhanced frames
                for enhanced_frame in enhanced_frames:
                    out.write(enhanced_frame)
                
                if processed_frames % 50 == 0:
                    self.logger.info(f"Enhanced {processed_frames}/{frame_count} frames")
            
            # Cleanup
            cap.release()
            out.release()
            
            self.logger.info(f"Video frame enhancement completed: {processed_frames} frames processed")
            return True
            
        except Exception as e:
            self.logger.error(f"Video frame enhancement failed: {str(e)}")
            return False
    
    def _enhance_frame_batch(self, frames: list, fidelity_weight: float) -> list:
        """
        Enhance a batch of frames efficiently.
        
        Args:
            frames: List of frame arrays
            fidelity_weight: Weight for fidelity vs quality
            
        Returns:
            List of enhanced frame arrays
        """
        enhanced_frames = []
        
        for frame in frames:
            # Apply enhanced processing for faces
            enhanced_frame = self._apply_advanced_enhancement(frame)
            enhanced_frames.append(enhanced_frame)
        
        return enhanced_frames
    
    def _apply_advanced_enhancement(self, img: np.ndarray) -> np.ndarray:
        """
        Apply advanced image enhancement optimized for faces and 1080p.
        
        Args:
            img: Input image array
            
        Returns:
            Enhanced image array
        """
        # Convert to LAB color space for better enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel for better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back
        enhanced_lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Apply face-specific sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel * 0.1)
        enhanced = cv2.addWeighted(enhanced, 0.85, sharpened, 0.15, 0)
        
        # Noise reduction while preserving details
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return enhanced
    
    def _fallback_enhance_image(self, input_path: str, output_path: str) -> bool:
        """
        Fallback enhancement using OpenCV and basic techniques.
        
        Args:
            input_path: Path to input image
            output_path: Path to save enhanced image
            
        Returns:
            True if enhancement successful, False otherwise
        """
        try:
            self.logger.info("Using fallback image enhancement")
            
            # Read image
            img = cv2.imread(input_path)
            if img is None:
                self.logger.error(f"Could not read image: {input_path}")
                return False
            
            # Apply basic enhancement
            enhanced = self._apply_basic_enhancement(img)
            
            # Save enhanced image
            success = cv2.imwrite(output_path, enhanced)
            if success:
                self.logger.info(f"Fallback enhanced image saved to: {output_path}")
                return True
            else:
                self.logger.error(f"Failed to save enhanced image: {output_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Fallback enhancement failed: {str(e)}")
            return False
    
    def _apply_basic_enhancement(self, img: np.ndarray) -> np.ndarray:
        """
        Apply basic image enhancement using OpenCV.
        
        Args:
            img: Input image array
            
        Returns:
            Enhanced image array
        """
        # Convert to LAB color space for better enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Apply bilateral filter for noise reduction
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Slight sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel * 0.1)
        enhanced = cv2.addWeighted(img, 0.7, enhanced, 0.3, 0)
        
        # Upscale if requested
        if self.upscale > 1:
            height, width = enhanced.shape[:2]
            new_height, new_width = height * self.upscale, width * self.upscale
            enhanced = cv2.resize(enhanced, (new_width, new_height), 
                                interpolation=cv2.INTER_LANCZOS4)
        
        return enhanced
    
    def enhance_video_frames(self, 
                           video_path: str, 
                           output_path: str,
                           frame_interval: int = 1,
                           fidelity_weight: float = 0.5) -> bool:
        """
        Enhance video by processing individual frames.
        
        Args:
            video_path: Path to input video
            output_path: Path to save enhanced video
            frame_interval: Process every N frames (1 = all frames)
            fidelity_weight: Weight for fidelity vs quality (0-1)
            
        Returns:
            True if enhancement successful, False otherwise
        """
        try:
            self.logger.info(f"Enhancing video: {video_path}")
            
            # Create temporary directory for frame processing
            with tempfile.TemporaryDirectory() as temp_dir:
                frames_dir = Path(temp_dir) / "frames"
                enhanced_dir = Path(temp_dir) / "enhanced"
                frames_dir.mkdir()
                enhanced_dir.mkdir()
                
                # Extract frames
                self.logger.info("Extracting video frames...")
                if not self._extract_frames(video_path, frames_dir, frame_interval):
                    return False
                
                # Get video properties
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                
                # Enhance each frame
                frame_files = sorted(frames_dir.glob("*.jpg"))
                total_frames = len(frame_files)
                
                self.logger.info(f"Enhancing {total_frames} frames...")
                for i, frame_file in enumerate(frame_files):
                    if i % 10 == 0:
                        self.logger.info(f"Processing frame {i+1}/{total_frames}")
                    
                    enhanced_frame = enhanced_dir / frame_file.name
                    self.enhance_image(str(frame_file), str(enhanced_frame), fidelity_weight)
                
                # Reconstruct video
                self.logger.info("Reconstructing enhanced video...")
                return self._reconstruct_video(enhanced_dir, output_path, fps, width, height)
                
        except Exception as e:
            self.logger.error(f"Video enhancement failed: {str(e)}")
            return False
    
    def _extract_frames(self, video_path: str, output_dir: Path, interval: int) -> bool:
        """Extract frames from video."""
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % interval == 0:
                    frame_file = output_dir / f"frame_{extracted_count:06d}.jpg"
                    cv2.imwrite(str(frame_file), frame)
                    extracted_count += 1
                
                frame_count += 1
            
            cap.release()
            self.logger.info(f"Extracted {extracted_count} frames")
            return extracted_count > 0
            
        except Exception as e:
            self.logger.error(f"Frame extraction failed: {str(e)}")
            return False
    
    def _reconstruct_video(self, frames_dir: Path, output_path: str, 
                          fps: float, width: int, height: int) -> bool:
        """Reconstruct video from enhanced frames."""
        try:
            # Use ffmpeg for better quality video reconstruction
            frame_pattern = str(frames_dir / "frame_%06d.jpg")
            
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", frame_pattern,
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Enhanced video saved to: {output_path}")
                return True
            else:
                self.logger.error(f"Video reconstruction failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Video reconstruction failed: {str(e)}")
            return False
    
    def get_enhancement_info(self) -> Dict[str, Any]:
        """Get information about the enhancement capabilities."""
        return {
            "codeformer_available": self.available,
            "codeformer_path": str(self.codeformer_path) if self.codeformer_path else None,
            "device": self.device,
            "upscale_factor": self.upscale,
            "background_enhance": self.background_enhance,
            "face_upsample": self.face_upsample,
            "fallback_method": "OpenCV basic enhancement"
        }

def main():
    """Test the CodeFormer enhancer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test CodeFormer Integration")
    parser.add_argument("--input", required=True, help="Input image/video path")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("--codeformer-path", help="Path to CodeFormer repository")
    parser.add_argument("--fidelity", type=float, default=0.5, help="Fidelity weight (0-1)")
    parser.add_argument("--device", default="auto", help="Device (cpu/cuda/auto)")
    parser.add_argument("--upscale", type=int, default=2, help="Upscale factor")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create enhancer
    enhancer = CodeFormerEnhancer(
        codeformer_path=args.codeformer_path,
        device=args.device,
        upscale=args.upscale
    )
    
    # Print info
    info = enhancer.get_enhancement_info()
    print("CodeFormer Enhancement Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Determine if input is image or video
    input_path = Path(args.input)
    if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        # Enhance image
        success = enhancer.enhance_image(args.input, args.output, args.fidelity)
    elif input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        # Enhance video
        success = enhancer.enhance_video_frames(args.input, args.output, 
                                              fidelity_weight=args.fidelity)
    else:
        print(f"Unsupported file format: {input_path.suffix}")
        sys.exit(1)
    
    if success:
        print(f"Enhancement completed: {args.output}")
    else:
        print("Enhancement failed")
        sys.exit(1)

if __name__ == "__main__":
    main()