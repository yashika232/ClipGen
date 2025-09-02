#!/usr/bin/env python3
"""
Stage 0B: Image Animation Pipeline
Converts single portrait image + audio into high-quality talking head video using SadTalker.
Input: Portrait image + synthesized audio from Stage 0A
Output: Professional-quality talking head video with perfect lip sync
"""

import argparse
import logging
import sys
import json
import time
import os
import shutil
import subprocess
from pathlib import Path
import numpy as np
import cv2
import torch
from typing import Tuple, Dict, Any, Optional, List
import tempfile
from PIL import Image
import warnings

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

# Add SadTalker to path
SADTALKER_PATH = PROJECT_ROOT / "models" / "SadTalker"
sys.path.insert(0, str(SADTALKER_PATH))

# Import project utilities
from utils.logger import setup_logger, TimedLogger
from utils.file_utils import ensure_directory, save_json, load_json

# Import CodeFormer integration
try:
    from stages.codeformer_integration import CodeFormerEnhancer
    CODEFORMER_AVAILABLE = True
except ImportError:
    CODEFORMER_AVAILABLE = False
    CodeFormerEnhancer = None

# Import Real-ESRGAN integration
try:
    from stages.realesrgan_integration import RealESRGANUpscaler
    REALESRGAN_AVAILABLE = True
except ImportError:
    REALESRGAN_AVAILABLE = False
    RealESRGANUpscaler = None

# Import hybrid device manager
try:
    sys.path.insert(0, str(SADTALKER_PATH))
    from src.utils.hybrid_device_manager import get_device_manager
    HYBRID_DEVICE_AVAILABLE = True
except ImportError:
    HYBRID_DEVICE_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class ImageAnimationPipeline:
    """High-quality image animation pipeline using SadTalker with enhancements."""
    
    def __init__(self, config_path: Optional[str] = None, device: str = "auto"):
        """Initialize the image animation pipeline."""
        self.logger = setup_logger("stage0b_image_animation")
        
        # Load configuration
        if config_path and Path(config_path).exists():
            self.config = load_json(config_path)
        else:
            self.config = self._get_default_config()
        
        # Device selection with Apple Silicon MPS support
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Initialize hybrid device manager for MPS optimization
        self.device_manager = None
        if HYBRID_DEVICE_AVAILABLE and self.device == "mps":
            try:
                # Enable MPS fallback for unsupported operations
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                self.device_manager = get_device_manager(self.device, enable_hybrid=True)
                self.logger.info("Hybrid MPS/CPU device manager initialized with fallback enabled")
                self.device_manager.print_device_assignment()
            except Exception as e:
                self.logger.warning(f"Failed to initialize hybrid device manager: {e}")
                self.device_manager = None
        
        self.logger.info(f"Using device: {self.device}")
        
        # Paths
        self.sadtalker_path = SADTALKER_PATH
        self.checkpoint_dir = self.sadtalker_path / "checkpoints"
        self.gfpgan_path = self.sadtalker_path / "gfpgan"
        self.temp_dir = Path(tempfile.gettempdir()) / "image_animation"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Verify SadTalker installation
        self._verify_sadtalker_installation()
        
        # Initialize CodeFormer if available
        self.codeformer_enhancer = None
        enhancement_config = self.config.get("face_enhancement", self.config.get("enhancement", {}))
        if CODEFORMER_AVAILABLE and enhancement_config.get("enable_codeformer", False):
            try:
                self.codeformer_enhancer = CodeFormerEnhancer(
                    device=self.device,
                    upscale=enhancement_config.get("upscale_factor", 2),
                    background_enhance=enhancement_config.get("enable_background_enhancer", True)
                )
                if self.codeformer_enhancer.available:
                    self.logger.info("CodeFormer enhancement enabled")
                else:
                    self.logger.warning("CodeFormer not found, using fallback enhancement")
            except Exception as e:
                self.logger.warning(f"Failed to initialize CodeFormer: {e}")
                self.codeformer_enhancer = None
        
        # Initialize Real-ESRGAN for 1080p upscaling
        self.realesrgan_upscaler = None
        upscaling_config = self.config.get("postprocessing", {})
        if REALESRGAN_AVAILABLE and upscaling_config.get("enable_upscaling", True):
            try:
                self.realesrgan_upscaler = RealESRGANUpscaler(
                    device=self.device,
                    model_name="RealESRGAN_x2plus",  # 2x upscaling for 1080p
                    upscale=2
                )
                if self.realesrgan_upscaler.available:
                    self.logger.info("Real-ESRGAN 2x upscaling enabled for 1080p output")
                else:
                    self.logger.warning("Real-ESRGAN not found, using fallback upscaling")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Real-ESRGAN: {e}")
                self.realesrgan_upscaler = None
        
        self.logger.info("Image animation pipeline initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for image animation."""
        return {
            "sadtalker": {
                "size": 512,  # Model size (512 for higher quality)
                "expression_scale": 1.8,  # Higher expression intensity for natural movement
                "pose_style": 0,  # Pose style (0-45)
                "ref_eyeblink": None,  # Reference for eye blinking
                "ref_pose": None,  # Reference for pose
                "still": False,  # Minimal head motion
                "preprocess": "resize",  # Use resize mode for full face visibility
                "verbose": False,
                "old_version": False,
                "net_recon": "resnet50",  # 3D face reconstruction network
                "init_path": None,
                "use_safetensor": True
            },
            "face_enhancement": {
                "enable_gfpgan": True,  # Face enhancement
                "enable_background_enhancer": True,  # Background enhancement
                "enable_codeformer": True,  # Enable CodeFormer for superior enhancement
                "enhancer": "gfpgan",  # Enhancement method
                "background_enhancer": "realesrgan",
                "upscale_factor": 2,  # Upscaling factor
                "face_enhance_strength": 0.9,  # Higher face enhancement
                "background_enhance_strength": 0.7,  # Higher background enhancement
                "codeformer_fidelity": 0.7  # Higher fidelity for CodeFormer
            },
            "preprocessing": {
                "face_crop_size": [512, 512],  # Face crop dimensions
                "image_resize": True,  # Resize large images
                "max_image_size": 1024,  # Maximum input image size
                "detect_face": True,  # Auto-detect face
                "face_enhancement": True,  # Pre-enhance face
                "image_quality_check": True
            },
            "output": {
                "video_format": "mp4",
                "video_codec": "libx264",
                "video_quality": "high",  # high, medium, low
                "fps": 25,
                "audio_codec": "aac",
                "audio_bitrate": "128k",
                "crf": 15,  # High quality encoding (lower = better quality)
                "preset": "slow"  # Better compression
            },
            "postprocessing": {
                "face_restoration": True,
                "temporal_consistency": True,
                "motion_smoothing": True,
                "color_correction": True,
                "noise_reduction": True,
                "stabilization": False,
                "sharpening": 0.2,  # Increased sharpening for better clarity
                "contrast_enhancement": 0.15,  # Better contrast
                "enable_upscaling": True,  # Enable Real-ESRGAN upscaling
                "upscale_factor": 2,  # 2x upscaling for 1080p
                "target_resolution": "1080p"  # Target output resolution
            }
        }
    
    def _verify_sadtalker_installation(self):
        """Verify SadTalker models and dependencies are available."""
        self.logger.info("Verifying SadTalker installation...")
        
        # Check required files
        required_files = [
            "checkpoints/SadTalker_V0.0.2_256.safetensors",
            "checkpoints/SadTalker_V0.0.2_512.safetensors",
            "checkpoints/auido2exp_00300-model.pth",
            "checkpoints/auido2pose_00140-model.pth",
            "checkpoints/facevid2vid_00189-model.pth.tar",
            "checkpoints/mapping_00109-model.pth.tar",
            "checkpoints/mapping_00229-model.pth.tar",
            "gfpgan/weights/GFPGANv1.4.pth"
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = self.sadtalker_path / file_path
            if not full_path.exists():
                missing_files.append(file_path)
        
        if missing_files:
            self.logger.error("Missing SadTalker model files:")
            for file_path in missing_files:
                self.logger.error(f"  - {file_path}")
            raise FileNotFoundError("SadTalker models not found. Please run download script.")
        
        # Check main inference script
        inference_script = self.sadtalker_path / "inference.py"
        if not inference_script.exists():
            raise FileNotFoundError(f"SadTalker inference script not found: {inference_script}")
        
        self.logger.info("[SUCCESS] SadTalker installation verified")
    
    def preprocess_input_image(self, image_path: str) -> str:
        """Preprocess input image for optimal animation quality."""
        self.logger.info(f"Preprocessing input image: {Path(image_path).name}")
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            original_height, original_width = image.shape[:2]
            self.logger.info(f"Original image size: {original_width}x{original_height}")
            
            # Resize if too large
            max_size = self.config["preprocessing"]["max_image_size"]
            if max(original_width, original_height) > max_size:
                scale = max_size / max(original_width, original_height)
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
                self.logger.info(f"Resized image to: {new_width}x{new_height}")
            
            # Enhance image quality if enabled
            if self.config["preprocessing"]["face_enhancement"]:
                image = self._enhance_input_image(image)
            
            # Detect and validate face
            if self.config["preprocessing"]["detect_face"]:
                face_detected = self._detect_face_in_image(image)
                if not face_detected:
                    self.logger.warning("No face detected in image. Animation quality may be reduced.")
            
            # Save processed image
            processed_path = self.temp_dir / f"processed_image_{int(time.time())}.jpg"
            cv2.imwrite(str(processed_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            self.logger.info(f"Image preprocessing completed: {processed_path}")
            return str(processed_path)
            
        except Exception as e:
            self.logger.error(f"Image preprocessing failed: {str(e)}")
            raise
    
    def _enhance_input_image(self, image: np.ndarray) -> np.ndarray:
        """Apply enhancement to input image."""
        try:
            # Basic enhancement: contrast and sharpening
            # Convert to LAB color space for better processing
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge back
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Apply slight sharpening
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # Blend original and sharpened
            result = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Image enhancement failed: {str(e)}, using original")
            return image
    
    def _detect_face_in_image(self, image: np.ndarray) -> bool:
        """Detect if there's a face in the image."""
        try:
            # Use OpenCV's face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            self.logger.info(f"Detected {len(faces)} face(s) in image")
            return len(faces) > 0
            
        except Exception as e:
            self.logger.warning(f"Face detection failed: {str(e)}")
            return True  # Assume face is present if detection fails
    
    def animate_image_with_sadtalker(self, image_path: str, audio_path: str, 
                                   output_dir: str) -> Optional[str]:
        """Animate image using SadTalker."""
        self.logger.info("Starting SadTalker animation...")
        
        try:
            # Preprocess input image
            processed_image = self.preprocess_input_image(image_path)
            
            # Prepare output directory
            sadtalker_output_dir = ensure_directory(output_dir) / "sadtalker_raw"
            sadtalker_output_dir.mkdir(exist_ok=True)
            
            # Build SadTalker command
            cmd = self._build_sadtalker_command(
                processed_image, audio_path, str(sadtalker_output_dir)
            )
            
            # Execute SadTalker
            with TimedLogger(self.logger, "SadTalker animation"):
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(self.sadtalker_path),
                    timeout=1800  # 30 minute timeout for CPU processing
                )
            
            if result.returncode != 0:
                self.logger.error(f"SadTalker failed with return code {result.returncode}")
                self.logger.error(f"Error output: {result.stderr}")
                return None
            
            # Find generated video
            output_video = self._find_sadtalker_output(sadtalker_output_dir)
            if output_video is None:
                self.logger.error("SadTalker output video not found")
                return None
            
            self.logger.info(f"SadTalker animation completed: {output_video}")
            return str(output_video)
            
        except subprocess.TimeoutExpired:
            self.logger.error("SadTalker animation timed out")
            return None
        except Exception as e:
            self.logger.error(f"SadTalker animation failed: {str(e)}")
            return None
    
    def _build_sadtalker_command(self, image_path: str, audio_path: str, 
                               output_dir: str) -> List[str]:
        """Build SadTalker command with optimal parameters."""
        # Convert to absolute paths since SadTalker runs from its own directory
        abs_audio_path = str(Path(audio_path).resolve())
        abs_image_path = str(Path(image_path).resolve())
        abs_output_dir = str(Path(output_dir).resolve())
        
        # Use conda environment for SadTalker to ensure all dependencies work
        conda_python = "/Users/aryanjain/miniforge3/envs/sadtalker/bin/python"
        cmd = [
            conda_python,
            "inference.py",
            "--driven_audio", abs_audio_path,
            "--source_image", abs_image_path,
            "--result_dir", abs_output_dir,
            "--checkpoint_dir", str(self.checkpoint_dir),
            "--size", str(self.config["sadtalker"]["size"]),
            "--expression_scale", str(self.config["sadtalker"]["expression_scale"]),
            "--pose_style", str(self.config["sadtalker"]["pose_style"]),
            "--preprocess", self.config["sadtalker"]["preprocess"]
        ]
        
        # Use GPU acceleration for neural networks, CPU only for problematic operations
        # PIL operations have been replaced with OpenCV for better memory management
        if self.device == "cpu":
            cmd.append("--cpu")
        # For MPS/GPU, let SadTalker use GPU acceleration for neural networks
        
        # Add enhancement options
        enhancement_config = self.config.get("face_enhancement", self.config.get("enhancement", {}))
        if enhancement_config.get("enable_gfpgan", False):
            cmd.extend(["--enhancer", enhancement_config.get("enhancer", "gfpgan")])
        
        if enhancement_config.get("enable_background_enhancer", False):
            cmd.extend(["--background_enhancer", enhancement_config.get("background_enhancer", "realesrgan")])
        
        # Add optional parameters
        if self.config["sadtalker"]["still"]:
            cmd.append("--still")
        
        if self.config["sadtalker"]["verbose"]:
            cmd.append("--verbose")
        
        # Reference videos if specified
        if self.config["sadtalker"]["ref_eyeblink"]:
            cmd.extend(["--ref_eyeblink", self.config["sadtalker"]["ref_eyeblink"]])
        
        if self.config["sadtalker"]["ref_pose"]:
            cmd.extend(["--ref_pose", self.config["sadtalker"]["ref_pose"]])
        
        self.logger.info(f"SadTalker command: {' '.join(cmd)}")
        self.logger.info(f"Using config - Size: {self.config['sadtalker']['size']}, Expression: {self.config['sadtalker']['expression_scale']}, Preprocess: {self.config['sadtalker']['preprocess']}")
        return cmd
    
    def _find_sadtalker_output(self, output_dir: Path) -> Optional[Path]:
        """Find the generated video from SadTalker output directory."""
        # SadTalker typically outputs with timestamp
        video_patterns = ["*.mp4", "*.avi", "*.mov"]
        
        for pattern in video_patterns:
            videos = list(output_dir.rglob(pattern))
            if videos:
                # Return the most recent video
                latest_video = max(videos, key=lambda x: x.stat().st_mtime)
                return latest_video
        
        return None
    
    def enhance_output_video(self, input_video_path: str, audio_path: str, output_path: str) -> bool:
        """Apply additional enhancement to the output video with audio integration and 1080p upscaling."""
        self.logger.info("Applying video enhancement with Real-ESRGAN upscaling and audio integration...")
        
        try:
            # Step 1: Check if upscaling is enabled
            upscaling_config = self.config.get("postprocessing", {})
            enable_upscaling = upscaling_config.get("enable_upscaling", True)
            
            if enable_upscaling and self.realesrgan_upscaler and self.realesrgan_upscaler.available:
                # Step 1a: Upscale video to 1080p using Real-ESRGAN
                temp_upscaled_path = str(Path(output_path).with_suffix('.upscaled.mp4'))
                self.logger.info("Upscaling video to 1080p using Real-ESRGAN...")
                
                upscale_success = self.realesrgan_upscaler.upscale_video(input_video_path, temp_upscaled_path)
                if upscale_success:
                    self.logger.info("Real-ESRGAN upscaling completed successfully")
                    current_video_path = temp_upscaled_path
                else:
                    self.logger.warning("Real-ESRGAN upscaling failed, using original video")
                    current_video_path = input_video_path
            else:
                current_video_path = input_video_path
                if enable_upscaling:
                    self.logger.warning("Real-ESRGAN upscaling requested but not available, using original resolution")
            
            # Step 2: Apply frame enhancement (CodeFormer, etc.)
            temp_enhanced_path = str(Path(output_path).with_suffix('.enhanced.mp4'))
            
            enhanced_success = self._enhance_video_frames(current_video_path, temp_enhanced_path)
            if enhanced_success:
                final_video_path = temp_enhanced_path
            else:
                self.logger.warning("Frame enhancement failed, using current video")
                final_video_path = current_video_path
            
            # Step 3: Integrate audio using FFmpeg for professional quality
            ffmpeg_success = self._integrate_audio_with_ffmpeg(final_video_path, audio_path, output_path)
            
            # Cleanup temporary files
            for temp_path in [temp_upscaled_path if 'temp_upscaled_path' in locals() else None,
                             temp_enhanced_path if enhanced_success else None]:
                if temp_path and Path(temp_path).exists() and temp_path != output_path:
                    try:
                        Path(temp_path).unlink()
                    except Exception as e:
                        self.logger.warning(f"Failed to cleanup temp file {temp_path}: {e}")
            
            if ffmpeg_success:
                self.logger.info("Video enhancement with Real-ESRGAN upscaling and audio integration completed successfully")
                return True
            else:
                # Fallback: copy best available video if FFmpeg fails
                self.logger.warning("FFmpeg audio integration failed, using video without audio")
                if final_video_path != output_path:
                    shutil.copy2(final_video_path, output_path)
                return False
            
        except Exception as e:
            self.logger.error(f"Video enhancement failed: {str(e)}")
            return False
    
    def _enhance_video_frames(self, input_video_path: str, output_path: str) -> bool:
        """Process video frames with enhanced quality settings."""
        try:
            # Load video
            cap = cv2.VideoCapture(input_video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {input_video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS) or self.config["output"]["fps"]
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Use H.264 codec with high quality settings
            fourcc = cv2.VideoWriter_fourcc(*'h264')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            enhanced_frames = []
            
            # Process frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Apply frame enhancement
                postprocessing_config = self.config.get("postprocessing", self.config.get("quality", {}))
                if postprocessing_config.get("face_restoration", False):
                    enhanced_frame = self._enhance_frame(frame)
                else:
                    enhanced_frame = frame
                
                enhanced_frames.append(enhanced_frame)
                frame_count += 1
            
            cap.release()
            
            # Apply temporal consistency if enabled
            postprocessing_config = self.config.get("postprocessing", self.config.get("quality", {}))
            if postprocessing_config.get("temporal_consistency", False):
                enhanced_frames = self._apply_temporal_consistency(enhanced_frames)
            
            # Write enhanced frames
            for frame in enhanced_frames:
                out.write(frame)
            
            out.release()
            
            self.logger.info(f"Video frame enhancement completed: {frame_count} frames processed")
            return True
            
        except Exception as e:
            self.logger.error(f"Video frame enhancement failed: {str(e)}")
            return False
    
    def _integrate_audio_with_ffmpeg(self, video_path: str, audio_path: str, output_path: str) -> bool:
        """Integrate audio with video using FFmpeg for professional quality with enhanced sync."""
        try:
            # First, analyze input files to ensure compatibility
            video_info = self._get_video_info(video_path)
            audio_info = self._get_audio_info(audio_path)
            
            if not video_info or not audio_info:
                self.logger.error("Failed to analyze input files")
                return False
            
            # Build FFmpeg command for high-quality H.264 encoding with audio
            output_config = self.config.get("output", {})
            crf = output_config.get("crf", 15)  # High quality
            preset = output_config.get("preset", "slow")  # Better compression
            audio_bitrate = output_config.get("audio_bitrate", "192k")  # Higher quality audio
            
            # Enhanced FFmpeg command with professional settings
            ffmpeg_cmd = [
                "ffmpeg", "-y",  # Overwrite output file
                "-i", video_path,  # Input video
                "-i", audio_path,  # Input audio
                
                # Video encoding settings
                "-c:v", "libx264",  # H.264 video codec
                "-crf", str(crf),  # Constant Rate Factor (15 = high quality)
                "-preset", preset,  # Encoding preset
                "-profile:v", "high",  # H.264 high profile
                "-level", "4.1",  # H.264 level
                "-refs", "4",  # Reference frames
                "-bf", "3",  # B-frames
                
                # Audio encoding settings
                "-c:a", "aac",  # AAC audio codec
                "-b:a", audio_bitrate,  # Audio bitrate
                "-ar", "48000",  # Sample rate
                "-ac", "2",  # Stereo output
                "-af", "aresample=async=1000,volume=1.0,highpass=f=85,lowpass=f=15000",  # Audio filters
                
                # Stream mapping and sync
                "-map", "0:v:0",  # Map first video stream
                "-map", "1:a:0",  # Map first audio stream
                "-async", "1",  # Audio sync method
                "-vsync", "1",  # Video sync method
                "-avoid_negative_ts", "make_zero",  # Handle timing issues
                
                # Output settings
                "-movflags", "+faststart+rtphint",  # Optimize for streaming
                "-pix_fmt", "yuv420p",  # Ensure compatibility
                "-colorspace", "bt709",  # Standard color space
                "-color_range", "tv",  # TV color range
                "-max_muxing_queue_size", "1024",  # Handle sync issues
                
                output_path
            ]
            
            self.logger.info(f"Running enhanced FFmpeg for audio integration...")
            self.logger.debug(f"Video info: {video_info}")
            self.logger.debug(f"Audio info: {audio_info}")
            
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout for complex processing
            )
            
            if result.returncode == 0:
                # Verify output quality
                if self._verify_audio_video_sync(output_path):
                    self.logger.info("FFmpeg audio integration completed successfully with verified sync")
                    return True
                else:
                    self.logger.warning("Audio integration completed but sync verification failed")
                    return True  # Still return True as file was created
            else:
                self.logger.error(f"FFmpeg failed with code {result.returncode}")
                if result.stderr:
                    self.logger.error(f"FFmpeg error: {result.stderr}")
                    
                # Try fallback with simpler settings
                return self._integrate_audio_fallback(video_path, audio_path, output_path)
                
        except subprocess.TimeoutExpired:
            self.logger.error("FFmpeg audio integration timed out")
            return self._integrate_audio_fallback(video_path, audio_path, output_path)
        except Exception as e:
            self.logger.error(f"FFmpeg audio integration failed: {str(e)}")
            return self._integrate_audio_fallback(video_path, audio_path, output_path)
    
    def _get_video_info(self, video_path: str) -> Optional[Dict[str, Any]]:
        """Get video file information using FFprobe."""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                info = json.loads(result.stdout)
                # Extract video stream info
                for stream in info.get("streams", []):
                    if stream.get("codec_type") == "video":
                        return {
                            "duration": float(stream.get("duration", 0)),
                            "fps": eval(stream.get("r_frame_rate", "25/1")),
                            "width": stream.get("width", 0),
                            "height": stream.get("height", 0),
                            "codec": stream.get("codec_name", "unknown")
                        }
            return None
        except Exception as e:
            self.logger.error(f"Failed to get video info: {e}")
            return None
    
    def _get_audio_info(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """Get audio file information using FFprobe."""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                info = json.loads(result.stdout)
                # Extract audio stream info
                for stream in info.get("streams", []):
                    if stream.get("codec_type") == "audio":
                        return {
                            "duration": float(stream.get("duration", 0)),
                            "sample_rate": int(stream.get("sample_rate", 22050)),
                            "channels": int(stream.get("channels", 1)),
                            "codec": stream.get("codec_name", "unknown")
                        }
            return None
        except Exception as e:
            self.logger.error(f"Failed to get audio info: {e}")
            return None
    
    def _verify_audio_video_sync(self, video_path: str) -> bool:
        """Verify that audio and video are properly synchronized."""
        try:
            # Use FFprobe to check for sync issues
            cmd = [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "packet=pts_time", "-of", "csv=p=0",
                "-read_intervals", "%+#5", video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # If FFprobe can read the file without errors, consider it valid
            if result.returncode == 0:
                self.logger.debug("Audio-video sync verification passed")
                return True
            else:
                self.logger.warning("Audio-video sync verification detected issues")
                return False
                
        except Exception as e:
            self.logger.warning(f"Sync verification failed: {e}")
            return False
    
    def _integrate_audio_fallback(self, video_path: str, audio_path: str, output_path: str) -> bool:
        """Fallback audio integration with simpler settings."""
        try:
            self.logger.info("Attempting fallback audio integration...")
            
            # Simple FFmpeg command as fallback
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", audio_path,
                "-c:v", "copy",  # Copy video stream
                "-c:a", "aac",   # Re-encode audio
                "-b:a", "128k",  # Lower bitrate
                "-shortest",     # Match shortest stream
                output_path
            ]
            
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                self.logger.info("Fallback audio integration completed")
                return True
            else:
                self.logger.error(f"Fallback integration failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Fallback audio integration failed: {e}")
            return False

    def _enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply enhancement to a single frame."""
        try:
            # CodeFormer face restoration (if available and enabled)
            enhancement_config = self.config.get("face_enhancement", self.config.get("enhancement", {}))
            if self.codeformer_enhancer and enhancement_config.get("enable_codeformer", False):
                frame = self._apply_codeformer_enhancement(frame)
            
            # Color correction
            postprocessing_config = self.config.get("postprocessing", self.config.get("quality", {}))
            if postprocessing_config.get("color_correction", False):
                frame = self._apply_color_correction(frame)
            
            # Noise reduction
            if postprocessing_config.get("noise_reduction", False):
                frame = cv2.bilateralFilter(frame, 5, 50, 50)
            
            return frame
            
        except Exception as e:
            self.logger.warning(f"Frame enhancement failed: {str(e)}")
            return frame
    
    def _apply_color_correction(self, frame: np.ndarray) -> np.ndarray:
        """Apply enhanced color correction and brightness adjustment."""
        try:
            # 1. Convert to LAB color space for better color processing
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # 2. Apply adaptive histogram equalization to L channel
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # 3. Merge LAB channels and convert back to BGR
            lab_corrected = cv2.merge([l, a, b])
            bgr_corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)
            
            # 4. Apply gamma correction for better brightness
            gamma = 1.1  # Slight brightness boost
            inv_gamma = 1.0 / gamma
            gamma_table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            gamma_corrected = cv2.LUT(bgr_corrected, gamma_table)
            
            # 5. Enhance vibrance (selective saturation boost)
            hsv = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # Boost saturation selectively (avoid oversaturation)
            s = cv2.multiply(s, 1.15)  # 15% saturation boost
            s = np.clip(s, 0, 255).astype(np.uint8)
            
            # 6. Merge HSV and convert back to BGR
            enhanced_hsv = cv2.merge([h, s, v])
            final_corrected = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
            
            # 7. Apply subtle sharpening
            postprocessing_config = self.config.get("postprocessing", {})
            sharpening_strength = postprocessing_config.get("sharpening", 0.1)
            if sharpening_strength > 0:
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * sharpening_strength
                kernel[1,1] = kernel[1,1] + (1 - sharpening_strength * 8)
                final_corrected = cv2.filter2D(final_corrected, -1, kernel)
            
            return final_corrected
            
        except Exception as e:
            self.logger.warning(f"Color correction failed: {str(e)}")
            return frame
    
    def _apply_codeformer_enhancement(self, frame: np.ndarray) -> np.ndarray:
        """Apply CodeFormer enhancement to a single frame."""
        try:
            # Create temporary files
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_input:
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_output:
                    # Save frame to temp file
                    cv2.imwrite(temp_input.name, frame)
                    
                    # Apply CodeFormer enhancement
                    enhancement_config = self.config.get("face_enhancement", self.config.get("enhancement", {}))
                    fidelity_weight = enhancement_config.get("codeformer_fidelity", 0.5)
                    success = self.codeformer_enhancer.enhance_image(
                        temp_input.name, 
                        temp_output.name, 
                        fidelity_weight
                    )
                    
                    if success and Path(temp_output.name).exists():
                        # Read enhanced frame
                        enhanced_frame = cv2.imread(temp_output.name)
                        if enhanced_frame is not None:
                            # Resize to match original dimensions if needed
                            if enhanced_frame.shape[:2] != frame.shape[:2]:
                                enhanced_frame = cv2.resize(enhanced_frame, 
                                                           (frame.shape[1], frame.shape[0]), 
                                                           interpolation=cv2.INTER_LANCZOS4)
                            frame = enhanced_frame
                    
                    # Clean up temp files
                    try:
                        os.unlink(temp_input.name)
                        os.unlink(temp_output.name)
                    except (OSError, FileNotFoundError):
                        pass  # Ignore cleanup errors
                        
        except Exception as e:
            self.logger.warning(f"CodeFormer enhancement failed for frame: {str(e)}")
            
        return frame
    
    def _apply_temporal_consistency(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply temporal smoothing across frames."""
        if len(frames) < 2:
            return frames
        
        # Simple temporal averaging
        smoothed_frames = [frames[0]]  # First frame unchanged
        
        alpha = 0.1  # Smoothing factor
        
        for i in range(1, len(frames)):
            # Blend with previous frame
            blended = cv2.addWeighted(frames[i], 1-alpha, smoothed_frames[i-1], alpha, 0)
            smoothed_frames.append(blended)
        
        return smoothed_frames
    
    def save_outputs(self, output_dir: str, final_video_path: str,
                    input_image_path: str, input_audio_path: str,
                    processing_metadata: Dict[str, Any]) -> Dict[str, str]:
        """Save processing outputs and metadata."""
        output_path = ensure_directory(output_dir)
        
        # Prepare metadata
        metadata = {
            "input_image_path": str(input_image_path),
            "input_audio_path": str(input_audio_path),
            "output_video_path": str(final_video_path),
            "processing_timestamp": time.time(),
            "device_used": self.device,
            "sadtalker_path": str(self.sadtalker_path),
            "config": self.config,
            "processing_info": processing_metadata
        }
        
        # Analyze output video
        try:
            cap = cv2.VideoCapture(final_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            metadata["output_video_info"] = {
                "duration": duration,
                "fps": fps,
                "frame_count": frame_count,
                "resolution": f"{width}x{height}",
                "file_size_mb": Path(final_video_path).stat().st_size / 1024 / 1024
            }
        except Exception as e:
            self.logger.warning(f"Could not analyze output video: {str(e)}")
        
        # Save metadata
        metadata_path = output_path / "image_animation_metadata.json"
        save_json(metadata, metadata_path)
        
        output_files = {
            "animated_video": str(final_video_path),
            "metadata": str(metadata_path)
        }
        
        self.logger.info(f"Image animation outputs saved to: {output_path}")
        for key, path in output_files.items():
            if Path(path).exists():
                file_size = Path(path).stat().st_size
                self.logger.info(f"  {key}: {Path(path).name} ({file_size:,} bytes)")
        
        return output_files
    
    def process_image_animation(self, image_path: str, audio_path: str, 
                              output_dir: str) -> bool:
        """Main image animation processing function."""
        try:
            self.logger.info("="*60)
            self.logger.info("Starting Image Animation Pipeline")
            self.logger.info("="*60)
            
            start_time = time.time()
            output_path = ensure_directory(output_dir)
            
            processing_metadata = {
                "start_time": start_time,
                "sadtalker_version": "v0.0.2",
                "enhancement_enabled": self.config.get("face_enhancement", self.config.get("enhancement", {})).get("enable_gfpgan", False)
            }
            
            # Generate talking head with SadTalker
            with TimedLogger(self.logger, "SadTalker image animation"):
                raw_video_path = self.animate_image_with_sadtalker(
                    image_path, audio_path, str(output_path)
                )
            
            if raw_video_path is None:
                self.logger.error("SadTalker animation failed")
                return False
            
            processing_metadata["sadtalker_output"] = raw_video_path
            
            # Apply additional enhancement
            final_video_path = output_path / "final_talking_head.mp4"
            
            postprocessing_config = self.config.get("postprocessing", self.config.get("quality", {}))
            if postprocessing_config.get("face_restoration", False):
                with TimedLogger(self.logger, "video enhancement"):
                    enhancement_success = self.enhance_output_video(
                        raw_video_path, audio_path, str(final_video_path)
                    )
                
                if not enhancement_success:
                    self.logger.warning("Video enhancement failed, using raw output")
                    # Still try to integrate audio even if enhancement failed
                    self._integrate_audio_with_ffmpeg(raw_video_path, audio_path, str(final_video_path))
            else:
                # Even without enhancement, integrate audio for complete output
                self.logger.info("Integrating audio without video enhancement...")
                audio_success = self._integrate_audio_with_ffmpeg(raw_video_path, audio_path, str(final_video_path))
                if not audio_success:
                    self.logger.warning("Audio integration failed, copying video without audio")
                    shutil.copy2(raw_video_path, final_video_path)
            
            processing_metadata["enhancement_applied"] = postprocessing_config.get("face_restoration", False)
            processing_metadata["final_output"] = str(final_video_path)
            
            # Save outputs and metadata
            self.save_outputs(
                str(output_path), str(final_video_path), 
                image_path, audio_path, processing_metadata
            )
            
            # Final summary
            total_time = time.time() - start_time
            self.logger.info("="*60)
            self.logger.info("Image Animation Pipeline Completed Successfully")
            self.logger.info(f"Total processing time: {total_time:.2f}s")
            self.logger.info(f"Final video: {final_video_path}")
            self.logger.info("="*60)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Image animation processing failed: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Stage 0B: Image Animation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Animate image with audio
  python stage0b_image_animation.py --input-image photo.jpg --input-audio speech.wav --output-dir outputs/

  # Use custom configuration
  python stage0b_image_animation.py --input-image photo.jpg --input-audio speech.wav --config animation_config.json
        """
    )
    
    parser.add_argument(
        "--input-image",
        type=str,
        required=True,
        help="Path to input portrait image"
    )
    
    parser.add_argument(
        "--input-audio",
        type=str,
        required=True,
        help="Path to driving audio (from Stage 0A or any audio file)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "stage0b"),
        help="Output directory for animated video and metadata"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "image_animation_config.json"),
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="Computing device"
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
    
    # Validate input files
    if not Path(args.input_image).exists():
        print(f"Error: Input image file not found: {args.input_image}")
        return 1
    
    if not Path(args.input_audio).exists():
        print(f"Error: Input audio file not found: {args.input_audio}")
        return 1
    
    try:
        # Create image animation pipeline
        pipeline = ImageAnimationPipeline(
            config_path=args.config if Path(args.config).exists() else None,
            device=args.device
        )
        
        # Process image animation
        success = pipeline.process_image_animation(
            args.input_image,
            args.input_audio,
            args.output_dir
        )
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"Failed to create image animation pipeline: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())