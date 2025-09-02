#!/usr/bin/env python3
"""
Intelligent Pipeline Routing System
Automatically optimizes video synthesis pipeline based on input characteristics,
hardware capabilities, and quality requirements.
"""

import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import time
from dataclasses import dataclass, asdict
from enum import Enum
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

from utils.logger import setup_logger

class QualityTier(Enum):
    """Quality tiers for pipeline optimization."""
    DRAFT = "draft"
    STANDARD = "standard"
    HIGH = "high"
    PREMIUM = "premium"

class ProcessingMode(Enum):
    """Processing modes for different use cases."""
    SPEED = "speed"
    BALANCED = "balanced"
    QUALITY = "quality"
    MAXIMUM = "maximum"

@dataclass
class InputAnalysis:
    """Analysis results for input media."""
    # Image analysis
    image_resolution: Tuple[int, int]
    image_quality_score: float
    face_count: int
    face_quality_scores: List[float]
    image_complexity: float
    
    # Audio analysis
    audio_duration: float
    audio_sample_rate: int
    audio_channels: int
    audio_quality_score: float
    
    # Combined metrics
    overall_quality_score: float
    processing_complexity: float

@dataclass
class HardwareProfile:
    """Hardware capabilities profile."""
    device_type: str  # cpu, cuda, mps
    memory_gb: float
    compute_score: float
    supports_gpu_acceleration: bool
    supports_batch_processing: bool
    estimated_processing_speed: float

@dataclass
class PipelineConfiguration:
    """Optimized pipeline configuration."""
    # Core settings
    quality_tier: QualityTier
    processing_mode: ProcessingMode
    
    # SadTalker settings
    sadtalker_size: int
    expression_scale: float
    preprocess_mode: str
    
    # Enhancement settings
    enable_face_enhancement: bool
    enable_background_enhancement: bool
    enable_temporal_consistency: bool
    
    # Real-ESRGAN settings
    realesrgan_scale: int
    enable_batch_processing: bool
    batch_size: int
    
    # CodeFormer settings
    codeformer_fidelity: float
    enable_frame_enhancement: bool
    
    # InsightFace settings
    face_detection_threshold: float
    max_faces: int
    
    # Performance settings
    target_resolution: str
    estimated_processing_time: float
    memory_usage_mb: float

class IntelligentRouter:
    """
    Intelligent pipeline routing system that optimizes processing based on
    input characteristics and hardware capabilities.
    """
    
    def __init__(self, hardware_profile: Optional[HardwareProfile] = None):
        """
        Initialize the intelligent router.
        
        Args:
            hardware_profile: Optional hardware profile, auto-detected if None
        """
        self.logger = setup_logger("intelligent_router")
        self.hardware_profile = hardware_profile or self._detect_hardware_profile()
        
        # Load quality presets
        self.quality_presets = self._load_quality_presets()
        
        self.logger.info(f"Intelligent router initialized with {self.hardware_profile.device_type} device")
    
    def _detect_hardware_profile(self) -> HardwareProfile:
        """Auto-detect hardware capabilities."""
        # Detect device
        if torch.cuda.is_available():
            device_type = "cuda"
            supports_gpu = True
        elif torch.backends.mps.is_available():
            device_type = "mps"
            supports_gpu = True
        else:
            device_type = "cpu"
            supports_gpu = False
        
        # Estimate memory (simplified)
        try:
            if device_type == "cuda":
                memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            else:
                # Use system memory as approximation
                import psutil
                memory_gb = psutil.virtual_memory().total / (1024**3)
        except Exception:
            memory_gb = 16.0  # Default assumption
        
        # Compute score based on device and memory
        if device_type == "cuda":
            compute_score = min(memory_gb / 8.0, 2.0)  # Scale with GPU memory
        elif device_type == "mps":
            compute_score = min(memory_gb / 16.0, 1.5)  # Apple Silicon unified memory
        else:
            compute_score = min(memory_gb / 32.0, 1.0)  # CPU processing
        
        # Estimate processing speed (frames per second)
        speed_multipliers = {"cuda": 10.0, "mps": 5.0, "cpu": 1.0}
        estimated_speed = compute_score * speed_multipliers[device_type]
        
        return HardwareProfile(
            device_type=device_type,
            memory_gb=memory_gb,
            compute_score=compute_score,
            supports_gpu_acceleration=supports_gpu,
            supports_batch_processing=memory_gb > 8.0,
            estimated_processing_speed=estimated_speed
        )
    
    def _load_quality_presets(self) -> Dict[QualityTier, Dict]:
        """Load quality presets for different tiers."""
        return {
            QualityTier.DRAFT: {
                "sadtalker_size": 256,
                "expression_scale": 1.0,
                "realesrgan_scale": 1,
                "enable_face_enhancement": False,
                "enable_background_enhancement": False,
                "enable_temporal_consistency": False,
                "codeformer_fidelity": 0.5,
                "target_resolution": "720p",
                "face_detection_threshold": 0.5
            },
            QualityTier.STANDARD: {
                "sadtalker_size": 512,
                "expression_scale": 1.5,
                "realesrgan_scale": 2,
                "enable_face_enhancement": True,
                "enable_background_enhancement": False,
                "enable_temporal_consistency": True,
                "codeformer_fidelity": 0.6,
                "target_resolution": "1080p",
                "face_detection_threshold": 0.6
            },
            QualityTier.HIGH: {
                "sadtalker_size": 512,
                "expression_scale": 1.8,
                "realesrgan_scale": 2,
                "enable_face_enhancement": True,
                "enable_background_enhancement": True,
                "enable_temporal_consistency": True,
                "codeformer_fidelity": 0.7,
                "target_resolution": "1080p",
                "face_detection_threshold": 0.7
            },
            QualityTier.PREMIUM: {
                "sadtalker_size": 512,
                "expression_scale": 2.0,
                "realesrgan_scale": 2,
                "enable_face_enhancement": True,
                "enable_background_enhancement": True,
                "enable_temporal_consistency": True,
                "codeformer_fidelity": 0.8,
                "target_resolution": "1080p",
                "face_detection_threshold": 0.8
            }
        }
    
    def analyze_input(self, image_path: str, audio_path: str) -> InputAnalysis:
        """
        Analyze input image and audio to determine optimal processing strategy.
        
        Args:
            image_path: Path to input image
            audio_path: Path to input audio
            
        Returns:
            Input analysis results
        """
        self.logger.info(f"Analyzing input: {Path(image_path).name}, {Path(audio_path).name}")
        
        # Analyze image
        image_analysis = self._analyze_image(image_path)
        
        # Analyze audio
        audio_analysis = self._analyze_audio(audio_path)
        
        # Combine analyses
        overall_quality = (image_analysis["quality_score"] + audio_analysis["quality_score"]) / 2
        processing_complexity = self._calculate_processing_complexity(image_analysis, audio_analysis)
        
        return InputAnalysis(
            image_resolution=image_analysis["resolution"],
            image_quality_score=image_analysis["quality_score"],
            face_count=image_analysis["face_count"],
            face_quality_scores=image_analysis["face_quality_scores"],
            image_complexity=image_analysis["complexity"],
            audio_duration=audio_analysis["duration"],
            audio_sample_rate=audio_analysis["sample_rate"],
            audio_channels=audio_analysis["channels"],
            audio_quality_score=audio_analysis["quality_score"],
            overall_quality_score=overall_quality,
            processing_complexity=processing_complexity
        )
    
    def _analyze_image(self, image_path: str) -> Dict:
        """Analyze image characteristics."""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            height, width = image.shape[:2]
            resolution = (width, height)
            
            # Analyze image quality using multiple metrics
            quality_metrics = []
            
            # Sharpness (Laplacian variance)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            quality_metrics.append(min(sharpness / 1000.0, 1.0))
            
            # Contrast (standard deviation)
            contrast = np.std(gray) / 255.0
            quality_metrics.append(contrast)
            
            # Resolution quality factor
            pixel_count = width * height
            resolution_factor = min(pixel_count / (1920 * 1080), 1.0)
            quality_metrics.append(resolution_factor)
            
            # Overall image quality score
            quality_score = np.mean(quality_metrics)
            
            # Face detection for complexity analysis
            try:
                from stages.insightface_integration import InsightFaceDetector
                detector = InsightFaceDetector(device="cpu")  # Use CPU for analysis
                faces = detector.detect_faces(image, max_faces=5)
                face_count = len(faces)
                face_quality_scores = [face.get("confidence", 0.5) for face in faces]
            except Exception as e:
                self.logger.warning(f"Face detection failed during analysis: {e}")
                face_count = 1  # Assume one face
                face_quality_scores = [0.7]
            
            # Image complexity (edge density)
            edges = cv2.Canny(gray, 50, 150)
            complexity = np.sum(edges > 0) / (width * height)
            
            return {
                "resolution": resolution,
                "quality_score": quality_score,
                "face_count": face_count,
                "face_quality_scores": face_quality_scores,
                "complexity": complexity
            }
            
        except Exception as e:
            self.logger.error(f"Image analysis failed: {e}")
            return {
                "resolution": (512, 512),
                "quality_score": 0.5,
                "face_count": 1,
                "face_quality_scores": [0.5],
                "complexity": 0.5
            }
    
    def _analyze_audio(self, audio_path: str) -> Dict:
        """Analyze audio characteristics."""
        try:
            # Get audio file info using FFprobe
            import subprocess
            
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise ValueError("FFprobe failed")
            
            import json
            info = json.loads(result.stdout)
            
            # Extract audio stream info
            audio_stream = None
            for stream in info.get("streams", []):
                if stream.get("codec_type") == "audio":
                    audio_stream = stream
                    break
            
            if not audio_stream:
                raise ValueError("No audio stream found")
            
            # Get audio properties
            duration = float(info["format"].get("duration", 0))
            sample_rate = int(audio_stream.get("sample_rate", 44100))
            channels = int(audio_stream.get("channels", 2))
            bit_rate = int(audio_stream.get("bit_rate", 128000))
            
            # Calculate quality score based on audio properties
            quality_factors = []
            
            # Sample rate quality
            sr_quality = min(sample_rate / 48000.0, 1.0)
            quality_factors.append(sr_quality)
            
            # Bit rate quality
            br_quality = min(bit_rate / 320000.0, 1.0)
            quality_factors.append(br_quality)
            
            # Duration reasonableness (not too short, not too long)
            if duration < 1.0:
                duration_quality = duration
            elif duration > 300.0:  # 5 minutes
                duration_quality = 0.8
            else:
                duration_quality = 1.0
            quality_factors.append(duration_quality)
            
            quality_score = np.mean(quality_factors)
            
            return {
                "duration": duration,
                "sample_rate": sample_rate,
                "channels": channels,
                "quality_score": quality_score
            }
            
        except Exception as e:
            self.logger.error(f"Audio analysis failed: {e}")
            return {
                "duration": 10.0,
                "sample_rate": 44100,
                "channels": 2,
                "quality_score": 0.7
            }
    
    def _calculate_processing_complexity(self, image_analysis: Dict, audio_analysis: Dict) -> float:
        """Calculate overall processing complexity."""
        complexity_factors = []
        
        # Image complexity factors
        width, height = image_analysis["resolution"]
        resolution_complexity = (width * height) / (1920 * 1080)
        complexity_factors.append(resolution_complexity)
        
        complexity_factors.append(image_analysis["complexity"])
        complexity_factors.append(image_analysis["face_count"] / 5.0)  # Normalize to max 5 faces
        
        # Audio complexity factors
        duration_complexity = min(audio_analysis["duration"] / 60.0, 2.0)  # Normalize to 2 minutes
        complexity_factors.append(duration_complexity)
        
        return min(np.mean(complexity_factors), 2.0)  # Cap at 2.0x complexity
    
    def determine_optimal_configuration(self, 
                                      input_analysis: InputAnalysis,
                                      user_preferences: Optional[Dict] = None) -> PipelineConfiguration:
        """
        Determine optimal pipeline configuration based on analysis and preferences.
        
        Args:
            input_analysis: Input analysis results
            user_preferences: Optional user preferences override
            
        Returns:
            Optimized pipeline configuration
        """
        preferences = user_preferences or {}
        
        # Determine quality tier based on input quality and user preference
        requested_quality = preferences.get("quality_tier")
        if requested_quality:
            quality_tier = QualityTier(requested_quality)
        else:
            # Auto-determine based on input quality
            if input_analysis.overall_quality_score > 0.8:
                quality_tier = QualityTier.HIGH
            elif input_analysis.overall_quality_score > 0.6:
                quality_tier = QualityTier.STANDARD
            else:
                quality_tier = QualityTier.DRAFT
        
        # Determine processing mode based on hardware and complexity
        processing_mode = self._determine_processing_mode(input_analysis, preferences)
        
        # Get base configuration from preset
        base_config = self.quality_presets[quality_tier].copy()
        
        # Adjust configuration based on hardware capabilities
        base_config = self._adjust_for_hardware(base_config, input_analysis)
        
        # Apply user preferences
        base_config.update(preferences.get("overrides", {}))
        
        # Calculate estimates
        processing_time = self._estimate_processing_time(input_analysis, base_config)
        memory_usage = self._estimate_memory_usage(input_analysis, base_config)
        
        return PipelineConfiguration(
            quality_tier=quality_tier,
            processing_mode=processing_mode,
            sadtalker_size=base_config["sadtalker_size"],
            expression_scale=base_config["expression_scale"],
            preprocess_mode="resize",
            enable_face_enhancement=base_config["enable_face_enhancement"],
            enable_background_enhancement=base_config["enable_background_enhancement"],
            enable_temporal_consistency=base_config["enable_temporal_consistency"],
            realesrgan_scale=base_config["realesrgan_scale"],
            enable_batch_processing=self.hardware_profile.supports_batch_processing,
            batch_size=self._calculate_optimal_batch_size(input_analysis),
            codeformer_fidelity=base_config["codeformer_fidelity"],
            enable_frame_enhancement=base_config["enable_face_enhancement"],
            face_detection_threshold=base_config["face_detection_threshold"],
            max_faces=min(input_analysis.face_count * 2, 5),
            target_resolution=base_config["target_resolution"],
            estimated_processing_time=processing_time,
            memory_usage_mb=memory_usage
        )
    
    def _determine_processing_mode(self, input_analysis: InputAnalysis, 
                                 preferences: Dict) -> ProcessingMode:
        """Determine optimal processing mode."""
        user_mode = preferences.get("processing_mode")
        if user_mode:
            return ProcessingMode(user_mode)
        
        # Auto-determine based on complexity and hardware
        if input_analysis.processing_complexity > 1.5:
            if self.hardware_profile.compute_score > 1.5:
                return ProcessingMode.BALANCED
            else:
                return ProcessingMode.SPEED
        elif self.hardware_profile.compute_score > 1.8:
            return ProcessingMode.QUALITY
        else:
            return ProcessingMode.BALANCED
    
    def _adjust_for_hardware(self, config: Dict, input_analysis: InputAnalysis) -> Dict:
        """Adjust configuration based on hardware capabilities."""
        # Reduce settings for lower-end hardware
        if self.hardware_profile.compute_score < 1.0:
            config["sadtalker_size"] = min(config["sadtalker_size"], 256)
            config["enable_background_enhancement"] = False
            if self.hardware_profile.memory_gb < 8.0:
                config["enable_temporal_consistency"] = False
        
        # Boost settings for high-end hardware
        elif self.hardware_profile.compute_score > 1.8:
            if input_analysis.overall_quality_score > 0.7:
                config["expression_scale"] = min(config["expression_scale"] * 1.1, 2.0)
                config["codeformer_fidelity"] = min(config["codeformer_fidelity"] + 0.1, 0.9)
        
        return config
    
    def _calculate_optimal_batch_size(self, input_analysis: InputAnalysis) -> int:
        """Calculate optimal batch size based on hardware and input."""
        if not self.hardware_profile.supports_batch_processing:
            return 1
        
        # Base batch size on memory and complexity
        base_batch_size = max(1, int(self.hardware_profile.memory_gb / 4.0))
        
        # Adjust for complexity
        complexity_factor = 1.0 / max(input_analysis.processing_complexity, 0.5)
        optimal_batch_size = max(1, int(base_batch_size * complexity_factor))
        
        return min(optimal_batch_size, 8)  # Cap at 8
    
    def _estimate_processing_time(self, input_analysis: InputAnalysis, config: Dict) -> float:
        """Estimate processing time in minutes."""
        # Base time factors
        duration_factor = input_analysis.audio_duration / 10.0  # 10-second baseline
        complexity_factor = input_analysis.processing_complexity
        quality_factor = {"draft": 0.5, "standard": 1.0, "high": 1.5, "premium": 2.0}.get(
            config.get("target_resolution", "standard"), 1.0
        )
        
        # Hardware speed factor
        speed_factor = 1.0 / max(self.hardware_profile.estimated_processing_speed, 0.1)
        
        # Base processing time (minutes)
        base_time = 2.0  # 2 minutes for baseline 10-second video
        
        estimated_time = base_time * duration_factor * complexity_factor * quality_factor * speed_factor
        
        return max(estimated_time, 0.5)  # Minimum 30 seconds
    
    def _estimate_memory_usage(self, input_analysis: InputAnalysis, config: Dict) -> float:
        """Estimate memory usage in MB."""
        # Base memory usage
        base_memory = 2048  # 2GB baseline
        
        # Adjust for resolution
        width, height = input_analysis.image_resolution
        resolution_factor = (width * height) / (512 * 512)
        
        # Adjust for batch processing
        batch_factor = config.get("batch_size", 1)
        
        # Adjust for quality settings
        quality_multiplier = 1.0
        if config.get("enable_face_enhancement", False):
            quality_multiplier += 0.3
        if config.get("enable_background_enhancement", False):
            quality_multiplier += 0.3
        if config.get("enable_temporal_consistency", False):
            quality_multiplier += 0.2
        
        estimated_memory = base_memory * resolution_factor * batch_factor * quality_multiplier
        
        return min(estimated_memory, self.hardware_profile.memory_gb * 1024 * 0.8)  # Cap at 80% of available
    
    def generate_config_file(self, config: PipelineConfiguration, output_path: str):
        """Generate a configuration file for the optimized pipeline."""
        config_dict = {
            "metadata": {
                "generated_by": "intelligent_router",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "quality_tier": config.quality_tier.value,
                "processing_mode": config.processing_mode.value,
                "estimated_processing_time_minutes": config.estimated_processing_time,
                "estimated_memory_usage_mb": config.memory_usage_mb
            },
            "sadtalker": {
                "size": config.sadtalker_size,
                "expression_scale": config.expression_scale,
                "pose_style": 0,
                "preprocess": config.preprocess_mode,
                "verbose": True
            },
            "face_enhancement": {
                "enable_gfpgan": config.enable_face_enhancement,
                "enable_background_enhancer": config.enable_background_enhancement,
                "enable_codeformer": config.enable_frame_enhancement,
                "codeformer_fidelity": config.codeformer_fidelity,
                "enhancer": "gfpgan",
                "background_enhancer": "realesrgan"
            },
            "realesrgan": {
                "upscale_factor": config.realesrgan_scale,
                "enable_temporal_consistency": config.enable_temporal_consistency,
                "enable_batch_processing": config.enable_batch_processing,
                "batch_size": config.batch_size
            },
            "insightface": {
                "detection_threshold": config.face_detection_threshold,
                "max_faces": config.max_faces
            },
            "output": {
                "target_resolution": config.target_resolution,
                "video_codec": "libx264",
                "crf": 15,
                "preset": "medium"
            },
            "performance": {
                "device": self.hardware_profile.device_type,
                "enable_gpu_acceleration": self.hardware_profile.supports_gpu_acceleration,
                "memory_optimization": True
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        self.logger.info(f"Generated optimized configuration: {output_path}")
    
    def get_routing_summary(self, input_analysis: InputAnalysis, 
                          config: PipelineConfiguration) -> Dict:
        """Generate a summary of routing decisions."""
        return {
            "input_analysis": asdict(input_analysis),
            "hardware_profile": asdict(self.hardware_profile),
            "pipeline_configuration": asdict(config),
            "optimization_summary": {
                "quality_tier": config.quality_tier.value,
                "processing_mode": config.processing_mode.value,
                "key_optimizations": self._get_key_optimizations(config),
                "performance_estimates": {
                    "processing_time_minutes": config.estimated_processing_time,
                    "memory_usage_mb": config.memory_usage_mb,
                    "target_resolution": config.target_resolution
                }
            }
        }
    
    def _get_key_optimizations(self, config: PipelineConfiguration) -> List[str]:
        """Get list of key optimizations applied."""
        optimizations = []
        
        if config.enable_batch_processing:
            optimizations.append(f"Batch processing (size={config.batch_size})")
        
        if config.enable_temporal_consistency:
            optimizations.append("Temporal consistency for smooth video")
        
        if config.enable_face_enhancement:
            optimizations.append("Face enhancement with CodeFormer")
        
        if config.enable_background_enhancement:
            optimizations.append("Background enhancement with Real-ESRGAN")
        
        if config.realesrgan_scale > 1:
            optimizations.append(f"Real-ESRGAN {config.realesrgan_scale}x upscaling")
        
        if self.hardware_profile.supports_gpu_acceleration:
            optimizations.append(f"GPU acceleration ({self.hardware_profile.device_type.upper()})")
        
        return optimizations


def main():
    """Test the intelligent routing system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Intelligent Pipeline Routing")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--audio", required=True, help="Input audio path")
    parser.add_argument("--output-config", default="optimized_config.json", help="Output config file")
    parser.add_argument("--quality-tier", choices=["draft", "standard", "high", "premium"], 
                       help="Force specific quality tier")
    parser.add_argument("--processing-mode", choices=["speed", "balanced", "quality", "maximum"],
                       help="Force specific processing mode")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize router
    router = IntelligentRouter()
    
    # Analyze input
    analysis = router.analyze_input(args.image, args.audio)
    
    # Determine configuration
    preferences = {}
    if args.quality_tier:
        preferences["quality_tier"] = args.quality_tier
    if args.processing_mode:
        preferences["processing_mode"] = args.processing_mode
    
    config = router.determine_optimal_configuration(analysis, preferences)
    
    # Generate configuration file
    router.generate_config_file(config, args.output_config)
    
    # Print summary
    summary = router.get_routing_summary(analysis, config)
    
    print("\\nTarget: INTELLIGENT ROUTING SUMMARY")
    print("=" * 50)
    print(f"Quality Tier: {config.quality_tier.value.title()}")
    print(f"Processing Mode: {config.processing_mode.value.title()}")
    print(f"Target Resolution: {config.target_resolution}")
    print(f"Estimated Time: {config.estimated_processing_time:.1f} minutes")
    print(f"Memory Usage: {config.memory_usage_mb:.0f} MB")
    
    print("\\nTools Key Optimizations:")
    for opt in summary["optimization_summary"]["key_optimizations"]:
        print(f"  • {opt}")
    
    print(f"\\nFile: Configuration saved to: {args.output_config}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())