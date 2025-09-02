#!/usr/bin/env python3
"""
Production Mode Manager - Real Models Only
Manages production-only video processing with real models
NO SIMULATION OR FALLBACK MODES - Production pipeline only
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from enum import Enum
import logging
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from pipeline_exceptions import (
    PipelineException, EnvironmentException, CondaEnvironmentError,
    ProcessingTimeoutError, VideoGenerationError, ExceptionHandler
)


class ProcessingMode(Enum):
    """Available processing modes - PRODUCTION ONLY."""
    PRODUCTION = "production"


class StageCapability(Enum):
    """Stage processing capability status - REAL MODELS ONLY."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class ProductionModeManager:
    """Manages production-only processing with real models."""
    
    def __init__(self, base_dir: str = None, quick_startup: bool = False):
        """Initialize production mode manager.
        
        Args:
            base_dir: Base directory for the pipeline
            quick_startup: Skip slow environment checks for fast API startup
        """
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        self.logger = logging.getLogger(__name__)
        self.quick_startup = quick_startup
        
        # Conda environments for real models
        self.conda_envs = {
            'xtts': '/Users/aryanjain/miniforge3/envs/xtts_voice_cloning/bin/python',
            'sadtalker': '/opt/miniconda3/envs/sadtalker/bin/python', 
            'insightface': '/opt/miniconda3/envs/sadtalker/bin/python',
            'realesrgan': '/Users/aryanjain/miniforge3/envs/realesrgan_real/bin/python',
            'codeformer': '/Users/aryanjain/miniforge3/envs/realesrgan_real/bin/python',
            'manim': '/Users/aryanjain/miniforge3/envs/video-audio-processing/bin/python',
            'ffmpeg': '/opt/homebrew/bin/ffmpeg'
        }
        
        # Real model paths
        self.model_paths = {
            'sadtalker': '/Users/aryanjain/Documents/video-synthesis-pipeline copy/real_models/SadTalker',
            'realesrgan': '/Users/aryanjain/Documents/video-synthesis-pipeline copy/real_models/Real-ESRGAN/weights',
            'codeformer': '/Users/aryanjain/Documents/video-synthesis-pipeline copy/models/codeformer/weights/CodeFormer',
            'insightface': '~/.insightface/models/buffalo_l'
        }
    
    def get_processing_mode(self) -> str:
        """Get current processing mode - always production."""
        return ProcessingMode.PRODUCTION.value
    
    def set_processing_mode(self, mode: str) -> None:
        """Set processing mode - only production is supported."""
        if mode.lower() != 'production':
            self.logger.warning(f"Only production mode is supported. Ignoring mode: {mode}")
        # In production-only manager, mode is always production - nothing to set
    
    def assess_pipeline_capabilities(self) -> Dict[str, Any]:
        """Assess which pipeline stages can run in production with real models."""
        capabilities = {}
        
        # Check each stage
        stages = ['xtts', 'insightface', 'sadtalker', 'realesrgan', 'codeformer', 'manim', 'ffmpeg']
        
        for stage in stages:
            capabilities[stage] = {
                'mode': ProcessingMode.PRODUCTION.value,
                'available': self._check_stage_availability(stage),
                'real_models_only': True,
                'fallback_enabled': False
            }
        
        return {
            'processing_mode': ProcessingMode.PRODUCTION.value,
            'total_stages': len(stages),
            'available_stages': sum(1 for s in capabilities.values() if s['available']),
            'real_models_required': True,
            'stage_details': capabilities,
            'timestamp': datetime.now().isoformat()
        }
    
    def _check_stage_availability(self, stage_name: str) -> bool:
        """Check if a stage is available for production processing."""
        try:
            # Quick startup mode - just check paths exist
            if self.quick_startup:
                if stage_name == 'ffmpeg':
                    return Path('/opt/homebrew/bin/ffmpeg').exists()
                elif stage_name in self.conda_envs:
                    return Path(self.conda_envs[stage_name]).exists()
                return False
            
            # Full check mode
            if stage_name == 'ffmpeg':
                # Check FFmpeg availability
                result = subprocess.run(['which', 'ffmpeg'], capture_output=True, text=True, timeout=5)
                return result.returncode == 0
            
            elif stage_name in self.conda_envs:
                # Check conda environment
                python_path = self.conda_envs[stage_name]
                if not Path(python_path).exists():
                    return False
                
                # Test basic import with reduced timeout
                test_commands = {
                    'xtts': 'import TTS; from TTS.api import TTS',
                    'sadtalker': 'import torch; import cv2; import numpy as np',
                    'insightface': 'import insightface; from insightface.app import FaceAnalysis',
                    'realesrgan': 'from realesrgan import RealESRGANer; from basicsr.archs.rrdbnet_arch import RRDBNet',
                    'codeformer': 'import torch; import cv2',
                    'manim': 'import manim'
                }
                
                if stage_name in test_commands:
                    result = subprocess.run(
                        [python_path, '-c', test_commands[stage_name]],
                        capture_output=True, text=True, timeout=10  # Reduced from 30 to 10 seconds
                    )
                    return result.returncode == 0
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Error checking {stage_name} availability: {e}")
            return False
    
    def get_stage_processing_mode(self, stage_name: str) -> str:
        """Get processing mode for a specific stage - always production."""
        return ProcessingMode.PRODUCTION.value
    
    def estimate_processing_time(self, stage_name: str, input_size: float = 1.0) -> float:
        """Estimate processing time for a stage in production mode."""
        # Base times for real model processing (in seconds)
        base_times = {
            'xtts': 30.0,
            'insightface': 5.0,
            'sadtalker': 60.0,
            'realesrgan': 120.0,
            'codeformer': 90.0,
            'manim': 45.0,
            'ffmpeg': 20.0
        }
        
        base_time = base_times.get(stage_name, 30.0)
        return base_time * input_size
    
    def get_stage_output_spec(self, stage_name: str) -> Dict[str, Any]:
        """Get expected output specification for a stage in production mode."""
        specs = {
            'xtts': {
                'production': {'audio_file': 'wav', 'metadata': 'json'}
            },
            'insightface': {
                'production': {'face_crop': 'jpg', 'face_metadata': 'json'}
            },
            'sadtalker': {
                'production': {'video_file': 'mp4', 'video_metadata': 'json'}
            },
            'realesrgan': {
                'production': {'enhanced_video': 'mp4', 'enhancement_metadata': 'json'}
            },
            'codeformer': {
                'production': {'restored_video': 'mp4', 'restoration_metadata': 'json'}
            },
            'manim': {
                'production': {'animation_video': 'mp4', 'animation_metadata': 'json'}
            },
            'ffmpeg': {
                'production': {'final_video': 'mp4', 'assembly_metadata': 'json'}
            }
        }
        
        return specs.get(stage_name, {}).get('production', {})
    
    def create_stage_result(self, stage_name: str, success: bool, **kwargs) -> Dict[str, Any]:
        """Create standardized stage result for production processing."""
        return {
            'stage': stage_name,
            'processing_mode': ProcessingMode.PRODUCTION.value,
            'success': success,
            'timestamp': time.time(),
            'real_models_used': True,
            'fallback_used': False,
            **kwargs
        }
    
    def validate_production_environment(self) -> Dict[str, Any]:
        """Validate that production environment is ready."""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        capabilities = self.assess_pipeline_capabilities()
        unavailable_stages = [
            stage for stage, info in capabilities['stage_details'].items() 
            if not info['available']
        ]
        
        if unavailable_stages:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Unavailable stages: {unavailable_stages}")
            validation_results['recommendations'].append("Install missing dependencies for unavailable stages")
        
        return validation_results
    
    def get_system_capabilities(self) -> Dict[str, Any]:
        """Get comprehensive system capabilities in expected format for frontend API."""
        # Get base capabilities
        base_capabilities = self.assess_pipeline_capabilities()
        
        # Format for frontend API expectations
        capabilities = {
            'processing_mode': base_capabilities['processing_mode'],
            'environment_status': {
                'available_environments': base_capabilities['available_stages'],
                'total_environments': base_capabilities['total_stages'],
                'environment_details': base_capabilities['stage_details']
            },
            'production_readiness': {
                'production_ready': base_capabilities['available_stages'] >= 5,  # At least 5 stages must be available
                'real_models_only': True,
                'fallback_disabled': True,
                'missing_stages': [
                    stage for stage, info in base_capabilities['stage_details'].items()
                    if not info['available']
                ]
            },
            'system_info': {
                'total_stages': base_capabilities['total_stages'],
                'available_stages': base_capabilities['available_stages'],
                'real_models_required': True,
                'timestamp': base_capabilities['timestamp']
            }
        }
        
        return capabilities
    
    def validate_environment_for_production(self) -> Dict[str, Any]:
        """Validate environment for production (alias for existing method)."""
        return self.validate_production_environment()


def main():
    """Test production mode manager."""
    print("Target: Testing Production Mode Manager - Real Models Only")
    print("=" * 60)
    
    manager = ProductionModeManager()
    
    # Assess capabilities
    print("Status: Assessing pipeline capabilities...")
    capabilities = manager.assess_pipeline_capabilities()
    
    print(f"Processing Mode: {capabilities['processing_mode']}")
    print(f"Available Stages: {capabilities['available_stages']}/{capabilities['total_stages']}")
    print(f"Real Models Required: {capabilities['real_models_required']}")
    
    print("\nSearch Stage Details:")
    for stage, info in capabilities['stage_details'].items():
        status = "[SUCCESS]" if info['available'] else "[ERROR]"
        print(f"  {status} {stage}: {info['mode']} mode, real models only")
    
    # Validate environment
    print("\n[EMOJI] Validating production environment...")
    validation = manager.validate_production_environment()
    
    if validation['valid']:
        print("[SUCCESS] Production environment is ready!")
    else:
        print("[ERROR] Production environment has issues:")
        for error in validation['errors']:
            print(f"  - {error}")


if __name__ == "__main__":
    main()