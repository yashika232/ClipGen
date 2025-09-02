#!/usr/bin/env python3
"""
Real-ESRGAN Model Manager
Handles model downloads, local fallbacks, and URL updates for Real-ESRGAN models
"""

import os
import sys
import requests
import hashlib
from pathlib import Path
import logging
from typing import Dict, Optional, List
import json

# Real-ESRGAN Model Registry with working URLs (updated 2025)
REALESRGAN_MODEL_REGISTRY = {
    "RealESRGAN_x4plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "description": "4x upscaling model for general images",
        "scale": 4,
        "type": "general"
    },
    "RealESRGAN_x2plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "description": "2x upscaling model for general images", 
        "scale": 2,
        "type": "general"
    },
    "realesr-general-x4v3": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "description": "Tiny 4x model (low memory usage)",
        "scale": 4,
        "type": "general_tiny"
    },
    "RealESRGAN_x4plus_anime_6B": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "description": "4x upscaling model optimized for anime images",
        "scale": 4,
        "type": "anime"
    },
    "realesr-animevideov3": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
        "description": "Anime video model with XS size",
        "scale": 4,
        "type": "anime_video"
    }
}

class RealESRGANModelManager:
    """Manages Real-ESRGAN model downloads and local storage."""
    
    def __init__(self, models_dir: str = None):
        self.models_dir = Path(models_dir) if models_dir else Path.home() / ".cache" / "realesrgan_models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def download_model(self, model_name: str, force_download: bool = False) -> Optional[str]:
        """
        Download a Real-ESRGAN model with progress tracking.
        
        Args:
            model_name: Name of the model to download
            force_download: Force re-download even if model exists
            
        Returns:
            Path to downloaded model file, or None if failed
        """
        if model_name not in REALESRGAN_MODEL_REGISTRY:
            self.logger.error(f"Unknown model: {model_name}")
            return None
            
        model_info = REALESRGAN_MODEL_REGISTRY[model_name]
        model_path = self.models_dir / f"{model_name}.pth"
        
        # Check if model already exists
        if model_path.exists() and not force_download:
            self.logger.info(f"[SUCCESS] Model already exists: {model_path}")
            return str(model_path)
        
        # Download model
        try:
            self.logger.info(f"Download Downloading {model_name} from {model_info['url']}")
            
            response = requests.get(model_info['url'], stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            print(f"\rDownload Downloading {model_name}: {progress:.1f}%", end='', flush=True)
            
            print()  # New line after progress
            self.logger.info(f"[SUCCESS] Downloaded {model_name}: {model_path}")
            return str(model_path)
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to download {model_name}: {e}")
            if model_path.exists():
                model_path.unlink()  # Remove partial download
            return None
    
    def get_model_path(self, model_name: str, auto_download: bool = True) -> Optional[str]:
        """
        Get path to a model, downloading if necessary.
        
        Args:
            model_name: Name of the model
            auto_download: Automatically download if not found locally
            
        Returns:
            Path to model file, or None if not available
        """
        model_path = self.models_dir / f"{model_name}.pth"
        
        if model_path.exists():
            return str(model_path)
        
        if auto_download:
            return self.download_model(model_name)
        
        return None
    
    def list_available_models(self) -> Dict[str, Dict]:
        """List all available models in the registry."""
        return REALESRGAN_MODEL_REGISTRY.copy()
    
    def list_local_models(self) -> List[str]:
        """List models that are stored locally."""
        local_models = []
        for model_name in REALESRGAN_MODEL_REGISTRY:
            model_path = self.models_dir / f"{model_name}.pth"
            if model_path.exists():
                local_models.append(model_name)
        return local_models
    
    def verify_model_integrity(self, model_name: str) -> bool:
        """Verify that a model file is not corrupted."""
        model_path = self.models_dir / f"{model_name}.pth"
        
        if not model_path.exists():
            return False
        
        try:
            # Basic check: file size should be reasonable (> 1MB)
            file_size = model_path.stat().st_size
            if file_size < 1024 * 1024:  # Less than 1MB is suspicious
                self.logger.warning(f"[WARNING] Model file seems too small: {file_size} bytes")
                return False
            
            # Try to load with torch to verify format
            try:
                import torch
                torch.load(model_path, map_location='cpu')
                return True
            except Exception as e:
                self.logger.warning(f"[WARNING] Model file appears corrupted: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"[ERROR] Error verifying model: {e}")
            return False

def patch_realesrgan_urls():
    """
    Patch Real-ESRGAN library to use correct model URLs.
    This function patches the library to use the working URLs from our registry.
    """
    try:
        # Try to patch the Real-ESRGAN library's model URLs
        import realesrgan
        from realesrgan.utils import RealESRGANer
        
        # Store original method
        original_init = RealESRGANer.__init__
        
        def patched_init(self, scale, model_path, dni_weight=None, model=None, tile=0,
                        tile_pad=10, pre_pad=0, half=False, device=None, gpu_id=None):
            """Patched RealESRGANer init that uses correct model URLs."""
            
            # If model_path is a model name without .pth, try to get from registry
            if model_path and not model_path.endswith('.pth'):
                model_name = model_path
                if model_name in REALESRGAN_MODEL_REGISTRY:
                    manager = RealESRGANModelManager()
                    downloaded_path = manager.get_model_path(model_name, auto_download=True)
                    if downloaded_path:
                        model_path = downloaded_path
                        logging.info(f"[SUCCESS] Using downloaded model: {model_path}")
            
            # Call original init with updated model_path
            original_init(self, scale=scale, model_path=model_path, dni_weight=dni_weight,
                         model=model, tile=tile, tile_pad=tile_pad, pre_pad=pre_pad,
                         half=half, device=device, gpu_id=gpu_id)
        
        # Apply patch
        RealESRGANer.__init__ = patched_init
        logging.info("[SUCCESS] Patched Real-ESRGAN library with correct model URLs")
        return True
        
    except ImportError:
        logging.warning("[WARNING] Real-ESRGAN library not available for patching")
        return False
    except Exception as e:
        logging.error(f"[ERROR] Failed to patch Real-ESRGAN library: {e}")
        return False

def create_model_config_file(output_path: str):
    """Create a configuration file with model information."""
    config = {
        "models": REALESRGAN_MODEL_REGISTRY,
        "default_model": "RealESRGAN_x2plus",
        "cache_dir": str(Path.home() / ".cache" / "realesrgan_models"),
        "updated": "2025-07-07"
    }
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logging.info(f"[SUCCESS] Created model config file: {output_path}")

def main():
    """Test and demonstrate model manager functionality."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-ESRGAN Model Manager")
    parser.add_argument("--download", help="Download a specific model")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--local", action="store_true", help="List local models")
    parser.add_argument("--verify", help="Verify a local model")
    parser.add_argument("--patch", action="store_true", help="Test URL patching")
    parser.add_argument("--config", help="Create config file at specified path")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    manager = RealESRGANModelManager()
    
    if args.list:
        print("Endpoints Available Real-ESRGAN models:")
        for name, info in manager.list_available_models().items():
            print(f"  {name}: {info['description']} ({info['scale']}x, {info['type']})")
    
    if args.local:
        local_models = manager.list_local_models()
        print(f"Storage Local models ({len(local_models)}):")
        for model in local_models:
            print(f"  [SUCCESS] {model}")
    
    if args.download:
        model_path = manager.download_model(args.download)
        if model_path:
            print(f"[SUCCESS] Downloaded: {model_path}")
        else:
            print(f"[ERROR] Failed to download: {args.download}")
    
    if args.verify:
        if manager.verify_model_integrity(args.verify):
            print(f"[SUCCESS] Model verified: {args.verify}")
        else:
            print(f"[ERROR] Model verification failed: {args.verify}")
    
    if args.patch:
        if patch_realesrgan_urls():
            print("[SUCCESS] Real-ESRGAN URL patching successful")
        else:
            print("[ERROR] Real-ESRGAN URL patching failed")
    
    if args.config:
        create_model_config_file(args.config)

if __name__ == "__main__":
    main()