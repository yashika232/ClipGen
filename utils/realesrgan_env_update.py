#!/usr/bin/env python3
"""
Real-ESRGAN Environment Update Utility
Updates the Real-ESRGAN conda environment with working models and patches
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import logging
import json

def copy_downloaded_models_to_env(models_dir: str, target_env: str):
    """
    Copy downloaded models to the Real-ESRGAN conda environment.
    
    Args:
        models_dir: Directory containing downloaded models
        target_env: Name of the conda environment
    """
    try:
        # Get conda environment path
        result = subprocess.run(
            ["conda", "info", "--envs", "--json"],
            capture_output=True,
            text=True,
            check=True
        )
        
        env_info = json.loads(result.stdout)
        env_path = None
        
        # Find the target environment path
        for env in env_info["envs"]:
            if target_env in env:
                env_path = Path(env)
                break
        
        if not env_path:
            logging.error(f"[ERROR] Environment not found: {target_env}")
            return False
        
        # Find weights directory in environment
        weights_dirs = [
            env_path / "lib" / "python3.9" / "site-packages" / "weights",
            env_path / "lib" / "python3.10" / "site-packages" / "weights",
            env_path / "lib" / "python3.11" / "site-packages" / "weights",
            env_path / "weights"
        ]
        
        target_weights_dir = None
        for weights_dir in weights_dirs:
            if weights_dir.parent.exists():
                target_weights_dir = weights_dir
                break
        
        if not target_weights_dir:
            # Create weights directory
            target_weights_dir = env_path / "weights"
            target_weights_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Assets: Target weights directory: {target_weights_dir}")
        
        # Copy models
        models_source = Path(models_dir)
        copied_models = []
        
        for model_file in models_source.glob("*.pth"):
            target_path = target_weights_dir / model_file.name
            shutil.copy2(str(model_file), str(target_path))
            copied_models.append(model_file.name)
            logging.info(f"[SUCCESS] Copied model: {model_file.name}")
        
        logging.info(f"[SUCCESS] Copied {len(copied_models)} models to {target_env} environment")
        return True
        
    except Exception as e:
        logging.error(f"[ERROR] Failed to copy models: {e}")
        return False

def patch_realesrgan_environment(env_name: str):
    """
    Patch the Real-ESRGAN environment to use correct model URLs.
    
    Args:
        env_name: Name of the conda environment
    """
    try:
        # Create a patch script for the environment
        patch_script = f"""
import sys
import os
from pathlib import Path

# Model URL corrections
MODEL_URL_FIXES = {{
    "RealESRGAN_x2plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "realesr-general-x4v3.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
}}

def patch_download_utils():
    try:
        # Import after being in the environment
        from basicsr.utils import download_util
        
        # Store original function
        original_load_file_from_url = download_util.load_file_from_url
        
        def patched_load_file_from_url(url, model_dir=None, progress=True, file_name=None, save_dir=None):
            # Check if we need to fix the URL
            if file_name and file_name in MODEL_URL_FIXES:
                corrected_url = MODEL_URL_FIXES[file_name]
                print(f"Tools Correcting URL for {{file_name}}: {{corrected_url}}")
                url = corrected_url
            
            return original_load_file_from_url(url, model_dir, progress, file_name, save_dir)
        
        # Apply patch
        download_util.load_file_from_url = patched_load_file_from_url
        print("[SUCCESS] Applied Real-ESRGAN URL patches")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to patch download utils: {{e}}")
        return False

if __name__ == "__main__":
    patch_download_utils()
"""
        
        # Run patch in the environment
        result = subprocess.run([
            "conda", "run", "-n", env_name, "python", "-c", patch_script
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logging.info("[SUCCESS] Successfully patched Real-ESRGAN environment")
            return True
        else:
            logging.error(f"[ERROR] Failed to patch environment: {result.stderr}")
            return False
            
    except Exception as e:
        logging.error(f"[ERROR] Error patching environment: {e}")
        return False

def test_realesrgan_in_env(env_name: str):
    """
    Test Real-ESRGAN functionality in the environment.
    
    Args:
        env_name: Name of the conda environment
    """
    test_script = """
import sys
sys.path.insert(0, '/Users/aryanjain/Documents/video-synthesis-pipeline copy/utils')

try:
    from realesrgan_model_manager import RealESRGANModelManager, patch_realesrgan_urls
    
    # Apply URL patches
    patch_realesrgan_urls()
    
    # Test model manager
    manager = RealESRGANModelManager()
    model_path = manager.get_model_path("RealESRGAN_x2plus", auto_download=False)
    
    if model_path:
        print(f"[SUCCESS] Model available: {model_path}")
    else:
        print("[ERROR] Model not available")
    
    # Test Real-ESRGAN library
    from realesrgan import RealESRGANer
    
    upsampler = RealESRGANer(
        scale=2,
        model_path=model_path or "RealESRGAN_x2plus",
        model=None,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        device='cpu'
    )
    
    print("[SUCCESS] Real-ESRGAN initialized successfully")
    print("REALESRGAN_TEST_SUCCESS")
    
except Exception as e:
    print(f"[ERROR] Real-ESRGAN test failed: {e}")
    import traceback
    traceback.print_exc()
    print("REALESRGAN_TEST_FAILED")
"""
    
    try:
        result = subprocess.run([
            "conda", "run", "-n", env_name, "python", "-c", test_script
        ], capture_output=True, text=True, timeout=60)
        
        if "REALESRGAN_TEST_SUCCESS" in result.stdout:
            logging.info("[SUCCESS] Real-ESRGAN test passed in environment")
            return True
        else:
            logging.error(f"[ERROR] Real-ESRGAN test failed: {result.stderr}")
            logging.error(f"[ERROR] Stdout: {result.stdout}")
            return False
            
    except Exception as e:
        logging.error(f"[ERROR] Error testing Real-ESRGAN: {e}")
        return False

def update_realesrgan_environment():
    """Main function to update the Real-ESRGAN environment with working models."""
    
    logging.basicConfig(level=logging.INFO)
    
    # Configuration
    env_name = "realesrgan"
    models_cache_dir = Path.home() / ".cache" / "realesrgan_models"
    
    logging.info("Tools Updating Real-ESRGAN environment with working models...")
    
    # Step 1: Ensure models are downloaded
    from realesrgan_model_manager import RealESRGANModelManager
    manager = RealESRGANModelManager()
    
    # Download essential models
    essential_models = ["RealESRGAN_x2plus", "RealESRGAN_x4plus"]
    for model_name in essential_models:
        model_path = manager.get_model_path(model_name, auto_download=True)
        if model_path:
            logging.info(f"[SUCCESS] Model ready: {model_name}")
        else:
            logging.error(f"[ERROR] Failed to get model: {model_name}")
    
    # Step 2: Copy models to environment
    if copy_downloaded_models_to_env(str(models_cache_dir), env_name):
        logging.info("[SUCCESS] Models copied to environment")
    else:
        logging.error("[ERROR] Failed to copy models")
        return False
    
    # Step 3: Test environment
    if test_realesrgan_in_env(env_name):
        logging.info("[SUCCESS] Real-ESRGAN environment updated successfully")
        return True
    else:
        logging.error("[ERROR] Environment update failed")
        return False

def main():
    """CLI interface for the environment updater."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Update Real-ESRGAN environment")
    parser.add_argument("--env", default="realesrgan", help="Conda environment name")
    parser.add_argument("--test-only", action="store_true", help="Only test environment")
    parser.add_argument("--copy-models", action="store_true", help="Only copy models")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.test_only:
        success = test_realesrgan_in_env(args.env)
        print(f"Test result: {'[SUCCESS] PASSED' if success else '[ERROR] FAILED'}")
    elif args.copy_models:
        models_dir = Path.home() / ".cache" / "realesrgan_models"
        success = copy_downloaded_models_to_env(str(models_dir), args.env)
        print(f"Copy result: {'[SUCCESS] SUCCESS' if success else '[ERROR] FAILED'}")
    else:
        success = update_realesrgan_environment()
        print(f"Update result: {'[SUCCESS] SUCCESS' if success else '[ERROR] FAILED'}")

if __name__ == "__main__":
    main()