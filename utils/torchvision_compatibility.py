#!/usr/bin/env python3
"""
Torchvision Compatibility Layer
Fixes 'torchvision.transforms.functional_tensor' import errors in PyTorch 2.7+ / Torchvision 0.22+
Provides backward compatibility for legacy dependencies like BasicSR and GFPGAN
"""

import sys
import types
import warnings
import logging

logger = logging.getLogger(__name__)

def install_torchvision_compatibility():
    """
    Install compatibility layer for torchvision.transforms.functional_tensor
    
    This function addresses the issue where legacy packages (BasicSR, GFPGAN) 
    try to import from 'torchvision.transforms.functional_tensor' which was 
    deprecated in torchvision 0.15 and removed in 0.17+.
    
    Creates a virtual module that redirects imports to the correct location.
    """
    
    # Check if compatibility layer is already installed
    if 'torchvision.transforms.functional_tensor' in sys.modules:
        logger.debug("Torchvision compatibility layer already installed")
        return True
    
    try:
        # First, try to import from the deprecated location (for older torchvision)
        import torchvision.transforms.functional_tensor
        logger.info("[SUCCESS] Using native torchvision.transforms.functional_tensor (older torchvision)")
        return True
        
    except ImportError:
        # If that fails, create compatibility layer using functional module
        logger.info("Tools Installing torchvision compatibility layer for functional_tensor")
        
        try:
            # Import all functions from the current functional module
            import torchvision.transforms.functional as F
            
            # Create a virtual functional_tensor module
            functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
            
            # Map commonly used functions that were in functional_tensor
            # Based on official PyTorch documentation: https://pytorch.org/vision/stable/transforms.html
            compatibility_functions = [
                'rgb_to_grayscale',
                'to_tensor', 
                'normalize',
                'resize',
                'crop',
                'center_crop',
                'pad',
                'rotate',
                'affine',
                'hflip',
                'vflip',
                'adjust_brightness',
                'adjust_contrast',
                'adjust_gamma',
                'adjust_hue',
                'adjust_saturation',
                'gaussian_blur',
                'to_pil_image',
                'perspective',
                'elastic_transform',
                'posterize',
                'solarize',
                'autocontrast',
                'equalize',
                'invert'
            ]
            
            # Copy functions from functional to functional_tensor
            copied_functions = []
            for func_name in compatibility_functions:
                if hasattr(F, func_name):
                    setattr(functional_tensor, func_name, getattr(F, func_name))
                    copied_functions.append(func_name)
            
            # Add the virtual module to sys.modules
            sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor
            
            logger.info(f"[SUCCESS] Torchvision compatibility layer installed successfully")
            logger.debug(f"   Mapped functions: {copied_functions}")
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to install torchvision compatibility layer: {e}")
            return False

def ensure_basicsr_compatibility():
    """
    Ensure BasicSR compatibility with latest torchvision
    Specifically addresses rgb_to_grayscale import issues
    """
    
    try:
        # Test the specific import that causes issues in BasicSR
        from torchvision.transforms.functional_tensor import rgb_to_grayscale
        logger.debug("[SUCCESS] BasicSR compatibility: rgb_to_grayscale import successful")
        return True
        
    except ImportError as e:
        logger.warning(f"[WARNING] BasicSR compatibility issue detected: {e}")
        
        # Install compatibility layer if not already done
        if install_torchvision_compatibility():
            # Test again after installing compatibility layer
            try:
                from torchvision.transforms.functional_tensor import rgb_to_grayscale
                logger.info("[SUCCESS] BasicSR compatibility restored with compatibility layer")
                return True
            except ImportError:
                logger.error("[ERROR] BasicSR compatibility could not be restored")
                return False
        else:
            return False

def ensure_gfpgan_compatibility():
    """
    Ensure GFPGAN compatibility with latest torchvision
    Tests imports that GFPGAN typically uses
    """
    
    try:
        # Test GFPGAN's typical imports
        import gfpgan
        logger.debug("[SUCCESS] GFPGAN module imports successfully")
        
        # Try to create a basic GFPGAN instance to test deeper imports
        # from gfpgan import GFPGANer
        # Note: We'll test this in a controlled way to avoid loading models
        
        return True
        
    except ImportError as e:
        logger.error(f"[ERROR] GFPGAN compatibility issue: {e}")
        
        # If it's a functional_tensor issue, try installing compatibility layer
        if 'functional_tensor' in str(e):
            logger.info("Tools Attempting to fix GFPGAN functional_tensor compatibility")
            return install_torchvision_compatibility()
        else:
            logger.warning(f"[WARNING] GFPGAN issue not related to functional_tensor: {e}")
            return False

def test_compatibility_layer():
    """Test the compatibility layer with common operations"""
    
    logger.info("🧪 Testing torchvision compatibility layer...")
    
    try:
        # Test basic imports
        from torchvision.transforms.functional_tensor import rgb_to_grayscale
        logger.info("   [SUCCESS] rgb_to_grayscale import successful")
        
        # Test with a simple tensor operation
        import torch
        test_tensor = torch.randn(3, 32, 32)  # RGB tensor
        grayscale_result = rgb_to_grayscale(test_tensor)
        
        if grayscale_result.shape[0] == 1:  # Should be single channel
            logger.info("   [SUCCESS] rgb_to_grayscale operation successful")
            logger.info(f"   Input shape: {test_tensor.shape} -> Output shape: {grayscale_result.shape}")
            return True
        else:
            logger.error(f"   [ERROR] rgb_to_grayscale output shape incorrect: {grayscale_result.shape}")
            return False
            
    except Exception as e:
        logger.error(f"   [ERROR] Compatibility layer test failed: {e}")
        return False

def get_compatibility_status():
    """Get current compatibility status"""
    
    status = {
        'compatibility_layer_installed': False,
        'basicsr_compatible': False,
        'gfpgan_compatible': False,
        'test_passed': False,
        'torch_version': None,
        'torchvision_version': None
    }
    
    try:
        import torch
        import torchvision
        status['torch_version'] = torch.__version__
        status['torchvision_version'] = torchvision.__version__
        
        # Check if compatibility layer is installed
        status['compatibility_layer_installed'] = 'torchvision.transforms.functional_tensor' in sys.modules
        
        # Test BasicSR compatibility
        status['basicsr_compatible'] = ensure_basicsr_compatibility()
        
        # Test GFPGAN compatibility
        status['gfpgan_compatible'] = ensure_gfpgan_compatibility()
        
        # Test compatibility layer functionality
        if status['compatibility_layer_installed']:
            status['test_passed'] = test_compatibility_layer()
        
    except Exception as e:
        logger.error(f"Error getting compatibility status: {e}")
    
    return status

def print_compatibility_report():
    """Print a detailed compatibility report"""
    
    print("\n" + "=" * 60)
    print("Tools TORCHVISION COMPATIBILITY REPORT")
    print("=" * 60)
    
    status = get_compatibility_status()
    
    print(f"PyTorch Version: {status['torch_version']}")
    print(f"Torchvision Version: {status['torchvision_version']}")
    
    print(f"\nCompatibility Status:")
    print(f"  {'[SUCCESS]' if status['compatibility_layer_installed'] else '[ERROR]'} Compatibility Layer: {'Installed' if status['compatibility_layer_installed'] else 'Not Installed'}")
    print(f"  {'[SUCCESS]' if status['basicsr_compatible'] else '[ERROR]'} BasicSR: {'Compatible' if status['basicsr_compatible'] else 'Issues Detected'}")
    print(f"  {'[SUCCESS]' if status['gfpgan_compatible'] else '[ERROR]'} GFPGAN: {'Compatible' if status['gfpgan_compatible'] else 'Issues Detected'}")
    print(f"  {'[SUCCESS]' if status['test_passed'] else '[ERROR]'} Functionality Test: {'Passed' if status['test_passed'] else 'Failed'}")
    
    overall_status = all([
        status['compatibility_layer_installed'],
        status['basicsr_compatible'], 
        status['test_passed']
    ])
    
    print(f"\nTarget: Overall Status: {'[SUCCESS] COMPATIBLE' if overall_status else '[ERROR] ISSUES DETECTED'}")
    
    if not overall_status:
        print("\nINFO: Recommended Actions:")
        if not status['compatibility_layer_installed']:
            print("  • Run install_torchvision_compatibility()")
        if not status['basicsr_compatible']:
            print("  • Check BasicSR installation and version")
        if not status['test_passed']:
            print("  • Review compatibility layer implementation")
    
    return overall_status

# Auto-install compatibility layer when module is imported
def auto_install():
    """Automatically install compatibility layer when module is imported"""
    try:
        success = install_torchvision_compatibility()
        if success:
            logger.info("Tools Torchvision compatibility layer auto-installed")
        return success
    except Exception as e:
        logger.warning(f"[WARNING] Auto-install failed: {e}")
        return False

# Main execution for testing
if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("🧪 Testing Torchvision Compatibility Layer")
    
    # Install compatibility layer
    success = install_torchvision_compatibility()
    
    if success:
        # Generate and print report
        overall_compatible = print_compatibility_report()
        
        if overall_compatible:
            print("\nSUCCESS SUCCESS: Torchvision compatibility layer working!")
        else:
            print("\n[WARNING] PARTIAL: Some compatibility issues remain")
    else:
        print("\n[ERROR] FAILED: Could not install compatibility layer")

# Auto-install when imported (can be disabled by setting environment variable)
import os
if os.environ.get('DISABLE_TORCHVISION_AUTOINSTALL', '').lower() not in ('1', 'true', 'yes'):
    auto_install()