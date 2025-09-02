#!/usr/bin/env python3
"""
TorchVision Compatibility Fix
Resolves compatibility issues with newer TorchVision versions (0.17+) where functional_tensor was removed
"""

import sys
import logging
from pathlib import Path


def patch_torchvision_functional_tensor():
    """
    Create a compatibility layer for torchvision.transforms.functional_tensor.
    
    This module was deprecated in TorchVision 0.15 and removed in 0.17.
    Functions have been moved to torchvision.transforms.functional.
    """
    try:
        import torchvision
        from packaging import version
        
        # Check if we're using a TorchVision version that removed functional_tensor
        torchvision_version = version.parse(torchvision.__version__)
        if torchvision_version >= version.parse("0.17.0"):
            
            # Try to import functional_tensor to see if it already exists
            try:
                import torchvision.transforms.functional_tensor
                logging.info("[SUCCESS] torchvision.transforms.functional_tensor already available")
                return True
            except ImportError:
                # Create compatibility module
                _create_functional_tensor_compatibility()
                logging.info("[SUCCESS] Created torchvision.transforms.functional_tensor compatibility layer")
                return True
        else:
            logging.info(f"[SUCCESS] TorchVision {torchvision.__version__} has native functional_tensor support")
            return True
            
    except Exception as e:
        logging.error(f"[ERROR] Failed to patch TorchVision functional_tensor: {e}")
        return False


def _create_functional_tensor_compatibility():
    """Create a functional_tensor compatibility module."""
    import types
    import torchvision.transforms.functional as F
    
    # Create a mock functional_tensor module
    functional_tensor_module = types.ModuleType('torchvision.transforms.functional_tensor')
    
    # Map commonly used functions that were moved from functional_tensor to functional
    function_mappings = {
        'rgb_to_grayscale': getattr(F, 'rgb_to_grayscale', None),
        'adjust_brightness': getattr(F, 'adjust_brightness', None),
        'adjust_contrast': getattr(F, 'adjust_contrast', None),
        'adjust_saturation': getattr(F, 'adjust_saturation', None),
        'adjust_hue': getattr(F, 'adjust_hue', None),
        'adjust_gamma': getattr(F, 'adjust_gamma', None),
        'normalize': getattr(F, 'normalize', None),
        'resize': getattr(F, 'resize', None),
        'crop': getattr(F, 'crop', None),
        'center_crop': getattr(F, 'center_crop', None),
        'resized_crop': getattr(F, 'resized_crop', None),
        'horizontal_flip': getattr(F, 'hflip', None),
        'vertical_flip': getattr(F, 'vflip', None),
        'rotate': getattr(F, 'rotate', None),
        'affine': getattr(F, 'affine', None),
        'perspective': getattr(F, 'perspective', None),
        'elastic_transform': getattr(F, 'elastic_transform', None),
        'gaussian_blur': getattr(F, 'gaussian_blur', None),
    }
    
    # Add available functions to the mock module
    for func_name, func in function_mappings.items():
        if func is not None:
            setattr(functional_tensor_module, func_name, func)
    
    # Special handling for functions that might have different names
    if hasattr(F, 'to_grayscale') and not hasattr(functional_tensor_module, 'rgb_to_grayscale'):
        setattr(functional_tensor_module, 'rgb_to_grayscale', F.to_grayscale)
    
    # Add the module to sys.modules
    sys.modules['torchvision.transforms.functional_tensor'] = functional_tensor_module
    
    # Also add to the torchvision.transforms namespace
    import torchvision.transforms
    torchvision.transforms.functional_tensor = functional_tensor_module


def patch_basicsr_degradations():
    """
    Specifically patch BasicSR's degradations.py file to use the correct import.
    """
    try:
        # First apply the general TorchVision patch
        patch_torchvision_functional_tensor()
        
        # Additional specific patches for BasicSR
        import torchvision.transforms.functional as F
        
        # Make sure rgb_to_grayscale is available
        if not hasattr(F, 'rgb_to_grayscale'):
            # If rgb_to_grayscale is not in functional, it might be to_grayscale
            if hasattr(F, 'to_grayscale'):
                F.rgb_to_grayscale = F.to_grayscale
                logging.info("[SUCCESS] Mapped rgb_to_grayscale to to_grayscale")
        
        logging.info("[SUCCESS] Applied BasicSR degradations compatibility patches")
        return True
        
    except Exception as e:
        logging.error(f"[ERROR] Failed to patch BasicSR degradations: {e}")
        return False


def apply_torchvision_compatibility_fixes():
    """
    Apply all TorchVision compatibility fixes.
    This should be called before importing BasicSR or CodeFormer.
    """
    try:
        # Apply the main functional_tensor patch
        success = patch_torchvision_functional_tensor()
        
        # Apply BasicSR-specific patches
        if success:
            success = patch_basicsr_degradations()
        
        if success:
            logging.info("[SUCCESS] TorchVision compatibility fixes applied successfully")
        else:
            logging.warning("[WARNING] Some TorchVision compatibility fixes failed")
        
        return success
        
    except Exception as e:
        logging.error(f"[ERROR] Failed to apply TorchVision compatibility fixes: {e}")
        return False


def test_torchvision_compatibility():
    """Test if TorchVision compatibility fixes work."""
    try:
        # Apply fixes first
        apply_torchvision_compatibility_fixes()
        
        # Test importing functional_tensor
        from torchvision.transforms.functional_tensor import rgb_to_grayscale
        logging.info("[SUCCESS] torchvision.transforms.functional_tensor.rgb_to_grayscale import successful")
        
        # Test BasicSR import
        import basicsr.data.degradations
        logging.info("[SUCCESS] basicsr.data.degradations import successful")
        
        # Test CodeFormer specific imports
        import basicsr.archs.codeformer_arch
        logging.info("[SUCCESS] basicsr.archs.codeformer_arch import successful")
        
        return True
        
    except Exception as e:
        logging.error(f"[ERROR] TorchVision compatibility test failed: {e}")
        return False


def main():
    """Test the TorchVision compatibility fixes."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TorchVision Compatibility Fix")
    parser.add_argument("--test", action="store_true", help="Test compatibility fixes")
    parser.add_argument("--apply", action="store_true", help="Apply compatibility fixes")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.test:
        success = test_torchvision_compatibility()
        print(f"Test result: {'[SUCCESS] PASSED' if success else '[ERROR] FAILED'}")
    elif args.apply:
        success = apply_torchvision_compatibility_fixes()
        print(f"Apply result: {'[SUCCESS] SUCCESS' if success else '[ERROR] FAILED'}")
    else:
        print("Use --test to test compatibility or --apply to apply fixes")


if __name__ == "__main__":
    main()