#!/usr/bin/env python3
"""
Apply TorchVision compatibility fixes within the current environment.
This script should be run before importing BasicSR or CodeFormer modules.
"""

import sys
import logging
import os

def apply_torchvision_fixes():
    """Apply TorchVision compatibility fixes - to be called within current environment."""
    try:
        # Add utils to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        
        from torchvision_compatibility_fix import apply_torchvision_compatibility_fixes
        
        # Apply the fixes
        success = apply_torchvision_compatibility_fixes()
        
        if success:
            print("[SUCCESS] TorchVision compatibility fixes applied successfully")
            
            # Test imports
            try:
                from torchvision.transforms.functional_tensor import rgb_to_grayscale
                print("[SUCCESS] torchvision.transforms.functional_tensor import successful")
                
                import basicsr.data.degradations
                print("[SUCCESS] basicsr.data.degradations import successful")
                
                print("TORCHVISION_FIX_SUCCESS")
                return True
                
            except ImportError as e:
                print(f"[WARNING] Import test failed: {e}")
                print("TORCHVISION_FIX_PARTIAL")
                return True  # Fixes applied but some imports still fail
                
        else:
            print("[ERROR] Failed to apply TorchVision compatibility fixes")
            print("TORCHVISION_FIX_FAILED")
            return False
            
    except Exception as e:
        print(f"[ERROR] Error applying TorchVision fixes: {e}")
        import traceback
        traceback.print_exc()
        print("TORCHVISION_FIX_ERROR")
        return False

if __name__ == "__main__":
    success = apply_torchvision_fixes()
    sys.exit(0 if success else 1)