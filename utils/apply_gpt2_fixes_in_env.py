#!/usr/bin/env python3
"""
Apply GPT2InferenceModel fixes within the TTS environment
This script should be run in the voice cloning environment where transformers is available
"""

import sys
import warnings

def main():
    """Apply GPT2InferenceModel fixes and test them."""
    try:
        # Apply basic compatibility fixes first
        print("Tools Applying basic compatibility fixes...")
        
        # PyTorch weights_only fix
        import torch
        original_load = torch.load
        def patched_load(f, map_location=None, pickle_module=None, weights_only=None, **kwargs):
            if weights_only is None:
                weights_only = False
            return original_load(f, map_location=map_location, pickle_module=pickle_module, 
                               weights_only=weights_only, **kwargs)
        torch.load = patched_load
        
        # Suppress warnings
        warnings.filterwarnings("ignore", message=".*GPT2InferenceModel.*GenerationMixin.*")
        warnings.filterwarnings("ignore", message=".*trust_remote_code.*")
        warnings.filterwarnings("ignore", message=".*PreTrainedModel.*")
        
        print("[SUCCESS] Basic compatibility fixes applied")
        
        # Now apply GPT2InferenceModel specific fixes
        print("Tools Applying GPT2InferenceModel fixes...")
        
        from transformers import GenerationMixin, GPT2PreTrainedModel
        from TTS.tts.layers.xtts.gpt_inference import GPT2InferenceModel
        
        print(f"Endpoints Transformers version: {sys.modules['transformers'].__version__}")
        print(f"Endpoints GPT2InferenceModel base classes: {GPT2InferenceModel.__bases__}")
        
        # Check current state
        has_generate_before = hasattr(GPT2InferenceModel, 'generate')
        print(f"Endpoints GPT2InferenceModel has 'generate' method: {has_generate_before}")
        
        if not has_generate_before:
            print("Tools Adding GenerationMixin methods to GPT2InferenceModel...")
            
            # Essential generation methods to add
            generation_methods = [
                'generate', 'sample', 'greedy_search', 'beam_search', 'beam_sample',
                '_get_logits_warper', '_get_logits_processor', '_get_stopping_criteria',
                '_prepare_model_inputs', '_prepare_attention_mask_for_generation',
                '_expand_inputs_for_generation', '_extract_past_from_model_output',
                '_update_model_kwargs_for_generation', '_reorder_cache'
            ]
            
            # Copy methods from GenerationMixin to GPT2InferenceModel
            methods_added = 0
            for method_name in generation_methods:
                if hasattr(GenerationMixin, method_name):
                    method = getattr(GenerationMixin, method_name)
                    if callable(method):
                        setattr(GPT2InferenceModel, method_name, method)
                        methods_added += 1
            
            print(f"[SUCCESS] Added {methods_added} generation methods to GPT2InferenceModel")
            
            # Add necessary attributes
            if not hasattr(GPT2InferenceModel, '_supports_cache_class'):
                GPT2InferenceModel._supports_cache_class = False
            
            if not hasattr(GPT2InferenceModel, 'can_generate'):
                GPT2InferenceModel.can_generate = lambda self: True
            
            print("[SUCCESS] Added necessary generation attributes")
        
        # Verify the fix
        has_generate_after = hasattr(GPT2InferenceModel, 'generate')
        print(f"Endpoints GPT2InferenceModel has 'generate' method after fix: {has_generate_after}")
        
        if has_generate_after:
            print("[SUCCESS] GPT2InferenceModel fix successful!")
            print("GPT2_FIX_SUCCESS")
            return True
        else:
            print("[ERROR] GPT2InferenceModel fix failed!")
            print("GPT2_FIX_FAILED")
            return False
            
    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        print("GPT2_FIX_FAILED")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        print("GPT2_FIX_FAILED")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)