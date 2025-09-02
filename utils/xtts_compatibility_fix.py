#!/usr/bin/env python3
"""
XTTS Compatibility Fix for GPT2InferenceModel
Fixes the 'GPT2InferenceModel' object has no attribute 'generate' issue
Enhanced version for Transformers 4.53.0+ compatibility
"""

import torch
import warnings
import sys
from types import MethodType

def apply_xtts_compatibility_fixes():
    """Apply enhanced compatibility fixes for XTTS-v2 and PyTorch/Transformers versions."""
    
    # Fix 1: PyTorch weights_only parameter compatibility
    original_load = torch.load
    def patched_load(f, map_location=None, pickle_module=None, weights_only=None, **kwargs):
        if weights_only is None:
            weights_only = False  # Use False for compatibility with older models
        return original_load(f, map_location=map_location, pickle_module=pickle_module, 
                           weights_only=weights_only, **kwargs)
    torch.load = patched_load
    
    # Fix 2: Suppress specific warnings that don't affect functionality
    warnings.filterwarnings("ignore", message=".*GPT2InferenceModel.*GenerationMixin.*")
    warnings.filterwarnings("ignore", message=".*trust_remote_code.*")
    warnings.filterwarnings("ignore", message=".*PreTrainedModel.*")
    warnings.filterwarnings("ignore", message=".*torch.load.*weights_only.*")
    
    # Fix 3: Set up safe globals for torch serialization if needed
    try:
        from TTS.tts.configs.xtts_config import XttsConfig
        torch.serialization.add_safe_globals([XttsConfig])
    except ImportError:
        pass  # XttsConfig may not be available in all environments
    
    print("[SUCCESS] Enhanced XTTS compatibility fixes applied")

def apply_gpt2_inference_fixes():
    """Apply GPT2InferenceModel specific fixes - to be called within TTS environment."""
    try:
        # Import required modules (only available in TTS environment)
        from transformers import GenerationMixin, GPT2PreTrainedModel
        from TTS.tts.layers.xtts.gpt_inference import GPT2InferenceModel
        
        # Check if GPT2InferenceModel already has generate method
        if not hasattr(GPT2InferenceModel, 'generate'):
            # Add GenerationMixin methods to GPT2InferenceModel
            _add_generation_methods_to_gpt2_inference(GPT2InferenceModel, GenerationMixin)
            print("[SUCCESS] Enhanced GPT2InferenceModel with GenerationMixin methods")
            
            # Also ensure it inherits from GenerationMixin in the class hierarchy  
            if GenerationMixin not in GPT2InferenceModel.__mro__:
                # Create a new class that properly inherits from both
                class EnhancedGPT2InferenceModel(GPT2InferenceModel, GenerationMixin):
                    pass
                
                # Replace the original class in the module
                import TTS.tts.layers.xtts.gpt_inference as gpt_inference_module
                gpt_inference_module.GPT2InferenceModel = EnhancedGPT2InferenceModel
                print("[SUCCESS] Enhanced GPT2InferenceModel class hierarchy")
                
            return True
        else:
            print("[SUCCESS] GPT2InferenceModel already has generate method")
            return True
            
    except ImportError as e:
        print(f"[WARNING] Could not enhance GPT2InferenceModel: {e}")
        print("[WARNING] Will rely on fallback mechanism")
        return False
    except Exception as e:
        print(f"[WARNING] Error applying GPT2 fixes: {e}")
        return False

def _add_generation_methods_to_gpt2_inference(gpt2_inference_class, generation_mixin_class):
    """Add GenerationMixin methods to GPT2InferenceModel class."""
    
    # List of essential generation methods to add
    generation_methods = [
        'generate', 'sample', 'greedy_search', 'beam_search', 'beam_sample',
        'group_beam_search', 'constrained_beam_search', '_get_logits_warper',
        '_get_logits_processor', '_get_stopping_criteria', '_prepare_model_inputs',
        '_prepare_attention_mask_for_generation', '_prepare_encoder_decoder_kwargs_for_generation',
        '_expand_inputs_for_generation', '_extract_past_from_model_output',
        '_update_model_kwargs_for_generation', '_reorder_cache'
    ]
    
    # Copy methods from GenerationMixin to GPT2InferenceModel
    for method_name in generation_methods:
        if hasattr(generation_mixin_class, method_name):
            method = getattr(generation_mixin_class, method_name)
            if callable(method):
                # Copy the method to the target class
                setattr(gpt2_inference_class, method_name, method)
    
    # Ensure the class has the necessary attributes for generation
    if not hasattr(gpt2_inference_class, '_supports_cache_class'):
        gpt2_inference_class._supports_cache_class = False
    
    if not hasattr(gpt2_inference_class, 'can_generate'):
        gpt2_inference_class.can_generate = lambda self: True
    
    # Add a method to check if generation is properly set up
    def _verify_generation_setup(self):
        """Verify that generation methods are properly configured."""
        required_methods = ['generate', 'prepare_inputs_for_generation']
        missing_methods = [m for m in required_methods if not hasattr(self, m)]
        if missing_methods:
            raise AttributeError(f"GPT2InferenceModel missing required methods: {missing_methods}")
        return True
    
    gpt2_inference_class._verify_generation_setup = _verify_generation_setup

def create_xtts_wrapper():
    """Create a wrapper for XTTS that handles compatibility issues."""
    
    class XTTSWrapper:
        def __init__(self, model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False):
            self.model_name = model_name
            self.gpu = gpu
            self.model = None
            self.synthesizer = None
            self.available = False
            self.gpt2_fix_applied = False
            
            # Initialize with compatibility fixes
            self._initialize_with_fixes()
        
        def _initialize_with_fixes(self):
            """Initialize XTTS model with enhanced compatibility fixes."""
            try:
                apply_xtts_compatibility_fixes()
                
                from TTS.api import TTS
                self.model = TTS(self.model_name, gpu=self.gpu)
                self.synthesizer = self.model
                self.available = True
                
                # Apply runtime GPT2 fixes
                self._apply_runtime_gpt2_fixes()
                
                print(f"[SUCCESS] XTTS model loaded: {self.model_name}")
                
            except Exception as e:
                print(f"[ERROR] XTTS loading failed: {e}")
                self.model = None
                self.available = False
        
        def _apply_runtime_gpt2_fixes(self):
            """Apply GPT2InferenceModel fixes at runtime."""
            try:
                # Check if the model has a TTS component with GPT2InferenceModel
                if hasattr(self.model, 'synthesizer') and hasattr(self.model.synthesizer, 'tts_model'):
                    tts_model = self.model.synthesizer.tts_model
                    
                    # Look for GPT2InferenceModel in the TTS model
                    if hasattr(tts_model, 'gpt') and hasattr(tts_model.gpt, '__class__'):
                        gpt_class = tts_model.gpt.__class__
                        if 'GPT2InferenceModel' in str(gpt_class):
                            # Apply fixes directly to the loaded model instance
                            self._fix_gpt2_instance(tts_model.gpt)
                            print("[SUCCESS] Applied runtime GPT2InferenceModel fixes")
                            self.gpt2_fix_applied = True
                        
            except Exception as e:
                print(f"[WARNING] Could not apply runtime GPT2 fixes: {e}")
                self.gpt2_fix_applied = False
        
        def _fix_gpt2_instance(self, gpt2_instance):
            """Apply GenerationMixin methods to a specific GPT2InferenceModel instance."""
            try:
                from transformers import GenerationMixin
                
                # Check if instance already has generate method
                if hasattr(gpt2_instance, 'generate'):
                    return True
                
                # Essential generation methods to add
                generation_methods = [
                    'generate', 'sample', 'greedy_search', 'beam_search', 'beam_sample',
                    '_get_logits_warper', '_get_logits_processor', '_get_stopping_criteria',
                    '_prepare_model_inputs', '_prepare_attention_mask_for_generation',
                    '_expand_inputs_for_generation', '_extract_past_from_model_output',
                    '_update_model_kwargs_for_generation', '_reorder_cache'
                ]
                
                # Copy methods from GenerationMixin to the instance
                methods_added = 0
                for method_name in generation_methods:
                    if hasattr(GenerationMixin, method_name):
                        method = getattr(GenerationMixin, method_name)
                        if callable(method):
                            # Bind method to the instance
                            import types
                            bound_method = types.MethodType(method, gpt2_instance)
                            setattr(gpt2_instance, method_name, bound_method)
                            methods_added += 1
                
                # Add necessary attributes
                if not hasattr(gpt2_instance, '_supports_cache_class'):
                    gpt2_instance._supports_cache_class = False
                
                if not hasattr(gpt2_instance, 'can_generate'):
                    gpt2_instance.can_generate = lambda: True
                
                print(f"[SUCCESS] Fixed GPT2InferenceModel instance with {methods_added} methods")
                return True
                
            except Exception as e:
                print(f"[WARNING] Failed to fix GPT2InferenceModel instance: {e}")
                return False
        
        def tts(self, text, language="en", speed=1.0, **kwargs):
            """TTS method that returns audio data directly."""
            if not self.available:
                raise RuntimeError("XTTS model not available")
            
            try:
                # Re-apply runtime fixes before synthesis
                if not self.gpt2_fix_applied:
                    self._apply_runtime_gpt2_fixes()
                
                # Use the model's tts method if available
                if hasattr(self.model, 'tts'):
                    return self.model.tts(text=text, language=language, speed=speed, **kwargs)
                else:
                    # Use synthesizer's tts method as fallback
                    return self.synthesizer.tts(text=text, language=language, speed=speed, **kwargs)
                    
            except AttributeError as e:
                if "'GPT2InferenceModel' object has no attribute 'generate'" in str(e):
                    print("[WARNING] GPT2InferenceModel compatibility issue detected")
                    success = self._emergency_gpt2_fix()
                    if success:
                        try:
                            if hasattr(self.model, 'tts'):
                                return self.model.tts(text=text, language=language, speed=speed, **kwargs)
                            else:
                                return self.synthesizer.tts(text=text, language=language, speed=speed, **kwargs)
                        except Exception:
                            print("[WARNING] Retry failed, using fallback approach")
                    
                    # Fallback synthesis
                    return self._fallback_tts(text, language, **kwargs)
                else:
                    raise e
                    
            except Exception as e:
                error_str = str(e)
                if any(keyword in error_str.lower() for keyword in ['generate', 'gpt2', 'transformers', 'inference']):
                    print(f"[WARNING] XTTS compatibility issue: {e}")
                    return self._fallback_tts(text, language, **kwargs)
                else:
                    raise e

        def tts_to_file(self, text, speaker_wav=None, language="en", file_path="output.wav", **kwargs):
            """Enhanced TTS with comprehensive error handling and compatibility fixes."""
            if not self.available:
                raise RuntimeError("XTTS model not available")
            
            try:
                # Re-apply runtime fixes before synthesis
                if not self.gpt2_fix_applied:
                    self._apply_runtime_gpt2_fixes()
                
                # Try normal synthesis first
                return self.model.tts_to_file(
                    text=text,
                    speaker_wav=speaker_wav,
                    language=language,
                    file_path=file_path,
                    **kwargs
                )
                
            except AttributeError as e:
                if "'GPT2InferenceModel' object has no attribute 'generate'" in str(e):
                    print("[WARNING] GPT2InferenceModel compatibility issue detected")
                    print("[WARNING] Attempting advanced compatibility fix...")
                    
                    # Try to fix the specific instance that caused the error
                    success = self._emergency_gpt2_fix()
                    if success:
                        print("[SUCCESS] Emergency GPT2 fix applied, retrying synthesis...")
                        try:
                            return self.model.tts_to_file(
                                text=text,
                                speaker_wav=speaker_wav,
                                language=language,
                                file_path=file_path,
                                **kwargs
                            )
                        except Exception:
                            print("[WARNING] Retry failed, using fallback approach")
                    
                    # If all else fails, use fallback
                    return self._fallback_synthesis(text, speaker_wav, language, file_path, **kwargs)
                else:
                    raise e
                    
            except Exception as e:
                # Handle other potential compatibility issues
                error_str = str(e)
                if any(keyword in error_str.lower() for keyword in ['generate', 'gpt2', 'transformers', 'inference']):
                    print(f"[WARNING] XTTS compatibility issue: {e}")
                    return self._fallback_synthesis(text, speaker_wav, language, file_path, **kwargs)
                else:
                    raise e
        
        def _emergency_gpt2_fix(self):
            """Emergency GPT2InferenceModel fix during runtime."""
            try:
                # Try to find and fix any GPT2InferenceModel instances in the model
                if hasattr(self.model, 'synthesizer') and hasattr(self.model.synthesizer, 'tts_model'):
                    tts_model = self.model.synthesizer.tts_model
                    
                    # Check all attributes for GPT2InferenceModel instances
                    for attr_name in dir(tts_model):
                        if not attr_name.startswith('_'):
                            attr = getattr(tts_model, attr_name, None)
                            if attr and hasattr(attr, '__class__'):
                                if 'GPT2InferenceModel' in str(attr.__class__):
                                    print(f"Tools Fixing GPT2InferenceModel in {attr_name}")
                                    self._fix_gpt2_instance(attr)
                    
                    return True
                    
            except Exception as e:
                print(f"[WARNING] Emergency GPT2 fix failed: {e}")
                return False
        
        def _fallback_tts(self, text, language="en", **kwargs):
            """Fallback TTS that returns audio data."""
            fallback_models = [
                "tts_models/en/ljspeech/tacotron2-DDC",
                "tts_models/en/ljspeech/glow-tts",
                "tts_models/en/ljspeech/speedy-speech"
            ]
            
            for fallback_model in fallback_models:
                try:
                    from TTS.api import TTS
                    fallback_tts = TTS(fallback_model, gpu=False)
                    
                    print(f"[WARNING] Using fallback TTS model: {fallback_model} (no voice cloning)")
                    return fallback_tts.tts(text=text, language=language, **kwargs)
                    
                except Exception as e:
                    print(f"[WARNING] Fallback model {fallback_model} failed: {e}")
                    continue
            
            # If all fallbacks fail, raise the error
            raise RuntimeError("All XTTS fallback models failed")

        def _fallback_synthesis(self, text, speaker_wav, language, file_path, **kwargs):
            """Advanced fallback synthesis with multiple fallback options."""
            fallback_models = [
                "tts_models/en/ljspeech/tacotron2-DDC",
                "tts_models/en/ljspeech/glow-tts",
                "tts_models/en/ljspeech/speedy-speech"
            ]
            
            for fallback_model in fallback_models:
                try:
                    from TTS.api import TTS
                    fallback_tts = TTS(fallback_model, gpu=False)
                    
                    print(f"[WARNING] Using fallback TTS model: {fallback_model} (no voice cloning)")
                    return fallback_tts.tts_to_file(text=text, file_path=file_path)
                    
                except Exception as e:
                    print(f"[WARNING] Fallback model {fallback_model} failed: {e}")
                    continue
            
            # If all fallbacks fail, raise the error
            raise RuntimeError("All XTTS fallback models failed")
    
    return XTTSWrapper

if __name__ == "__main__":
    # Test the compatibility fixes
    apply_xtts_compatibility_fixes()
    
    # Test XTTS wrapper
    wrapper_class = create_xtts_wrapper()
    xtts = wrapper_class()
    
    if xtts.available:
        print("[SUCCESS] XTTS compatibility wrapper working")
    else:
        print("[ERROR] XTTS compatibility wrapper failed")