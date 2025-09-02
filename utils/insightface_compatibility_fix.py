#!/usr/bin/env python3
"""
InsightFace Compatibility Fix
Fixes InsightFace model loading and buffalo_l initialization issues
"""

import os
import warnings
import numpy as np

def apply_insightface_compatibility_fixes():
    """Apply compatibility fixes for InsightFace."""
    
    # Suppress InsightFace warnings
    warnings.filterwarnings("ignore", message=".*insightface.*")
    warnings.filterwarnings("ignore", message=".*onnx.*")
    warnings.filterwarnings("ignore", message=".*buffalo.*")
    
    # Set environment variables for better compatibility
    os.environ['INSIGHTFACE_DISABLE_WARNINGS'] = '1'
    
    print("[SUCCESS] InsightFace compatibility fixes applied")

def create_insightface_wrapper():
    """Create a wrapper for InsightFace that handles compatibility issues."""
    
    class InsightFaceWrapper:
        def __init__(self, model_name='buffalo_l', providers=None, device='cpu', det_size=(640, 640)):
            apply_insightface_compatibility_fixes()
            
            self.available = False
            self.app = None
            
            try:
                import insightface
                
                # Set up providers with fallback
                if providers is None:
                    if device == 'cuda':
                        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    else:
                        providers = ['CPUExecutionProvider']
                
                # Initialize with error handling
                try:
                    # Try with specified model
                    self.app = insightface.app.FaceAnalysis(
                        name=model_name,
                        providers=providers
                    )
                    self.app.prepare(ctx_id=0, det_size=det_size)
                    
                    # Fix RetinaFace providers attribute issue
                    self._fix_retinaface_providers()
                    
                    self.available = True
                    print(f"[SUCCESS] InsightFace loaded with {model_name} model")
                    
                except Exception as e:
                    print(f"[WARNING] {model_name} model failed: {e}")
                    
                    # Try fallback with default model
                    try:
                        self.app = insightface.app.FaceAnalysis(providers=providers)
                        self.app.prepare(ctx_id=0, det_size=det_size)
                        
                        # Fix RetinaFace providers attribute issue
                        self._fix_retinaface_providers()
                        
                        self.available = True
                        print("[SUCCESS] InsightFace loaded with default model")
                        
                    except Exception as e2:
                        print(f"[ERROR] InsightFace default model also failed: {e2}")
                        self.available = False
                        
            except ImportError:
                print("[ERROR] InsightFace not installed")
                self.available = False
            except Exception as e:
                print(f"[ERROR] InsightFace initialization failed: {e}")
                self.available = False
        
        def _fix_retinaface_providers(self):
            """Fix RetinaFace providers attribute issue."""
            try:
                # Check if app has models and fix providers attribute
                if hasattr(self.app, 'models') and self.app.models:
                    for model_name, model in self.app.models.items():
                        if hasattr(model, 'session') and not hasattr(model, 'providers'):
                            # Add providers attribute if missing
                            if hasattr(model.session, 'get_providers'):
                                model.providers = model.session.get_providers()
                            else:
                                model.providers = ['CPUExecutionProvider']
                            
                        # Also fix for detection models specifically
                        if 'det' in model_name.lower() and hasattr(model, 'session'):
                            if not hasattr(model, 'providers'):
                                model.providers = getattr(model.session, 'get_providers', lambda: ['CPUExecutionProvider'])()
                
            except Exception as e:
                print(f"[WARNING] RetinaFace providers fix warning: {e}")
                # Continue anyway, this is just a compatibility fix
        
        def get(self, image, max_num=0):
            """Get faces from image with error handling."""
            if not self.available:
                return []
            
            try:
                faces = self.app.get(image, max_num=max_num)
                return faces
            except Exception as e:
                print(f"[WARNING] Face detection failed: {e}")
                return []
        
        def detect_faces(self, image, max_faces=5):
            """Detect faces with simplified output format."""
            faces = self.get(image, max_num=max_faces)
            
            results = []
            for face in faces:
                face_info = {
                    'bbox': face.bbox.tolist() if hasattr(face, 'bbox') else [],
                    'confidence': float(face.det_score) if hasattr(face, 'det_score') else 0.0,
                    'landmarks': face.kps.tolist() if hasattr(face, 'kps') else [],
                    'age': getattr(face, 'age', None),
                    'gender': getattr(face, 'gender', None)
                }
                results.append(face_info)
            
            return results
    
    return InsightFaceWrapper

if __name__ == "__main__":
    # Test the compatibility fixes
    apply_insightface_compatibility_fixes()
    
    # Test InsightFace wrapper
    wrapper_class = create_insightface_wrapper()
    detector = wrapper_class()
    
    if detector.available:
        print("[SUCCESS] InsightFace compatibility wrapper working")
        
        # Test with dummy image
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        faces = detector.detect_faces(test_image)
        print(f"Test detection: {len(faces)} faces found")
    else:
        print("[ERROR] InsightFace compatibility wrapper failed")