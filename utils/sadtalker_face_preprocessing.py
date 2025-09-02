#!/usr/bin/env python3
"""
SadTalker Face Detection Preprocessing
Improves face detection success rate by preprocessing images and adjusting detection parameters
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from pathlib import Path
import logging

def preprocess_face_image(image_path: str, output_path: str = None, enhance_contrast: bool = True, 
                         resize_to: tuple = None, enhance_brightness: float = 1.0) -> str:
    """
    Preprocess face image to improve SadTalker face detection success rate.
    
    Args:
        image_path: Path to input image
        output_path: Path for output image (if None, overwrites input)
        enhance_contrast: Whether to enhance image contrast
        resize_to: Target size (width, height) for resizing
        enhance_brightness: Brightness enhancement factor (1.0 = no change)
    
    Returns:
        Path to processed image
    """
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Resize if specified
        if resize_to:
            image = image.resize(resize_to, Image.Resampling.LANCZOS)
        
        # Enhance brightness if needed
        if enhance_brightness != 1.0:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(enhance_brightness)
        
        # Enhance contrast for better face detection
        if enhance_contrast:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.3)  # Moderate contrast enhancement
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.2)  # Slight sharpness enhancement
        
        # Apply histogram equalization for better feature visibility
        image_array = np.array(image)
        image_array = apply_adaptive_histogram_equalization(image_array)
        image = Image.fromarray(image_array)
        
        # Save processed image
        if output_path is None:
            output_path = image_path
        
        image.save(output_path, "JPEG", quality=95)
        
        logging.info(f"[SUCCESS] Preprocessed face image: {Path(image_path).name} -> {Path(output_path).name}")
        return output_path
        
    except Exception as e:
        logging.error(f"[ERROR] Face preprocessing failed: {e}")
        return image_path  # Return original path if preprocessing fails

def apply_adaptive_histogram_equalization(image_array: np.ndarray) -> np.ndarray:
    """Apply adaptive histogram equalization to improve image contrast."""
    try:
        # Convert to YUV color space
        yuv = cv2.cvtColor(image_array, cv2.COLOR_RGB2YUV)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to Y channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        
        return enhanced
        
    except Exception:
        # If enhancement fails, return original
        return image_array

def create_multiple_detection_variants(image_path: str, output_dir: str) -> list:
    """
    Create multiple variants of an image with different preprocessing settings
    to increase the chances of successful face detection.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save variants
        
    Returns:
        List of paths to created variants
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    base_name = Path(image_path).stem
    variants = []
    
    # Variant 1: Original with mild enhancement
    variant_1 = output_path / f"{base_name}_enhanced.jpg"
    preprocess_face_image(image_path, str(variant_1), enhance_contrast=True, enhance_brightness=1.1)
    variants.append(str(variant_1))
    
    # Variant 2: Higher contrast
    variant_2 = output_path / f"{base_name}_high_contrast.jpg"
    preprocess_face_image(image_path, str(variant_2), enhance_contrast=True, enhance_brightness=1.2)
    variants.append(str(variant_2))
    
    # Variant 3: Resized to optimal size for detection
    variant_3 = output_path / f"{base_name}_resized.jpg"
    preprocess_face_image(image_path, str(variant_3), resize_to=(512, 512), enhance_contrast=True)
    variants.append(str(variant_3))
    
    # Variant 4: Darker version (sometimes helps with overexposed faces)
    variant_4 = output_path / f"{base_name}_darker.jpg"
    preprocess_face_image(image_path, str(variant_4), enhance_brightness=0.8, enhance_contrast=True)
    variants.append(str(variant_4))
    
    logging.info(f"[SUCCESS] Created {len(variants)} detection variants for {Path(image_path).name}")
    return variants

def patch_sadtalker_detection_confidence(sadtalker_path: str, new_confidence: float = 0.7):
    """
    Patch SadTalker's face detection confidence threshold to be more permissive.
    
    Args:
        sadtalker_path: Path to SadTalker directory
        new_confidence: New confidence threshold (lower = more permissive)
    """
    try:
        # Find the face detection file
        detection_file = Path(sadtalker_path) / "src" / "face3d" / "extract_kp_videos_safe.py"
        
        if not detection_file.exists():
            logging.warning(f"[WARNING] SadTalker detection file not found: {detection_file}")
            return False
        
        # Read the file
        with open(detection_file, 'r') as f:
            content = f.read()
        
        # Replace the confidence threshold
        # Look for the pattern: detect_faces(images, 0.97)
        import re
        pattern = r'detect_faces\(images,\s*([0-9.]+)\)'
        match = re.search(pattern, content)
        
        if match:
            old_confidence = match.group(1)
            new_content = re.sub(pattern, f'detect_faces(images, {new_confidence})', content)
            
            # Create backup
            backup_file = detection_file.with_suffix('.py.backup')
            if not backup_file.exists():
                with open(backup_file, 'w') as f:
                    f.write(content)
            
            # Write updated content
            with open(detection_file, 'w') as f:
                f.write(new_content)
            
            logging.info(f"[SUCCESS] Updated SadTalker detection confidence: {old_confidence} -> {new_confidence}")
            return True
        else:
            logging.warning("[WARNING] Could not find face detection confidence pattern in SadTalker code")
            return False
            
    except Exception as e:
        logging.error(f"[ERROR] Failed to patch SadTalker detection confidence: {e}")
        return False

def main():
    """Test face preprocessing functions."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess face images for better SadTalker detection")
    parser.add_argument("--image", help="Input image path")
    parser.add_argument("--output", help="Output image path")
    parser.add_argument("--variants", help="Create detection variants in this directory")
    parser.add_argument("--patch-confidence", type=float, help="Patch SadTalker confidence threshold")
    parser.add_argument("--sadtalker-path", help="Path to SadTalker directory for patching")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.patch_confidence and args.sadtalker_path:
        patch_sadtalker_detection_confidence(args.sadtalker_path, args.patch_confidence)
    
    if args.variants:
        create_multiple_detection_variants(args.image, args.variants)
    else:
        preprocess_face_image(args.image, args.output)

if __name__ == "__main__":
    main()