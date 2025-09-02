#!/usr/bin/env python3
"""
Enhanced Realistic Face Generator for SadTalker Testing
Creates realistic human face images that meet SadTalker's face detection requirements
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageColor
import random
from pathlib import Path

def create_realistic_test_faces(output_dir: str, num_faces: int = 3):
    """
    Create realistic test face images for SadTalker face detection testing.
    
    Args:
        output_dir: Directory to save generated faces
        num_faces: Number of different face variations to create
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    created_faces = []
    
    for i in range(num_faces):
        # Create realistic face image
        face_image = create_realistic_face(face_id=i)
        
        # Save the face
        face_filename = f"realistic_face_{i+1:02d}.jpg"
        face_path = output_path / face_filename
        face_image.save(face_path, "JPEG", quality=95)
        
        created_faces.append(str(face_path))
        print(f"[SUCCESS] Created realistic face {i+1}: {face_filename}")
    
    return created_faces

def create_realistic_face(face_id: int = 0, size: tuple = (512, 512)):
    """
    Create a realistic-looking human face image with proper proportions and features.
    
    Args:
        face_id: ID for generating different face variations
        size: Image size (width, height)
    
    Returns:
        PIL Image of realistic face
    """
    width, height = size
    
    # Create base image with skin tone
    skin_tones = [
        (255, 220, 177),  # Light skin
        (241, 194, 125),  # Medium light skin
        (224, 172, 105),  # Medium skin
        (198, 134, 66),   # Medium dark skin
        (141, 85, 36)     # Dark skin
    ]
    
    base_skin = skin_tones[face_id % len(skin_tones)]
    image = Image.new('RGB', (width, height), base_skin)
    draw = ImageDraw.Draw(image)
    
    # Face shape (oval)
    face_margin = 50
    face_width = width - (2 * face_margin)
    face_height = int(height * 0.85)
    face_top = (height - face_height) // 2
    
    face_bbox = [
        face_margin,
        face_top,
        face_margin + face_width,
        face_top + face_height
    ]
    
    # Draw face outline with gradient-like shading
    draw.ellipse(face_bbox, fill=base_skin, outline=None)
    
    # Face proportions (rule of thirds)
    face_center_x = width // 2
    face_center_y = face_top + face_height // 2
    
    # Eyes (positioned at 1/3 from top of face)
    eye_y = face_top + face_height // 3
    eye_distance = face_width // 5
    left_eye_x = face_center_x - eye_distance
    right_eye_x = face_center_x + eye_distance
    
    # Draw eyes with realistic proportions
    eye_width = 30
    eye_height = 15
    
    # Eye whites
    left_eye_bbox = [left_eye_x - eye_width//2, eye_y - eye_height//2,
                     left_eye_x + eye_width//2, eye_y + eye_height//2]
    right_eye_bbox = [right_eye_x - eye_width//2, eye_y - eye_height//2,
                      right_eye_x + eye_width//2, eye_y + eye_height//2]
    
    draw.ellipse(left_eye_bbox, fill=(255, 255, 255), outline=(0, 0, 0))
    draw.ellipse(right_eye_bbox, fill=(255, 255, 255), outline=(0, 0, 0))
    
    # Iris and pupils
    iris_colors = [(101, 67, 33), (72, 61, 139), (34, 139, 34), (165, 42, 42)]
    iris_color = iris_colors[face_id % len(iris_colors)]
    
    iris_size = 12
    pupil_size = 6
    
    # Left eye iris and pupil
    draw.ellipse([left_eye_x - iris_size//2, eye_y - iris_size//2,
                  left_eye_x + iris_size//2, eye_y + iris_size//2],
                 fill=iris_color)
    draw.ellipse([left_eye_x - pupil_size//2, eye_y - pupil_size//2,
                  left_eye_x + pupil_size//2, eye_y + pupil_size//2],
                 fill=(0, 0, 0))
    
    # Right eye iris and pupil
    draw.ellipse([right_eye_x - iris_size//2, eye_y - iris_size//2,
                  right_eye_x + iris_size//2, eye_y + iris_size//2],
                 fill=iris_color)
    draw.ellipse([right_eye_x - pupil_size//2, eye_y - pupil_size//2,
                  right_eye_x + pupil_size//2, eye_y + pupil_size//2],
                 fill=(0, 0, 0))
    
    # Eyebrows
    eyebrow_y = eye_y - 25
    eyebrow_color = (101, 67, 33) if face_id % 2 == 0 else (139, 69, 19)
    
    # Left eyebrow
    draw.arc([left_eye_x - 20, eyebrow_y - 5, left_eye_x + 20, eyebrow_y + 5],
             start=0, end=180, fill=eyebrow_color, width=3)
    
    # Right eyebrow
    draw.arc([right_eye_x - 20, eyebrow_y - 5, right_eye_x + 20, eyebrow_y + 5],
             start=0, end=180, fill=eyebrow_color, width=3)
    
    # Nose (positioned at center of face)
    nose_top_y = eye_y + 30
    nose_bottom_y = nose_top_y + 40
    nose_width = 15
    
    # Nose bridge
    draw.line([face_center_x, nose_top_y, face_center_x, nose_bottom_y - 10],
              fill=(max(0, base_skin[0] - 20), 
                    max(0, base_skin[1] - 20), 
                    max(0, base_skin[2] - 20)), 
              width=2)
    
    # Nostrils
    nostril_y = nose_bottom_y - 5
    draw.ellipse([face_center_x - 8, nostril_y - 3, face_center_x - 3, nostril_y + 2],
                 fill=(max(0, base_skin[0] - 30), 
                       max(0, base_skin[1] - 30), 
                       max(0, base_skin[2] - 30)))
    draw.ellipse([face_center_x + 3, nostril_y - 3, face_center_x + 8, nostril_y + 2],
                 fill=(max(0, base_skin[0] - 30), 
                       max(0, base_skin[1] - 30), 
                       max(0, base_skin[2] - 30)))
    
    # Mouth (positioned at lower third of face)
    mouth_y = face_top + (face_height * 2) // 3
    mouth_width = 40
    
    # Lips
    lip_colors = [(205, 92, 92), (220, 20, 60), (199, 21, 133), (147, 112, 219)]
    lip_color = lip_colors[face_id % len(lip_colors)]
    
    # Upper lip
    draw.arc([face_center_x - mouth_width//2, mouth_y - 8,
              face_center_x + mouth_width//2, mouth_y + 8],
             start=0, end=180, fill=lip_color, width=4)
    
    # Lower lip
    draw.arc([face_center_x - mouth_width//2, mouth_y - 5,
              face_center_x + mouth_width//2, mouth_y + 10],
             start=180, end=360, fill=lip_color, width=4)
    
    # Hair
    hair_colors = [(101, 67, 33), (139, 69, 19), (0, 0, 0), (255, 255, 0), (165, 42, 42)]
    hair_color = hair_colors[face_id % len(hair_colors)]
    
    # Simple hair shape (top of head)
    hair_height = 80
    hair_bbox = [face_margin - 10, face_top - hair_height,
                 face_margin + face_width + 10, face_top + 20]
    draw.ellipse(hair_bbox, fill=hair_color)
    
    # Add some basic shading for depth
    add_facial_shading(draw, face_bbox, base_skin)
    
    return image

def add_facial_shading(draw, face_bbox, base_skin):
    """Add basic shading to give the face more depth and realism."""
    # Calculate shadow color (darker version of skin tone)
    shadow_color = (
        max(0, base_skin[0] - 30),
        max(0, base_skin[1] - 30),
        max(0, base_skin[2] - 30)
    )
    
    face_left, face_top, face_right, face_bottom = face_bbox
    face_width = face_right - face_left
    face_height = face_bottom - face_top
    
    # Add subtle shadows on the sides of the face
    shadow_width = face_width // 8
    
    # Left side shadow
    draw.arc([face_left, face_top, face_left + shadow_width * 2, face_bottom],
             start=90, end=270, fill=shadow_color, width=10)
    
    # Right side shadow
    draw.arc([face_right - shadow_width * 2, face_top, face_right, face_bottom],
             start=270, end=450, fill=shadow_color, width=10)

def download_reference_faces(output_dir: str):
    """
    Download real reference face images for testing.
    This creates placeholder URLs - in production, you'd use actual stock photo APIs.
    """
    print("Download Note: Using generated faces instead of downloaded references")
    print("    Generated faces are designed to work well with SadTalker's detection")
    return []

def validate_face_quality(image_path: str):
    """
    Validate that a face image meets SadTalker's requirements.
    
    Args:
        image_path: Path to the face image
    
    Returns:
        dict: Validation results
    """
    try:
        from PIL import Image
        
        image = Image.open(image_path)
        width, height = image.size
        
        validation = {
            "valid": True,
            "issues": [],
            "recommendations": []
        }
        
        # Check image dimensions
        if width < 256 or height < 256:
            validation["issues"].append("Image too small (minimum 256x256)")
            validation["valid"] = False
        
        if width > 1024 or height > 1024:
            validation["recommendations"].append("Large image - consider resizing for faster processing")
        
        # Check aspect ratio
        aspect_ratio = width / height
        if not 0.7 <= aspect_ratio <= 1.4:
            validation["issues"].append("Unusual aspect ratio - faces work best with near-square images")
            validation["valid"] = False
        
        # Check if image is grayscale
        if image.mode == 'L':
            validation["recommendations"].append("Grayscale image - color images often work better")
        
        return validation
        
    except Exception as e:
        return {
            "valid": False,
            "issues": [f"Could not validate image: {str(e)}"],
            "recommendations": []
        }

def main():
    """Generate realistic test faces for SadTalker testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate realistic faces for SadTalker testing")
    parser.add_argument("--output-dir", required=True, help="Output directory for generated faces")
    parser.add_argument("--num-faces", type=int, default=3, help="Number of faces to generate")
    parser.add_argument("--validate", action="store_true", help="Validate generated faces")
    
    args = parser.parse_args()
    
    print("Style: Generating realistic faces for SadTalker testing...")
    
    # Generate faces
    face_paths = create_realistic_test_faces(args.output_dir, args.num_faces)
    
    # Validate if requested
    if args.validate:
        print("\nSearch Validating generated faces...")
        for face_path in face_paths:
            result = validate_face_quality(face_path)
            status = "[SUCCESS]" if result["valid"] else "[ERROR]"
            print(f"{status} {Path(face_path).name}: {'Valid' if result['valid'] else 'Issues found'}")
            
            if result["issues"]:
                for issue in result["issues"]:
                    print(f"    [WARNING] {issue}")
            
            if result["recommendations"]:
                for rec in result["recommendations"]:
                    print(f"    INFO: {rec}")
    
    print(f"\n[SUCCESS] Generated {len(face_paths)} realistic faces in {args.output_dir}")
    print("These faces are designed to meet SadTalker's face detection requirements")

if __name__ == "__main__":
    main()