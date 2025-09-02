#!/usr/bin/env python3
"""
Create realistic test face images for SadTalker testing
Uses dlib face landmarks to create more detectable synthetic faces
"""

import cv2
import numpy as np
from pathlib import Path
import math

def create_realistic_test_face(output_path: str, size: int = 512) -> bool:
    """Create a realistic test face that face detectors can recognize."""
    
    try:
        # Create image with proper background
        image = np.ones((size, size, 3), dtype=np.uint8) * 250  # Light background
        
        # Face parameters
        center_x, center_y = size // 2, size // 2
        face_width = int(size * 0.35)
        face_height = int(size * 0.45)
        
        # Create face mask for proper face detection
        face_mask = np.zeros((size, size), dtype=np.uint8)
        cv2.ellipse(face_mask, (center_x, center_y), (face_width, face_height), 0, 0, 360, 255, -1)
        
        # Base face color (skin tone)
        skin_color = (220, 185, 165)
        
        # Apply face color
        face_region = cv2.bitwise_and(image, image, mask=face_mask)
        face_colored = np.zeros_like(image)
        face_colored[face_mask > 0] = skin_color
        
        # Blend with background
        image = cv2.addWeighted(image, 0.3, face_colored, 0.7, 0)
        
        # Add facial features with proper proportions for face detection
        
        # Eyes (critical for face detection)
        eye_y = center_y - int(face_height * 0.15)
        eye_spacing = int(face_width * 0.3)
        eye_width = int(face_width * 0.12)
        eye_height = int(face_height * 0.08)
        
        # Left eye
        left_eye_x = center_x - eye_spacing
        cv2.ellipse(image, (left_eye_x, eye_y), (eye_width, eye_height), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (left_eye_x, eye_y), (eye_width-3, eye_height-2), 0, 0, 360, (200, 180, 160), 2)
        cv2.circle(image, (left_eye_x, eye_y), int(eye_width*0.6), (120, 80, 60), -1)  # Iris
        cv2.circle(image, (left_eye_x, eye_y), int(eye_width*0.3), (20, 20, 20), -1)   # Pupil
        cv2.circle(image, (left_eye_x-2, eye_y-2), int(eye_width*0.15), (255, 255, 255), -1)  # Highlight
        
        # Right eye
        right_eye_x = center_x + eye_spacing
        cv2.ellipse(image, (right_eye_x, eye_y), (eye_width, eye_height), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (right_eye_x, eye_y), (eye_width-3, eye_height-2), 0, 0, 360, (200, 180, 160), 2)
        cv2.circle(image, (right_eye_x, eye_y), int(eye_width*0.6), (120, 80, 60), -1)  # Iris
        cv2.circle(image, (right_eye_x, eye_y), int(eye_width*0.3), (20, 20, 20), -1)   # Pupil
        cv2.circle(image, (right_eye_x-2, eye_y-2), int(eye_width*0.15), (255, 255, 255), -1)  # Highlight
        
        # Eyebrows (important for face structure)
        brow_y = eye_y - int(face_height * 0.08)
        brow_width = int(eye_width * 1.3)
        brow_height = int(face_height * 0.04)
        
        cv2.ellipse(image, (left_eye_x, brow_y), (brow_width, brow_height), 0, 0, 180, (100, 70, 50), -1)
        cv2.ellipse(image, (right_eye_x, brow_y), (brow_width, brow_height), 0, 0, 180, (100, 70, 50), -1)
        
        # Nose (critical facial landmark)
        nose_width = int(face_width * 0.08)
        nose_height = int(face_height * 0.15)
        nose_y = center_y
        
        # Nose bridge
        cv2.ellipse(image, (center_x, nose_y), (nose_width, nose_height), 0, 0, 360, (200, 165, 145), -1)
        
        # Nostrils
        nostril_y = nose_y + int(nose_height * 0.6)
        nostril_spacing = int(nose_width * 0.8)
        cv2.ellipse(image, (center_x - nostril_spacing//2, nostril_y), (3, 5), 0, 0, 360, (150, 120, 100), -1)
        cv2.ellipse(image, (center_x + nostril_spacing//2, nostril_y), (3, 5), 0, 0, 360, (150, 120, 100), -1)
        
        # Mouth (important for face detection)
        mouth_y = center_y + int(face_height * 0.25)
        mouth_width = int(face_width * 0.15)
        mouth_height = int(face_height * 0.04)
        
        # Mouth shape
        cv2.ellipse(image, (center_x, mouth_y), (mouth_width, mouth_height), 0, 0, 360, (160, 110, 110), -1)
        cv2.ellipse(image, (center_x, mouth_y-2), (mouth_width-2, mouth_height-1), 0, 0, 360, (140, 90, 90), -1)
        
        # Add subtle face shading for depth
        # Left side shading
        shade_mask = np.zeros((size, size), dtype=np.uint8)
        cv2.ellipse(shade_mask, (center_x - face_width//4, center_y), (face_width//2, face_height), 0, 0, 360, 255, -1)
        shade_overlay = image.copy()
        shade_overlay[shade_mask > 0] = [max(0, c-15) for c in skin_color]
        image = cv2.addWeighted(image, 0.8, shade_overlay, 0.2, 0)
        
        # Hair (helps with face boundary detection)
        hair_mask = np.zeros((size, size), dtype=np.uint8)
        cv2.ellipse(hair_mask, (center_x, center_y - int(face_height * 0.7)), 
                   (int(face_width * 1.1), int(face_height * 0.6)), 0, 0, 180, 255, -1)
        
        # Remove face area from hair
        cv2.ellipse(hair_mask, (center_x, center_y), (face_width, face_height), 0, 0, 360, 0, -1)
        
        # Apply hair color
        hair_color = (80, 60, 40)
        image[hair_mask > 0] = hair_color
        
        # Add some texture to make it more realistic
        noise = np.random.randint(-10, 10, image.shape, dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Apply slight blur for more natural look
        image = cv2.GaussianBlur(image, (3, 3), 0.5)
        
        # Save image
        success = cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        if success:
            print(f"[SUCCESS] Created realistic test face: {output_path}")
            return True
        else:
            print(f"[ERROR] Failed to save test face: {output_path}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Error creating test face: {e}")
        return False

def verify_face_detectability(image_path: str) -> bool:
    """Verify that the created face can be detected by OpenCV face detection."""
    
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return False
        
        # Try OpenCV face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        print(f"OpenCV detected {len(faces)} faces")
        
        # Also try profile face detection
        profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        profile_faces = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        print(f"OpenCV detected {len(profile_faces)} profile faces")
        
        return len(faces) > 0 or len(profile_faces) > 0
        
    except Exception as e:
        print(f"Face detection verification failed: {e}")
        return False

if __name__ == "__main__":
    # Test the face creation
    test_dir = Path("test_outputs/realistic_faces")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    test_image_path = str(test_dir / "realistic_test_face.jpg")
    
    if create_realistic_test_face(test_image_path):
        if verify_face_detectability(test_image_path):
            print("[SUCCESS] Realistic test face created and verified!")
        else:
            print("[WARNING] Face created but may not be detectable")
    else:
        print("[ERROR] Failed to create test face")