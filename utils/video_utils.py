"""
Video processing utilities for the pipeline.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
import subprocess
import json
import sys

def get_video_info(video_path: str) -> Dict[str, Any]:
    """
    Get comprehensive video information.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary containing video information
        
    Raises:
        ValueError: If video cannot be opened
    """
    if not Path(video_path).exists():
        raise ValueError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get basic video properties
    info = {
        'path': video_path,
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': None,
        'codec': None
    }
    
    # Calculate duration
    if info['fps'] > 0:
        info['duration'] = info['frame_count'] / info['fps']
    
    # Try to get codec information
    fourcc = cap.get(cv2.CAP_PROP_FOURCC)
    if fourcc:
        codec_bytes = int(fourcc).to_bytes(4, byteorder='little')
        try:
            info['codec'] = codec_bytes.decode('ascii').rstrip('\x00')
        except:
            info['codec'] = 'unknown'
    
    cap.release()
    
    return info


def extract_frames(video_path: str, output_dir: str,
                  start_frame: int = 0, end_frame: Optional[int] = None,
                  step: int = 1, image_format: str = 'jpg') -> List[str]:
    """
    Extract frames from video and save as images.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        start_frame: Starting frame number
        end_frame: Ending frame number (None for all frames)
        step: Frame step size (1 = every frame, 2 = every other frame, etc.)
        image_format: Output image format ('jpg', 'png')
        
    Returns:
        List of extracted frame file paths
        
    Raises:
        ValueError: If video cannot be opened
    """
    from .file_utils import ensure_directory
    
    if not Path(video_path).exists():
        raise ValueError(f"Video file not found: {video_path}")
    
    output_path = ensure_directory(output_dir)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frame_paths = []
    frame_num = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_num >= start_frame:
            if end_frame is not None and frame_num > end_frame:
                break
            
            if (frame_num - start_frame) % step == 0:
                frame_filename = f"frame_{frame_num:06d}.{image_format}"
                frame_path = output_path / frame_filename
                
                # Save frame with appropriate quality settings
                if image_format.lower() == 'jpg':
                    cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                elif image_format.lower() == 'png':
                    cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                else:
                    cv2.imwrite(str(frame_path), frame)
                
                frame_paths.append(str(frame_path))
                extracted_count += 1
        
        frame_num += 1
    
    cap.release()
    return frame_paths


def create_video_from_frames(frame_paths: List[str], output_path: str,
                           fps: float = 30.0, codec: str = 'mp4v',
                           quality: Optional[int] = None) -> bool:
    """
    Create video from a sequence of frame images.
    
    Args:
        frame_paths: List of frame image paths (must be in order)
        output_path: Output video path
        fps: Frames per second
        codec: Video codec ('mp4v', 'XVID', 'H264')
        quality: Video quality (0-100, higher is better)
        
    Returns:
        True if successful, False otherwise
    """
    if not frame_paths:
        return False
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        return False
    
    height, width, channels = first_frame.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        return False
    
    try:
        for i, frame_path in enumerate(frame_paths):
            frame = cv2.imread(frame_path)
            if frame is not None:
                # Ensure frame has correct dimensions
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                out.write(frame)
            else:
                print(f"Warning: Could not read frame {i}: {frame_path}")
        
        out.release()
        return True
    except Exception as e:
        print(f"Error creating video: {e}")
        out.release()
        return False


def resize_video(input_path: str, output_path: str,
                width: int, height: int, maintain_aspect: bool = True) -> bool:
    """
    Resize video to new dimensions.
    
    Args:
        input_path: Input video path
        output_path: Output video path
        width: Target width
        height: Target height
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        True if successful, False otherwise
    """
    if not Path(input_path).exists():
        return False
    
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        return False
    
    # Get original dimensions
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate new dimensions maintaining aspect ratio
    if maintain_aspect:
        aspect = orig_width / orig_height
        if width / height > aspect:
            width = int(height * aspect)
        else:
            height = int(width / aspect)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        cap.release()
        return False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            resized_frame = cv2.resize(frame, (width, height))
            out.write(resized_frame)
        
        cap.release()
        out.release()
        return True
    except Exception:
        cap.release()
        out.release()
        return False


def get_video_thumbnail(video_path: str, output_path: str,
                       frame_number: Optional[int] = None) -> bool:
    """
    Extract a thumbnail image from video.
    
    Args:
        video_path: Path to video file
        output_path: Path to save thumbnail
        frame_number: Specific frame to extract (None for middle frame)
        
    Returns:
        True if successful, False otherwise
    """
    if not Path(video_path).exists():
        return False
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return False
    
    try:
        # If no frame specified, use middle frame
        if frame_number is None:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_number = total_frames // 2
        
        # Seek to desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if ret:
            cv2.imwrite(output_path, frame)
            cap.release()
            return True
        else:
            cap.release()
            return False
    except Exception:
        cap.release()
        return False


def test_video_utils():
    """Test function for video utilities."""
    print("Testing video utilities...")
    
    # This test requires an actual video file to work properly
    # For now, just test the structure
    print("Video utilities loaded successfully!")
    print("Available functions:")
    print("- get_video_info()")
    print("- extract_frames()")
    print("- create_video_from_frames()")
    print("- resize_video()")
    print("- get_video_thumbnail()")


if __name__ == "__main__":
    test_video_utils()
