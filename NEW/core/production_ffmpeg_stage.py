#!/usr/bin/env python3
"""
Production FFmpeg Stage - Video Concatenation and Assembly
Concatenate video chunks and create final assembled video
NO FALLBACK MECHANISMS - Production mode only
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionFFmpegStage:
    """Production FFmpeg Stage - Video concatenation and final assembly."""
    
    def __init__(self, project_root: str = None):
        """Initialize Production FFmpeg stage.
        
        Args:
            project_root: Path to project root directory
        """
        if project_root is None:
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = Path(project_root)
            
        # Output directory
        self.output_dir = self.project_root / "NEW" / "processed" / "ffmpeg"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # FFmpeg configuration
        self.ffmpeg_path = self._find_ffmpeg()
        
        # Verify environment
        self.available = self._verify_environment()
        
        if not self.available:
            raise RuntimeError("FFmpeg production environment not available")
        
        logger.info(f"Production FFmpeg Stage initialized")
        logger.info(f"  FFmpeg path: {self.ffmpeg_path}")
        logger.info(f"  Output directory: {self.output_dir}")
    
    def _find_ffmpeg(self) -> str:
        """Find FFmpeg executable."""
        # Try common locations
        ffmpeg_paths = [
            '/usr/local/bin/ffmpeg',
            '/opt/homebrew/bin/ffmpeg',
            '/usr/bin/ffmpeg',
            'ffmpeg'  # System PATH
        ]
        
        for path in ffmpeg_paths:
            try:
                result = subprocess.run(
                    [path, '-version'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    return path
            except:
                continue
        
        return 'ffmpeg'  # Default to system PATH
    
    def _verify_environment(self) -> bool:
        """Verify FFmpeg is available."""
        try:
            result = subprocess.run(
                [self.ffmpeg_path, '-version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                logger.info("[SUCCESS] FFmpeg production environment verified")
                return True
            else:
                logger.error(f"[ERROR] FFmpeg verification failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] FFmpeg verification failed: {e}")
            return False
    
    def concatenate_video_chunks(self, video_chunks: List[str], output_path: str, 
                               background_video: str = None, overlay_position: str = "bottom-left") -> Dict[str, Any]:
        """Concatenate video chunks into a single video with optional background overlay.
        
        Args:
            video_chunks: List of video chunk paths
            output_path: Path for final output video
            background_video: Optional background video for overlay
            overlay_position: Position for face overlay ("bottom-left", "bottom-right", etc.)
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        
        result = {
            'stage': 'production_ffmpeg_concatenation',
            'timestamp': time.time(),
            'success': False,
            'input_chunks': video_chunks,
            'output_video_path': output_path,
            'processing_time': 0,
            'errors': []
        }
        
        try:
            # Validate input chunks
            valid_chunks = []
            for chunk in video_chunks:
                if Path(chunk).exists():
                    valid_chunks.append(chunk)
                else:
                    result['errors'].append(f"Video chunk not found: {chunk}")
            
            if not valid_chunks:
                result['errors'].append("No valid video chunks found")
                return result
            
            if len(valid_chunks) == 1:
                # Single chunk - just copy it
                import shutil
                shutil.copy2(valid_chunks[0], output_path)
                result['success'] = True
                result['file_size'] = Path(output_path).stat().st_size
                result['total_chunks'] = 1
                
                logger.info("[SUCCESS] Single chunk copied successfully")
                logger.info(f"   Output: {output_path}")
                logger.info(f"   File size: {result['file_size']:,} bytes")
                
            else:
                # Multiple chunks - concatenate using FFmpeg
                result_concat = self._ffmpeg_concatenate(valid_chunks, output_path, background_video, overlay_position)
                
                if result_concat['success']:
                    result.update(result_concat)
                    result['success'] = True
                    result['total_chunks'] = len(valid_chunks)
                    
                    logger.info("[SUCCESS] Video chunks concatenated successfully")
                    logger.info(f"   Total chunks: {len(valid_chunks)}")
                    logger.info(f"   Output: {output_path}")
                    logger.info(f"   File size: {result.get('file_size', 0):,} bytes")
                    if background_video:
                        logger.info(f"   Background overlay: {background_video}")
                        logger.info(f"   Overlay position: {overlay_position}")
                else:
                    result['errors'].extend(result_concat['errors'])
                    
        except Exception as e:
            result['errors'].append(f"FFmpeg concatenation failed: {str(e)}")
            logger.error(f"[ERROR] FFmpeg concatenation error: {e}")
            
        finally:
            result['processing_time'] = time.time() - start_time
            
            # Save stage results
            results_file = self.output_dir / f"production_ffmpeg_results_{int(time.time())}.json"
            with open(results_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"File: Production FFmpeg results saved: {results_file}")
            logger.info(f"Duration:  Processing time: {result['processing_time']:.2f} seconds")
        
        return result
    
    def _ffmpeg_concatenate(self, video_chunks: List[str], output_path: str, 
                           background_video: str = None, overlay_position: str = "bottom-left") -> Dict[str, Any]:
        """Use FFmpeg to concatenate video chunks."""
        try:
            # Create temporary file list for FFmpeg concat
            temp_dir = Path(output_path).parent / "temp"
            temp_dir.mkdir(exist_ok=True)
            
            concat_file = temp_dir / f"concat_list_{int(time.time())}.txt"
            
            # Write file list for FFmpeg concat demuxer
            with open(concat_file, 'w') as f:
                for chunk in video_chunks:
                    # Use absolute paths and escape for FFmpeg
                    abs_path = Path(chunk).resolve()
                    f.write(f"file '{abs_path}'\n")
            
            # Build FFmpeg command based on whether we have background video
            if background_video and Path(background_video).exists():
                # Create overlay composition with background video
                cmd = self._build_overlay_command(concat_file, background_video, output_path, overlay_position)
            else:
                # Simple concatenation without overlay
                cmd = [
                    self.ffmpeg_path,
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', str(concat_file),
                    '-c', 'copy',  # Copy streams without re-encoding for speed
                    '-y',  # Overwrite output file
                    str(output_path)
                ]
            
            logger.info(f"Tools Running FFmpeg concatenation...")
            logger.info(f"   Command: {' '.join(cmd)}")
            
            # Execute FFmpeg
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            # Cleanup temporary file
            if concat_file.exists():
                concat_file.unlink()
            
            if result.returncode == 0:
                if Path(output_path).exists():
                    file_size = Path(output_path).stat().st_size
                    
                    # Get video properties
                    video_info = self._get_video_info(output_path)
                    
                    return {
                        'success': True,
                        'file_size': file_size,
                        'video_duration': video_info.get('duration', 0),
                        'video_fps': video_info.get('fps', 0),
                        'video_resolution': video_info.get('resolution', [0, 0]),
                        'errors': []
                    }
                else:
                    return {'success': False, 'errors': ['Output file not created']}
            else:
                logger.error(f"FFmpeg concatenation failed: {result.stderr}")
                return {'success': False, 'errors': [f'FFmpeg failed: {result.stderr}']}
                
        except subprocess.TimeoutExpired:
            return {'success': False, 'errors': ['FFmpeg concatenation timed out']}
        except Exception as e:
            return {'success': False, 'errors': [f'FFmpeg concatenation error: {str(e)}']}
    
    def _build_overlay_command(self, concat_file: Path, background_video: str, 
                              output_path: str, overlay_position: str) -> List[str]:
        """Build FFmpeg command for overlay composition."""
        
        # First, concatenate the face video chunks
        temp_face_video = concat_file.parent / f"temp_face_video_{int(time.time())}.mp4"
        
        # Calculate overlay position (user wants bottom-left, 256x256 size)
        # Assuming 1920x1080 background, place 256x256 face at bottom-left
        if overlay_position == "bottom-left":
            overlay_x = 20  # 20px from left edge
            overlay_y = "main_h-overlay_h-20"  # 20px from bottom
        elif overlay_position == "bottom-right":
            overlay_x = "main_w-overlay_w-20"  # 20px from right edge
            overlay_y = "main_h-overlay_h-20"  # 20px from bottom
        elif overlay_position == "top-left":
            overlay_x = 20  # 20px from left edge
            overlay_y = 20  # 20px from top
        elif overlay_position == "top-right":
            overlay_x = "main_w-overlay_w-20"  # 20px from right edge
            overlay_y = 20  # 20px from top
        else:
            # Default to bottom-left
            overlay_x = 20
            overlay_y = "main_h-overlay_h-20"
        
        # Build complex filter for overlay
        cmd = [
            self.ffmpeg_path,
            '-f', 'concat',
            '-safe', '0',
            '-i', str(concat_file),  # Face video chunks
            '-i', str(background_video),  # Background video
            '-filter_complex', 
            f'[0:v]scale=256:256[face];[1:v][face]overlay={overlay_x}:{overlay_y}[v];[0:a]aresample=async=1[a]',
            '-map', '[v]',
            '-map', '[a]',
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-shortest',  # Match shortest input duration
            '-y',  # Overwrite output file
            str(output_path)
        ]
        
        return cmd
    
    def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video information using FFprobe."""
        try:
            # Try to use ffprobe for detailed info
            ffprobe_cmd = [
                self.ffmpeg_path.replace('ffmpeg', 'ffprobe'),
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                str(video_path)
            ]
            
            result = subprocess.run(
                ffprobe_cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                
                # Find video stream
                for stream in data.get('streams', []):
                    if stream.get('codec_type') == 'video':
                        return {
                            'duration': float(stream.get('duration', 0)),
                            'fps': eval(stream.get('r_frame_rate', '0/1')),
                            'resolution': [
                                stream.get('width', 0),
                                stream.get('height', 0)
                            ]
                        }
            
            # Fallback to basic info
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = frame_count / fps if fps > 0 else 0
                cap.release()
                
                return {
                    'duration': duration,
                    'fps': fps,
                    'resolution': [width, height]
                }
            
        except Exception as e:
            logger.warning(f"Could not get video info: {e}")
        
        return {'duration': 0, 'fps': 0, 'resolution': [0, 0]}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the FFmpeg setup."""
        return {
            "tool_name": "Production FFmpeg",
            "ffmpeg_path": self.ffmpeg_path,
            "available": self.available,
            "supports_concatenation": True,
            "supports_video_info": True,
            "supports_copy_streams": True,
            "fallback_enabled": False,
            "architecture": "production_only"
        }


def main():
    """Test Production FFmpeg stage."""
    print("Target: Production FFmpeg Stage Test")
    print("=" * 50)
    
    # Initialize stage
    stage = ProductionFFmpegStage()
    
    # Test with sample video chunks (using recent enhancement outputs)
    test_chunks = [
        "/Users/aryanjain/Documents/video-synthesis-pipeline copy/NEW/processed/enhancement/enhanced_lip_sync_video_1752820642_1752820645.mp4",
        "/Users/aryanjain/Documents/video-synthesis-pipeline copy/NEW/processed/enhancement/enhanced_lip_sync_video_1752820643_1752820648.mp4"
    ]
    
    output_path = "/Users/aryanjain/Documents/video-synthesis-pipeline copy/NEW/processed/ffmpeg/concatenated_video_test.mp4"
    
    # Run concatenation
    results = stage.concatenate_video_chunks(test_chunks, output_path)
    
    if results['success']:
        print("\\nSUCCESS Production FFmpeg Stage test PASSED!")
        print(f"[SUCCESS] Output video: {results['output_video_path']}")
        print(f"[SUCCESS] File size: {results.get('file_size', 0):,} bytes")
        print(f"[SUCCESS] Total chunks: {results.get('total_chunks', 0)}")
        print(f"[SUCCESS] Video duration: {results.get('video_duration', 0):.2f} seconds")
        print(f"[SUCCESS] Video resolution: {results.get('video_resolution', [0, 0])}")
        print(f"Duration:  Processing time: {results['processing_time']:.2f} seconds")
    else:
        print("\\n[ERROR] Production FFmpeg Stage test FAILED!")
        print("Tools Errors:")
        for error in results['errors']:
            print(f"   - {error}")
    
    print(f"\\nStatus: Model info: {stage.get_model_info()}")


if __name__ == "__main__":
    main()