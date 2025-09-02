#!/usr/bin/env python3
"""
Advanced Chunked Audio Processor
Integrates existing chunking logic with NEW pipeline
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any
import logging

# Import existing chunking utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "useless_files" / "experimental_scripts"))

try:
    from xtts_clean_chunker import XTTSCleanChunker
except ImportError:
    XTTSCleanChunker = None


class ChunkedAudioProcessor:
    """Advanced chunked audio processor using proven chunking strategies."""
    
    def __init__(self):
        """Initialize chunked audio processor."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize existing chunker if available
        self.xtts_chunker = XTTSCleanChunker() if XTTSCleanChunker else None
        
        if self.xtts_chunker:
            self.logger.info("[SUCCESS] XTTS Clean Chunker available")
        else:
            self.logger.warning("[WARNING] XTTS Clean Chunker not available, using fallback")
    
    def should_chunk_audio(self, audio_path: str, max_duration: int = 60) -> bool:
        """Determine if audio should be chunked.
        
        Args:
            audio_path: Path to audio file
            max_duration: Maximum duration before chunking (seconds)
            
        Returns:
            True if chunking is needed, False otherwise
        """
        try:
            # Get audio duration using ffprobe
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                duration = float(result.stdout.strip())
                self.logger.info(f"Audio duration: {duration:.1f} seconds")
                return duration > max_duration
            else:
                self.logger.warning("Could not determine audio duration, assuming chunking needed")
                return True
        
        except Exception as e:
            self.logger.warning(f"Error checking audio duration: {e}, assuming chunking needed")
            return True
    
    def create_clean_chunks(self, audio_path: str, output_dir: str, 
                          chunk_duration: int = 10) -> List[str]:
        """Create clean non-overlapping audio chunks.
        
        Args:
            audio_path: Path to input audio file
            output_dir: Directory for output chunks
            chunk_duration: Duration per chunk in seconds
            
        Returns:
            List of chunk file paths
        """
        try:
            if self.xtts_chunker:
                # Use existing proven chunker
                chunk_paths = self.xtts_chunker.create_clean_chunks(audio_path, output_dir)
                self.logger.info(f"[SUCCESS] Created {len(chunk_paths)} clean chunks using XTTS chunker")
                return chunk_paths
            else:
                # Fallback to basic chunking
                return self._basic_clean_chunking(audio_path, output_dir, chunk_duration)
        
        except Exception as e:
            self.logger.error(f"Clean chunking failed: {e}")
            return []
    
    def _basic_clean_chunking(self, audio_path: str, output_dir: str, 
                            chunk_duration: int) -> List[str]:
        """Basic clean chunking using FFmpeg.
        
        Args:
            audio_path: Path to input audio file
            output_dir: Directory for output chunks
            chunk_duration: Duration per chunk in seconds
            
        Returns:
            List of chunk file paths
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get total duration
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise ValueError("Could not get audio duration")
            
            total_duration = float(result.stdout.strip())
            num_chunks = int(total_duration / chunk_duration) + 1
            
            chunk_paths = []
            timestamp = int(time.time())
            
            for i in range(num_chunks):
                start_time = i * chunk_duration
                if start_time >= total_duration:
                    break
                
                chunk_path = output_dir / f"clean_chunk_{timestamp}_{i:03d}.wav"
                
                # Calculate actual chunk duration (handle last chunk)
                actual_duration = min(chunk_duration, total_duration - start_time)
                
                # Use high-quality audio settings for clean chunks
                cmd = [
                    'ffmpeg', '-y',
                    '-i', audio_path,
                    '-ss', str(start_time),
                    '-t', str(actual_duration),
                    '-af', 'afade=in:st=0:d=0.1,afade=out:st=' + str(actual_duration - 0.1) + ':d=0.1',
                    '-c:a', 'pcm_s16le',
                    '-ar', '16000',
                    str(chunk_path)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0 and chunk_path.exists():
                    chunk_paths.append(str(chunk_path))
                else:
                    self.logger.warning(f"Failed to create clean chunk {i}")
            
            self.logger.info(f"[SUCCESS] Created {len(chunk_paths)} basic clean chunks")
            return chunk_paths
        
        except Exception as e:
            self.logger.error(f"Basic clean chunking failed: {e}")
            return []
    
    def preprocess_audio_for_sadtalker(self, audio_path: str, output_path: str) -> Dict[str, Any]:
        """Preprocess audio for SadTalker compatibility.
        
        Args:
            audio_path: Path to input audio
            output_path: Path for preprocessed output
            
        Returns:
            Preprocessing results dictionary
        """
        try:
            # Convert audio to SadTalker-compatible format
            # 16kHz, PCM 16-bit, mono
            cmd = [
                'ffmpeg', '-y',
                '-i', audio_path,
                '-ar', '16000',
                '-ac', '1',
                '-c:a', 'pcm_s16le',
                '-f', 'wav',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and Path(output_path).exists():
                self.logger.info(f"[SUCCESS] Audio preprocessed for SadTalker: {output_path}")
                return {"success": True, "output_path": output_path}
            else:
                return {"success": False, "error": result.stderr}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def concatenate_audio_chunks(self, chunk_paths: List[str], output_path: str) -> Dict[str, Any]:
        """Concatenate audio chunks with perfect sync.
        
        Args:
            chunk_paths: List of chunk file paths
            output_path: Path for concatenated output
            
        Returns:
            Concatenation results dictionary
        """
        try:
            # Create concat file for FFmpeg
            concat_file = Path(output_path).with_suffix('.txt')
            
            with open(concat_file, 'w') as f:
                for chunk_path in chunk_paths:
                    f.write(f"file '{chunk_path}'\\n")
            
            # Concatenate with perfect sync (no gaps or overlaps)
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_file),
                '-c', 'copy',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Cleanup concat file
            concat_file.unlink()
            
            if result.returncode == 0:
                self.logger.info(f"[SUCCESS] Audio chunks concatenated: {output_path}")
                return {"success": True}
            else:
                return {"success": False, "error": result.stderr}
        
        except Exception as e:
            return {"success": False, "error": str(e)}


def main():
    """Test chunked audio processor."""
    processor = ChunkedAudioProcessor()
    
    # Test with a sample audio file
    test_audio = "/path/to/test/audio.wav"  # Replace with actual test file
    
    if Path(test_audio).exists():
        print(f"Testing with: {test_audio}")
        
        # Check if chunking is needed
        should_chunk = processor.should_chunk_audio(test_audio)
        print(f"Should chunk: {should_chunk}")
        
        if should_chunk:
            # Create chunks
            output_dir = Path("test_chunks")
            chunks = processor.create_clean_chunks(test_audio, str(output_dir))
            
            if chunks:
                print(f"Created {len(chunks)} chunks")
                
                # Test concatenation
                output_path = "test_concatenated.wav"
                result = processor.concatenate_audio_chunks(chunks, output_path)
                
                if result["success"]:
                    print("[SUCCESS] Concatenation test successful")
                else:
                    print(f"[ERROR] Concatenation failed: {result['error']}")
            else:
                print("[ERROR] No chunks created")
    else:
        print("No test audio file available")


if __name__ == "__main__":
    main()