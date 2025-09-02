#!/usr/bin/env python3
"""
Voice Cloning Stage for NEW Video Pipeline
Integrates XTTS voice cloning with NEW metadata system
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

# Add paths for existing XTTS integration
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "INTEGRATED_PIPELINE" / "src"))

from metadata_manager import MetadataManager

try:
    from xtts_stage import XTTSStage
except ImportError:
    # Fallback if direct import fails
    XTTSStage = None


class VoiceCloningStage:
    """Voice cloning stage that integrates with NEW metadata system."""
    
    def __init__(self, metadata_manager: MetadataManager = None):
        """Initialize voice cloning stage.
        
        Args:
            metadata_manager: Metadata manager instance
        """
        self.metadata_manager = metadata_manager or MetadataManager()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Output directory
        self.output_dir = self.metadata_manager.output_dir / "voice_cloning"
        self.output_dir.mkdir(exist_ok=True)
        
        # Try to import existing XTTS stage
        self.xtts_stage = None
        if XTTSStage:
            try:
                self.xtts_stage = XTTSStage()
                self.logger.info("[SUCCESS] XTTS stage imported successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize XTTS stage: {e}")
    
    def should_chunk_audio(self, script_text: str, max_duration: int = 60) -> bool:
        """Determine if audio should be chunked based on script length.
        
        Args:
            script_text: Script text to analyze
            max_duration: Maximum duration before chunking (seconds)
            
        Returns:
            True if chunking is needed, False otherwise
        """
        # Estimate duration (approximately 150 words per minute)
        words = len(script_text.split())
        estimated_duration = (words / 150) * 60
        
        self.logger.info(f"Estimated audio duration: {estimated_duration:.1f} seconds ({words} words)")
        return estimated_duration > max_duration
    
    def extract_script_text(self, metadata: Dict[str, Any]) -> str:
        """Extract clean script text from metadata.
        
        Args:
            metadata: Full metadata dictionary
            
        Returns:
            Clean script text for voice synthesis
        """
        script_generated = metadata.get("script_generated", {})
        
        # Get core content
        core_content = script_generated.get("core_content", "")
        
        if not core_content:
            # Fallback to combining sections
            sections = []
            for section in ["hook", "objectives", "interactive", "summary"]:
                section_content = script_generated.get(section, "")
                if section_content:
                    sections.append(section_content)
            core_content = "\n\n".join(sections)
        
        # Clean up the script for voice synthesis
        cleaned_script = self._clean_script_for_voice(core_content)
        return cleaned_script
    
    def _clean_script_for_voice(self, script: str) -> str:
        """Clean script text for voice synthesis.
        
        Args:
            script: Raw script text
            
        Returns:
            Cleaned script text
        """
        # Remove markdown formatting
        cleaned = script.replace("**", "").replace("*", "")
        cleaned = cleaned.replace("#", "").replace("`", "")
        
        # Remove stage directions and timestamps
        lines = []
        for line in cleaned.split('\n'):
            line = line.strip()
            
            # Skip empty lines and stage directions
            if not line or line.startswith('(') or '--' in line:
                continue
            
            # Remove narrator labels
            if line.startswith("**Narrator:**"):
                line = line.replace("**Narrator:**", "").strip()
            
            if line:
                lines.append(line)
        
        return " ".join(lines)
    
    def chunk_script_text(self, script_text: str, chunk_duration: int = 10) -> List[str]:
        """Chunk script text into segments for voice synthesis.
        
        Args:
            script_text: Full script text
            chunk_duration: Target duration per chunk (seconds)
            
        Returns:
            List of script chunks
        """
        # Estimate words per chunk (150 words per minute)
        words_per_chunk = int((chunk_duration / 60) * 150)
        
        words = script_text.split()
        chunks = []
        
        for i in range(0, len(words), words_per_chunk):
            chunk_words = words[i:i + words_per_chunk]
            chunk_text = " ".join(chunk_words)
            chunks.append(chunk_text)
        
        self.logger.info(f"Script chunked into {len(chunks)} segments")
        return chunks
    
    def synthesize_voice_with_xtts(self, script_text: str, voice_sample_path: str,
                                 output_path: str, emotion: str = "inspired", 
                                 tone: str = "professional") -> Dict[str, Any]:
        """Synthesize voice using XTTS.
        
        Args:
            script_text: Text to synthesize
            voice_sample_path: Path to voice sample
            output_path: Output path for synthesized audio
            emotion: Emotion parameter
            tone: Tone parameter
            
        Returns:
            Synthesis results dictionary
        """
        try:
            if self.xtts_stage:
                # Use existing XTTS stage
                result = self.xtts_stage.synthesize_speech(
                    text=script_text,
                    voice_reference=voice_sample_path,
                    output_path=output_path,
                    emotion=emotion,
                    tone=tone
                )
                return result
            else:
                # Fallback to subprocess call
                return self._synthesize_with_subprocess(script_text, voice_sample_path, output_path)
        
        except Exception as e:
            self.logger.error(f"XTTS synthesis failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _synthesize_with_subprocess(self, script_text: str, voice_sample_path: str,
                                  output_path: str) -> Dict[str, Any]:
        """Fallback synthesis using subprocess.
        
        Args:
            script_text: Text to synthesize
            voice_sample_path: Path to voice sample
            output_path: Output path for synthesized audio
            
        Returns:
            Synthesis results dictionary
        """
        try:
            # Use existing XTTS stage via subprocess
            xtts_script = Path(__file__).parent.parent.parent / "INTEGRATED_PIPELINE" / "src" / "xtts_stage.py"
            
            if not xtts_script.exists():
                return {"success": False, "error": "XTTS stage script not found"}
            
            # Create temporary script file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(script_text)
                script_file = f.name
            
            try:
                # Run XTTS via subprocess
                cmd = [
                    sys.executable, str(xtts_script),
                    "--text-file", script_file,
                    "--voice-reference", voice_sample_path,
                    "--output", output_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    return {"success": True, "output_path": output_path}
                else:
                    return {"success": False, "error": result.stderr}
            
            finally:
                # Cleanup temporary file
                os.unlink(script_file)
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def process_voice_cloning(self) -> Dict[str, Any]:
        """Process voice cloning stage using metadata.
        
        Returns:
            Processing results dictionary
        """
        start_time = time.time()
        
        # Update status to processing
        self.metadata_manager.update_stage_status("voice_cloning", "processing")
        
        try:
            # Load metadata
            metadata = self.metadata_manager.load_metadata()
            if metadata is None:
                raise ValueError("Failed to load metadata")
            
            # Get voice sample path
            voice_sample_path = self.metadata_manager.get_stage_input("voice_cloning", "input_audio")
            if voice_sample_path is None:
                raise ValueError("Voice sample not found in metadata")
            
            # Extract script text
            script_text = self.extract_script_text(metadata)
            if not script_text:
                raise ValueError("No script text found in metadata")
            
            # Get emotion and tone parameters
            emotion = metadata.get("emotion", "inspired")
            tone = metadata.get("tone", "professional")
            
            self.logger.info(f"Processing voice cloning:")
            self.logger.info(f"  Voice sample: {voice_sample_path}")
            self.logger.info(f"  Script length: {len(script_text)} characters")
            self.logger.info(f"  Emotion: {emotion}, Tone: {tone}")
            
            # Check if chunking is needed
            pipeline_config = metadata.get("pipeline_config", {})
            auto_chunking = pipeline_config.get("auto_chunking", True)
            chunk_duration = pipeline_config.get("chunk_duration", 10)
            
            if auto_chunking and self.should_chunk_audio(script_text):
                # Process with chunking
                result = self._process_chunked_voice_cloning(
                    script_text, voice_sample_path, emotion, tone, chunk_duration
                )
            else:
                # Process as single file
                result = self._process_single_voice_cloning(
                    script_text, voice_sample_path, emotion, tone
                )
            
            processing_time = time.time() - start_time
            
            if result["success"]:
                # Update metadata with success
                update_data = {
                    "output_voice": result.get("output_path"),
                    "chunks": result.get("chunks", []),
                    "duration": result.get("duration", 0),
                    "processing_time": processing_time,
                    "error": None
                }
                
                self.metadata_manager.update_stage_status("voice_cloning", "completed", update_data)
                
                # Update video generation inputs
                self.metadata_manager.update_stage_status("video_generation", None, {
                    "input_script": script_text,
                    "input_voice": result.get("output_path")
                })
                
                self.logger.info(f"[SUCCESS] Voice cloning completed in {processing_time:.1f}s")
                return result
            else:
                # Update metadata with failure
                update_data = {
                    "processing_time": processing_time,
                    "error": result.get("error")
                }
                
                self.metadata_manager.update_stage_status("voice_cloning", "failed", update_data)
                
                self.logger.error(f"[ERROR] Voice cloning failed: {result.get('error')}")
                return result
        
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            # Update metadata with error
            update_data = {
                "processing_time": processing_time,
                "error": error_msg
            }
            
            self.metadata_manager.update_stage_status("voice_cloning", "failed", update_data)
            
            self.logger.error(f"[ERROR] Voice cloning stage failed: {error_msg}")
            return {"success": False, "error": error_msg}
    
    def _process_single_voice_cloning(self, script_text: str, voice_sample_path: str,
                                    emotion: str, tone: str) -> Dict[str, Any]:
        """Process voice cloning as single file.
        
        Args:
            script_text: Script text to synthesize
            voice_sample_path: Path to voice sample
            emotion: Emotion parameter
            tone: Tone parameter
            
        Returns:
            Processing results dictionary
        """
        timestamp = int(time.time())
        output_path = self.output_dir / f"voice_synthesis_{timestamp}.wav"
        
        result = self.synthesize_voice_with_xtts(
            script_text, voice_sample_path, str(output_path), emotion, tone
        )
        
        if result["success"]:
            # Convert to relative path
            relative_path = output_path.relative_to(self.metadata_manager.new_dir)
            
            return {
                "success": True,
                "output_path": str(relative_path),
                "chunks": [],
                "duration": 0  # TODO: Get actual duration
            }
        else:
            return result
    
    def _process_chunked_voice_cloning(self, script_text: str, voice_sample_path: str,
                                     emotion: str, tone: str, chunk_duration: int) -> Dict[str, Any]:
        """Process voice cloning with chunking.
        
        Args:
            script_text: Script text to synthesize
            voice_sample_path: Path to voice sample
            emotion: Emotion parameter
            tone: Tone parameter
            chunk_duration: Duration per chunk in seconds
            
        Returns:
            Processing results dictionary
        """
        # Chunk the script
        script_chunks = self.chunk_script_text(script_text, chunk_duration)
        
        timestamp = int(time.time())
        chunk_paths = []
        
        # Process each chunk
        for i, chunk_text in enumerate(script_chunks):
            chunk_output = self.output_dir / f"voice_chunk_{timestamp}_{i:03d}.wav"
            
            self.logger.info(f"Processing chunk {i+1}/{len(script_chunks)}")
            
            result = self.synthesize_voice_with_xtts(
                chunk_text, voice_sample_path, str(chunk_output), emotion, tone
            )
            
            if result["success"]:
                relative_path = chunk_output.relative_to(self.metadata_manager.new_dir)
                chunk_paths.append(str(relative_path))
            else:
                self.logger.error(f"Failed to process chunk {i+1}: {result.get('error')}")
                return {"success": False, "error": f"Chunk {i+1} failed: {result.get('error')}"}
        
        # Concatenate chunks using existing logic
        final_output = self.output_dir / f"voice_synthesis_combined_{timestamp}.wav"
        concat_result = self._concatenate_audio_chunks(chunk_paths, final_output)
        
        if concat_result["success"]:
            relative_path = final_output.relative_to(self.metadata_manager.new_dir)
            
            return {
                "success": True,
                "output_path": str(relative_path),
                "chunks": chunk_paths,
                "duration": 0  # TODO: Get actual duration
            }
        else:
            return concat_result
    
    def _concatenate_audio_chunks(self, chunk_paths: List[str], output_path: Path) -> Dict[str, Any]:
        """Concatenate audio chunks using FFmpeg.
        
        Args:
            chunk_paths: List of chunk file paths (relative to NEW dir)
            output_path: Output path for concatenated audio
            
        Returns:
            Concatenation results dictionary
        """
        try:
            # Convert to absolute paths
            abs_chunk_paths = []
            for chunk_path in chunk_paths:
                abs_path = self.metadata_manager.new_dir / chunk_path
                if abs_path.exists():
                    abs_chunk_paths.append(str(abs_path))
                else:
                    self.logger.warning(f"Chunk not found: {abs_path}")
            
            if not abs_chunk_paths:
                return {"success": False, "error": "No valid chunks found for concatenation"}
            
            # Create concat file for FFmpeg
            concat_file = output_path.with_suffix('.txt')
            with open(concat_file, 'w') as f:
                for chunk_path in abs_chunk_paths:
                    f.write(f"file '{chunk_path}'\\n")
            
            # Run FFmpeg concatenation
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_file),
                '-c', 'copy',
                str(output_path)
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
    """Test voice cloning stage."""
    stage = VoiceCloningStage()
    
    # Check metadata status
    metadata = stage.metadata_manager.load_metadata()
    if metadata:
        print("[SUCCESS] Metadata loaded successfully")
        
        # Check if voice cloning is ready
        voice_sample = stage.metadata_manager.get_stage_input("voice_cloning", "input_audio")
        if voice_sample:
            print(f"Recording Voice sample found: {voice_sample}")
            
            # Process voice cloning
            result = stage.process_voice_cloning()
            
            if result["success"]:
                print("[SUCCESS] Voice cloning completed successfully")
                print(f"Output: {result.get('output_path')}")
            else:
                print(f"[ERROR] Voice cloning failed: {result.get('error')}")
        else:
            print("[ERROR] No voice sample found in metadata")
    else:
        print("[ERROR] Failed to load metadata")


if __name__ == "__main__":
    main()