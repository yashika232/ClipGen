#!/usr/bin/env python3
"""
Enhanced Voice Cloning Stage for Video Synthesis Pipeline
Integrates with metadata-driven architecture and uses generated clean scripts
"""

import os
import sys
import subprocess
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import wave

# Add the core directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_metadata_manager import EnhancedMetadataManager

# Try to import existing XTTS components
try:
    # Import the new production XTTS stage
    from production_xtts_stage import ProductionXTTSStage
    XTTS_AVAILABLE = True
    XTTS_TYPE = "production_xtts_stage"
except ImportError:
    try:
        # Fallback to legacy system
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "legacy_docs" / "INTEGRATED_PIPELINE" / "src"))
        from xtts_stage import XTTSStage
        XTTS_AVAILABLE = True
        XTTS_TYPE = "xtts_stage"
    except ImportError:
        XTTS_AVAILABLE = False
        XTTS_TYPE = None


class EnhancedVoiceCloningStage:
    """Enhanced voice cloning stage with metadata-driven architecture."""
    
    def __init__(self, base_dir: str = None):
        """Initialize enhanced voice cloning stage.
        
        Args:
            base_dir: Base directory for the pipeline. Defaults to NEW directory.
        """
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        # Initialize metadata manager
        self.metadata_manager = EnhancedMetadataManager(str(self.base_dir))
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Output directories
        self.voice_output_dir = self.base_dir / "processed" / "voice_chunks"
        self.voice_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to initialize XTTS stage
        self.xtts_stage = None
        self.xtts_type = XTTS_TYPE
        if XTTS_AVAILABLE:
            try:
                if XTTS_TYPE == "production_xtts_stage":
                    # Initialize the ProductionXTTSStage
                    self.xtts_stage = ProductionXTTSStage(project_root=str(self.base_dir))
                    self.logger.info("[SUCCESS] ProductionXTTSStage initialized successfully")
                elif XTTS_TYPE == "xtts_stage":
                    # Initialize the legacy XTTSStage
                    self.xtts_stage = XTTSStage()
                    self.logger.info("[SUCCESS] XTTSStage initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize XTTS stage: {e}")
                self.xtts_stage = None
                self.xtts_type = None
        
        # Voice cloning configuration
        self.voice_config = {
            'max_chunk_duration': 60,  # seconds
            'words_per_minute': 150,   # for duration estimation
            'chunk_overlap': 1,        # seconds overlap between chunks
            'audio_format': 'wav',
            'sample_rate': 22050
        }
        
        self.logger.info("STARTING Enhanced Voice Cloning Stage initialized")
    
    def process_voice_cloning(self) -> Dict[str, Any]:
        """Process voice cloning using metadata-driven approach.
        
        Returns:
            Dictionary with processing results
        """
        try:
            # Update stage status to processing
            self.metadata_manager.update_stage_status(
                "voice_cloning", 
                "processing",
                {"input_audio": "user_assets/voices/"}
            )
            
            # Load current session metadata
            metadata = self.metadata_manager.load_metadata()
            if not metadata:
                raise ValueError("No active session found")
            
            # Get clean script from generated content
            generated_content = metadata.get('generated_content', {})
            clean_script = generated_content.get('clean_script')
            
            if not clean_script:
                raise ValueError("No clean script found in generated content")
            
            # Get user assets
            user_assets = metadata.get('user_assets', {})
            voice_sample_path = user_assets.get('voice_sample')
            
            if not voice_sample_path:
                raise ValueError("No voice sample found in user assets")
            
            # Convert to absolute path
            voice_sample_full_path = self.base_dir / voice_sample_path
            if not voice_sample_full_path.exists():
                raise ValueError(f"Voice sample file not found: {voice_sample_full_path}")
            
            # Get user preferences
            user_inputs = metadata.get('user_inputs', {})
            tone = user_inputs.get('tone', 'professional')
            emotion = user_inputs.get('emotion', 'confident')
            content_type = user_inputs.get('content_type', 'Short-Form Video Reel')
            
            self.logger.info(f"Target: Processing voice cloning:")
            self.logger.info(f"   Voice sample: {voice_sample_path}")
            self.logger.info(f"   Script length: {len(clean_script)} characters")
            self.logger.info(f"   Tone: {tone}, Emotion: {emotion}")
            self.logger.info(f"   Content type: {content_type}")
            
            # Determine processing strategy
            pipeline_config = metadata.get('pipeline_config', {})
            auto_chunking = pipeline_config.get('auto_chunking', True)
            chunk_duration = pipeline_config.get('chunk_duration', 10)
            
            processing_start = time.time()
            
            if auto_chunking and self._should_chunk_script(clean_script):
                # Process with chunking for longer content
                result = self._process_chunked_voice_cloning(
                    clean_script, str(voice_sample_full_path), tone, emotion, chunk_duration
                )
            else:
                # Process as single file for shorter content
                result = self._process_single_voice_cloning(
                    clean_script, str(voice_sample_full_path), tone, emotion
                )
            
            processing_time = time.time() - processing_start
            
            if result['success']:
                # Update metadata with successful results
                stage_data = {
                    "output_voice": result['output_path'],
                    "chunks": result.get('chunks', []),
                    "processing_duration": processing_time,
                    "script_length": len(clean_script),
                    "synthesis_method": "chunked" if result.get('chunks') else "single",
                    "voice_config": {
                        "tone": tone,
                        "emotion": emotion,
                        "content_type": content_type
                    }
                }
                
                self.metadata_manager.update_stage_status(
                    "voice_cloning",
                    "completed",
                    stage_data
                )
                
                self.logger.info(f"[SUCCESS] Voice cloning completed successfully in {processing_time:.1f}s")
                self.logger.info(f"   Output: {result['output_path']}")
                
                return {
                    'success': True,
                    'output_path': result['output_path'],
                    'chunks': result.get('chunks', []),
                    'processing_time': processing_time,
                    'stage_updated': True
                }
            else:
                # Update metadata with failure
                error_data = {
                    "processing_duration": processing_time,
                    "error_details": result.get('error', 'Unknown error')
                }
                
                self.metadata_manager.update_stage_status(
                    "voice_cloning",
                    "failed",
                    error_data
                )
                
                self.logger.error(f"[ERROR] Voice cloning failed: {result.get('error')}")
                return result
        
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"[ERROR] Voice cloning stage error: {error_msg}")
            
            # Update metadata with error
            self.metadata_manager.update_stage_status(
                "voice_cloning",
                "failed",
                {"error": error_msg}
            )
            
            return {
                'success': False,
                'error': error_msg,
                'stage_updated': True
            }
    
    def _should_chunk_script(self, script: str, max_duration: int = 60) -> bool:
        """Determine if script should be chunked based on estimated duration.
        
        Args:
            script: Script text to analyze
            max_duration: Maximum duration before chunking (seconds)
            
        Returns:
            True if chunking is recommended
        """
        words = len(script.split())
        estimated_duration = (words / self.voice_config['words_per_minute']) * 60
        
        self.logger.info(f"   Estimated duration: {estimated_duration:.1f}s ({words} words)")
        return estimated_duration > max_duration
    
    def _process_single_voice_cloning(self, script: str, voice_sample_path: str,
                                    tone: str, emotion: str) -> Dict[str, Any]:
        """Process voice cloning as a single file.
        
        Args:
            script: Clean script text
            voice_sample_path: Path to voice sample file
            tone: Voice tone parameter
            emotion: Voice emotion parameter
            
        Returns:
            Processing results
        """
        try:
            timestamp = int(datetime.now().timestamp())
            output_filename = f"voice_synthesis_{timestamp}.wav"
            output_path = self.voice_output_dir / output_filename
            
            self.logger.info("Recording Synthesizing single voice file...")
            
            # Synthesize voice
            synthesis_result = self._synthesize_voice(
                script, voice_sample_path, str(output_path), tone, emotion
            )
            
            if synthesis_result['success']:
                # Get relative path for metadata
                relative_path = output_path.relative_to(self.base_dir)
                
                return {
                    'success': True,
                    'output_path': str(relative_path),
                    'synthesis_method': 'single'
                }
            else:
                return synthesis_result
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _process_chunked_voice_cloning(self, script: str, voice_sample_path: str,
                                     tone: str, emotion: str, chunk_duration: int) -> Dict[str, Any]:
        """Process voice cloning with script chunking.
        
        Args:
            script: Clean script text
            voice_sample_path: Path to voice sample file
            tone: Voice tone parameter
            emotion: Voice emotion parameter
            chunk_duration: Target duration per chunk (seconds)
            
        Returns:
            Processing results
        """
        try:
            # Split script into chunks
            script_chunks = self._chunk_script(script, chunk_duration)
            
            timestamp = int(datetime.now().timestamp())
            chunk_paths = []
            
            self.logger.info(f"Recording Synthesizing {len(script_chunks)} voice chunks...")
            
            # Process each chunk
            for i, chunk_text in enumerate(script_chunks):
                chunk_filename = f"voice_chunk_{timestamp}_{i:03d}.wav"
                chunk_path = self.voice_output_dir / chunk_filename
                
                self.logger.info(f"   Processing chunk {i+1}/{len(script_chunks)}")
                
                synthesis_result = self._synthesize_voice(
                    chunk_text, voice_sample_path, str(chunk_path), tone, emotion
                )
                
                if synthesis_result['success']:
                    relative_path = chunk_path.relative_to(self.base_dir)
                    chunk_paths.append(str(relative_path))
                else:
                    return {
                        'success': False,
                        'error': f"Chunk {i+1} synthesis failed: {synthesis_result.get('error')}"
                    }
            
            # Concatenate chunks into final audio
            final_filename = f"voice_synthesis_combined_{timestamp}.wav"
            final_path = self.voice_output_dir / final_filename
            
            concat_result = self._concatenate_audio_chunks(chunk_paths, final_path)
            
            if concat_result['success']:
                relative_final_path = final_path.relative_to(self.base_dir)
                
                return {
                    'success': True,
                    'output_path': str(relative_final_path),
                    'chunks': chunk_paths,
                    'synthesis_method': 'chunked'
                }
            else:
                return concat_result
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _chunk_script(self, script: str, chunk_duration: int) -> List[str]:
        """Split script into chunks based on target duration.
        
        Args:
            script: Script text to chunk
            chunk_duration: Target duration per chunk (seconds)
            
        Returns:
            List of script chunks
        """
        # Calculate target words per chunk
        words_per_chunk = int((chunk_duration / 60) * self.voice_config['words_per_minute'])
        
        # Split by sentences for more natural chunks
        sentences = script.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            # If adding this sentence would exceed the target, start a new chunk
            if current_word_count + sentence_words > words_per_chunk and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_word_count = sentence_words
            else:
                current_chunk.append(sentence)
                current_word_count += sentence_words
        
        # Add remaining sentences as final chunk
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        
        self.logger.info(f"   Script split into {len(chunks)} natural chunks")
        return chunks
    
    def _synthesize_voice(self, text: str, voice_sample_path: str, output_path: str,
                         tone: str, emotion: str) -> Dict[str, Any]:
        """Synthesize voice using XTTS system - PRODUCTION MODE ONLY.
        
        Args:
            text: Text to synthesize
            voice_sample_path: Reference voice sample
            output_path: Output file path
            tone: Voice tone
            emotion: Voice emotion
            
        Returns:
            Synthesis results
        """
        try:
            if self.xtts_stage:
                # Use XTTS - PRODUCTION MODE ONLY
                return self._synthesize_with_xtts(text, voice_sample_path, output_path, tone, emotion)
            else:
                # FAIL FAST - NO FALLBACK SYNTHESIS
                self.logger.error("[ERROR] XTTS not available - PRODUCTION MODE REQUIRES XTTS")
                return {'success': False, 'error': 'XTTS not available - production mode requires XTTS. No fallback synthesis allowed.'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _synthesize_with_xtts(self, text: str, voice_sample_path: str, output_path: str,
                             tone: str, emotion: str) -> Dict[str, Any]:
        """Synthesize using Production XTTS stage.
        
        Args:
            text: Text to synthesize
            voice_sample_path: Reference voice sample
            output_path: Output file path
            tone: Voice tone
            emotion: Voice emotion
            
        Returns:
            Synthesis results
        """
        try:
            # Prepare XTTS parameters based on emotion and tone
            xtts_params = {
                'temperature': 0.7,
                'speed': 1.0,
                'repetition_penalty': 1.1,
                'top_k': 50,
                'top_p': 0.8,
                'length_penalty': 1.0
            }
            
            # Adjust parameters based on emotion and tone
            if emotion == 'confident':
                xtts_params['temperature'] = 0.6
                xtts_params['speed'] = 1.1
            elif emotion == 'inspired':
                xtts_params['temperature'] = 0.8
                xtts_params['speed'] = 1.0
            elif emotion == 'professional':
                xtts_params['temperature'] = 0.5
                xtts_params['speed'] = 0.9
            
            if tone == 'professional':
                xtts_params['repetition_penalty'] = 1.2
            elif tone == 'friendly':
                xtts_params['temperature'] = 0.8
            elif tone == 'motivational':
                xtts_params['speed'] = 1.1
            
            # Prepare script data
            script_data = {
                'text_chunks': [text],
                'clean_script': text,
                'tone': tone,
                'emotion': emotion
            }
            
            # Use ProductionXTTSStage
            result = self.xtts_stage.process_voice_cloning(
                voice_sample_path=voice_sample_path,
                script_data=script_data,
                xtts_params=xtts_params
            )
            
            if result['success']:
                # Copy the output file to the desired location
                import shutil
                output_audio_path = result['output_audio_path']
                shutil.copy2(output_audio_path, output_path)
                
                return {
                    'success': True,
                    'output_path': output_path,
                    'original_path': output_audio_path,
                    'file_size': result.get('file_size', 0),
                    'processing_time': result.get('processing_time', 0)
                }
            else:
                return {
                    'success': False,
                    'error': f"Production XTTS failed: {result.get('errors', 'Unknown error')}"
                }
            
        except Exception as e:
            return {'success': False, 'error': f"XTTS synthesis failed: {str(e)}"}
    
    def _synthesize_with_fallback(self, text: str, voice_sample_path: str, output_path: str) -> Dict[str, Any]:
        """Fallback synthesis method when XTTS is not available.
        
        Args:
            text: Text to synthesize
            voice_sample_path: Reference voice sample
            output_path: Output file path
            
        Returns:
            Synthesis results
        """
        try:
            # Create a simple test audio file as fallback
            self.logger.warning("Using fallback synthesis - creating test audio file")
            
            # Generate a simple sine wave as placeholder
            import numpy as np
            duration = max(len(text.split()) / self.voice_config['words_per_minute'] * 60, 5)
            sample_rate = self.voice_config['sample_rate']
            
            # Generate test tone
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio_data = np.sin(2 * np.pi * 440 * t) * 0.3  # 440 Hz tone
            
            # Add some variation
            audio_data += np.sin(2 * np.pi * 880 * t) * 0.1
            audio_data = (audio_data * 32767).astype(np.int16)
            
            # Save as WAV
            with wave.open(output_path, 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            return {'success': True, 'method': 'fallback_test_audio'}
            
        except Exception as e:
            return {'success': False, 'error': f"Fallback synthesis failed: {str(e)}"}
    
    def _concatenate_audio_chunks(self, chunk_paths: List[str], output_path: Path) -> Dict[str, Any]:
        """Concatenate audio chunks using FFmpeg or fallback method.
        
        Args:
            chunk_paths: List of chunk paths (relative to base_dir)
            output_path: Output path for concatenated audio
            
        Returns:
            Concatenation results
        """
        try:
            # Convert to absolute paths and verify existence
            abs_chunk_paths = []
            for chunk_path in chunk_paths:
                abs_path = self.base_dir / chunk_path
                if abs_path.exists():
                    abs_chunk_paths.append(abs_path)
                else:
                    self.logger.warning(f"Chunk not found: {abs_path}")
            
            if not abs_chunk_paths:
                return {'success': False, 'error': 'No valid chunks found for concatenation'}
            
            # Try FFmpeg first
            ffmpeg_result = self._concatenate_with_ffmpeg(abs_chunk_paths, output_path)
            if ffmpeg_result['success']:
                return ffmpeg_result
            
            # Fallback to Python-based concatenation
            self.logger.info("FFmpeg failed, using Python-based concatenation")
            return self._concatenate_with_python(abs_chunk_paths, output_path)
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _concatenate_with_ffmpeg(self, chunk_paths: List[Path], output_path: Path) -> Dict[str, Any]:
        """Concatenate audio using FFmpeg.
        
        Args:
            chunk_paths: List of absolute chunk paths
            output_path: Output path
            
        Returns:
            Concatenation results
        """
        try:
            # Create concat file
            concat_file = output_path.with_suffix('.txt')
            with open(concat_file, 'w') as f:
                for chunk_path in chunk_paths:
                    f.write(f"file '{chunk_path}'\n")
            
            # Run FFmpeg
            cmd = [
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                '-i', str(concat_file), '-c', 'copy', str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            # Cleanup
            concat_file.unlink(missing_ok=True)
            
            if result.returncode == 0:
                self.logger.info("[SUCCESS] Audio concatenated with FFmpeg")
                return {'success': True, 'method': 'ffmpeg'}
            else:
                return {'success': False, 'error': f"FFmpeg failed: {result.stderr}"}
                
        except Exception as e:
            return {'success': False, 'error': f"FFmpeg concatenation failed: {str(e)}"}
    
    def _concatenate_with_python(self, chunk_paths: List[Path], output_path: Path) -> Dict[str, Any]:
        """Concatenate audio using Python wave module.
        
        Args:
            chunk_paths: List of absolute chunk paths
            output_path: Output path
            
        Returns:
            Concatenation results
        """
        try:
            # Read all chunks and concatenate
            concatenated_frames = b''
            sample_rate = None
            
            for chunk_path in chunk_paths:
                with wave.open(str(chunk_path), 'rb') as wav_file:
                    if sample_rate is None:
                        sample_rate = wav_file.getframerate()
                    
                    frames = wav_file.readframes(wav_file.getnframes())
                    concatenated_frames += frames
            
            # Write concatenated audio
            with wave.open(str(output_path), 'wb') as output_wav:
                output_wav.setnchannels(1)
                output_wav.setsampwidth(2)
                output_wav.setframerate(sample_rate or self.voice_config['sample_rate'])
                output_wav.writeframes(concatenated_frames)
            
            self.logger.info("[SUCCESS] Audio concatenated with Python")
            return {'success': True, 'method': 'python'}
            
        except Exception as e:
            return {'success': False, 'error': f"Python concatenation failed: {str(e)}"}
    
    def get_voice_cloning_status(self) -> Dict[str, Any]:
        """Get current voice cloning stage status.
        
        Returns:
            Status information
        """
        try:
            stage_status = self.metadata_manager.get_stage_status("voice_cloning")
            if stage_status:
                return {
                    'success': True,
                    'stage_status': stage_status
                }
            else:
                return {
                    'success': False,
                    'error': 'No voice cloning stage found in metadata'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def validate_voice_cloning_prerequisites(self) -> Dict[str, Any]:
        """Validate that voice cloning can be started.
        
        Returns:
            Validation results
        """
        try:
            # Check metadata
            metadata = self.metadata_manager.load_metadata()
            if not metadata:
                return {'valid': False, 'errors': ['No active session found']}
            
            errors = []
            warnings = []
            
            # Check for clean script
            generated_content = metadata.get('generated_content', {})
            if not generated_content.get('clean_script'):
                errors.append('No clean script generated yet')
            
            # Check for voice sample
            user_assets = metadata.get('user_assets', {})
            voice_sample = user_assets.get('voice_sample')
            if not voice_sample:
                errors.append('No voice sample uploaded')
            elif voice_sample:
                voice_path = self.base_dir / voice_sample
                if not voice_path.exists():
                    errors.append(f'Voice sample file not found: {voice_sample}')
            
            # Check XTTS availability
            if not self.xtts_stage:
                warnings.append('XTTS not available - will use fallback synthesis')
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings,
                'ready_for_processing': len(errors) == 0
            }
            
        except Exception as e:
            return {
                'valid': False,
                'errors': [str(e)]
            }


def main():
    """Test the enhanced voice cloning stage."""
    print("🧪 Testing Enhanced Voice Cloning Stage")
    print("=" * 50)
    
    # Initialize stage
    stage = EnhancedVoiceCloningStage()
    
    # Check prerequisites
    prereq_result = stage.validate_voice_cloning_prerequisites()
    print(f"[SUCCESS] Prerequisites check:")
    print(f"   Valid: {prereq_result['valid']}")
    if prereq_result.get('errors'):
        print(f"   Errors: {prereq_result['errors']}")
    if prereq_result.get('warnings'):
        print(f"   Warnings: {prereq_result['warnings']}")
    
    # Check current status
    status_result = stage.get_voice_cloning_status()
    if status_result['success']:
        stage_info = status_result['stage_status']
        print(f"[SUCCESS] Current status: {stage_info.get('status', 'unknown')}")
    else:
        print(f"[ERROR] Status check failed: {status_result['error']}")
    
    # Process voice cloning if prerequisites are met
    if prereq_result['valid']:
        print("\nTarget: Starting voice cloning processing...")
        result = stage.process_voice_cloning()
        
        if result['success']:
            print("[SUCCESS] Voice cloning completed successfully!")
            print(f"   Output: {result['output_path']}")
            if result.get('chunks'):
                print(f"   Chunks: {len(result['chunks'])}")
        else:
            print(f"[ERROR] Voice cloning failed: {result['error']}")
    else:
        print("\n[ERROR] Prerequisites not met - skipping processing")
    
    print("\nSUCCESS Enhanced Voice Cloning Stage testing completed!")


if __name__ == "__main__":
    main()