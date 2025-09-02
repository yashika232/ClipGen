#!/usr/bin/env python3
"""
Production XTTS Stage - Voice Cloning Implementation for Production Mode
Based on legacy INTEGRATED_PIPELINE implementation - Main model only, no fallback mechanisms
"""

import os
import sys
import time
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProductionXTTSStage:
    """Production XTTS Voice Cloning Stage - Main Model Only."""
    
    def __init__(self, project_root: str = None, output_dir: str = None):
        """Initialize Production XTTS stage.
        
        Args:
            project_root: Path to project root directory
            output_dir: Path to output directory
        """
        if project_root is None:
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = Path(project_root)
            
        if output_dir is None:
            self.output_dir = self.project_root / "NEW" / "processed" / "voice_chunks"
        else:
            self.output_dir = Path(output_dir)
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # XTTS configuration - using correct conda environment
        self.conda_python = Path("/Users/aryanjain/miniforge3/envs/xtts_voice_cloning/bin/python")
        self.max_text_length = 2000
        self.supported_languages = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"]
        
        logger.info(f"Production XTTS Stage initialized")
        logger.info(f"  Project root: {self.project_root}")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Python environment: {self.conda_python}")
    
    def validate_inputs(self, voice_sample_path: str, script_data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate inputs for XTTS voice cloning.
        
        Args:
            voice_sample_path: Path to user's voice sample
            script_data: Processed script data
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate voice sample
        voice_path = Path(voice_sample_path)
        if not voice_path.exists():
            errors.append(f"Voice sample not found: {voice_sample_path}")
        elif not voice_path.suffix.lower() in ['.wav', '.mp3', '.m4a', '.flac']:
            errors.append(f"Unsupported audio format: {voice_path.suffix}")
        elif voice_path.stat().st_size < 1000:
            errors.append(f"Voice sample too small: {voice_path.stat().st_size} bytes")
        
        # Validate script data - check for text chunks
        text_chunks = script_data.get('text_chunks', [])
        if not text_chunks:
            # Try to get from clean script
            clean_script = script_data.get('clean_script', '')
            if clean_script:
                # Create text chunks from clean script
                script_data['text_chunks'] = [clean_script]
                text_chunks = script_data['text_chunks']
            else:
                errors.append("No text chunks or clean script in script data")
        
        if text_chunks:
            for i, chunk in enumerate(text_chunks):
                if len(chunk) > self.max_text_length:
                    errors.append(f"Text chunk {i} too long: {len(chunk)} characters")
                elif len(chunk) < 5:
                    errors.append(f"Text chunk {i} too short: {len(chunk)} characters")
        
        return len(errors) == 0, errors
    
    def check_python_environment(self) -> bool:
        """Check if XTTS Python environment exists and has TTS installed.
        
        Returns:
            True if environment exists and TTS available, False otherwise
        """
        try:
            if not self.conda_python.exists():
                logger.error(f"Python environment not found: {self.conda_python}")
                return False
            
            # Test TTS import
            result = subprocess.run(
                [str(self.conda_python), '-c', 'import TTS; from TTS.api import TTS; print("TTS available")'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and "TTS available" in result.stdout:
                logger.info("[SUCCESS] Production XTTS environment and dependencies available")
                return True
            else:
                logger.error(f"TTS not available in environment: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error checking Python environment: {e}")
            return False
    
    def create_xtts_script(self, voice_sample_path: str, text_chunks: List[str], 
                          output_path: str, xtts_params: Dict[str, Any]) -> str:
        """Create XTTS voice cloning script.
        
        Args:
            voice_sample_path: Path to voice sample
            text_chunks: List of text chunks to synthesize
            output_path: Output audio file path
            xtts_params: XTTS parameters
            
        Returns:
            XTTS script content
        """
        script_content = f'''
import os
import sys
import warnings
import time
warnings.filterwarnings('ignore')

# Environment setup
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['COQUI_TOS_AGREED'] = '1'

print("STARTING Loading XTTS dependencies...")

try:
    import torch
    print(f"[SUCCESS] PyTorch loaded: {{torch.__version__}}")
    
    import torchaudio
    print(f"[SUCCESS] TorchAudio loaded: {{torchaudio.__version__}}")
    
    import numpy as np
    print("[SUCCESS] NumPy loaded")
    
    from TTS.api import TTS
    print("[SUCCESS] TTS API loaded")
    
    # Apply compatibility patches
    print("Tools Applying compatibility patches...")
    
    # Fix torch.load weights_only issue for XTTS compatibility
    original_load = torch.load
    def patched_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    torch.load = patched_load
    
    # Additional compatibility fixes for XTTS
    import torch.serialization
    try:
        from TTS.tts.configs.xtts_config import XttsConfig
        torch.serialization.add_safe_globals([XttsConfig])
    except ImportError:
        pass  # XttsConfig might not be available in all versions
    
    print("[SUCCESS] Compatibility patches applied")
    
except ImportError as e:
    print(f"[ERROR] DEPENDENCY ERROR: {{e}}")
    print("   Please install dependencies in conda environment")
    sys.exit(1)

try:
    # Initialize XTTS model
    print("Package Loading XTTS-v2 model...")
    start_time = time.time()
    
    # Use CPU for reliability on macOS
    device = "cpu"
    print(f"Tools Using device: {{device}}")
    
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    
    load_time = time.time() - start_time
    print(f"[SUCCESS] XTTS model loaded in {{load_time:.2f}} seconds")
    
    # Process text chunks
    text_chunks = {text_chunks}
    voice_sample = "{voice_sample_path}"
    output_path = "{output_path}"
    
    print(f"Target: Processing {{len(text_chunks)}} text chunks...")
    print(f"   Reference voice: {{voice_sample}}")
    print(f"   Output path: {{output_path}}")
    
    # XTTS parameters
    temperature = {xtts_params.get('temperature', 0.7)}
    length_penalty = {xtts_params.get('length_penalty', 1.0)}
    repetition_penalty = {xtts_params.get('repetition_penalty', 1.1)}
    top_k = {xtts_params.get('top_k', 50)}
    top_p = {xtts_params.get('top_p', 0.8)}
    speed = {xtts_params.get('speed', 1.0)}
    
    print(f"   Temperature: {{temperature}}")
    print(f"   Speed: {{speed}}")
    
    # Process chunks
    audio_segments = []
    
    for i, text_chunk in enumerate(text_chunks):
        print(f"   Processing chunk {{i+1}}/{{len(text_chunks)}} ({{len(text_chunk)}} chars)...")
        
        chunk_start = time.time()
        
        # Create temporary file for this chunk
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            chunk_output = tmp_file.name
        
        try:
            # Synthesize this chunk
            tts.tts_to_file(
                text=text_chunk,
                speaker_wav=voice_sample,
                language="en",
                file_path=chunk_output,
                speed=speed
            )
            
            # Load and store audio data
            import torchaudio
            audio_data, sample_rate = torchaudio.load(chunk_output)
            audio_segments.append(audio_data)
            
            chunk_time = time.time() - chunk_start
            print(f"     Chunk {{i+1}} completed in {{chunk_time:.2f}} seconds")
            
        except Exception as e:
            print(f"[ERROR] Error processing chunk {{i+1}}: {{e}}")
            sys.exit(1)
        finally:
            # Clean up temporary file
            if os.path.exists(chunk_output):
                os.unlink(chunk_output)
    
    # Combine audio segments
    print("[EMOJI] Combining audio segments...")
    
    if audio_segments:
        # Concatenate all audio segments
        combined_audio = torch.cat(audio_segments, dim=1)
        
        # Save combined audio
        torchaudio.save(output_path, combined_audio, sample_rate)
        
        print(f"[SUCCESS] Combined audio saved to {{output_path}}")
        
        # Verify output
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            duration = combined_audio.shape[1] / sample_rate
            
            print(f"XTTS_SUCCESS: {{file_size}} bytes")
            print(f"AUDIO_DURATION: {{duration:.2f}} seconds")
            print(f"SAMPLE_RATE: {{sample_rate}} Hz")
            print(f"OUTPUT_FILE: {{output_path}}")
        else:
            print("[ERROR] XTTS_ERROR: No output file generated")
            sys.exit(1)
    else:
        print("[ERROR] XTTS_ERROR: No audio segments generated")
        sys.exit(1)
        
except Exception as e:
    import traceback
    print(f"[ERROR] XTTS_ERROR: {{str(e)}}")
    traceback.print_exc()
    sys.exit(1)
'''
        
        return script_content
    
    def run_xtts_voice_cloning(self, voice_sample_path: str, script_data: Dict[str, Any],
                             xtts_params: Dict[str, Any]) -> Optional[str]:
        """Run XTTS voice cloning process.
        
        Args:
            voice_sample_path: Path to user's voice sample
            script_data: Processed script data
            xtts_params: XTTS parameters
            
        Returns:
            Path to generated audio file, or None if failed
        """
        logger.info("Recording Starting Production XTTS voice cloning...")
        
        # Generate output filename
        timestamp = int(time.time())
        output_filename = f"xtts_voice_{timestamp}.wav"
        output_path = self.output_dir / output_filename
        
        # Create XTTS script
        text_chunks = script_data.get('text_chunks', [])
        xtts_script = self.create_xtts_script(
            voice_sample_path, text_chunks, str(output_path), xtts_params
        )
        
        try:
            # Run XTTS in conda environment
            cmd = [
                str(self.conda_python), '-c', xtts_script
            ]
            
            logger.info(f"Tools Running XTTS in conda environment: {self.conda_python}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            if result.returncode == 0 and 'XTTS_SUCCESS' in result.stdout:
                if output_path.exists():
                    file_size = output_path.stat().st_size
                    logger.info(f"[SUCCESS] Production XTTS voice cloning successful!")
                    logger.info(f"   Output: {output_path}")
                    logger.info(f"   File size: {file_size:,} bytes")
                    
                    # Extract additional info from output
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'AUDIO_DURATION:' in line:
                            logger.info(f"   Duration: {line.split('AUDIO_DURATION:')[1].strip()}")
                        elif 'SAMPLE_RATE:' in line:
                            logger.info(f"   Sample rate: {line.split('SAMPLE_RATE:')[1].strip()}")
                    
                    return str(output_path)
                else:
                    logger.error("[ERROR] Production XTTS voice cloning failed: No output file generated")
                    return None
            else:
                logger.error("[ERROR] Production XTTS voice cloning failed!")
                logger.error(f"   Return code: {result.returncode}")
                logger.error(f"   STDOUT: {result.stdout}")
                logger.error(f"   STDERR: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("[ERROR] Production XTTS voice cloning timeout (30 minutes)")
            return None
        except Exception as e:
            logger.error(f"[ERROR] Production XTTS voice cloning exception: {e}")
            return None
    
    def process_voice_cloning(self, voice_sample_path: str, script_data: Dict[str, Any],
                            xtts_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process voice cloning with Production XTTS.
        
        Args:
            voice_sample_path: Path to user's voice sample
            script_data: Processed script data
            xtts_params: XTTS parameters
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        
        result = {
            'stage': 'production_xtts_voice_cloning',
            'timestamp': time.time(),
            'success': False,
            'input_voice_sample': voice_sample_path,
            'input_script_data': script_data,
            'xtts_params': xtts_params or {},
            'output_audio_path': None,
            'processing_time': 0,
            'errors': []
        }
        
        try:
            # Set default parameters if not provided
            if xtts_params is None:
                xtts_params = {
                    'temperature': 0.7,
                    'length_penalty': 1.0,
                    'repetition_penalty': 1.1,
                    'top_k': 50,
                    'top_p': 0.8,
                    'speed': 1.0
                }
            
            # Validate inputs
            logger.info("Endpoints Validating Production XTTS inputs...")
            is_valid, errors = self.validate_inputs(voice_sample_path, script_data)
            if not is_valid:
                result['errors'] = errors
                logger.error(f"[ERROR] Input validation failed: {errors}")
                return result
            
            # Check Python environment
            logger.info("Search Checking Python environment...")
            if not self.check_python_environment():
                result['errors'].append(f"Python environment '{self.conda_python}' not available or TTS not installed")
                logger.error(f"[ERROR] Python environment check failed")
                return result
            
            # Run voice cloning
            output_path = self.run_xtts_voice_cloning(voice_sample_path, script_data, xtts_params)
            
            if output_path:
                result['success'] = True
                result['output_audio_path'] = output_path
                
                # Add file info
                output_file = Path(output_path)
                result['file_size'] = output_file.stat().st_size
                result['file_name'] = output_file.name
                
                logger.info("SUCCESS Production XTTS Stage completed successfully!")
            else:
                result['errors'].append("Production XTTS voice cloning failed")
                logger.error("[ERROR] Production XTTS Stage failed")
                
        except Exception as e:
            result['errors'].append(f"Unexpected error: {str(e)}")
            logger.error(f"[ERROR] Production XTTS Stage error: {e}")
            
        finally:
            result['processing_time'] = time.time() - start_time
            
            # Save stage results
            results_file = self.output_dir / f"production_xtts_results_{int(time.time())}.json"
            with open(results_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"File: Production XTTS stage results saved: {results_file}")
            logger.info(f"Duration:  Processing time: {result['processing_time']:.2f} seconds")
        
        return result


def main():
    """Test Production XTTS stage."""
    print("Target: Production XTTS Stage Test - Main Model Only")
    print("=" * 50)
    
    # Test with sample data
    stage = ProductionXTTSStage()
    
    # Sample script data (should be provided by user in production)
    script_data = {
        'clean_script': 'This is a test of production XTTS voice cloning functionality.',
        'text_chunks': ['This is a test of production XTTS voice cloning functionality.'],
        'tone': 'professional',  # USER CONFIGURABLE
        'emotion': 'inspired'     # USER CONFIGURABLE
    }
    
    # Sample voice path (USER CONFIGURABLE - should be provided by user)
    voice_sample = "/Users/aryanjain/Documents/video-synthesis-pipeline copy/MYC.wav"
    
    # Test parameters (USER CONFIGURABLE)
    xtts_params = {
        'temperature': 0.7,  # USER CONFIGURABLE
        'speed': 1.0         # USER CONFIGURABLE
    }
    
    # Run processing
    results = stage.process_voice_cloning(voice_sample, script_data, xtts_params)
    
    if results['success']:
        print("\nSUCCESS Production XTTS Stage test PASSED!")
        print(f"[SUCCESS] Output audio: {results['output_audio_path']}")
        print(f"Status: File size: {results.get('file_size', 0):,} bytes")
    else:
        print("\n[ERROR] Production XTTS Stage test FAILED!")
        print("Tools Errors:")
        for error in results['errors']:
            print(f"   - {error}")


if __name__ == "__main__":
    main()