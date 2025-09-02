#!/usr/bin/env python3
"""
Multi-Engine TTS Manager with Fallback Support
Implements robust voice cloning with multiple engine fallback strategy
"""

import os
import sys
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import time
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiEngineTTSManager:
    """
    Multi-engine TTS system with robust fallback support
    Based on AllTalk TTS patterns and Coqui TTS best practices
    """
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.engines = {}
        self.engine_priority = []
        self.current_engine = None
        self.fallback_count = 0
        self.max_fallback_attempts = 3
        
        # Engine configurations - prioritizing working environment as main
        self.engine_configs = {
            'xtts_current': {
                'name': 'XTTS Current (Main)',
                'venv_path': self.project_root / 'venv_stage0a_voice_synthesis' / 'bin' / 'python',
                'description': 'Main XTTS environment (PyTorch 2.5.1 + Transformers 4.49.0)',
                'priority': 1,
                'timeout': 300,  # Extended to 5 minutes for long scripts
                'supports_cloning': True
            },
            'xtts_advanced': {
                'name': 'XTTS Advanced (Fallback)',
                'venv_path': self.project_root / 'CHECK' / 'venv_voice_cloning_advanced' / 'bin' / 'python',
                'description': 'XTTS-v2 advanced environment (fallback)',
                'priority': 2,
                'timeout': 300,  # Extended to 5 minutes for long scripts
                'supports_cloning': True
            },
            'coqui_fresh': {
                'name': 'Coqui Fresh',
                'venv_path': None,  # Will be created
                'description': 'Fresh Coqui TTS installation (fallback)',
                'priority': 3,
                'timeout': 300,  # Extended to 5 minutes for long scripts
                'supports_cloning': True
            },
            'basic_tts_fallback': {
                'name': 'Basic TTS Fallback',
                'venv_path': None,  # System-wide
                'description': 'System basic TTS (macOS say command)',
                'priority': 3,
                'timeout': 120,  # Extended to 2 minutes for long scripts
                'supports_cloning': False
            },
            'tacotron2_emergency': {
                'name': 'Tacotron2 Emergency',
                'venv_path': None,  # System-wide
                'description': 'Emergency Tacotron2 engine (last resort)',
                'priority': 4,
                'timeout': 180,  # Extended to 3 minutes for long scripts
                'supports_cloning': False
            }
        }
        
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize and validate all TTS engines"""
        logger.info("Target: Initializing Multi-Engine TTS Manager")
        
        for engine_id, config in self.engine_configs.items():
            try:
                engine_status = self._validate_engine(engine_id, config)
                if engine_status['available']:
                    self.engines[engine_id] = {
                        'config': config,
                        'status': engine_status,
                        'last_used': None,
                        'failure_count': 0
                    }
                    self.engine_priority.append(engine_id)
                    logger.info(f"[SUCCESS] {config['name']} - Available")
                else:
                    logger.warning(f"[ERROR] {config['name']} - Not available: {engine_status.get('error', 'Unknown error')}")
            except Exception as e:
                logger.error(f"[ERROR] Failed to initialize {config['name']}: {e}")
        
        # Sort engines by priority
        self.engine_priority.sort(key=lambda x: self.engine_configs[x]['priority'])
        
        if self.engines:
            logger.info(f"STARTING {len(self.engines)} engines available: {', '.join(self.engine_priority)}")
            self.current_engine = self.engine_priority[0]
        else:
            logger.error("[WARNING] No TTS engines available!")
    
    def _validate_engine(self, engine_id: str, config: Dict) -> Dict:
        """Validate a specific TTS engine"""
        try:
            if engine_id == 'coqui_fresh':
                return self._validate_coqui_fresh()
            elif engine_id == 'basic_tts_fallback':
                return self._validate_basic_tts_engine()
            elif engine_id == 'tacotron2_emergency':
                return self._validate_tacotron2_emergency()
            else:
                return self._validate_xtts_engine(config)
        except Exception as e:
            return {'available': False, 'error': str(e)}
    
    def _validate_xtts_engine(self, config: Dict) -> Dict:
        """Validate XTTS engine in specific environment"""
        venv_path = config['venv_path']
        
        if not venv_path.exists():
            return {'available': False, 'error': f'Virtual environment not found: {venv_path}'}
        
        # Quick validation script
        validation_script = '''
import sys
import warnings
warnings.filterwarnings('ignore')

try:
    from TTS.api import TTS
    import torch
    import transformers
    
    print("VALIDATION_SUCCESS")
    print(f"PyTorch: {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    
except Exception as e:
    print(f"VALIDATION_ERROR: {e}")
    sys.exit(1)
'''
        
        try:
            result = subprocess.run(
                [str(venv_path), '-c', validation_script],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and 'VALIDATION_SUCCESS' in result.stdout:
                return {
                    'available': True,
                    'versions': self._extract_versions(result.stdout),
                    'validated_at': time.time()
                }
            else:
                return {'available': False, 'error': result.stderr or 'Validation failed'}
                
        except subprocess.TimeoutExpired:
            return {'available': False, 'error': 'Validation timeout'}
        except Exception as e:
            return {'available': False, 'error': str(e)}
    
    def _validate_coqui_fresh(self) -> Dict:
        """Validate or create fresh Coqui TTS installation"""
        try:
            # Check if fresh TTS is already available
            import TTS
            return {
                'available': True,
                'type': 'system',
                'version': TTS.__version__
            }
        except ImportError:
            # Need to install fresh TTS
            return {
                'available': False,
                'error': 'Fresh Coqui TTS installation needed',
                'install_required': True
            }
    
    def _validate_basic_tts_engine(self) -> Dict:
        """Validate basic TTS engine (system commands)"""
        try:
            import subprocess
            import sys
            
            if sys.platform == "darwin":  # macOS
                # Test say command
                result = subprocess.run(['say', '--version'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0 or 'say' in result.stderr:
                    return {
                        'available': True,
                        'type': 'system',
                        'platform': 'macOS',
                        'supports_cloning': False
                    }
            elif sys.platform.startswith("linux"):  # Linux
                # Test espeak
                result = subprocess.run(['espeak', '--version'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return {
                        'available': True,
                        'type': 'system',
                        'platform': 'Linux',
                        'supports_cloning': False
                    }
            
            return {'available': False, 'error': 'No system TTS available'}
            
        except Exception as e:
            return {'available': False, 'error': str(e)}
    
    def _validate_tacotron2_emergency(self) -> Dict:
        """Validate emergency Tacotron2 engine"""
        try:
            # Simple test for basic TTS availability
            import torch
            return {
                'available': True,
                'type': 'emergency',
                'supports_cloning': False
            }
        except ImportError:
            return {'available': False, 'error': 'PyTorch not available'}
    
    def _extract_versions(self, output: str) -> Dict:
        """Extract version information from validation output"""
        versions = {}
        for line in output.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                versions[key.strip()] = value.strip()
        return versions
    
    def clone_voice(self, reference_audio: str, text: str, language: str = "en", 
                   output_file: str = None, use_voice_focused: bool = True, **kwargs) -> Optional[str]:
        """
        Clone voice using the best available engine with fallback support
        Includes optimized voice_focused preprocessing by default
        """
        self.fallback_count = 0
        
        # Apply voice_focused optimization to reference audio if requested
        if use_voice_focused:
            try:
                reference_audio = self._apply_voice_focused_optimization(reference_audio)
                logger.info("Target: Applied voice_focused optimization to reference audio")
            except Exception as e:
                logger.warning(f"[WARNING] Voice optimization failed, using original: {e}")
        
        for attempt in range(self.max_fallback_attempts):
            if not self.current_engine:
                logger.error("[ERROR] No engines available for voice cloning")
                return None
            
            try:
                logger.info(f"Recording Attempting voice cloning with {self.engines[self.current_engine]['config']['name']} (attempt {attempt + 1})")
                
                result = self._clone_with_engine(
                    self.current_engine,
                    reference_audio,
                    text,
                    language,
                    output_file,
                    **kwargs
                )
                
                if result:
                    logger.info(f"[SUCCESS] Voice cloning successful with {self.engines[self.current_engine]['config']['name']}")
                    self.engines[self.current_engine]['last_used'] = time.time()
                    return result
                else:
                    logger.warning(f"[WARNING] Voice cloning failed with {self.engines[self.current_engine]['config']['name']}")
                    self._handle_engine_failure()
                    
            except Exception as e:
                logger.error(f"[ERROR] Exception in {self.engines[self.current_engine]['config']['name']}: {e}")
                self._handle_engine_failure()
                
        logger.error("[ERROR] All fallback attempts exhausted")
        return None
    
    def _clone_with_engine(self, engine_id: str, reference_audio: str, text: str, 
                          language: str, output_file: str = None, **kwargs) -> Optional[str]:
        """Clone voice using a specific engine"""
        config = self.engines[engine_id]['config']
        
        if engine_id in ['xtts_advanced', 'xtts_current']:
            return self._clone_with_xtts(config, reference_audio, text, language, output_file, **kwargs)
        elif engine_id == 'coqui_fresh':
            return self._clone_with_coqui_fresh(reference_audio, text, language, output_file, **kwargs)
        elif engine_id == 'basic_tts_fallback':
            return self._clone_with_basic_tts(text, output_file, **kwargs)
        elif engine_id == 'tacotron2_emergency':
            return self._clone_with_tacotron2_emergency(text, output_file, **kwargs)
        else:
            raise ValueError(f"Unknown engine: {engine_id}")
    
    def _clone_with_xtts(self, config: Dict, reference_audio: str, text: str, 
                        language: str, output_file: str = None, **kwargs) -> Optional[str]:
        """Clone voice using XTTS engine"""
        venv_path = config['venv_path']
        
        if not output_file:
            output_file = f"/tmp/xtts_output_{int(time.time())}.wav"
        
        # Fixed XTTS cloning script with proper text escaping
        # Escape text properly to avoid syntax errors
        escaped_text = text.replace('"', '\\"').replace("'", "\\'").replace('\n', ' ').replace('\r', ' ')
        
        cloning_script = f'''
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Environment setup
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['COQUI_TOS_AGREED'] = '1'

import torch

# Apply basic compatibility fixes for weights_only and GPT2 issues
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return original_load(*args, **kwargs)
torch.load = patched_load

# Fix for GPT2 generation compatibility issues
import transformers
if hasattr(transformers, 'GPT2LMHeadModel'):
    original_generate = transformers.GPT2LMHeadModel.generate
    def patched_generate(self, *args, **kwargs):
        # Ensure generate method is available
        if hasattr(self, '_generate'):
            return self._generate(*args, **kwargs)
        elif hasattr(super(transformers.GPT2LMHeadModel, self), 'generate'):
            return super(transformers.GPT2LMHeadModel, self).generate(*args, **kwargs)
        else:
            # Fallback to base generation
            from transformers.generation.utils import GenerationMixin
            return GenerationMixin.generate(self, *args, **kwargs)
    transformers.GPT2LMHeadModel.generate = patched_generate

try:
    from pathlib import Path
    from TTS.api import TTS
    
    # Initialize TTS with error handling
    print("Loading XTTS-v2 model...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    print("[SUCCESS] XTTS model loaded successfully")
    
    # Prepare text for synthesis
    synthesis_text = "{escaped_text}"
    
    # Perform voice cloning with robust error handling
    print("Performing voice cloning...")
    tts.tts_to_file(
        text=synthesis_text,
        speaker_wav="{reference_audio}",
        language="{language}",
        file_path="{output_file}"
    )
    
    # Verify output
    if Path("{output_file}").exists():
        file_size = Path("{output_file}").stat().st_size
        print(f"CLONING_SUCCESS: {{file_size}} bytes")
    else:
        print("CLONING_ERROR: No output file generated")
        sys.exit(1)
        
except Exception as e:
    import traceback
    print(f"CLONING_ERROR: {{str(e)}}")
    traceback.print_exc()
    sys.exit(1)
'''
        
        try:
            result = subprocess.run(
                [str(venv_path), '-c', cloning_script],
                capture_output=True,
                text=True,
                timeout=config['timeout']
            )
            
            if result.returncode == 0 and 'CLONING_SUCCESS' in result.stdout:
                if Path(output_file).exists():
                    logger.info(f"[SUCCESS] XTTS cloning successful: {output_file}")
                    return output_file
                else:
                    logger.error("[ERROR] XTTS cloning failed: No output file")
                    return None
            else:
                logger.error(f"[ERROR] XTTS cloning failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error(f"[ERROR] XTTS cloning timeout after {config['timeout']} seconds")
            return None
        except Exception as e:
            logger.error(f"[ERROR] XTTS cloning exception: {e}")
            return None
    
    def _clone_with_coqui_fresh(self, reference_audio: str, text: str, 
                               language: str, output_file: str = None, **kwargs) -> Optional[str]:
        """Clone voice using fresh Coqui TTS installation"""
        try:
            from TTS.api import TTS
            
            if not output_file:
                output_file = f"/tmp/coqui_output_{int(time.time())}.wav"
            
            # Initialize fresh TTS
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
            
            # Perform cloning
            tts.tts_to_file(
                text=text,
                speaker_wav=reference_audio,
                language=language,
                file_path=output_file
            )
            
            if Path(output_file).exists():
                logger.info(f"[SUCCESS] Coqui Fresh cloning successful: {output_file}")
                return output_file
            else:
                logger.error("[ERROR] Coqui Fresh cloning failed: No output file")
                return None
                
        except Exception as e:
            logger.error(f"[ERROR] Coqui Fresh cloning failed: {e}")
            return None
    
    def _clone_with_tacotron2_emergency(self, text: str, output_file: str = None, **kwargs) -> Optional[str]:
        """Emergency TTS using Tacotron2 (no voice cloning)"""
        try:
            from TTS.api import TTS
            
            if not output_file:
                output_file = f"/tmp/tacotron2_output_{int(time.time())}.wav"
            
            # Emergency TTS without cloning
            tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=False)
            tts.tts_to_file(text=text, file_path=output_file)
            
            if Path(output_file).exists():
                logger.info(f"[SUCCESS] Emergency TTS successful: {output_file}")
                return output_file
            else:
                logger.error("[ERROR] Emergency TTS failed: No output file")
                return None
                
        except Exception as e:
            logger.error(f"[ERROR] Emergency TTS failed: {e}")
            return None
    
    def _clone_with_basic_tts(self, text: str, output_file: str = None, **kwargs) -> Optional[str]:
        """Basic TTS using system commands (macOS say command)"""
        try:
            import subprocess
            import tempfile
            
            if not output_file:
                output_file = f"/tmp/basic_tts_output_{int(time.time())}.wav"
            
            # Split long text into chunks for better processing
            max_length = 2000  # Reasonable limit for system TTS
            if len(text) > max_length:
                # Split into sentences and process in chunks
                sentences = text.split('. ')
                chunks = []
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk + sentence) < max_length:
                        current_chunk += sentence + ". "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + ". "
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Generate audio for each chunk and combine
                chunk_files = []
                for i, chunk in enumerate(chunks):
                    chunk_file = f"/tmp/chunk_{i}_{int(time.time())}.wav"
                    if self._generate_basic_audio(chunk, chunk_file):
                        chunk_files.append(chunk_file)
                
                # Combine chunk files
                if chunk_files:
                    self._combine_audio_files(chunk_files, output_file)
                    
                    # Cleanup chunk files
                    for chunk_file in chunk_files:
                        try:
                            Path(chunk_file).unlink()
                        except:
                            pass
                else:
                    return None
            else:
                # Generate single audio file
                if not self._generate_basic_audio(text, output_file):
                    return None
            
            if Path(output_file).exists():
                logger.info(f"[SUCCESS] Basic TTS successful: {output_file}")
                return output_file
            else:
                logger.error("[ERROR] Basic TTS failed: No output file")
                return None
                
        except Exception as e:
            logger.error(f"[ERROR] Basic TTS failed: {e}")
            return None
    
    def _generate_basic_audio(self, text: str, output_file: str) -> bool:
        """Generate audio using system TTS command"""
        try:
            import subprocess
            import sys
            
            # Clean text for TTS
            clean_text = text.replace('"', '').replace("'", "").strip()
            
            if sys.platform == "darwin":  # macOS
                # Use macOS say command with better voice
                cmd = ["say", "-v", "Alex", "-o", output_file, "--data-format=LEF32@22050", clean_text]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0 and Path(output_file).exists():
                    # Convert to wav format if needed
                    if not output_file.endswith('.wav'):
                        wav_output = output_file.replace('.aiff', '.wav')
                        subprocess.run(['ffmpeg', '-i', output_file, '-y', wav_output], 
                                     capture_output=True)
                        Path(output_file).unlink()
                        output_file = wav_output
                    return True
                else:
                    logger.error(f"say command failed: {result.stderr}")
                    return False
                    
            elif sys.platform.startswith("linux"):  # Linux
                # Use espeak or festival
                cmd = ["espeak", "-w", output_file, clean_text]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                return result.returncode == 0 and Path(output_file).exists()
                
            else:  # Windows or other
                logger.warning("Basic TTS not implemented for this platform")
                return False
                
        except Exception as e:
            logger.error(f"Basic audio generation failed: {e}")
            return False
    
    def _combine_audio_files(self, input_files: List[str], output_file: str):
        """Combine multiple audio files into one"""
        try:
            import subprocess
            
            # Create file list for ffmpeg
            file_list = "/tmp/audio_files.txt"
            with open(file_list, 'w') as f:
                for file_path in input_files:
                    f.write(f"file '{file_path}'\n")
            
            # Use ffmpeg to concatenate
            cmd = [
                'ffmpeg', '-f', 'concat', '-safe', '0', 
                '-i', file_list, '-c', 'copy', '-y', output_file
            ]
            
            subprocess.run(cmd, capture_output=True, timeout=60)
            
            # Cleanup
            Path(file_list).unlink()
            
        except Exception as e:
            logger.error(f"Audio combination failed: {e}")
    
    def _handle_engine_failure(self):
        """Handle engine failure and switch to next available engine"""
        if self.current_engine:
            self.engines[self.current_engine]['failure_count'] += 1
            logger.warning(f"[WARNING] Engine failure: {self.engines[self.current_engine]['config']['name']}")
        
        # Find next available engine
        current_index = self.engine_priority.index(self.current_engine) if self.current_engine in self.engine_priority else -1
        next_engines = self.engine_priority[current_index + 1:]
        
        for engine_id in next_engines:
            if engine_id in self.engines:
                self.current_engine = engine_id
                self.fallback_count += 1
                logger.info(f"[EMOJI] Switching to fallback engine: {self.engines[engine_id]['config']['name']}")
                return
        
        # No more engines available
        self.current_engine = None
        logger.error("[ERROR] No more engines available for fallback")
    
    def _apply_voice_focused_optimization(self, audio_path: str) -> str:
        """Apply voice_focused optimization to reference audio"""
        try:
            import numpy as np
            import soundfile as sf
            import librosa
            from scipy.signal import butter, filtfilt
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            
            # Resample to 22050 Hz
            if sr != 22050:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
                sr = 22050
            
            # High-pass filter to remove low-frequency noise
            nyquist = sr // 2
            high_cutoff = 80 / nyquist
            b, a = butter(4, high_cutoff, btype='high')
            audio_filtered = filtfilt(b, a, audio)
            
            # Low-pass filter at 8000 Hz for voice
            low_cutoff = 8000 / nyquist
            b, a = butter(4, low_cutoff, btype='low')
            audio_filtered = filtfilt(b, a, audio_filtered)
            
            # Dynamic range compression
            threshold = 0.4
            ratio = 3.0
            
            audio_abs = np.abs(audio_filtered)
            mask = audio_abs > threshold
            
            compressed = np.copy(audio_filtered)
            compressed[mask] = np.sign(audio_filtered[mask]) * (
                threshold + (audio_abs[mask] - threshold) / ratio
            )
            
            # Normalize to target RMS
            target_rms = 0.2
            current_rms = np.sqrt(np.mean(compressed**2))
            if current_rms > 0:
                compressed = compressed * (target_rms / current_rms)
            
            # Clip to prevent distortion
            audio_final = np.clip(compressed, -0.95, 0.95)
            
            # Save optimized audio
            output_path = f"/tmp/voice_focused_{int(time.time())}_{os.getpid()}.wav"
            sf.write(output_path, audio_final, 22050)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Voice_focused optimization failed: {e}")
            return audio_path  # Return original if optimization fails
    
    def get_status(self) -> Dict:
        """Get current status of all engines"""
        return {
            'current_engine': self.current_engine,
            'available_engines': len(self.engines),
            'fallback_count': self.fallback_count,
            'engines': {
                engine_id: {
                    'name': info['config']['name'],
                    'available': True,
                    'failure_count': info['failure_count'],
                    'last_used': info['last_used']
                }
                for engine_id, info in self.engines.items()
            }
        }
    
    def install_coqui_fresh(self) -> bool:
        """Install fresh Coqui TTS if needed"""
        try:
            logger.info("Package Installing fresh Coqui TTS...")
            
            # Create fresh environment
            fresh_venv = self.project_root / "venv_coqui_fresh"
            if not fresh_venv.exists():
                subprocess.run([
                    sys.executable, "-m", "venv", str(fresh_venv)
                ], check=True)
            
            # Install Coqui TTS
            pip_path = fresh_venv / "bin" / "pip"
            subprocess.run([
                str(pip_path), "install", "--upgrade", "pip"
            ], check=True)
            
            subprocess.run([
                str(pip_path), "install", "TTS"
            ], check=True)
            
            # Update configuration
            self.engine_configs['coqui_fresh']['venv_path'] = fresh_venv / "bin" / "python"
            
            logger.info("[SUCCESS] Fresh Coqui TTS installed successfully")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to install fresh Coqui TTS: {e}")
            return False

# Convenience functions
def create_multi_engine_tts(project_root: str = None) -> MultiEngineTTSManager:
    """Create and initialize multi-engine TTS manager"""
    return MultiEngineTTSManager(project_root)

def test_voice_cloning(reference_audio: str, text: str, language: str = "en") -> Optional[str]:
    """Test voice cloning with multi-engine fallback"""
    manager = create_multi_engine_tts()
    return manager.clone_voice(reference_audio, text, language)

if __name__ == "__main__":
    # Test the multi-engine TTS manager
    print("🧪 Testing Multi-Engine TTS Manager")
    
    manager = create_multi_engine_tts()
    print(f"Status: {manager.get_status()}")
    
    # Create test reference audio
    import numpy as np
    import soundfile as sf
    
    sr = 22050
    test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 3, 3 * sr)) * 0.3
    test_ref = "/tmp/test_multi_engine_ref.wav"
    sf.write(test_ref, test_audio, sr)
    
    # Test cloning
    result = manager.clone_voice(
        test_ref,
        "Testing multi-engine voice cloning with fallback support.",
        "en"
    )
    
    if result:
        print(f"[SUCCESS] Multi-engine test successful: {result}")
    else:
        print("[ERROR] Multi-engine test failed")
    
    # Cleanup
    if Path(test_ref).exists():
        os.remove(test_ref)