#!/usr/bin/env python3
"""
Stage 0A: Voice Synthesis Pipeline
Handles voice cloning and text-to-speech synthesis using XTTS-v2.
Input: User voice sample + script text
Output: High-quality synthesized audio in user's cloned voice
"""

import argparse
import logging
import sys
import json
import time
import os
from pathlib import Path
import numpy as np
import torch
import torchaudio
import soundfile as sf
from typing import Tuple, Dict, Any, Optional, List
import tempfile
import warnings

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

# Import project utilities
from utils.logger import setup_logger, TimedLogger
from utils.file_utils import ensure_directory, save_json, load_json
from utils.xtts_compatibility_fix import apply_xtts_compatibility_fixes, create_xtts_wrapper

# Suppress TTS warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class VoiceCloningPipeline:
    """Advanced voice cloning pipeline using XTTS-v2."""
    
    def __init__(self, config_path: Optional[str] = None, device: str = "auto"):
        """Initialize the voice cloning pipeline."""
        self.logger = setup_logger("stage0a_voice_synthesis")
        
        # Load configuration
        if config_path and Path(config_path).exists():
            self.config = load_json(config_path)
        else:
            self.config = self._get_default_config()
        
        # Device selection
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize TTS model
        self.tts_model = None
        self.speaker_embedding = None
        self.temp_dir = Path(tempfile.gettempdir()) / "voice_cloning"
        self.temp_dir.mkdir(exist_ok=True)
        
        self.logger.info("Voice cloning pipeline initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for voice cloning."""
        return {
            "tts_model": {
                "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
                "use_cuda": torch.cuda.is_available(),
                "language": "en",
                "speed": 1.0,
                "temperature": 0.7,
                "length_penalty": 1.0,
                "repetition_penalty": 1.1,
                "top_k": 50,
                "top_p": 0.85
            },
            "voice_cloning": {
                "min_audio_length": 5.0,  # Minimum seconds of reference audio
                "max_audio_length": 30.0,  # Maximum seconds for processing
                "target_sample_rate": 22050,
                "normalize_audio": True,
                "remove_silence": True,
                "speaker_embedding_dim": 512
            },
            "audio_processing": {
                "sample_rate": 22050,
                "bit_depth": 16,
                "channels": 1,
                "format": "wav",
                "noise_reduction": True,
                "voice_activity_detection": True
            },
            "synthesis": {
                "chunk_size": 250,  # Characters per chunk for long texts
                "overlap_size": 25,  # Character overlap between chunks
                "add_silence": 0.5,  # Seconds of silence between sentences
                "emotion_strength": 0.8,
                "speaking_rate": 1.0
            },
            "quality": {
                "enable_enhancement": True,
                "vocal_isolation": True,
                "noise_gate_threshold": -40,  # dB
                "dynamic_range_compression": True
            }
        }
    
    def _load_tts_model(self):
        """Load XTTS-v2 model for voice cloning with enhanced compatibility fixes."""
        if self.tts_model is not None:
            return
        
        try:
            with TimedLogger(self.logger, "loading XTTS-v2 model"):
                self.logger.info("Loading XTTS-v2 model with enhanced compatibility fixes...")
                
                # Apply basic compatibility fixes first
                apply_xtts_compatibility_fixes()
                
                # Apply enhanced GPT2InferenceModel fixes
                self._apply_enhanced_gpt2_fixes()
                
                # Use the compatibility wrapper
                XTTSWrapper = create_xtts_wrapper()
                model_name = self.config["tts_model"]["model_name"]
                
                self.tts_model = XTTSWrapper(
                    model_name=model_name,
                    gpu=self.device == "cuda"
                )
                
                if self.tts_model.available:
                    self.logger.info(f"[SUCCESS] XTTS-v2 model loaded with enhanced compatibility: {model_name}")
                else:
                    raise RuntimeError("XTTS model failed to initialize")
                
        except ImportError:
            self.logger.error("TTS library not found. Please install Coqui TTS: pip install TTS")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load TTS model: {str(e)}")
            raise
    
    def _apply_enhanced_gpt2_fixes(self):
        """Apply enhanced GPT2InferenceModel compatibility fixes."""
        try:
            from transformers import GenerationMixin, GPT2PreTrainedModel
            from TTS.tts.layers.xtts.gpt_inference import GPT2InferenceModel
            
            # Check if GPT2InferenceModel already has generate method
            if not hasattr(GPT2InferenceModel, 'generate'):
                self.logger.info("Tools Applying enhanced GPT2InferenceModel fixes...")
                
                # Essential generation methods to add
                generation_methods = [
                    'generate', 'sample', 'greedy_search', 'beam_search', 'beam_sample',
                    '_get_logits_warper', '_get_logits_processor', '_get_stopping_criteria',
                    '_prepare_model_inputs', '_prepare_attention_mask_for_generation',
                    '_expand_inputs_for_generation', '_extract_past_from_model_output',
                    '_update_model_kwargs_for_generation', '_reorder_cache'
                ]
                
                # Copy methods from GenerationMixin to GPT2InferenceModel
                methods_added = 0
                for method_name in generation_methods:
                    if hasattr(GenerationMixin, method_name):
                        method = getattr(GenerationMixin, method_name)
                        if callable(method):
                            setattr(GPT2InferenceModel, method_name, method)
                            methods_added += 1
                
                # Add necessary attributes
                if not hasattr(GPT2InferenceModel, '_supports_cache_class'):
                    GPT2InferenceModel._supports_cache_class = False
                
                if not hasattr(GPT2InferenceModel, 'can_generate'):
                    GPT2InferenceModel.can_generate = lambda self: True
                
                self.logger.info(f"[SUCCESS] Enhanced GPT2InferenceModel with {methods_added} generation methods")
            else:
                self.logger.info("[SUCCESS] GPT2InferenceModel already has generate method")
                
        except ImportError as e:
            self.logger.warning(f"[WARNING] Could not apply enhanced GPT2 fixes: {e}")
            self.logger.warning("[WARNING] Will rely on fallback mechanism")
        except Exception as e:
            self.logger.warning(f"[WARNING] Error applying enhanced GPT2 fixes: {e}")
    
    def preprocess_reference_audio(self, audio_path: str) -> str:
        """Preprocess reference audio for optimal voice cloning."""
        self.logger.info(f"Preprocessing reference audio: {Path(audio_path).name}")
        
        try:
            # Load audio
            audio, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # Resample to target sample rate
            target_sr = self.config["voice_cloning"]["target_sample_rate"]
            if sample_rate != target_sr:
                resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
                audio = resampler(audio)
                sample_rate = target_sr
            
            # Normalize audio if configured
            if self.config["voice_cloning"]["normalize_audio"]:
                audio = audio / torch.max(torch.abs(audio))
            
            # Check duration
            duration = audio.shape[1] / sample_rate
            min_duration = self.config["voice_cloning"]["min_audio_length"]
            max_duration = self.config["voice_cloning"]["max_audio_length"]
            
            if duration < min_duration:
                self.logger.warning(f"Reference audio too short ({duration:.1f}s). Minimum: {min_duration}s")
            elif duration > max_duration:
                self.logger.info(f"Trimming reference audio from {duration:.1f}s to {max_duration}s")
                max_samples = int(max_duration * sample_rate)
                audio = audio[:, :max_samples]
            
            # Remove silence if enabled
            if self.config["voice_cloning"]["remove_silence"]:
                audio = self._remove_silence(audio, sample_rate)
            
            # Apply noise reduction if enabled
            if self.config["quality"]["noise_reduction"]:
                audio = self._apply_noise_reduction(audio)
            
            # Save processed audio
            processed_path = self.temp_dir / f"processed_reference_{int(time.time())}.wav"
            torchaudio.save(str(processed_path), audio, sample_rate)
            
            final_duration = audio.shape[1] / sample_rate
            self.logger.info(f"Reference audio processed: {final_duration:.1f}s at {sample_rate}Hz")
            
            return str(processed_path)
            
        except Exception as e:
            self.logger.error(f"Failed to preprocess reference audio: {str(e)}")
            raise
    
    def _remove_silence(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Remove silence from audio using voice activity detection."""
        try:
            # Simple energy-based VAD
            frame_length = int(0.025 * sample_rate)  # 25ms frames
            hop_length = int(0.010 * sample_rate)    # 10ms hop
            
            # Calculate energy per frame
            audio_sq = audio.squeeze()
            frames = audio_sq.unfold(0, frame_length, hop_length)
            energy = torch.mean(frames ** 2, dim=1)
            
            # Threshold for voice activity
            threshold = torch.quantile(energy, 0.2)  # Bottom 20% is likely silence
            voice_frames = energy > threshold
            
            # Expand frame decisions back to samples
            voice_samples = torch.zeros_like(audio_sq, dtype=torch.bool)
            for i, is_voice in enumerate(voice_frames):
                start = i * hop_length
                end = min(start + frame_length, len(voice_samples))
                voice_samples[start:end] = is_voice
            
            # Keep voiced sections
            voiced_audio = audio_sq[voice_samples]
            
            if len(voiced_audio) > 0:
                return voiced_audio.unsqueeze(0)
            else:
                return audio  # Return original if no voice detected
                
        except Exception as e:
            self.logger.warning(f"Silence removal failed: {str(e)}, using original audio")
            return audio
    
    def _apply_noise_reduction(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply basic noise reduction."""
        try:
            # Simple spectral gating
            # This is a basic implementation - for production, consider using noisereduce library
            
            # Apply high-pass filter to remove low-frequency noise
            try:
                from torchaudio.transforms import HighpassFilter
                highpass = HighpassFilter(sample_rate=self.config["voice_cloning"]["target_sample_rate"], 
                                        cutoff_freq=80.0)
                audio = highpass(audio)
            except ImportError:
                # Fallback: Simple high-pass using scipy if available
                try:
                    from scipy.signal import butter, sosfilt
                    sos = butter(4, 80.0, btype='high', fs=self.config["voice_cloning"]["target_sample_rate"], output='sos')
                    audio_np = audio.cpu().numpy() if hasattr(audio, 'cpu') else audio
                    filtered = sosfilt(sos, audio_np)
                    audio = torch.from_numpy(filtered).float() if hasattr(audio, 'cpu') else filtered
                except ImportError:
                    # No filtering available, skip
                    pass
            
            return audio
            
        except Exception as e:
            self.logger.warning(f"Noise reduction failed: {str(e)}, using original audio")
            return audio
    
    def extract_speaker_embedding(self, reference_audio_path: str) -> np.ndarray:
        """Extract speaker embedding from reference audio."""
        self.logger.info("Extracting speaker embedding from reference audio")
        
        try:
            # Preprocess reference audio
            processed_audio_path = self.preprocess_reference_audio(reference_audio_path)
            
            # Load TTS model if not already loaded
            self._load_tts_model()
            
            # Extract speaker embedding using XTTS
            with TimedLogger(self.logger, "speaker embedding extraction"):
                # Use TTS model's get_conditioning_latents method
                gpt_cond_latent, speaker_embedding = self.tts_model.synthesizer.tts_model.get_conditioning_latents(
                    audio_path=processed_audio_path
                )
                self.speaker_embedding = speaker_embedding
            
            embedding_shape = self.speaker_embedding.shape
            self.logger.info(f"Speaker embedding extracted: shape {embedding_shape}")
            
            return self.speaker_embedding.cpu().numpy()
            
        except Exception as e:
            self.logger.error(f"Failed to extract speaker embedding: {str(e)}")
            raise
    
    def synthesize_speech(self, text: str, output_path: str, 
                         reference_audio_path: Optional[str] = None) -> bool:
        """Synthesize speech using cloned voice."""
        self.logger.info(f"Synthesizing speech: {len(text)} characters")
        
        try:
            # Load TTS model
            self._load_tts_model()
            
            # Use reference audio for cloning if provided
            if reference_audio_path:
                reference_audio_path = self.preprocess_reference_audio(reference_audio_path)
            
            # Split long text into chunks
            text_chunks = self._split_text_into_chunks(text)
            self.logger.info(f"Split text into {len(text_chunks)} chunks")
            
            audio_segments = []
            
            for i, chunk in enumerate(text_chunks):
                self.logger.info(f"Synthesizing chunk {i+1}/{len(text_chunks)}")
                
                with TimedLogger(self.logger, f"synthesis chunk {i+1}"):
                    # Synthesize audio for this chunk
                    if reference_audio_path:
                        # Voice cloning mode - use conditioning latents
                        gpt_cond_latent, speaker_embedding = self.tts_model.synthesizer.tts_model.get_conditioning_latents(
                            audio_path=reference_audio_path
                        )
                        result = self.tts_model.synthesizer.tts_model.inference(
                            text=chunk,
                            language=self.config["tts_model"]["language"],
                            gpt_cond_latent=gpt_cond_latent,
                            speaker_embedding=speaker_embedding,
                            temperature=self.config["tts_model"]["temperature"],
                            length_penalty=self.config["tts_model"]["length_penalty"],
                            repetition_penalty=self.config["tts_model"]["repetition_penalty"],
                            top_k=self.config["tts_model"]["top_k"],
                            top_p=self.config["tts_model"]["top_p"],
                            speed=self.config["tts_model"]["speed"]
                        )
                        # Extract audio from result
                        if isinstance(result, dict):
                            audio_chunk = result.get("wav", result.get("audio", result))
                        elif isinstance(result, tuple):
                            audio_chunk = result[0]
                        else:
                            audio_chunk = result
                    else:
                        # Standard TTS mode
                        audio_chunk = self.tts_model.tts(
                            text=chunk,
                            language=self.config["tts_model"]["language"],
                            speed=self.config["tts_model"]["speed"]
                        )
                    
                    audio_segments.append(audio_chunk)
            
            # Concatenate audio segments
            with TimedLogger(self.logger, "audio concatenation"):
                final_audio = self._concatenate_audio_segments(audio_segments)
            
            # Apply post-processing
            if self.config["quality"]["enable_enhancement"]:
                final_audio = self._enhance_synthesized_audio(final_audio)
            
            # Save final audio
            sample_rate = self.config["audio_processing"]["sample_rate"]
            sf.write(output_path, final_audio, sample_rate)
            
            duration = len(final_audio) / sample_rate
            self.logger.info(f"Speech synthesis completed: {duration:.2f}s audio saved to {output_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Speech synthesis failed: {str(e)}")
            return False
    
    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Split long text into manageable chunks for synthesis."""
        chunk_size = self.config["synthesis"]["chunk_size"]
        overlap_size = self.config["synthesis"]["overlap_size"]
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Find a good break point (sentence end, period, etc.)
            if end < len(text):
                # Look for sentence boundaries
                for break_char in ['. ', '! ', '? ', '\n', '; ']:
                    break_pos = text.rfind(break_char, start, end)
                    if break_pos > start + chunk_size // 2:  # Don't break too early
                        end = break_pos + len(break_char)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap_size if end < len(text) else end
        
        return chunks
    
    def _concatenate_audio_segments(self, audio_segments: List[np.ndarray]) -> np.ndarray:
        """Concatenate audio segments with proper spacing."""
        if not audio_segments:
            return np.array([])
        
        if len(audio_segments) == 1:
            return audio_segments[0]
        
        # Add silence between segments
        silence_duration = self.config["synthesis"]["add_silence"]
        sample_rate = self.config["audio_processing"]["sample_rate"]
        silence_samples = int(silence_duration * sample_rate)
        silence = np.zeros(silence_samples)
        
        # Concatenate with silence
        result = []
        for i, segment in enumerate(audio_segments):
            result.append(segment)
            if i < len(audio_segments) - 1:  # Don't add silence after last segment
                result.append(silence)
        
        return np.concatenate(result)
    
    def _enhance_synthesized_audio(self, audio) -> np.ndarray:
        """Apply post-processing enhancement to synthesized audio."""
        try:
            # Convert to numpy array if needed
            if isinstance(audio, dict):
                self.logger.warning(f"Audio enhancement received dict, skipping enhancement")
                return audio
            elif hasattr(audio, 'cpu'):
                audio = audio.cpu().numpy()
            elif not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            
            # Apply dynamic range compression
            if self.config["quality"]["dynamic_range_compression"]:
                audio = self._apply_compression(audio)
            
            # Apply noise gate
            noise_threshold = self.config["quality"]["noise_gate_threshold"]
            audio = self._apply_noise_gate(audio, noise_threshold)
            
            return audio
            
        except Exception as e:
            self.logger.warning(f"Audio enhancement failed: {str(e)}, using original audio")
            return audio
    
    def _apply_compression(self, audio: np.ndarray) -> np.ndarray:
        """Apply dynamic range compression."""
        # Simple compression: reduce dynamic range
        ratio = 4.0  # 4:1 compression ratio
        threshold = 0.1  # Compression threshold
        
        compressed = np.copy(audio)
        mask = np.abs(compressed) > threshold
        compressed[mask] = np.sign(compressed[mask]) * (
            threshold + (np.abs(compressed[mask]) - threshold) / ratio
        )
        
        return compressed
    
    def _apply_noise_gate(self, audio: np.ndarray, threshold_db: float) -> np.ndarray:
        """Apply noise gate to remove low-level noise."""
        threshold_linear = 10 ** (threshold_db / 20)
        
        # Simple gate: zero out samples below threshold
        gated_audio = np.where(np.abs(audio) > threshold_linear, audio, 0)
        
        return gated_audio
    
    def save_outputs(self, output_dir: str, synthesized_audio_path: str,
                    reference_audio_path: str, text: str, 
                    speaker_embedding: Optional[np.ndarray] = None) -> Dict[str, str]:
        """Save processing outputs and metadata."""
        output_path = ensure_directory(output_dir)
        
        # Save metadata
        metadata = {
            "input_text": text,
            "input_text_length": len(text),
            "reference_audio_path": str(reference_audio_path),
            "synthesized_audio_path": str(synthesized_audio_path),
            "processing_timestamp": time.time(),
            "device_used": self.device,
            "config": self.config
        }
        
        if speaker_embedding is not None:
            embedding_path = output_path / "speaker_embedding.npy"
            np.save(embedding_path, speaker_embedding)
            metadata["speaker_embedding_path"] = str(embedding_path)
            metadata["speaker_embedding_shape"] = list(speaker_embedding.shape)
        
        # Calculate audio statistics
        try:
            audio, sr = sf.read(synthesized_audio_path)
            duration = len(audio) / sr
            metadata["output_duration"] = duration
            metadata["output_sample_rate"] = sr
            metadata["output_channels"] = 1 if audio.ndim == 1 else audio.shape[1]
        except Exception as e:
            self.logger.warning(f"Could not analyze output audio: {str(e)}")
        
        metadata_path = output_path / "voice_synthesis_metadata.json"
        save_json(metadata, metadata_path)
        
        output_files = {
            "synthesized_audio": str(synthesized_audio_path),
            "metadata": str(metadata_path)
        }
        
        if speaker_embedding is not None:
            output_files["speaker_embedding"] = str(embedding_path)
        
        self.logger.info(f"Voice synthesis outputs saved to: {output_path}")
        for key, path in output_files.items():
            if Path(path).exists():
                file_size = Path(path).stat().st_size
                self.logger.info(f"  {key}: {Path(path).name} ({file_size:,} bytes)")
        
        return output_files
    
    def process_voice_cloning(self, reference_audio_path: str, text: str, 
                            output_dir: str) -> bool:
        """Main voice cloning processing function."""
        try:
            self.logger.info("="*60)
            self.logger.info("Starting Voice Cloning & Speech Synthesis Pipeline")
            self.logger.info("="*60)
            
            start_time = time.time()
            output_path = ensure_directory(output_dir)
            
            # Extract speaker embedding
            with TimedLogger(self.logger, "speaker embedding extraction"):
                speaker_embedding = self.extract_speaker_embedding(reference_audio_path)
            
            # Synthesize speech
            output_audio_path = output_path / "synthesized_speech.wav"
            with TimedLogger(self.logger, "speech synthesis"):
                success = self.synthesize_speech(
                    text, 
                    str(output_audio_path), 
                    reference_audio_path
                )
            
            if not success:
                return False
            
            # Save outputs and metadata
            self.save_outputs(
                output_dir, 
                str(output_audio_path), 
                reference_audio_path, 
                text, 
                speaker_embedding
            )
            
            # Final summary
            total_time = time.time() - start_time
            self.logger.info("="*60)
            self.logger.info("Voice Cloning Pipeline Completed Successfully")
            self.logger.info(f"Total processing time: {total_time:.2f}s")
            self.logger.info(f"Output audio: {output_audio_path}")
            self.logger.info("="*60)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Voice cloning processing failed: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Stage 0A: Voice Synthesis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clone voice and synthesize speech
  python stage0a_voice_synthesis.py --reference-audio voice_sample.wav --text "Hello world!" --output-dir outputs/

  # Use custom configuration
  python stage0a_voice_synthesis.py --reference-audio voice.wav --text "Script text" --config voice_config.json
        """
    )
    
    parser.add_argument(
        "--reference-audio",
        type=str,
        required=True,
        help="Path to reference audio for voice cloning (5-30 seconds recommended)"
    )
    
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to synthesize in the cloned voice"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "stage0a"),
        help="Output directory for synthesized audio and metadata"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "voice_synthesis_config.json"),
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="Computing device"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input files
    if not Path(args.reference_audio).exists():
        print(f"Error: Reference audio file not found: {args.reference_audio}")
        return 1
    
    if not args.text.strip():
        print("Error: Text cannot be empty")
        return 1
    
    try:
        # Create voice cloning pipeline
        pipeline = VoiceCloningPipeline(
            config_path=args.config if Path(args.config).exists() else None,
            device=args.device
        )
        
        # Process voice cloning
        success = pipeline.process_voice_cloning(
            args.reference_audio,
            args.text,
            args.output_dir
        )
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"Failed to create voice cloning pipeline: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())