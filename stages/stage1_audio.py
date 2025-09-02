#!/usr/bin/env python3
"""
Stage 1: Audio Processing
Extracts transcription and rich audio embeddings from input audio.
"""

import argparse
import logging
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
import whisper
import librosa
import soundfile as sf
from typing import Tuple, Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

# Import only the specific utils we need (avoid cv2 dependencies)
from utils.logger import setup_logger, TimedLogger
from utils.file_utils import ensure_directory, save_json

class AudioProcessor:
    """Audio processing pipeline for transcription and feature extraction."""
    
    def __init__(self, config_path: Optional[str] = None, model_size: str = "base"):
        """Initialize the audio processor."""
        self.model_size = model_size
        self.sample_rate = 16000
        self.logger = setup_logger("stage1_audio")
        
        # Load configuration
        if config_path and Path(config_path).exists():
            from utils.file_utils import load_json
            self.config = load_json(config_path)
        else:
            self.config = self._get_default_config()
        
        # Device selection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize Whisper model
        with TimedLogger(self.logger, f"loading Whisper model ({model_size})"):
            self.whisper_model = whisper.load_model(model_size, device=self.device)
        
        self.logger.info("AudioProcessor initialized successfully")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "whisper": {
                "model_size": self.model_size,
                "language": "en",
                "word_timestamps": True,
                "verbose": False
            },
            "audio_processing": {
                "sample_rate": 16000,
                "n_mfcc": 13,
                "n_mels": 80,
                "n_fft": 2048,
                "hop_length": 512,
                "normalize": True
            },
            "features": {
                "mfcc": True,
                "mel_spectrogram": True,
                "spectral_features": True,
                "zero_crossing_rate": True,
                "rms_energy": True
            }
        }
    
    def load_audio(self, audio_path: str) -> np.ndarray:
        """Load and preprocess audio file."""
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        self.logger.info(f"Loading audio from: {audio_path}")
        
        try:
            # Load audio using librosa
            audio, sr = librosa.load(
                audio_path, 
                sr=self.sample_rate, 
                mono=True
            )
            
            # Normalize audio if configured
            if self.config["audio_processing"]["normalize"]:
                audio = librosa.util.normalize(audio)
            
            # Basic validation
            if len(audio) == 0:
                raise ValueError("Loaded audio is empty")
            
            duration = len(audio) / self.sample_rate
            self.logger.info(f"Audio loaded: {len(audio):,} samples at {sr} Hz ({duration:.2f}s)")
            
            return audio
            
        except Exception as e:
            raise ValueError(f"Failed to load audio file {audio_path}: {str(e)}")
    
    def extract_transcript(self, audio: np.ndarray) -> Dict[str, Any]:
        """Extract transcript using Whisper."""
        with TimedLogger(self.logger, "transcription with Whisper"):
            result = self.whisper_model.transcribe(
                audio,
                language=self.config["whisper"]["language"],
                word_timestamps=self.config["whisper"]["word_timestamps"],
                verbose=self.config["whisper"]["verbose"]
            )
        
        transcript_data = {
            "text": result["text"].strip(),
            "language": result["language"],
            "segments": result.get("segments", []),
            "duration": result.get("duration", len(audio) / self.sample_rate)
        }
        
        if "segments" in result:
            word_count = sum(len(segment.get("words", [])) for segment in result["segments"])
            transcript_data["word_count"] = word_count
        
        self.logger.info(f"Transcript extracted: {len(result['text'])} characters")
        if "segments" in transcript_data:
            self.logger.info(f"Found {len(transcript_data['segments'])} segments")
        
        return transcript_data
    
    def extract_audio_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract rich audio features for lip-sync."""
        with TimedLogger(self.logger, "audio feature extraction"):
            features = []
            config = self.config["audio_processing"]
            
            # 1. MFCC features
            if self.config["features"]["mfcc"]:
                mfccs = librosa.feature.mfcc(
                    y=audio, 
                    sr=self.sample_rate,
                    n_mfcc=config["n_mfcc"],
                    n_fft=config["n_fft"],
                    hop_length=config["hop_length"]
                )
                features.append(mfccs)
            
            # 2. Mel spectrogram
            if self.config["features"]["mel_spectrogram"]:
                mel_spec = librosa.feature.melspectrogram(
                    y=audio,
                    sr=self.sample_rate,
                    n_mels=config["n_mels"],
                    n_fft=config["n_fft"],
                    hop_length=config["hop_length"]
                )
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                features.append(mel_spec_db)
            
            # 3. Spectral features
            if self.config["features"]["spectral_features"]:
                spectral_centroids = librosa.feature.spectral_centroid(
                    y=audio, sr=self.sample_rate, hop_length=config["hop_length"]
                )
                spectral_rolloff = librosa.feature.spectral_rolloff(
                    y=audio, sr=self.sample_rate, hop_length=config["hop_length"]
                )
                spectral_bandwidth = librosa.feature.spectral_bandwidth(
                    y=audio, sr=self.sample_rate, hop_length=config["hop_length"]
                )
                features.extend([spectral_centroids, spectral_rolloff, spectral_bandwidth])
            
            # 4. Zero crossing rate
            if self.config["features"]["zero_crossing_rate"]:
                zcr = librosa.feature.zero_crossing_rate(audio, hop_length=config["hop_length"])
                features.append(zcr)
            
            # 5. RMS energy
            if self.config["features"]["rms_energy"]:
                rms = librosa.feature.rms(y=audio, hop_length=config["hop_length"])
                features.append(rms)
            
            # Concatenate all features
            if not features:
                raise ValueError("No features were extracted. Check configuration.")
            
            audio_features = np.vstack(features)
            self.logger.info(f"Audio features extracted: {audio_features.shape}")
            return audio_features
    
    def save_outputs(self, output_dir: str, transcript_data: Dict[str, Any], 
                    audio_features: np.ndarray, audio_path: str) -> Dict[str, str]:
        """Save all outputs to specified directory."""
        output_path = ensure_directory(output_dir)
        
        # Save transcript text
        transcript_path = output_path / "transcript.txt"
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(transcript_data["text"])
        
        # Save detailed transcript data
        transcript_detail_path = output_path / "transcript_detail.json"
        save_json(transcript_data, transcript_detail_path)
        
        # Save audio features
        features_path = output_path / "audio_feats.npy"
        np.save(features_path, audio_features)
        
        # Save metadata
        metadata = {
            "input_audio_path": str(audio_path),
            "processing_timestamp": time.time(),
            "audio_duration": transcript_data.get("duration", 0),
            "feature_shape": list(audio_features.shape),
            "sample_rate": self.sample_rate,
            "model_size": self.model_size,
            "device_used": self.device,
            "config": self.config
        }
        
        metadata_path = output_path / "audio_metadata.json"
        save_json(metadata, metadata_path)
        
        output_files = {
            "transcript": str(transcript_path),
            "transcript_detail": str(transcript_detail_path),
            "audio_features": str(features_path),
            "metadata": str(metadata_path)
        }
        
        self.logger.info(f"Outputs saved to: {output_path}")
        for key, path in output_files.items():
            file_size = Path(path).stat().st_size
            self.logger.info(f"  {key}: {Path(path).name} ({file_size:,} bytes)")
        
        return output_files
    
    def process_audio(self, input_path: str, output_dir: str) -> bool:
        """Main processing function."""
        try:
            self.logger.info("="*50)
            self.logger.info("Starting Audio Processing Pipeline")
            self.logger.info("="*50)
            
            start_time = time.time()
            
            # Load audio
            audio = self.load_audio(input_path)
            
            # Extract transcript
            transcript_data = self.extract_transcript(audio)
            
            # Extract audio features
            audio_features = self.extract_audio_features(audio)
            
            # Save outputs
            output_files = self.save_outputs(output_dir, transcript_data, audio_features, input_path)
            
            # Final summary
            total_time = time.time() - start_time
            self.logger.info("="*50)
            self.logger.info("Audio Processing Pipeline Completed Successfully")
            self.logger.info(f"Total processing time: {total_time:.2f}s")
            self.logger.info(f"Output files: {len(output_files)}")
            self.logger.info("="*50)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Audio processing failed: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Stage 1: Audio Processing")
    parser.add_argument("--input-audio", type=str, required=True,
                       help="Path to input audio file")
    parser.add_argument("--output-dir", type=str,
                       default=str(PROJECT_ROOT / "outputs" / "stage1"),
                       help="Output directory")
    parser.add_argument("--model-size", type=str, default="base",
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size")
    parser.add_argument("--config", type=str,
                       default=str(PROJECT_ROOT / "configs" / "audio_config.json"),
                       help="Path to configuration file")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not Path(args.input_audio).exists():
        print(f"Error: Input audio file not found: {args.input_audio}")
        return 1
    
    try:
        processor = AudioProcessor(
            config_path=args.config if Path(args.config).exists() else None,
            model_size=args.model_size
        )
        
        success = processor.process_audio(args.input_audio, args.output_dir)
        return 0 if success else 1
        
    except Exception as e:
        print(f"Failed to create audio processor: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
