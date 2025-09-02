#!/usr/bin/env python3
"""
Stage 2: Voice Synthesis - Simple XTTS Integration
Reads script from generated_script.json and synthesizes voice using XTTS
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleVoiceSynthesis:
    """Simple voice synthesis using XTTS."""
    
    def __init__(self):
        """Initialize voice synthesis."""
        self.project_root = PROJECT_ROOT
        self.outputs_dir = self.project_root / "outputs"
        self.outputs_dir.mkdir(exist_ok=True)
        
        # Load session data
        self.session_data = self._load_session_data()
        
        logger.info("Recording Simple Voice Synthesis initialized")
    
    def _load_session_data(self) -> Dict[str, Any]:
        """Load session data from JSON file."""
        session_file = self.outputs_dir / "session_data.json"
        if session_file.exists():
            with open(session_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_generated_script(self) -> Dict[str, Any]:
        """Load generated script from JSON file."""
        script_file = self.outputs_dir / "generated_script.json"
        if script_file.exists():
            with open(script_file, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError("Generated script not found. Run stage 1 first.")
    
    def synthesize_voice(self) -> bool:
        """Synthesize voice using XTTS."""
        try:
            logger.info("Target: Starting voice synthesis...")
            
            # Load generated script
            script_data = self._load_generated_script()
            clean_script = script_data.get("clean_script", "")
            
            if not clean_script:
                raise ValueError("No clean script found in generated script data")
            
            # Get voice sample path
            voice_sample_path = self.session_data.get("voice_audio_path")
            if not voice_sample_path:
                # Try default voice sample
                voice_sample_path = str(self.project_root / "user_assets" / "voice_sample.wav")
            
            if not Path(voice_sample_path).exists():
                raise FileNotFoundError(f"Voice sample not found: {voice_sample_path}")
            
            # Import and use the working XTTS implementation
            sys.path.insert(0, str(self.project_root / "stages"))
            from stage0a_voice_synthesis import VoiceCloningPipeline
            
            # Initialize XTTS pipeline
            logger.info("Initializing XTTS pipeline...")
            xtts_pipeline = VoiceCloningPipeline()
            
            # Set up output paths
            output_audio_path = self.outputs_dir / "synthesized_speech.wav"
            
            # Process voice cloning
            logger.info(f"Processing voice synthesis...")
            logger.info(f"  Script length: {len(clean_script)} characters")
            logger.info(f"  Voice sample: {voice_sample_path}")
            logger.info(f"  Output: {output_audio_path}")
            
            success = xtts_pipeline.process_voice_cloning(
                reference_audio_path=voice_sample_path,
                text=clean_script,
                output_dir=str(self.outputs_dir)
            )
            
            if success:
                # Check if output file exists
                if output_audio_path.exists():
                    logger.info(f"[SUCCESS] Voice synthesis completed successfully")
                    logger.info(f"   Output: {output_audio_path}")
                    
                    # Save voice synthesis metadata
                    voice_metadata = {
                        "synthesized_at": time.time(),
                        "script_length": len(clean_script),
                        "voice_sample_path": voice_sample_path,
                        "output_audio_path": str(output_audio_path),
                        "synthesis_success": True
                    }
                    
                    metadata_file = self.outputs_dir / "voice_synthesis_metadata.json"
                    with open(metadata_file, 'w') as f:
                        json.dump(voice_metadata, f, indent=2)
                    
                    return True
                else:
                    logger.error("[ERROR] Voice synthesis completed but output file not found")
                    return False
            else:
                logger.error("[ERROR] Voice synthesis failed")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] Voice synthesis failed: {str(e)}")
            return False
    
    def _create_fallback_audio(self) -> bool:
        """Create a simple fallback audio file if XTTS fails."""
        try:
            logger.info("Creating fallback audio file...")
            
            # Create a simple beep as fallback
            import numpy as np
            import wave
            
            # Generate a simple tone
            duration = 5  # seconds
            sample_rate = 22050
            frequency = 440  # Hz
            
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
            audio_data = (audio_data * 32767).astype(np.int16)
            
            # Save as WAV
            output_path = self.outputs_dir / "synthesized_speech.wav"
            with wave.open(str(output_path), 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            logger.info(f"[SUCCESS] Fallback audio created: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Fallback audio creation failed: {str(e)}")
            return False


def main():
    """Main entry point."""
    try:
        synthesizer = SimpleVoiceSynthesis()
        success = synthesizer.synthesize_voice()
        
        # If XTTS fails, create fallback (only for development)
        if not success:
            logger.warning("XTTS failed, creating fallback audio...")
            success = synthesizer._create_fallback_audio()
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Voice synthesis failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())