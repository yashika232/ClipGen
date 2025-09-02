# Simple Video Synthesis Pipeline

A simplified, working video synthesis pipeline that generates talking head videos with optional background animations.

## Features

- **Simple Stage-by-Stage Processing**: Clear progression from script to final video
- **Minimal Metadata Usage**: Only 2 metadata touchpoints as requested
- **Main Models Only**: No fallback mechanisms - fail fast if models unavailable
- **Environment Isolation**: Each stage runs in its own conda environment
- **Easy Configuration**: Single JSON config file

## Pipeline Stages

1. **Script Generation** → `outputs/generated_script.json`
2. **Voice Synthesis** → `outputs/synthesized_speech.wav`
3. **Face Processing** → `outputs/face_crop.jpg`
4. **Video Generation** → `outputs/talking_head.mp4`
5. **Enhancement** → `outputs/enhanced_video.mp4`
6. **Background Animation** → `outputs/background_animation.mp4` (optional)
7. **Final Assembly** → `outputs/final_video.mp4`

## Quick Start

### 1. Setup Configuration

Edit `config.json` and add your Gemini API key:

```json
{
  "gemini_api_key": "your_gemini_api_key_here"
}
```

### 2. Prepare Assets

Create `user_assets/` directory with:
- `face_image.jpg` - Clear face image
- `voice_sample.wav` - Voice sample for cloning (optional)

### 3. Run Pipeline

```bash
python simple_pipeline.py \
  --title "Machine Learning Basics" \
  --topic "Introduction to ML" \
  --face-image user_assets/face_image.jpg \
  --voice-audio user_assets/voice_sample.wav
```

### 4. Optional: Include Background Animation

```bash
python simple_pipeline.py \
  --title "Machine Learning Basics" \
  --topic "Introduction to ML" \
  --face-image user_assets/face_image.jpg \
  --voice-audio user_assets/voice_sample.wav \
  --include-animation
```

## Requirements

### Conda Environments

Create the following conda environments:

```bash
# Voice synthesis
conda create -n xtts python=3.9
conda activate xtts
# Install XTTS dependencies here

# Video generation
conda create -n sadtalker python=3.8
conda activate sadtalker
# Install SadTalker dependencies here

# Enhancement
conda create -n enhancement python=3.9
conda activate enhancement
# Install Real-ESRGAN and CodeFormer dependencies here

# Background animation
conda create -n manim python=3.9
conda activate manim
pip install manim

# Audio processing
conda create -n video-audio-processing python=3.9
conda activate video-audio-processing
pip install librosa soundfile scipy
```

### System Dependencies

- FFmpeg
- OpenCV
- MediaPipe

## Configuration

Edit `config.json` to customize:

- **API Keys**: Gemini API key
- **Conda Environments**: Environment names for each stage
- **Timeouts**: Processing timeouts for each stage
- **Model Paths**: Paths to model directories
- **Quality Settings**: Video and audio quality settings

## Architecture

### Metadata Usage (Only 2 Places)

1. **XTTS Stage**: Reads `clean_script` from `generated_script.json`
2. **Manim Stage**: Reads `manim_code` from `generated_script.json`

### No Complex Metadata Management

- Simple JSON files for data passing between stages
- No enhanced metadata manager
- No complex import dependencies
- Each stage is self-contained

### Main Models Only

- **No Fallback Mechanisms**: Pipeline fails if main model unavailable
- **Fail Fast**: Clear error messages, no silent failures
- **Environment Isolation**: Each model runs in dedicated conda environment

## Troubleshooting

### Common Issues

1. **Conda Environment Not Found**
   ```bash
   conda env list
   # Check if required environments exist
   ```

2. **Gemini API Key Error**
   ```bash
   # Set environment variable
   export GEMINI_API_KEY="your_key_here"
   ```

3. **Model Not Found**
   ```bash
   # Check model paths in config.json
   # Ensure models are downloaded to correct directories
   ```

### Debug Mode

Run with verbose logging:

```bash
python simple_pipeline.py --verbose [other args]
```

### Check Outputs

Each stage creates output files in `outputs/`:
- `session_data.json` - Session information
- `generated_script.json` - Generated script and Manim code
- `synthesized_speech.wav` - Voice synthesis output
- `face_crop.jpg` - Processed face image
- `talking_head.mp4` - Generated video
- `enhanced_video.mp4` - Enhanced video
- `background_animation.mp4` - Background animation (if enabled)
- `final_video.mp4` - Final assembled video

## Development

### Adding New Stages

1. Create new stage file in `stages/`
2. Update `simple_pipeline.py` to include new stage
3. Add conda environment to `config.json`
4. Update timeouts and model paths

### Testing Individual Stages

```bash
# Test script generation
python stages/stage1_script_generation.py

# Test voice synthesis
python stages/stage2_voice_synthesis.py

# Test face processing
python stages/stage3_face_processing.py
```

## Differences from Complex System

- **Removed**: Complex NEW/core/ enhanced classes
- **Removed**: Multi-layer metadata management
- **Removed**: Fallback mechanisms
- **Simplified**: Direct file-based communication between stages
- **Simplified**: Single configuration file
- **Simplified**: Clear stage-by-stage progression

## Next Steps

1. Test the pipeline with your content
2. Adjust conda environments as needed
3. Customize Manim animations in generated code
4. Add more sophisticated error handling if needed

This simplified approach focuses on working functionality over complex abstractions.