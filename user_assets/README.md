# User Assets Directory

This directory contains user-uploaded files organized by session.

## Structure
- `session_[timestamp]_[uuid]/` - Individual session directories
  - `face_image.jpg` - User's face image
  - `voice_sample.wav` - User's voice sample (optional)
  - `user_inputs.json` - Complete user input data

## Usage
Files are automatically organized here when using the interactive input script:
```bash
python get_user_input.py
```

## Cleanup
Old session directories can be safely deleted after video generation is complete.