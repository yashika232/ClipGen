# 🎬 Video Synthesis Pipeline

> **AI-powered talking head video generation with production-ready REST API**

A complete video synthesis pipeline that transforms text scripts and face images into professional talking head videos using state-of-the-art AI models including XTTS voice cloning, SadTalker lip-sync, and Real-ESRGAN enhancement.

## ✨ Key Features

- **🗣️ Voice Cloning**: Natural voice synthesis using XTTS technology
- **👤 Face Animation**: Realistic lip-sync and facial expressions with SadTalker
- **🎥 Video Enhancement**: AI-powered upscaling with Real-ESRGAN
- **🌐 REST API**: Production-ready Flask API with WebSocket support
- **👥 Multi-User**: Session-based isolation and concurrent processing
- **⚡ Real-Time**: Live progress updates via WebSocket
- **🔒 Secure**: File validation, virus scanning, and resource limits

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.8+ required
python --version

# Install system dependencies (macOS)
brew install ffmpeg libmagic

# Install system dependencies (Ubuntu)
sudo apt-get update
sudo apt-get install ffmpeg libmagic1 clamav clamav-daemon
```

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/video-synthesis-pipeline.git
cd video-synthesis-pipeline

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export GEMINI_API_KEY="your_gemini_api_key"
export PRODUCTION_MODE="true"
```

### Start the API Server
```bash
# Start the production API server
python start_api_server.py

# API available at: http://localhost:5000
# WebSocket at: ws://localhost:5000/socket.io
# Health check: http://localhost:5000/health
```

## 📚 API Usage

### Create Session & Generate Video
```python
import requests
import socketio

# Create session
response = requests.post('http://localhost:5000/session/create', 
    json={'user_id': 'user123'})
session_id = response.json()['session_id']

# Upload face image
with open('face.jpg', 'rb') as f:
    files = {'file': f}
    data = {'session_id': session_id}
    requests.post('http://localhost:5000/upload/face', 
        files=files, data=data)

# Upload audio or start with text
requests.post('http://localhost:5000/process/start',
    json={
        'session_id': session_id,
        'script_text': 'Hello, this is my talking head video!'
    })

# Real-time progress via WebSocket
sio = socketio.Client()
sio.connect('http://localhost:5000')
sio.emit('subscribe_progress', {'session_id': session_id})

@sio.on('progress_update')
def on_progress(data):
    print(f"Progress: {data['progress_percent']}% - {data['message']}")
```

### JavaScript/React Integration
```javascript
// Frontend integration example
const socket = io('ws://localhost:5000');

// Create session
const response = await fetch('/session/create', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({user_id: 'user123'})
});
const {session_id} = await response.json();

// Track progress
socket.emit('subscribe_progress', {session_id});
socket.on('progress_update', (data) => {
    console.log(`${data.progress_percent}%: ${data.message}`);
});

// Start processing
await fetch('/process/start', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        session_id: session_id,
        script_text: 'Your video script here'
    })
});
```

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │◄───┤  Flask API      │◄───┤ Session Manager │
│   Application   │    │  + WebSocket    │    │ + File Security │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   AI Pipeline       │
                    │                     │
                    │  Text → XTTS        │
                    │    ↓                │
                    │  Audio → SadTalker  │
                    │    ↓                │
                    │  Video → Real-ESRGAN│
                    │    ↓                │
                    │  Enhanced Video     │
                    └─────────────────────┘
```

## 📋 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health and status |
| `GET` | `/capabilities` | Production capabilities |
| `POST` | `/session/create` | Create user session |
| `GET` | `/session/{id}/status` | Get session status |
| `POST` | `/upload/face` | Upload face image |
| `POST` | `/upload/audio` | Upload audio file |
| `POST` | `/process/start` | Start video generation |
| `GET` | `/outputs/{session_id}` | List generated files |
| `GET` | `/download/{session_id}/{filename}` | Download files |
| `POST` | `/cleanup` | Manual cleanup |

## 🛠️ Development

### Environment Setup
The pipeline uses separate conda environments for different AI models:

```bash
# Required conda environments
conda create -n xtts_voice_cloning python=3.9
conda create -n sadtalker python=3.8  
conda create -n realesrgan_real python=3.9
conda create -n video-audio-processing python=3.9
```

### Running Tests
```bash
# Production readiness tests
python test_production_video_generation.py

# API endpoint tests  
python test_api_endpoints.py

# Session management tests
python test_session_management.py
```

### Frontend Development
```bash
# Terminal 1: Start API server
python frontend_api_websocket.py

# Terminal 2: Start your frontend application
cd genify-dashboard-verse-main
npm install
npm run dev

# Terminal 3: Test API endpoints
python test_api_endpoints.py
```

## 📁 Project Structure

```
video-synthesis-pipeline/
├── 📄 README.md                    # This file
├── 📦 requirements.txt             # Python dependencies
├── 🚀 start_api_server.py          # Quick start script
├── 🌐 frontend_api_websocket.py    # Main API server
├── 🧪 test_*.py                    # Test suites
│
├── NEW/                            # Core production system
│   ├── core/                       # Production managers
│   │   ├── production_mode_manager.py
│   │   ├── concurrent_session_manager.py
│   │   ├── realtime_progress_manager.py
│   │   └── secure_file_validator.py
│   └── video_outputs/              # Generated content
│
├── stages/                         # Pipeline stages
│   ├── stage0a_voice_synthesis.py
│   ├── stage2_face.py
│   ├── stage3_lipsync.py
│   └── stage4_enhance.py
│
├── genify-dashboard-verse-main/    # React frontend
├── datasets/                       # Sample data
├── configs/                        # Configuration files
└── requirements/                   # Detailed requirements
```

## ⚙️ Configuration

### Environment Variables
```bash
# Required
export GEMINI_API_KEY="your_gemini_api_key"
export PRODUCTION_MODE="true"

# Optional
export MAX_CONCURRENT_SESSIONS="5"
export SESSION_TIMEOUT_HOURS="2"
export MAX_FILE_SIZE_MB="50"
export ENABLE_VIRUS_SCANNING="true"
```

### Performance Settings
- **Processing Time**: ~4-5 minutes per video
- **Memory Usage**: 4-8GB during processing
- **Concurrent Sessions**: 5 (configurable)
- **File Size Limit**: 50MB per upload
- **Supported Formats**: JPG, PNG, GIF, BMP | WAV, MP3, FLAC, OGG

## 🔒 Security Features

- ✅ **File Validation**: Multi-layer file type and content verification
- ✅ **Virus Scanning**: Optional ClamAV integration
- ✅ **Session Isolation**: Complete user session separation
- ✅ **Resource Limits**: Memory and processing constraints
- ✅ **Input Sanitization**: Secure file handling and path validation
- ✅ **CORS Configuration**: Cross-origin request management

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation for API changes
- Ensure all tests pass before submitting

## 📊 Performance & Monitoring

### Health Check
```bash
# Check system status
curl http://localhost:5000/health

# Expected response
{
    "healthy": true,
    "production_ready": true,
    "memory_usage_percent": 45.2,
    "active_sessions": 2,
    "system_load": "normal"
}
```

### Resource Monitoring
```bash
# Monitor system resources
curl http://localhost:5000/health | jq '.memory_usage_percent'

# Trigger cleanup if needed
curl -X POST http://localhost:5000/cleanup
```

## 🐛 Troubleshooting

### Common Issues

**Q: "Module not found" errors during startup**
```bash
# Install missing dependencies
pip install -r requirements.txt

# Check conda environments
conda env list
```

**Q: "Memory limit exceeded" during processing**
```bash
# Reduce concurrent sessions
export MAX_CONCURRENT_SESSIONS="2"

# Or trigger manual cleanup
curl -X POST http://localhost:5000/cleanup
```

**Q: WebSocket connection fails**
```bash
# Check if port 5000 is available
lsof -i :5000

# Restart the API server
python start_api_server.py
```

For more detailed troubleshooting, see `nonuseful/debug_legacy/legacy_docs/PRODUCTION_README.md`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **SadTalker**: Lip-sync and facial animation technology
- **XTTS**: Advanced text-to-speech synthesis
- **Real-ESRGAN**: AI-powered video enhancement
- **OpenCV**: Computer vision processing
- **Flask**: Web framework for API development

## 📞 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/your-username/video-synthesis-pipeline/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/your-username/video-synthesis-pipeline/discussions)
- 📧 **Email**: support@yourproject.com

---

**Ready to create amazing talking head videos?** 🚀

```bash
# Get started in 3 commands
git clone https://github.com/your-username/video-synthesis-pipeline.git
cd video-synthesis-pipeline && pip install -r requirements.txt
python start_api_server.py
```

*Built with ❤️ using AI and modern web technologies*# video-synthesis-pipeline-copy
# video-synthesis-pipeline-copy
