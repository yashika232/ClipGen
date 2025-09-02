#!/usr/bin/env python3
"""
Interactive Input Starter for Production Pipeline
Checks if production API is running and provides user-friendly guidance
"""

import os
import sys
import subprocess
import requests
import time
from pathlib import Path

def check_api_server():
    """Check if the production API server is running."""
    try:
        response = requests.get("http://localhost:5002/health", timeout=3)
        return response.status_code == 200
    except:
        return False

def check_frontend_server():
    """Check if the frontend server is running."""
    try:
        response = requests.get("http://localhost:8080", timeout=3)
        return response.status_code == 200
    except:
        return False

def main():
    print("VIDEO PIPELINE Interactive Video Generation Input")
    print("=" * 50)
    
    # Check if production API is running
    print("Search Checking production servers...")
    
    api_running = check_api_server()
    frontend_running = check_frontend_server()
    
    print(f"   Tools API Server (5002): {'[SUCCESS] Running' if api_running else '[ERROR] Not Running'}")
    print(f"   Frontend Frontend (8080): {'[SUCCESS] Running' if frontend_running else '[ERROR] Not Running'}")
    
    if not api_running:
        print("\\n[WARNING]  Production API server is not running!")
        print("\\nSTARTING To start the full production system:")
        print("   python start_pipeline.py")
        print("\\nTools To start just the API server:")
        print("   python frontend_api_websocket.py")
        print("\\nINFO: Then run this script again to use interactive input.")
        return 1
    
    if not frontend_running:
        print("\\n[WARNING]  Frontend server is not running!")
        print("   You can still use the CLI, but the web interface won't be available.")
        print("\\nSTARTING To start the full system (API + Frontend):")
        print("   python start_pipeline.py")
    
    print("\\n" + "=" * 50)
    
    # Ask user how they want to proceed
    print("\\nEndpoints How would you like to provide input?")
    print("   1. Interactive CLI (guided prompts)")
    print("   2. Direct command line (advanced)")
    print("   3. Use existing session file")
    print("   4. Use web interface")
    
    choice = input("\\nChoose option (1-4): ").strip()
    
    if choice == "1":
        # Run interactive CLI
        print("\\nTarget: Starting interactive input collection...")
        try:
            from production_user_input import main as interactive_main
            interactive_main()
        except ImportError:
            print("[ERROR] Production input module not found")
            return 1
    
    elif choice == "2":
        # Show command line help
        print("\\nComputer Direct API Usage Examples:")
        print("\\n1. Create a session:")
        print('   curl -X POST http://localhost:5002/session/create -H "Content-Type: application/json" -d \'{"user_id": "cli_user"}\'')
        print("\\n2. Upload face image:")
        print('   curl -X POST -F "file=@/path/to/face.jpg" -F "session_id=your_session_id" http://localhost:5002/upload/face')
        print("\\n3. Upload audio (optional):")
        print('   curl -X POST -F "file=@/path/to/voice.wav" -F "session_id=your_session_id" http://localhost:5002/upload/audio')
        print("\\n4. Start processing:")
        print('   curl -X POST http://localhost:5002/process/start -H "Content-Type: application/json" -d \'{"session_id": "your_session_id", "config": {"title": "My Video", "topic": "Test Topic"}}\'')
    
    elif choice == "3":
        # Session file usage
        print("\\nAssets: Session File Usage:")
        print("\\nIf you have a session file from the simple pipeline:")
        print("   python simple_pipeline.py --session-file path/to/user_inputs.json")
        print("\\nNote: This uses the basic pipeline, not the production API")
    
    elif choice == "4":
        # Web interface
        if frontend_running:
            print("\\nAPI Opening web interface...")
            print("   URL: http://localhost:8080")
            try:
                import webbrowser
                webbrowser.open("http://localhost:8080")
            except:
                pass
        else:
            print("\\n[ERROR] Web interface is not running.")
            print("   Start with: python start_pipeline.py")
    
    else:
        print("\\n[ERROR] Invalid choice. Please run the script again.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())