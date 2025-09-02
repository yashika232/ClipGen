#!/usr/bin/env python3
"""
Video Synthesis Pipeline API Server Startup Script
Quick start script for the production-ready API server
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    
    required_packages = [
        'flask', 'flask-cors', 'flask-socketio', 
        'python-magic', 'psutil', 'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  [FOUND] {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  [MISSING] {package}")
    
    if missing_packages:
        print(f"\n[WARNING] Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        for package in missing_packages:
            try:
                subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                             check=True, capture_output=True)
                print(f"  [SUCCESS] Installed {package}")
            except subprocess.CalledProcessError as e:
                print(f"  [ERROR] Failed to install {package}: {e}")
                return False
    
    return True

def check_production_readiness():
    """Check if the system is production ready."""
    print("\nChecking production readiness...")
    
    try:
        # Add paths
        sys.path.insert(0, str(Path(__file__).parent / "NEW" / "core"))
        
        from production_mode_manager import ProductionModeManager
        
        manager = ProductionModeManager()
        capabilities = manager.get_system_capabilities()
        
        print(f"  Environments: {capabilities['environment_status']['available_environments']}/{capabilities['environment_status']['total_environments']}")
        print(f"  Production ready: {'[READY]' if capabilities['production_readiness']['production_ready'] else '[PARTIAL]'}")
        
        if capabilities['production_readiness']['production_ready']:
            print("  [SUCCESS] All systems ready for production video generation!")
        else:
            print("  [WARNING] Some environments missing - will use hybrid mode")
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] Production check failed: {e}")
        return False

def start_api_server():
    """Start the API server."""
    print("\nStarting Video Synthesis Pipeline API Server...")
    print("=" * 60)
    
    try:
        # Import and start the API
        from frontend_api_websocket import app, socketio, production_manager
        
        # Set production mode
        production_manager.set_processing_mode('production')
        
        print("[SUCCESS] API server initialized successfully!")
        print("\nAvailable Endpoints:")
        print("  Health Check: http://localhost:5000/health")
        print("  Capabilities: http://localhost:5000/capabilities")
        print("  WebSocket: ws://localhost:5000/socket.io")
        print("\nDocumentation: See LOCAL_PROTOTYPE_READY_SUMMARY.md")
        print("\n[READY] Ready for frontend integration!")
        print("=" * 60)
        
        # Start the server
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
        
    except KeyboardInterrupt:
        print("\n\n[STOPPED] Server stopped by user")
        print("Thanks for using the Video Synthesis Pipeline!")
    except Exception as e:
        print(f"\n[ERROR] Failed to start server: {e}")
        return False
    
    return True

def main():
    """Main startup function."""
    print("VIDEO SYNTHESIS PIPELINE - API Server Startup")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n[ERROR] Dependency check failed. Please install missing packages manually.")
        return 1
    
    # Check production readiness
    if not check_production_readiness():
        print("\n[WARNING] Production readiness check failed. Server may have limited functionality.")
    
    # Start server
    if not start_api_server():
        print("\n[ERROR] Failed to start API server.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())