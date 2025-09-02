#!/bin/bash

# Video Synthesis Pipeline - Bash Startup Script
# Alternative startup script for Unix/Linux systems
# Provides the same functionality as start_pipeline.py but in bash

set -e  # Exit on error

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_PORT=5002
FRONTEND_PORT=8080
FRONTEND_DIR="$PROJECT_ROOT/genify-dashboard-verse-main"
LOG_DIR="$PROJECT_ROOT/logs"
BACKEND_PID_FILE="$LOG_DIR/backend.pid"
FRONTEND_PID_FILE="$LOG_DIR/frontend.pid"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# Print banner
print_banner() {
    echo -e "${CYAN}🚀$(printf '=%.0s' {1..78})🚀${NC}"
    echo -e "${CYAN}🎬          VIDEO SYNTHESIS PIPELINE - BASH STARTUP          🎬${NC}"
    echo -e "${CYAN}🚀$(printf '=%.0s' {1..78})🚀${NC}"
    echo -e "${PURPLE}📅 Started: $(date)${NC}"
    echo -e "${PURPLE}💻 Platform: $(uname -s) $(uname -r)${NC}"
    echo -e "${PURPLE}🐚 Shell: $SHELL${NC}"
    echo -e "${PURPLE}📁 Project: $PROJECT_ROOT${NC}"
    echo
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python dependencies
check_python_dependencies() {
    log_info "Checking Python dependencies..."
    
    if ! command_exists python3; then
        log_error "Python 3 is not installed"
        return 1
    fi
    
    local python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    log_info "Python version: $python_version"
    
    # Check required packages
    local required_packages=("flask" "flask-cors" "flask-socketio" "requests" "psutil")
    local missing_packages=()
    
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" 2>/dev/null; then
            missing_packages+=("$package")
            log_warning "Missing package: $package"
        else
            log_info "✅ $package"
        fi
    done
    
    if [ ${#missing_packages[@]} -gt 0 ]; then
        log_warning "Missing Python packages: ${missing_packages[*]}"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return 1
        fi
    fi
    
    log_info "✅ Python dependencies check completed!"
    echo
}

# Check Node.js dependencies
check_node_dependencies() {
    log_info "Checking Node.js dependencies..."
    
    if ! command_exists node; then
        log_error "Node.js is not installed"
        log_error "Install Node.js from: https://nodejs.org/"
        return 1
    fi
    
    local node_version=$(node --version)
    log_info "Node.js version: $node_version"
    
    if ! command_exists npm; then
        log_error "npm is not installed"
        return 1
    fi
    
    local npm_version=$(npm --version)
    log_info "npm version: $npm_version"
    
    if [ ! -d "$FRONTEND_DIR" ]; then
        log_error "Frontend directory not found: $FRONTEND_DIR"
        return 1
    fi
    
    log_info "✅ Frontend directory: $FRONTEND_DIR"
    
    if [ ! -f "$FRONTEND_DIR/package.json" ]; then
        log_error "package.json not found: $FRONTEND_DIR/package.json"
        return 1
    fi
    
    log_info "✅ package.json found"
    
    if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
        log_warning "node_modules not found - will install dependencies"
        install_npm_dependencies
    else
        log_info "✅ node_modules found"
    fi
    
    log_info "✅ Node.js dependencies check completed!"
    echo
}

# Install npm dependencies
install_npm_dependencies() {
    log_info "Installing npm dependencies..."
    
    cd "$FRONTEND_DIR"
    if npm install; then
        log_info "✅ npm dependencies installed successfully!"
    else
        log_error "Failed to install npm dependencies"
        return 1
    fi
    cd "$PROJECT_ROOT"
    echo
}

# Check if port is in use
is_port_in_use() {
    local port=$1
    if command_exists lsof; then
        lsof -i :$port >/dev/null 2>&1
    elif command_exists netstat; then
        netstat -tuln | grep ":$port " >/dev/null 2>&1
    else
        # Fallback: try to connect to the port
        (echo > /dev/tcp/localhost/$port) >/dev/null 2>&1
    fi
}

# Kill processes using a port
kill_port_processes() {
    local port=$1
    log_info "Cleaning up port $port..."
    
    if command_exists lsof; then
        local pids=$(lsof -ti :$port 2>/dev/null)
        if [ -n "$pids" ]; then
            log_info "Found processes on port $port: $pids"
            echo "$pids" | xargs kill -9 2>/dev/null || true
            sleep 1
            log_info "✅ Port $port cleaned up"
        fi
    else
        log_warning "lsof not available - cannot clean up port $port"
    fi
}

# Check port availability
check_ports() {
    log_info "Checking port availability..."
    
    # Check backend port
    if is_port_in_use $BACKEND_PORT; then
        log_warning "Port $BACKEND_PORT is in use - cleaning up..."
        kill_port_processes $BACKEND_PORT
        if is_port_in_use $BACKEND_PORT; then
            log_warning "Port $BACKEND_PORT still in use - will try anyway"
        else
            log_info "✅ Port $BACKEND_PORT now available for backend"
        fi
    else
        log_info "✅ Port $BACKEND_PORT available for backend"
    fi
    
    # Check frontend port
    if is_port_in_use $FRONTEND_PORT; then
        log_warning "Port $FRONTEND_PORT is in use - cleaning up..."
        kill_port_processes $FRONTEND_PORT
        if is_port_in_use $FRONTEND_PORT; then
            log_warning "Port $FRONTEND_PORT still in use - Vite will find alternative"
        else
            log_info "✅ Port $FRONTEND_PORT now available for frontend"
        fi
    else
        log_info "✅ Port $FRONTEND_PORT available for frontend"
    fi
    
    echo
}

# Start backend server
start_backend() {
    log_info "Starting backend API server..."
    
    # Create log directory
    mkdir -p "$LOG_DIR"
    
    # Start backend in background
    cd "$PROJECT_ROOT"
    python3 frontend_api_websocket.py > "$LOG_DIR/backend.log" 2>&1 &
    local backend_pid=$!
    echo $backend_pid > "$BACKEND_PID_FILE"
    
    log_info "📡 Backend process started (PID: $backend_pid)"
    log_info "🌐 Backend URL: http://localhost:$BACKEND_PORT"
    
    # Wait for backend to be ready
    log_info "⏳ Waiting for backend to be ready..."
    local max_wait=60
    local wait_time=0
    
    while [ $wait_time -lt $max_wait ]; do
        if curl -s "http://localhost:$BACKEND_PORT/health" >/dev/null 2>&1; then
            log_info "✅ Backend API server is ready!"
            return 0
        fi
        sleep 2
        wait_time=$((wait_time + 2))
    done
    
    log_error "Backend failed to start within timeout"
    return 1
}

# Start frontend server
start_frontend() {
    log_info "Starting frontend development server..."
    
    # Start frontend in background
    cd "$FRONTEND_DIR"
    npm run dev > "$LOG_DIR/frontend.log" 2>&1 &
    local frontend_pid=$!
    echo $frontend_pid > "$FRONTEND_PID_FILE"
    
    log_info "📡 Frontend process started (PID: $frontend_pid)"
    log_info "🌐 Frontend URL: http://localhost:$FRONTEND_PORT"
    
    # Wait for frontend to be ready
    log_info "⏳ Waiting for frontend to be ready..."
    local max_wait=120
    local wait_time=0
    
    while [ $wait_time -lt $max_wait ]; do
        if curl -s "http://localhost:$FRONTEND_PORT" >/dev/null 2>&1; then
            log_info "✅ Frontend development server is ready!"
            return 0
        fi
        sleep 3
        wait_time=$((wait_time + 3))
    done
    
    log_error "Frontend failed to start within timeout"
    return 1
}

# Open browser
open_browser() {
    log_info "Opening browser..."
    
    local url="http://localhost:$FRONTEND_PORT"
    
    if command_exists xdg-open; then
        xdg-open "$url" 2>/dev/null &
    elif command_exists open; then
        open "$url" 2>/dev/null &
    elif command_exists start; then
        start "$url" 2>/dev/null &
    else
        log_warning "Could not open browser automatically"
        log_info "💡 Please manually open: $url"
    fi
    
    log_info "✅ Browser opened to: $url"
    echo
}

# Show service status
show_status() {
    echo -e "${CYAN}📊 SERVICE STATUS${NC}"
    echo "$(printf '=%.0s' {1..50})"
    
    # Check backend
    if [ -f "$BACKEND_PID_FILE" ] && kill -0 $(cat "$BACKEND_PID_FILE") 2>/dev/null; then
        echo -e "${GREEN}🔧 Backend API: ✅ Running${NC}"
    else
        echo -e "${RED}🔧 Backend API: ❌ Not Running${NC}"
    fi
    echo "   └── URL: http://localhost:$BACKEND_PORT"
    echo "   └── Health: http://localhost:$BACKEND_PORT/health"
    
    # Check frontend
    if [ -f "$FRONTEND_PID_FILE" ] && kill -0 $(cat "$FRONTEND_PID_FILE") 2>/dev/null; then
        echo -e "${GREEN}🎨 Frontend: ✅ Running${NC}"
    else
        echo -e "${RED}🎨 Frontend: ❌ Not Running${NC}"
    fi
    echo "   └── URL: http://localhost:$FRONTEND_PORT"
    
    echo -e "${BLUE}📡 WebSocket: ws://localhost:$BACKEND_PORT/socket.io${NC}"
    echo "$(printf '=%.0s' {1..50})"
    echo
}

# Show usage information
show_usage_info() {
    echo -e "${CYAN}📖 USAGE INFORMATION${NC}"
    echo "$(printf '=%.0s' {1..50})"
    echo -e "${PURPLE}🎬 Video Generation Pipeline:${NC}"
    echo "   1. Generate Script - Enter topic, tone, emotion"
    echo "   2. Generate Thumbnails - AI-powered thumbnail creation"
    echo "   3. Upload Files - Face image and audio file"
    echo "   4. Generate Video - Complete pipeline processing"
    echo "   5. Download Results - Final video and assets"
    echo
    echo -e "${PURPLE}🔧 Development:${NC}"
    echo "   • Backend logs: $LOG_DIR/backend.log"
    echo "   • Frontend logs: $LOG_DIR/frontend.log"
    echo "   • API testing: http://localhost:$BACKEND_PORT/health"
    echo
    echo -e "${PURPLE}🛑 To stop: ${NC}"
    echo "   • Press Ctrl+C"
    echo "   • Run: python3 stop_pipeline.py"
    echo "   • Run: ./stop_pipeline.sh"
    echo "$(printf '=%.0s' {1..50})"
    echo
}

# Cleanup function
cleanup() {
    echo
    log_info "🧹 Cleaning up..."
    
    # Kill backend process
    if [ -f "$BACKEND_PID_FILE" ]; then
        local backend_pid=$(cat "$BACKEND_PID_FILE")
        if kill -0 "$backend_pid" 2>/dev/null; then
            log_info "🔧 Stopping backend server..."
            kill "$backend_pid" 2>/dev/null || true
            sleep 2
            kill -9 "$backend_pid" 2>/dev/null || true
        fi
        rm -f "$BACKEND_PID_FILE"
    fi
    
    # Kill frontend process
    if [ -f "$FRONTEND_PID_FILE" ]; then
        local frontend_pid=$(cat "$FRONTEND_PID_FILE")
        if kill -0 "$frontend_pid" 2>/dev/null; then
            log_info "🎨 Stopping frontend server..."
            kill "$frontend_pid" 2>/dev/null || true
            sleep 2
            kill -9 "$frontend_pid" 2>/dev/null || true
        fi
        rm -f "$FRONTEND_PID_FILE"
    fi
    
    log_info "✅ Cleanup completed"
    echo -e "${CYAN}🎬 Thanks for using Video Synthesis Pipeline!${NC}"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Main execution
main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  -h, --help     Show this help message"
                echo "  --no-browser   Don't open browser automatically"
                echo "  --backend-only Start only backend server"
                echo "  --frontend-only Start only frontend server"
                exit 0
                ;;
            --no-browser)
                NO_BROWSER=true
                shift
                ;;
            --backend-only)
                BACKEND_ONLY=true
                shift
                ;;
            --frontend-only)
                FRONTEND_ONLY=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Main startup sequence
    print_banner
    
    if [ "$FRONTEND_ONLY" != true ]; then
        check_python_dependencies
    fi
    
    if [ "$BACKEND_ONLY" != true ]; then
        check_node_dependencies
    fi
    
    check_ports
    
    # Start services
    if [ "$FRONTEND_ONLY" != true ]; then
        start_backend
    fi
    
    if [ "$BACKEND_ONLY" != true ]; then
        start_frontend
    fi
    
    # Show status and usage
    show_status
    
    if [ "$NO_BROWSER" != true ] && [ "$BACKEND_ONLY" != true ]; then
        open_browser
    fi
    
    show_usage_info
    
    # Monitor services
    log_info "👀 Monitoring services... (Press Ctrl+C to stop)"
    echo
    
    while true; do
        # Check if processes are still running
        if [ "$FRONTEND_ONLY" != true ] && [ -f "$BACKEND_PID_FILE" ]; then
            if ! kill -0 $(cat "$BACKEND_PID_FILE") 2>/dev/null; then
                log_error "Backend process died unexpectedly"
                cleanup
            fi
        fi
        
        if [ "$BACKEND_ONLY" != true ] && [ -f "$FRONTEND_PID_FILE" ]; then
            if ! kill -0 $(cat "$FRONTEND_PID_FILE") 2>/dev/null; then
                log_error "Frontend process died unexpectedly"
                cleanup
            fi
        fi
        
        sleep 5
    done
}

# Run main function
main "$@"