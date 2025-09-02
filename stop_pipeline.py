#!/usr/bin/env python3
"""
Stop Pipeline Script - Clean Shutdown Utility
Provides graceful shutdown for the video synthesis pipeline with proper cleanup,
logging finalization, and resource management.
"""

import os
import sys
import signal
import subprocess
import time
import psutil
import requests
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Import logging system
from pipeline_logger import get_logger, LogComponent, set_session_context


class PipelineShutdown:
    """Handles graceful shutdown of the video synthesis pipeline."""
    
    def __init__(self):
        self.logger = get_logger()
        self.project_root = Path(__file__).parent
        self.backend_port = 5002
        self.frontend_port = 8080
        self.shutdown_timeout = 30  # seconds
        
        # Set session context for logging
        set_session_context("shutdown_session", "system")
        
        self.logger.info(
            LogComponent.SYSTEM,
            "shutdown_initiated",
            "Pipeline shutdown process initiated"
        )
    
    def find_pipeline_processes(self) -> List[Dict[str, Any]]:
        """Find all running pipeline processes."""
        pipeline_processes = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    
                    # Check for pipeline-related processes
                    if any(keyword in cmdline.lower() for keyword in [
                        'frontend_api_websocket.py',
                        'start_pipeline.py',
                        'gemini_script_generator.py',
                        'ai_thumbnail_generator.py',
                        'npm run dev',
                        'vite',
                        'react-scripts'
                    ]):
                        pipeline_processes.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'cmdline': cmdline,
                            'create_time': proc.info['create_time']
                        })
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
        except Exception as e:
            self.logger.error(
                LogComponent.SYSTEM,
                "process_discovery_failed",
                f"Failed to discover pipeline processes: {e}",
                error=e
            )
        
        self.logger.info(
            LogComponent.SYSTEM,
            "processes_discovered",
            f"Found {len(pipeline_processes)} pipeline processes",
            metadata={'process_count': len(pipeline_processes)}
        )
        
        return pipeline_processes
    
    def check_port_usage(self) -> Dict[str, Any]:
        """Check which ports are in use by the pipeline."""
        port_info = {}
        
        for port in [self.backend_port, self.frontend_port]:
            try:
                connections = psutil.net_connections(kind='inet')
                port_processes = []
                
                for conn in connections:
                    if conn.laddr.port == port and conn.status == 'LISTEN':
                        try:
                            proc = psutil.Process(conn.pid)
                            port_processes.append({
                                'pid': conn.pid,
                                'name': proc.name(),
                                'cmdline': ' '.join(proc.cmdline())
                            })
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
                
                port_info[port] = {
                    'in_use': len(port_processes) > 0,
                    'processes': port_processes
                }
                
            except Exception as e:
                self.logger.warning(
                    LogComponent.SYSTEM,
                    "port_check_failed",
                    f"Failed to check port {port}: {e}",
                    metadata={'port': port},
                    error=e
                )
                port_info[port] = {'in_use': False, 'processes': []}
        
        return port_info
    
    def graceful_api_shutdown(self) -> bool:
        """Attempt graceful shutdown via API endpoint."""
        try:
            self.logger.info(
                LogComponent.SYSTEM,
                "api_shutdown_attempt",
                "Attempting graceful shutdown via API"
            )
            
            # Try to call shutdown endpoint
            response = requests.post(
                f"http://localhost:{self.backend_port}/api/shutdown",
                json={'graceful': True},
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.info(
                    LogComponent.SYSTEM,
                    "api_shutdown_success",
                    "Graceful shutdown initiated via API"
                )
                return True
            else:
                self.logger.warning(
                    LogComponent.SYSTEM,
                    "api_shutdown_failed",
                    f"API shutdown failed with status {response.status_code}"
                )
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.info(
                LogComponent.SYSTEM,
                "api_shutdown_unavailable",
                f"API shutdown not available: {e}"
            )
            return False
    
    def terminate_process(self, pid: int, name: str, graceful: bool = True) -> bool:
        """Terminate a specific process with optional graceful shutdown."""
        try:
            proc = psutil.Process(pid)
            
            if graceful:
                # Try graceful termination first
                self.logger.info(
                    LogComponent.SYSTEM,
                    "process_termination_graceful",
                    f"Sending SIGTERM to process {name} (PID: {pid})"
                )
                proc.terminate()
                
                # Wait for graceful shutdown
                try:
                    proc.wait(timeout=self.shutdown_timeout)
                    self.logger.info(
                        LogComponent.SYSTEM,
                        "process_terminated_gracefully",
                        f"Process {name} (PID: {pid}) terminated gracefully"
                    )
                    return True
                except psutil.TimeoutExpired:
                    self.logger.warning(
                        LogComponent.SYSTEM,
                        "process_termination_timeout",
                        f"Process {name} (PID: {pid}) did not terminate gracefully"
                    )
            
            # Force kill if graceful failed
            self.logger.info(
                LogComponent.SYSTEM,
                "process_termination_forced",
                f"Force killing process {name} (PID: {pid})"
            )
            proc.kill()
            proc.wait(timeout=10)
            
            self.logger.info(
                LogComponent.SYSTEM,
                "process_killed",
                f"Process {name} (PID: {pid}) killed successfully"
            )
            return True
            
        except psutil.NoSuchProcess:
            self.logger.info(
                LogComponent.SYSTEM,
                "process_already_dead",
                f"Process {name} (PID: {pid}) already terminated"
            )
            return True
        except Exception as e:
            self.logger.error(
                LogComponent.SYSTEM,
                "process_termination_failed",
                f"Failed to terminate process {name} (PID: {pid}): {e}",
                error=e
            )
            return False
    
    def kill_port_processes(self, port: int) -> bool:
        """Kill all processes using a specific port."""
        try:
            # Use lsof to find processes using the port
            result = subprocess.run(
                ['lsof', '-ti', f':{port}'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                success_count = 0
                
                for pid_str in pids:
                    if pid_str:
                        try:
                            pid = int(pid_str)
                            if self.terminate_process(pid, f"port_{port}_process"):
                                success_count += 1
                        except ValueError:
                            continue
                
                self.logger.info(
                    LogComponent.SYSTEM,
                    "port_processes_killed",
                    f"Killed {success_count}/{len(pids)} processes on port {port}",
                    metadata={'port': port, 'success_count': success_count, 'total_count': len(pids)}
                )
                return success_count > 0
            
            return True  # No processes found
            
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            self.logger.warning(
                LogComponent.SYSTEM,
                "port_cleanup_failed",
                f"Could not clean up port {port} using lsof",
                metadata={'port': port}
            )
            return False
    
    def cleanup_temp_files(self):
        """Clean up temporary files and directories."""
        temp_dirs = [
            'temp',
            'uploads',
            'generated_thumbnails',
            'sessions',
            'quarantine'
        ]
        
        cleaned_files = 0
        
        for temp_dir in temp_dirs:
            temp_path = self.project_root / temp_dir
            if temp_path.exists():
                try:
                    for file_path in temp_path.glob('*'):
                        if file_path.is_file():
                            # Keep files from today
                            if file_path.stat().st_mtime < (time.time() - 24 * 60 * 60):
                                file_path.unlink()
                                cleaned_files += 1
                        elif file_path.is_dir() and not any(file_path.iterdir()):
                            # Remove empty directories
                            file_path.rmdir()
                            
                except Exception as e:
                    self.logger.warning(
                        LogComponent.SYSTEM,
                        "temp_cleanup_failed",
                        f"Failed to clean up {temp_dir}: {e}",
                        metadata={'temp_dir': str(temp_dir)},
                        error=e
                    )
        
        if cleaned_files > 0:
            self.logger.info(
                LogComponent.SYSTEM,
                "temp_files_cleaned",
                f"Cleaned up {cleaned_files} temporary files",
                metadata={'files_cleaned': cleaned_files}
            )
    
    def finalize_logs(self):
        """Finalize logs and perform cleanup."""
        try:
            # Flush any remaining logs
            self.logger.info(
                LogComponent.SYSTEM,
                "log_finalization",
                "Finalizing logs and performing cleanup"
            )
            
            # Clean up old logs
            self.logger.cleanup_old_logs()
            
            # Final log entry
            self.logger.info(
                LogComponent.SYSTEM,
                "shutdown_complete",
                "Pipeline shutdown completed successfully",
                metadata={
                    'shutdown_time': datetime.now().isoformat(),
                    'total_duration': time.time()
                }
            )
            
        except Exception as e:
            print(f"Error during log finalization: {e}")
    
    def run_shutdown(self, force: bool = False):
        """Execute the complete shutdown process."""
        start_time = time.time()
        
        print("[STOPPED] Video Synthesis Pipeline - Shutdown Process")
        print("=" * 60)
        
        # Step 1: Discover running processes
        print("Search Discovering pipeline processes...")
        processes = self.find_pipeline_processes()
        
        if not processes and not force:
            print("[SUCCESS] No pipeline processes found - system appears to be stopped")
            self.logger.info(
                LogComponent.SYSTEM,
                "shutdown_not_needed",
                "No pipeline processes found - system already stopped"
            )
            return
        
        # Step 2: Check port usage
        print("Ports Checking port usage...")
        port_info = self.check_port_usage()
        
        # Step 3: Attempt graceful shutdown
        if not force:
            print("Integration Attempting graceful shutdown...")
            if self.graceful_api_shutdown():
                print("Processing Waiting for graceful shutdown...")
                time.sleep(5)
                
                # Re-check processes
                remaining_processes = self.find_pipeline_processes()
                if not remaining_processes:
                    print("[SUCCESS] Graceful shutdown completed successfully")
                    self.cleanup_temp_files()
                    self.finalize_logs()
                    return
        
        # Step 4: Force termination
        print("[EMOJI] Force terminating processes...")
        terminated_count = 0
        
        for process in processes:
            if self.terminate_process(process['pid'], process['name'], graceful=not force):
                terminated_count += 1
        
        print(f"[EMOJI] Terminated {terminated_count}/{len(processes)} processes")
        
        # Step 5: Clean up ports
        print("🧹 Cleaning up ports...")
        for port in [self.backend_port, self.frontend_port]:
            if port_info.get(port, {}).get('in_use'):
                self.kill_port_processes(port)
        
        # Step 6: Clean up temporary files
        print("[EMOJI]  Cleaning up temporary files...")
        self.cleanup_temp_files()
        
        # Step 7: Finalize logs
        print("Step Finalizing logs...")
        self.finalize_logs()
        
        # Summary
        duration = time.time() - start_time
        print(f"[SUCCESS] Shutdown completed in {duration:.2f} seconds")
        print("VIDEO PIPELINE Video Synthesis Pipeline stopped successfully!")


def main():
    """Main entry point for the shutdown script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Stop the video synthesis pipeline')
    parser.add_argument('--force', '-f', action='store_true', 
                       help='Force shutdown without attempting graceful termination')
    parser.add_argument('--timeout', '-t', type=int, default=30,
                       help='Timeout for graceful shutdown (default: 30 seconds)')
    
    args = parser.parse_args()
    
    try:
        shutdown = PipelineShutdown()
        shutdown.shutdown_timeout = args.timeout
        shutdown.run_shutdown(force=args.force)
        
    except KeyboardInterrupt:
        print("\n[STOPPED] Shutdown interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Shutdown failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()