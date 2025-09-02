#!/usr/bin/env python3
"""
5-hour timeout monitor for the pipeline process
"""
import time
import subprocess
import signal
import os
from datetime import datetime, timedelta

def monitor_pipeline():
    # Find the current pipeline process
    try:
        result = subprocess.run(['pgrep', '-f', 'start_pipeline.py'], capture_output=True, text=True)
        if result.returncode == 0:
            pid = int(result.stdout.strip().split('\n')[0])
            print(f"Target: Found pipeline process PID: {pid}")
        else:
            print("[ERROR] No pipeline process found")
            return
    except Exception as e:
        print(f"[ERROR] Error finding process: {e}")
        return
    
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=5)
    
    print(f"⏰ Pipeline started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏰ Will terminate at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Processing Running for 5 hours (18000 seconds)")
    print("="*60)
    
    try:
        # Monitor for 5 hours (18000 seconds)
        for i in range(18000):
            time.sleep(1)
            
            # Check if process is still running
            try:
                os.kill(pid, 0)  # This doesn't kill, just checks if process exists
            except OSError:
                print(f"\n[EMOJI] Pipeline process {pid} has stopped naturally")
                return
            
            # Print progress every 30 minutes (1800 seconds)
            if i > 0 and i % 1800 == 0:
                elapsed_hours = i / 3600
                remaining_hours = 5 - elapsed_hours
                current_time = datetime.now()
                print(f"\nProcessing Progress: {elapsed_hours:.1f}h elapsed, {remaining_hours:.1f}h remaining")
                print(f"Date: Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 5 hours have passed, terminate the process
        print(f"\n⏰ 5-hour timeout reached!")
        print(f"[EMOJI] Terminating pipeline process {pid}...")
        
        try:
            os.kill(pid, signal.SIGTERM)
            print("[SUCCESS] Sent SIGTERM signal")
            
            # Wait 10 seconds for graceful shutdown
            time.sleep(10)
            
            # Check if it's still running
            try:
                os.kill(pid, 0)
                print("[WARNING] Process still running, sending SIGKILL...")
                os.kill(pid, signal.SIGKILL)
                print("[SUCCESS] Process force killed")
            except OSError:
                print("[SUCCESS] Process terminated gracefully")
                
        except OSError as e:
            print(f"[ERROR] Error terminating process: {e}")
    
    except KeyboardInterrupt:
        print(f"\n[WARNING] Monitor interrupted by user")
        return
    
    finally:
        end_time_actual = datetime.now()
        total_runtime = end_time_actual - start_time
        print(f"\nStatus: Total runtime: {total_runtime}")
        print(f"[EMOJI] Monitor stopped at: {end_time_actual.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    monitor_pipeline()