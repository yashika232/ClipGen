#!/usr/bin/env python3
"""
Script to run start_pipeline.py with a 5-hour timeout
"""
import subprocess
import signal
import time
import sys
import os
from datetime import datetime, timedelta

def run_with_timeout():
    print("STARTING Starting Video Synthesis Pipeline with 5-hour timeout...")
    print(f"Date: Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Start the pipeline process
    process = subprocess.Popen(
        [sys.executable, 'start_pipeline.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    start_time = time.time()
    timeout_seconds = 5 * 60 * 60  # 5 hours
    end_time = datetime.now() + timedelta(hours=5)
    
    print(f"⏰ Will run until: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"ID Process ID: {process.pid}")
    print("="*60)
    
    try:
        # Monitor the process
        while True:
            # Check if process is still running
            if process.poll() is not None:
                print("\n[EMOJI] Pipeline process completed naturally")
                break
                
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                print(f"\n⏰ 5-hour timeout reached. Stopping pipeline...")
                process.terminate()
                time.sleep(5)  # Give it time to terminate gracefully
                if process.poll() is None:
                    print("[EMOJI] Force killing process...")
                    process.kill()
                break
            
            # Read and display output
            try:
                line = process.stdout.readline()
                if line:
                    print(line.rstrip())
                else:
                    time.sleep(1)
            except:
                time.sleep(1)
                
            # Show remaining time every 30 minutes
            if int(elapsed) % 1800 == 0 and elapsed > 0:
                remaining = timeout_seconds - elapsed
                remaining_hours = remaining / 3600
                print(f"\nProcessing Time remaining: {remaining_hours:.1f} hours")
                
    except KeyboardInterrupt:
        print(f"\n[WARNING] Interrupted by user. Stopping pipeline...")
        process.terminate()
        time.sleep(5)
        if process.poll() is None:
            process.kill()
    
    finally:
        total_time = time.time() - start_time
        print(f"\nStatus: Total runtime: {total_time/3600:.2f} hours")
        print(f"[EMOJI] Pipeline stopped at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    run_with_timeout()