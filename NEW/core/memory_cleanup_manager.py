#!/usr/bin/env python3
"""
Memory and Cleanup Manager - Resource Management System
Comprehensive memory monitoring, cleanup, and resource management for production deployment
Prevents memory leaks, manages disk space, and maintains system health
"""

import os
import sys
import gc
import psutil
import shutil
import threading
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import tempfile
import subprocess

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from pipeline_exceptions import (
    InsufficientStorageError, InsufficientMemoryError, ProcessingTimeoutError
)


class CleanupPriority(Enum):
    """Cleanup priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ResourceType(Enum):
    """Types of resources being managed."""
    MEMORY = "memory"
    STORAGE = "storage"
    TEMP_FILES = "temp_files"
    SESSION_DATA = "session_data"
    CACHE = "cache"
    LOGS = "logs"


@dataclass
class ResourceUsage:
    """Resource usage tracking."""
    resource_type: ResourceType
    current_usage: float
    max_usage: float
    usage_percent: float
    last_updated: datetime
    trend: str  # "increasing", "decreasing", "stable"


@dataclass
class CleanupTask:
    """Cleanup task definition."""
    task_id: str
    priority: CleanupPriority
    resource_type: ResourceType
    target_path: str
    max_age: timedelta
    size_threshold: Optional[int]
    cleanup_function: Callable[[], bool]
    last_run: Optional[datetime]
    success_count: int
    failure_count: int


class MemoryCleanupManager:
    """Comprehensive memory and resource cleanup manager."""
    
    def __init__(self, base_dir: str = None):
        """Initialize memory and cleanup manager.
        
        Args:
            base_dir: Base directory for the pipeline
        """
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        # Resource monitoring settings
        self.memory_warning_threshold = 80  # Percent
        self.memory_critical_threshold = 90  # Percent
        self.storage_warning_threshold = 85  # Percent
        self.storage_critical_threshold = 95  # Percent
        
        # Cleanup intervals
        self.cleanup_intervals = {
            CleanupPriority.CRITICAL: timedelta(minutes=5),
            CleanupPriority.HIGH: timedelta(minutes=15),
            CleanupPriority.MEDIUM: timedelta(hours=1),
            CleanupPriority.LOW: timedelta(hours=6)
        }
        
        # Thread-safe tracking
        self._resource_lock = threading.RLock()
        self._cleanup_lock = threading.RLock()
        self._active_cleanups: Set[str] = set()
        
        # Resource tracking
        self._resource_usage: Dict[ResourceType, ResourceUsage] = {}
        self._cleanup_tasks: Dict[str, CleanupTask] = {}
        self._resource_history: List[Dict[str, Any]] = []
        
        # Directories for cleanup
        self.temp_dirs = [
            self.base_dir / "temp",
            self.base_dir / "sessions",
            self.base_dir / "cache",
            Path(tempfile.gettempdir()) / "video_synthesis_pipeline"
        ]
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize monitoring
        self._initialize_cleanup_tasks()
        self._start_monitoring_threads()
        
        self.logger.info("🧹 Memory and Cleanup Manager initialized")
    
    def _initialize_cleanup_tasks(self) -> None:
        """Initialize default cleanup tasks."""
        # Temporary files cleanup
        self.register_cleanup_task(
            task_id="temp_files_cleanup",
            priority=CleanupPriority.HIGH,
            resource_type=ResourceType.TEMP_FILES,
            target_path=str(self.base_dir / "temp"),
            max_age=timedelta(hours=1),
            cleanup_function=self._cleanup_temp_files
        )
        
        # Old session data cleanup
        self.register_cleanup_task(
            task_id="old_sessions_cleanup",
            priority=CleanupPriority.MEDIUM,
            resource_type=ResourceType.SESSION_DATA,
            target_path=str(self.base_dir / "sessions"),
            max_age=timedelta(hours=24),
            cleanup_function=self._cleanup_old_sessions
        )
        
        # Cache cleanup
        self.register_cleanup_task(
            task_id="cache_cleanup",
            priority=CleanupPriority.LOW,
            resource_type=ResourceType.CACHE,
            target_path=str(self.base_dir / "cache"),
            max_age=timedelta(days=7),
            cleanup_function=self._cleanup_cache
        )
        
        # Log rotation
        self.register_cleanup_task(
            task_id="log_rotation",
            priority=CleanupPriority.LOW,
            resource_type=ResourceType.LOGS,
            target_path=str(self.base_dir / "logs"),
            max_age=timedelta(days=30),
            cleanup_function=self._cleanup_logs
        )
        
        # Memory garbage collection
        self.register_cleanup_task(
            task_id="memory_gc",
            priority=CleanupPriority.HIGH,
            resource_type=ResourceType.MEMORY,
            target_path="",
            max_age=timedelta(minutes=30),
            cleanup_function=self._force_garbage_collection
        )
    
    def register_cleanup_task(self, task_id: str, priority: CleanupPriority,
                            resource_type: ResourceType, target_path: str,
                            max_age: timedelta, cleanup_function: Callable[[], bool],
                            size_threshold: Optional[int] = None) -> None:
        """Register a cleanup task.
        
        Args:
            task_id: Unique task identifier
            priority: Cleanup priority
            resource_type: Type of resource being cleaned
            target_path: Path to clean
            max_age: Maximum age before cleanup
            cleanup_function: Function to perform cleanup
            size_threshold: Optional size threshold for triggering cleanup
        """
        with self._cleanup_lock:
            task = CleanupTask(
                task_id=task_id,
                priority=priority,
                resource_type=resource_type,
                target_path=target_path,
                max_age=max_age,
                size_threshold=size_threshold,
                cleanup_function=cleanup_function,
                last_run=None,
                success_count=0,
                failure_count=0
            )
            
            self._cleanup_tasks[task_id] = task
            self.logger.info(f"Endpoints Registered cleanup task: {task_id}")
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage statistics.
        
        Returns:
            Comprehensive resource usage information
        """
        with self._resource_lock:
            # Update current usage
            self._update_resource_usage()
            
            # Compile usage statistics
            usage_stats = {}
            for resource_type, usage in self._resource_usage.items():
                usage_stats[resource_type.value] = {
                    'current_usage': usage.current_usage,
                    'max_usage': usage.max_usage,
                    'usage_percent': usage.usage_percent,
                    'trend': usage.trend,
                    'last_updated': usage.last_updated.isoformat(),
                    'status': self._get_usage_status(usage.usage_percent)
                }
            
            # Add system-wide statistics
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage(str(self.base_dir))
            
            system_stats = {
                'memory': {
                    'total_gb': round(memory.total / (1024**3), 2),
                    'available_gb': round(memory.available / (1024**3), 2),
                    'used_percent': memory.percent,
                    'status': self._get_usage_status(memory.percent)
                },
                'storage': {
                    'total_gb': round(disk.total / (1024**3), 2),
                    'free_gb': round(disk.free / (1024**3), 2),
                    'used_percent': round((disk.used / disk.total) * 100, 1),
                    'status': self._get_usage_status((disk.used / disk.total) * 100)
                },
                'process': {
                    'cpu_percent': psutil.cpu_percent(interval=0.1),
                    'memory_mb': round(psutil.Process().memory_info().rss / (1024**2), 1),
                    'open_files': len(psutil.Process().open_files()),
                    'threads': psutil.Process().num_threads()
                }
            }
            
            return {
                'resource_usage': usage_stats,
                'system_stats': system_stats,
                'cleanup_tasks': self._get_cleanup_task_status(),
                'last_updated': datetime.now().isoformat()
            }
    
    def force_cleanup(self, priority_level: Optional[CleanupPriority] = None,
                     resource_type: Optional[ResourceType] = None) -> Dict[str, Any]:
        """Force immediate cleanup operation.
        
        Args:
            priority_level: Optional priority filter
            resource_type: Optional resource type filter
            
        Returns:
            Cleanup results
        """
        cleanup_results = {
            'started_at': datetime.now().isoformat(),
            'tasks_run': 0,
            'tasks_successful': 0,
            'tasks_failed': 0,
            'space_freed_mb': 0,
            'errors': []
        }
        
        with self._cleanup_lock:
            for task_id, task in self._cleanup_tasks.items():
                # Apply filters
                if priority_level and task.priority != priority_level:
                    continue
                if resource_type and task.resource_type != resource_type:
                    continue
                
                # Skip if already running
                if task_id in self._active_cleanups:
                    continue
                
                # Run cleanup task
                try:
                    self._active_cleanups.add(task_id)
                    cleanup_results['tasks_run'] += 1
                    
                    self.logger.info(f"🧹 Running forced cleanup: {task_id}")
                    
                    # Get before stats
                    before_stats = self._get_target_stats(task.target_path)
                    
                    # Run cleanup
                    success = task.cleanup_function()
                    
                    # Get after stats
                    after_stats = self._get_target_stats(task.target_path)
                    space_freed = before_stats.get('size_mb', 0) - after_stats.get('size_mb', 0)
                    
                    if success:
                        cleanup_results['tasks_successful'] += 1
                        cleanup_results['space_freed_mb'] += max(0, space_freed)
                        task.success_count += 1
                        task.last_run = datetime.now()
                        self.logger.info(f"[SUCCESS] Cleanup successful: {task_id} (freed {space_freed:.1f}MB)")
                    else:
                        cleanup_results['tasks_failed'] += 1
                        task.failure_count += 1
                        cleanup_results['errors'].append(f"Cleanup failed: {task_id}")
                        self.logger.warning(f"[ERROR] Cleanup failed: {task_id}")
                
                except Exception as e:
                    cleanup_results['tasks_failed'] += 1
                    task.failure_count += 1
                    error_msg = f"Cleanup error in {task_id}: {str(e)}"
                    cleanup_results['errors'].append(error_msg)
                    self.logger.error(error_msg)
                
                finally:
                    self._active_cleanups.discard(task_id)
        
        cleanup_results['completed_at'] = datetime.now().isoformat()
        self.logger.info(f"🧹 Forced cleanup completed: {cleanup_results['tasks_successful']}/{cleanup_results['tasks_run']} successful")
        
        return cleanup_results
    
    def check_resource_health(self) -> Dict[str, Any]:
        """Check system resource health and recommend actions.
        
        Returns:
            Health check results with recommendations
        """
        health_check = {
            'overall_health': 'healthy',
            'issues': [],
            'warnings': [],
            'recommendations': [],
            'immediate_actions_needed': False
        }
        
        # Get current usage
        usage_stats = self.get_resource_usage()
        
        # Check memory
        memory_percent = usage_stats['system_stats']['memory']['used_percent']
        if memory_percent > self.memory_critical_threshold:
            health_check['overall_health'] = 'critical'
            health_check['immediate_actions_needed'] = True
            health_check['issues'].append(f"Critical memory usage: {memory_percent}%")
            health_check['recommendations'].append("Run immediate memory cleanup")
        elif memory_percent > self.memory_warning_threshold:
            health_check['overall_health'] = 'warning'
            health_check['warnings'].append(f"High memory usage: {memory_percent}%")
            health_check['recommendations'].append("Schedule memory cleanup")
        
        # Check storage
        storage_percent = usage_stats['system_stats']['storage']['used_percent']
        if storage_percent > self.storage_critical_threshold:
            health_check['overall_health'] = 'critical'
            health_check['immediate_actions_needed'] = True
            health_check['issues'].append(f"Critical storage usage: {storage_percent}%")
            health_check['recommendations'].append("Run immediate storage cleanup")
        elif storage_percent > self.storage_warning_threshold:
            health_check['overall_health'] = 'warning'
            health_check['warnings'].append(f"High storage usage: {storage_percent}%")
            health_check['recommendations'].append("Schedule storage cleanup")
        
        # Check process resources
        process_memory_mb = usage_stats['system_stats']['process']['memory_mb']
        if process_memory_mb > 4096:  # More than 4GB for process
            health_check['warnings'].append(f"High process memory usage: {process_memory_mb}MB")
            health_check['recommendations'].append("Run garbage collection")
        
        # Check open files
        open_files = usage_stats['system_stats']['process']['open_files']
        if open_files > 1000:
            health_check['warnings'].append(f"Many open files: {open_files}")
            health_check['recommendations'].append("Check for file handle leaks")
        
        # Check failed cleanup tasks
        failed_tasks = [
            task_id for task_id, task in self._cleanup_tasks.items()
            if task.failure_count > task.success_count and task.failure_count > 3
        ]
        if failed_tasks:
            health_check['warnings'].append(f"Cleanup tasks failing: {failed_tasks}")
            health_check['recommendations'].append("Review failing cleanup tasks")
        
        return health_check
    
    def emergency_cleanup(self) -> Dict[str, Any]:
        """Emergency cleanup when system resources are critically low.
        
        Returns:
            Emergency cleanup results
        """
        self.logger.warning("[EMOJI] Emergency cleanup initiated")
        
        emergency_results = {
            'started_at': datetime.now().isoformat(),
            'actions_taken': [],
            'space_freed_mb': 0,
            'memory_freed_mb': 0
        }
        
        try:
            # 1. Force garbage collection
            before_memory = psutil.Process().memory_info().rss / (1024**2)
            gc.collect()
            after_memory = psutil.Process().memory_info().rss / (1024**2)
            memory_freed = max(0, before_memory - after_memory)
            
            emergency_results['memory_freed_mb'] = memory_freed
            emergency_results['actions_taken'].append(f"Forced garbage collection (freed {memory_freed:.1f}MB)")
            
            # 2. Clean all temporary files immediately
            temp_cleaned = self._emergency_temp_cleanup()
            emergency_results['space_freed_mb'] += temp_cleaned
            emergency_results['actions_taken'].append(f"Emergency temp cleanup (freed {temp_cleaned:.1f}MB)")
            
            # 3. Force high-priority cleanups
            cleanup_results = self.force_cleanup(priority_level=CleanupPriority.HIGH)
            emergency_results['space_freed_mb'] += cleanup_results['space_freed_mb']
            emergency_results['actions_taken'].append(f"High-priority cleanup tasks completed")
            
            # 4. Clear system caches if possible
            try:
                if os.name == 'posix':  # Unix-like systems
                    subprocess.run(['sync'], check=False, timeout=10)
                emergency_results['actions_taken'].append("System cache sync completed")
            except Exception:
                pass
            
        except Exception as e:
            self.logger.error(f"Emergency cleanup error: {e}")
            emergency_results['actions_taken'].append(f"Error during emergency cleanup: {str(e)}")
        
        emergency_results['completed_at'] = datetime.now().isoformat()
        self.logger.warning(f"[EMOJI] Emergency cleanup completed: freed {emergency_results['space_freed_mb']:.1f}MB storage, {emergency_results['memory_freed_mb']:.1f}MB memory")
        
        return emergency_results
    
    def _update_resource_usage(self) -> None:
        """Update current resource usage tracking."""
        now = datetime.now()
        
        # Memory usage
        memory = psutil.virtual_memory()
        self._update_resource_entry(
            ResourceType.MEMORY,
            current=memory.used / (1024**3),  # GB
            maximum=memory.total / (1024**3),  # GB
            timestamp=now
        )
        
        # Storage usage
        disk = psutil.disk_usage(str(self.base_dir))
        self._update_resource_entry(
            ResourceType.STORAGE,
            current=disk.used / (1024**3),  # GB
            maximum=disk.total / (1024**3),  # GB
            timestamp=now
        )
        
        # Temp files usage
        temp_size = self._calculate_directory_size(self.base_dir / "temp")
        self._update_resource_entry(
            ResourceType.TEMP_FILES,
            current=temp_size / (1024**2),  # MB
            maximum=1024,  # 1GB max temp files
            timestamp=now
        )
    
    def _update_resource_entry(self, resource_type: ResourceType, current: float,
                             maximum: float, timestamp: datetime) -> None:
        """Update a specific resource usage entry."""
        usage_percent = (current / maximum) * 100 if maximum > 0 else 0
        
        # Calculate trend
        trend = "stable"
        if resource_type in self._resource_usage:
            old_usage = self._resource_usage[resource_type].usage_percent
            if usage_percent > old_usage + 5:
                trend = "increasing"
            elif usage_percent < old_usage - 5:
                trend = "decreasing"
        
        self._resource_usage[resource_type] = ResourceUsage(
            resource_type=resource_type,
            current_usage=current,
            max_usage=maximum,
            usage_percent=usage_percent,
            last_updated=timestamp,
            trend=trend
        )
    
    def _get_usage_status(self, usage_percent: float) -> str:
        """Get status string for usage percentage."""
        if usage_percent >= self.storage_critical_threshold:
            return "critical"
        elif usage_percent >= self.storage_warning_threshold:
            return "warning"
        elif usage_percent >= 50:
            return "moderate"
        else:
            return "good"
    
    def _get_cleanup_task_status(self) -> Dict[str, Any]:
        """Get status of all cleanup tasks."""
        task_status = {}
        
        for task_id, task in self._cleanup_tasks.items():
            task_status[task_id] = {
                'priority': task.priority.value,
                'resource_type': task.resource_type.value,
                'last_run': task.last_run.isoformat() if task.last_run else None,
                'success_count': task.success_count,
                'failure_count': task.failure_count,
                'is_active': task_id in self._active_cleanups,
                'target_path': task.target_path
            }
        
        return task_status
    
    def _get_target_stats(self, target_path: str) -> Dict[str, Any]:
        """Get statistics for a target path."""
        if not target_path or not Path(target_path).exists():
            return {'size_mb': 0, 'file_count': 0}
        
        try:
            path = Path(target_path)
            if path.is_file():
                return {
                    'size_mb': path.stat().st_size / (1024**2),
                    'file_count': 1
                }
            elif path.is_dir():
                total_size = 0
                file_count = 0
                
                for file_path in path.rglob('*'):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
                        file_count += 1
                
                return {
                    'size_mb': total_size / (1024**2),
                    'file_count': file_count
                }
        except Exception:
            pass
        
        return {'size_mb': 0, 'file_count': 0}
    
    def _calculate_directory_size(self, directory: Path) -> int:
        """Calculate total size of directory in bytes."""
        if not directory.exists():
            return 0
        
        total_size = 0
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception:
            pass
        
        return total_size
    
    # Cleanup task implementations
    
    def _cleanup_temp_files(self) -> bool:
        """Clean up temporary files."""
        try:
            files_removed = 0
            space_freed = 0
            cutoff_time = datetime.now() - timedelta(hours=1)
            
            for temp_dir in self.temp_dirs:
                if not temp_dir.exists():
                    continue
                
                for file_path in temp_dir.rglob('*'):
                    if file_path.is_file():
                        try:
                            file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                            if file_time < cutoff_time:
                                file_size = file_path.stat().st_size
                                file_path.unlink()
                                files_removed += 1
                                space_freed += file_size
                        except Exception:
                            continue
            
            self.logger.info(f"🧹 Temp cleanup: {files_removed} files, {space_freed/(1024**2):.1f}MB freed")
            return True
        
        except Exception as e:
            self.logger.error(f"Temp cleanup failed: {e}")
            return False
    
    def _cleanup_old_sessions(self) -> bool:
        """Clean up old session data."""
        try:
            sessions_removed = 0
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            sessions_dir = self.base_dir / "sessions"
            if not sessions_dir.exists():
                return True
            
            for session_dir in sessions_dir.iterdir():
                if session_dir.is_dir():
                    try:
                        dir_time = datetime.fromtimestamp(session_dir.stat().st_mtime)
                        if dir_time < cutoff_time:
                            shutil.rmtree(session_dir)
                            sessions_removed += 1
                    except Exception:
                        continue
            
            self.logger.info(f"🧹 Session cleanup: {sessions_removed} sessions removed")
            return True
        
        except Exception as e:
            self.logger.error(f"Session cleanup failed: {e}")
            return False
    
    def _cleanup_cache(self) -> bool:
        """Clean up cache files."""
        try:
            cache_dir = self.base_dir / "cache"
            if not cache_dir.exists():
                return True
            
            files_removed = 0
            cutoff_time = datetime.now() - timedelta(days=7)
            
            for cache_file in cache_dir.rglob('*'):
                if cache_file.is_file():
                    try:
                        file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                        if file_time < cutoff_time:
                            cache_file.unlink()
                            files_removed += 1
                    except Exception:
                        continue
            
            self.logger.info(f"🧹 Cache cleanup: {files_removed} files removed")
            return True
        
        except Exception as e:
            self.logger.error(f"Cache cleanup failed: {e}")
            return False
    
    def _cleanup_logs(self) -> bool:
        """Clean up old log files."""
        try:
            logs_dir = self.base_dir / "logs"
            if not logs_dir.exists():
                return True
            
            files_removed = 0
            cutoff_time = datetime.now() - timedelta(days=30)
            
            for log_file in logs_dir.glob('*.log*'):
                try:
                    file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if file_time < cutoff_time:
                        log_file.unlink()
                        files_removed += 1
                except Exception:
                    continue
            
            self.logger.info(f"🧹 Log cleanup: {files_removed} files removed")
            return True
        
        except Exception as e:
            self.logger.error(f"Log cleanup failed: {e}")
            return False
    
    def _force_garbage_collection(self) -> bool:
        """Force Python garbage collection."""
        try:
            before = psutil.Process().memory_info().rss
            collected = gc.collect()
            after = psutil.Process().memory_info().rss
            
            freed_mb = (before - after) / (1024**2)
            self.logger.info(f"🧹 Garbage collection: {collected} objects, {freed_mb:.1f}MB freed")
            return True
        
        except Exception as e:
            self.logger.error(f"Garbage collection failed: {e}")
            return False
    
    def _emergency_temp_cleanup(self) -> float:
        """Emergency cleanup of all temporary files."""
        space_freed = 0
        
        try:
            for temp_dir in self.temp_dirs:
                if temp_dir.exists():
                    before_size = self._calculate_directory_size(temp_dir)
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    temp_dir.mkdir(exist_ok=True)
                    space_freed += before_size / (1024**2)  # Convert to MB
        
        except Exception as e:
            self.logger.error(f"Emergency temp cleanup error: {e}")
        
        return space_freed
    
    def _start_monitoring_threads(self) -> None:
        """Start background monitoring and cleanup threads."""
        def resource_monitor():
            while True:
                try:
                    time.sleep(60)  # Update every minute
                    with self._resource_lock:
                        self._update_resource_usage()
                    
                    # Check if emergency action needed
                    health = self.check_resource_health()
                    if health['immediate_actions_needed']:
                        self.emergency_cleanup()
                
                except Exception as e:
                    self.logger.error(f"Resource monitor error: {e}")
        
        def scheduled_cleanup():
            while True:
                try:
                    time.sleep(300)  # Check every 5 minutes
                    
                    # Run scheduled cleanups
                    now = datetime.now()
                    for task_id, task in self._cleanup_tasks.items():
                        if task_id in self._active_cleanups:
                            continue
                        
                        interval = self.cleanup_intervals.get(task.priority, timedelta(hours=1))
                        if task.last_run is None or (now - task.last_run) >= interval:
                            try:
                                self._active_cleanups.add(task_id)
                                success = task.cleanup_function()
                                
                                if success:
                                    task.success_count += 1
                                else:
                                    task.failure_count += 1
                                
                                task.last_run = now
                            
                            except Exception as e:
                                task.failure_count += 1
                                self.logger.error(f"Scheduled cleanup error {task_id}: {e}")
                            
                            finally:
                                self._active_cleanups.discard(task_id)
                
                except Exception as e:
                    self.logger.error(f"Scheduled cleanup error: {e}")
        
        # Start monitoring threads
        monitor_thread = threading.Thread(target=resource_monitor, daemon=True)
        cleanup_thread = threading.Thread(target=scheduled_cleanup, daemon=True)
        
        monitor_thread.start()
        cleanup_thread.start()
        
        self.logger.info("[EMOJI] Started monitoring and cleanup threads")


def main():
    """Test the memory cleanup manager."""
    print("🧪 Testing Memory Cleanup Manager")
    print("=" * 50)
    
    manager = MemoryCleanupManager()
    
    # Get resource usage
    usage = manager.get_resource_usage()
    print(f"Memory usage: {usage['system_stats']['memory']['used_percent']:.1f}%")
    print(f"Storage usage: {usage['system_stats']['storage']['used_percent']:.1f}%")
    
    # Check health
    health = manager.check_resource_health()
    print(f"System health: {health['overall_health']}")
    
    # Test forced cleanup
    cleanup_results = manager.force_cleanup(priority_level=CleanupPriority.HIGH)
    print(f"Cleanup completed: {cleanup_results['tasks_successful']}/{cleanup_results['tasks_run']} successful")
    
    print("Memory Cleanup Manager ready for production!")


if __name__ == "__main__":
    main()