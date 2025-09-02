#!/usr/bin/env python3
"""
Concurrent Session Manager - Multi-User Session Isolation
Manages multiple user sessions with proper isolation, locking, and resource management
Designed for production frontend with multiple concurrent users
"""

import os
import sys
import json
import time
import threading
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import fcntl  # For file locking on Unix systems

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from pipeline_exceptions import (
    SessionException, SessionNotFoundError, SessionExpiredError,
    SessionConcurrencyError, SessionStateError, InsufficientMemoryError
)


class SessionState(Enum):
    """Session states for tracking lifecycle."""
    CREATED = "created"
    CONFIGURED = "configured" 
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    CLEANUP = "cleanup"


@dataclass
class SessionInfo:
    """Session information structure."""
    session_id: str
    user_id: Optional[str]
    state: SessionState
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    current_stage: Optional[str]
    processing_thread_id: Optional[int]
    resource_usage: Dict[str, Any]
    metadata_path: str
    isolation_directory: str


class ConcurrentSessionManager:
    """Manages multiple concurrent user sessions with proper isolation."""
    
    def __init__(self, base_dir: str = None, max_concurrent_sessions: int = 10):
        """Initialize concurrent session manager.
        
        Args:
            base_dir: Base directory for the pipeline
            max_concurrent_sessions: Maximum number of concurrent sessions
        """
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        self.max_concurrent_sessions = max_concurrent_sessions
        self.sessions_dir = self.base_dir / "sessions"
        self.sessions_dir.mkdir(exist_ok=True)
        
        # Thread-safe session storage
        self._sessions_lock = threading.RLock()
        self._active_sessions: Dict[str, SessionInfo] = {}
        self._processing_sessions: Set[str] = set()
        
        # Session expiry settings
        self.default_session_timeout = timedelta(hours=2)
        self.processing_session_timeout = timedelta(hours=1)
        self.cleanup_interval = timedelta(minutes=15)
        
        # Resource monitoring (optimized for script generation)
        self._resource_lock = threading.Lock()
        self._total_memory_usage = 0
        self._max_memory_per_session = 64 * 1024 * 1024  # 64MB per session (lightweight for script generation)
        self._max_total_memory = 1 * 1024 * 1024 * 1024  # 1GB total (allows many concurrent sessions)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Start cleanup thread
        self._start_cleanup_thread()
        
        # Load existing sessions
        self._load_existing_sessions()
        
        self.logger.info(f"[EMOJI] Concurrent Session Manager initialized - Max sessions: {max_concurrent_sessions}")
        self.logger.info(f"Storage Memory limits: {self._max_memory_per_session // (1024*1024)}MB per session, {self._max_total_memory // (1024*1024)}MB total")
    
    def create_session(self, user_id: Optional[str] = None, 
                      session_timeout: Optional[timedelta] = None) -> str:
        """Create a new isolated session.
        
        Args:
            user_id: Optional user identifier
            session_timeout: Custom session timeout
            
        Returns:
            New session ID
            
        Raises:
            InsufficientMemoryError: If system resources are exhausted
        """
        with self._sessions_lock:
            # Check session limits
            if len(self._active_sessions) >= self.max_concurrent_sessions:
                # Try to clean up expired sessions first
                self._cleanup_expired_sessions()
                
                if len(self._active_sessions) >= self.max_concurrent_sessions:
                    raise InsufficientMemoryError(
                        operation="create_session",
                        required_mb=self._max_memory_per_session // (1024 * 1024)
                    )
            
            # Generate unique session ID
            session_id = str(uuid.uuid4())
            while session_id in self._active_sessions:
                session_id = str(uuid.uuid4())
            
            # Create session isolation directory
            session_dir = self.sessions_dir / session_id
            session_dir.mkdir(exist_ok=True)
            
            # Set up session subdirectories
            for subdir in ['metadata', 'assets', 'processing', 'output', 'temp']:
                (session_dir / subdir).mkdir(exist_ok=True)
            
            # Create session info
            now = datetime.now()
            timeout = session_timeout or self.default_session_timeout
            
            session_info = SessionInfo(
                session_id=session_id,
                user_id=user_id,
                state=SessionState.CREATED,
                created_at=now,
                last_activity=now,
                expires_at=now + timeout,
                current_stage=None,
                processing_thread_id=None,
                resource_usage={'memory_mb': 0, 'disk_mb': 0},
                metadata_path=str(session_dir / 'metadata' / 'session_metadata.json'),
                isolation_directory=str(session_dir)
            )
            
            # Save session info
            self._save_session_info(session_info)
            self._active_sessions[session_id] = session_info
            
            self.logger.info(f"[SUCCESS] Session created: {session_id} (user: {user_id or 'anonymous'})")
            return session_id
    
    def get_session(self, session_id: str) -> SessionInfo:
        """Get session information.
        
        Args:
            session_id: Session ID to retrieve
            
        Returns:
            Session information
            
        Raises:
            SessionNotFoundError: If session doesn't exist
            SessionExpiredError: If session has expired
        """
        with self._sessions_lock:
            if session_id not in self._active_sessions:
                raise SessionNotFoundError(session_id)
            
            session_info = self._active_sessions[session_id]
            
            # Check if session has expired
            if datetime.now() > session_info.expires_at:
                self._expire_session(session_id)
                raise SessionExpiredError(session_id, session_info.expires_at.isoformat())
            
            return session_info
    
    def update_session_activity(self, session_id: str) -> None:
        """Update session last activity timestamp.
        
        Args:
            session_id: Session ID to update
        """
        with self._sessions_lock:
            if session_id in self._active_sessions:
                session_info = self._active_sessions[session_id]
                session_info.last_activity = datetime.now()
                
                # Extend expiry if processing
                if session_info.state == SessionState.PROCESSING:
                    session_info.expires_at = datetime.now() + self.processing_session_timeout
                
                self._save_session_info(session_info)
    
    def start_session_processing(self, session_id: str, stage_name: str) -> None:
        """Mark session as processing a specific stage.
        
        Args:
            session_id: Session ID
            stage_name: Name of the stage being processed
            
        Raises:
            SessionConcurrencyError: If session is already processing
        """
        with self._sessions_lock:
            session_info = self.get_session(session_id)
            
            # Check if session is already processing
            if session_id in self._processing_sessions:
                raise SessionConcurrencyError(session_id, session_info.current_stage or "unknown")
            
            # Update session state
            session_info.state = SessionState.PROCESSING
            session_info.current_stage = stage_name
            session_info.processing_thread_id = threading.get_ident()
            session_info.last_activity = datetime.now()
            session_info.expires_at = datetime.now() + self.processing_session_timeout
            
            self._processing_sessions.add(session_id)
            self._save_session_info(session_info)
            
            self.logger.info(f"[EMOJI] Session {session_id} started processing: {stage_name}")
    
    def complete_session_processing(self, session_id: str, success: bool = True) -> None:
        """Mark session processing as complete.
        
        Args:
            session_id: Session ID
            success: Whether processing was successful
        """
        with self._sessions_lock:
            if session_id in self._active_sessions:
                session_info = self._active_sessions[session_id]
                session_info.state = SessionState.COMPLETED if success else SessionState.FAILED
                session_info.current_stage = None
                session_info.processing_thread_id = None
                session_info.last_activity = datetime.now()
                
                self._processing_sessions.discard(session_id)
                self._save_session_info(session_info)
                
                status = "completed" if success else "failed"
                self.logger.info(f"[SUCCESS] Session {session_id} processing {status}")
    
    def allocate_session_resources(self, session_id: str, memory_mb: int) -> bool:
        """Allocate resources for a session.
        
        Args:
            session_id: Session ID
            memory_mb: Memory to allocate in MB
            
        Returns:
            True if allocation successful, False otherwise
        """
        with self._resource_lock:
            session_info = self.get_session(session_id)
            
            # Check if we can allocate the requested memory
            current_usage = session_info.resource_usage.get('memory_mb', 0)
            additional_memory = memory_mb - current_usage
            
            if (self._total_memory_usage + additional_memory) > (self._max_total_memory // (1024 * 1024)):
                return False
            
            if memory_mb > (self._max_memory_per_session // (1024 * 1024)):
                return False
            
            # Allocate resources
            self._total_memory_usage += additional_memory
            session_info.resource_usage['memory_mb'] = memory_mb
            
            self._save_session_info(session_info)
            return True
    
    def deallocate_session_resources(self, session_id: str) -> None:
        """Deallocate resources for a session.
        
        Args:
            session_id: Session ID
        """
        with self._resource_lock:
            if session_id in self._active_sessions:
                session_info = self._active_sessions[session_id]
                memory_mb = session_info.resource_usage.get('memory_mb', 0)
                
                self._total_memory_usage -= memory_mb
                self._total_memory_usage = max(0, self._total_memory_usage)  # Ensure non-negative
                
                session_info.resource_usage['memory_mb'] = 0
                self._save_session_info(session_info)
    
    def get_session_isolation_path(self, session_id: str, subpath: str = "") -> Path:
        """Get isolated path for session data.
        
        Args:
            session_id: Session ID
            subpath: Sub-path within session directory
            
        Returns:
            Isolated path for the session
        """
        session_info = self.get_session(session_id)
        base_path = Path(session_info.isolation_directory)
        
        if subpath:
            return base_path / subpath
        else:
            return base_path
    
    def list_active_sessions(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List active sessions, optionally filtered by user.
        
        Args:
            user_id: Optional user ID filter
            
        Returns:
            List of session information
        """
        with self._sessions_lock:
            sessions = []
            for session_id, session_info in self._active_sessions.items():
                if user_id is None or session_info.user_id == user_id:
                    sessions.append({
                        'session_id': session_id,
                        'user_id': session_info.user_id,
                        'state': session_info.state.value,
                        'created_at': session_info.created_at.isoformat(),
                        'last_activity': session_info.last_activity.isoformat(),
                        'current_stage': session_info.current_stage,
                        'is_processing': session_id in self._processing_sessions,
                        'resource_usage': session_info.resource_usage
                    })
            return sessions
    
    def cleanup_session(self, session_id: str, force: bool = False) -> bool:
        """Clean up a specific session.
        
        Args:
            session_id: Session ID to clean up
            force: Force cleanup even if session is processing
            
        Returns:
            True if cleanup successful, False otherwise
        """
        with self._sessions_lock:
            if session_id not in self._active_sessions:
                return False
            
            session_info = self._active_sessions[session_id]
            
            # Check if session is processing and force is not set
            if session_id in self._processing_sessions and not force:
                return False
            
            # Deallocate resources
            self.deallocate_session_resources(session_id)
            
            # Remove from processing set
            self._processing_sessions.discard(session_id)
            
            # Clean up session directory
            session_dir = Path(session_info.isolation_directory)
            if session_dir.exists():
                try:
                    # Remove session files
                    import shutil
                    shutil.rmtree(session_dir)
                except OSError as e:
                    self.logger.warning(f"Failed to remove session directory {session_dir}: {e}")
            
            # Remove from active sessions
            del self._active_sessions[session_id]
            
            self.logger.info(f"🧹 Session {session_id} cleaned up")
            return True
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status information.
        
        Returns:
            System status including resource usage and session counts
        """
        with self._sessions_lock, self._resource_lock:
            active_count = len(self._active_sessions)
            processing_count = len(self._processing_sessions)
            
            # Calculate resource usage
            total_memory_usage = sum(
                session.resource_usage.get('memory_mb', 0) 
                for session in self._active_sessions.values()
            )
            
            # Session states breakdown
            state_counts = {}
            for state in SessionState:
                state_counts[state.value] = sum(
                    1 for session in self._active_sessions.values() 
                    if session.state == state
                )
            
            return {
                'active_sessions': active_count,
                'processing_sessions': processing_count,
                'max_sessions': self.max_concurrent_sessions,
                'session_utilization': (active_count / self.max_concurrent_sessions) * 100,
                'memory_usage': {
                    'total_mb': total_memory_usage,
                    'max_total_mb': self._max_total_memory // (1024 * 1024),
                    'max_per_session_mb': self._max_memory_per_session // (1024 * 1024),
                    'utilization_percent': (total_memory_usage / (self._max_total_memory // (1024 * 1024))) * 100
                },
                'session_states': state_counts,
                'last_cleanup': getattr(self, '_last_cleanup_time', None),
                'system_healthy': active_count < self.max_concurrent_sessions and total_memory_usage < (self._max_total_memory // (1024 * 1024)) * 0.8
            }
    
    def _save_session_info(self, session_info: SessionInfo) -> None:
        """Save session info to disk."""
        try:
            session_file = Path(session_info.metadata_path)
            session_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to serializable format
            session_data = asdict(session_info)
            session_data['state'] = session_info.state.value
            session_data['created_at'] = session_info.created_at.isoformat()
            session_data['last_activity'] = session_info.last_activity.isoformat()
            session_data['expires_at'] = session_info.expires_at.isoformat()
            
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save session info for {session_info.session_id}: {e}")
    
    def _load_existing_sessions(self) -> None:
        """Load existing sessions from disk."""
        try:
            for session_dir in self.sessions_dir.iterdir():
                if session_dir.is_dir():
                    session_file = session_dir / 'metadata' / 'session_metadata.json'
                    if session_file.exists():
                        try:
                            with open(session_file, 'r') as f:
                                session_data = json.load(f)
                            
                            # Convert back to SessionInfo
                            session_info = SessionInfo(
                                session_id=session_data['session_id'],
                                user_id=session_data.get('user_id'),
                                state=SessionState(session_data['state']),
                                created_at=datetime.fromisoformat(session_data['created_at']),
                                last_activity=datetime.fromisoformat(session_data['last_activity']),
                                expires_at=datetime.fromisoformat(session_data['expires_at']),
                                current_stage=session_data.get('current_stage'),
                                processing_thread_id=session_data.get('processing_thread_id'),
                                resource_usage=session_data.get('resource_usage', {}),
                                metadata_path=session_data['metadata_path'],
                                isolation_directory=session_data['isolation_directory']
                            )
                            
                            # Check if session is still valid
                            if datetime.now() <= session_info.expires_at:
                                self._active_sessions[session_info.session_id] = session_info
                                self.logger.info(f"[EMOJI] Restored session: {session_info.session_id}")
                            else:
                                # Session expired, clean it up
                                self.logger.info(f"🧹 Expired session found: {session_info.session_id}")
                                
                        except Exception as e:
                            self.logger.warning(f"Failed to load session from {session_file}: {e}")
        except Exception as e:
            self.logger.error(f"Failed to load existing sessions: {e}")
    
    def _expire_session(self, session_id: str) -> None:
        """Mark session as expired and prepare for cleanup."""
        if session_id in self._active_sessions:
            session_info = self._active_sessions[session_id]
            session_info.state = SessionState.EXPIRED
            self._save_session_info(session_info)
    
    def _cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions."""
        now = datetime.now()
        expired_sessions = []
        
        for session_id, session_info in self._active_sessions.items():
            if now > session_info.expires_at:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.cleanup_session(session_id, force=True)
            self.logger.info(f"🧹 Cleaned up expired session: {session_id}")
    
    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(self.cleanup_interval.total_seconds())
                    self._cleanup_expired_sessions()
                    self._last_cleanup_time = datetime.now().isoformat()
                except Exception as e:
                    self.logger.error(f"Error in cleanup thread: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        self.logger.info("🧹 Started session cleanup thread")


def main():
    """Test the concurrent session manager."""
    print("🧪 Testing Concurrent Session Manager")
    print("=" * 50)
    
    manager = ConcurrentSessionManager(max_concurrent_sessions=3)
    
    # Create test sessions
    session1 = manager.create_session(user_id="user1")
    session2 = manager.create_session(user_id="user2")
    
    print(f"Created sessions: {session1}, {session2}")
    
    # Test session processing
    manager.start_session_processing(session1, "script_generation")
    print(f"Started processing: {session1}")
    
    # Get system status
    status = manager.get_system_status()
    print(f"System status: {status['active_sessions']} active, {status['processing_sessions']} processing")
    
    # Complete processing
    manager.complete_session_processing(session1)
    print(f"Completed processing: {session1}")
    
    # List sessions
    sessions = manager.list_active_sessions()
    print(f"Active sessions: {len(sessions)}")
    
    # Clean up
    manager.cleanup_session(session1)
    manager.cleanup_session(session2)
    print("Cleaned up sessions")


if __name__ == "__main__":
    main()