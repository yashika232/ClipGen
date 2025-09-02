#!/usr/bin/env python3
"""
Real-time Progress Manager - Live Updates for Frontend
Provides WebSocket-compatible real-time progress updates for video synthesis pipeline
Supports Server-Sent Events (SSE) and WebSocket protocols for live frontend communication
"""

import os
import sys
import json
import time
import asyncio
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import queue
import uuid

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from pipeline_exceptions import SessionNotFoundError, SessionStateError


class ProgressEventType(Enum):
    """Types of progress events."""
    STAGE_STARTED = "stage_started"
    STAGE_PROGRESS = "stage_progress"
    STAGE_COMPLETED = "stage_completed"
    STAGE_FAILED = "stage_failed"
    PIPELINE_COMPLETED = "pipeline_completed"
    PIPELINE_FAILED = "pipeline_failed"
    STATUS_UPDATE = "status_update"
    ERROR_OCCURRED = "error_occurred"
    WARNING_ISSUED = "warning_issued"


@dataclass
class ProgressEvent:
    """Progress event structure."""
    event_id: str
    session_id: str
    event_type: ProgressEventType
    timestamp: datetime
    stage_name: Optional[str]
    progress_percent: float
    message: str
    details: Dict[str, Any]
    estimated_completion: Optional[datetime]


class ProgressSubscription:
    """Represents a client subscription to progress updates."""
    
    def __init__(self, subscription_id: str, session_id: str, 
                 callback: Callable[[ProgressEvent], None]):
        self.subscription_id = subscription_id
        self.session_id = session_id
        self.callback = callback
        self.created_at = datetime.now()
        self.last_event = None
        self.is_active = True


class RealtimeProgressManager:
    """Manages real-time progress updates for multiple sessions."""
    
    def __init__(self, base_dir: str = None):
        """Initialize real-time progress manager.
        
        Args:
            base_dir: Base directory for the pipeline
        """
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        # Thread-safe event management
        self._events_lock = threading.RLock()
        self._subscriptions_lock = threading.RLock()
        
        # Event storage and subscriptions
        self._session_events: Dict[str, List[ProgressEvent]] = {}
        self._subscriptions: Dict[str, ProgressSubscription] = {}
        self._session_subscribers: Dict[str, Set[str]] = {}
        
        # Progress tracking
        self._session_progress: Dict[str, Dict[str, Any]] = {}
        self._stage_timings: Dict[str, Dict[str, float]] = {}
        
        # Pipeline stage configuration
        self.pipeline_stages = [
            "script_generation",
            "voice_cloning", 
            "face_processing",
            "video_generation",
            "video_enhancement",
            "background_animation",
            "final_assembly"
        ]
        
        # Stage weight for progress calculation (adds to 100%)
        self.stage_weights = {
            "script_generation": 15,
            "voice_cloning": 10,
            "face_processing": 10,
            "video_generation": 25,
            "video_enhancement": 20,
            "background_animation": 15,
            "final_assembly": 5
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Event cleanup configuration
        self.max_events_per_session = 1000
        self.event_retention_hours = 24
        
        # Start cleanup thread
        self._start_cleanup_thread()
        
        self.logger.info("Server Real-time Progress Manager initialized")
    
    def subscribe_to_session(self, session_id: str, 
                           callback: Callable[[ProgressEvent], None]) -> str:
        """Subscribe to progress updates for a session.
        
        Args:
            session_id: Session ID to subscribe to
            callback: Function to call when events occur
            
        Returns:
            Subscription ID for managing the subscription
        """
        subscription_id = str(uuid.uuid4())
        
        with self._subscriptions_lock:
            subscription = ProgressSubscription(
                subscription_id=subscription_id,
                session_id=session_id,
                callback=callback
            )
            
            self._subscriptions[subscription_id] = subscription
            
            if session_id not in self._session_subscribers:
                self._session_subscribers[session_id] = set()
            self._session_subscribers[session_id].add(subscription_id)
        
        self.logger.info(f"[EMOJI] New subscription: {subscription_id} for session {session_id}")
        
        # Send recent events to new subscriber
        self._send_recent_events(subscription_id, session_id)
        
        return subscription_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from progress updates.
        
        Args:
            subscription_id: Subscription ID to remove
            
        Returns:
            True if subscription was found and removed
        """
        with self._subscriptions_lock:
            if subscription_id not in self._subscriptions:
                return False
            
            subscription = self._subscriptions[subscription_id]
            session_id = subscription.session_id
            
            # Remove from subscriptions
            del self._subscriptions[subscription_id]
            
            # Remove from session subscribers
            if session_id in self._session_subscribers:
                self._session_subscribers[session_id].discard(subscription_id)
                if not self._session_subscribers[session_id]:
                    del self._session_subscribers[session_id]
        
        self.logger.info(f"[EMOJI] Unsubscribed: {subscription_id}")
        return True
    
    def start_stage(self, session_id: str, stage_name: str, 
                   estimated_duration: Optional[float] = None) -> None:
        """Signal that a pipeline stage has started.
        
        Args:
            session_id: Session ID
            stage_name: Name of the stage
            estimated_duration: Estimated duration in seconds
        """
        # Initialize session progress if needed
        if session_id not in self._session_progress:
            self._session_progress[session_id] = {
                'current_stage': None,
                'overall_progress': 0.0,
                'stage_progress': 0.0,
                'started_at': datetime.now(),
                'estimated_completion': None,
                'stages_completed': [],
                'current_stage_start': None
            }
        
        # Update progress tracking
        progress_info = self._session_progress[session_id]
        progress_info['current_stage'] = stage_name
        progress_info['stage_progress'] = 0.0
        progress_info['current_stage_start'] = datetime.now()
        
        # Calculate overall progress
        completed_stages = progress_info['stages_completed']
        completed_weight = sum(self.stage_weights.get(stage, 0) for stage in completed_stages)
        progress_info['overall_progress'] = completed_weight
        
        # Estimate completion time
        if estimated_duration:
            progress_info['estimated_completion'] = datetime.now() + timedelta(seconds=estimated_duration)
        
        # Create and emit event
        event = ProgressEvent(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            event_type=ProgressEventType.STAGE_STARTED,
            timestamp=datetime.now(),
            stage_name=stage_name,
            progress_percent=progress_info['overall_progress'],
            message=f"Started {stage_name.replace('_', ' ').title()}",
            details={
                'stage_index': self.pipeline_stages.index(stage_name) if stage_name in self.pipeline_stages else -1,
                'total_stages': len(self.pipeline_stages),
                'estimated_duration': estimated_duration
            },
            estimated_completion=progress_info['estimated_completion']
        )
        
        self._emit_event(event)
    
    def update_stage_progress(self, session_id: str, stage_name: str, 
                            progress_percent: float, message: str = "",
                            details: Optional[Dict[str, Any]] = None) -> None:
        """Update progress within a stage.
        
        Args:
            session_id: Session ID
            stage_name: Name of the current stage
            progress_percent: Progress within the stage (0-100)
            message: Optional progress message
            details: Additional progress details
        """
        if session_id not in self._session_progress:
            self.logger.warning(f"No progress tracking for session {session_id}")
            return
        
        progress_info = self._session_progress[session_id]
        progress_info['stage_progress'] = max(0, min(100, progress_percent))
        
        # Calculate overall progress
        completed_stages = progress_info['stages_completed']
        completed_weight = sum(self.stage_weights.get(stage, 0) for stage in completed_stages)
        current_stage_weight = self.stage_weights.get(stage_name, 0)
        stage_contribution = (progress_percent / 100) * current_stage_weight
        
        progress_info['overall_progress'] = completed_weight + stage_contribution
        
        # Update estimated completion
        if progress_info['current_stage_start'] and progress_percent > 0:
            elapsed = (datetime.now() - progress_info['current_stage_start']).total_seconds()
            estimated_stage_duration = (elapsed / progress_percent) * 100
            remaining_time = (100 - progress_percent) / 100 * estimated_stage_duration
            
            # Add estimated time for remaining stages
            remaining_stages = [s for s in self.pipeline_stages 
                              if s not in completed_stages and s != stage_name]
            remaining_stages_time = len(remaining_stages) * 60  # Rough estimate
            
            progress_info['estimated_completion'] = datetime.now() + timedelta(
                seconds=remaining_time + remaining_stages_time
            )
        
        # Create and emit event
        event = ProgressEvent(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            event_type=ProgressEventType.STAGE_PROGRESS,
            timestamp=datetime.now(),
            stage_name=stage_name,
            progress_percent=progress_info['overall_progress'],
            message=message or f"{stage_name.replace('_', ' ').title()}: {progress_percent:.1f}%",
            details={
                'stage_progress': progress_percent,
                'overall_progress': progress_info['overall_progress'],
                **(details or {})
            },
            estimated_completion=progress_info['estimated_completion']
        )
        
        self._emit_event(event)
    
    def complete_stage(self, session_id: str, stage_name: str, 
                      success: bool = True, message: str = "",
                      output_info: Optional[Dict[str, Any]] = None) -> None:
        """Signal that a pipeline stage has completed.
        
        Args:
            session_id: Session ID
            stage_name: Name of the completed stage
            success: Whether the stage completed successfully
            message: Completion message
            output_info: Information about stage outputs
        """
        if session_id not in self._session_progress:
            return
        
        progress_info = self._session_progress[session_id]
        
        if success:
            progress_info['stages_completed'].append(stage_name)
            event_type = ProgressEventType.STAGE_COMPLETED
            
            # Calculate updated overall progress
            completed_weight = sum(self.stage_weights.get(stage, 0) 
                                 for stage in progress_info['stages_completed'])
            progress_info['overall_progress'] = completed_weight
            
            default_message = f"Completed {stage_name.replace('_', ' ').title()}"
        else:
            event_type = ProgressEventType.STAGE_FAILED
            default_message = f"Failed {stage_name.replace('_', ' ').title()}"
        
        progress_info['current_stage'] = None
        progress_info['stage_progress'] = 0.0
        
        # Create and emit event
        event = ProgressEvent(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            event_type=event_type,
            timestamp=datetime.now(),
            stage_name=stage_name,
            progress_percent=progress_info['overall_progress'],
            message=message or default_message,
            details={
                'success': success,
                'output_info': output_info or {},
                'stages_completed': len(progress_info['stages_completed']),
                'total_stages': len(self.pipeline_stages)
            },
            estimated_completion=progress_info['estimated_completion']
        )
        
        self._emit_event(event)
    
    def complete_pipeline(self, session_id: str, success: bool = True,
                         final_output: Optional[str] = None,
                         processing_summary: Optional[Dict[str, Any]] = None) -> None:
        """Signal that the entire pipeline has completed.
        
        Args:
            session_id: Session ID
            success: Whether the pipeline completed successfully
            final_output: Path to final output file
            processing_summary: Summary of processing statistics
        """
        if session_id not in self._session_progress:
            return
        
        progress_info = self._session_progress[session_id]
        progress_info['overall_progress'] = 100.0 if success else progress_info['overall_progress']
        progress_info['completed_at'] = datetime.now()
        
        # Calculate total processing time
        processing_time = (datetime.now() - progress_info['started_at']).total_seconds()
        
        event_type = ProgressEventType.PIPELINE_COMPLETED if success else ProgressEventType.PIPELINE_FAILED
        message = "Pipeline completed successfully!" if success else "Pipeline failed"
        
        # Create and emit event
        event = ProgressEvent(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            event_type=event_type,
            timestamp=datetime.now(),
            stage_name=None,
            progress_percent=progress_info['overall_progress'],
            message=message,
            details={
                'success': success,
                'final_output': final_output,
                'processing_time_seconds': processing_time,
                'processing_summary': processing_summary or {},
                'total_stages_completed': len(progress_info['stages_completed'])
            },
            estimated_completion=None
        )
        
        self._emit_event(event)
    
    def emit_warning(self, session_id: str, warning_message: str,
                    details: Optional[Dict[str, Any]] = None) -> None:
        """Emit a warning event.
        
        Args:
            session_id: Session ID
            warning_message: Warning message
            details: Additional warning details
        """
        current_progress = self._session_progress.get(session_id, {}).get('overall_progress', 0.0)
        
        event = ProgressEvent(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            event_type=ProgressEventType.WARNING_ISSUED,
            timestamp=datetime.now(),
            stage_name=self._session_progress.get(session_id, {}).get('current_stage'),
            progress_percent=current_progress,
            message=warning_message,
            details=details or {},
            estimated_completion=self._session_progress.get(session_id, {}).get('estimated_completion')
        )
        
        self._emit_event(event)
    
    def emit_error(self, session_id: str, error_message: str,
                  error_code: Optional[str] = None,
                  details: Optional[Dict[str, Any]] = None) -> None:
        """Emit an error event.
        
        Args:
            session_id: Session ID
            error_message: Error message
            error_code: Optional error code
            details: Additional error details
        """
        current_progress = self._session_progress.get(session_id, {}).get('overall_progress', 0.0)
        
        event = ProgressEvent(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            event_type=ProgressEventType.ERROR_OCCURRED,
            timestamp=datetime.now(),
            stage_name=self._session_progress.get(session_id, {}).get('current_stage'),
            progress_percent=current_progress,
            message=error_message,
            details={
                'error_code': error_code,
                **(details or {})
            },
            estimated_completion=None
        )
        
        self._emit_event(event)
    
    def get_session_progress(self, session_id: str) -> Dict[str, Any]:
        """Get current progress for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Current progress information
        """
        if session_id not in self._session_progress:
            return {
                'session_id': session_id,
                'overall_progress': 0.0,
                'current_stage': None,
                'stage_progress': 0.0,
                'started_at': None,
                'estimated_completion': None,
                'stages_completed': [],
                'is_active': False
            }
        
        progress_info = self._session_progress[session_id]
        return {
            'session_id': session_id,
            'overall_progress': progress_info['overall_progress'],
            'current_stage': progress_info['current_stage'],
            'stage_progress': progress_info['stage_progress'],
            'started_at': progress_info['started_at'].isoformat() if progress_info['started_at'] else None,
            'estimated_completion': progress_info['estimated_completion'].isoformat() if progress_info['estimated_completion'] else None,
            'stages_completed': progress_info['stages_completed'],
            'is_active': progress_info['current_stage'] is not None,
            'completion_percentage': progress_info['overall_progress']
        }
    
    def get_session_events(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent events for a session.
        
        Args:
            session_id: Session ID
            limit: Maximum number of events to return
            
        Returns:
            List of recent events
        """
        with self._events_lock:
            events = self._session_events.get(session_id, [])
            recent_events = events[-limit:] if len(events) > limit else events
            
            return [self._event_to_dict(event) for event in recent_events]
    
    def _emit_event(self, event: ProgressEvent) -> None:
        """Emit an event to all subscribers.
        
        Args:
            event: Progress event to emit
        """
        session_id = event.session_id
        
        # Store event
        with self._events_lock:
            if session_id not in self._session_events:
                self._session_events[session_id] = []
            
            self._session_events[session_id].append(event)
            
            # Trim old events
            if len(self._session_events[session_id]) > self.max_events_per_session:
                self._session_events[session_id] = self._session_events[session_id][-self.max_events_per_session:]
        
        # Send to subscribers
        with self._subscriptions_lock:
            subscribers = self._session_subscribers.get(session_id, set())
            
            for subscription_id in subscribers.copy():  # Copy to avoid modification during iteration
                if subscription_id in self._subscriptions:
                    subscription = self._subscriptions[subscription_id]
                    try:
                        subscription.callback(event)
                        subscription.last_event = event
                    except Exception as e:
                        self.logger.error(f"Error sending event to subscriber {subscription_id}: {e}")
                        # Remove failed subscription
                        self._subscriptions.pop(subscription_id, None)
                        subscribers.discard(subscription_id)
    
    def _send_recent_events(self, subscription_id: str, session_id: str) -> None:
        """Send recent events to a new subscriber.
        
        Args:
            subscription_id: Subscription ID
            session_id: Session ID
        """
        recent_events = self.get_session_events(session_id, limit=10)
        
        if subscription_id in self._subscriptions:
            subscription = self._subscriptions[subscription_id]
            
            for event_dict in recent_events:
                try:
                    # Convert back to ProgressEvent
                    event = ProgressEvent(
                        event_id=event_dict['event_id'],
                        session_id=event_dict['session_id'],
                        event_type=ProgressEventType(event_dict['event_type']),
                        timestamp=datetime.fromisoformat(event_dict['timestamp']),
                        stage_name=event_dict.get('stage_name'),
                        progress_percent=event_dict['progress_percent'],
                        message=event_dict['message'],
                        details=event_dict['details'],
                        estimated_completion=datetime.fromisoformat(event_dict['estimated_completion']) if event_dict.get('estimated_completion') else None
                    )
                    
                    subscription.callback(event)
                except Exception as e:
                    self.logger.error(f"Error sending recent event to subscriber {subscription_id}: {e}")
                    break
    
    def _event_to_dict(self, event: ProgressEvent) -> Dict[str, Any]:
        """Convert event to dictionary for serialization.
        
        Args:
            event: Progress event
            
        Returns:
            Event as dictionary
        """
        return {
            'event_id': event.event_id,
            'session_id': event.session_id,
            'event_type': event.event_type.value,
            'timestamp': event.timestamp.isoformat(),
            'stage_name': event.stage_name,
            'progress_percent': event.progress_percent,
            'message': event.message,
            'details': event.details,
            'estimated_completion': event.estimated_completion.isoformat() if event.estimated_completion else None
        }
    
    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread for old events."""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(3600)  # Run every hour
                    self._cleanup_old_events()
                except Exception as e:
                    self.logger.error(f"Error in event cleanup: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        self.logger.info("🧹 Started event cleanup thread")
    
    def _cleanup_old_events(self) -> None:
        """Clean up old events to prevent memory buildup."""
        cutoff_time = datetime.now() - timedelta(hours=self.event_retention_hours)
        
        with self._events_lock:
            for session_id in list(self._session_events.keys()):
                events = self._session_events[session_id]
                recent_events = [e for e in events if e.timestamp > cutoff_time]
                
                if len(recent_events) != len(events):
                    self._session_events[session_id] = recent_events
                    self.logger.info(f"🧹 Cleaned up old events for session {session_id}")


# === SSE and WebSocket Support Functions ===

def create_sse_response(event: ProgressEvent) -> str:
    """Create Server-Sent Events formatted response.
    
    Args:
        event: Progress event
        
    Returns:
        SSE formatted string
    """
    event_dict = {
        'event_id': event.event_id,
        'session_id': event.session_id,
        'event_type': event.event_type.value,
        'timestamp': event.timestamp.isoformat(),
        'stage_name': event.stage_name,
        'progress_percent': event.progress_percent,
        'message': event.message,
        'details': event.details,
        'estimated_completion': event.estimated_completion.isoformat() if event.estimated_completion else None
    }
    
    return f"data: {json.dumps(event_dict)}\n\n"


async def websocket_event_sender(websocket, progress_manager: RealtimeProgressManager, session_id: str):
    """WebSocket event sender for async frameworks.
    
    Args:
        websocket: WebSocket connection
        progress_manager: Progress manager instance
        session_id: Session ID to subscribe to
    """
    event_queue = asyncio.Queue()
    
    def event_callback(event: ProgressEvent):
        try:
            asyncio.create_task(event_queue.put(event))
        except RuntimeError:
            # Event loop might be closed
            pass
    
    subscription_id = progress_manager.subscribe_to_session(session_id, event_callback)
    
    try:
        while True:
            event = await event_queue.get()
            event_dict = {
                'event_id': event.event_id,
                'session_id': event.session_id,
                'event_type': event.event_type.value,
                'timestamp': event.timestamp.isoformat(),
                'stage_name': event.stage_name,
                'progress_percent': event.progress_percent,
                'message': event.message,
                'details': event.details,
                'estimated_completion': event.estimated_completion.isoformat() if event.estimated_completion else None
            }
            
            await websocket.send_text(json.dumps(event_dict))
    
    except Exception as e:
        logging.getLogger(__name__).error(f"WebSocket error: {e}")
    finally:
        progress_manager.unsubscribe(subscription_id)


def main():
    """Test the real-time progress manager."""
    print("🧪 Testing Real-time Progress Manager")
    print("=" * 50)
    
    progress_manager = RealtimeProgressManager()
    
    # Test callback
    def test_callback(event: ProgressEvent):
        print(f"Server Event: {event.event_type.value} - {event.message} ({event.progress_percent:.1f}%)")
    
    # Subscribe to session
    session_id = "test-session-123"
    subscription_id = progress_manager.subscribe_to_session(session_id, test_callback)
    
    # Simulate pipeline progress
    progress_manager.start_stage(session_id, "script_generation", 10)
    time.sleep(1)
    
    progress_manager.update_stage_progress(session_id, "script_generation", 50, "Generating content...")
    time.sleep(1)
    
    progress_manager.complete_stage(session_id, "script_generation", True, "Script generated successfully")
    time.sleep(1)
    
    progress_manager.start_stage(session_id, "voice_cloning", 15)
    progress_manager.update_stage_progress(session_id, "voice_cloning", 75, "Processing voice...")
    progress_manager.complete_stage(session_id, "voice_cloning", True)
    
    progress_manager.complete_pipeline(session_id, True, "final_video.mp4")
    
    # Get final status
    status = progress_manager.get_session_progress(session_id)
    print(f"Final progress: {status['overall_progress']:.1f}%")
    
    # Cleanup
    progress_manager.unsubscribe(subscription_id)


if __name__ == "__main__":
    main()