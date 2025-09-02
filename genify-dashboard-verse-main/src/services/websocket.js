import { io } from 'socket.io-client';

class WebSocketService {
  constructor() {
    this.socket = null;
    this.isConnected = false;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectInterval = 1000;
    this.eventHandlers = new Map();
    this.heartbeatInterval = null;
  }

  connect(url = 'http://localhost:5002') {
    if (this.socket) {
      this.disconnect();
    }

    this.socket = io(url, {
      transports: ['websocket', 'polling'], // Allow fallback to polling
      timeout: 60000,
      reconnection: true,
      reconnectionAttempts: this.maxReconnectAttempts,
      reconnectionDelay: this.reconnectInterval,
      reconnectionDelayMax: 5000, // Max delay between attempts
      randomizationFactor: 0.5, // Randomize reconnection delays
      forceNew: true, // Force new connection
    });

    this.setupEventHandlers();
    
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('WebSocket connection timeout'));
      }, 10000);

      this.socket.on('connect', () => {
        clearTimeout(timeout);
        this.isConnected = true;
        this.reconnectAttempts = 0;
        console.log('🔗 WebSocket connected');
        resolve();
      });

      this.socket.on('connect_error', (error) => {
        clearTimeout(timeout);
        console.error('❌ WebSocket connection error:', error);
        reject(error);
      });
    });
  }

  setupEventHandlers() {
    if (!this.socket) return;

    this.socket.on('connect', () => {
      this.isConnected = true;
      this.reconnectAttempts = 0;
      console.log('✅ WebSocket connected');
      this.emit('connection_status', { connected: true });
      this.startHeartbeat();
    });

    this.socket.on('disconnect', (reason) => {
      this.isConnected = false;
      console.log('❌ WebSocket disconnected:', reason);
      this.emit('connection_status', { connected: false, reason });
      this.stopHeartbeat();
    });

    this.socket.on('reconnect', (attemptNumber) => {
      this.isConnected = true;
      console.log(`🔄 WebSocket reconnected after ${attemptNumber} attempts`);
      this.emit('connection_status', { connected: true, reconnected: true });
    });

    this.socket.on('reconnect_error', (error) => {
      console.error('🔄 WebSocket reconnection error:', error);
      this.reconnectAttempts++;
    });

    this.socket.on('reconnect_failed', () => {
      console.error('💔 WebSocket reconnection failed');
      this.emit('connection_status', { connected: false, failed: true });
    });

    // Pipeline-specific events
    this.socket.on('progress_update', (data) => {
      console.log('📊 Progress update:', data);
      this.emit('progress_update', data);
    });

    this.socket.on('upload_start', (data) => {
      console.log('📤 Upload started:', data);
      this.emit('upload_start', data);
    });

    this.socket.on('upload_success', (data) => {
      console.log('✅ Upload completed:', data);
      this.emit('upload_success', data);
    });

    this.socket.on('upload_failed', (data) => {
      console.error('❌ Upload failed:', data);
      this.emit('upload_failed', data);
    });

    this.socket.on('processing_started', (data) => {
      console.log('🚀 Processing started:', data);
      this.emit('processing_started', data);
    });

    this.socket.on('processing_stage_update', (data) => {
      console.log('🔄 Processing stage update:', data);
      this.emit('processing_stage_update', data);
    });

    this.socket.on('processing_completed', (data) => {
      console.log('🎉 Processing completed:', data);
      this.emit('processing_completed', data);
    });

    this.socket.on('processing_error', (data) => {
      console.error('❌ Processing error:', data);
      this.emit('processing_error', data);
    });

    this.socket.on('generation_started', (data) => {
      console.log('🎨 Generation started:', data);
      this.emit('generation_started', data);
    });

    this.socket.on('generation_progress', (data) => {
      console.log('📈 Generation progress:', data);
      this.emit('generation_progress', data);
    });

    this.socket.on('generation_completed', (data) => {
      console.log('✨ Generation completed:', data);
      this.emit('generation_completed', data);
    });

    this.socket.on('generation_error', (data) => {
      console.error('❌ Generation error:', data);
      this.emit('generation_error', data);
    });
  }

  disconnect() {
    this.stopHeartbeat();
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
      this.isConnected = false;
      console.log('🔌 WebSocket disconnected');
    }
  }

  startHeartbeat() {
    this.stopHeartbeat(); // Clear any existing heartbeat
    this.heartbeatInterval = setInterval(() => {
      if (this.socket && this.isConnected) {
        this.socket.emit('ping');
      }
    }, 30000); // Send ping every 30 seconds
  }

  stopHeartbeat() {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  // Subscribe to progress updates for a specific session
  subscribeToProgress(sessionId) {
    if (!this.socket || !this.isConnected) {
      console.error('❌ WebSocket not connected');
      return false;
    }

    this.socket.emit('subscribe_progress', { session_id: sessionId });
    console.log(`📋 Subscribed to progress for session: ${sessionId}`);
    return true;
  }

  // Unsubscribe from progress updates
  unsubscribeFromProgress(sessionId) {
    if (!this.socket || !this.isConnected) {
      console.error('❌ WebSocket not connected');
      return false;
    }

    this.socket.emit('unsubscribe_progress', { session_id: sessionId });
    console.log(`📋 Unsubscribed from progress for session: ${sessionId}`);
    return true;
  }

  // Event handler management
  on(event, handler) {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, new Set());
    }
    this.eventHandlers.get(event).add(handler);
  }

  off(event, handler) {
    if (this.eventHandlers.has(event)) {
      this.eventHandlers.get(event).delete(handler);
    }
  }

  emit(event, data) {
    if (this.eventHandlers.has(event)) {
      this.eventHandlers.get(event).forEach(handler => {
        try {
          handler(data);
        } catch (error) {
          console.error(`Error in event handler for ${event}:`, error);
        }
      });
    }
  }

  // Utility methods
  isConnected() {
    return this.isConnected && this.socket && this.socket.connected;
  }

  getConnectionStatus() {
    return {
      connected: this.isConnected,
      socketId: this.socket?.id || null,
      reconnectAttempts: this.reconnectAttempts,
    };
  }

  // Send custom message to server
  send(event, data) {
    if (!this.socket || !this.isConnected) {
      console.error('❌ WebSocket not connected');
      return false;
    }

    this.socket.emit(event, data);
    return true;
  }
}

// Create singleton instance
const webSocketService = new WebSocketService();

export default webSocketService;