/**
 * Frontend Logger Utility
 * Provides structured logging for the video synthesis pipeline frontend
 * with session correlation, error tracking, and performance monitoring.
 * Now includes memory monitoring to prevent heap out of memory errors.
 */

import memoryMonitor from './memoryMonitor.js';

// Log levels
const LOG_LEVELS = {
  DEBUG: 'DEBUG',
  INFO: 'INFO',
  WARNING: 'WARNING',
  ERROR: 'ERROR',
  CRITICAL: 'CRITICAL'
};

// Frontend components
const LOG_COMPONENTS = {
  APP_CONTEXT: 'app_context',
  VIDEO_GENERATOR: 'video_generator',
  FILE_UPLOADER: 'file_uploader',
  PROGRESS_TRACKER: 'progress_tracker',
  API_SERVICE: 'api_service',
  WEBSOCKET_SERVICE: 'websocket_service',
  UI_COMPONENT: 'ui_component',
  USER_ACTION: 'user_action',
  PERFORMANCE: 'performance',
  ERROR_BOUNDARY: 'error_boundary'
};

// Log storage types
const STORAGE_TYPES = {
  CONSOLE: 'console',
  LOCAL_STORAGE: 'localStorage',
  SESSION_STORAGE: 'sessionStorage',
  REMOTE: 'remote'
};

class FrontendLogger {
  constructor() {
    this.sessionId = null;
    this.userId = null;
    this.correlationId = null;
    this.logLevel = LOG_LEVELS.INFO;
    this.enabledStorageTypes = [STORAGE_TYPES.CONSOLE, STORAGE_TYPES.LOCAL_STORAGE];
    this.remoteEndpoint = null;
    this.logBuffer = [];
    this.maxBufferSize = 50;
    this.maxLogEntries = 200;
    this.maxStorageSize = 2 * 1024 * 1024; // 2MB limit for localStorage
    this.performanceMarks = new Map();
    
    // Initialize session info
    this.initializeSession();
    
    // Setup error handlers
    this.setupErrorHandlers();
    
    // Setup periodic cleanup
    this.setupPeriodicCleanup();
    
    // Register with memory monitor
    this.setupMemoryMonitoring();
  }

  /**
   * Initialize session information
   */
  initializeSession() {
    this.sessionId = this.getStoredValue('logger_session_id') || this.generateId();
    this.correlationId = this.generateId();
    this.storeValue('logger_session_id', this.sessionId);
    
    // Get user info if available
    this.userId = this.getStoredValue('user_id') || 'anonymous';
    
    this.info(LOG_COMPONENTS.APP_CONTEXT, 'logger_initialized', 'Frontend logger initialized', {
      sessionId: this.sessionId,
      userId: this.userId,
      correlationId: this.correlationId,
      userAgent: navigator.userAgent,
      url: window.location.href,
      timestamp: new Date().toISOString()
    });
  }

  /**
   * Generate unique ID
   */
  generateId() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
      const r = Math.random() * 16 | 0;
      const v = c == 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
  }

  /**
   * Set session context
   */
  setSessionContext(sessionId, userId = null, correlationId = null) {
    this.sessionId = sessionId;
    this.userId = userId || this.userId;
    this.correlationId = correlationId || this.generateId();
    
    this.storeValue('logger_session_id', sessionId);
    if (userId) {
      this.storeValue('user_id', userId);
    }
    
    this.info(LOG_COMPONENTS.APP_CONTEXT, 'session_context_updated', 'Session context updated', {
      sessionId: this.sessionId,
      userId: this.userId,
      correlationId: this.correlationId
    });
  }

  /**
   * Set log level
   */
  setLogLevel(level) {
    if (Object.values(LOG_LEVELS).includes(level)) {
      this.logLevel = level;
      this.info(LOG_COMPONENTS.APP_CONTEXT, 'log_level_changed', `Log level changed to ${level}`, {
        previousLevel: this.logLevel,
        newLevel: level
      });
    }
  }

  /**
   * Configure remote logging
   */
  configureRemoteLogging(endpoint, enabled = true) {
    this.remoteEndpoint = enabled ? endpoint : null;
    if (enabled) {
      this.enabledStorageTypes.push(STORAGE_TYPES.REMOTE);
    } else {
      this.enabledStorageTypes = this.enabledStorageTypes.filter(type => type !== STORAGE_TYPES.REMOTE);
    }
    
    this.info(LOG_COMPONENTS.APP_CONTEXT, 'remote_logging_configured', `Remote logging ${enabled ? 'enabled' : 'disabled'}`, {
      endpoint: endpoint,
      enabled: enabled
    });
  }

  /**
   * Check if should log at level
   */
  shouldLog(level) {
    const levels = Object.values(LOG_LEVELS);
    return levels.indexOf(level) >= levels.indexOf(this.logLevel);
  }

  /**
   * Create log entry
   */
  createLogEntry(level, component, event, message, metadata = {}, executionTime = null, error = null) {
    const entry = {
      timestamp: new Date().toISOString(),
      level: level,
      sessionId: this.sessionId,
      userId: this.userId,
      correlationId: this.correlationId,
      component: component,
      event: event,
      message: message,
      metadata: this.filterSensitiveData(metadata),
      executionTime: executionTime,
      url: window.location.href,
      userAgent: navigator.userAgent,
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight
      },
      memory: this.getMemoryInfo(),
      error: error ? {
        name: error.name,
        message: error.message,
        stack: error.stack,
        fileName: error.fileName,
        lineNumber: error.lineNumber,
        columnNumber: error.columnNumber
      } : null
    };

    return entry;
  }

  /**
   * Filter sensitive data from logs
   */
  filterSensitiveData(data) {
    const sensitiveKeys = ['password', 'token', 'apiKey', 'secret', 'authorization', 'cookie'];
    const filtered = {};
    
    for (const [key, value] of Object.entries(data)) {
      if (sensitiveKeys.some(sensitive => key.toLowerCase().includes(sensitive))) {
        filtered[key] = '***REDACTED***';
      } else if (typeof value === 'object' && value !== null) {
        filtered[key] = this.filterSensitiveData(value);
      } else {
        filtered[key] = value;
      }
    }
    
    return filtered;
  }

  /**
   * Get memory information (simplified to prevent memory leaks)
   */
  getMemoryInfo() {
    if (performance.memory) {
      return {
        usedMB: Math.round(performance.memory.usedJSHeapSize / 1024 / 1024),
        limitMB: Math.round(performance.memory.jsHeapSizeLimit / 1024 / 1024)
      };
    }
    return null;
  }

  /**
   * Log entry to different storage types
   */
  logToStorages(entry) {
    // Console logging
    if (this.enabledStorageTypes.includes(STORAGE_TYPES.CONSOLE)) {
      this.logToConsole(entry);
    }

    // Local storage logging
    if (this.enabledStorageTypes.includes(STORAGE_TYPES.LOCAL_STORAGE)) {
      this.logToLocalStorage(entry);
    }

    // Session storage logging
    if (this.enabledStorageTypes.includes(STORAGE_TYPES.SESSION_STORAGE)) {
      this.logToSessionStorage(entry);
    }

    // Remote logging
    if (this.enabledStorageTypes.includes(STORAGE_TYPES.REMOTE) && this.remoteEndpoint) {
      this.logToRemote(entry);
    }
  }

  /**
   * Log to console
   */
  logToConsole(entry) {
    const logMessage = `[${entry.timestamp}] [${entry.level}] [${entry.component}] ${entry.event}: ${entry.message}`;
    
    switch (entry.level) {
      case LOG_LEVELS.DEBUG:
        console.debug(logMessage, entry.metadata);
        break;
      case LOG_LEVELS.INFO:
        console.info(logMessage, entry.metadata);
        break;
      case LOG_LEVELS.WARNING:
        console.warn(logMessage, entry.metadata);
        break;
      case LOG_LEVELS.ERROR:
      case LOG_LEVELS.CRITICAL:
        console.error(logMessage, entry.metadata, entry.error);
        break;
      default:
        console.log(logMessage, entry.metadata);
    }
  }

  /**
   * Log to local storage
   */
  logToLocalStorage(entry) {
    try {
      const logs = this.getStoredLogs('frontend_logs') || [];
      logs.push(entry);
      
      // Keep only recent logs
      if (logs.length > this.maxLogEntries) {
        logs.splice(0, logs.length - this.maxLogEntries);
      }
      
      // Check storage size before storing
      const serialized = JSON.stringify(logs);
      if (serialized.length > this.maxStorageSize) {
        // Reduce logs by half if too large
        logs.splice(0, Math.floor(logs.length / 2));
        console.warn('Logger: Reduced log entries due to size limit');
      }
      
      this.storeValue('frontend_logs', logs);
    } catch (error) {
      // Clear logs if storage is full
      if (error.name === 'QuotaExceededError') {
        this.clearLogs();
        console.warn('Logger: Cleared logs due to storage quota exceeded');
      } else {
        console.error('Failed to log to localStorage:', error);
      }
    }
  }

  /**
   * Log to session storage
   */
  logToSessionStorage(entry) {
    try {
      const logs = this.getStoredLogs('session_logs', 'sessionStorage') || [];
      logs.push(entry);
      
      // Keep only recent logs
      if (logs.length > this.maxLogEntries) {
        logs.splice(0, logs.length - this.maxLogEntries);
      }
      
      sessionStorage.setItem('session_logs', JSON.stringify(logs));
    } catch (error) {
      console.error('Failed to log to sessionStorage:', error);
    }
  }

  /**
   * Log to remote endpoint
   */
  logToRemote(entry) {
    this.logBuffer.push(entry);
    
    // Send logs in batches
    if (this.logBuffer.length >= this.maxBufferSize) {
      this.flushRemoteLogs();
    }
  }

  /**
   * Flush remote logs
   */
  async flushRemoteLogs() {
    if (this.logBuffer.length === 0 || !this.remoteEndpoint) {
      return;
    }
    
    const logs = [...this.logBuffer];
    this.logBuffer = [];
    
    try {
      const response = await fetch(this.remoteEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          logs: logs,
          sessionId: this.sessionId,
          userId: this.userId,
          source: 'frontend'
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
    } catch (error) {
      console.error('Failed to send logs to remote endpoint:', error);
      // Re-add logs to buffer for retry
      this.logBuffer.unshift(...logs);
    }
  }

  /**
   * Core logging method
   */
  log(level, component, event, message, metadata = {}, executionTime = null, error = null) {
    if (!this.shouldLog(level)) {
      return;
    }

    const entry = this.createLogEntry(level, component, event, message, metadata, executionTime, error);
    this.logToStorages(entry);
  }

  /**
   * Log level methods
   */
  debug(component, event, message, metadata = {}, executionTime = null) {
    this.log(LOG_LEVELS.DEBUG, component, event, message, metadata, executionTime);
  }

  info(component, event, message, metadata = {}, executionTime = null) {
    this.log(LOG_LEVELS.INFO, component, event, message, metadata, executionTime);
  }

  warning(component, event, message, metadata = {}, executionTime = null) {
    this.log(LOG_LEVELS.WARNING, component, event, message, metadata, executionTime);
  }

  error(component, event, message, metadata = {}, executionTime = null, error = null) {
    this.log(LOG_LEVELS.ERROR, component, event, message, metadata, executionTime, error);
  }

  critical(component, event, message, metadata = {}, executionTime = null, error = null) {
    this.log(LOG_LEVELS.CRITICAL, component, event, message, metadata, executionTime, error);
  }

  /**
   * Performance monitoring
   */
  startPerformanceTimer(operationId) {
    this.performanceMarks.set(operationId, {
      startTime: performance.now(),
      startMemory: this.getMemoryInfo()
    });
  }

  endPerformanceTimer(operationId, component, event, message, metadata = {}) {
    const mark = this.performanceMarks.get(operationId);
    if (!mark) {
      return;
    }

    const executionTime = performance.now() - mark.startTime;
    const endMemory = this.getMemoryInfo();
    
    this.performanceMarks.delete(operationId);
    
    const performanceMetadata = {
      ...metadata,
      operationId: operationId,
      executionTime: executionTime,
      startMemory: mark.startMemory,
      endMemory: endMemory,
      memoryDelta: endMemory && mark.startMemory ? 
        endMemory.usedJSHeapSize - mark.startMemory.usedJSHeapSize : null
    };

    this.info(component, event, message, performanceMetadata, executionTime);
  }

  /**
   * User action logging
   */
  logUserAction(action, target, metadata = {}) {
    this.info(LOG_COMPONENTS.USER_ACTION, 'user_action', `User ${action} on ${target}`, {
      action: action,
      target: target,
      ...metadata
    });
  }

  /**
   * API call logging
   */
  logApiCall(method, url, status, executionTime, requestData = null, responseData = null, error = null) {
    const level = error ? LOG_LEVELS.ERROR : LOG_LEVELS.INFO;
    const event = error ? 'api_call_failed' : 'api_call_success';
    
    this.log(level, LOG_COMPONENTS.API_SERVICE, event, `${method} ${url} - ${status}`, {
      method: method,
      url: url,
      status: status,
      requestData: requestData,
      responseData: responseData
    }, executionTime, error);
  }

  /**
   * WebSocket event logging
   */
  logWebSocketEvent(event, data = {}) {
    this.info(LOG_COMPONENTS.WEBSOCKET_SERVICE, 'websocket_event', `WebSocket event: ${event}`, {
      event: event,
      data: data
    });
  }

  /**
   * State change logging
   */
  logStateChange(component, previousState, newState, action = null) {
    this.debug(LOG_COMPONENTS.APP_CONTEXT, 'state_change', `State changed in ${component}`, {
      component: component,
      previousState: previousState,
      newState: newState,
      action: action
    });
  }

  /**
   * Error boundary logging
   */
  logErrorBoundary(error, errorInfo, componentStack) {
    this.critical(LOG_COMPONENTS.ERROR_BOUNDARY, 'error_boundary_triggered', 'React error boundary triggered', {
      componentStack: componentStack,
      errorInfo: errorInfo
    }, null, error);
  }

  /**
   * Setup error handlers
   */
  setupErrorHandlers() {
    // Global error handler
    window.addEventListener('error', (event) => {
      this.error(LOG_COMPONENTS.APP_CONTEXT, 'global_error', 'Global JavaScript error', {
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno
      }, null, event.error);
    });

    // Unhandled promise rejection handler
    window.addEventListener('unhandledrejection', (event) => {
      this.error(LOG_COMPONENTS.APP_CONTEXT, 'unhandled_promise_rejection', 'Unhandled promise rejection', {
        reason: event.reason
      }, null, event.reason);
    });
  }

  /**
   * Setup periodic cleanup
   */
  setupPeriodicCleanup() {
    // Flush remote logs every 30 seconds
    setInterval(() => {
      this.flushRemoteLogs();
    }, 30000);

    // Clean up old logs every 5 minutes
    setInterval(() => {
      this.cleanupOldLogs();
    }, 300000);
  }

  /**
   * Clean up old logs
   */
  cleanupOldLogs() {
    try {
      const logs = this.getStoredLogs('frontend_logs') || [];
      const cutoffTime = new Date(Date.now() - 24 * 60 * 60 * 1000); // 24 hours ago
      
      const filteredLogs = logs.filter(log => new Date(log.timestamp) > cutoffTime);
      
      if (filteredLogs.length < logs.length) {
        this.storeValue('frontend_logs', filteredLogs);
        this.info(LOG_COMPONENTS.APP_CONTEXT, 'logs_cleaned', `Cleaned up ${logs.length - filteredLogs.length} old log entries`);
      }
    } catch (error) {
      console.error('Failed to clean up old logs:', error);
    }
  }

  /**
   * Get stored logs
   */
  getStoredLogs(key, storageType = 'localStorage') {
    try {
      const storage = storageType === 'sessionStorage' ? sessionStorage : localStorage;
      const stored = storage.getItem(key);
      return stored ? JSON.parse(stored) : null;
    } catch (error) {
      console.error(`Failed to get stored logs from ${storageType}:`, error);
      return null;
    }
  }

  /**
   * Store value with memory safety
   */
  storeValue(key, value) {
    try {
      const jsonString = memoryMonitor.safeJSONStringify(value, this.maxStorageSize);
      localStorage.setItem(key, jsonString);
    } catch (error) {
      if (error.name === 'QuotaExceededError') {
        this.clearLogs();
        console.warn('Storage quota exceeded, cleared logs and retrying');
        try {
          const jsonString = memoryMonitor.safeJSONStringify(value, this.maxStorageSize / 2);
          localStorage.setItem(key, jsonString);
        } catch (retryError) {
          console.error('Failed to store value after cleanup:', retryError);
        }
      } else {
        console.error('Failed to store value:', error);
      }
    }
  }

  /**
   * Setup memory monitoring integration
   */
  setupMemoryMonitoring() {
    // Register cleanup callback with memory monitor
    memoryMonitor.onMemoryPressure((level) => {
      if (level === 'warning') {
        // Reduce log entries
        this.cleanupOldLogs();
      } else if (level === 'critical') {
        // Aggressive cleanup
        this.clearLogs();
        this.info(LOG_COMPONENTS.APP_CONTEXT, 'memory_pressure_cleanup', 
          'Cleared logs due to critical memory pressure');
      }
    });
  }

  /**
   * Get stored value
   */
  getStoredValue(key) {
    try {
      const stored = localStorage.getItem(key);
      return stored ? JSON.parse(stored) : null;
    } catch (error) {
      console.error('Failed to get stored value:', error);
      return null;
    }
  }

  /**
   * Export logs
   */
  exportLogs() {
    const logs = this.getStoredLogs('frontend_logs') || [];
    const blob = new Blob([JSON.stringify(logs, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `frontend_logs_${this.sessionId}_${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    this.info(LOG_COMPONENTS.APP_CONTEXT, 'logs_exported', 'Logs exported successfully');
  }

  /**
   * Clear logs
   */
  clearLogs() {
    try {
      localStorage.removeItem('frontend_logs');
      sessionStorage.removeItem('session_logs');
      this.info(LOG_COMPONENTS.APP_CONTEXT, 'logs_cleared', 'All logs cleared');
    } catch (error) {
      console.error('Failed to clear logs:', error);
    }
  }
}

// Create global logger instance
const frontendLogger = new FrontendLogger();

// Export logger and constants
export {
  frontendLogger as default,
  LOG_LEVELS,
  LOG_COMPONENTS,
  STORAGE_TYPES
};

// Performance monitoring decorator
export const withPerformanceLogging = (component, operation) => {
  return (target, propertyKey, descriptor) => {
    const originalMethod = descriptor.value;
    
    descriptor.value = async function(...args) {
      const operationId = `${component}_${operation}_${Date.now()}`;
      frontendLogger.startPerformanceTimer(operationId);
      
      try {
        const result = await originalMethod.apply(this, args);
        frontendLogger.endPerformanceTimer(
          operationId,
          component,
          'operation_completed',
          `${operation} completed successfully`,
          { args: args, result: result }
        );
        return result;
      } catch (error) {
        frontendLogger.endPerformanceTimer(
          operationId,
          component,
          'operation_failed',
          `${operation} failed`,
          { args: args },
          error
        );
        throw error;
      }
    };
    
    return descriptor;
  };
};

// Convenience functions
export const logUserAction = (action, target, metadata = {}) => {
  frontendLogger.logUserAction(action, target, metadata);
};

export const logApiCall = (method, url, status, executionTime, requestData = null, responseData = null, error = null) => {
  frontendLogger.logApiCall(method, url, status, executionTime, requestData, responseData, error);
};

export const logWebSocketEvent = (event, data = {}) => {
  frontendLogger.logWebSocketEvent(event, data);
};

export const logStateChange = (component, previousState, newState, action = null) => {
  frontendLogger.logStateChange(component, previousState, newState, action);
};

export const setSessionContext = (sessionId, userId = null, correlationId = null) => {
  frontendLogger.setSessionContext(sessionId, userId, correlationId);
};

export const configureRemoteLogging = (endpoint, enabled = true) => {
  frontendLogger.configureRemoteLogging(endpoint, enabled);
};

export const exportLogs = () => {
  frontendLogger.exportLogs();
};

export const clearLogs = () => {
  frontendLogger.clearLogs();
};