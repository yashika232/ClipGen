/**
 * Memory Monitor Utility
 * Prevents JavaScript heap out of memory errors by monitoring and limiting memory usage
 */

class MemoryMonitor {
  constructor() {
    this.memoryThreshold = 512 * 1024 * 1024; // 512MB threshold
    this.isMonitoring = false;
    this.cleanupCallbacks = [];
    this.warningThreshold = 0.8; // 80% of threshold
    this.criticalThreshold = 0.9; // 90% of threshold
  }

  /**
   * Start memory monitoring
   */
  startMonitoring() {
    if (this.isMonitoring) return;
    
    this.isMonitoring = true;
    
    // Check memory every 10 seconds
    this.monitorInterval = setInterval(() => {
      this.checkMemoryUsage();
    }, 10000);
    
    console.info('Memory monitor started');
  }

  /**
   * Stop memory monitoring
   */
  stopMonitoring() {
    if (!this.isMonitoring) return;
    
    this.isMonitoring = false;
    
    if (this.monitorInterval) {
      clearInterval(this.monitorInterval);
      this.monitorInterval = null;
    }
    
    console.info('Memory monitor stopped');
  }

  /**
   * Check current memory usage
   */
  checkMemoryUsage() {
    if (!performance.memory) {
      return null;
    }

    const memory = performance.memory;
    const usedMemory = memory.usedJSHeapSize;
    const memoryUsagePercent = usedMemory / this.memoryThreshold;

    const status = {
      usedMB: Math.round(usedMemory / 1024 / 1024),
      totalMB: Math.round(memory.totalJSHeapSize / 1024 / 1024),
      limitMB: Math.round(memory.jsHeapSizeLimit / 1024 / 1024),
      thresholdMB: Math.round(this.memoryThreshold / 1024 / 1024),
      usagePercent: Math.round(memoryUsagePercent * 100),
      level: this.getMemoryLevel(memoryUsagePercent)
    };

    // Take action based on memory usage
    if (memoryUsagePercent >= this.criticalThreshold) {
      this.handleCriticalMemory(status);
    } else if (memoryUsagePercent >= this.warningThreshold) {
      this.handleWarningMemory(status);
    }

    return status;
  }

  /**
   * Get memory level indicator
   */
  getMemoryLevel(usagePercent) {
    if (usagePercent >= this.criticalThreshold) return 'critical';
    if (usagePercent >= this.warningThreshold) return 'warning';
    return 'normal';
  }

  /**
   * Handle warning level memory usage
   */
  handleWarningMemory(status) {
    console.warn('Memory usage warning:', status);
    this.triggerCleanup('warning');
  }

  /**
   * Handle critical level memory usage
   */
  handleCriticalMemory(status) {
    console.error('Critical memory usage detected:', status);
    
    // Aggressive cleanup
    this.triggerCleanup('critical');
    
    // Clear localStorage if possible
    this.clearStorageQuota();
    
    // Force garbage collection if available
    this.forceGarbageCollection();
  }

  /**
   * Trigger cleanup callbacks
   */
  triggerCleanup(level) {
    this.cleanupCallbacks.forEach(callback => {
      try {
        callback(level);
      } catch (error) {
        console.error('Error in cleanup callback:', error);
      }
    });
  }

  /**
   * Clear storage quota to free memory
   */
  clearStorageQuota() {
    try {
      // Clear old logs and data
      const keysToClean = [];
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key && (key.includes('log') || key.includes('cache') || key.includes('temp'))) {
          keysToClean.push(key);
        }
      }
      
      keysToClean.forEach(key => {
        localStorage.removeItem(key);
      });
      
      console.info(`Cleared ${keysToClean.length} localStorage items to free memory`);
    } catch (error) {
      console.error('Error clearing storage quota:', error);
    }
  }

  /**
   * Force garbage collection if available
   */
  forceGarbageCollection() {
    if (window.gc) {
      window.gc();
      console.info('Forced garbage collection');
    }
  }

  /**
   * Register cleanup callback
   */
  onMemoryPressure(callback) {
    this.cleanupCallbacks.push(callback);
  }

  /**
   * Unregister cleanup callback
   */
  removeMemoryPressureCallback(callback) {
    const index = this.cleanupCallbacks.indexOf(callback);
    if (index > -1) {
      this.cleanupCallbacks.splice(index, 1);
    }
  }

  /**
   * Set memory threshold
   */
  setMemoryThreshold(thresholdMB) {
    this.memoryThreshold = thresholdMB * 1024 * 1024;
    console.info(`Memory threshold set to ${thresholdMB}MB`);
  }

  /**
   * Get current memory status
   */
  getMemoryStatus() {
    return this.checkMemoryUsage();
  }

  /**
   * Check if JSON operation is safe
   */
  isSafeForJSONOperation(dataSize) {
    const status = this.checkMemoryUsage();
    if (!status) return true;
    
    const availableMemory = this.memoryThreshold - (status.usedMB * 1024 * 1024);
    return dataSize < availableMemory * 0.1; // Use only 10% of available memory
  }

  /**
   * Safe JSON stringify with size limits
   */
  safeJSONStringify(obj, maxSize = 1024 * 1024) { // 1MB default limit
    try {
      if (!this.isSafeForJSONOperation(maxSize)) {
        throw new Error('Insufficient memory for JSON operation');
      }
      
      const result = JSON.stringify(obj);
      
      if (result.length > maxSize) {
        throw new Error(`JSON result too large: ${result.length} bytes (limit: ${maxSize})`);
      }
      
      return result;
    } catch (error) {
      console.error('Safe JSON stringify failed:', error);
      return JSON.stringify({ error: 'JSON stringify failed due to memory constraints' });
    }
  }
}

// Create global memory monitor instance
const memoryMonitor = new MemoryMonitor();

// Auto-start monitoring when imported
memoryMonitor.startMonitoring();

export default memoryMonitor;

// Convenience exports
export const {
  startMonitoring,
  stopMonitoring,
  checkMemoryUsage,
  getMemoryStatus,
  onMemoryPressure,
  setMemoryThreshold,
  safeJSONStringify
} = memoryMonitor;