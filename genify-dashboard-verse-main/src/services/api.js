import axios from 'axios';

// API Configuration
const API_BASE_URL = 'http://localhost:5002';

// Create axios instance with default configuration
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5 minutes timeout for video processing
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for adding authentication or other headers
api.interceptors.request.use(
  (config) => {
    // Add timestamp to requests
    config.headers['X-Request-Time'] = new Date().toISOString();
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for handling errors
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    if (error.response) {
      // Server responded with error status
      const errorData = error.response.data;
      return Promise.reject({
        message: errorData.message || 'Server error occurred',
        details: errorData.details || {},
        status: error.response.status,
      });
    } else if (error.request) {
      // Request was made but no response received
      return Promise.reject({
        message: 'No response from server. Please check your connection.',
        details: {},
        status: 0,
      });
    } else {
      // Something else happened
      return Promise.reject({
        message: error.message || 'An unexpected error occurred',
        details: {},
        status: 0,
      });
    }
  }
);

// API Service Methods
export const apiService = {
  // System Health & Capabilities
  async getHealth() {
    const response = await api.get('/health');
    return response.data;
  },

  async getCapabilities() {
    const response = await api.get('/capabilities');
    return response.data;
  },

  // Session Management
  async createSession(userId = null) {
    const response = await api.post('/session/create', {
      user_id: userId || `user_${Date.now()}`,
    });
    return response.data;
  },

  async getSessionStatus(sessionId) {
    const response = await api.get(`/session/${sessionId}/status`);
    return response.data;
  },

  // File Upload
  async uploadFaceImage(file, sessionId) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('session_id', sessionId);

    const response = await api.post('/upload/face', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  async uploadAudioFile(file, sessionId) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('session_id', sessionId);

    const response = await api.post('/upload/audio', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  // Content Generation
  async startProcessing(sessionId, config) {
    const response = await api.post('/process/start', {
      session_id: sessionId,
      ...config,
    });
    return response.data;
  },

  // Script Generation (now using real backend endpoint)
  async generateScript(params) {
    const response = await api.post('/generate/script', params);
    const data = response.data;
    
    // Handle structured response from backend
    if (data.success && data.script) {
      return {
        script: data.content || data.script.core_content || data.script,
        content: data.content || data.script.core_content || data.script,
        session_id: data.session_id,
        sections: data.script,
        stats: data.stats
      };
    }
    
    return data;
  },

  // Thumbnail Generation (now using real backend endpoint)
  async generateThumbnail(params) {
    const response = await api.post('/generate/thumbnail', params);
    const data = response.data;
    
    // Handle structured response from backend
    if (data.success && data.thumbnails) {
      return {
        thumbnails: data.thumbnails,
        session_id: data.session_id,
        params: data.params
      };
    }
    
    return data;
  },

  // Video Generation (uses the actual backend process/start endpoint)
  async generateVideo(params) {
    const response = await api.post('/process/start', {
      session_id: params.sessionId,
      script_text: params.script_text,
      ...params
    });
    return response.data;
  },

  // Output Management
  async getOutputs(sessionId) {
    const response = await api.get(`/outputs/${sessionId}`);
    return response.data;
  },

  async downloadFile(sessionId, filename) {
    const response = await api.get(`/download/${sessionId}/${filename}`, {
      responseType: 'blob',
    });
    
    // Create download URL
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', filename);
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
    
    return { success: true, filename };
  },

  // Cleanup
  async cleanup(sessionId = null) {
    const response = await api.post('/cleanup', {
      session_id: sessionId,
    });
    return response.data;
  },

  // Utility Methods
  async checkServerStatus() {
    try {
      const health = await this.getHealth();
      return health.status === 'healthy';
    } catch (error) {
      return false;
    }
  },

  // Get available options for dropdowns
  async getAvailableOptions() {
    return {
      tones: ['professional', 'friendly', 'motivational', 'casual', 'enthusiastic', 'educational'],
      emotions: ['inspired', 'confident', 'curious', 'excited', 'calm', 'passionate', 'determined'],
      audiences: ['junior engineers', 'students', 'professionals', 'general public', 'experts', 'beginners', 'intermediate', 'advanced'],
      contentTypes: ['educational', 'promotional', 'tutorial', 'announcement', 'overview'],
      qualityLevels: ['draft', 'standard', 'high', 'premium'],
      thumbnailStyles: ['modern', 'minimalist', 'bold', 'professional', 'gaming', 'educational'],
    };
  },
};

export default apiService;