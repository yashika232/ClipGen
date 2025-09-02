import React, { createContext, useContext, useReducer, useEffect } from 'react';
import apiService from '../services/api';
import webSocketService from '../services/websocket';

// Initial state
const initialState = {
  // Connection status
  isConnected: false,
  serverStatus: 'unknown',
  
  // Session management
  currentSession: null,
  sessions: {},
  
  // User data
  user: null,
  
  // Progress tracking
  progress: {},
  
  // Generated content
  generatedContent: {
    thumbnails: [],
    scripts: [],
    videos: [],
  },
  
  // File uploads
  uploads: {},
  
  // UI state
  loading: false,
  error: null,
  
  // Available options
  availableOptions: {
    tones: [],
    emotions: [],
    audiences: [],
    contentTypes: [],
    qualityLevels: [],
    thumbnailStyles: [],
  },
};

// Action types
const actionTypes = {
  SET_CONNECTION_STATUS: 'SET_CONNECTION_STATUS',
  SET_SERVER_STATUS: 'SET_SERVER_STATUS',
  SET_USER: 'SET_USER',
  INITIALIZE_METADATA: 'INITIALIZE_METADATA',
  UPDATE_METADATA: 'UPDATE_METADATA',
  SYNC_METADATA: 'SYNC_METADATA',
  SET_CURRENT_SESSION: 'SET_CURRENT_SESSION',
  ADD_SESSION: 'ADD_SESSION',
  UPDATE_SESSION: 'UPDATE_SESSION',
  SET_PROGRESS: 'SET_PROGRESS',
  UPDATE_PROGRESS: 'UPDATE_PROGRESS',
  ADD_GENERATED_CONTENT: 'ADD_GENERATED_CONTENT',
  UPDATE_UPLOAD_STATUS: 'UPDATE_UPLOAD_STATUS',
  SET_LOADING: 'SET_LOADING',
  SET_ERROR: 'SET_ERROR',
  CLEAR_ERROR: 'CLEAR_ERROR',
  SET_AVAILABLE_OPTIONS: 'SET_AVAILABLE_OPTIONS',
};

// Reducer
const appReducer = (state, action) => {
  switch (action.type) {
    case actionTypes.SET_CONNECTION_STATUS:
      return {
        ...state,
        isConnected: action.payload,
      };
    
    case actionTypes.SET_SERVER_STATUS:
      return {
        ...state,
        serverStatus: action.payload,
      };
    
    case actionTypes.SET_USER:
      return {
        ...state,
        user: action.payload,
      };
    
    case actionTypes.SET_CURRENT_SESSION:
      return {
        ...state,
        currentSession: action.payload,
        sessions: {
          ...state.sessions,
          [action.payload.session_id]: action.payload
        }
      };
    
    case actionTypes.INITIALIZE_METADATA:
      return {
        ...state,
        sessions: {
          ...state.sessions,
          [action.payload.session_id]: {
            ...state.sessions[action.payload.session_id],
            metadata: action.payload.metadata
          }
        }
      };
    
    case actionTypes.UPDATE_METADATA:
      const { session_id, metadata } = action.payload;
      return {
        ...state,
        sessions: {
          ...state.sessions,
          [session_id]: {
            ...state.sessions[session_id],
            metadata: {
              ...state.sessions[session_id]?.metadata,
              ...metadata
            }
          }
        }
      };
    
    case actionTypes.SYNC_METADATA:
      return {
        ...state,
        sessions: {
          ...state.sessions,
          [action.payload.session_id]: {
            ...state.sessions[action.payload.session_id],
            metadata: action.payload.metadata,
            last_synced: action.payload.timestamp
          }
        }
      };
    
    case actionTypes.ADD_SESSION:
      return {
        ...state,
        sessions: {
          ...state.sessions,
          [action.payload.session_id]: action.payload,
        },
      };
    
    case actionTypes.UPDATE_SESSION:
      return {
        ...state,
        sessions: {
          ...state.sessions,
          [action.payload.session_id]: {
            ...state.sessions[action.payload.session_id],
            ...action.payload,
          },
        },
      };
    
    case actionTypes.SET_PROGRESS:
      return {
        ...state,
        progress: {
          ...state.progress,
          [action.payload.session_id]: action.payload.progress,
        },
      };
    
    case actionTypes.UPDATE_PROGRESS:
      return {
        ...state,
        progress: {
          ...state.progress,
          [action.payload.session_id]: {
            ...state.progress[action.payload.session_id],
            ...action.payload.progress,
          },
        },
      };
    
    case actionTypes.ADD_GENERATED_CONTENT:
      return {
        ...state,
        generatedContent: {
          ...state.generatedContent,
          [action.payload.type]: [
            ...state.generatedContent[action.payload.type],
            action.payload.content,
          ],
        },
      };
    
    case actionTypes.UPDATE_UPLOAD_STATUS:
      return {
        ...state,
        uploads: {
          ...state.uploads,
          [action.payload.session_id]: {
            ...state.uploads[action.payload.session_id],
            ...action.payload.upload,
          },
        },
      };
    
    case actionTypes.SET_LOADING:
      return {
        ...state,
        loading: action.payload,
      };
    
    case actionTypes.SET_ERROR:
      return {
        ...state,
        error: action.payload,
      };
    
    case actionTypes.CLEAR_ERROR:
      return {
        ...state,
        error: null,
      };
    
    case actionTypes.SET_AVAILABLE_OPTIONS:
      return {
        ...state,
        availableOptions: action.payload,
      };
    
    default:
      return state;
  }
};

// Create context
const AppContext = createContext();

// Custom hook to use the context
export const useApp = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
};

// Provider component
export const AppProvider = ({ children }) => {
  const [state, dispatch] = useReducer(appReducer, initialState);

  // Initialize WebSocket connection
  useEffect(() => {
    const initializeWebSocket = async () => {
      try {
        await webSocketService.connect();
        dispatch({ type: actionTypes.SET_CONNECTION_STATUS, payload: true });
        
        // Set up event handlers
        webSocketService.on('connection_status', (data) => {
          dispatch({ type: actionTypes.SET_CONNECTION_STATUS, payload: data.connected });
        });
        
        webSocketService.on('progress_update', (data) => {
          dispatch({
            type: actionTypes.UPDATE_PROGRESS,
            payload: {
              session_id: data.session_id,
              progress: data,
            },
          });
        });
        
        webSocketService.on('upload_start', (data) => {
          dispatch({
            type: actionTypes.UPDATE_UPLOAD_STATUS,
            payload: {
              session_id: data.session_id,
              upload: { status: 'uploading', ...data },
            },
          });
        });
        
        webSocketService.on('upload_success', (data) => {
          dispatch({
            type: actionTypes.UPDATE_UPLOAD_STATUS,
            payload: {
              session_id: data.session_id,
              upload: { status: 'completed', ...data },
            },
          });
        });
        
        webSocketService.on('upload_failed', (data) => {
          dispatch({
            type: actionTypes.UPDATE_UPLOAD_STATUS,
            payload: {
              session_id: data.session_id,
              upload: { status: 'failed', error: data.error },
            },
          });
        });
        
        webSocketService.on('processing_started', (data) => {
          dispatch({
            type: actionTypes.UPDATE_PROGRESS,
            payload: {
              session_id: data.session_id,
              progress: { status: 'processing', stage: 'starting', ...data },
            },
          });
        });
        
        webSocketService.on('processing_completed', (data) => {
          dispatch({
            type: actionTypes.UPDATE_PROGRESS,
            payload: {
              session_id: data.session_id,
              progress: { status: 'completed', ...data },
            },
          });
        });
        
        webSocketService.on('processing_error', (data) => {
          dispatch({
            type: actionTypes.SET_ERROR,
            payload: {
              message: data.message,
              session_id: data.session_id,
            },
          });
        });
        
      } catch (error) {
        console.error('Failed to initialize WebSocket:', error);
        dispatch({ type: actionTypes.SET_CONNECTION_STATUS, payload: false });
      }
    };

    initializeWebSocket();

    // Cleanup on unmount
    return () => {
      webSocketService.disconnect();
    };
  }, []);

  // Initialize session automatically
  useEffect(() => {
    const initializeSession = async () => {
      try {
        // Check server status first
        const health = await apiService.getHealth();
        dispatch({ type: actionTypes.SET_SERVER_STATUS, payload: health.healthy ? 'healthy' : 'unhealthy' });
        
        if (health.healthy) {
          // Create a new session automatically
          const session = await apiService.createSession();
          dispatch({ type: actionTypes.SET_CURRENT_SESSION, payload: session });
          
          // Load available options
          const options = await apiService.getAvailableOptions();
          dispatch({ type: actionTypes.SET_AVAILABLE_OPTIONS, payload: options });
          
          // Initialize metadata tracking
          dispatch({ 
            type: actionTypes.INITIALIZE_METADATA, 
            payload: { 
              session_id: session.session_id,
              metadata: {
                user_inputs: {},
                generated_content: {},
                pipeline_stages: {},
                target_duration: null
              }
            }
          });
          
          console.log('✅ Session initialized:', session.session_id);
        }
      } catch (error) {
        console.error('Failed to initialize session:', error);
        dispatch({ type: actionTypes.SET_ERROR, payload: { message: 'Failed to connect to server' } });
      }
    };

    initializeSession();
  }, []);

  // Check server status on load
  useEffect(() => {
    const checkServerStatus = async () => {
      try {
        const isHealthy = await apiService.checkServerStatus();
        dispatch({ type: actionTypes.SET_SERVER_STATUS, payload: isHealthy ? 'healthy' : 'unhealthy' });
      } catch (error) {
        dispatch({ type: actionTypes.SET_SERVER_STATUS, payload: 'error' });
      }
    };

    checkServerStatus();
  }, []);

  // Load available options on mount
  useEffect(() => {
    const loadAvailableOptions = async () => {
      try {
        const options = await apiService.getAvailableOptions();
        dispatch({ type: actionTypes.SET_AVAILABLE_OPTIONS, payload: options });
      } catch (error) {
        console.error('Failed to load available options:', error);
      }
    };

    loadAvailableOptions();
  }, []);

  // Actions
  const actions = {
    // Session management
    async createSession(userId) {
      try {
        dispatch({ type: actionTypes.SET_LOADING, payload: true });
        const session = await apiService.createSession(userId);
        
        dispatch({ type: actionTypes.ADD_SESSION, payload: session });
        dispatch({ type: actionTypes.SET_CURRENT_SESSION, payload: session });
        
        // Subscribe to WebSocket progress for this session
        webSocketService.subscribeToProgress(session.session_id);
        
        return session;
      } catch (error) {
        dispatch({ type: actionTypes.SET_ERROR, payload: error });
        throw error;
      } finally {
        dispatch({ type: actionTypes.SET_LOADING, payload: false });
      }
    },

    async getSessionStatus(sessionId) {
      try {
        const status = await apiService.getSessionStatus(sessionId);
        dispatch({
          type: actionTypes.UPDATE_SESSION,
          payload: { session_id: sessionId, ...status },
        });
        return status;
      } catch (error) {
        dispatch({ type: actionTypes.SET_ERROR, payload: error });
        throw error;
      }
    },

    // File uploads
    async uploadFaceImage(file, sessionId) {
      try {
        const result = await apiService.uploadFaceImage(file, sessionId);
        dispatch({
          type: actionTypes.UPDATE_UPLOAD_STATUS,
          payload: {
            session_id: sessionId,
            upload: { faceImage: result },
          },
        });
        return result;
      } catch (error) {
        dispatch({ type: actionTypes.SET_ERROR, payload: error });
        throw error;
      }
    },

    async uploadAudioFile(file, sessionId) {
      try {
        const result = await apiService.uploadAudioFile(file, sessionId);
        dispatch({
          type: actionTypes.UPDATE_UPLOAD_STATUS,
          payload: {
            session_id: sessionId,
            upload: { audioFile: result },
          },
        });
        return result;
      } catch (error) {
        dispatch({ type: actionTypes.SET_ERROR, payload: error });
        throw error;
      }
    },

    // Content generation
    async generateThumbnail(params) {
      try {
        dispatch({ type: actionTypes.SET_LOADING, payload: true });
        const result = await apiService.generateThumbnail(params);
        
        dispatch({
          type: actionTypes.ADD_GENERATED_CONTENT,
          payload: {
            type: 'thumbnails',
            content: result,
          },
        });

        // Update metadata with thumbnail generation results
        if (params.sessionId) {
          const thumbnailMetadata = {
            generated_content: {
              thumbnails: result.thumbnails || [],
              thumbnail_params: {
                prompt: params.prompt,
                style: params.style,
                quality: params.quality,
                count: params.count
              },
              thumbnail_stats: {
                generation_methods: result.generation_methods || [],
                enhanced_prompt: result.enhanced_prompt || params.prompt,
                total_generated: result.thumbnails?.length || 0
              },
              thumbnails_generated_at: new Date().toISOString()
            }
          };
          
          dispatch({
            type: actionTypes.UPDATE_METADATA,
            payload: { session_id: params.sessionId, metadata: thumbnailMetadata }
          });
        }
        
        return result;
      } catch (error) {
        dispatch({ type: actionTypes.SET_ERROR, payload: error });
        throw error;
      } finally {
        dispatch({ type: actionTypes.SET_LOADING, payload: false });
      }
    },

    async generateScript(params) {
      try {
        dispatch({ type: actionTypes.SET_LOADING, payload: true });
        const result = await apiService.generateScript(params);
        
        dispatch({
          type: actionTypes.ADD_GENERATED_CONTENT,
          payload: {
            type: 'scripts',
            content: result,
          },
        });

        // Update metadata with script generation results
        if (params.sessionId) {
          const scriptMetadata = {
            generated_content: {
              script: result.script || result.content,
              script_params: {
                topic: params.topic,
                duration: params.duration,
                tone: params.tone,
                emotion: params.emotion,
                audience: params.audience,
                content_type: params.contentType
              },
              script_stats: {
                word_count: result.word_count || 0,
                estimated_duration: result.estimated_duration || (params.duration * 60),
                generation_method: result.generation_method || 'unknown'
              },
              script_generated_at: new Date().toISOString()
            },
            target_duration: result.estimated_duration || (params.duration * 60)
          };
          
          dispatch({
            type: actionTypes.UPDATE_METADATA,
            payload: { session_id: params.sessionId, metadata: scriptMetadata }
          });
        }
        
        return result;
      } catch (error) {
        dispatch({ type: actionTypes.SET_ERROR, payload: error });
        throw error;
      } finally {
        dispatch({ type: actionTypes.SET_LOADING, payload: false });
      }
    },

    async generateVideo(params) {
      try {
        dispatch({ type: actionTypes.SET_LOADING, payload: true });
        const result = await apiService.generateVideo(params);
        
        dispatch({
          type: actionTypes.ADD_GENERATED_CONTENT,
          payload: {
            type: 'videos',
            content: result,
          },
        });
        
        return result;
      } catch (error) {
        dispatch({ type: actionTypes.SET_ERROR, payload: error });
        throw error;
      } finally {
        dispatch({ type: actionTypes.SET_LOADING, payload: false });
      }
    },

    // Processing
    async startProcessing(sessionId, config) {
      try {
        const result = await apiService.startProcessing(sessionId, config);
        return result;
      } catch (error) {
        dispatch({ type: actionTypes.SET_ERROR, payload: error });
        throw error;
      }
    },

    // Utility actions
    clearError() {
      dispatch({ type: actionTypes.CLEAR_ERROR });
    },

    setUser(user) {
      dispatch({ type: actionTypes.SET_USER, payload: user });
    },

    // Download file
    async downloadFile(sessionId, filename) {
      try {
        return await apiService.downloadFile(sessionId, filename);
      } catch (error) {
        dispatch({ type: actionTypes.SET_ERROR, payload: error });
        throw error;
      }
    },

    // Metadata management
    updateMetadata(sessionId, metadata) {
      dispatch({
        type: actionTypes.UPDATE_METADATA,
        payload: { session_id: sessionId, metadata }
      });
    },

    syncMetadata(sessionId, metadata, timestamp = null) {
      dispatch({
        type: actionTypes.SYNC_METADATA,
        payload: { 
          session_id: sessionId, 
          metadata,
          timestamp: timestamp || new Date().toISOString()
        }
      });
    },

    getSessionMetadata(sessionId) {
      return state.sessions[sessionId]?.metadata || {};
    },

    initializeMetadata(sessionId, metadata) {
      dispatch({
        type: actionTypes.INITIALIZE_METADATA,
        payload: { session_id: sessionId, metadata }
      });
    },
  };

  return (
    <AppContext.Provider value={{ state, actions }}>
      {children}
    </AppContext.Provider>
  );
};

export default AppContext;