import React, { useState, useRef, useCallback } from 'react';
import { Upload, X, CheckCircle, AlertCircle, Image, Music } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { useApp } from '@/contexts/AppContext';

const FileUploader = ({ 
  type = 'image', 
  accept = 'image/*', 
  maxSize = 50 * 1024 * 1024, // 50MB
  onUploadSuccess,
  onUploadError,
  sessionId 
}) => {
  const { actions } = useApp();
  const [dragActive, setDragActive] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState('idle'); // idle, uploading, success, error
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  // File validation
  const validateFile = useCallback((file) => {
    // Check file size
    if (file.size > maxSize) {
      throw new Error(`File size must be less than ${Math.round(maxSize / (1024 * 1024))}MB`);
    }

    // Check file type
    if (type === 'image') {
      const allowedTypes = [
        'image/jpeg', 'image/jpg', 'image/pjpeg', // JPEG variations
        'image/png', 'image/x-png', // PNG variations
        'image/gif', // GIF
        'image/bmp', 'image/x-bmp', 'image/x-bitmap' // BMP variations
      ];
      
      // Also check file extension as fallback for MIME type issues
      const fileName = file.name.toLowerCase();
      const allowedExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp'];
      const hasValidExtension = allowedExtensions.some(ext => fileName.endsWith(ext));
      
      if (!allowedTypes.includes(file.type) && !hasValidExtension) {
        throw new Error('Please upload a valid image file (JPEG, PNG, GIF, BMP)');
      }
    } else if (type === 'audio') {
      const allowedTypes = [
        'audio/wav', 'audio/wave', 'audio/x-wav', // WAV variations
        'audio/mp3', 'audio/mpeg', 'audio/mp4', // MP3 variations
        'audio/flac', 'audio/x-flac', // FLAC variations
        'audio/ogg', 'audio/ogg-audio', // OGG variations
        'audio/m4a', 'audio/mp4a-latm', 'audio/x-m4a' // M4A variations
      ];
      
      // Also check file extension as fallback for MIME type issues
      const fileName = file.name.toLowerCase();
      const allowedExtensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a'];
      const hasValidExtension = allowedExtensions.some(ext => fileName.endsWith(ext));
      
      if (!allowedTypes.includes(file.type) && !hasValidExtension) {
        throw new Error('Please upload a valid audio file (WAV, MP3, FLAC, OGG, M4A)');
      }
    }

    return true;
  }, [type, maxSize]);

  // Handle file upload
  const handleFileUpload = useCallback(async (file) => {
    try {
      validateFile(file);
      
      setUploadStatus('uploading');
      setUploadProgress(0);
      setError(null);
      
      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 200);

      let result;
      if (type === 'image') {
        result = await actions.uploadFaceImage(file, sessionId);
      } else if (type === 'audio') {
        result = await actions.uploadAudioFile(file, sessionId);
      }

      clearInterval(progressInterval);
      setUploadProgress(100);
      setUploadStatus('success');
      setUploadedFile(file);
      
      onUploadSuccess && onUploadSuccess(result, file);
      
    } catch (error) {
      setUploadStatus('error');
      setError(error.message);
      setUploadProgress(0);
      onUploadError && onUploadError(error);
    }
  }, [actions, sessionId, type, validateFile, onUploadSuccess, onUploadError]);

  // Drag and drop handlers
  const handleDragEnter = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  }, []);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFileUpload(files[0]);
    }
  }, [handleFileUpload]);

  // File input change handler
  const handleInputChange = useCallback((e) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileUpload(files[0]);
    }
  }, [handleFileUpload]);

  // Remove uploaded file
  const handleRemoveFile = useCallback(() => {
    setUploadedFile(null);
    setUploadStatus('idle');
    setUploadProgress(0);
    setError(null);
    if (inputRef.current) {
      inputRef.current.value = '';
    }
  }, []);

  // Open file dialog
  const openFileDialog = useCallback(() => {
    inputRef.current?.click();
  }, []);

  // Get file type icon
  const getFileIcon = () => {
    if (type === 'image') return <Image className="w-8 h-8" />;
    if (type === 'audio') return <Music className="w-8 h-8" />;
    return <Upload className="w-8 h-8" />;
  };

  // Get file type label
  const getFileTypeLabel = () => {
    if (type === 'image') return 'face image';
    if (type === 'audio') return 'audio file';
    return 'file';
  };

  return (
    <div className="w-full">
      <Card className="relative overflow-hidden">
        <CardContent className="p-0">
          {/* Upload Area */}
          <div
            className={`
              relative border-2 border-dashed transition-all duration-300 p-8
              ${dragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300'}
              ${uploadStatus === 'success' ? 'border-green-500 bg-green-50' : ''}
              ${uploadStatus === 'error' ? 'border-red-500 bg-red-50' : ''}
              ${uploadStatus === 'uploading' ? 'border-yellow-500 bg-yellow-50' : ''}
              hover:border-gray-400 cursor-pointer
            `}
            onDragEnter={handleDragEnter}
            onDragLeave={handleDragLeave}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={openFileDialog}
          >
            {/* Hidden file input */}
            <input
              ref={inputRef}
              type="file"
              accept={accept}
              onChange={handleInputChange}
              className="hidden"
            />

            {/* Upload Status Display */}
            <div className="text-center">
              {uploadStatus === 'idle' && (
                <>
                  <div className="text-gray-400 mb-4">
                    {getFileIcon()}
                  </div>
                  <h3 className="text-lg font-semibold text-gray-700 mb-2">
                    Upload {getFileTypeLabel()}
                  </h3>
                  <p className="text-sm text-gray-500 mb-4">
                    Drag and drop your {getFileTypeLabel()} here, or click to browse
                  </p>
                  <p className="text-xs text-gray-400">
                    Max size: {Math.round(maxSize / (1024 * 1024))}MB
                  </p>
                </>
              )}

              {uploadStatus === 'uploading' && (
                <>
                  <div className="text-yellow-500 mb-4">
                    <Upload className="w-8 h-8 animate-pulse" />
                  </div>
                  <h3 className="text-lg font-semibold text-gray-700 mb-2">
                    Uploading...
                  </h3>
                  <Progress value={uploadProgress} className="mb-2" />
                  <p className="text-sm text-gray-500">
                    {uploadProgress}% complete
                  </p>
                </>
              )}

              {uploadStatus === 'success' && uploadedFile && (
                <>
                  <div className="text-green-500 mb-4">
                    <CheckCircle className="w-8 h-8" />
                  </div>
                  <h3 className="text-lg font-semibold text-gray-700 mb-2">
                    Upload successful!
                  </h3>
                  <p className="text-sm text-gray-500 mb-4">
                    {uploadedFile.name}
                  </p>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleRemoveFile();
                    }}
                  >
                    <X className="w-4 h-4 mr-2" />
                    Remove
                  </Button>
                </>
              )}

              {uploadStatus === 'error' && (
                <>
                  <div className="text-red-500 mb-4">
                    <AlertCircle className="w-8 h-8" />
                  </div>
                  <h3 className="text-lg font-semibold text-gray-700 mb-2">
                    Upload failed
                  </h3>
                  <p className="text-sm text-red-500 mb-4">
                    {error}
                  </p>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleRemoveFile();
                    }}
                  >
                    Try again
                  </Button>
                </>
              )}
            </div>
          </div>

          {/* File info display */}
          {uploadedFile && uploadStatus === 'success' && (
            <div className="p-4 border-t bg-gray-50">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-700">
                    {uploadedFile.name}
                  </p>
                  <p className="text-xs text-gray-500">
                    {(uploadedFile.size / (1024 * 1024)).toFixed(2)} MB
                  </p>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleRemoveFile}
                >
                  <X className="w-4 h-4" />
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Error Alert */}
      {error && uploadStatus === 'error' && (
        <Alert className="mt-4" variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
    </div>
  );
};

export default FileUploader;