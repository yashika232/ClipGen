import React from 'react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { AlertCircle, RefreshCw } from 'lucide-react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { 
      hasError: false, 
      error: null, 
      errorInfo: null,
      errorType: 'unknown'
    };
  }

  static getDerivedStateFromError(error) {
    // Analyze error type for better user messaging
    let errorType = 'unknown';
    if (error.message?.includes('Network')) {
      errorType = 'network';
    } else if (error.message?.includes('session')) {
      errorType = 'session';
    } else if (error.message?.includes('timeout')) {
      errorType = 'timeout';
    } else if (error.name === 'ChunkLoadError') {
      errorType = 'chunk_load';
    }

    return { hasError: true, error, errorType };
  }

  componentDidCatch(error, errorInfo) {
    console.error('React Error Boundary caught an error:', error, errorInfo);
    this.setState({ errorInfo });
    
    // Log to external service if needed
    // logErrorToService(error, errorInfo);
  }

  handleRefresh = () => {
    window.location.reload();
  };

  handleRetry = () => {
    this.setState({ hasError: false, error: null, errorInfo: null, errorType: 'unknown' });
  };

  getErrorMessage() {
    const { errorType, error } = this.state;
    
    switch (errorType) {
      case 'network':
        return {
          title: 'Network Connection Error',
          message: 'Unable to connect to the video generation server. Please check your internet connection and try again.',
          actionText: 'Retry Connection'
        };
      case 'session':
        return {
          title: 'Session Error',
          message: 'Your session has expired or is invalid. Please refresh the page to start a new session.',
          actionText: 'Start New Session'
        };
      case 'timeout':
        return {
          title: 'Request Timeout',
          message: 'The server is taking too long to respond. This might be due to high load or processing time.',
          actionText: 'Try Again'
        };
      case 'chunk_load':
        return {
          title: 'Loading Error',
          message: 'Failed to load application resources. Please refresh the page.',
          actionText: 'Refresh Page'
        };
      default:
        return {
          title: 'Something went wrong',
          message: error?.message || 'An unexpected error occurred in the video generator. Please try refreshing the page.',
          actionText: 'Refresh Page'
        };
    }
  }

  render() {
    if (this.state.hasError) {
      const errorDetails = this.getErrorMessage();
      
      return (
        <div className="min-h-screen flex items-center justify-center bg-gray-50 p-4">
          <div className="max-w-md w-full">
            <Alert className="border-red-200 bg-red-50">
              <AlertCircle className="h-4 w-4 text-red-600" />
              <AlertDescription className="text-red-800">
                <div className="mb-4">
                  <h3 className="font-semibold mb-2">{errorDetails.title}</h3>
                  <p className="text-sm mb-3">
                    {errorDetails.message}
                  </p>
                  {process.env.NODE_ENV === 'development' && (
                    <details className="text-xs bg-red-100 p-2 rounded mb-3">
                      <summary className="cursor-pointer font-medium">Technical Details</summary>
                      <pre className="mt-2 whitespace-pre-wrap">
                        {this.state.error?.stack || this.state.error?.message}
                      </pre>
                    </details>
                  )}
                </div>
                <div className="flex gap-2">
                  {this.state.errorType !== 'chunk_load' && (
                    <Button 
                      onClick={this.handleRetry}
                      variant="outline"
                      className="flex-1"
                    >
                      Try Again
                    </Button>
                  )}
                  <Button 
                    onClick={this.handleRefresh}
                    className="flex-1 bg-red-600 hover:bg-red-700"
                  >
                    <RefreshCw className="w-4 h-4 mr-2" />
                    {errorDetails.actionText}
                  </Button>
                </div>
              </AlertDescription>
            </Alert>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;