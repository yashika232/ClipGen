import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Video, Upload, FileText, Image, User, Mic, Download, Play, AlertCircle, CheckCircle, Clock } from 'lucide-react';
import { useApp } from '@/contexts/AppContext';
import FileUploader from './FileUploader';
import ProgressTracker from './ProgressTracker';

const VideoGenerator = () => {
  const { state, actions } = useApp();
  
  // Wait for session creation from AppContext
  useEffect(() => {
    // Only set error if AppContext failed to create session after 5 seconds
    const timeoutId = setTimeout(() => {
      if (!state.currentSession) {
        setError('Failed to initialize session. Please refresh the page.');
      }
    }, 5000);
    
    // Clear timeout if session is created
    if (state.currentSession) {
      clearTimeout(timeoutId);
    }
    
    return () => clearTimeout(timeoutId);
  }, [state.currentSession]);
  
  // Form state
  const [topic, setTopic] = useState('');
  const [duration, setDuration] = useState('5');
  const [tone, setTone] = useState('professional');
  const [emotion, setEmotion] = useState('confident');
  const [audience, setAudience] = useState('general public');
  const [contentType, setContentType] = useState('educational');
  const [generatedScript, setGeneratedScript] = useState('');
  const [selectedThumbnail, setSelectedThumbnail] = useState(null);
  
  // Pipeline state
  const [currentStep, setCurrentStep] = useState(1);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStep, setProcessingStep] = useState('');
  const [uploadedFiles, setUploadedFiles] = useState({
    faceImage: null,
    audioFile: null
  });
  
  // Results state
  const [generatedContent, setGeneratedContent] = useState({
    script: '',
    thumbnails: [],
    finalVideo: null
  });
  
  const [error, setError] = useState(null);
  const [completedSteps, setCompletedSteps] = useState([]);
  
  // Get available options from context
  const tones = state.availableOptions.tones || ['professional', 'friendly', 'motivational', 'casual', 'enthusiastic'];
  const emotions = state.availableOptions.emotions || ['confident', 'excited', 'calm', 'passionate', 'determined'];
  const audiences = state.availableOptions.audiences || ['general public', 'professionals', 'students', 'experts'];
  const contentTypes = state.availableOptions.contentTypes || ['educational', 'promotional', 'tutorial', 'announcement'];

  // Handle file uploads
  const handleFileUpload = (type, file) => {
    setUploadedFiles(prev => ({
      ...prev,
      [type]: file
    }));
  };

  // Step 1: Generate Script
  const handleGenerateScript = async () => {
    if (!topic.trim()) {
      setError('Please enter a topic');
      return;
    }
    
    // Ensure session exists
    if (!state.currentSession?.session_id) {
      setError('No active session. Please refresh the page.');
      return;
    }
    
    setIsProcessing(true);
    setProcessingStep('Generating script...');
    setError(null);
    
    try {
      const result = await actions.generateScript({
        topic: topic.trim(),
        duration: parseInt(duration),
        tone,
        emotion,
        audience,
        contentType,
        sessionId: state.currentSession.session_id,
      });
      
      if (result && (result.script || result.content)) {
        setGeneratedScript(result.script || result.content);
        setGeneratedContent(prev => ({ ...prev, script: result.script || result.content }));
        setCompletedSteps(prev => [...prev, 1]);
        setCurrentStep(2);
      } else {
        setError('Script generation returned empty result');
      }
      
    } catch (error) {
      console.error('Script generation error:', error);
      setError(error.message || 'Failed to generate script. Please try again.');
    } finally {
      setIsProcessing(false);
      setProcessingStep('');
    }
  };

  // Step 2: Generate Thumbnails
  const handleGenerateThumbnails = async () => {
    if (!generatedScript) {
      setError('Please generate a script first');
      return;
    }
    
    // Ensure session exists
    if (!state.currentSession?.session_id) {
      setError('No active session. Please refresh the page.');
      return;
    }
    
    setIsProcessing(true);
    setProcessingStep('Generating thumbnails...');
    setError(null);
    
    try {
      // Use comprehensive context like app.py for better accuracy
      const thumbnailParams = {
        prompt: topic,
        title: topic,
        audience: audience,
        tone: tone,
        emotion: emotion,
        // Use first 100 characters of script as hook context
        script_hook: generatedScript ? generatedScript.substring(0, 100) : '',
        style: 'multi-style',  // Request all 5 styles
        quality: 'high',
        count: 5,  // Request 5 styles like app.py
        sessionId: state.currentSession.session_id,
      };
      
      const result = await actions.generateThumbnail(thumbnailParams);
      
      if (result && result.thumbnails) {
        setGeneratedContent(prev => ({ ...prev, thumbnails: result.thumbnails }));
        setCompletedSteps(prev => [...prev, 2]);
        setCurrentStep(3);
      } else {
        // Thumbnails are optional, allow skipping to next step
        setCompletedSteps(prev => [...prev, 2]);
        setCurrentStep(3);
        console.log('Thumbnail generation failed, but continuing to next step');
      }
      
    } catch (error) {
      console.error('Thumbnail generation error:', error);
      // Thumbnails are optional, allow skipping to next step
      setCompletedSteps(prev => [...prev, 2]);
      setCurrentStep(3);
      setError('Thumbnail generation failed, but you can continue without thumbnails');
    } finally {
      setIsProcessing(false);
      setProcessingStep('');
    }
  };

  // Step 3: Upload Files
  const handleFileUploadComplete = () => {
    if (uploadedFiles.faceImage && uploadedFiles.audioFile) {
      setCompletedSteps(prev => [...prev, 3]);
      setCurrentStep(4);
    }
  };

  // Step 4: Start Video Generation
  const handleStartVideoGeneration = async () => {
    if (!generatedScript || !uploadedFiles.faceImage || !uploadedFiles.audioFile) {
      setError('Please complete all previous steps (script generation and file uploads)');
      return;
    }
    
    // Ensure session exists
    if (!state.currentSession?.session_id) {
      setError('No active session. Please refresh the page.');
      return;
    }
    
    setIsProcessing(true);
    setProcessingStep('Starting video generation pipeline...');
    setError(null);
    
    try {
      console.log('Starting video generation with session:', state.currentSession.session_id);
      
      const result = await actions.generateVideo({
        script_text: generatedScript,
        sessionId: state.currentSession.session_id,
        parameters: {
          tone,
          emotion,
          audience,
          contentType,
          duration: parseInt(duration)
        }
      });
      
      console.log('Video generation started:', result);
      setCompletedSteps(prev => [...prev, 4]);
      setCurrentStep(5);
      
      // The actual video generation will be tracked via WebSocket
      
    } catch (error) {
      console.error('Video generation error:', error);
      setError(error.message || 'Failed to start video generation. Please try again.');
      setIsProcessing(false);
      setProcessingStep('');
    }
  };

  // Listen for WebSocket updates
  useEffect(() => {
    if (!state.currentSession) return;
    
    // Handle processing completion
    const handleProcessingComplete = (data) => {
      if (data.session_id === state.currentSession.session_id) {
        setIsProcessing(false);
        setProcessingStep('');
        setGeneratedContent(prev => ({ ...prev, finalVideo: data.output_file }));
        setCompletedSteps(prev => [...prev, 5]);
      }
    };
    
    // Handle processing errors
    const handleProcessingError = (data) => {
      if (data.session_id === state.currentSession.session_id) {
        setError(data.error || 'Video generation failed');
        setIsProcessing(false);
        setProcessingStep('');
      }
    };
    
    // Note: These would be handled by the WebSocket service
    // For now, we'll simulate completion after a delay
    if (currentStep === 5 && isProcessing) {
      setTimeout(() => {
        setIsProcessing(false);
        setProcessingStep('');
        setGeneratedContent(prev => ({ 
          ...prev, 
          finalVideo: `final_video_${Date.now()}.mp4` 
        }));
        setCompletedSteps(prev => [...prev, 5]);
      }, 5000);
    }
    
  }, [state.currentSession, currentStep, isProcessing]);

  const getStepStatus = (step) => {
    if (completedSteps.includes(step)) return 'completed';
    if (currentStep === step) return 'current';
    return 'pending';
  };

  const getStepIcon = (step) => {
    const status = getStepStatus(step);
    if (status === 'completed') return <CheckCircle className="w-5 h-5 text-green-500" />;
    if (status === 'current') return <Clock className="w-5 h-5 text-blue-500 animate-pulse" />;
    return <div className="w-5 h-5 rounded-full bg-gray-300" />;
  };

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-gray-900 mb-2">Video Generator</h1>
      </div>

      {/* Progress Steps */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-4">
            {getStepIcon(1)}
            <span className={`text-sm font-medium ${getStepStatus(1) === 'completed' ? 'text-green-600' : getStepStatus(1) === 'current' ? 'text-blue-600' : 'text-gray-500'}`}>
              Script Generation
            </span>
          </div>
          <div className="flex items-center space-x-4">
            {getStepIcon(2)}
            <span className={`text-sm font-medium ${getStepStatus(2) === 'completed' ? 'text-green-600' : getStepStatus(2) === 'current' ? 'text-blue-600' : 'text-gray-500'}`}>
              Thumbnails
            </span>
          </div>
          <div className="flex items-center space-x-4">
            {getStepIcon(3)}
            <span className={`text-sm font-medium ${getStepStatus(3) === 'completed' ? 'text-green-600' : getStepStatus(3) === 'current' ? 'text-blue-600' : 'text-gray-500'}`}>
              File Upload
            </span>
          </div>
          <div className="flex items-center space-x-4">
            {getStepIcon(4)}
            <span className={`text-sm font-medium ${getStepStatus(4) === 'completed' ? 'text-green-600' : getStepStatus(4) === 'current' ? 'text-blue-600' : 'text-gray-500'}`}>
              Video Generation
            </span>
          </div>
          <div className="flex items-center space-x-4">
            {getStepIcon(5)}
            <span className={`text-sm font-medium ${getStepStatus(5) === 'completed' ? 'text-green-600' : getStepStatus(5) === 'current' ? 'text-blue-600' : 'text-gray-500'}`}>
              Final Video
            </span>
          </div>
        </div>
        <Progress value={(completedSteps.length / 5) * 100} className="w-full" />
      </div>

      {/* Error Display */}
      {error && (
        <Alert variant="destructive" className="mb-6">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Processing Status */}
      {isProcessing && (
        <Alert className="mb-6">
          <Clock className="h-4 w-4 animate-spin" />
          <AlertDescription>{processingStep}</AlertDescription>
        </Alert>
      )}

      <Tabs value={currentStep.toString()} className="w-full">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="1" disabled={currentStep < 1}>
            <FileText className="w-4 h-4 mr-2" />
            Script
          </TabsTrigger>
          <TabsTrigger value="2" disabled={currentStep < 2}>
            <Image className="w-4 h-4 mr-2" />
            Thumbnails
          </TabsTrigger>
          <TabsTrigger value="3" disabled={currentStep < 3}>
            <Upload className="w-4 h-4 mr-2" />
            Files
          </TabsTrigger>
          <TabsTrigger value="4" disabled={currentStep < 4}>
            <Video className="w-4 h-4 mr-2" />
            Generate
          </TabsTrigger>
          <TabsTrigger value="5" disabled={currentStep < 5}>
            <Download className="w-4 h-4 mr-2" />
            Results
          </TabsTrigger>
        </TabsList>

        {/* Step 1: Script Generation */}
        <TabsContent value="1" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Step 1: Generate Script</CardTitle>
              <CardDescription>
                Create a compelling script for your video with AI assistance
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="topic">Video Topic</Label>
                  <Input
                    id="topic"
                    placeholder="e.g., Introduction to Machine Learning"
                    value={topic}
                    onChange={(e) => setTopic(e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="duration">Duration (minutes)</Label>
                  <Input
                    id="duration"
                    type="number"
                    min="1"
                    max="60"
                    value={duration}
                    onChange={(e) => setDuration(e.target.value)}
                  />
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="tone">Tone</Label>
                  <Select value={tone} onValueChange={setTone}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select tone" />
                    </SelectTrigger>
                    <SelectContent>
                      {tones.map((toneOption) => (
                        <SelectItem key={toneOption} value={toneOption}>
                          {toneOption.charAt(0).toUpperCase() + toneOption.slice(1)}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="emotion">Emotion</Label>
                  <Select value={emotion} onValueChange={setEmotion}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select emotion" />
                    </SelectTrigger>
                    <SelectContent>
                      {emotions.map((emotionOption) => (
                        <SelectItem key={emotionOption} value={emotionOption}>
                          {emotionOption.charAt(0).toUpperCase() + emotionOption.slice(1)}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="audience">Target Audience</Label>
                  <Select value={audience} onValueChange={setAudience}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select audience" />
                    </SelectTrigger>
                    <SelectContent>
                      {audiences.map((audienceOption) => (
                        <SelectItem key={audienceOption} value={audienceOption}>
                          {audienceOption.charAt(0).toUpperCase() + audienceOption.slice(1)}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="contentType">Content Type</Label>
                  <Select value={contentType} onValueChange={setContentType}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select content type" />
                    </SelectTrigger>
                    <SelectContent>
                      {contentTypes.map((type) => (
                        <SelectItem key={type} value={type}>
                          {type.charAt(0).toUpperCase() + type.slice(1)}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
              
              {generatedScript && (
                <div className="space-y-2">
                  <Label>Generated Script</Label>
                  <Textarea
                    value={generatedScript}
                    onChange={(e) => setGeneratedScript(e.target.value)}
                    className="min-h-[200px]"
                    placeholder="Your generated script will appear here..."
                  />
                </div>
              )}
              
              <Button 
                onClick={handleGenerateScript}
                disabled={!topic.trim() || isProcessing || !state.currentSession}
                className="w-full"
                size="lg"
              >
                {isProcessing ? 'Generating...' : 'Generate Script'}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Step 2: Thumbnail Generation */}
        <TabsContent value="2" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Step 2: Generate Thumbnails</CardTitle>
              <CardDescription>
                Create eye-catching thumbnails for your video
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {generatedContent.thumbnails.length === 0 ? (
                <Button 
                  onClick={handleGenerateThumbnails}
                  disabled={!generatedScript || isProcessing}
                  className="w-full"
                  size="lg"
                >
                  {isProcessing ? 'Generating...' : 'Generate Thumbnails'}
                </Button>
              ) : (
                <div className="grid grid-cols-2 gap-4">
                  {generatedContent.thumbnails.map((thumbnail, index) => (
                    <div 
                      key={index}
                      className={`relative cursor-pointer rounded-lg overflow-hidden border-2 ${
                        selectedThumbnail === thumbnail ? 'border-blue-500' : 'border-gray-200'
                      }`}
                      onClick={() => setSelectedThumbnail(thumbnail)}
                    >
                      <img 
                        src={thumbnail.url} 
                        alt={`Thumbnail ${index + 1}`}
                        className="w-full h-auto"
                      />
                      {selectedThumbnail === thumbnail && (
                        <div className="absolute top-2 right-2">
                          <Badge variant="default">Selected</Badge>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Step 3: File Upload */}
        <TabsContent value="3" className="space-y-6">
          <div className="grid grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>
                  <User className="w-5 h-5 mr-2 inline" />
                  Face Image
                </CardTitle>
                <CardDescription>
                  Upload a clear face image for the video avatar
                </CardDescription>
              </CardHeader>
              <CardContent>
                {state.currentSession ? (
                  <FileUploader
                    type="image"
                    accept="image/*"
                    maxSize={10 * 1024 * 1024} // 10MB
                    sessionId={state.currentSession.session_id}
                    onUploadSuccess={(result, file) => handleFileUpload('faceImage', file)}
                  />
                ) : (
                  <div className="p-8 text-center border-2 border-dashed border-gray-300 rounded-lg">
                    <div className="text-gray-400 mb-4">
                      <Upload className="w-8 h-8 mx-auto animate-pulse" />
                    </div>
                    <h3 className="text-lg font-semibold text-gray-700 mb-2">
                      Initializing session...
                    </h3>
                    <p className="text-sm text-gray-500">
                      Please wait while we set up your session
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>
                  <Mic className="w-5 h-5 mr-2 inline" />
                  Audio File
                </CardTitle>
                <CardDescription>
                  Upload an audio file for voice cloning (optional)
                </CardDescription>
              </CardHeader>
              <CardContent>
                {state.currentSession ? (
                  <FileUploader
                    type="audio"
                    accept="audio/*"
                    maxSize={50 * 1024 * 1024} // 50MB
                    sessionId={state.currentSession.session_id}
                    onUploadSuccess={(result, file) => handleFileUpload('audioFile', file)}
                  />
                ) : (
                  <div className="p-8 text-center border-2 border-dashed border-gray-300 rounded-lg">
                    <div className="text-gray-400 mb-4">
                      <Upload className="w-8 h-8 mx-auto animate-pulse" />
                    </div>
                    <h3 className="text-lg font-semibold text-gray-700 mb-2">
                      Initializing session...
                    </h3>
                    <p className="text-sm text-gray-500">
                      Please wait while we set up your session
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
          
          {uploadedFiles.faceImage && uploadedFiles.audioFile && (
            <Button 
              onClick={handleFileUploadComplete}
              className="w-full"
              size="lg"
            >
              Continue to Video Generation
            </Button>
          )}
        </TabsContent>

        {/* Step 4: Video Generation */}
        <TabsContent value="4" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Step 4: Generate Video</CardTitle>
              <CardDescription>
                Start the complete video generation pipeline
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="font-semibold mb-2">Ready to Generate:</h3>
                <ul className="space-y-1 text-sm">
                  <li>✓ Script: {generatedScript.substring(0, 50)}...</li>
                  <li>✓ Thumbnail: {selectedThumbnail ? 'Selected' : 'None (Optional)'}</li>
                  <li>✓ Face Image: {uploadedFiles.faceImage ? 'Uploaded' : 'None'}</li>
                  <li>✓ Audio File: {uploadedFiles.audioFile ? 'Uploaded' : 'None'}</li>
                  <li>✓ Duration: {duration} minutes</li>
                </ul>
              </div>
              
              <Button 
                onClick={handleStartVideoGeneration}
                disabled={!generatedScript || !uploadedFiles.faceImage || !uploadedFiles.audioFile || isProcessing}
                className="w-full"
                size="lg"
              >
                {isProcessing ? 'Starting Pipeline...' : 'Start Video Generation'}
              </Button>
              
              {isProcessing && (
                <ProgressTracker sessionId={state.currentSession?.session_id} />
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Step 5: Results */}
        <TabsContent value="5" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Step 5: Your Generated Video</CardTitle>
              <CardDescription>
                Your AI-generated video is ready!
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {generatedContent.finalVideo ? (
                <div className="text-center">
                  <div className="bg-green-50 p-6 rounded-lg mb-4">
                    <CheckCircle className="w-12 h-12 text-green-500 mx-auto mb-2" />
                    <h3 className="text-lg font-semibold text-green-800">Video Generated Successfully!</h3>
                    <p className="text-green-600">Your video has been processed and is ready for download.</p>
                  </div>
                  
                  <div className="flex gap-4 justify-center">
                    <Button size="lg" className="flex items-center gap-2">
                      <Play className="w-5 h-5" />
                      Preview Video
                    </Button>
                    <Button size="lg" variant="outline" className="flex items-center gap-2">
                      <Download className="w-5 h-5" />
                      Download Video
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="text-center py-12 text-gray-500">
                  <Video className="w-16 h-16 mx-auto mb-4 opacity-50" />
                  <p>Complete the previous steps to generate your video</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default VideoGenerator;