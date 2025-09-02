import React, { useEffect, useState } from 'react';
import { CheckCircle, Clock, Play, Sparkles, Video, Image, Mic, FileText } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { useApp } from '@/contexts/AppContext';

const ProgressTracker = ({ sessionId, onComplete, onError }) => {
  const { state } = useApp();
  const [stages, setStages] = useState([
    {
      id: 'script_generation',
      name: 'Script Generation',
      description: 'Generating script content with Gemini AI',
      icon: <FileText className="w-5 h-5" />,
      status: 'pending',
      progress: 0,
      estimatedTime: 30,
      actualTime: 0,
    },
    {
      id: 'voice_cloning',
      name: 'Voice Cloning',
      description: 'Creating voice with XTTS technology',
      icon: <Mic className="w-5 h-5" />,
      status: 'pending',
      progress: 0,
      estimatedTime: 30,
      actualTime: 0,
    },
    {
      id: 'video_generation',
      name: 'Video Generation',
      description: 'Creating talking head video with SadTalker',
      icon: <Video className="w-5 h-5" />,
      status: 'pending',
      progress: 0,
      estimatedTime: 60,
      actualTime: 0,
    },
    {
      id: 'video_enhancement',
      name: 'Video Enhancement',
      description: 'Enhancing video quality with Real-ESRGAN & CodeFormer',
      icon: <Sparkles className="w-5 h-5" />,
      status: 'pending',
      progress: 0,
      estimatedTime: 90,
      actualTime: 0,
    },
    {
      id: 'final_assembly',
      name: 'Final Assembly',
      description: 'Assembling final video with audio synchronization',
      icon: <Play className="w-5 h-5" />,
      status: 'pending',
      progress: 0,
      estimatedTime: 30,
      actualTime: 0,
    },
  ]);

  const [overallProgress, setOverallProgress] = useState(0);
  const [currentStage, setCurrentStage] = useState('script_generation');
  const [startTime, setStartTime] = useState(null);
  const [totalEstimatedTime, setTotalEstimatedTime] = useState(240); // 4 minutes
  const [isProcessing, setIsProcessing] = useState(false);

  // Listen for progress updates from the app context
  useEffect(() => {
    if (!sessionId || !state.progress[sessionId]) return;

    const progressData = state.progress[sessionId];
    
    // Update overall progress
    if (progressData.overall_progress !== undefined) {
      setOverallProgress(progressData.overall_progress);
    }

    // Update current stage
    if (progressData.current_stage) {
      setCurrentStage(progressData.current_stage);
    }

    // Update specific stage progress
    if (progressData.stage_name && progressData.progress_percent !== undefined) {
      setStages(prevStages => 
        prevStages.map(stage => {
          if (stage.id === progressData.stage_name) {
            return {
              ...stage,
              progress: progressData.progress_percent,
              status: progressData.progress_percent === 100 ? 'completed' : 'in_progress',
              actualTime: progressData.elapsed_time || 0,
            };
          }
          return stage;
        })
      );
    }

    // Handle processing status
    if (progressData.status === 'processing' && !isProcessing) {
      setIsProcessing(true);
      setStartTime(Date.now());
    } else if (progressData.status === 'completed') {
      setIsProcessing(false);
      setStages(prevStages => 
        prevStages.map(stage => ({
          ...stage,
          status: 'completed',
          progress: 100,
        }))
      );
      onComplete && onComplete();
    } else if (progressData.status === 'error') {
      setIsProcessing(false);
      onError && onError(progressData.error);
    }
  }, [sessionId, state.progress, isProcessing, onComplete, onError]);

  // Calculate elapsed time
  const [elapsedTime, setElapsedTime] = useState(0);
  useEffect(() => {
    if (!startTime || !isProcessing) return;

    const interval = setInterval(() => {
      setElapsedTime(Math.floor((Date.now() - startTime) / 1000));
    }, 1000);

    return () => clearInterval(interval);
  }, [startTime, isProcessing]);

  // Format time
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Get status color
  const getStatusColor = (status) => {
    switch (status) {
      case 'completed': return 'bg-green-500';
      case 'in_progress': return 'bg-blue-500';
      case 'pending': return 'bg-gray-300';
      case 'error': return 'bg-red-500';
      default: return 'bg-gray-300';
    }
  };

  // Get status badge variant
  const getStatusBadge = (status) => {
    switch (status) {
      case 'completed': return 'default';
      case 'in_progress': return 'secondary';
      case 'pending': return 'outline';
      case 'error': return 'destructive';
      default: return 'outline';
    }
  };

  // Calculate estimated remaining time
  const calculateRemainingTime = () => {
    if (!isProcessing || overallProgress === 0) return totalEstimatedTime;
    
    const progressRatio = overallProgress / 100;
    const estimatedTotalTime = elapsedTime / progressRatio;
    return Math.max(0, estimatedTotalTime - elapsedTime);
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Video className="w-5 h-5" />
          Video Generation Progress
        </CardTitle>
        
        {/* Overall Progress */}
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span>Overall Progress</span>
            <span>{Math.round(overallProgress)}%</span>
          </div>
          <Progress value={overallProgress} className="h-2" />
          
          {/* Time Information */}
          <div className="flex justify-between text-xs text-gray-500">
            <span>Elapsed: {formatTime(elapsedTime)}</span>
            <span>
              {isProcessing ? (
                `Est. remaining: ${formatTime(calculateRemainingTime())}`
              ) : (
                `Total est. time: ${formatTime(totalEstimatedTime)}`
              )}
            </span>
          </div>
        </div>
      </CardHeader>

      <CardContent>
        <div className="space-y-4">
          {stages.map((stage, index) => (
            <div key={stage.id} className="flex items-start gap-4">
              {/* Stage Icon & Status */}
              <div className="flex flex-col items-center">
                <div className={`
                  w-10 h-10 rounded-full flex items-center justify-center text-white transition-all duration-300
                  ${getStatusColor(stage.status)}
                  ${stage.status === 'in_progress' ? 'animate-pulse' : ''}
                `}>
                  {stage.status === 'completed' ? (
                    <CheckCircle className="w-5 h-5" />
                  ) : stage.status === 'in_progress' ? (
                    <div className="w-5 h-5 animate-spin rounded-full border-2 border-white border-t-transparent" />
                  ) : (
                    stage.icon
                  )}
                </div>
                
                {/* Connector Line */}
                {index < stages.length - 1 && (
                  <div className={`
                    w-0.5 h-12 mt-2 transition-all duration-300
                    ${stage.status === 'completed' ? 'bg-green-500' : 'bg-gray-200'}
                  `} />
                )}
              </div>

              {/* Stage Content */}
              <div className="flex-1 space-y-2">
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-semibold text-gray-900">{stage.name}</h4>
                    <p className="text-sm text-gray-500">{stage.description}</p>
                  </div>
                  <Badge variant={getStatusBadge(stage.status)}>
                    {stage.status.replace('_', ' ')}
                  </Badge>
                </div>

                {/* Stage Progress */}
                {stage.status === 'in_progress' && (
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span>Progress</span>
                      <span>{Math.round(stage.progress)}%</span>
                    </div>
                    <Progress value={stage.progress} className="h-1" />
                  </div>
                )}

                {/* Time Information */}
                <div className="flex justify-between text-xs text-gray-400">
                  <span>Est. time: {formatTime(stage.estimatedTime)}</span>
                  {stage.actualTime > 0 && (
                    <span>Actual: {formatTime(stage.actualTime)}</span>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Status Message */}
        {isProcessing && (
          <div className="mt-6 p-4 bg-blue-50 rounded-lg">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 animate-spin rounded-full border-2 border-blue-600 border-t-transparent" />
              <span className="text-sm font-medium text-blue-900">
                Processing your video...
              </span>
            </div>
            <p className="text-xs text-blue-700 mt-1">
              Current stage: {stages.find(s => s.id === currentStage)?.name || 'Processing'}
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default ProgressTracker;