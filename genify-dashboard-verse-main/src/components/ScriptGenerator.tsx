
import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { Copy, Download, Sparkles, AlertCircle, CheckCircle, FileText, Mic } from 'lucide-react';
import { useApp } from '@/contexts/AppContext';

const ScriptGenerator = () => {
  const { state, actions } = useApp();
  const [topic, setTopic] = useState('');
  const [duration, setDuration] = useState('5');
  const [tone, setTone] = useState('professional');
  const [emotion, setEmotion] = useState('confident');
  const [audience, setAudience] = useState('general public');
  const [contentType, setContentType] = useState('educational');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedScript, setGeneratedScript] = useState('');
  const [generationProgress, setGenerationProgress] = useState(0);
  const [error, setError] = useState(null);
  const [wordCount, setWordCount] = useState(0);

  // Get available options from context
  const tones = state.availableOptions.tones || ['professional', 'friendly', 'motivational', 'casual', 'enthusiastic', 'educational'];
  const emotions = state.availableOptions.emotions || ['inspired', 'confident', 'curious', 'excited', 'calm', 'passionate', 'determined'];
  const audiences = state.availableOptions.audiences || ['junior engineers', 'students', 'professionals', 'general public', 'experts', 'beginners'];
  const contentTypes = state.availableOptions.contentTypes || ['educational', 'promotional', 'tutorial', 'announcement', 'overview'];

  // Update word count when script changes
  useEffect(() => {
    if (generatedScript) {
      const words = generatedScript.trim().split(/\s+/).length;
      setWordCount(words);
    }
  }, [generatedScript]);

  // Handle script generation
  const handleGenerate = async () => {
    if (!topic.trim()) {
      setError('Please enter a topic');
      return;
    }

    setIsGenerating(true);
    setError(null);
    setGenerationProgress(0);

    try {
      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setGenerationProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 15;
        });
      }, 300);

      const result = await actions.generateScript({
        topic: topic.trim(),
        duration: parseInt(duration),
        tone,
        emotion,
        audience,
        contentType,
        sessionId: state.currentSession?.session_id,
      });

      clearInterval(progressInterval);
      setGenerationProgress(100);
      
      // Handle natural script response - simplified approach
      console.log('Script generation result:', result);
      
      const scriptText = result.script || result.content || '';
      if (!scriptText || typeof scriptText !== 'string') {
        console.error('No script content received:', result);
        throw new Error('No script content was generated. Please try again.');
      }
      
      setGeneratedScript(scriptText);
      
      // Reset progress after a brief delay
      setTimeout(() => {
        setGenerationProgress(0);
      }, 1000);

    } catch (error) {
      setError(error.message || 'Failed to generate script');
      setGenerationProgress(0);
    } finally {
      setIsGenerating(false);
    }
  };

  // Handle copy to clipboard
  const handleCopy = async () => {
    try {
      const textToCopy = generatedScript || 'No script content available';
      await navigator.clipboard.writeText(textToCopy);
      // Could add a toast notification here
    } catch (error) {
      setError('Failed to copy script to clipboard');
    }
  };

  // Handle download
  const handleDownload = () => {
    const content = generatedScript || 'No script content available';
    const blob = new Blob([content], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `script_${topic.replace(/\s+/g, '_').toLowerCase()}_${Date.now()}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  };

  // Handle clear script
  const handleClear = () => {
    setGeneratedScript('');
    setError(null);
  };

  // Calculate estimated reading time
  const calculateReadingTime = () => {
    const wordsPerMinute = 160; // Average speaking rate
    return Math.ceil(wordCount / wordsPerMinute);
  };

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-gray-900 mb-2">Script Generator</h1>
        <p className="text-gray-600 text-lg">Create natural, conversational scripts that sound like real people talking - perfect for your videos.</p>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        {/* Input Panel */}
        <Card>
          <CardHeader>
            <CardTitle>Generate Script</CardTitle>
            <CardDescription>
              Tell us your topic and style preferences - we'll create a natural, flowing script that sounds conversational and engaging.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-2">
              <Label htmlFor="topic">Topic/Subject</Label>
              <Input
                id="topic"
                placeholder="e.g., How to Start a YouTube Channel"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
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

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="tone">Tone & Style</Label>
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

            {/* Progress Bar */}
            {isGenerating && (
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Generating script...</span>
                  <span>{generationProgress}%</span>
                </div>
                <Progress value={generationProgress} />
              </div>
            )}

            {/* Error Display */}
            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <Button 
              onClick={handleGenerate}
              disabled={!topic.trim() || isGenerating || !state.currentSession}
              className="w-full bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700"
              size="lg"
            >
              {isGenerating ? (
                <>
                  <Sparkles className="w-4 h-4 mr-2 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <FileText className="w-4 h-4 mr-2" />
                  Generate Script
                </>
              )}
            </Button>

            {!state.currentSession && (
              <Alert>
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>
                  Please create a session first to generate scripts.
                </AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>

        {/* Results Panel */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>Generated Script</span>
              {generatedScript && (
                <div className="flex gap-2">
                  <Button variant="outline" size="sm" onClick={handleCopy}>
                    <Copy className="w-4 h-4 mr-1" />
                    Copy
                  </Button>
                  <Button variant="outline" size="sm" onClick={handleDownload}>
                    <Download className="w-4 h-4 mr-1" />
                    Download
                  </Button>
                  <Button variant="outline" size="sm" onClick={handleClear}>
                    Clear
                  </Button>
                </div>
              )}
            </CardTitle>
            <CardDescription>
              {generatedScript
                ? `Your script is ready! ${wordCount} words, ~${calculateReadingTime()} min reading time.`
                : 'Your generated script will appear here.'
              }
            </CardDescription>
          </CardHeader>
          <CardContent>
            {generatedScript ? (
              <div className="space-y-4">
                {/* Simple Script Editor */}
                <Textarea
                  value={generatedScript}
                  onChange={(e) => setGeneratedScript(e.target.value)}
                  className="min-h-[400px] text-base leading-relaxed p-4"
                  placeholder="Your generated script will appear here. You can edit it directly..."
                />
                
                {/* Script Stats */}
                <div className="flex justify-between text-sm text-gray-500 border-t pt-2">
                  <span>Words: {wordCount}</span>
                  <span>Est. reading time: {calculateReadingTime()} min</span>
                  <span>Target duration: {duration} min</span>
                </div>

                {/* Action Buttons */}
                <div className="flex gap-2">
                  <Button 
                    onClick={handleCopy}
                    variant="outline"
                    size="sm"
                  >
                    <Copy className="w-4 h-4 mr-1" />
                    Copy Script
                  </Button>
                  <Button 
                    onClick={handleDownload}
                    variant="outline"
                    size="sm"
                  >
                    <Download className="w-4 h-4 mr-1" />
                    Download
                  </Button>
                  <Button 
                    onClick={handleClear}
                    variant="outline"
                    size="sm"
                  >
                    Clear
                  </Button>
                </div>
              </div>
            ) : (
              <div className="text-center py-12 text-gray-500">
                <div className="w-16 h-16 mx-auto mb-4 opacity-50 bg-blue-100 rounded-full flex items-center justify-center">
                  <FileText className="w-8 h-8 text-blue-500" />
                </div>
                <p>No script generated yet. Fill out the form and click generate!</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default ScriptGenerator;
