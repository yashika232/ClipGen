
import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { LayoutDashboard, Download, Sparkles, AlertCircle, CheckCircle } from 'lucide-react';
import { useApp } from '@/contexts/AppContext';

const ThumbnailGenerator = () => {
  const { state, actions } = useApp();
  const [prompt, setPrompt] = useState('');
  const [style, setStyle] = useState('modern');
  const [quality, setQuality] = useState('high');
  const [count, setCount] = useState(4);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedThumbnails, setGeneratedThumbnails] = useState([]);
  const [generationProgress, setGenerationProgress] = useState(0);
  const [error, setError] = useState(null);

  // Get available options from context
  const thumbnailStyles = state.availableOptions.thumbnailStyles || ['modern', 'minimalist', 'bold', 'professional', 'gaming', 'educational'];
  const qualityLevels = state.availableOptions.qualityLevels || ['draft', 'standard', 'high', 'premium'];

  // Handle thumbnail generation
  const handleGenerate = async () => {
    if (!prompt.trim()) {
      setError('Please enter a thumbnail description');
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
          return prev + 10;
        });
      }, 500);

      const result = await actions.generateThumbnail({
        prompt: prompt.trim(),
        style,
        quality,
        count,
        sessionId: state.currentSession?.session_id,
      });

      clearInterval(progressInterval);
      setGenerationProgress(100);
      
      // Add generated thumbnails to state
      setGeneratedThumbnails(prev => [...prev, ...result.thumbnails]);
      
      // Reset progress after a brief delay
      setTimeout(() => {
        setGenerationProgress(0);
      }, 1000);

    } catch (error) {
      setError(error.message || 'Failed to generate thumbnails');
      setGenerationProgress(0);
    } finally {
      setIsGenerating(false);
    }
  };

  // Handle thumbnail download
  const handleDownload = async (thumbnail) => {
    try {
      if (thumbnail.url) {
        // Download from URL
        const response = await fetch(thumbnail.url);
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `thumbnail_${Date.now()}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
      } else if (thumbnail.sessionId && thumbnail.filename) {
        // Download from session
        await actions.downloadFile(thumbnail.sessionId, thumbnail.filename);
      }
    } catch (error) {
      setError('Failed to download thumbnail');
    }
  };

  // Clear generated thumbnails
  const handleClear = () => {
    setGeneratedThumbnails([]);
    setError(null);
  };

  // Listen for WebSocket generation updates
  useEffect(() => {
    if (!state.currentSession) return;

    const handleGenerationProgress = (data) => {
      if (data.type === 'thumbnail' && data.session_id === state.currentSession.session_id) {
        setGenerationProgress(data.progress || 0);
      }
    };

    const handleGenerationCompleted = (data) => {
      if (data.type === 'thumbnail' && data.session_id === state.currentSession.session_id) {
        setGenerationProgress(100);
        if (data.result?.thumbnails) {
          setGeneratedThumbnails(prev => [...prev, ...data.result.thumbnails]);
        }
      }
    };

    const handleGenerationError = (data) => {
      if (data.type === 'thumbnail' && data.session_id === state.currentSession.session_id) {
        setError(data.error || 'Generation failed');
        setGenerationProgress(0);
      }
    };

    // Note: These would be handled by the WebSocket service
    // For now, we'll rely on the API calls
    
  }, [state.currentSession]);

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-gray-900 mb-2">Thumbnail Generator</h1>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        {/* Input Panel */}
        <Card>
          <CardHeader>
            <CardTitle>Generate Thumbnail</CardTitle>
            <CardDescription>
              Describe what you want your thumbnail to look like, and our AI will create multiple high-quality options for you.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-2">
              <Label htmlFor="prompt">Thumbnail Description</Label>
              <Textarea
                id="prompt"
                placeholder="e.g., A tech tutorial thumbnail with bright colors, bold text saying 'How to Code', and a laptop in the background"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                className="min-h-[100px]"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="style">Style</Label>
                <Select value={style} onValueChange={setStyle}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select style" />
                  </SelectTrigger>
                  <SelectContent>
                    {thumbnailStyles.map((styleOption) => (
                      <SelectItem key={styleOption} value={styleOption}>
                        {styleOption.charAt(0).toUpperCase() + styleOption.slice(1)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="quality">Quality</Label>
                <Select value={quality} onValueChange={setQuality}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select quality" />
                  </SelectTrigger>
                  <SelectContent>
                    {qualityLevels.map((qualityOption) => (
                      <SelectItem key={qualityOption} value={qualityOption}>
                        {qualityOption.charAt(0).toUpperCase() + qualityOption.slice(1)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="count">Number of Thumbnails</Label>
              <Select value={count.toString()} onValueChange={(value) => setCount(parseInt(value))}>
                <SelectTrigger>
                  <SelectValue placeholder="Select count" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="2">2 thumbnails</SelectItem>
                  <SelectItem value="4">4 thumbnails</SelectItem>
                  <SelectItem value="6">6 thumbnails</SelectItem>
                  <SelectItem value="8">8 thumbnails</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Progress Bar */}
            {isGenerating && (
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Generating thumbnails...</span>
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
              disabled={!prompt.trim() || isGenerating || !state.currentSession}
              className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700"
              size="lg"
            >
              {isGenerating ? (
                <>
                  <Sparkles className="w-4 h-4 mr-2 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4 mr-2" />
                  Generate Thumbnails
                </>
              )}
            </Button>

            {!state.currentSession && (
              <Alert>
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>
                  Please create a session first to generate thumbnails.
                </AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>

        {/* Results Panel */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>Generated Thumbnails</span>
              {generatedThumbnails.length > 0 && (
                <Button variant="outline" size="sm" onClick={handleClear}>
                  Clear All
                </Button>
              )}
            </CardTitle>
            <CardDescription>
              {generatedThumbnails.length > 0 
                ? 'Click on any thumbnail to download it.'
                : 'Your generated thumbnails will appear here.'
              }
            </CardDescription>
          </CardHeader>
          <CardContent>
            {generatedThumbnails.length > 0 ? (
              <div className="grid grid-cols-2 gap-4">
                {generatedThumbnails.map((thumbnail, index) => (
                  <div 
                    key={index}
                    className="relative group cursor-pointer rounded-lg overflow-hidden hover:scale-105 transition-transform duration-200"
                  >
                    <img 
                      src={thumbnail.url || thumbnail} 
                      alt={`Generated thumbnail ${index + 1}`}
                      className="w-full h-auto rounded-lg"
                    />
                    <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex items-center justify-center">
                      <Button 
                        size="sm" 
                        className="bg-white text-black hover:bg-gray-100"
                        onClick={() => handleDownload(thumbnail)}
                      >
                        <Download className="w-4 h-4 mr-1" />
                        Download
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-12 text-gray-500">
                <LayoutDashboard className="w-16 h-16 mx-auto mb-4 opacity-50" />
                <p>No thumbnails generated yet. Fill out the form and click generate!</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default ThumbnailGenerator;
