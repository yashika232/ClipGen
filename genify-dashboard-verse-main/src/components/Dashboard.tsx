
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { LayoutDashboard, LogIn, Sparkles, Zap, Palette, Video, FileText } from 'lucide-react';
import ThumbnailGenerator from './ThumbnailGenerator';
import ScriptGenerator from './ScriptGenerator';
import VideoGenerator from './VideoGenerator';
import ErrorBoundary from './ErrorBoundary';

interface DashboardProps {
  user: { name: string; email: string };
  onLogout: () => void;
}

type ActiveTab = 'overview' | 'thumbnail' | 'script' | 'video';

const Dashboard = ({ user, onLogout }: DashboardProps) => {
  const [activeTab, setActiveTab] = useState<ActiveTab>('overview');

  const renderContent = () => {
    switch (activeTab) {
      case 'thumbnail':
        return <ThumbnailGenerator />;
      case 'script':
        return <ScriptGenerator />;
      case 'video':
        return (
          <ErrorBoundary>
            <VideoGenerator />
          </ErrorBoundary>
        );
      default:
        return (
          <div className="grid md:grid-cols-3 gap-6 animate-fade-in">
            <Card 
              className="cursor-pointer hover:shadow-lg transition-all duration-500 hover:scale-110 bg-gradient-to-br from-purple-50 to-purple-100 border-purple-200 hover:shadow-purple-500/25 group animate-fade-in animation-delay-100"
              onClick={() => setActiveTab('thumbnail')}
            >
              <CardHeader>
                <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg flex items-center justify-center mb-4 group-hover:animate-spin transition-all duration-300">
                  <Palette className="w-6 h-6 text-white" />
                </div>
                <CardTitle className="text-purple-900 group-hover:text-purple-700 transition-colors duration-300">Thumbnail Generator</CardTitle>
                <CardDescription className="group-hover:text-purple-600 transition-colors duration-300">
                  Create stunning thumbnails for your content with AI-powered design tools.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Button className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 hover:scale-105 transform transition-all duration-300">
                  Start Creating
                  <Sparkles className="w-4 h-4 ml-2 animate-pulse" />
                </Button>
              </CardContent>
            </Card>

            <Card 
              className="cursor-pointer hover:shadow-lg transition-all duration-500 hover:scale-110 bg-gradient-to-br from-blue-50 to-blue-100 border-blue-200 hover:shadow-blue-500/25 group animate-fade-in animation-delay-200"
              onClick={() => setActiveTab('script')}
            >
              <CardHeader>
                <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-lg flex items-center justify-center mb-4 group-hover:animate-pulse transition-all duration-300">
                  <FileText className="w-6 h-6 text-white" />
                </div>
                <CardTitle className="text-blue-900 group-hover:text-blue-700 transition-colors duration-300">Script Generator</CardTitle>
                <CardDescription className="group-hover:text-blue-600 transition-colors duration-300">
                  Generate compelling scripts for videos, podcasts, and presentations.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Button className="w-full bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 hover:scale-105 transform transition-all duration-300">
                  Start Writing
                  <Zap className="w-4 h-4 ml-2 animate-bounce" />
                </Button>
              </CardContent>
            </Card>

            <Card 
              className="cursor-pointer hover:shadow-lg transition-all duration-500 hover:scale-110 bg-gradient-to-br from-indigo-50 to-indigo-100 border-indigo-200 hover:shadow-indigo-500/25 group animate-fade-in animation-delay-300"
              onClick={() => setActiveTab('video')}
            >
              <CardHeader>
                <div className="w-12 h-12 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-lg flex items-center justify-center mb-4 group-hover:animate-bounce transition-all duration-300">
                  <Video className="w-6 h-6 text-white" />
                </div>
                <CardTitle className="text-indigo-900 group-hover:text-indigo-700 transition-colors duration-300">Video Generator</CardTitle>
                <CardDescription className="group-hover:text-indigo-600 transition-colors duration-300">
                  Create amazing videos with AI-powered editing and generation tools.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Button className="w-full bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 hover:scale-105 transform transition-all duration-300">
                  Create Video
                  <Video className="w-4 h-4 ml-2 animate-pulse" />
                </Button>
              </CardContent>
            </Card>
          </div>
        );
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 relative overflow-hidden">
      {/* Animated Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 left-10 w-32 h-32 bg-purple-200/20 rounded-full animate-pulse"></div>
        <div className="absolute top-40 right-20 w-24 h-24 bg-blue-200/20 rounded-full animate-bounce"></div>
        <div className="absolute bottom-32 left-32 w-20 h-20 bg-indigo-200/20 rounded-full animate-ping"></div>
      </div>

      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-50 animate-fade-in">
        <div className="container mx-auto px-6 py-4 flex justify-between items-center">
          <div className="flex items-center gap-8">
            <div 
              className="text-2xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent cursor-pointer hover:scale-110 transform transition-all duration-300"
              onClick={() => setActiveTab('overview')}
            >
              ClipGen
              <Sparkles className="inline w-6 h-6 ml-2 text-purple-500 animate-spin" />
            </div>
            <nav className="hidden md:flex gap-6">
              <button
                onClick={() => setActiveTab('overview')}
                className={`px-4 py-2 rounded-lg transition-all duration-300 hover:scale-105 ${
                  activeTab === 'overview' 
                    ? 'bg-purple-100 text-purple-700 animate-pulse' 
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                }`}
              >
                Dashboard
              </button>
              <button
                onClick={() => setActiveTab('thumbnail')}
                className={`px-4 py-2 rounded-lg transition-all duration-300 hover:scale-105 ${
                  activeTab === 'thumbnail' 
                    ? 'bg-purple-100 text-purple-700 animate-pulse' 
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                }`}
              >
                Thumbnails
              </button>
              <button
                onClick={() => setActiveTab('script')}
                className={`px-4 py-2 rounded-lg transition-all duration-300 hover:scale-105 ${
                  activeTab === 'script' 
                    ? 'bg-purple-100 text-purple-700 animate-pulse' 
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                }`}
              >
                Scripts
              </button>
              <button
                onClick={() => setActiveTab('video')}
                className={`px-4 py-2 rounded-lg transition-all duration-300 hover:scale-105 ${
                  activeTab === 'video' 
                    ? 'bg-purple-100 text-purple-700 animate-pulse' 
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                }`}
              >
                Videos
              </button>
            </nav>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-gray-600 animate-fade-in">Welcome, {user.name}! 👋</span>
            <Button 
              variant="outline" 
              onClick={onLogout}
              className="text-gray-600 hover:text-gray-900 hover:scale-105 transform transition-all duration-300"
            >
              <LogIn className="w-4 h-4 mr-2 animate-bounce" />
              Logout
            </Button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-6 py-8 relative z-10">
        {activeTab === 'overview' && (
          <div className="mb-8 animate-fade-in">
            <h1 className="text-4xl font-bold text-gray-900 mb-2 hover:scale-105 transform transition-all duration-300">
              Welcome back, {user.name}! 🎨
            </h1>
            <p className="text-gray-600 text-lg">
              Choose a tool below to start creating amazing content with AI.
            </p>
          </div>
        )}
        
        {renderContent()}
      </main>
    </div>
  );
};

export default Dashboard;
