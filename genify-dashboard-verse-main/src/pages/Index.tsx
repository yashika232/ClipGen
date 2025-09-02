
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { ArrowDown, ArrowUp, LayoutDashboard, LogIn, UserPlus, Sparkles, Zap, Palette } from 'lucide-react';
import AuthModal from '@/components/AuthModal';
import Dashboard from '@/components/Dashboard';

const Index = () => {
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);
  const [authMode, setAuthMode] = useState<'signin' | 'signup'>('signin');
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [user, setUser] = useState<{ name: string; email: string } | null>(null);

  const handleAuthSuccess = (userData: { name: string; email: string }) => {
    setUser(userData);
    setIsLoggedIn(true);
    setIsAuthModalOpen(false);
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
    setUser(null);
  };

  const openAuthModal = (mode: 'signin' | 'signup') => {
    setAuthMode(mode);
    setIsAuthModalOpen(true);
  };

  if (isLoggedIn && user) {
    return <Dashboard user={user} onLogout={handleLogout} />;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 overflow-hidden">
      {/* Animated Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 left-10 w-32 h-32 bg-purple-400/10 rounded-full animate-pulse"></div>
        <div className="absolute top-40 right-20 w-24 h-24 bg-blue-400/10 rounded-full animate-bounce"></div>
        <div className="absolute bottom-32 left-32 w-20 h-20 bg-indigo-400/10 rounded-full animate-ping"></div>
        <div className="absolute bottom-20 right-40 w-16 h-16 bg-purple-400/10 rounded-full animate-pulse"></div>
      </div>

      {/* Header */}
      <header className="container mx-auto px-6 py-8 flex justify-between items-center relative z-10 animate-fade-in">
        <div className="text-2xl font-bold text-white transform hover:scale-110 transition-all duration-300 cursor-pointer">
          Clip<span className="text-purple-400 animate-pulse">Gen</span>
          <Sparkles className="inline w-6 h-6 ml-2 text-yellow-400 animate-spin" />
        </div>
        <div className="flex gap-4">
          <Button 
            variant="outline" 
            onClick={() => openAuthModal('signin')}
            className="bg-transparent border-white/20 text-white hover:bg-white/10 hover:scale-105 transform transition-all duration-300 hover:shadow-lg hover:shadow-purple-500/25"
          >
            <LogIn className="w-4 h-4 mr-2 animate-bounce" />
            Sign In
          </Button>
          <Button 
            onClick={() => openAuthModal('signup')}
            className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white border-0 hover:scale-105 transform transition-all duration-300 hover:shadow-lg hover:shadow-blue-500/25"
          >
            <UserPlus className="w-4 h-4 mr-2 animate-pulse" />
            Get Started
          </Button>
        </div>
      </header>

      {/* Hero Section */}
      <section className="container mx-auto px-6 py-20 text-center relative z-10">
        <div className="max-w-4xl mx-auto animate-fade-in">
          <h1 className="text-6xl font-bold text-white mb-8 leading-tight transform hover:scale-105 transition-all duration-500">
            Create Amazing Content with
            <span className="bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent animate-pulse"> AI Magic</span>
            <Zap className="inline w-12 h-12 ml-4 text-yellow-400 animate-bounce" />
          </h1>
          <p className="text-xl text-blue-100 mb-12 max-w-2xl mx-auto animate-fade-in animation-delay-300">
            Generate stunning thumbnails, compelling scripts, and engaging videos with our powerful AI tools. Transform your content creation workflow today.
          </p>
          <Button 
            size="lg"
            onClick={() => openAuthModal('signup')}
            className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white border-0 px-8 py-6 text-lg font-semibold rounded-xl shadow-2xl hover:shadow-purple-500/25 transition-all duration-300 hover:scale-110 animate-bounce"
          >
            Start Creating for Free
            <ArrowDown className="w-5 h-5 ml-2 animate-bounce" />
          </Button>
        </div>
      </section>

      {/* Features Section */}
      <section className="container mx-auto px-6 py-20 relative z-10">
        <div className="text-center mb-16 animate-fade-in">
          <h2 className="text-4xl font-bold text-white mb-6 hover:scale-105 transform transition-all duration-300">
            Everything You Need to Create
          </h2>
          <p className="text-blue-100 text-lg max-w-2xl mx-auto">
            Our AI-powered tools help you create professional content in minutes, not hours.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          <Card className="bg-white/10 border-white/20 backdrop-blur-sm hover:bg-white/15 transition-all duration-500 hover:scale-110 hover:shadow-2xl hover:shadow-purple-500/25 animate-fade-in animation-delay-100 group">
            <CardHeader>
              <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg flex items-center justify-center mb-4 group-hover:animate-spin transition-all duration-300">
                <Palette className="w-6 h-6 text-white" />
              </div>
              <CardTitle className="text-white text-xl group-hover:text-purple-300 transition-colors duration-300">Thumbnail Generation</CardTitle>
              <CardDescription className="text-blue-100 group-hover:text-blue-200 transition-colors duration-300">
                Create eye-catching thumbnails that boost your click-through rates with AI-powered design suggestions.
              </CardDescription>
            </CardHeader>
          </Card>

          <Card className="bg-white/10 border-white/20 backdrop-blur-sm hover:bg-white/15 transition-all duration-500 hover:scale-110 hover:shadow-2xl hover:shadow-blue-500/25 animate-fade-in animation-delay-200 group">
            <CardHeader>
              <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-lg flex items-center justify-center mb-4 group-hover:animate-pulse transition-all duration-300">
                <LayoutDashboard className="w-6 h-6 text-white" />
              </div>
              <CardTitle className="text-white text-xl group-hover:text-blue-300 transition-colors duration-300">Script Generation</CardTitle>
              <CardDescription className="text-blue-100 group-hover:text-blue-200 transition-colors duration-300">
                Generate compelling scripts for videos, podcasts, and presentations with intelligent AI writing.
              </CardDescription>
            </CardHeader>
          </Card>

          <Card className="bg-white/10 border-white/20 backdrop-blur-sm hover:bg-white/15 transition-all duration-500 hover:scale-110 hover:shadow-2xl hover:shadow-indigo-500/25 animate-fade-in animation-delay-300 group">
            <CardHeader>
              <div className="w-12 h-12 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-lg flex items-center justify-center mb-4 group-hover:animate-bounce transition-all duration-300">
                <Zap className="w-6 h-6 text-white" />
              </div>
              <CardTitle className="text-white text-xl group-hover:text-indigo-300 transition-colors duration-300">Video Generation</CardTitle>
              <CardDescription className="text-blue-100 group-hover:text-blue-200 transition-colors duration-300">
                Transform your ideas into stunning videos with AI-powered video creation and editing tools.
              </CardDescription>
            </CardHeader>
          </Card>
        </div>
      </section>

      {/* CTA Section */}
      <section className="container mx-auto px-6 py-20 text-center relative z-10 animate-fade-in">
        <div className="max-w-3xl mx-auto bg-white/10 backdrop-blur-sm rounded-2xl p-12 border border-white/20 hover:bg-white/15 transition-all duration-500 hover:scale-105 hover:shadow-2xl hover:shadow-purple-500/25">
          <h2 className="text-3xl font-bold text-white mb-6 hover:scale-105 transform transition-all duration-300">
            Ready to Transform Your Content?
            <Sparkles className="inline w-8 h-8 ml-2 text-yellow-400 animate-spin" />
          </h2>
          <p className="text-blue-100 mb-8 text-lg">
            Join thousands of creators who are already using ClipGen to create amazing content.
          </p>
          <Button 
            size="lg"
            onClick={() => openAuthModal('signup')}
            className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white border-0 px-8 py-6 text-lg font-semibold rounded-xl shadow-2xl hover:shadow-purple-500/25 transition-all duration-300 hover:scale-110 animate-pulse"
          >
            Get Started Now
            <ArrowUp className="w-5 h-5 ml-2 animate-bounce" />
          </Button>
        </div>
      </section>

      <AuthModal 
        isOpen={isAuthModalOpen}
        onClose={() => setIsAuthModalOpen(false)}
        mode={authMode}
        onAuthSuccess={handleAuthSuccess}
      />
    </div>
  );
};

export default Index;
