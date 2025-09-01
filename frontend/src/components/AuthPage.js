import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import LoginForm from './LoginForm';
import RegisterForm from './RegisterForm';

const AuthPage = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [isTransitioning, setIsTransitioning] = useState(false);
  const { user, isAuthenticated } = useAuth();

  // Redirect if already authenticated
  useEffect(() => {
    if (isAuthenticated && user) {
      window.location.href = '/dashboard';
    }
  }, [isAuthenticated, user]);

  // Handle form switching with smooth transition
  const handleSwitchForm = (toLogin = true) => {
    setIsTransitioning(true);
    
    setTimeout(() => {
      setIsLogin(toLogin);
      setIsTransitioning(false);
    }, 150);
  };

  // Handle successful authentication
  const handleAuthSuccess = (userData) => {
    console.log('Authentication successful:', userData);
    // The AuthContext will handle the redirect
  };

  // Don't render if already authenticated
  if (isAuthenticated && user) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="loading-spinner mx-auto mb-4"></div>
          <p className="text-gray-600">Redirecting to dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50">
      {/* Background Pattern */}
      <div className="absolute inset-0 bg-grid-pattern opacity-5"></div>
      
      {/* Header */}
      <div className="relative z-10 pt-4 sm:pt-8 pb-2 sm:pb-4">
        <div className="max-w-7xl mx-auto px-3 sm:px-4 lg:px-8">
          <div className="text-center">
            <div className="flex items-center justify-center space-x-2 mb-3 sm:mb-4">
              <div className="h-8 w-8 sm:h-10 sm:w-10 bg-gradient-to-r from-blue-600 to-green-600 rounded-lg flex items-center justify-center">
                <svg className="h-4 w-4 sm:h-6 sm:w-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                </svg>
              </div>
              <h1 className="text-lg sm:text-2xl font-bold bg-gradient-to-r from-blue-600 to-green-600 bg-clip-text text-transparent">
                Stock Prediction AI
              </h1>
            </div>
            <p className="text-sm sm:text-base text-gray-600 max-w-2xl mx-auto px-2">
              Advanced AI-powered stock market predictions with real-time analysis, 
              technical indicators, and sentiment-driven insights for the Indian stock market.
            </p>
          </div>
        </div>
      </div>

      {/* Form Toggle Tabs */}
      <div className="relative z-10 max-w-md mx-auto px-3 sm:px-4">
        <div className="bg-white rounded-lg shadow-sm p-1 mb-4 sm:mb-8">
          <div className="grid grid-cols-2 gap-1">
            <button
              onClick={() => handleSwitchForm(true)}
              disabled={isTransitioning}
              className={`py-2 sm:py-2.5 px-3 sm:px-4 text-xs sm:text-sm font-medium rounded-md transition-all duration-200 ${
                isLogin
                  ? 'bg-blue-600 text-white shadow-sm'
                  : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
              } disabled:opacity-50`}
            >
              Sign In
            </button>
            <button
              onClick={() => handleSwitchForm(false)}
              disabled={isTransitioning}
              className={`py-2 sm:py-2.5 px-3 sm:px-4 text-xs sm:text-sm font-medium rounded-md transition-all duration-200 ${
                !isLogin
                  ? 'bg-green-600 text-white shadow-sm'
                  : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
              } disabled:opacity-50`}
            >
              Sign Up
            </button>
          </div>
        </div>
      </div>

      {/* Form Container */}
      <div className={`relative z-10 transition-opacity duration-150 ${
        isTransitioning ? 'opacity-0' : 'opacity-100'
      }`}>
        {isLogin ? (
          <LoginForm
            onSuccess={handleAuthSuccess}
            onSwitchToRegister={() => handleSwitchForm(false)}
          />
        ) : (
          <RegisterForm
            onSuccess={handleAuthSuccess}
            onSwitchToLogin={() => handleSwitchForm(true)}
          />
        )}
      </div>

      {/* Features Section */}
      <div className="relative z-10 mt-8 sm:mt-16 pb-8 sm:pb-16">
        <div className="max-w-7xl mx-auto px-3 sm:px-4 lg:px-8">
          <div className="text-center mb-6 sm:mb-12">
            <h2 className="text-xl sm:text-3xl font-bold text-gray-900 mb-2 sm:mb-4">
              Why Choose Our Platform?
            </h2>
            <p className="text-sm sm:text-base text-gray-600 max-w-2xl mx-auto px-2">
              Experience the power of AI-driven stock market analysis with cutting-edge technology
            </p>
          </div>
          
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6 lg:gap-8">
            {/* Feature 1 */}
            <div className="text-center p-3 sm:p-0">
              <div className="mx-auto h-10 w-10 sm:h-12 sm:w-12 bg-blue-100 rounded-lg flex items-center justify-center mb-3 sm:mb-4">
                <svg className="h-5 w-5 sm:h-6 sm:w-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <h3 className="text-base sm:text-lg font-semibold text-gray-900 mb-2">AI-Powered Predictions</h3>
              <p className="text-gray-600 text-xs sm:text-sm leading-relaxed">
                Advanced machine learning models including XGBoost, Transformers, and Deep Neural Networks
              </p>
            </div>

            {/* Feature 2 */}
            <div className="text-center p-3 sm:p-0">
              <div className="mx-auto h-10 w-10 sm:h-12 sm:w-12 bg-green-100 rounded-lg flex items-center justify-center mb-3 sm:mb-4">
                <svg className="h-5 w-5 sm:h-6 sm:w-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <h3 className="text-base sm:text-lg font-semibold text-gray-900 mb-2">Real-Time Analysis</h3>
              <p className="text-gray-600 text-xs sm:text-sm leading-relaxed">
                Live market data integration with instant predictions and technical indicator updates
              </p>
            </div>

            {/* Feature 3 */}
            <div className="text-center p-3 sm:p-0">
              <div className="mx-auto h-10 w-10 sm:h-12 sm:w-12 bg-purple-100 rounded-lg flex items-center justify-center mb-3 sm:mb-4">
                <svg className="h-5 w-5 sm:h-6 sm:w-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z" />
                </svg>
              </div>
              <h3 className="text-base sm:text-lg font-semibold text-gray-900 mb-2">News Sentiment</h3>
              <p className="text-gray-600 text-xs sm:text-sm leading-relaxed">
                Financial news analysis with sentiment scoring to enhance prediction accuracy
              </p>
            </div>

            {/* Feature 4 */}
            <div className="text-center p-3 sm:p-0">
              <div className="mx-auto h-10 w-10 sm:h-12 sm:w-12 bg-orange-100 rounded-lg flex items-center justify-center mb-3 sm:mb-4">
                <svg className="h-5 w-5 sm:h-6 sm:w-6 text-orange-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <h3 className="text-base sm:text-lg font-semibold text-gray-900 mb-2">Technical Indicators</h3>
              <p className="text-gray-600 text-xs sm:text-sm leading-relaxed">
                Comprehensive technical analysis with RSI, MACD, Bollinger Bands, and more
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="relative z-10 bg-gray-50 border-t border-gray-200">
        <div className="max-w-7xl mx-auto py-4 sm:py-8 px-3 sm:px-4 lg:px-8">
          <div className="text-center">
            <p className="text-gray-500 text-xs sm:text-sm leading-relaxed px-2">
              © 2024 Stock Prediction AI. Built with advanced machine learning for the Indian stock market.
            </p>
            <div className="mt-3 sm:mt-4 flex flex-wrap justify-center gap-3 sm:gap-6">
              <a href="/terms" className="text-gray-400 hover:text-gray-500 text-xs sm:text-sm px-1">
                Terms of Service
              </a>
              <a href="/privacy" className="text-gray-400 hover:text-gray-500 text-xs sm:text-sm px-1">
                Privacy Policy
              </a>
              <a href="/contact" className="text-gray-400 hover:text-gray-500 text-xs sm:text-sm px-1">
                Contact Us
              </a>
              <a href="/api-docs" className="text-gray-400 hover:text-gray-500 text-xs sm:text-sm px-1">
                API Documentation
              </a>
            </div>
          </div>
        </div>
      </footer>

      {/* Custom Styles */}
      <style jsx>{`
        .bg-grid-pattern {
          background-image: radial-gradient(circle, #e5e7eb 1px, transparent 1px);
          background-size: 20px 20px;
        }
        
        .loading-spinner {
          border: 2px solid #f3f4f6;
          border-top: 2px solid #3b82f6;
          border-radius: 50%;
          width: 20px;
          height: 20px;
          animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        
        @media (max-width: 640px) {
          .min-h-screen {
            min-height: 100vh;
            min-height: -webkit-fill-available;
          }
        }
      `}</style>
    </div>
  );
};

export default AuthPage;