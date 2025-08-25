import React, { Suspense } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { LanguageDetector } from './components/common/LanguageSwitcher';
import './i18n'; // Initialize i18n

// Import components (these would be created later)
const Dashboard = React.lazy(() => import('./pages/Dashboard'));
const Predictions = React.lazy(() => import('./pages/Predictions'));
const Portfolio = React.lazy(() => import('./pages/Portfolio'));
const Watchlist = React.lazy(() => import('./pages/Watchlist'));
const News = React.lazy(() => import('./pages/News'));
const Analysis = React.lazy(() => import('./pages/Analysis'));
const Settings = React.lazy(() => import('./pages/Settings'));

// Layout components
import Header from './components/layout/Header';
import Sidebar from './components/layout/Sidebar';
import Footer from './components/layout/Footer';

// Loading component
const LoadingSpinner = () => {
  const { t } = useTranslation();
  
  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-50 dark:bg-gray-900">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
        <p className="text-gray-600 dark:text-gray-400 text-lg">
          {t('app.loading')}
        </p>
      </div>
    </div>
  );
};

// Error Boundary
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('App Error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex items-center justify-center min-h-screen bg-gray-50 dark:bg-gray-900">
          <div className="text-center max-w-md mx-auto p-6">
            <div className="text-red-500 text-6xl mb-4">⚠️</div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
              Something went wrong
            </h1>
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              We're sorry, but something unexpected happened. Please refresh the page or try again later.
            </p>
            <button
              onClick={() => window.location.reload()}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              Refresh Page
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

// Main Layout Component
const Layout = ({ children }) => {
  const [sidebarOpen, setSidebarOpen] = React.useState(false);
  const { t, i18n } = useTranslation();

  // Update document direction based on language
  React.useEffect(() => {
    const direction = i18n.dir();
    document.documentElement.dir = direction;
    document.documentElement.lang = i18n.language;
  }, [i18n.language]);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Sidebar */}
      <Sidebar 
        isOpen={sidebarOpen} 
        onClose={() => setSidebarOpen(false)} 
      />
      
      {/* Main Content */}
      <div className={`transition-all duration-300 ${
        sidebarOpen ? 'lg:ml-64' : 'lg:ml-16'
      }`}>
        {/* Header */}
        <Header 
          onMenuClick={() => setSidebarOpen(!sidebarOpen)}
          sidebarOpen={sidebarOpen}
        />
        
        {/* Page Content */}
        <main className="px-4 py-6 sm:px-6 lg:px-8">
          <div className="mx-auto max-w-7xl">
            {children}
          </div>
        </main>
        
        {/* Footer */}
        <Footer />
      </div>
      
      {/* Mobile sidebar backdrop */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 z-40 bg-black bg-opacity-50 lg:hidden" 
          onClick={() => setSidebarOpen(false)}
        />
      )}
    </div>
  );
};

// Placeholder components for pages that don't exist yet
const PlaceholderPage = ({ title, description }) => {
  const { t } = useTranslation();
  
  return (
    <div className="text-center py-12">
      <div className="text-6xl mb-4">🚧</div>
      <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
        {title}
      </h1>
      <p className="text-gray-600 dark:text-gray-400 mb-6">
        {description}
      </p>
      <div className="text-sm text-gray-500 dark:text-gray-400">
        {t('app.loading')}
      </div>
    </div>
  );
};

// Create placeholder components
const DashboardPlaceholder = () => {
  const { t } = useTranslation();
  return (
    <PlaceholderPage 
      title={t('navigation.dashboard')} 
      description="Real-time market overview and portfolio summary"
    />
  );
};

const PredictionsPlaceholder = () => {
  const { t } = useTranslation();
  return (
    <PlaceholderPage 
      title={t('navigation.predictions')} 
      description="AI-powered stock predictions and analysis"
    />
  );
};

const PortfolioPlaceholder = () => {
  const { t } = useTranslation();
  return (
    <PlaceholderPage 
      title={t('navigation.portfolio')} 
      description="Your investment portfolio and performance tracking"
    />
  );
};

const WatchlistPlaceholder = () => {
  const { t } = useTranslation();
  return (
    <PlaceholderPage 
      title={t('navigation.watchlist')} 
      description="Track your favorite stocks and market movements"
    />
  );
};

const NewsPlaceholder = () => {
  const { t } = useTranslation();
  return (
    <PlaceholderPage 
      title={t('navigation.news')} 
      description="Latest market news with sentiment analysis"
    />
  );
};

const AnalysisPlaceholder = () => {
  const { t } = useTranslation();
  return (
    <PlaceholderPage 
      title={t('navigation.analysis')} 
      description="Technical and fundamental analysis tools"
    />
  );
};

const SettingsPlaceholder = () => {
  const { t } = useTranslation();
  return (
    <PlaceholderPage 
      title={t('navigation.settings')} 
      description="Application settings and preferences"
    />
  );
};

// Main App Component
function App() {
  const { t } = useTranslation();

  return (
    <ErrorBoundary>
      <LanguageDetector>
        <Router>
          <Layout>
            <Suspense fallback={<LoadingSpinner />}>
              <Routes>
                {/* Default route */}
                <Route path="/" element={<Navigate to="/dashboard" replace />} />
                
                {/* Main routes */}
                <Route path="/dashboard" element={<DashboardPlaceholder />} />
                <Route path="/predictions" element={<PredictionsPlaceholder />} />
                <Route path="/portfolio" element={<PortfolioPlaceholder />} />
                <Route path="/watchlist" element={<WatchlistPlaceholder />} />
                <Route path="/news" element={<NewsPlaceholder />} />
                <Route path="/analysis" element={<AnalysisPlaceholder />} />
                <Route path="/settings" element={<SettingsPlaceholder />} />
                
                {/* Catch all route */}
                <Route path="*" element={
                  <div className="text-center py-12">
                    <div className="text-6xl mb-4">404</div>
                    <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
                      {t('errors.notFound')}
                    </h1>
                    <p className="text-gray-600 dark:text-gray-400 mb-6">
                      The page you're looking for doesn't exist.
                    </p>
                    <button
                      onClick={() => window.history.back()}
                      className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                    >
                      {t('navigation.back')}
                    </button>
                  </div>
                } />
              </Routes>
            </Suspense>
          </Layout>
        </Router>
      </LanguageDetector>
    </ErrorBoundary>
  );
}

export default App;