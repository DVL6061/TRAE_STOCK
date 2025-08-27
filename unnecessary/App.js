import React, { useState, useEffect } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

// Layout components
import Header from '../frontend/src/components/layout/Header';
import Sidebar from '../frontend/src/components/layout/Sidebar';
import Footer from '../frontend/src/components/layout/Footer';

// Page components
import Dashboard from '../frontend/src/pages/Dashboard';
import StockDetail from '../frontend/src/pages/StockDetail';
import PredictionPage from '../frontend/src/pages/PredictionPage';
import NewsPage from '../frontend/src/pages/NewsPage';
import SettingsPage from '../frontend/src/pages/SettingsPage';
import NotFoundPage from '../frontend/src/pages/NotFoundPage';

// Context
import { ThemeProvider } from '../frontend/src/contexts/ThemeContext';

function App() {
  const { t } = useTranslation();
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  // Simulate initial loading
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 1000);

    return () => clearTimeout(timer);
  }, []);

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen bg-neutral-50 dark:bg-neutral-900">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-primary-500 border-t-transparent rounded-full animate-spin mx-auto"></div>
          <h2 className="mt-4 text-xl font-semibold text-neutral-800 dark:text-neutral-200">
            {t('loading')}
          </h2>
        </div>
      </div>
    );
  }

  return (
    <ThemeProvider>
      <div className="flex flex-col min-h-screen bg-neutral-50 dark:bg-neutral-900">
        <Header toggleSidebar={toggleSidebar} />
        
        <div className="flex flex-1 overflow-hidden">
          <Sidebar isOpen={isSidebarOpen} toggleSidebar={toggleSidebar} />
          
          <main className="flex-1 overflow-y-auto p-4 md:p-6">
            <Routes>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/stock/:ticker" element={<StockDetail />} />
              <Route path="/predictions" element={<PredictionPage />} />
              <Route path="/news" element={<NewsPage />} />
              <Route path="/settings" element={<SettingsPage />} />
              <Route path="*" element={<NotFoundPage />} />
            </Routes>
          </main>
        </div>
        
        <Footer />
      </div>
      
      <ToastContainer 
        position="bottom-right"
        autoClose={5000}
        hideProgressBar={false}
        newestOnTop
        closeOnClick
        rtl={false}
        pauseOnFocusLoss
        draggable
        pauseOnHover
        theme="colored"
      />
    </ThemeProvider>
  );
}

export default App;