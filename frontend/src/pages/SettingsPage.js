import React, { useState, useEffect, useContext } from 'react';
import { useTranslation } from 'react-i18next';
import { ThemeContext } from '../contexts/ThemeContext';
import {
  FiSun,
  FiMoon,
  FiGlobe,
  FiBell,
  FiLock,
  FiDatabase,
  FiCpu,
  FiSave,
  FiRefreshCw,
  FiAlertTriangle,
  FiInfo,
  FiCheckCircle,
  FiToggleLeft,
  FiToggleRight,
  FiSliders,
  FiUser,
} from 'react-icons/fi';

const SettingsPage = () => {
  const { t, i18n } = useTranslation();
  const { theme, setTheme } = useContext(ThemeContext);
  const [isLoading, setIsLoading] = useState(true);
  const [saveSuccess, setSaveSuccess] = useState(false);
  
  // Settings state
  const [settings, setSettings] = useState({
    // Appearance
    theme: 'system',
    language: 'en',
    compactMode: false,
    showAnimations: true,
    
    // Notifications
    enableNotifications: true,
    priceAlerts: true,
    newsAlerts: true,
    predictionAlerts: true,
    emailNotifications: false,
    
    // Privacy
    saveSearchHistory: true,
    shareAnalytics: true,
    
    // Data
    dataRefreshInterval: '5',
    cacheStrategy: 'balanced',
    downloadHistoricalData: false,
    
    // Prediction
    defaultTimeframe: 'short_term',
    showConfidenceIntervals: true,
    showContributingFactors: true,
    showTechnicalIndicators: true,
    showNewsImpact: true,
    
    // Advanced
    enableExperimentalFeatures: false,
    debugMode: false,
    apiEndpoint: 'production',
  });
  
  // Simulate loading settings
  useEffect(() => {
    setIsLoading(true);
    
    // Simulate API call delay
    const timer = setTimeout(() => {
      // Update theme and language from actual context/i18n
      setSettings(prev => ({
        ...prev,
        theme: theme,
        language: i18n.language,
      }));
      setIsLoading(false);
    }, 1000);
    
    return () => clearTimeout(timer);
  }, [theme, i18n.language]);
  
  // Handle settings change
  const handleSettingChange = (category, setting, value) => {
    setSettings(prev => ({
      ...prev,
      [setting]: value,
    }));
    
    // Apply theme and language changes immediately
    if (setting === 'theme') {
      setTheme(value);
    } else if (setting === 'language') {
      i18n.changeLanguage(value);
    }
  };
  
  // Handle save settings
  const handleSaveSettings = () => {
    // Simulate saving settings
    setSaveSuccess(false);
    setIsLoading(true);
    
    // Simulate API call delay
    setTimeout(() => {
      setIsLoading(false);
      setSaveSuccess(true);
      
      // Hide success message after 3 seconds
      setTimeout(() => {
        setSaveSuccess(false);
      }, 3000);
    }, 1000);
  };
  
  // Handle reset settings
  const handleResetSettings = () => {
    // Simulate resetting settings
    setIsLoading(true);
    
    // Simulate API call delay
    setTimeout(() => {
      setSettings({
        // Appearance
        theme: 'system',
        language: 'en',
        compactMode: false,
        showAnimations: true,
        
        // Notifications
        enableNotifications: true,
        priceAlerts: true,
        newsAlerts: true,
        predictionAlerts: true,
        emailNotifications: false,
        
        // Privacy
        saveSearchHistory: true,
        shareAnalytics: true,
        
        // Data
        dataRefreshInterval: '5',
        cacheStrategy: 'balanced',
        downloadHistoricalData: false,
        
        // Prediction
        defaultTimeframe: 'short_term',
        showConfidenceIntervals: true,
        showContributingFactors: true,
        showTechnicalIndicators: true,
        showNewsImpact: true,
        
        // Advanced
        enableExperimentalFeatures: false,
        debugMode: false,
        apiEndpoint: 'production',
      });
      
      // Apply theme and language changes immediately
      setTheme('system');
      i18n.changeLanguage('en');
      
      setIsLoading(false);
      setSaveSuccess(true);
      
      // Hide success message after 3 seconds
      setTimeout(() => {
        setSaveSuccess(false);
      }, 3000);
    }, 1000);
  };
  
  // Toggle component
  const Toggle = ({ value, onChange }) => (
    <button
      type="button"
      className={`relative inline-flex h-6 w-11 items-center rounded-full focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 ${value ? 'bg-primary-600' : 'bg-neutral-200 dark:bg-neutral-700'}`}
      onClick={() => onChange(!value)}
    >
      <span
        className={`${value ? 'translate-x-6' : 'translate-x-1'} inline-block h-4 w-4 transform rounded-full bg-white transition-transform`}
      />
    </button>
  );
  
  // Loading skeleton
  if (isLoading) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="animate-pulse">
          <div className="h-8 bg-neutral-200 dark:bg-neutral-700 rounded w-1/3 mb-6"></div>
          <div className="h-4 bg-neutral-200 dark:bg-neutral-700 rounded w-2/3 mb-8"></div>
          
          {[...Array(5)].map((_, index) => (
            <div key={index} className="mb-8">
              <div className="h-6 bg-neutral-200 dark:bg-neutral-700 rounded w-1/4 mb-4"></div>
              <div className="border border-neutral-200 dark:border-neutral-700 rounded-lg p-6 mb-4">
                <div className="space-y-6">
                  {[...Array(3)].map((_, i) => (
                    <div key={i} className="flex justify-between items-center">
                      <div className="space-y-2">
                        <div className="h-5 bg-neutral-200 dark:bg-neutral-700 rounded w-40"></div>
                        <div className="h-4 bg-neutral-200 dark:bg-neutral-700 rounded w-64"></div>
                      </div>
                      <div className="h-6 w-11 bg-neutral-200 dark:bg-neutral-700 rounded-full"></div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ))}
          
          <div className="h-10 bg-neutral-200 dark:bg-neutral-700 rounded w-32 mb-4"></div>
        </div>
      </div>
    );
  }
  
  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-neutral-900 dark:text-neutral-100 mb-2">
          {t('settings')}
        </h1>
        <p className="text-neutral-500 dark:text-neutral-400">
          {t('settings_description')}
        </p>
      </div>
      
      {/* Success message */}
      {saveSuccess && (
        <div className="mb-6 bg-success-50 dark:bg-success-900/20 border border-success-200 dark:border-success-800 rounded-lg p-4 flex items-start">
          <FiCheckCircle className="h-5 w-5 text-success-500 mt-0.5 mr-3 flex-shrink-0" />
          <div>
            <h3 className="text-success-800 dark:text-success-400 font-medium">
              {t('settings_saved')}
            </h3>
            <p className="text-success-700 dark:text-success-300 text-sm mt-1">
              {t('settings_saved_description')}
            </p>
          </div>
        </div>
      )}
      
      {/* Appearance settings */}
      <div className="mb-8">
        <h2 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4 flex items-center">
          <FiSun className="mr-2 h-5 w-5 text-primary-500" />
          {t('appearance')}
        </h2>
        
        <div className="border border-neutral-200 dark:border-neutral-700 rounded-lg p-6 mb-4">
          <div className="space-y-6">
            {/* Theme */}
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
              <div className="mb-2 sm:mb-0">
                <label className="block text-neutral-900 dark:text-neutral-100 font-medium">
                  {t('theme')}
                </label>
                <p className="text-neutral-500 dark:text-neutral-400 text-sm mt-1">
                  {t('theme_description')}
                </p>
              </div>
              <div className="flex space-x-2">
                <button
                  type="button"
                  className={`px-3 py-2 rounded-md flex items-center ${settings.theme === 'light' ? 'bg-primary-100 dark:bg-primary-900 text-primary-800 dark:text-primary-200' : 'bg-neutral-100 dark:bg-neutral-800 text-neutral-700 dark:text-neutral-300'}`}
                  onClick={() => handleSettingChange('appearance', 'theme', 'light')}
                >
                  <FiSun className="mr-2 h-4 w-4" />
                  {t('light')}
                </button>
                <button
                  type="button"
                  className={`px-3 py-2 rounded-md flex items-center ${settings.theme === 'dark' ? 'bg-primary-100 dark:bg-primary-900 text-primary-800 dark:text-primary-200' : 'bg-neutral-100 dark:bg-neutral-800 text-neutral-700 dark:text-neutral-300'}`}
                  onClick={() => handleSettingChange('appearance', 'theme', 'dark')}
                >
                  <FiMoon className="mr-2 h-4 w-4" />
                  {t('dark')}
                </button>
                <button
                  type="button"
                  className={`px-3 py-2 rounded-md flex items-center ${settings.theme === 'system' ? 'bg-primary-100 dark:bg-primary-900 text-primary-800 dark:text-primary-200' : 'bg-neutral-100 dark:bg-neutral-800 text-neutral-700 dark:text-neutral-300'}`}
                  onClick={() => handleSettingChange('appearance', 'theme', 'system')}
                >
                  <FiCpu className="mr-2 h-4 w-4" />
                  {t('system')}
                </button>
              </div>
            </div>
            
            {/* Language */}
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
              <div className="mb-2 sm:mb-0">
                <label className="block text-neutral-900 dark:text-neutral-100 font-medium">
                  {t('language')}
                </label>
                <p className="text-neutral-500 dark:text-neutral-400 text-sm mt-1">
                  {t('language_description')}
                </p>
              </div>
              <div className="flex space-x-2">
                <button
                  type="button"
                  className={`px-3 py-2 rounded-md flex items-center ${settings.language === 'en' ? 'bg-primary-100 dark:bg-primary-900 text-primary-800 dark:text-primary-200' : 'bg-neutral-100 dark:bg-neutral-800 text-neutral-700 dark:text-neutral-300'}`}
                  onClick={() => handleSettingChange('appearance', 'language', 'en')}
                >
                  <span className="mr-2">🇬🇧</span>
                  English
                </button>
                <button
                  type="button"
                  className={`px-3 py-2 rounded-md flex items-center ${settings.language === 'hi' ? 'bg-primary-100 dark:bg-primary-900 text-primary-800 dark:text-primary-200' : 'bg-neutral-100 dark:bg-neutral-800 text-neutral-700 dark:text-neutral-300'}`}
                  onClick={() => handleSettingChange('appearance', 'language', 'hi')}
                >
                  <span className="mr-2">🇮🇳</span>
                  हिन्दी
                </button>
              </div>
            </div>
            
            {/* Compact Mode */}
            <div className="flex justify-between items-center">
              <div>
                <label className="block text-neutral-900 dark:text-neutral-100 font-medium">
                  {t('compact_mode')}
                </label>
                <p className="text-neutral-500 dark:text-neutral-400 text-sm mt-1">
                  {t('compact_mode_description')}
                </p>
              </div>
              <Toggle
                value={settings.compactMode}
                onChange={(value) => handleSettingChange('appearance', 'compactMode', value)}
              />
            </div>
            
            {/* Show Animations */}
            <div className="flex justify-between items-center">
              <div>
                <label className="block text-neutral-900 dark:text-neutral-100 font-medium">
                  {t('show_animations')}
                </label>
                <p className="text-neutral-500 dark:text-neutral-400 text-sm mt-1">
                  {t('show_animations_description')}
                </p>
              </div>
              <Toggle
                value={settings.showAnimations}
                onChange={(value) => handleSettingChange('appearance', 'showAnimations', value)}
              />
            </div>
          </div>
        </div>
      </div>
      
      {/* Notification settings */}
      <div className="mb-8">
        <h2 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4 flex items-center">
          <FiBell className="mr-2 h-5 w-5 text-primary-500" />
          {t('notifications')}
        </h2>
        
        <div className="border border-neutral-200 dark:border-neutral-700 rounded-lg p-6 mb-4">
          <div className="space-y-6">
            {/* Enable Notifications */}
            <div className="flex justify-between items-center">
              <div>
                <label className="block text-neutral-900 dark:text-neutral-100 font-medium">
                  {t('enable_notifications')}
                </label>
                <p className="text-neutral-500 dark:text-neutral-400 text-sm mt-1">
                  {t('enable_notifications_description')}
                </p>
              </div>
              <Toggle
                value={settings.enableNotifications}
                onChange={(value) => handleSettingChange('notifications', 'enableNotifications', value)}
              />
            </div>
            
            {settings.enableNotifications && (
              <>
                {/* Price Alerts */}
                <div className="flex justify-between items-center pl-6 border-l-2 border-neutral-200 dark:border-neutral-700">
                  <div>
                    <label className="block text-neutral-900 dark:text-neutral-100 font-medium">
                      {t('price_alerts')}
                    </label>
                    <p className="text-neutral-500 dark:text-neutral-400 text-sm mt-1">
                      {t('price_alerts_description')}
                    </p>
                  </div>
                  <Toggle
                    value={settings.priceAlerts}
                    onChange={(value) => handleSettingChange('notifications', 'priceAlerts', value)}
                  />
                </div>
                
                {/* News Alerts */}
                <div className="flex justify-between items-center pl-6 border-l-2 border-neutral-200 dark:border-neutral-700">
                  <div>
                    <label className="block text-neutral-900 dark:text-neutral-100 font-medium">
                      {t('news_alerts')}
                    </label>
                    <p className="text-neutral-500 dark:text-neutral-400 text-sm mt-1">
                      {t('news_alerts_description')}
                    </p>
                  </div>
                  <Toggle
                    value={settings.newsAlerts}
                    onChange={(value) => handleSettingChange('notifications', 'newsAlerts', value)}
                  />
                </div>
                
                {/* Prediction Alerts */}
                <div className="flex justify-between items-center pl-6 border-l-2 border-neutral-200 dark:border-neutral-700">
                  <div>
                    <label className="block text-neutral-900 dark:text-neutral-100 font-medium">
                      {t('prediction_alerts')}
                    </label>
                    <p className="text-neutral-500 dark:text-neutral-400 text-sm mt-1">
                      {t('prediction_alerts_description')}
                    </p>
                  </div>
                  <Toggle
                    value={settings.predictionAlerts}
                    onChange={(value) => handleSettingChange('notifications', 'predictionAlerts', value)}
                  />
                </div>
              </>
            )}
            
            {/* Email Notifications */}
            <div className="flex justify-between items-center">
              <div>
                <label className="block text-neutral-900 dark:text-neutral-100 font-medium">
                  {t('email_notifications')}
                </label>
                <p className="text-neutral-500 dark:text-neutral-400 text-sm mt-1">
                  {t('email_notifications_description')}
                </p>
              </div>
              <Toggle
                value={settings.emailNotifications}
                onChange={(value) => handleSettingChange('notifications', 'emailNotifications', value)}
              />
            </div>
          </div>
        </div>
      </div>
      
      {/* Privacy settings */}
      <div className="mb-8">
        <h2 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4 flex items-center">
          <FiLock className="mr-2 h-5 w-5 text-primary-500" />
          {t('privacy')}
        </h2>
        
        <div className="border border-neutral-200 dark:border-neutral-700 rounded-lg p-6 mb-4">
          <div className="space-y-6">
            {/* Save Search History */}
            <div className="flex justify-between items-center">
              <div>
                <label className="block text-neutral-900 dark:text-neutral-100 font-medium">
                  {t('save_search_history')}
                </label>
                <p className="text-neutral-500 dark:text-neutral-400 text-sm mt-1">
                  {t('save_search_history_description')}
                </p>
              </div>
              <Toggle
                value={settings.saveSearchHistory}
                onChange={(value) => handleSettingChange('privacy', 'saveSearchHistory', value)}
              />
            </div>
            
            {/* Share Analytics */}
            <div className="flex justify-between items-center">
              <div>
                <label className="block text-neutral-900 dark:text-neutral-100 font-medium">
                  {t('share_analytics')}
                </label>
                <p className="text-neutral-500 dark:text-neutral-400 text-sm mt-1">
                  {t('share_analytics_description')}
                </p>
              </div>
              <Toggle
                value={settings.shareAnalytics}
                onChange={(value) => handleSettingChange('privacy', 'shareAnalytics', value)}
              />
            </div>
          </div>
        </div>
      </div>
      
      {/* Data settings */}
      <div className="mb-8">
        <h2 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4 flex items-center">
          <FiDatabase className="mr-2 h-5 w-5 text-primary-500" />
          {t('data')}
        </h2>
        
        <div className="border border-neutral-200 dark:border-neutral-700 rounded-lg p-6 mb-4">
          <div className="space-y-6">
            {/* Data Refresh Interval */}
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
              <div className="mb-2 sm:mb-0">
                <label className="block text-neutral-900 dark:text-neutral-100 font-medium">
                  {t('data_refresh_interval')}
                </label>
                <p className="text-neutral-500 dark:text-neutral-400 text-sm mt-1">
                  {t('data_refresh_interval_description')}
                </p>
              </div>
              <select
                className="input py-2 pl-3 pr-10 sm:w-auto"
                value={settings.dataRefreshInterval}
                onChange={(e) => handleSettingChange('data', 'dataRefreshInterval', e.target.value)}
              >
                <option value="1">{t('1_minute')}</option>
                <option value="5">{t('5_minutes')}</option>
                <option value="15">{t('15_minutes')}</option>
                <option value="30">{t('30_minutes')}</option>
                <option value="60">{t('60_minutes')}</option>
              </select>
            </div>
            
            {/* Cache Strategy */}
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
              <div className="mb-2 sm:mb-0">
                <label className="block text-neutral-900 dark:text-neutral-100 font-medium">
                  {t('cache_strategy')}
                </label>
                <p className="text-neutral-500 dark:text-neutral-400 text-sm mt-1">
                  {t('cache_strategy_description')}
                </p>
              </div>
              <select
                className="input py-2 pl-3 pr-10 sm:w-auto"
                value={settings.cacheStrategy}
                onChange={(e) => handleSettingChange('data', 'cacheStrategy', e.target.value)}
              >
                <option value="aggressive">{t('aggressive_caching')}</option>
                <option value="balanced">{t('balanced_caching')}</option>
                <option value="minimal">{t('minimal_caching')}</option>
                <option value="disabled">{t('disable_caching')}</option>
              </select>
            </div>
            
            {/* Download Historical Data */}
            <div className="flex justify-between items-center">
              <div>
                <label className="block text-neutral-900 dark:text-neutral-100 font-medium">
                  {t('download_historical_data')}
                </label>
                <p className="text-neutral-500 dark:text-neutral-400 text-sm mt-1">
                  {t('download_historical_data_description')}
                </p>
              </div>
              <Toggle
                value={settings.downloadHistoricalData}
                onChange={(value) => handleSettingChange('data', 'downloadHistoricalData', value)}
              />
            </div>
          </div>
        </div>
      </div>
      
      {/* Prediction settings */}
      <div className="mb-8">
        <h2 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4 flex items-center">
          <FiTrendingUp className="mr-2 h-5 w-5 text-primary-500" />
          {t('prediction')}
        </h2>
        
        <div className="border border-neutral-200 dark:border-neutral-700 rounded-lg p-6 mb-4">
          <div className="space-y-6">
            {/* Default Timeframe */}
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
              <div className="mb-2 sm:mb-0">
                <label className="block text-neutral-900 dark:text-neutral-100 font-medium">
                  {t('default_timeframe')}
                </label>
                <p className="text-neutral-500 dark:text-neutral-400 text-sm mt-1">
                  {t('default_timeframe_description')}
                </p>
              </div>
              <select
                className="input py-2 pl-3 pr-10 sm:w-auto"
                value={settings.defaultTimeframe}
                onChange={(e) => handleSettingChange('prediction', 'defaultTimeframe', e.target.value)}
              >
                <option value="intraday">{t('intraday')}</option>
                <option value="short_term">{t('short_term')}</option>
                <option value="medium_term">{t('medium_term')}</option>
                <option value="long_term">{t('long_term')}</option>
              </select>
            </div>
            
            {/* Show Confidence Intervals */}
            <div className="flex justify-between items-center">
              <div>
                <label className="block text-neutral-900 dark:text-neutral-100 font-medium">
                  {t('show_confidence_intervals')}
                </label>
                <p className="text-neutral-500 dark:text-neutral-400 text-sm mt-1">
                  {t('show_confidence_intervals_description')}
                </p>
              </div>
              <Toggle
                value={settings.showConfidenceIntervals}
                onChange={(value) => handleSettingChange('prediction', 'showConfidenceIntervals', value)}
              />
            </div>
            
            {/* Show Contributing Factors */}
            <div className="flex justify-between items-center">
              <div>
                <label className="block text-neutral-900 dark:text-neutral-100 font-medium">
                  {t('show_contributing_factors')}
                </label>
                <p className="text-neutral-500 dark:text-neutral-400 text-sm mt-1">
                  {t('show_contributing_factors_description')}
                </p>
              </div>
              <Toggle
                value={settings.showContributingFactors}
                onChange={(value) => handleSettingChange('prediction', 'showContributingFactors', value)}
              />
            </div>
            
            {/* Show Technical Indicators */}
            <div className="flex justify-between items-center">
              <div>
                <label className="block text-neutral-900 dark:text-neutral-100 font-medium">
                  {t('show_technical_indicators')}
                </label>
                <p className="text-neutral-500 dark:text-neutral-400 text-sm mt-1">
                  {t('show_technical_indicators_description')}
                </p>
              </div>
              <Toggle
                value={settings.showTechnicalIndicators}
                onChange={(value) => handleSettingChange('prediction', 'showTechnicalIndicators', value)}
              />
            </div>
            
            {/* Show News Impact */}
            <div className="flex justify-between items-center">
              <div>
                <label className="block text-neutral-900 dark:text-neutral-100 font-medium">
                  {t('show_news_impact')}
                </label>
                <p className="text-neutral-500 dark:text-neutral-400 text-sm mt-1">
                  {t('show_news_impact_description')}
                </p>
              </div>
              <Toggle
                value={settings.showNewsImpact}
                onChange={(value) => handleSettingChange('prediction', 'showNewsImpact', value)}
              />
            </div>
          </div>
        </div>
      </div>
      
      {/* Advanced settings */}
      <div className="mb-8">
        <h2 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4 flex items-center">
          <FiSliders className="mr-2 h-5 w-5 text-primary-500" />
          {t('advanced')}
        </h2>
        
        <div className="border border-neutral-200 dark:border-neutral-700 rounded-lg p-6 mb-4">
          <div className="space-y-6">
            {/* Experimental Features */}
            <div className="flex justify-between items-center">
              <div>
                <label className="block text-neutral-900 dark:text-neutral-100 font-medium">
                  {t('enable_experimental_features')}
                </label>
                <p className="text-neutral-500 dark:text-neutral-400 text-sm mt-1">
                  {t('enable_experimental_features_description')}
                </p>
              </div>
              <Toggle
                value={settings.enableExperimentalFeatures}
                onChange={(value) => handleSettingChange('advanced', 'enableExperimentalFeatures', value)}
              />
            </div>
            
            {/* Debug Mode */}
            <div className="flex justify-between items-center">
              <div>
                <label className="block text-neutral-900 dark:text-neutral-100 font-medium">
                  {t('debug_mode')}
                </label>
                <p className="text-neutral-500 dark:text-neutral-400 text-sm mt-1">
                  {t('debug_mode_description')}
                </p>
              </div>
              <Toggle
                value={settings.debugMode}
                onChange={(value) => handleSettingChange('advanced', 'debugMode', value)}
              />
            </div>
            
            {/* API Endpoint */}
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
              <div className="mb-2 sm:mb-0">
                <label className="block text-neutral-900 dark:text-neutral-100 font-medium">
                  {t('api_endpoint')}
                </label>
                <p className="text-neutral-500 dark:text-neutral-400 text-sm mt-1">
                  {t('api_endpoint_description')}
                </p>
              </div>
              <select
                className="input py-2 pl-3 pr-10 sm:w-auto"
                value={settings.apiEndpoint}
                onChange={(e) => handleSettingChange('advanced', 'apiEndpoint', e.target.value)}
              >
                <option value="production">{t('production')}</option>
                <option value="staging">{t('staging')}</option>
                <option value="development">{t('development')}</option>
                <option value="local">{t('local')}</option>
              </select>
            </div>
          </div>
        </div>
        
        {/* Warning for advanced settings */}
        <div className="bg-warning-50 dark:bg-warning-900/20 border border-warning-200 dark:border-warning-800 rounded-lg p-4 flex items-start">
          <FiAlertTriangle className="h-5 w-5 text-warning-500 mt-0.5 mr-3 flex-shrink-0" />
          <div>
            <h3 className="text-warning-800 dark:text-warning-400 font-medium">
              {t('advanced_settings_warning')}
            </h3>
            <p className="text-warning-700 dark:text-warning-300 text-sm mt-1">
              {t('advanced_settings_warning_description')}
            </p>
          </div>
        </div>
      </div>
      
      {/* User profile section */}
      <div className="mb-8">
        <h2 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4 flex items-center">
          <FiUser className="mr-2 h-5 w-5 text-primary-500" />
          {t('user_profile')}
        </h2>
        
        <div className="border border-neutral-200 dark:border-neutral-700 rounded-lg p-6 mb-4">
          <div className="flex items-center mb-6">
            <div className="h-16 w-16 rounded-full bg-primary-100 dark:bg-primary-900 flex items-center justify-center text-primary-600 dark:text-primary-400 text-2xl font-bold mr-4">
              U
            </div>
            <div>
              <h3 className="text-lg font-medium text-neutral-900 dark:text-neutral-100">
                {t('user')}
              </h3>
              <p className="text-neutral-500 dark:text-neutral-400">
                user@example.com
              </p>
            </div>
            <button className="btn btn-outline ml-auto">
              {t('edit_profile')}
            </button>
          </div>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-1">
                {t('account_type')}
              </label>
              <div className="text-neutral-900 dark:text-neutral-100 flex items-center">
                <span className="mr-2">{t('free_tier')}</span>
                <span className="px-2 py-1 bg-primary-100 dark:bg-primary-900 text-primary-800 dark:text-primary-200 text-xs rounded-full">
                  {t('upgrade_available')}
                </span>
              </div>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-1">
                {t('joined_date')}
              </label>
              <div className="text-neutral-900 dark:text-neutral-100">
                {new Date().toLocaleDateString()}
              </div>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-1">
                {t('data_usage')}
              </label>
              <div className="text-neutral-900 dark:text-neutral-100">
                {t('data_usage_info')}
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Action buttons */}
      <div className="flex flex-col sm:flex-row space-y-4 sm:space-y-0 sm:space-x-4">
        <button
          type="button"
          className="btn btn-primary flex items-center justify-center"
          onClick={handleSaveSettings}
        >
          <FiSave className="mr-2 h-4 w-4" />
          {t('save_settings')}
        </button>
        
        <button
          type="button"
          className="btn btn-outline flex items-center justify-center"
          onClick={handleResetSettings}
        >
          <FiRefreshCw className="mr-2 h-4 w-4" />
          {t('reset_to_defaults')}
        </button>
      </div>
      
      {/* Info section */}
      <div className="mt-8 bg-neutral-50 dark:bg-neutral-800 rounded-lg p-4 border border-neutral-200 dark:border-neutral-700">
        <h4 className="font-medium text-neutral-900 dark:text-neutral-100 mb-2 flex items-center">
          <FiInfo className="mr-2 h-4 w-4 text-primary-500" />
          {t('about_settings')}
        </h4>
        <p className="text-sm text-neutral-600 dark:text-neutral-400">
          {t('settings_info')}
        </p>
      </div>
    </div>
  );
};

export default SettingsPage;