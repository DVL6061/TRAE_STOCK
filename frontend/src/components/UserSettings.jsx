import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import {
  UserIcon,
  PaintBrushIcon,
  BellIcon,
  ChartBarIcon,
  ShieldCheckIcon,
  CogIcon,
  CheckIcon,
  XMarkIcon,
  ExclamationTriangleIcon,
  ArrowDownTrayIcon,
  ArrowPathIcon
} from '@heroicons/react/24/outline';

const UserSettings = () => {
  const { t, i18n } = useTranslation();
  const [activeTab, setActiveTab] = useState('profile');
  const [settings, setSettings] = useState({
    // Profile settings
    firstName: '',
    lastName: '',
    email: '',
    phone: '',
    bio: '',
    
    // Appearance settings
    theme: 'auto',
    language: 'en',
    fontSize: 'medium',
    chartTheme: 'lightCharts',
    compactMode: false,
    
    // Notification settings
    emailNotifications: true,
    pushNotifications: true,
    priceAlerts: true,
    newsAlerts: true,
    portfolioUpdates: true,
    marketOpenClose: false,
    weeklyReports: true,
    tradingSignals: true,
    
    // Trading settings
    defaultOrderType: 'marketOrder',
    riskTolerance: 'mediumRisk',
    stopLossPercentage: 5,
    takeProfitPercentage: 10,
    enablePaperTrading: false,
    
    // Security settings
    twoFactorAuth: false,
    loginAlerts: true,
    sessionTimeout: 30,
    apiAccess: false,
    dataExport: true
  });
  
  const [showResetConfirm, setShowResetConfirm] = useState(false);
  const [notification, setNotification] = useState({ show: false, type: '', message: '' });

  const tabs = [
    { id: 'profile', name: t('settings.profile'), icon: UserIcon },
    { id: 'appearance', name: t('settings.appearance'), icon: PaintBrushIcon },
    { id: 'notifications', name: t('settings.notifications'), icon: BellIcon },
    { id: 'trading', name: t('settings.trading'), icon: ChartBarIcon },
    { id: 'security', name: t('settings.security'), icon: ShieldCheckIcon },
    { id: 'advanced', name: t('settings.advanced'), icon: CogIcon }
  ];

  useEffect(() => {
    // Load settings from localStorage or API
    const savedSettings = localStorage.getItem('userSettings');
    if (savedSettings) {
      setSettings(prev => ({ ...prev, ...JSON.parse(savedSettings) }));
    }
  }, []);

  const handleSettingChange = (key, value) => {
    setSettings(prev => {
      const newSettings = { ...prev, [key]: value };
      localStorage.setItem('userSettings', JSON.stringify(newSettings));
      return newSettings;
    });
    
    // Handle language change
    if (key === 'language') {
      i18n.changeLanguage(value);
    }
    
    // Handle theme change
    if (key === 'theme') {
      document.documentElement.setAttribute('data-theme', value);
    }
  };

  const showNotification = (type, message) => {
    setNotification({ show: true, type, message });
    setTimeout(() => setNotification({ show: false, type: '', message: '' }), 3000);
  };

  const handleProfileUpdate = () => {
    // Simulate API call
    setTimeout(() => {
      showNotification('success', t('settings.profileUpdated'));
    }, 500);
  };

  const handleResetToDefaults = () => {
    const defaultSettings = {
      theme: 'auto',
      language: 'en',
      fontSize: 'medium',
      chartTheme: 'lightCharts',
      compactMode: false,
      emailNotifications: true,
      pushNotifications: true,
      priceAlerts: true,
      newsAlerts: true,
      portfolioUpdates: true,
      marketOpenClose: false,
      weeklyReports: true,
      tradingSignals: true,
      defaultOrderType: 'marketOrder',
      riskTolerance: 'mediumRisk',
      stopLossPercentage: 5,
      takeProfitPercentage: 10,
      enablePaperTrading: false,
      twoFactorAuth: false,
      loginAlerts: true,
      sessionTimeout: 30,
      apiAccess: false,
      dataExport: true
    };
    
    setSettings(prev => ({ ...prev, ...defaultSettings }));
    localStorage.setItem('userSettings', JSON.stringify(defaultSettings));
    setShowResetConfirm(false);
    showNotification('success', t('settings.resetSuccess'));
  };

  const handleExportData = () => {
    try {
      const dataToExport = {
        settings,
        exportDate: new Date().toISOString(),
        version: '1.0'
      };
      
      const blob = new Blob([JSON.stringify(dataToExport, null, 2)], {
        type: 'application/json'
      });
      
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `stock-app-settings-${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
      showNotification('success', t('settings.exportSuccess'));
    } catch (error) {
      showNotification('error', t('settings.exportError'));
    }
  };

  const renderProfileTab = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            {t('settings.firstName')}
          </label>
          <input
            type="text"
            value={settings.firstName}
            onChange={(e) => handleSettingChange('firstName', e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            {t('settings.lastName')}
          </label>
          <input
            type="text"
            value={settings.lastName}
            onChange={(e) => handleSettingChange('lastName', e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
          />
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            {t('settings.email')}
          </label>
          <input
            type="email"
            value={settings.email}
            onChange={(e) => handleSettingChange('email', e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            {t('settings.phone')}
          </label>
          <input
            type="tel"
            value={settings.phone}
            onChange={(e) => handleSettingChange('phone', e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
          />
        </div>
      </div>
      
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          {t('settings.bio')}
        </label>
        <textarea
          value={settings.bio}
          onChange={(e) => handleSettingChange('bio', e.target.value)}
          placeholder={t('settings.bioPlaceholder')}
          rows={4}
          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
        />
      </div>
      
      <button
        onClick={handleProfileUpdate}
        className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
      >
        {t('common.update')}
      </button>
    </div>
  );

  const renderAppearanceTab = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            {t('settings.theme')}
          </label>
          <select
            value={settings.theme}
            onChange={(e) => handleSettingChange('theme', e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
          >
            <option value="light">{t('settings.lightTheme')}</option>
            <option value="dark">{t('settings.darkTheme')}</option>
            <option value="auto">{t('settings.autoTheme')}</option>
          </select>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            {t('settings.language')}
          </label>
          <select
            value={settings.language}
            onChange={(e) => handleSettingChange('language', e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
          >
            <option value="en">English</option>
            <option value="hi">हिंदी</option>
          </select>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            {t('settings.fontSize')}
          </label>
          <select
            value={settings.fontSize}
            onChange={(e) => handleSettingChange('fontSize', e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
          >
            <option value="small">{t('settings.small')}</option>
            <option value="medium">{t('settings.medium')}</option>
            <option value="large">{t('settings.large')}</option>
          </select>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            {t('settings.chartTheme')}
          </label>
          <select
            value={settings.chartTheme}
            onChange={(e) => handleSettingChange('chartTheme', e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
          >
            <option value="lightCharts">{t('settings.lightCharts')}</option>
            <option value="darkCharts">{t('settings.darkCharts')}</option>
            <option value="colorfulCharts">{t('settings.colorfulCharts')}</option>
          </select>
        </div>
      </div>
      
      <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
        <div>
          <h4 className="font-medium text-gray-900 dark:text-white">{t('settings.compactMode')}</h4>
          <p className="text-sm text-gray-500 dark:text-gray-400">{t('settings.compactModeDesc')}</p>
        </div>
        <button
          onClick={() => handleSettingChange('compactMode', !settings.compactMode)}
          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
            settings.compactMode ? 'bg-blue-600' : 'bg-gray-200 dark:bg-gray-700'
          }`}
        >
          <span
            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
              settings.compactMode ? 'translate-x-6' : 'translate-x-1'
            }`}
          />
        </button>
      </div>
    </div>
  );

  const renderNotificationsTab = () => {
    const notificationSettings = [
      { key: 'emailNotifications', title: t('settings.emailNotifications'), desc: t('settings.emailNotificationsDesc') },
      { key: 'pushNotifications', title: t('settings.pushNotifications'), desc: t('settings.pushNotificationsDesc') },
      { key: 'priceAlerts', title: t('settings.priceAlerts'), desc: t('settings.priceAlertsDesc') },
      { key: 'newsAlerts', title: t('settings.newsAlerts'), desc: t('settings.newsAlertsDesc') },
      { key: 'portfolioUpdates', title: t('settings.portfolioUpdates'), desc: t('settings.portfolioUpdatesDesc') },
      { key: 'marketOpenClose', title: t('settings.marketOpenClose'), desc: t('settings.marketOpenCloseDesc') },
      { key: 'weeklyReports', title: t('settings.weeklyReports'), desc: t('settings.weeklyReportsDesc') },
      { key: 'tradingSignals', title: t('settings.tradingSignals'), desc: t('settings.tradingSignalsDesc') }
    ];

    return (
      <div className="space-y-4">
        {notificationSettings.map((setting) => (
          <div key={setting.key} className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
            <div>
              <h4 className="font-medium text-gray-900 dark:text-white">{setting.title}</h4>
              <p className="text-sm text-gray-500 dark:text-gray-400">{setting.desc}</p>
            </div>
            <button
              onClick={() => handleSettingChange(setting.key, !settings[setting.key])}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                settings[setting.key] ? 'bg-blue-600' : 'bg-gray-200 dark:bg-gray-700'
              }`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  settings[setting.key] ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>
        ))}
      </div>
    );
  };

  const renderTradingTab = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            {t('settings.defaultOrderType')}
          </label>
          <select
            value={settings.defaultOrderType}
            onChange={(e) => handleSettingChange('defaultOrderType', e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
          >
            <option value="marketOrder">{t('settings.marketOrder')}</option>
            <option value="limitOrder">{t('settings.limitOrder')}</option>
            <option value="stopOrder">{t('settings.stopOrder')}</option>
          </select>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            {t('settings.riskTolerance')}
          </label>
          <select
            value={settings.riskTolerance}
            onChange={(e) => handleSettingChange('riskTolerance', e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
          >
            <option value="lowRisk">{t('settings.lowRisk')}</option>
            <option value="mediumRisk">{t('settings.mediumRisk')}</option>
            <option value="highRisk">{t('settings.highRisk')}</option>
          </select>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            {t('settings.stopLossPercentage')}
          </label>
          <input
            type="number"
            min="0"
            max="100"
            step="0.1"
            value={settings.stopLossPercentage}
            onChange={(e) => handleSettingChange('stopLossPercentage', parseFloat(e.target.value))}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            {t('settings.takeProfitPercentage')}
          </label>
          <input
            type="number"
            min="0"
            max="1000"
            step="0.1"
            value={settings.takeProfitPercentage}
            onChange={(e) => handleSettingChange('takeProfitPercentage', parseFloat(e.target.value))}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
          />
        </div>
      </div>
      
      <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
        <div>
          <h4 className="font-medium text-gray-900 dark:text-white">{t('settings.enablePaperTrading')}</h4>
          <p className="text-sm text-gray-500 dark:text-gray-400">{t('settings.paperTradingDesc')}</p>
        </div>
        <button
          onClick={() => handleSettingChange('enablePaperTrading', !settings.enablePaperTrading)}
          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
            settings.enablePaperTrading ? 'bg-blue-600' : 'bg-gray-200 dark:bg-gray-700'
          }`}
        >
          <span
            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
              settings.enablePaperTrading ? 'translate-x-6' : 'translate-x-1'
            }`}
          />
        </button>
      </div>
    </div>
  );

  const renderSecurityTab = () => {
    const securitySettings = [
      { key: 'twoFactorAuth', title: t('settings.twoFactorAuth'), desc: t('settings.twoFactorAuthDesc') },
      { key: 'loginAlerts', title: t('settings.loginAlerts'), desc: t('settings.loginAlertsDesc') },
      { key: 'apiAccess', title: t('settings.apiAccess'), desc: t('settings.apiAccessDesc') },
      { key: 'dataExport', title: t('settings.dataExport'), desc: t('settings.dataExportDesc') }
    ];

    return (
      <div className="space-y-6">
        <div className="space-y-4">
          {securitySettings.map((setting) => (
            <div key={setting.key} className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <div>
                <h4 className="font-medium text-gray-900 dark:text-white">{setting.title}</h4>
                <p className="text-sm text-gray-500 dark:text-gray-400">{setting.desc}</p>
              </div>
              <button
                onClick={() => handleSettingChange(setting.key, !settings[setting.key])}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  settings[setting.key] ? 'bg-blue-600' : 'bg-gray-200 dark:bg-gray-700'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    settings[setting.key] ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>
          ))}
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            {t('settings.sessionTimeout')} (minutes)
          </label>
          <input
            type="number"
            min="5"
            max="480"
            step="5"
            value={settings.sessionTimeout}
            onChange={(e) => handleSettingChange('sessionTimeout', parseInt(e.target.value))}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
          />
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">{t('settings.sessionTimeoutDesc')}</p>
        </div>
      </div>
    );
  };

  const renderAdvancedTab = () => (
    <div className="space-y-6">
      <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-6">
        <div className="flex items-center mb-4">
          <ExclamationTriangleIcon className="h-6 w-6 text-red-600 dark:text-red-400 mr-2" />
          <h3 className="text-lg font-semibold text-red-800 dark:text-red-200">{t('settings.dangerZone')}</h3>
        </div>
        <p className="text-red-700 dark:text-red-300 mb-6">{t('settings.dangerZoneDesc')}</p>
        
        <div className="space-y-4">
          <button
            onClick={handleExportData}
            className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <ArrowDownTrayIcon className="h-4 w-4 mr-2" />
            {t('settings.exportData')}
          </button>
          
          <button
            onClick={() => setShowResetConfirm(true)}
            className="flex items-center px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
          >
            <ArrowPathIcon className="h-4 w-4 mr-2" />
            {t('settings.resetToDefaults')}
          </button>
        </div>
      </div>
    </div>
  );

  const renderTabContent = () => {
    switch (activeTab) {
      case 'profile': return renderProfileTab();
      case 'appearance': return renderAppearanceTab();
      case 'notifications': return renderNotificationsTab();
      case 'trading': return renderTradingTab();
      case 'security': return renderSecurityTab();
      case 'advanced': return renderAdvancedTab();
      default: return renderProfileTab();
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">{t('settings.title')}</h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">{t('settings.subtitle')}</p>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg overflow-hidden">
          <div className="flex flex-col lg:flex-row">
            {/* Sidebar */}
            <div className="lg:w-1/4 bg-gray-50 dark:bg-gray-700 border-r border-gray-200 dark:border-gray-600">
              <nav className="p-4 space-y-2">
                {tabs.map((tab) => {
                  const Icon = tab.icon;
                  return (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id)}
                      className={`w-full flex items-center px-3 py-2 text-left rounded-lg transition-colors ${
                        activeTab === tab.id
                          ? 'bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300'
                          : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-600'
                      }`}
                    >
                      <Icon className="h-5 w-5 mr-3" />
                      {tab.name}
                    </button>
                  );
                })}
              </nav>
            </div>

            {/* Main content */}
            <div className="lg:w-3/4 p-6">
              {renderTabContent()}
            </div>
          </div>
        </div>
      </div>

      {/* Notification */}
      {notification.show && (
        <div className="fixed top-4 right-4 z-50">
          <div className={`flex items-center p-4 rounded-lg shadow-lg ${
            notification.type === 'success' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
          }`}>
            {notification.type === 'success' ? (
              <CheckIcon className="h-5 w-5 mr-2" />
            ) : (
              <XMarkIcon className="h-5 w-5 mr-2" />
            )}
            {notification.message}
          </div>
        </div>
      )}

      {/* Reset confirmation modal */}
      {showResetConfirm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 max-w-md mx-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              {t('common.confirm')}
            </h3>
            <p className="text-gray-600 dark:text-gray-400 mb-6">
              {t('settings.resetConfirm')}
            </p>
            <div className="flex space-x-4">
              <button
                onClick={() => setShowResetConfirm(false)}
                className="flex-1 px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
              >
                {t('common.cancel')}
              </button>
              <button
                onClick={handleResetToDefaults}
                className="flex-1 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
              >
                {t('common.confirm')}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default UserSettings;