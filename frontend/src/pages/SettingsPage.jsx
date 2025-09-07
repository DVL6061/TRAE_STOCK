import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { useAuth } from '../contexts/AuthContext';
import { toast } from 'react-toastify';
import {
  UserIcon,
  BellIcon,
  PaintBrushIcon,
  ChartBarIcon,
  ShieldCheckIcon,
  CogIcon,
  EyeIcon,
  EyeSlashIcon,
  CheckIcon,
  XMarkIcon
} from '@heroicons/react/24/outline';

const SettingsPage = () => {
  const { t, i18n } = useTranslation();
  const { user, updateUserProfile } = useAuth();
  const [activeTab, setActiveTab] = useState('profile');
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  
  // Profile settings
  const [profileData, setProfileData] = useState({
    firstName: user?.firstName || '',
    lastName: user?.lastName || '',
    email: user?.email || '',
    phone: user?.phone || '',
    bio: user?.bio || '',
    avatar: user?.avatar || ''
  });

  // Theme and appearance settings
  const [themeSettings, setThemeSettings] = useState({
    theme: localStorage.getItem('theme') || 'dark',
    language: i18n.language || 'en',
    fontSize: localStorage.getItem('fontSize') || 'medium',
    chartTheme: localStorage.getItem('chartTheme') || 'dark',
    compactMode: localStorage.getItem('compactMode') === 'true'
  });

  // Notification settings
  const [notificationSettings, setNotificationSettings] = useState({
    emailNotifications: true,
    pushNotifications: true,
    priceAlerts: true,
    newsAlerts: true,
    portfolioUpdates: true,
    marketOpenClose: false,
    weeklyReports: true,
    tradingSignals: true
  });

  // Trading preferences
  const [tradingSettings, setTradingSettings] = useState({
    defaultOrderType: 'market',
    riskTolerance: 'medium',
    autoInvest: false,
    stopLossPercentage: 5,
    takeProfitPercentage: 15,
    maxPositionSize: 10000,
    preferredTimeframe: '1D',
    enablePaperTrading: false
  });

  // Security settings
  const [securitySettings, setSecuritySettings] = useState({
    twoFactorAuth: false,
    loginAlerts: true,
    sessionTimeout: 30,
    apiAccess: false,
    dataExport: false
  });

  const tabs = [
    { id: 'profile', name: t('settings.profile'), icon: UserIcon },
    { id: 'appearance', name: t('settings.appearance'), icon: PaintBrushIcon },
    { id: 'notifications', name: t('settings.notifications'), icon: BellIcon },
    { id: 'trading', name: t('settings.trading'), icon: ChartBarIcon },
    { id: 'security', name: t('settings.security'), icon: ShieldCheckIcon },
    { id: 'advanced', name: t('settings.advanced'), icon: CogIcon }
  ];

  const handleProfileUpdate = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      await updateUserProfile(profileData);
      toast.success(t('settings.profileUpdated'));
    } catch (error) {
      toast.error(t('settings.updateError'));
    } finally {
      setLoading(false);
    }
  };

  const handleThemeChange = (key, value) => {
    setThemeSettings(prev => ({ ...prev, [key]: value }));
    localStorage.setItem(key, value);
    
    if (key === 'theme') {
      document.documentElement.classList.toggle('dark', value === 'dark');
    }
    if (key === 'language') {
      i18n.changeLanguage(value);
    }
  };

  const handleNotificationChange = (key, value) => {
    setNotificationSettings(prev => ({ ...prev, [key]: value }));
  };

  const handleTradingChange = (key, value) => {
    setTradingSettings(prev => ({ ...prev, [key]: value }));
  };

  const handleSecurityChange = (key, value) => {
    setSecuritySettings(prev => ({ ...prev, [key]: value }));
  };

  const resetToDefaults = () => {
    if (window.confirm(t('settings.resetConfirm'))) {
      setThemeSettings({
        theme: 'dark',
        language: 'en',
        fontSize: 'medium',
        chartTheme: 'dark',
        compactMode: false
      });
      toast.success(t('settings.resetSuccess'));
    }
  };

  const exportData = async () => {
    try {
      const data = {
        profile: profileData,
        settings: {
          theme: themeSettings,
          notifications: notificationSettings,
          trading: tradingSettings,
          security: securitySettings
        },
        exportDate: new Date().toISOString()
      };
      
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `settings-backup-${new Date().toISOString().split('T')[0]}.json`;
      a.click();
      URL.revokeObjectURL(url);
      
      toast.success(t('settings.exportSuccess'));
    } catch (error) {
      toast.error(t('settings.exportError'));
    }
  };

  const renderProfileTab = () => (
    <div className="space-y-6">
      <form onSubmit={handleProfileUpdate} className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              {t('settings.firstName')}
            </label>
            <input
              type="text"
              value={profileData.firstName}
              onChange={(e) => setProfileData(prev => ({ ...prev, firstName: e.target.value }))}
              className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              {t('settings.lastName')}
            </label>
            <input
              type="text"
              value={profileData.lastName}
              onChange={(e) => setProfileData(prev => ({ ...prev, lastName: e.target.value }))}
              className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
            />
          </div>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            {t('settings.email')}
          </label>
          <input
            type="email"
            value={profileData.email}
            onChange={(e) => setProfileData(prev => ({ ...prev, email: e.target.value }))}
            className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            {t('settings.phone')}
          </label>
          <input
            type="tel"
            value={profileData.phone}
            onChange={(e) => setProfileData(prev => ({ ...prev, phone: e.target.value }))}
            className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            {t('settings.bio')}
          </label>
          <textarea
            value={profileData.bio}
            onChange={(e) => setProfileData(prev => ({ ...prev, bio: e.target.value }))}
            rows={4}
            className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
            placeholder={t('settings.bioPlaceholder')}
          />
        </div>
        
        <button
          type="submit"
          disabled={loading}
          className="w-full md:w-auto px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
        >
          {loading ? t('common.saving') : t('common.save')}
        </button>
      </form>
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
            value={themeSettings.theme}
            onChange={(e) => handleThemeChange('theme', e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
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
            value={themeSettings.language}
            onChange={(e) => handleThemeChange('language', e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
          >
            <option value="en">English</option>
            <option value="hi">हिंदी</option>
          </select>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            {t('settings.fontSize')}
          </label>
          <select
            value={themeSettings.fontSize}
            onChange={(e) => handleThemeChange('fontSize', e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
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
            value={themeSettings.chartTheme}
            onChange={(e) => handleThemeChange('chartTheme', e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
          >
            <option value="light">{t('settings.lightCharts')}</option>
            <option value="dark">{t('settings.darkCharts')}</option>
            <option value="colorful">{t('settings.colorfulCharts')}</option>
          </select>
        </div>
      </div>
      
      <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
        <div>
          <h4 className="font-medium text-gray-900 dark:text-white">{t('settings.compactMode')}</h4>
          <p className="text-sm text-gray-500 dark:text-gray-400">{t('settings.compactModeDesc')}</p>
        </div>
        <button
          onClick={() => handleThemeChange('compactMode', !themeSettings.compactMode)}
          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
            themeSettings.compactMode ? 'bg-blue-600' : 'bg-gray-200 dark:bg-gray-700'
          }`}
        >
          <span
            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
              themeSettings.compactMode ? 'translate-x-6' : 'translate-x-1'
            }`}
          />
        </button>
      </div>
    </div>
  );

  const renderNotificationsTab = () => (
    <div className="space-y-6">
      {Object.entries(notificationSettings).map(([key, value]) => (
        <div key={key} className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <div>
            <h4 className="font-medium text-gray-900 dark:text-white">
              {t(`settings.${key}`)}
            </h4>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              {t(`settings.${key}Desc`)}
            </p>
          </div>
          <button
            onClick={() => handleNotificationChange(key, !value)}
            className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
              value ? 'bg-blue-600' : 'bg-gray-200 dark:bg-gray-700'
            }`}
          >
            <span
              className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                value ? 'translate-x-6' : 'translate-x-1'
              }`}
            />
          </button>
        </div>
      ))}
    </div>
  );

  const renderTradingTab = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            {t('settings.defaultOrderType')}
          </label>
          <select
            value={tradingSettings.defaultOrderType}
            onChange={(e) => handleTradingChange('defaultOrderType', e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
          >
            <option value="market">{t('settings.marketOrder')}</option>
            <option value="limit">{t('settings.limitOrder')}</option>
            <option value="stop">{t('settings.stopOrder')}</option>
          </select>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            {t('settings.riskTolerance')}
          </label>
          <select
            value={tradingSettings.riskTolerance}
            onChange={(e) => handleTradingChange('riskTolerance', e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
          >
            <option value="low">{t('settings.lowRisk')}</option>
            <option value="medium">{t('settings.mediumRisk')}</option>
            <option value="high">{t('settings.highRisk')}</option>
          </select>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            {t('settings.stopLossPercentage')}
          </label>
          <input
            type="number"
            min="1"
            max="50"
            value={tradingSettings.stopLossPercentage}
            onChange={(e) => handleTradingChange('stopLossPercentage', parseInt(e.target.value))}
            className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            {t('settings.takeProfitPercentage')}
          </label>
          <input
            type="number"
            min="5"
            max="100"
            value={tradingSettings.takeProfitPercentage}
            onChange={(e) => handleTradingChange('takeProfitPercentage', parseInt(e.target.value))}
            className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
          />
        </div>
      </div>
      
      <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
        <div>
          <h4 className="font-medium text-gray-900 dark:text-white">{t('settings.enablePaperTrading')}</h4>
          <p className="text-sm text-gray-500 dark:text-gray-400">{t('settings.paperTradingDesc')}</p>
        </div>
        <button
          onClick={() => handleTradingChange('enablePaperTrading', !tradingSettings.enablePaperTrading)}
          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
            tradingSettings.enablePaperTrading ? 'bg-blue-600' : 'bg-gray-200 dark:bg-gray-700'
          }`}
        >
          <span
            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
              tradingSettings.enablePaperTrading ? 'translate-x-6' : 'translate-x-1'
            }`}
          />
        </button>
      </div>
    </div>
  );

  const renderSecurityTab = () => (
    <div className="space-y-6">
      {Object.entries(securitySettings).map(([key, value]) => (
        <div key={key} className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <div>
            <h4 className="font-medium text-gray-900 dark:text-white">
              {t(`settings.${key}`)}
            </h4>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              {t(`settings.${key}Desc`)}
            </p>
          </div>
          <button
            onClick={() => handleSecurityChange(key, !value)}
            className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
              value ? 'bg-blue-600' : 'bg-gray-200 dark:bg-gray-700'
            }`}
          >
            <span
              className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                value ? 'translate-x-6' : 'translate-x-1'
              }`}
            />
          </button>
        </div>
      ))}
    </div>
  );

  const renderAdvancedTab = () => (
    <div className="space-y-6">
      <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
        <h3 className="text-lg font-medium text-yellow-800 dark:text-yellow-200 mb-2">
          {t('settings.dangerZone')}
        </h3>
        <p className="text-sm text-yellow-700 dark:text-yellow-300 mb-4">
          {t('settings.dangerZoneDesc')}
        </p>
        
        <div className="space-y-4">
          <button
            onClick={exportData}
            className="w-full md:w-auto px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            {t('settings.exportData')}
          </button>
          
          <button
            onClick={resetToDefaults}
            className="w-full md:w-auto px-4 py-2 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700 transition-colors ml-0 md:ml-4"
          >
            {t('settings.resetToDefaults')}
          </button>
        </div>
      </div>
    </div>
  );

  const renderTabContent = () => {
    switch (activeTab) {
      case 'profile':
        return renderProfileTab();
      case 'appearance':
        return renderAppearanceTab();
      case 'notifications':
        return renderNotificationsTab();
      case 'trading':
        return renderTradingTab();
      case 'security':
        return renderSecurityTab();
      case 'advanced':
        return renderAdvancedTab();
      default:
        return renderProfileTab();
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            {t('settings.title')}
          </h1>
          <p className="mt-2 text-gray-600 dark:text-gray-400">
            {t('settings.subtitle')}
          </p>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg overflow-hidden">
          <div className="flex flex-col lg:flex-row">
            {/* Sidebar */}
            <div className="lg:w-1/4 bg-gray-50 dark:bg-gray-700">
              <nav className="space-y-1 p-4">
                {tabs.map((tab) => {
                  const Icon = tab.icon;
                  return (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id)}
                      className={`w-full flex items-center px-4 py-3 text-sm font-medium rounded-lg transition-colors ${
                        activeTab === tab.id
                          ? 'bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-200'
                          : 'text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-600'
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
    </div>
  );
};

export default SettingsPage;