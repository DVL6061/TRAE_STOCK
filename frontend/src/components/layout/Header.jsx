import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { 
  Bars3Icon, 
  BellIcon, 
  MagnifyingGlassIcon,
  UserCircleIcon,
  SunIcon,
  MoonIcon,
  ComputerDesktopIcon,
  Cog6ToothIcon,
  ArrowRightOnRectangleIcon
} from '@heroicons/react/24/outline';
import LanguageSwitcher from '../common/LanguageSwitcher';

const Header = ({ onMenuClick, sidebarOpen }) => {
  const { t } = useTranslation();
  const [searchQuery, setSearchQuery] = useState('');
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [showNotifications, setShowNotifications] = useState(false);
  const [theme, setTheme] = useState('light'); // This would come from a theme context

  // Mock user data
  const user = {
    name: 'John Doe',
    email: 'john.doe@example.com',
    avatar: null
  };

  // Mock notifications
  const notifications = [
    {
      id: 1,
      type: 'price_alert',
      title: 'RELIANCE crossed ₹2,500',
      message: 'Your price alert for RELIANCE has been triggered',
      time: '2 minutes ago',
      read: false
    },
    {
      id: 2,
      type: 'prediction',
      title: 'New AI Prediction Available',
      message: 'TCS prediction updated with 85% confidence',
      time: '15 minutes ago',
      read: false
    },
    {
      id: 3,
      type: 'news',
      title: 'Market News Update',
      message: 'RBI announces new monetary policy',
      time: '1 hour ago',
      read: true
    }
  ];

  const unreadCount = notifications.filter(n => !n.read).length;

  const handleSearch = (e) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      // Handle search functionality
      console.log('Searching for:', searchQuery);
      // This would typically navigate to search results or trigger a search
    }
  };

  const handleThemeChange = (newTheme) => {
    setTheme(newTheme);
    // This would update the theme context
    console.log('Theme changed to:', newTheme);
  };

  const getNotificationIcon = (type) => {
    switch (type) {
      case 'price_alert':
        return '📈';
      case 'prediction':
        return '🤖';
      case 'news':
        return '📰';
      default:
        return '🔔';
    }
  };

  return (
    <header className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
      <div className="px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          {/* Left side - Menu button and Logo */}
          <div className="flex items-center gap-4">
            <button
              type="button"
              onClick={onMenuClick}
              className="rounded-md p-2 text-gray-400 hover:bg-gray-100 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:hover:bg-gray-700 dark:hover:text-gray-300"
              aria-label="Toggle sidebar"
            >
              <Bars3Icon className="h-6 w-6" />
            </button>
            
            {/* Logo and title */}
            <div className="flex items-center gap-3">
              <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                <span className="text-white font-bold text-sm">AI</span>
              </div>
              <div className="hidden sm:block">
                <h1 className="text-xl font-bold text-gray-900 dark:text-white">
                  {t('app.title')}
                </h1>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  {t('app.subtitle')}
                </p>
              </div>
            </div>
          </div>

          {/* Center - Search */}
          <div className="flex-1 max-w-lg mx-4">
            <form onSubmit={handleSearch} className="relative">
              <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3">
                <MagnifyingGlassIcon className="h-5 w-5 text-gray-400" />
              </div>
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="block w-full rounded-lg border border-gray-300 bg-white py-2 pl-10 pr-3 text-sm placeholder-gray-500 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white dark:placeholder-gray-400"
                placeholder={t('predictions.searchStock')}
              />
            </form>
          </div>

          {/* Right side - Actions */}
          <div className="flex items-center gap-2">
            {/* Language Switcher */}
            <LanguageSwitcher variant="compact" showLabel={false} />

            {/* Theme Switcher */}
            <div className="relative">
              <button
                type="button"
                onClick={() => {
                  const themes = ['light', 'dark', 'auto'];
                  const currentIndex = themes.indexOf(theme);
                  const nextTheme = themes[(currentIndex + 1) % themes.length];
                  handleThemeChange(nextTheme);
                }}
                className="rounded-lg p-2 text-gray-400 hover:bg-gray-100 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:hover:bg-gray-700 dark:hover:text-gray-300"
                aria-label={t('settings.theme')}
              >
                {theme === 'light' && <SunIcon className="h-5 w-5" />}
                {theme === 'dark' && <MoonIcon className="h-5 w-5" />}
                {theme === 'auto' && <ComputerDesktopIcon className="h-5 w-5" />}
              </button>
            </div>

            {/* Notifications */}
            <div className="relative">
              <button
                type="button"
                onClick={() => setShowNotifications(!showNotifications)}
                className="relative rounded-lg p-2 text-gray-400 hover:bg-gray-100 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:hover:bg-gray-700 dark:hover:text-gray-300"
                aria-label={t('settings.notifications')}
              >
                <BellIcon className="h-5 w-5" />
                {unreadCount > 0 && (
                  <span className="absolute -top-1 -right-1 h-4 w-4 rounded-full bg-red-500 text-xs text-white flex items-center justify-center">
                    {unreadCount > 9 ? '9+' : unreadCount}
                  </span>
                )}
              </button>

              {/* Notifications Dropdown */}
              {showNotifications && (
                <>
                  <div 
                    className="fixed inset-0 z-10" 
                    onClick={() => setShowNotifications(false)}
                  />
                  <div className="absolute right-0 z-20 mt-2 w-80 rounded-lg bg-white shadow-lg ring-1 ring-black ring-opacity-5 dark:bg-gray-800 dark:ring-gray-600">
                    <div className="p-4 border-b border-gray-200 dark:border-gray-700">
                      <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                        {t('settings.notifications')}
                      </h3>
                    </div>
                    <div className="max-h-96 overflow-y-auto">
                      {notifications.length > 0 ? (
                        notifications.map((notification) => (
                          <div
                            key={notification.id}
                            className={`p-4 border-b border-gray-100 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer ${
                              !notification.read ? 'bg-blue-50 dark:bg-blue-900/10' : ''
                            }`}
                          >
                            <div className="flex items-start gap-3">
                              <span className="text-lg">
                                {getNotificationIcon(notification.type)}
                              </span>
                              <div className="flex-1 min-w-0">
                                <p className="text-sm font-medium text-gray-900 dark:text-white">
                                  {notification.title}
                                </p>
                                <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                                  {notification.message}
                                </p>
                                <p className="text-xs text-gray-400 dark:text-gray-500 mt-2">
                                  {notification.time}
                                </p>
                              </div>
                              {!notification.read && (
                                <div className="h-2 w-2 rounded-full bg-blue-600" />
                              )}
                            </div>
                          </div>
                        ))
                      ) : (
                        <div className="p-8 text-center text-gray-500 dark:text-gray-400">
                          <BellIcon className="h-12 w-12 mx-auto mb-4 opacity-50" />
                          <p>{t('errors.noNews')}</p>
                        </div>
                      )}
                    </div>
                  </div>
                </>
              )}
            </div>

            {/* User Menu */}
            <div className="relative">
              <button
                type="button"
                onClick={() => setShowUserMenu(!showUserMenu)}
                className="flex items-center gap-2 rounded-lg p-2 text-gray-400 hover:bg-gray-100 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:hover:bg-gray-700 dark:hover:text-gray-300"
                aria-label="User menu"
              >
                {user.avatar ? (
                  <img 
                    src={user.avatar} 
                    alt={user.name}
                    className="h-6 w-6 rounded-full"
                  />
                ) : (
                  <UserCircleIcon className="h-6 w-6" />
                )}
                <span className="hidden sm:block text-sm font-medium text-gray-700 dark:text-gray-200">
                  {user.name}
                </span>
              </button>

              {/* User Dropdown */}
              {showUserMenu && (
                <>
                  <div 
                    className="fixed inset-0 z-10" 
                    onClick={() => setShowUserMenu(false)}
                  />
                  <div className="absolute right-0 z-20 mt-2 w-56 rounded-lg bg-white shadow-lg ring-1 ring-black ring-opacity-5 dark:bg-gray-800 dark:ring-gray-600">
                    <div className="p-4 border-b border-gray-200 dark:border-gray-700">
                      <p className="text-sm font-medium text-gray-900 dark:text-white">
                        {user.name}
                      </p>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        {user.email}
                      </p>
                    </div>
                    <div className="py-1">
                      <button
                        className="flex w-full items-center gap-3 px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 dark:text-gray-200 dark:hover:bg-gray-700"
                        onClick={() => {
                          setShowUserMenu(false);
                          // Navigate to profile
                        }}
                      >
                        <UserCircleIcon className="h-4 w-4" />
                        {t('navigation.profile')}
                      </button>
                      <button
                        className="flex w-full items-center gap-3 px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 dark:text-gray-200 dark:hover:bg-gray-700"
                        onClick={() => {
                          setShowUserMenu(false);
                          // Navigate to settings
                        }}
                      >
                        <Cog6ToothIcon className="h-4 w-4" />
                        {t('navigation.settings')}
                      </button>
                      <hr className="my-1 border-gray-200 dark:border-gray-700" />
                      <button
                        className="flex w-full items-center gap-3 px-4 py-2 text-sm text-red-600 hover:bg-red-50 dark:text-red-400 dark:hover:bg-red-900/20"
                        onClick={() => {
                          setShowUserMenu(false);
                          // Handle logout
                        }}
                      >
                        <ArrowRightOnRectangleIcon className="h-4 w-4" />
                        {t('navigation.logout')}
                      </button>
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Market Status Bar */}
      <div className="bg-gray-50 dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700">
        <div className="px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between py-2 text-sm">
            <div className="flex items-center gap-6">
              <div className="flex items-center gap-2">
                <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
                <span className="text-gray-600 dark:text-gray-400">
                  {t('dashboard.marketOpen')}
                </span>
              </div>
              <div className="text-gray-600 dark:text-gray-400">
                NIFTY 50: <span className="text-green-600 font-medium">19,845.65 (+0.85%)</span>
              </div>
              <div className="text-gray-600 dark:text-gray-400">
                SENSEX: <span className="text-green-600 font-medium">66,598.91 (+0.73%)</span>
              </div>
            </div>
            <div className="text-gray-500 dark:text-gray-400">
              {t('predictions.lastUpdated')}: {new Date().toLocaleTimeString()}
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;