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
    <header className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-3 sm:px-4 lg:px-8">
        <div className="flex h-14 sm:h-16 items-center justify-between">
          {/* Left side - Menu button and Logo */}
          <div className="flex items-center gap-2 sm:gap-4">
            <button
              type="button"
              onClick={onMenuClick}
              className="lg:hidden rounded-md p-1.5 sm:p-2 text-gray-400 hover:bg-gray-100 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:hover:bg-gray-700 dark:hover:text-gray-300"
              aria-label="Toggle sidebar"
            >
              <Bars3Icon className="h-5 w-5 sm:h-6 sm:w-6" />
            </button>
            
            {/* Logo and title */}
            <div className="flex items-center gap-1.5 sm:gap-2">
              <div className="h-6 w-6 sm:h-7 sm:w-7 lg:h-8 lg:w-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                <span className="text-white font-bold text-xs sm:text-sm">AI</span>
              </div>
              <div className="hidden xs:block">
                <h1 className="text-sm sm:text-lg lg:text-xl font-bold text-gray-900 dark:text-white">
                  {t('app.title')}
                </h1>
                <p className="text-xs text-gray-500 dark:text-gray-400 hidden sm:block">
                  {t('app.subtitle')}
                </p>
              </div>
            </div>
          </div>

          {/* Center - Search */}
          <div className="flex-1 max-w-xs sm:max-w-lg mx-2 sm:mx-4 lg:mx-8">
            <form onSubmit={handleSearch} className="relative">
              <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-2 sm:pl-3">
                <MagnifyingGlassIcon className="h-4 w-4 sm:h-5 sm:w-5 text-gray-400" />
              </div>
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="block w-full rounded-lg border border-gray-300 bg-white py-1.5 sm:py-2 pl-7 sm:pl-10 pr-2 sm:pr-3 text-xs sm:text-sm placeholder-gray-500 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white dark:placeholder-gray-400"
                placeholder={t('predictions.searchStock')}
              />
            </form>
          </div>

          {/* Right side - Actions */}
          <div className="flex items-center gap-1 sm:gap-2 lg:gap-4">
            {/* Language Switcher */}
            <div className="hidden md:block">
              <LanguageSwitcher variant="compact" showLabel={false} />
            </div>

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
                className="rounded-lg p-1.5 sm:p-2 text-gray-400 hover:bg-gray-100 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:hover:bg-gray-700 dark:hover:text-gray-300"
                aria-label={t('settings.theme')}
                title={`Current theme: ${theme}`}
              >
                {theme === 'light' && <SunIcon className="h-4 w-4 sm:h-5 sm:w-5" />}
                {theme === 'dark' && <MoonIcon className="h-4 w-4 sm:h-5 sm:w-5" />}
                {theme === 'auto' && <ComputerDesktopIcon className="h-4 w-4 sm:h-5 sm:w-5" />}
              </button>
            </div>

            {/* Notifications */}
            <div className="relative">
              <button
                type="button"
                onClick={() => setShowNotifications(!showNotifications)}
                className="relative rounded-lg p-1.5 sm:p-2 text-gray-400 hover:bg-gray-100 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:hover:bg-gray-700 dark:hover:text-gray-300"
                aria-label={t('settings.notifications')}
              >
                <BellIcon className="h-4 w-4 sm:h-5 sm:w-5" />
                {unreadCount > 0 && (
                  <span className="absolute -top-0.5 -right-0.5 sm:-top-1 sm:-right-1 h-3 w-3 sm:h-4 sm:w-4 rounded-full bg-red-500 text-xs text-white flex items-center justify-center">
                    <span className="text-xs sm:text-xs">{unreadCount > 9 ? '9+' : unreadCount}</span>
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
                  <div className="absolute right-0 z-20 mt-2 w-72 sm:w-80 rounded-lg bg-white shadow-lg ring-1 ring-black ring-opacity-5 dark:bg-gray-800 dark:ring-gray-600">
                    <div className="p-3 sm:p-4 border-b border-gray-200 dark:border-gray-700">
                      <h3 className="text-base sm:text-lg font-semibold text-gray-900 dark:text-white">
                        {t('settings.notifications')} ({unreadCount})
                      </h3>
                    </div>
                    <div className="max-h-80 sm:max-h-96 overflow-y-auto">
                      {notifications.length > 0 ? (
                        notifications.map((notification) => (
                          <div
                            key={notification.id}
                            className={`p-3 sm:p-4 border-b border-gray-100 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer ${
                              !notification.read ? 'bg-blue-50 dark:bg-blue-900/10' : ''
                            }`}
                          >
                            <div className="flex items-start gap-2 sm:gap-3">
                              <span className="text-base sm:text-lg">
                                {getNotificationIcon(notification.type)}
                              </span>
                              <div className="flex-1 min-w-0">
                                <p className="text-xs sm:text-sm font-medium text-gray-900 dark:text-white">
                                  {notification.title}
                                </p>
                                <p className="text-xs sm:text-sm text-gray-500 dark:text-gray-400 mt-1">
                                  {notification.message}
                                </p>
                                <p className="text-xs text-gray-400 dark:text-gray-500 mt-1 sm:mt-2">
                                  {notification.time}
                                </p>
                              </div>
                              {!notification.read && (
                                <div className="h-1.5 w-1.5 sm:h-2 sm:w-2 rounded-full bg-blue-600" />
                              )}
                            </div>
                          </div>
                        ))
                      ) : (
                        <div className="p-6 sm:p-8 text-center text-gray-500 dark:text-gray-400">
                          <BellIcon className="h-8 w-8 sm:h-12 sm:w-12 mx-auto mb-3 sm:mb-4 opacity-50" />
                          <p className="text-sm">{t('errors.noNews')}</p>
                        </div>
                      )}
                    </div>
                    {notifications.length > 0 && (
                      <div className="px-3 sm:px-4 py-2 border-t border-gray-200 dark:border-gray-700">
                        <button
                          onClick={() => {
                            // Mark all as read functionality
                            setShowNotifications(false);
                          }}
                          className="text-xs sm:text-sm text-blue-600 hover:text-blue-500 dark:text-blue-400 dark:hover:text-blue-300"
                        >
                          {t('notifications.markAllRead')}
                        </button>
                      </div>
                    )}
                  </div>
                </>
              )}
            </div>

            {/* User Menu */}
            <div className="relative">
              <button
                type="button"
                onClick={() => setShowUserMenu(!showUserMenu)}
                className="flex items-center gap-1 sm:gap-2 rounded-lg p-1.5 sm:p-2 text-gray-400 hover:bg-gray-100 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:hover:bg-gray-700 dark:hover:text-gray-300"
                aria-label="User menu"
              >
                {user.avatar ? (
                  <img 
                    src={user.avatar} 
                    alt={user.name}
                    className="h-5 w-5 sm:h-6 sm:w-6 rounded-full"
                  />
                ) : (
                  <UserCircleIcon className="h-5 w-5 sm:h-6 sm:w-6" />
                )}
                <span className="hidden md:block text-sm font-medium text-gray-700 dark:text-gray-200">
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
                  <div className="absolute right-0 z-20 mt-2 w-48 sm:w-56 rounded-lg bg-white shadow-lg ring-1 ring-black ring-opacity-5 dark:bg-gray-800 dark:ring-gray-600">
                    <div className="p-3 sm:p-4 border-b border-gray-200 dark:border-gray-700">
                      <p className="text-xs sm:text-sm font-medium text-gray-900 dark:text-white">
                        {user.name}
                      </p>
                      <p className="text-xs sm:text-sm text-gray-500 dark:text-gray-400">
                        {user.email}
                      </p>
                    </div>
                    <div className="py-1">
                      <button
                        className="flex w-full items-center gap-2 sm:gap-3 px-3 sm:px-4 py-2 text-xs sm:text-sm text-gray-700 hover:bg-gray-100 dark:text-gray-200 dark:hover:bg-gray-700"
                        onClick={() => {
                          setShowUserMenu(false);
                          // Navigate to profile
                        }}
                      >
                        <UserCircleIcon className="h-3.5 w-3.5 sm:h-4 sm:w-4" />
                        {t('navigation.profile')}
                      </button>
                      <button
                        className="flex w-full items-center gap-2 sm:gap-3 px-3 sm:px-4 py-2 text-xs sm:text-sm text-gray-700 hover:bg-gray-100 dark:text-gray-200 dark:hover:bg-gray-700"
                        onClick={() => {
                          setShowUserMenu(false);
                          // Navigate to settings
                        }}
                      >
                        <Cog6ToothIcon className="h-3.5 w-3.5 sm:h-4 sm:w-4" />
                        {t('navigation.settings')}
                      </button>
                      <hr className="my-1 border-gray-200 dark:border-gray-700" />
                      <button
                        className="flex w-full items-center gap-2 sm:gap-3 px-3 sm:px-4 py-2 text-xs sm:text-sm text-red-600 hover:bg-red-50 dark:text-red-400 dark:hover:bg-red-900/20"
                        onClick={() => {
                          setShowUserMenu(false);
                          // Handle logout
                        }}
                      >
                        <ArrowRightOnRectangleIcon className="h-3.5 w-3.5 sm:h-4 sm:w-4" />
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
        <div className="px-3 sm:px-4 lg:px-8">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between py-2 gap-2 sm:gap-0 text-xs sm:text-sm">
            <div className="flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-4 lg:gap-6">
              <div className="flex items-center gap-1.5 sm:gap-2">
                <div className="h-1.5 w-1.5 sm:h-2 sm:w-2 rounded-full bg-green-500 animate-pulse" />
                <span className="text-gray-600 dark:text-gray-400 text-xs sm:text-sm">
                  {t('dashboard.marketOpen')}
                </span>
              </div>
              <div className="flex gap-3 sm:gap-4 lg:gap-6">
                <div className="text-gray-600 dark:text-gray-400 text-xs sm:text-sm">
                  <span className="hidden sm:inline">NIFTY 50: </span>
                  <span className="sm:hidden">NIFTY: </span>
                  <span className="text-green-600 font-medium">19,845.65 (+0.85%)</span>
                </div>
                <div className="text-gray-600 dark:text-gray-400 text-xs sm:text-sm">
                  SENSEX: <span className="text-green-600 font-medium">66,598.91 (+0.73%)</span>
                </div>
              </div>
            </div>
            <div className="text-gray-500 dark:text-gray-400 text-xs sm:text-sm">
              <span className="hidden sm:inline">{t('predictions.lastUpdated')}: </span>
              <span className="sm:hidden">Updated: </span>
              {new Date().toLocaleTimeString()}
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;