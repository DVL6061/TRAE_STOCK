import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useTheme } from '../../contexts/ThemeContext';

// Icons
import { 
  FiMenu, 
  FiSun, 
  FiMoon, 
  FiSearch, 
  FiBell, 
  FiUser,
  FiSettings,
  FiLogOut,
  FiGlobe
} from 'react-icons/fi';

const Header = ({ toggleSidebar }) => {
  const { t, i18n } = useTranslation();
  const { theme, toggleTheme } = useTheme();
  const [searchQuery, setSearchQuery] = useState('');
  const [showDropdown, setShowDropdown] = useState(false);
  const [showLanguageDropdown, setShowLanguageDropdown] = useState(false);
  const [showNotifications, setShowNotifications] = useState(false);

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    // Handle search logic here
    console.log('Searching for:', searchQuery);
    setSearchQuery('');
  };

  const toggleDropdown = () => {
    setShowDropdown(!showDropdown);
    // Close other dropdowns
    setShowLanguageDropdown(false);
    setShowNotifications(false);
  };

  const toggleLanguageDropdown = () => {
    setShowLanguageDropdown(!showLanguageDropdown);
    // Close other dropdowns
    setShowDropdown(false);
    setShowNotifications(false);
  };

  const toggleNotifications = () => {
    setShowNotifications(!showNotifications);
    // Close other dropdowns
    setShowDropdown(false);
    setShowLanguageDropdown(false);
  };

  const changeLanguage = (lng) => {
    i18n.changeLanguage(lng);
    setShowLanguageDropdown(false);
  };

  return (
    <header className="bg-white dark:bg-neutral-800 shadow-sm sticky top-0 z-10">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Left section - Logo and menu toggle */}
          <div className="flex items-center">
            <button
              onClick={toggleSidebar}
              className="p-2 rounded-md text-neutral-600 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-700 focus:outline-none"
              aria-label="Toggle sidebar"
            >
              <FiMenu className="h-6 w-6" />
            </button>
            
            <Link to="/" className="ml-2 flex items-center">
              <span className="text-primary-600 dark:text-primary-400 font-bold text-xl">
                {t('app_name')}
              </span>
            </Link>
          </div>

          {/* Middle section - Search */}
          <div className="hidden md:block flex-1 max-w-md mx-4">
            <form onSubmit={handleSearchSubmit} className="relative">
              <input
                type="text"
                placeholder={t('search')}
                className="w-full pl-10 pr-4 py-2 rounded-lg border border-neutral-300 dark:border-neutral-600 bg-neutral-50 dark:bg-neutral-700 text-neutral-900 dark:text-neutral-100 focus:outline-none focus:ring-2 focus:ring-primary-500"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
              <div className="absolute left-3 top-2.5 text-neutral-500 dark:text-neutral-400">
                <FiSearch className="h-5 w-5" />
              </div>
            </form>
          </div>

          {/* Right section - Theme toggle, notifications, language, profile */}
          <div className="flex items-center space-x-3">
            {/* Theme toggle */}
            <button
              onClick={toggleTheme}
              className="p-2 rounded-full text-neutral-600 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-700 focus:outline-none"
              aria-label="Toggle theme"
            >
              {theme === 'dark' ? (
                <FiSun className="h-5 w-5" />
              ) : (
                <FiMoon className="h-5 w-5" />
              )}
            </button>

            {/* Language selector */}
            <div className="relative">
              <button
                onClick={toggleLanguageDropdown}
                className="p-2 rounded-full text-neutral-600 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-700 focus:outline-none"
                aria-label="Change language"
              >
                <FiGlobe className="h-5 w-5" />
              </button>

              {showLanguageDropdown && (
                <div className="absolute right-0 mt-2 w-48 bg-white dark:bg-neutral-800 rounded-md shadow-lg py-1 z-10 border border-neutral-200 dark:border-neutral-700">
                  <button
                    onClick={() => changeLanguage('en')}
                    className={`block px-4 py-2 text-sm w-full text-left ${i18n.language === 'en' ? 'bg-neutral-100 dark:bg-neutral-700 text-primary-600 dark:text-primary-400' : 'text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-700'}`}
                  >
                    {t('english')}
                  </button>
                  <button
                    onClick={() => changeLanguage('hi')}
                    className={`block px-4 py-2 text-sm w-full text-left ${i18n.language === 'hi' ? 'bg-neutral-100 dark:bg-neutral-700 text-primary-600 dark:text-primary-400' : 'text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-700'}`}
                  >
                    {t('hindi')}
                  </button>
                </div>
              )}
            </div>

            {/* Notifications */}
            <div className="relative">
              <button
                onClick={toggleNotifications}
                className="p-2 rounded-full text-neutral-600 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-700 focus:outline-none"
                aria-label="View notifications"
              >
                <div className="relative">
                  <FiBell className="h-5 w-5" />
                  <span className="absolute -top-1 -right-1 bg-danger-500 text-white text-xs rounded-full h-4 w-4 flex items-center justify-center">
                    3
                  </span>
                </div>
              </button>

              {showNotifications && (
                <div className="absolute right-0 mt-2 w-80 bg-white dark:bg-neutral-800 rounded-md shadow-lg py-1 z-10 border border-neutral-200 dark:border-neutral-700">
                  <div className="px-4 py-2 border-b border-neutral-200 dark:border-neutral-700">
                    <h3 className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
                      {t('notifications')}
                    </h3>
                  </div>
                  <div className="max-h-60 overflow-y-auto">
                    {/* Sample notifications */}
                    <div className="px-4 py-3 hover:bg-neutral-100 dark:hover:bg-neutral-700 border-b border-neutral-200 dark:border-neutral-700">
                      <p className="text-sm text-neutral-900 dark:text-neutral-100 font-medium">
                        {t('prediction_results')}
                      </p>
                      <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-1">
                        TATAMOTORS.NS: {t('strong_buy')} - ₹950.25
                      </p>
                      <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-1">
                        10 {t('minutes')} {t('ago')}
                      </p>
                    </div>
                    <div className="px-4 py-3 hover:bg-neutral-100 dark:hover:bg-neutral-700 border-b border-neutral-200 dark:border-neutral-700">
                      <p className="text-sm text-neutral-900 dark:text-neutral-100 font-medium">
                        {t('news_alert')}
                      </p>
                      <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-1">
                        {t('high_impact')} {t('news')} {t('for')} TATAMOTORS.NS
                      </p>
                      <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-1">
                        30 {t('minutes')} {t('ago')}
                      </p>
                    </div>
                    <div className="px-4 py-3 hover:bg-neutral-100 dark:hover:bg-neutral-700">
                      <p className="text-sm text-neutral-900 dark:text-neutral-100 font-medium">
                        {t('market_update')}
                      </p>
                      <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-1">
                        NIFTY 50 {t('up')} 1.2% {t('today')}
                      </p>
                      <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-1">
                        1 {t('hour')} {t('ago')}
                      </p>
                    </div>
                  </div>
                  <div className="px-4 py-2 border-t border-neutral-200 dark:border-neutral-700">
                    <Link
                      to="/notifications"
                      className="text-sm text-primary-600 dark:text-primary-400 hover:text-primary-800 dark:hover:text-primary-300"
                      onClick={() => setShowNotifications(false)}
                    >
                      {t('view_all_notifications')}
                    </Link>
                  </div>
                </div>
              )}
            </div>

            {/* Profile dropdown */}
            <div className="relative ml-3">
              <button
                onClick={toggleDropdown}
                className="flex items-center text-sm rounded-full focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
                id="user-menu"
                aria-expanded="false"
                aria-haspopup="true"
              >
                <span className="sr-only">Open user menu</span>
                <div className="h-8 w-8 rounded-full bg-primary-500 flex items-center justify-center text-white">
                  <FiUser className="h-5 w-5" />
                </div>
              </button>

              {showDropdown && (
                <div
                  className="origin-top-right absolute right-0 mt-2 w-48 rounded-md shadow-lg py-1 bg-white dark:bg-neutral-800 ring-1 ring-black ring-opacity-5 focus:outline-none z-10 border border-neutral-200 dark:border-neutral-700"
                  role="menu"
                  aria-orientation="vertical"
                  aria-labelledby="user-menu"
                >
                  <Link
                    to="/profile"
                    className="flex items-center px-4 py-2 text-sm text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-700"
                    role="menuitem"
                    onClick={() => setShowDropdown(false)}
                  >
                    <FiUser className="mr-3 h-4 w-4" />
                    {t('profile')}
                  </Link>
                  <Link
                    to="/settings"
                    className="flex items-center px-4 py-2 text-sm text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-700"
                    role="menuitem"
                    onClick={() => setShowDropdown(false)}
                  >
                    <FiSettings className="mr-3 h-4 w-4" />
                    {t('settings')}
                  </Link>
                  <button
                    className="flex w-full items-center px-4 py-2 text-sm text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-700"
                    role="menuitem"
                    onClick={() => {
                      setShowDropdown(false);
                      // Handle logout logic here
                    }}
                  >
                    <FiLogOut className="mr-3 h-4 w-4" />
                    {t('logout')}
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
        
        {/* Mobile search - visible only on small screens */}
        <div className="md:hidden pb-3">
          <form onSubmit={handleSearchSubmit} className="relative">
            <input
              type="text"
              placeholder={t('search')}
              className="w-full pl-10 pr-4 py-2 rounded-lg border border-neutral-300 dark:border-neutral-600 bg-neutral-50 dark:bg-neutral-700 text-neutral-900 dark:text-neutral-100 focus:outline-none focus:ring-2 focus:ring-primary-500"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            <div className="absolute left-3 top-2.5 text-neutral-500 dark:text-neutral-400">
              <FiSearch className="h-5 w-5" />
            </div>
          </form>
        </div>
      </div>
    </header>
  );
};

export default Header;