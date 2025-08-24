import React from 'react';
import { NavLink } from 'react-router-dom';
import { useTranslation } from 'react-i18next';

// Icons
import {
  FiHome,
  FiTrendingUp,
  FiBarChart2,
  FiFileText,
  FiSettings,
  FiX,
  FiClock,
  FiStar,
  FiActivity,
  FiInfo
} from 'react-icons/fi';

const Sidebar = ({ isOpen, toggleSidebar }) => {
  const { t } = useTranslation();

  // Navigation items
  const navItems = [
    {
      name: t('dashboard'),
      path: '/dashboard',
      icon: <FiHome className="h-5 w-5" />
    },
    {
      name: t('stocks'),
      path: '/stock/TATAMOTORS.NS',
      icon: <FiTrendingUp className="h-5 w-5" />
    },
    {
      name: t('predictions'),
      path: '/predictions',
      icon: <FiBarChart2 className="h-5 w-5" />
    },
    {
      name: t('news'),
      path: '/news',
      icon: <FiFileText className="h-5 w-5" />
    },
    {
      name: t('settings'),
      path: '/settings',
      icon: <FiSettings className="h-5 w-5" />
    }
  ];

  // Prediction timeframes
  const timeframes = [
    {
      name: t('intraday'),
      path: '/predictions?timeframe=intraday',
      icon: <FiClock className="h-4 w-4" />
    },
    {
      name: t('short_term'),
      path: '/predictions?timeframe=short_term',
      icon: <FiActivity className="h-4 w-4" />
    },
    {
      name: t('medium_term'),
      path: '/predictions?timeframe=medium_term',
      icon: <FiTrendingUp className="h-4 w-4" />
    },
    {
      name: t('long_term'),
      path: '/predictions?timeframe=long_term',
      icon: <FiBarChart2 className="h-4 w-4" />
    }
  ];

  // Watchlist stocks
  const watchlistStocks = [
    { symbol: 'TATAMOTORS.NS', name: 'Tata Motors' },
    { symbol: 'RELIANCE.NS', name: 'Reliance Industries' },
    { symbol: 'INFY.NS', name: 'Infosys' },
    { symbol: 'HDFCBANK.NS', name: 'HDFC Bank' },
    { symbol: 'TCS.NS', name: 'Tata Consultancy Services' }
  ];

  return (
    <>
      {/* Overlay for mobile */}
      {isOpen && (
        <div
          className="fixed inset-0 z-20 bg-black bg-opacity-50 lg:hidden"
          onClick={toggleSidebar}
        ></div>
      )}

      {/* Sidebar */}
      <aside
        className={`fixed top-0 left-0 z-30 h-full w-64 transform bg-white dark:bg-neutral-800 shadow-lg transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:h-auto ${isOpen ? 'translate-x-0' : '-translate-x-full'}`}
      >
        <div className="h-full flex flex-col overflow-y-auto">
          {/* Sidebar header */}
          <div className="flex items-center justify-between px-4 h-16 border-b border-neutral-200 dark:border-neutral-700">
            <h2 className="text-xl font-bold text-primary-600 dark:text-primary-400">
              {t('app_name')}
            </h2>
            <button
              onClick={toggleSidebar}
              className="p-2 rounded-md text-neutral-600 dark:text-neutral-400 hover:bg-neutral-100 dark:hover:bg-neutral-700 focus:outline-none lg:hidden"
              aria-label="Close sidebar"
            >
              <FiX className="h-5 w-5" />
            </button>
          </div>

          {/* Navigation */}
          <nav className="flex-1 px-2 py-4 space-y-1">
            {navItems.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) =>
                  `flex items-center px-3 py-2 rounded-md text-sm font-medium ${isActive ? 'bg-primary-100 dark:bg-primary-900 text-primary-600 dark:text-primary-300' : 'text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-700'}`
                }
                onClick={() => {
                  if (window.innerWidth < 1024) {
                    toggleSidebar();
                  }
                }}
              >
                <span className="mr-3">{item.icon}</span>
                {item.name}
              </NavLink>
            ))}
          </nav>

          {/* Prediction Timeframes */}
          <div className="px-3 py-2 border-t border-neutral-200 dark:border-neutral-700">
            <h3 className="text-xs font-semibold text-neutral-500 dark:text-neutral-400 uppercase tracking-wider">
              {t('timeframe')}
            </h3>
            <div className="mt-2 space-y-1">
              {timeframes.map((timeframe) => (
                <NavLink
                  key={timeframe.path}
                  to={timeframe.path}
                  className={({ isActive }) =>
                    `flex items-center px-3 py-1.5 rounded-md text-xs font-medium ${isActive ? 'bg-primary-100 dark:bg-primary-900 text-primary-600 dark:text-primary-300' : 'text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-700'}`
                  }
                  onClick={() => {
                    if (window.innerWidth < 1024) {
                      toggleSidebar();
                    }
                  }}
                >
                  <span className="mr-2">{timeframe.icon}</span>
                  {timeframe.name}
                </NavLink>
              ))}
            </div>
          </div>

          {/* Watchlist */}
          <div className="px-3 py-2 border-t border-neutral-200 dark:border-neutral-700">
            <h3 className="text-xs font-semibold text-neutral-500 dark:text-neutral-400 uppercase tracking-wider flex items-center">
              <FiStar className="h-3 w-3 mr-1" />
              {t('watchlist')}
            </h3>
            <div className="mt-2 space-y-1">
              {watchlistStocks.map((stock) => (
                <NavLink
                  key={stock.symbol}
                  to={`/stock/${stock.symbol}`}
                  className={({ isActive }) =>
                    `flex items-center justify-between px-3 py-1.5 rounded-md text-xs font-medium ${isActive ? 'bg-primary-100 dark:bg-primary-900 text-primary-600 dark:text-primary-300' : 'text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-700'}`
                  }
                  onClick={() => {
                    if (window.innerWidth < 1024) {
                      toggleSidebar();
                    }
                  }}
                >
                  <span>{stock.name}</span>
                  <span className="text-xs font-semibold text-success-600 dark:text-success-400">
                    +2.5%
                  </span>
                </NavLink>
              ))}
            </div>
          </div>

          {/* App info */}
          <div className="px-3 py-4 border-t border-neutral-200 dark:border-neutral-700 mt-auto">
            <div className="flex items-center text-xs text-neutral-500 dark:text-neutral-400">
              <FiInfo className="h-3 w-3 mr-1" />
              <span>v1.0.0</span>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
};

export default Sidebar;