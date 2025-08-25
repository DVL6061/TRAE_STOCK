import React from 'react';
import { useTranslation } from 'react-i18next';
import { NavLink, useLocation } from 'react-router-dom';
import {
  HomeIcon,
  ChartBarIcon,
  BriefcaseIcon,
  BookmarkIcon,
  NewspaperIcon,
  ChartPieIcon,
  Cog6ToothIcon,
  XMarkIcon,
  CpuChipIcon,
  GlobeAltIcon,
  BoltIcon,
  TrendingUpIcon
} from '@heroicons/react/24/outline';
import {
  HomeIcon as HomeIconSolid,
  ChartBarIcon as ChartBarIconSolid,
  BriefcaseIcon as BriefcaseIconSolid,
  BookmarkIcon as BookmarkIconSolid,
  NewspaperIcon as NewspaperIconSolid,
  ChartPieIcon as ChartPieIconSolid,
  Cog6ToothIcon as Cog6ToothIconSolid
} from '@heroicons/react/24/solid';

const Sidebar = ({ isOpen, onClose }) => {
  const { t } = useTranslation();
  const location = useLocation();

  const navigation = [
    {
      name: t('navigation.dashboard'),
      href: '/dashboard',
      icon: HomeIcon,
      iconSolid: HomeIconSolid,
      description: 'Market overview and portfolio summary'
    },
    {
      name: t('navigation.predictions'),
      href: '/predictions',
      icon: CpuChipIcon,
      iconSolid: CpuChipIcon,
      description: 'AI-powered stock predictions',
      badge: 'AI'
    },
    {
      name: t('navigation.portfolio'),
      href: '/portfolio',
      icon: BriefcaseIcon,
      iconSolid: BriefcaseIconSolid,
      description: 'Your investment portfolio'
    },
    {
      name: t('navigation.watchlist'),
      href: '/watchlist',
      icon: BookmarkIcon,
      iconSolid: BookmarkIconSolid,
      description: 'Track your favorite stocks'
    },
    {
      name: t('navigation.news'),
      href: '/news',
      icon: NewspaperIcon,
      iconSolid: NewspaperIconSolid,
      description: 'Market news with sentiment analysis',
      badge: 'Live'
    },
    {
      name: t('navigation.analysis'),
      href: '/analysis',
      icon: ChartPieIcon,
      iconSolid: ChartPieIconSolid,
      description: 'Technical and fundamental analysis'
    }
  ];

  const bottomNavigation = [
    {
      name: t('navigation.settings'),
      href: '/settings',
      icon: Cog6ToothIcon,
      iconSolid: Cog6ToothIconSolid,
      description: 'Application settings'
    }
  ];

  const isCurrentPath = (path) => {
    return location.pathname === path;
  };

  const NavItem = ({ item, onClick }) => {
    const Icon = isCurrentPath(item.href) ? item.iconSolid : item.icon;
    const isActive = isCurrentPath(item.href);

    return (
      <NavLink
        to={item.href}
        onClick={onClick}
        className={({ isActive: linkActive }) => `
          group flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-all duration-200
          ${linkActive || isActive
            ? 'bg-blue-50 text-blue-700 shadow-sm dark:bg-blue-900/20 dark:text-blue-300'
            : 'text-gray-700 hover:bg-gray-100 hover:text-gray-900 dark:text-gray-300 dark:hover:bg-gray-800 dark:hover:text-white'
          }
        `}
        title={isOpen ? '' : item.name}
      >
        <div className="relative flex items-center">
          <Icon className={`h-5 w-5 flex-shrink-0 transition-colors ${
            isActive 
              ? 'text-blue-600 dark:text-blue-400' 
              : 'text-gray-400 group-hover:text-gray-500 dark:group-hover:text-gray-300'
          }`} />
          {item.badge && (
            <span className={`absolute -top-1 -right-1 h-2 w-2 rounded-full ${
              item.badge === 'AI' ? 'bg-purple-500' : 'bg-green-500'
            } animate-pulse`} />
          )}
        </div>
        
        <div className={`flex-1 transition-all duration-300 ${
          isOpen ? 'opacity-100 translate-x-0' : 'opacity-0 -translate-x-2 lg:hidden'
        }`}>
          <div className="flex items-center justify-between">
            <span className="truncate">{item.name}</span>
            {item.badge && (
              <span className={`ml-2 inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${
                item.badge === 'AI' 
                  ? 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300'
                  : 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300'
              }`}>
                {item.badge}
              </span>
            )}
          </div>
          {isOpen && (
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5 truncate">
              {item.description}
            </p>
          )}
        </div>
      </NavLink>
    );
  };

  return (
    <>
      {/* Mobile backdrop */}
      {isOpen && (
        <div 
          className="fixed inset-0 z-40 bg-black bg-opacity-50 lg:hidden" 
          onClick={onClose}
        />
      )}
      
      {/* Sidebar */}
      <div className={`
        fixed inset-y-0 left-0 z-50 flex flex-col bg-white dark:bg-gray-900 shadow-xl transition-all duration-300
        ${isOpen ? 'w-64' : 'w-16 lg:w-16'}
        lg:translate-x-0 ${isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
      `}>
        {/* Header */}
        <div className={`flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700 ${
          isOpen ? 'px-4' : 'px-2 lg:px-2'
        }`}>
          <div className={`flex items-center gap-3 transition-all duration-300 ${
            isOpen ? 'opacity-100' : 'opacity-0 lg:opacity-100'
          }`}>
            <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center flex-shrink-0">
              <TrendingUpIcon className="h-5 w-5 text-white" />
            </div>
            {isOpen && (
              <div>
                <h2 className="text-lg font-bold text-gray-900 dark:text-white">
                  {t('app.title')}
                </h2>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  {t('app.version')}
                </p>
              </div>
            )}
          </div>
          
          {/* Close button (mobile only) */}
          <button
            type="button"
            onClick={onClose}
            className="lg:hidden rounded-md p-2 text-gray-400 hover:bg-gray-100 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:hover:bg-gray-800 dark:hover:text-gray-300"
          >
            <XMarkIcon className="h-5 w-5" />
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 overflow-y-auto p-4 space-y-1">
          {/* Main Navigation */}
          <div className="space-y-1">
            {navigation.map((item) => (
              <NavItem 
                key={item.href} 
                item={item} 
                onClick={() => {
                  // Close mobile sidebar on navigation
                  if (window.innerWidth < 1024) {
                    onClose();
                  }
                }}
              />
            ))}
          </div>

          {/* Divider */}
          <div className={`border-t border-gray-200 dark:border-gray-700 my-4 transition-all duration-300 ${
            isOpen ? 'mx-0' : 'mx-2'
          }`} />

          {/* AI Features Section */}
          {isOpen && (
            <div className="mb-4">
              <h3 className="px-3 text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
                AI Features
              </h3>
              <div className="space-y-2">
                <div className="flex items-center gap-3 px-3 py-2 text-sm text-gray-600 dark:text-gray-400">
                  <CpuChipIcon className="h-4 w-4 text-purple-500" />
                  <span>XGBoost Model</span>
                  <div className="ml-auto h-2 w-2 rounded-full bg-green-500" />
                </div>
                <div className="flex items-center gap-3 px-3 py-2 text-sm text-gray-600 dark:text-gray-400">
                  <BoltIcon className="h-4 w-4 text-blue-500" />
                  <span>Informer Model</span>
                  <div className="ml-auto h-2 w-2 rounded-full bg-green-500" />
                </div>
                <div className="flex items-center gap-3 px-3 py-2 text-sm text-gray-600 dark:text-gray-400">
                  <ChartBarIcon className="h-4 w-4 text-green-500" />
                  <span>DQN Trading</span>
                  <div className="ml-auto h-2 w-2 rounded-full bg-green-500" />
                </div>
                <div className="flex items-center gap-3 px-3 py-2 text-sm text-gray-600 dark:text-gray-400">
                  <GlobeAltIcon className="h-4 w-4 text-orange-500" />
                  <span>FinGPT Sentiment</span>
                  <div className="ml-auto h-2 w-2 rounded-full bg-green-500" />
                </div>
              </div>
            </div>
          )}

          {/* Market Status */}
          {isOpen && (
            <div className="bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20 rounded-lg p-3 mb-4">
              <div className="flex items-center gap-2 mb-2">
                <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
                <span className="text-sm font-medium text-gray-900 dark:text-white">
                  {t('dashboard.marketOpen')}
                </span>
              </div>
              <div className="text-xs text-gray-600 dark:text-gray-400">
                <div>NIFTY: <span className="text-green-600 font-medium">19,845</span></div>
                <div>SENSEX: <span className="text-green-600 font-medium">66,598</span></div>
              </div>
            </div>
          )}
        </nav>

        {/* Bottom Navigation */}
        <div className="border-t border-gray-200 dark:border-gray-700 p-4">
          <div className="space-y-1">
            {bottomNavigation.map((item) => (
              <NavItem 
                key={item.href} 
                item={item} 
                onClick={() => {
                  if (window.innerWidth < 1024) {
                    onClose();
                  }
                }}
              />
            ))}
          </div>
          
          {/* Collapse/Expand Button (Desktop only) */}
          <button
            type="button"
            onClick={() => {
              // This would toggle the sidebar state in the parent component
              // For now, we'll just log it
              console.log('Toggle sidebar collapse');
            }}
            className={`hidden lg:flex items-center justify-center w-full mt-4 p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors ${
              isOpen ? 'text-sm' : ''
            }`}
            title={isOpen ? 'Collapse sidebar' : 'Expand sidebar'}
          >
            {isOpen ? (
              <>
                <ChartBarIcon className="h-4 w-4 mr-2" />
                <span className="text-xs">Collapse</span>
              </>
            ) : (
              <ChartBarIcon className="h-4 w-4" />
            )}
          </button>
        </div>
      </div>
    </>
  );
};

export default Sidebar;