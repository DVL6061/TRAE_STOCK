import React from 'react';
import { useTranslation } from 'react-i18next';
import {
  HeartIcon,
  GlobeAltIcon,
  ShieldCheckIcon,
  CpuChipIcon,
  ChartBarIcon,
  BoltIcon
} from '@heroicons/react/24/outline';

const Footer = () => {
  const { t, i18n } = useTranslation();
  const currentYear = new Date().getFullYear();

  const footerLinks = {
    product: [
      { name: t('footer.links.features'), href: '#features' },
      { name: t('footer.links.pricing'), href: '#pricing' },
      { name: t('footer.links.api'), href: '#api' },
      { name: t('footer.links.documentation'), href: '#docs' }
    ],
    company: [
      { name: t('footer.links.about'), href: '#about' },
      { name: t('footer.links.blog'), href: '#blog' },
      { name: t('footer.links.careers'), href: '#careers' },
      { name: t('footer.links.contact'), href: '#contact' }
    ],
    legal: [
      { name: t('footer.links.privacy'), href: '#privacy' },
      { name: t('footer.links.terms'), href: '#terms' },
      { name: t('footer.links.disclaimer'), href: '#disclaimer' },
      { name: t('footer.links.compliance'), href: '#compliance' }
    ],
    support: [
      { name: t('footer.links.help'), href: '#help' },
      { name: t('footer.links.community'), href: '#community' },
      { name: t('footer.links.status'), href: '#status' },
      { name: t('footer.links.feedback'), href: '#feedback' }
    ]
  };

  const aiModels = [
    {
      name: 'XGBoost',
      icon: ChartBarIcon,
      description: t('footer.models.xgboost'),
      status: 'active'
    },
    {
      name: 'Informer',
      icon: BoltIcon,
      description: t('footer.models.informer'),
      status: 'active'
    },
    {
      name: 'DQN',
      icon: CpuChipIcon,
      description: t('footer.models.dqn'),
      status: 'active'
    }
  ];

  const marketData = {
    lastUpdate: new Date().toLocaleString(i18n.language === 'hi' ? 'hi-IN' : 'en-IN'),
    dataProvider: 'Angel One Smart API',
    newsProvider: 'CNBC, Moneycontrol, Mint'
  };

  return (
    <footer className="bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700">
      <div className="max-w-7xl mx-auto px-3 sm:px-4 lg:px-8">
        {/* Main Footer Content */}
        <div className="py-6 sm:py-8 lg:py-12">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6 lg:gap-8">
            {/* Brand Section */}
            <div className="col-span-1 sm:col-span-2 lg:col-span-1">
              <div className="flex items-center gap-2 sm:gap-3 mb-3 sm:mb-4">
                <div className="h-8 w-8 sm:h-10 sm:w-10 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                  <ChartBarIcon className="h-4 w-4 sm:h-6 sm:w-6 text-white" />
                </div>
                <div>
                  <h3 className="text-base sm:text-lg font-bold text-gray-900 dark:text-white">
                    {t('app.title')}
                  </h3>
                  <p className="text-xs sm:text-sm text-gray-500 dark:text-gray-400">
                    {t('app.tagline')}
                  </p>
                </div>
              </div>
              <p className="text-xs sm:text-sm text-gray-600 dark:text-gray-400 mb-3 sm:mb-4">
                {t('footer.description')}
              </p>
              
              {/* AI Models Status */}
              <div className="space-y-1 sm:space-y-2">
                <h4 className="text-xs sm:text-sm font-semibold text-gray-900 dark:text-white mb-1 sm:mb-2">
                  {t('footer.aiModels')}
                </h4>
                {aiModels.map((model) => {
                  const Icon = model.icon;
                  return (
                    <div key={model.name} className="flex items-center gap-1 sm:gap-2 text-xs sm:text-sm">
                      <Icon className="h-3 w-3 sm:h-4 sm:w-4 text-blue-500" />
                      <span className="text-gray-700 dark:text-gray-300">{model.name}</span>
                      <div className={`h-1.5 w-1.5 sm:h-2 sm:w-2 rounded-full ${
                        model.status === 'active' ? 'bg-green-500' : 'bg-red-500'
                      }`} />
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Product Links */}
            <div>
              <h3 className="text-xs sm:text-sm font-semibold text-gray-900 dark:text-white uppercase tracking-wider mb-2 sm:mb-4">
                {t('footer.sections.product')}
              </h3>
              <ul className="space-y-2 sm:space-y-3">
                {footerLinks.product.map((link) => (
                  <li key={link.name}>
                    <a
                      href={link.href}
                      className="text-xs sm:text-sm text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white transition-colors"
                    >
                      {link.name}
                    </a>
                  </li>
                ))}
              </ul>
            </div>

            {/* Company Links */}
            <div>
              <h3 className="text-xs sm:text-sm font-semibold text-gray-900 dark:text-white uppercase tracking-wider mb-2 sm:mb-4">
                {t('footer.sections.company')}
              </h3>
              <ul className="space-y-2 sm:space-y-3">
                {footerLinks.company.map((link) => (
                  <li key={link.name}>
                    <a
                      href={link.href}
                      className="text-xs sm:text-sm text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white transition-colors"
                    >
                      {link.name}
                    </a>
                  </li>
                ))}
              </ul>
            </div>

            {/* Support Links */}
            <div>
              <h3 className="text-xs sm:text-sm font-semibold text-gray-900 dark:text-white uppercase tracking-wider mb-2 sm:mb-4">
                {t('footer.sections.support')}
              </h3>
              <ul className="space-y-2 sm:space-y-3">
                {footerLinks.support.map((link) => (
                  <li key={link.name}>
                    <a
                      href={link.href}
                      className="text-xs sm:text-sm text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white transition-colors"
                    >
                      {link.name}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>

        {/* Market Data Status */}
        <div className="py-4 sm:py-6 border-t border-gray-200 dark:border-gray-700">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 sm:gap-4">
            <div className="flex items-center gap-2">
              <GlobeAltIcon className="h-4 w-4 sm:h-5 sm:w-5 text-blue-500" />
              <div>
                <p className="text-xs sm:text-sm font-medium text-gray-900 dark:text-white">
                  {t('footer.dataProvider')}
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  {marketData.dataProvider}
                </p>
              </div>
            </div>
            
            <div className="flex items-center gap-2">
              <ShieldCheckIcon className="h-4 w-4 sm:h-5 sm:w-5 text-green-500" />
              <div>
                <p className="text-xs sm:text-sm font-medium text-gray-900 dark:text-white">
                  {t('footer.lastUpdate')}
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  {marketData.lastUpdate}
                </p>
              </div>
            </div>
            
            <div className="flex items-center gap-2 sm:col-span-2 lg:col-span-1">
              <CpuChipIcon className="h-4 w-4 sm:h-5 sm:w-5 text-purple-500" />
              <div>
                <p className="text-xs sm:text-sm font-medium text-gray-900 dark:text-white">
                  {t('footer.aiPowered')}
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  FinGPT + ML Models
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Legal Links */}
        <div className="py-3 sm:py-4 border-t border-gray-200 dark:border-gray-700">
          <div className="flex flex-col sm:flex-row justify-between items-center gap-3 sm:gap-4">
            <div className="flex flex-wrap gap-3 sm:gap-6 justify-center sm:justify-start">
              {footerLinks.legal.map((link) => (
                <a
                  key={link.name}
                  href={link.href}
                  className="text-xs text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 transition-colors"
                >
                  {link.name}
                </a>
              ))}
            </div>
            
            <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
              <span>{t('footer.language')}: {i18n.language === 'hi' ? 'हिंदी' : 'English'}</span>
              <span className="hidden sm:inline">•</span>
              <span className="hidden sm:inline">{t('footer.region')}: India</span>
            </div>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="py-4 sm:py-6 border-t border-gray-200 dark:border-gray-700">
          <div className="flex flex-col sm:flex-row justify-between items-center gap-3 sm:gap-4">
            <div className="flex items-center gap-1 sm:gap-2 text-xs sm:text-sm text-gray-600 dark:text-gray-400">
              <span>© {currentYear} {t('app.title')}.</span>
              <span className="hidden sm:inline">{t('footer.allRightsReserved')}</span>
            </div>
            
            <div className="flex items-center gap-1 sm:gap-2 text-xs sm:text-sm text-gray-600 dark:text-gray-400">
              <span>{t('footer.madeWith')}</span>
              <HeartIcon className="h-3 w-3 sm:h-4 sm:w-4 text-red-500" />
              <span>{t('footer.inIndia')}</span>
            </div>
          </div>
        </div>

        {/* Disclaimer */}
        <div className="py-3 sm:py-4 border-t border-gray-200 dark:border-gray-700">
          <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-3 sm:p-4">
            <p className="text-xs text-yellow-800 dark:text-yellow-200">
              <strong>{t('footer.disclaimer.title')}:</strong> {t('footer.disclaimer.text')}
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;