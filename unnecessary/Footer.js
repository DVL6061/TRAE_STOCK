import React from 'react';
import { useTranslation } from 'react-i18next';
import { Link } from 'react-router-dom';

// Icons
import { FiGithub, FiHeart } from 'react-icons/fi';

const Footer = () => {
  const { t } = useTranslation();
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-white dark:bg-neutral-800 border-t border-neutral-200 dark:border-neutral-700 py-4 mt-auto">
      <div className="container mx-auto px-4">
        <div className="flex flex-col md:flex-row items-center justify-between">
          <div className="text-center md:text-left mb-4 md:mb-0">
            <p className="text-sm text-neutral-600 dark:text-neutral-400">
              &copy; {currentYear} {t('app_name')}. {t('all_rights_reserved')}.
            </p>
            <p className="text-xs text-neutral-500 dark:text-neutral-500 mt-1">
              {t('disclaimer')}: {t('prediction_disclaimer')}
            </p>
          </div>
          
          <div className="flex items-center space-x-4">
            <Link to="/about" className="text-sm text-neutral-600 dark:text-neutral-400 hover:text-primary-600 dark:hover:text-primary-400">
              {t('about')}
            </Link>
            <Link to="/privacy" className="text-sm text-neutral-600 dark:text-neutral-400 hover:text-primary-600 dark:hover:text-primary-400">
              {t('privacy_policy')}
            </Link>
            <Link to="/terms" className="text-sm text-neutral-600 dark:text-neutral-400 hover:text-primary-600 dark:hover:text-primary-400">
              {t('terms_of_service')}
            </Link>
            <a 
              href="https://github.com" 
              target="_blank" 
              rel="noopener noreferrer" 
              className="text-sm text-neutral-600 dark:text-neutral-400 hover:text-primary-600 dark:hover:text-primary-400 flex items-center"
            >
              <FiGithub className="mr-1 h-4 w-4" />
              GitHub
            </a>
          </div>
        </div>
        
        <div className="mt-4 text-center text-xs text-neutral-500 dark:text-neutral-500 flex items-center justify-center">
          {t('made_with')} <FiHeart className="mx-1 text-danger-500" /> {t('using')} React, FastAPI, ML & AI
        </div>
      </div>
    </footer>
  );
};

export default Footer;