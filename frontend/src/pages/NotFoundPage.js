import React from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { FiHome, FiAlertTriangle, FiSearch, FiArrowLeft } from 'react-icons/fi';

const NotFoundPage = () => {
  const { t } = useTranslation();

  return (
    <div className="min-h-[80vh] flex items-center justify-center px-4 py-12">
      <div className="text-center max-w-md">
        <div className="flex justify-center mb-6">
          <div className="h-24 w-24 rounded-full bg-neutral-100 dark:bg-neutral-800 flex items-center justify-center">
            <FiAlertTriangle className="h-12 w-12 text-warning-500" />
          </div>
        </div>
        
        <h1 className="text-4xl font-bold text-neutral-900 dark:text-neutral-100 mb-4">
          404
        </h1>
        
        <h2 className="text-2xl font-semibold text-neutral-800 dark:text-neutral-200 mb-4">
          {t('page_not_found')}
        </h2>
        
        <p className="text-neutral-600 dark:text-neutral-400 mb-8">
          {t('page_not_found_description')}
        </p>
        
        <div className="flex flex-col sm:flex-row space-y-4 sm:space-y-0 sm:space-x-4 justify-center">
          <Link
            to="/"
            className="btn btn-primary flex items-center justify-center"
          >
            <FiHome className="mr-2" />
            {t('go_to_home')}
          </Link>
          
          <Link
            to="/"
            className="btn btn-outline flex items-center justify-center"
            onClick={() => window.history.back()}
          >
            <FiArrowLeft className="mr-2" />
            {t('go_back')}
          </Link>
        </div>
        
        <div className="mt-12">
          <h3 className="text-lg font-medium text-neutral-800 dark:text-neutral-200 mb-4">
            {t('looking_for_something')}
          </h3>
          
          <div className="relative max-w-xs mx-auto">
            <input
              type="text"
              className="input w-full pl-10"
              placeholder={t('search_placeholder')}
            />
            <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
              <FiSearch className="text-neutral-500" />
            </div>
          </div>
          
          <div className="mt-6 text-sm">
            <p className="text-neutral-600 dark:text-neutral-400 mb-2">
              {t('popular_pages')}:
            </p>
            
            <div className="flex flex-wrap justify-center gap-2">
              <Link
                to="/"
                className="px-3 py-1 bg-neutral-100 dark:bg-neutral-800 rounded-full text-neutral-700 dark:text-neutral-300 hover:bg-neutral-200 dark:hover:bg-neutral-700"
              >
                {t('dashboard')}
              </Link>
              
              <Link
                to="/stock/TATAMOTORS.NS"
                className="px-3 py-1 bg-neutral-100 dark:bg-neutral-800 rounded-full text-neutral-700 dark:text-neutral-300 hover:bg-neutral-200 dark:hover:bg-neutral-700"
              >
                Tata Motors
              </Link>
              
              <Link
                to="/news"
                className="px-3 py-1 bg-neutral-100 dark:bg-neutral-800 rounded-full text-neutral-700 dark:text-neutral-300 hover:bg-neutral-200 dark:hover:bg-neutral-700"
              >
                {t('news')}
              </Link>
              
              <Link
                to="/predictions"
                className="px-3 py-1 bg-neutral-100 dark:bg-neutral-800 rounded-full text-neutral-700 dark:text-neutral-300 hover:bg-neutral-200 dark:hover:bg-neutral-700"
              >
                {t('predictions')}
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NotFoundPage;