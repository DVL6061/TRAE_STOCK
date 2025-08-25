import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { ChevronDownIcon, GlobeAltIcon } from '@heroicons/react/24/outline';
import { changeLanguage, getCurrentLanguage, getAvailableLanguages } from '../../i18n';

const LanguageSwitcher = ({ className = '', showLabel = true, variant = 'default' }) => {
  const { t } = useTranslation();
  const [isOpen, setIsOpen] = useState(false);
  const currentLanguage = getCurrentLanguage();
  const availableLanguages = getAvailableLanguages();

  const handleLanguageChange = async (languageCode) => {
    try {
      await changeLanguage(languageCode);
      setIsOpen(false);
      
      // Show success notification
      if (window.showNotification) {
        window.showNotification({
          type: 'success',
          message: t('success.settingsUpdated'),
          duration: 2000
        });
      }
    } catch (error) {
      console.error('Failed to change language:', error);
      
      // Show error notification
      if (window.showNotification) {
        window.showNotification({
          type: 'error',
          message: t('errors.unknownError'),
          duration: 3000
        });
      }
    }
  };

  const getLanguageFlag = (code) => {
    const flags = {
      'en': '🇺🇸',
      'hi': '🇮🇳'
    };
    return flags[code] || '🌐';
  };

  const getLanguageName = (code) => {
    const names = {
      'en': 'English',
      'hi': 'हिंदी'
    };
    return names[code] || code.toUpperCase();
  };

  const getVariantClasses = () => {
    switch (variant) {
      case 'compact':
        return {
          button: 'px-2 py-1 text-sm',
          dropdown: 'min-w-[120px]',
          item: 'px-3 py-2 text-sm'
        };
      case 'large':
        return {
          button: 'px-4 py-3 text-base',
          dropdown: 'min-w-[160px]',
          item: 'px-4 py-3 text-base'
        };
      default:
        return {
          button: 'px-3 py-2 text-sm',
          dropdown: 'min-w-[140px]',
          item: 'px-3 py-2 text-sm'
        };
    }
  };

  const variantClasses = getVariantClasses();

  return (
    <div className={`relative inline-block text-left ${className}`}>
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className={`
          inline-flex items-center justify-center gap-2 rounded-lg border border-gray-300 
          bg-white text-gray-700 shadow-sm hover:bg-gray-50 focus:outline-none 
          focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-all duration-200
          dark:bg-gray-800 dark:border-gray-600 dark:text-gray-200 dark:hover:bg-gray-700
          ${variantClasses.button}
        `}
        aria-expanded={isOpen}
        aria-haspopup="true"
        aria-label={t('settings.language')}
      >
        <GlobeAltIcon className="h-4 w-4" />
        <span className="flex items-center gap-1">
          <span className="text-base leading-none">
            {getLanguageFlag(currentLanguage)}
          </span>
          {showLabel && (
            <span className="font-medium">
              {getLanguageName(currentLanguage)}
            </span>
          )}
        </span>
        <ChevronDownIcon 
          className={`h-4 w-4 transition-transform duration-200 ${
            isOpen ? 'rotate-180' : ''
          }`} 
        />
      </button>

      {isOpen && (
        <>
          {/* Backdrop */}
          <div 
            className="fixed inset-0 z-10" 
            onClick={() => setIsOpen(false)}
            aria-hidden="true"
          />
          
          {/* Dropdown */}
          <div className={`
            absolute right-0 z-20 mt-2 origin-top-right rounded-lg bg-white shadow-lg 
            ring-1 ring-black ring-opacity-5 focus:outline-none
            dark:bg-gray-800 dark:ring-gray-600
            ${variantClasses.dropdown}
          `}>
            <div className="py-1" role="menu" aria-orientation="vertical">
              {availableLanguages.map((lang) => {
                const isActive = lang.code === currentLanguage;
                
                return (
                  <button
                    key={lang.code}
                    onClick={() => handleLanguageChange(lang.code)}
                    className={`
                      w-full flex items-center gap-3 text-left transition-colors duration-150
                      hover:bg-gray-100 dark:hover:bg-gray-700
                      ${isActive 
                        ? 'bg-blue-50 text-blue-700 dark:bg-blue-900/20 dark:text-blue-300' 
                        : 'text-gray-700 dark:text-gray-200'
                      }
                      ${variantClasses.item}
                    `}
                    role="menuitem"
                    disabled={isActive}
                  >
                    <span className="text-base leading-none">
                      {getLanguageFlag(lang.code)}
                    </span>
                    <span className="flex-1 font-medium">
                      {getLanguageName(lang.code)}
                    </span>
                    {isActive && (
                      <div className="h-2 w-2 rounded-full bg-blue-600 dark:bg-blue-400" />
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        </>
      )}
    </div>
  );
};

// Hook for using language switcher functionality
export const useLanguageSwitcher = () => {
  const { t } = useTranslation();
  const currentLanguage = getCurrentLanguage();
  const availableLanguages = getAvailableLanguages();

  const switchLanguage = async (languageCode) => {
    try {
      await changeLanguage(languageCode);
      return { success: true };
    } catch (error) {
      console.error('Failed to switch language:', error);
      return { success: false, error };
    }
  };

  const getLanguageInfo = (code = currentLanguage) => {
    const info = {
      'en': {
        name: 'English',
        nativeName: 'English',
        flag: '🇺🇸',
        direction: 'ltr',
        code: 'en'
      },
      'hi': {
        name: 'Hindi',
        nativeName: 'हिंदी',
        flag: '🇮🇳',
        direction: 'ltr',
        code: 'hi'
      }
    };
    return info[code] || info['en'];
  };

  return {
    currentLanguage,
    availableLanguages,
    switchLanguage,
    getLanguageInfo,
    t
  };
};

// Language detection component for initial setup
export const LanguageDetector = ({ children }) => {
  const [isDetecting, setIsDetecting] = useState(true);
  const { t } = useTranslation();

  React.useEffect(() => {
    const detectAndSetLanguage = async () => {
      try {
        // Check if language is already set
        const savedLanguage = localStorage.getItem('i18nextLng');
        if (savedLanguage && ['en', 'hi'].includes(savedLanguage)) {
          setIsDetecting(false);
          return;
        }

        // Detect browser language
        const browserLanguage = navigator.language || navigator.languages[0];
        const languageCode = browserLanguage.startsWith('hi') ? 'hi' : 'en';
        
        await changeLanguage(languageCode);
        setIsDetecting(false);
      } catch (error) {
        console.error('Language detection failed:', error);
        // Fallback to English
        await changeLanguage('en');
        setIsDetecting(false);
      }
    };

    detectAndSetLanguage();
  }, []);

  if (isDetecting) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-50 dark:bg-gray-900">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600 dark:text-gray-400">
            {t('app.loading')}
          </p>
        </div>
      </div>
    );
  }

  return children;
};

export default LanguageSwitcher;