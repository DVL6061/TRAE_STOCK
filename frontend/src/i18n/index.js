import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';

// Import language resources
import enTranslations from './locales/en.json';
import hiTranslations from './locales/hi.json';

// Language resources
const resources = {
  en: {
    translation: enTranslations
  },
  hi: {
    translation: hiTranslations
  }
};

// Initialize i18next
i18n
  .use(LanguageDetector) // Detect user language
  .use(initReactI18next) // Pass i18n instance to react-i18next
  .init({
    resources,
    
    // Default language
    fallbackLng: 'en',
    
    // Language detection options
    detection: {
      // Order of language detection methods
      order: ['localStorage', 'navigator', 'htmlTag'],
      
      // Cache user language
      caches: ['localStorage'],
      
      // Key to store language in localStorage
      lookupLocalStorage: 'i18nextLng',
      
      // Don't lookup from path, subdomain, etc.
      lookupFromPathIndex: 0,
      lookupFromSubdomainIndex: 0,
    },
    
    // Interpolation options
    interpolation: {
      escapeValue: false, // React already escapes values
      
      // Custom format functions
      format: function(value, format, lng) {
        if (format === 'currency') {
          // Format currency based on language
          if (lng === 'hi') {
            return new Intl.NumberFormat('hi-IN', {
              style: 'currency',
              currency: 'INR',
              minimumFractionDigits: 2
            }).format(value);
          } else {
            return new Intl.NumberFormat('en-IN', {
              style: 'currency',
              currency: 'INR',
              minimumFractionDigits: 2
            }).format(value);
          }
        }
        
        if (format === 'number') {
          // Format numbers based on language
          if (lng === 'hi') {
            return new Intl.NumberFormat('hi-IN').format(value);
          } else {
            return new Intl.NumberFormat('en-IN').format(value);
          }
        }
        
        if (format === 'percent') {
          // Format percentages
          if (lng === 'hi') {
            return new Intl.NumberFormat('hi-IN', {
              style: 'percent',
              minimumFractionDigits: 2
            }).format(value / 100);
          } else {
            return new Intl.NumberFormat('en-IN', {
              style: 'percent',
              minimumFractionDigits: 2
            }).format(value / 100);
          }
        }
        
        if (format === 'date') {
          // Format dates
          const date = new Date(value);
          if (lng === 'hi') {
            return date.toLocaleDateString('hi-IN');
          } else {
            return date.toLocaleDateString('en-IN');
          }
        }
        
        if (format === 'time') {
          // Format time
          const date = new Date(value);
          if (lng === 'hi') {
            return date.toLocaleTimeString('hi-IN');
          } else {
            return date.toLocaleTimeString('en-IN');
          }
        }
        
        return value;
      }
    },
    
    // React options
    react: {
      // Wait for translation to be loaded before rendering
      useSuspense: false,
      
      // Bind i18n instance to component
      bindI18n: 'languageChanged',
      
      // Bind store to component
      bindI18nStore: '',
      
      // How to transform the key
      transEmptyNodeValue: '',
      transSupportBasicHtmlNodes: true,
      transKeepBasicHtmlNodesFor: ['br', 'strong', 'i', 'em', 'span'],
    },
    
    // Debug mode (disable in production)
    debug: process.env.NODE_ENV === 'development',
    
    // Namespace options
    defaultNS: 'translation',
    ns: ['translation'],
    
    // Key separator
    keySeparator: '.',
    
    // Nested separator
    nsSeparator: ':',
    
    // Return objects for nested keys
    returnObjects: false,
    
    // Return null for missing keys
    returnNull: false,
    
    // Return empty string for missing keys
    returnEmptyString: false,
    
    // Postprocess missing keys
    postProcess: false,
    
    // Load path for loading translations
    // (not used since we're importing directly)
    // load: 'languageOnly',
    
    // Preload languages
    preload: ['en', 'hi'],
    
    // Clean code
    cleanCode: true,
    
    // Lowercase language codes
    lowerCaseLng: true,
    
    // Non-explicit support
    nonExplicitSupportedLngs: false,
    
    // Load languages synchronously
    initImmediate: false,
  });

// Export language utilities
export const supportedLanguages = [
  {
    code: 'en',
    name: 'English',
    nativeName: 'English',
    flag: '🇺🇸',
    rtl: false
  },
  {
    code: 'hi',
    name: 'Hindi',
    nativeName: 'हिन्दी',
    flag: '🇮🇳',
    rtl: false
  }
];

// Get current language info
export const getCurrentLanguageInfo = () => {
  const currentLang = i18n.language || 'en';
  return supportedLanguages.find(lang => lang.code === currentLang) || supportedLanguages[0];
};

// Change language function
export const changeLanguage = (langCode) => {
  return i18n.changeLanguage(langCode);
};

// Get available languages
export const getAvailableLanguages = () => {
  return supportedLanguages;
};

// Format currency helper
export const formatCurrency = (value, language = null) => {
  const lng = language || i18n.language || 'en';
  return i18n.format(value, 'currency', lng);
};

// Format number helper
export const formatNumber = (value, language = null) => {
  const lng = language || i18n.language || 'en';
  return i18n.format(value, 'number', lng);
};

// Format percentage helper
export const formatPercent = (value, language = null) => {
  const lng = language || i18n.language || 'en';
  return i18n.format(value, 'percent', lng);
};

// Format date helper
export const formatDate = (value, language = null) => {
  const lng = language || i18n.language || 'en';
  return i18n.format(value, 'date', lng);
};

// Format time helper
export const formatTime = (value, language = null) => {
  const lng = language || i18n.language || 'en';
  return i18n.format(value, 'time', lng);
};

// Check if current language is RTL
export const isRTL = () => {
  const currentLang = getCurrentLanguageInfo();
  return currentLang.rtl;
};

// Get text direction
export const getTextDirection = () => {
  return isRTL() ? 'rtl' : 'ltr';
};

// Translation key helpers
export const t = i18n.t.bind(i18n);

// Namespace helpers
export const createNamespacedT = (namespace) => {
  return (key, options) => i18n.t(`${namespace}:${key}`, options);
};

// Pluralization helpers
export const tPlural = (key, count, options = {}) => {
  return i18n.t(key, { count, ...options });
};

// Context helpers
export const tContext = (key, context, options = {}) => {
  return i18n.t(`${key}_${context}`, options);
};

// Export i18n instance
export default i18n;

// Language change event listeners
const languageChangeListeners = new Set();

// Add language change listener
export const addLanguageChangeListener = (callback) => {
  languageChangeListeners.add(callback);
  
  // Return unsubscribe function
  return () => {
    languageChangeListeners.delete(callback);
  };
};

// Notify language change listeners
i18n.on('languageChanged', (lng) => {
  languageChangeListeners.forEach(callback => {
    try {
      callback(lng);
    } catch (error) {
      console.error('Error in language change listener:', error);
    }
  });
});

// Initialize language direction on DOM
i18n.on('languageChanged', (lng) => {
  const direction = getTextDirection();
  document.documentElement.setAttribute('dir', direction);
  document.documentElement.setAttribute('lang', lng);
});

// Set initial direction
const initialDirection = getTextDirection();
document.documentElement.setAttribute('dir', initialDirection);
document.documentElement.setAttribute('lang', i18n.language || 'en');

// Export for debugging in development
if (process.env.NODE_ENV === 'development') {
  window.i18n = i18n;
  window.i18nUtils = {
    supportedLanguages,
    getCurrentLanguageInfo,
    changeLanguage,
    formatCurrency,
    formatNumber,
    formatPercent,
    formatDate,
    formatTime,
    isRTL,
    getTextDirection
  };
}