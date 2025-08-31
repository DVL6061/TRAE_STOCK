import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { toast } from 'react-toastify';

// Validation rules
const validationRules = {
  required: (value) => {
    if (typeof value === 'string') {
      return value.trim().length > 0;
    }
    return value !== null && value !== undefined && value !== '';
  },
  
  email: (value) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(value);
  },
  
  minLength: (min) => (value) => {
    return typeof value === 'string' && value.length >= min;
  },
  
  maxLength: (max) => (value) => {
    return typeof value === 'string' && value.length <= max;
  },
  
  numeric: (value) => {
    return !isNaN(parseFloat(value)) && isFinite(value);
  },
  
  positive: (value) => {
    return parseFloat(value) > 0;
  },
  
  range: (min, max) => (value) => {
    const num = parseFloat(value);
    return num >= min && num <= max;
  },
  
  stockSymbol: (value) => {
    const symbolRegex = /^[A-Z]{1,10}$/;
    return symbolRegex.test(value.toUpperCase());
  },
  
  phoneNumber: (value) => {
    const phoneRegex = /^[+]?[1-9]?[0-9]{7,15}$/;
    return phoneRegex.test(value.replace(/\s/g, ''));
  },
  
  password: (value) => {
    // At least 8 characters, 1 uppercase, 1 lowercase, 1 number
    const passwordRegex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[a-zA-Z\d@$!%*?&]{8,}$/;
    return passwordRegex.test(value);
  },
  
  confirmPassword: (originalPassword) => (value) => {
    return value === originalPassword;
  },
  
  date: (value) => {
    const date = new Date(value);
    return date instanceof Date && !isNaN(date);
  },
  
  futureDate: (value) => {
    const date = new Date(value);
    const now = new Date();
    return date > now;
  },
  
  pastDate: (value) => {
    const date = new Date(value);
    const now = new Date();
    return date < now;
  }
};

// Error messages
const getErrorMessage = (rule, params, t) => {
  const messages = {
    required: t('fieldRequired'),
    email: t('invalidEmail'),
    minLength: t('minLengthError', { min: params }),
    maxLength: t('maxLengthError', { max: params }),
    numeric: t('mustBeNumeric'),
    positive: t('mustBePositive'),
    range: t('valueOutOfRange', { min: params[0], max: params[1] }),
    stockSymbol: t('invalidStockSymbol'),
    phoneNumber: t('invalidPhoneNumber'),
    password: t('passwordRequirements'),
    confirmPassword: t('passwordsDoNotMatch'),
    date: t('invalidDate'),
    futureDate: t('dateMustBeFuture'),
    pastDate: t('dateMustBePast')
  };
  
  return messages[rule] || t('invalidInput');
};

// Custom hook for form validation
export const useFormValidation = (initialValues = {}, validationSchema = {}) => {
  const { t } = useTranslation();
  const [values, setValues] = useState(initialValues);
  const [errors, setErrors] = useState({});
  const [touched, setTouched] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Validate single field
  const validateField = (name, value) => {
    const fieldRules = validationSchema[name];
    if (!fieldRules) return null;

    for (const rule of fieldRules) {
      let isValid = false;
      let errorMessage = '';

      if (typeof rule === 'string') {
        // Simple rule like 'required', 'email'
        isValid = validationRules[rule](value);
        errorMessage = getErrorMessage(rule, null, t);
      } else if (typeof rule === 'object') {
        // Rule with parameters like { rule: 'minLength', params: 5 }
        const { rule: ruleName, params, message } = rule;
        
        if (Array.isArray(params)) {
          isValid = validationRules[ruleName](...params)(value);
        } else if (params !== undefined) {
          isValid = validationRules[ruleName](params)(value);
        } else {
          isValid = validationRules[ruleName](value);
        }
        
        errorMessage = message || getErrorMessage(ruleName, params, t);
      } else if (typeof rule === 'function') {
        // Custom validation function
        const result = rule(value, values);
        isValid = result === true;
        errorMessage = typeof result === 'string' ? result : t('invalidInput');
      }

      if (!isValid) {
        return errorMessage;
      }
    }

    return null;
  };

  // Validate all fields
  const validateForm = () => {
    const newErrors = {};
    let isValid = true;

    Object.keys(validationSchema).forEach(fieldName => {
      const error = validateField(fieldName, values[fieldName]);
      if (error) {
        newErrors[fieldName] = error;
        isValid = false;
      }
    });

    setErrors(newErrors);
    return isValid;
  };

  // Handle input change
  const handleChange = (name, value) => {
    setValues(prev => ({ ...prev, [name]: value }));
    
    // Clear error when user starts typing
    if (errors[name]) {
      setErrors(prev => ({ ...prev, [name]: null }));
    }
  };

  // Handle input blur
  const handleBlur = (name) => {
    setTouched(prev => ({ ...prev, [name]: true }));
    
    const error = validateField(name, values[name]);
    setErrors(prev => ({ ...prev, [name]: error }));
  };

  // Handle form submission
  const handleSubmit = async (onSubmit) => {
    setIsSubmitting(true);
    
    // Mark all fields as touched
    const allTouched = {};
    Object.keys(validationSchema).forEach(key => {
      allTouched[key] = true;
    });
    setTouched(allTouched);

    const isValid = validateForm();
    
    if (isValid) {
      try {
        await onSubmit(values);
        toast.success(t('formSubmittedSuccessfully'));
      } catch (error) {
        toast.error(error.message || t('submissionError'));
      }
    } else {
      toast.error(t('pleaseFixErrors'));
    }
    
    setIsSubmitting(false);
  };

  // Reset form
  const resetForm = () => {
    setValues(initialValues);
    setErrors({});
    setTouched({});
    setIsSubmitting(false);
  };

  return {
    values,
    errors,
    touched,
    isSubmitting,
    handleChange,
    handleBlur,
    handleSubmit,
    resetForm,
    validateForm,
    setValues
  };
};

// Reusable Input Component with validation
export const ValidatedInput = ({
  name,
  label,
  type = 'text',
  placeholder,
  value,
  error,
  touched,
  onChange,
  onBlur,
  required = false,
  disabled = false,
  className = '',
  ...props
}) => {
  const { t } = useTranslation();
  const hasError = touched && error;

  return (
    <div className={`mb-4 ${className}`}>
      {label && (
        <label 
          htmlFor={name} 
          className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2"
        >
          {label}
          {required && <span className="text-red-500 ml-1">*</span>}
        </label>
      )}
      
      <input
        id={name}
        name={name}
        type={type}
        value={value || ''}
        placeholder={placeholder}
        onChange={(e) => onChange(name, e.target.value)}
        onBlur={() => onBlur(name)}
        disabled={disabled}
        className={`w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 transition-colors ${
          hasError
            ? 'border-red-500 focus:ring-red-500 focus:border-red-500'
            : 'border-gray-300 focus:ring-blue-500 focus:border-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white'
        } ${disabled ? 'bg-gray-100 cursor-not-allowed' : ''}`}
        {...props}
      />
      
      {hasError && (
        <p className="mt-1 text-sm text-red-600 dark:text-red-400">
          {error}
        </p>
      )}
    </div>
  );
};

// Reusable Select Component with validation
export const ValidatedSelect = ({
  name,
  label,
  options = [],
  value,
  error,
  touched,
  onChange,
  onBlur,
  required = false,
  disabled = false,
  placeholder = 'Select an option',
  className = ''
}) => {
  const hasError = touched && error;

  return (
    <div className={`mb-4 ${className}`}>
      {label && (
        <label 
          htmlFor={name} 
          className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2"
        >
          {label}
          {required && <span className="text-red-500 ml-1">*</span>}
        </label>
      )}
      
      <select
        id={name}
        name={name}
        value={value || ''}
        onChange={(e) => onChange(name, e.target.value)}
        onBlur={() => onBlur(name)}
        disabled={disabled}
        className={`w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 transition-colors ${
          hasError
            ? 'border-red-500 focus:ring-red-500 focus:border-red-500'
            : 'border-gray-300 focus:ring-blue-500 focus:border-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white'
        } ${disabled ? 'bg-gray-100 cursor-not-allowed' : ''}`}
      >
        <option value="">{placeholder}</option>
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
      
      {hasError && (
        <p className="mt-1 text-sm text-red-600 dark:text-red-400">
          {error}
        </p>
      )}
    </div>
  );
};

// Trading Form Component
export const TradingForm = ({ onSubmit, initialData = {} }) => {
  const { t } = useTranslation();
  
  const validationSchema = {
    symbol: ['required', { rule: 'stockSymbol' }],
    quantity: [
      'required', 
      'numeric', 
      'positive',
      { rule: 'range', params: [1, 10000] }
    ],
    price: [
      'required', 
      'numeric', 
      'positive',
      { rule: 'range', params: [0.01, 100000] }
    ],
    orderType: ['required'],
    timeInForce: ['required']
  };

  const {
    values,
    errors,
    touched,
    isSubmitting,
    handleChange,
    handleBlur,
    handleSubmit,
    resetForm
  } = useFormValidation({
    symbol: '',
    quantity: '',
    price: '',
    orderType: 'market',
    timeInForce: 'day',
    ...initialData
  }, validationSchema);

  const orderTypeOptions = [
    { value: 'market', label: t('marketOrder') },
    { value: 'limit', label: t('limitOrder') },
    { value: 'stop', label: t('stopOrder') },
    { value: 'stop_limit', label: t('stopLimitOrder') }
  ];

  const timeInForceOptions = [
    { value: 'day', label: t('day') },
    { value: 'gtc', label: t('goodTillCanceled') },
    { value: 'ioc', label: t('immediateOrCancel') },
    { value: 'fok', label: t('fillOrKill') }
  ];

  return (
    <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg">
      <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-6">
        {t('placeOrder')}
      </h3>
      
      <form onSubmit={(e) => {
        e.preventDefault();
        handleSubmit(onSubmit);
      }}>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <ValidatedInput
            name="symbol"
            label={t('stockSymbol')}
            placeholder="e.g., RELIANCE"
            value={values.symbol}
            error={errors.symbol}
            touched={touched.symbol}
            onChange={handleChange}
            onBlur={handleBlur}
            required
          />
          
          <ValidatedInput
            name="quantity"
            label={t('quantity')}
            type="number"
            placeholder="Enter quantity"
            value={values.quantity}
            error={errors.quantity}
            touched={touched.quantity}
            onChange={handleChange}
            onBlur={handleBlur}
            required
          />
          
          <ValidatedInput
            name="price"
            label={t('price')}
            type="number"
            step="0.01"
            placeholder="Enter price"
            value={values.price}
            error={errors.price}
            touched={touched.price}
            onChange={handleChange}
            onBlur={handleBlur}
            required
            disabled={values.orderType === 'market'}
          />
          
          <ValidatedSelect
            name="orderType"
            label={t('orderType')}
            options={orderTypeOptions}
            value={values.orderType}
            error={errors.orderType}
            touched={touched.orderType}
            onChange={handleChange}
            onBlur={handleBlur}
            required
          />
          
          <ValidatedSelect
            name="timeInForce"
            label={t('timeInForce')}
            options={timeInForceOptions}
            value={values.timeInForce}
            error={errors.timeInForce}
            touched={touched.timeInForce}
            onChange={handleChange}
            onBlur={handleBlur}
            required
          />
        </div>
        
        <div className="flex justify-between mt-6">
          <button
            type="button"
            onClick={resetForm}
            className="px-4 py-2 text-gray-600 bg-gray-200 rounded-lg hover:bg-gray-300 transition-colors dark:bg-gray-600 dark:text-gray-300 dark:hover:bg-gray-500"
            disabled={isSubmitting}
          >
            {t('reset')}
          </button>
          
          <button
            type="submit"
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            disabled={isSubmitting}
          >
            {isSubmitting ? t('placing') : t('placeOrder')}
          </button>
        </div>
      </form>
    </div>
  );
};

export default {
  useFormValidation,
  ValidatedInput,
  ValidatedSelect,
  TradingForm
};