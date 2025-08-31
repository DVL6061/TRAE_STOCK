import React, { useEffect, useState } from 'react';
import { useAuth } from '../contexts/AuthContext';

const ProtectedRoute = ({ 
  children, 
  requiredRole = null, 
  requiredPermissions = [], 
  fallback = null,
  redirectTo = '/auth',
  showLoading = true 
}) => {
  const { 
    user, 
    isAuthenticated, 
    isLoading, 
    hasRole, 
    hasPermission, 
    hasAnyPermission,
    validateToken 
  } = useAuth();
  
  const [isValidating, setIsValidating] = useState(true);
  const [validationError, setValidationError] = useState(null);

  // Validate token on mount and periodically
  useEffect(() => {
    const validateUserToken = async () => {
      try {
        setIsValidating(true);
        setValidationError(null);
        
        if (isAuthenticated) {
          const isValid = await validateToken();
          if (!isValid) {
            setValidationError('Session expired. Please log in again.');
            setTimeout(() => {
              window.location.href = redirectTo;
            }, 2000);
            return;
          }
        }
      } catch (error) {
        console.error('Token validation error:', error);
        setValidationError('Authentication error. Please log in again.');
        setTimeout(() => {
          window.location.href = redirectTo;
        }, 2000);
      } finally {
        setIsValidating(false);
      }
    };

    validateUserToken();

    // Set up periodic token validation (every 5 minutes)
    const interval = setInterval(validateUserToken, 5 * 60 * 1000);
    
    return () => clearInterval(interval);
  }, [isAuthenticated, validateToken, redirectTo]);

  // Check if user has required role
  const hasRequiredRole = () => {
    if (!requiredRole) return true;
    return hasRole(requiredRole);
  };

  // Check if user has required permissions
  const hasRequiredPermissions = () => {
    if (!requiredPermissions || requiredPermissions.length === 0) return true;
    
    // Check if user has ALL required permissions
    return requiredPermissions.every(permission => hasPermission(permission));
  };

  // Loading state
  if (isLoading || isValidating) {
    if (!showLoading) return null;
    
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="loading-spinner mx-auto mb-4"></div>
          <h2 className="text-lg font-semibold text-gray-900 mb-2">Loading...</h2>
          <p className="text-gray-600">Verifying your authentication status</p>
        </div>
        
        <style jsx>{`
          .loading-spinner {
            border: 3px solid #f3f4f6;
            border-top: 3px solid #3b82f6;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
          }
          
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
        `}</style>
      </div>
    );
  }

  // Validation error state
  if (validationError) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="max-w-md w-full">
          <div className="bg-white shadow-lg rounded-lg p-6 text-center">
            <div className="mx-auto h-12 w-12 bg-red-100 rounded-full flex items-center justify-center mb-4">
              <svg className="h-6 w-6 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <h2 className="text-lg font-semibold text-gray-900 mb-2">Authentication Error</h2>
            <p className="text-gray-600 mb-4">{validationError}</p>
            <div className="loading-spinner mx-auto mb-2"></div>
            <p className="text-sm text-gray-500">Redirecting to login...</p>
          </div>
        </div>
        
        <style jsx>{`
          .loading-spinner {
            border: 2px solid #f3f4f6;
            border-top: 2px solid #ef4444;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            animation: spin 1s linear infinite;
          }
          
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
        `}</style>
      </div>
    );
  }

  // Not authenticated
  if (!isAuthenticated || !user) {
    if (fallback) {
      return fallback;
    }
    
    // Redirect to auth page
    window.location.href = redirectTo;
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="loading-spinner mx-auto mb-4"></div>
          <p className="text-gray-600">Redirecting to login...</p>
        </div>
        
        <style jsx>{`
          .loading-spinner {
            border: 2px solid #f3f4f6;
            border-top: 2px solid #3b82f6;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            animation: spin 1s linear infinite;
          }
          
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
        `}</style>
      </div>
    );
  }

  // Check role requirements
  if (!hasRequiredRole()) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="max-w-md w-full">
          <div className="bg-white shadow-lg rounded-lg p-6 text-center">
            <div className="mx-auto h-12 w-12 bg-yellow-100 rounded-full flex items-center justify-center mb-4">
              <svg className="h-6 w-6 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16c-.77.833.192 2.5 1.732 2.5z" />
              </svg>
            </div>
            <h2 className="text-lg font-semibold text-gray-900 mb-2">Access Denied</h2>
            <p className="text-gray-600 mb-4">
              You don't have the required role ({requiredRole}) to access this page.
            </p>
            <div className="space-y-2">
              <button
                onClick={() => window.history.back()}
                className="w-full px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500"
              >
                Go Back
              </button>
              <button
                onClick={() => window.location.href = '/dashboard'}
                className="w-full px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                Go to Dashboard
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Check permission requirements
  if (!hasRequiredPermissions()) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="max-w-md w-full">
          <div className="bg-white shadow-lg rounded-lg p-6 text-center">
            <div className="mx-auto h-12 w-12 bg-red-100 rounded-full flex items-center justify-center mb-4">
              <svg className="h-6 w-6 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728L5.636 5.636m12.728 12.728L18.364 5.636M5.636 18.364l12.728-12.728" />
              </svg>
            </div>
            <h2 className="text-lg font-semibold text-gray-900 mb-2">Insufficient Permissions</h2>
            <p className="text-gray-600 mb-2">
              You don't have the required permissions to access this page.
            </p>
            <p className="text-sm text-gray-500 mb-4">
              Required: {requiredPermissions.join(', ')}
            </p>
            <div className="space-y-2">
              <button
                onClick={() => window.history.back()}
                className="w-full px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500"
              >
                Go Back
              </button>
              <button
                onClick={() => window.location.href = '/dashboard'}
                className="w-full px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                Go to Dashboard
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // All checks passed, render children
  return children;
};

// Higher-order component for easier usage
export const withProtectedRoute = (Component, options = {}) => {
  return function ProtectedComponent(props) {
    return (
      <ProtectedRoute {...options}>
        <Component {...props} />
      </ProtectedRoute>
    );
  };
};

// Specific role-based components
export const AdminRoute = ({ children, ...props }) => (
  <ProtectedRoute requiredRole="admin" {...props}>
    {children}
  </ProtectedRoute>
);

export const ModeratorRoute = ({ children, ...props }) => (
  <ProtectedRoute requiredRole="moderator" {...props}>
    {children}
  </ProtectedRoute>
);

export const PremiumRoute = ({ children, ...props }) => (
  <ProtectedRoute requiredPermissions={['premium_features']} {...props}>
    {children}
  </ProtectedRoute>
);

export const TradingRoute = ({ children, ...props }) => (
  <ProtectedRoute requiredPermissions={['trading_access']} {...props}>
    {children}
  </ProtectedRoute>
);

export default ProtectedRoute;