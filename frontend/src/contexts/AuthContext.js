import React, { createContext, useContext, useReducer, useEffect } from 'react';
import axios from 'axios';

// Auth Context
const AuthContext = createContext();

// Auth Actions
const AUTH_ACTIONS = {
  LOGIN_START: 'LOGIN_START',
  LOGIN_SUCCESS: 'LOGIN_SUCCESS',
  LOGIN_FAILURE: 'LOGIN_FAILURE',
  LOGOUT: 'LOGOUT',
  REGISTER_START: 'REGISTER_START',
  REGISTER_SUCCESS: 'REGISTER_SUCCESS',
  REGISTER_FAILURE: 'REGISTER_FAILURE',
  UPDATE_PROFILE: 'UPDATE_PROFILE',
  CLEAR_ERROR: 'CLEAR_ERROR',
  SET_LOADING: 'SET_LOADING'
};

// Initial State
const initialState = {
  user: null,
  token: localStorage.getItem('token'),
  isAuthenticated: false,
  isLoading: false,
  error: null,
  loginAttempts: 0,
  lastLoginAttempt: null
};

// Auth Reducer
const authReducer = (state, action) => {
  switch (action.type) {
    case AUTH_ACTIONS.LOGIN_START:
    case AUTH_ACTIONS.REGISTER_START:
      return {
        ...state,
        isLoading: true,
        error: null
      };
    
    case AUTH_ACTIONS.LOGIN_SUCCESS:
      return {
        ...state,
        user: action.payload.user,
        token: action.payload.token,
        isAuthenticated: true,
        isLoading: false,
        error: null,
        loginAttempts: 0,
        lastLoginAttempt: null
      };
    
    case AUTH_ACTIONS.REGISTER_SUCCESS:
      return {
        ...state,
        user: action.payload.user,
        token: action.payload.token,
        isAuthenticated: true,
        isLoading: false,
        error: null
      };
    
    case AUTH_ACTIONS.LOGIN_FAILURE:
      return {
        ...state,
        user: null,
        token: null,
        isAuthenticated: false,
        isLoading: false,
        error: action.payload,
        loginAttempts: state.loginAttempts + 1,
        lastLoginAttempt: new Date().toISOString()
      };
    
    case AUTH_ACTIONS.REGISTER_FAILURE:
      return {
        ...state,
        user: null,
        token: null,
        isAuthenticated: false,
        isLoading: false,
        error: action.payload
      };
    
    case AUTH_ACTIONS.LOGOUT:
      return {
        ...state,
        user: null,
        token: null,
        isAuthenticated: false,
        isLoading: false,
        error: null
      };
    
    case AUTH_ACTIONS.UPDATE_PROFILE:
      return {
        ...state,
        user: { ...state.user, ...action.payload },
        error: null
      };
    
    case AUTH_ACTIONS.CLEAR_ERROR:
      return {
        ...state,
        error: null
      };
    
    case AUTH_ACTIONS.SET_LOADING:
      return {
        ...state,
        isLoading: action.payload
      };
    
    default:
      return state;
  }
};

// Auth Provider Component
export const AuthProvider = ({ children }) => {
  const [state, dispatch] = useReducer(authReducer, initialState);

  // API Base URL
  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

  // Configure axios defaults
  useEffect(() => {
    if (state.token) {
      axios.defaults.headers.common['Authorization'] = `Bearer ${state.token}`;
      localStorage.setItem('token', state.token);
    } else {
      delete axios.defaults.headers.common['Authorization'];
      localStorage.removeItem('token');
    }
  }, [state.token]);

  // Check if user is authenticated on app load
  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      validateToken(token);
    }
  }, []);

  // Validate token with backend
  const validateToken = async (token) => {
    try {
      dispatch({ type: AUTH_ACTIONS.SET_LOADING, payload: true });
      
      const response = await axios.get(`${API_BASE_URL}/auth/validate`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      
      if (response.data.valid) {
        dispatch({
          type: AUTH_ACTIONS.LOGIN_SUCCESS,
          payload: {
            user: response.data.user,
            token: token
          }
        });
      } else {
        dispatch({ type: AUTH_ACTIONS.LOGOUT });
      }
    } catch (error) {
      console.error('Token validation failed:', error);
      dispatch({ type: AUTH_ACTIONS.LOGOUT });
    } finally {
      dispatch({ type: AUTH_ACTIONS.SET_LOADING, payload: false });
    }
  };

  // Login function
  const login = async (credentials) => {
    try {
      dispatch({ type: AUTH_ACTIONS.LOGIN_START });
      
      // Check rate limiting
      if (state.loginAttempts >= 5) {
        const lastAttempt = new Date(state.lastLoginAttempt);
        const now = new Date();
        const timeDiff = (now - lastAttempt) / 1000 / 60; // minutes
        
        if (timeDiff < 15) {
          throw new Error(`Too many login attempts. Please try again in ${Math.ceil(15 - timeDiff)} minutes.`);
        }
      }
      
      const response = await axios.post(`${API_BASE_URL}/auth/login`, credentials);
      
      const { user, access_token } = response.data;
      
      dispatch({
        type: AUTH_ACTIONS.LOGIN_SUCCESS,
        payload: {
          user,
          token: access_token
        }
      });
      
      return { success: true, user };
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message || 'Login failed';
      dispatch({
        type: AUTH_ACTIONS.LOGIN_FAILURE,
        payload: errorMessage
      });
      return { success: false, error: errorMessage };
    }
  };

  // Register function
  const register = async (userData) => {
    try {
      dispatch({ type: AUTH_ACTIONS.REGISTER_START });
      
      const response = await axios.post(`${API_BASE_URL}/auth/register`, userData);
      
      const { user, access_token } = response.data;
      
      dispatch({
        type: AUTH_ACTIONS.REGISTER_SUCCESS,
        payload: {
          user,
          token: access_token
        }
      });
      
      return { success: true, user };
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message || 'Registration failed';
      dispatch({
        type: AUTH_ACTIONS.REGISTER_FAILURE,
        payload: errorMessage
      });
      return { success: false, error: errorMessage };
    }
  };

  // Logout function
  const logout = async () => {
    try {
      // Call logout endpoint to invalidate token on server
      if (state.token) {
        await axios.post(`${API_BASE_URL}/auth/logout`);
      }
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      dispatch({ type: AUTH_ACTIONS.LOGOUT });
    }
  };

  // Update profile function
  const updateProfile = async (profileData) => {
    try {
      dispatch({ type: AUTH_ACTIONS.SET_LOADING, payload: true });
      
      const response = await axios.put(`${API_BASE_URL}/auth/profile`, profileData);
      
      dispatch({
        type: AUTH_ACTIONS.UPDATE_PROFILE,
        payload: response.data.user
      });
      
      return { success: true, user: response.data.user };
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message || 'Profile update failed';
      return { success: false, error: errorMessage };
    } finally {
      dispatch({ type: AUTH_ACTIONS.SET_LOADING, payload: false });
    }
  };

  // Change password function
  const changePassword = async (passwordData) => {
    try {
      dispatch({ type: AUTH_ACTIONS.SET_LOADING, payload: true });
      
      const response = await axios.put(`${API_BASE_URL}/auth/change-password`, passwordData);
      
      return { success: true, message: response.data.message };
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message || 'Password change failed';
      return { success: false, error: errorMessage };
    } finally {
      dispatch({ type: AUTH_ACTIONS.SET_LOADING, payload: false });
    }
  };

  // Forgot password function
  const forgotPassword = async (email) => {
    try {
      dispatch({ type: AUTH_ACTIONS.SET_LOADING, payload: true });
      
      const response = await axios.post(`${API_BASE_URL}/auth/forgot-password`, { email });
      
      return { success: true, message: response.data.message };
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message || 'Password reset request failed';
      return { success: false, error: errorMessage };
    } finally {
      dispatch({ type: AUTH_ACTIONS.SET_LOADING, payload: false });
    }
  };

  // Reset password function
  const resetPassword = async (token, newPassword) => {
    try {
      dispatch({ type: AUTH_ACTIONS.SET_LOADING, payload: true });
      
      const response = await axios.post(`${API_BASE_URL}/auth/reset-password`, {
        token,
        new_password: newPassword
      });
      
      return { success: true, message: response.data.message };
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message || 'Password reset failed';
      return { success: false, error: errorMessage };
    } finally {
      dispatch({ type: AUTH_ACTIONS.SET_LOADING, payload: false });
    }
  };

  // Clear error function
  const clearError = () => {
    dispatch({ type: AUTH_ACTIONS.CLEAR_ERROR });
  };

  // Check if user has specific permission
  const hasPermission = (permission) => {
    if (!state.user || !state.user.permissions) return false;
    return state.user.permissions.includes(permission);
  };

  // Check if user has specific role
  const hasRole = (role) => {
    if (!state.user || !state.user.role) return false;
    return state.user.role === role;
  };

  // Get user preferences
  const getUserPreferences = () => {
    return state.user?.preferences || {};
  };

  // Update user preferences
  const updatePreferences = async (preferences) => {
    try {
      const response = await axios.put(`${API_BASE_URL}/auth/preferences`, preferences);
      
      dispatch({
        type: AUTH_ACTIONS.UPDATE_PROFILE,
        payload: { preferences: response.data.preferences }
      });
      
      return { success: true, preferences: response.data.preferences };
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message || 'Preferences update failed';
      return { success: false, error: errorMessage };
    }
  };

  // Context value
  const value = {
    // State
    user: state.user,
    token: state.token,
    isAuthenticated: state.isAuthenticated,
    isLoading: state.isLoading,
    error: state.error,
    loginAttempts: state.loginAttempts,
    
    // Actions
    login,
    register,
    logout,
    updateProfile,
    changePassword,
    forgotPassword,
    resetPassword,
    clearError,
    
    // Utilities
    hasPermission,
    hasRole,
    getUserPreferences,
    updatePreferences
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

// Custom hook to use auth context
export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

// HOC for protected components
export const withAuth = (WrappedComponent) => {
  return function AuthenticatedComponent(props) {
    const { isAuthenticated, isLoading } = useAuth();
    
    if (isLoading) {
      return (
        <div className="flex items-center justify-center min-h-screen">
          <div className="loading-spinner"></div>
          <span className="ml-2">Loading...</span>
        </div>
      );
    }
    
    if (!isAuthenticated) {
      return (
        <div className="flex items-center justify-center min-h-screen">
          <div className="text-center">
            <h2 className="text-xl font-semibold mb-4">Authentication Required</h2>
            <p className="text-gray-600 mb-4">Please log in to access this page.</p>
            <button 
              onClick={() => window.location.href = '/login'}
              className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
            >
              Go to Login
            </button>
          </div>
        </div>
      );
    }
    
    return <WrappedComponent {...props} />;
  };
};

// Permission-based component wrapper
export const withPermission = (permission) => (WrappedComponent) => {
  return function PermissionComponent(props) {
    const { hasPermission, isAuthenticated } = useAuth();
    
    if (!isAuthenticated || !hasPermission(permission)) {
      return (
        <div className="flex items-center justify-center min-h-screen">
          <div className="text-center">
            <h2 className="text-xl font-semibold mb-4">Access Denied</h2>
            <p className="text-gray-600">You don't have permission to access this resource.</p>
          </div>
        </div>
      );
    }
    
    return <WrappedComponent {...props} />;
  };
};

// Role-based component wrapper
export const withRole = (role) => (WrappedComponent) => {
  return function RoleComponent(props) {
    const { hasRole, isAuthenticated } = useAuth();
    
    if (!isAuthenticated || !hasRole(role)) {
      return (
        <div className="flex items-center justify-center min-h-screen">
          <div className="text-center">
            <h2 className="text-xl font-semibold mb-4">Access Denied</h2>
            <p className="text-gray-600">You don't have the required role to access this resource.</p>
          </div>
        </div>
      );
    }
    
    return <WrappedComponent {...props} />;
  };
};

export default AuthContext;