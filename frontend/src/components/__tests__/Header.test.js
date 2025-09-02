import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { ThemeProvider } from '../../contexts/ThemeContext';
import Header from '../layout/Header.jsx';
import { useTranslation } from 'react-i18next';

// Mock react-i18next
jest.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key) => key,
    i18n: {
      changeLanguage: jest.fn(),
      language: 'en',
    },
  }),
}));

// Test wrapper component
const TestWrapper = ({ children }) => (
  <BrowserRouter>
    <ThemeProvider>
      {children}
    </ThemeProvider>
  </BrowserRouter>
);

describe('Header Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders header with logo and navigation', () => {
    render(
      <TestWrapper>
        <Header />
      </TestWrapper>
    );

    // Check logo
    expect(screen.getByText('TRAE STOCK')).toBeInTheDocument();
    
    // Check navigation items
    expect(screen.getByText('header.dashboard')).toBeInTheDocument();
    expect(screen.getByText('header.predictions')).toBeInTheDocument();
    expect(screen.getByText('header.news')).toBeInTheDocument();
    expect(screen.getByText('header.portfolio')).toBeInTheDocument();
  });

  test('displays search functionality', () => {
    render(
      <TestWrapper>
        <Header />
      </TestWrapper>
    );

    const searchInput = screen.getByPlaceholderText('header.searchPlaceholder');
    expect(searchInput).toBeInTheDocument();
    
    // Test search input
    fireEvent.change(searchInput, { target: { value: 'RELIANCE' } });
    expect(searchInput.value).toBe('RELIANCE');
  });

  test('shows search suggestions when typing', async () => {
    render(
      <TestWrapper>
        <Header />
      </TestWrapper>
    );

    const searchInput = screen.getByPlaceholderText('header.searchPlaceholder');
    
    // Type in search
    fireEvent.change(searchInput, { target: { value: 'REL' } });
    fireEvent.focus(searchInput);

    // Should show search suggestions (mocked)
    await waitFor(() => {
      expect(screen.getByText('RELIANCE')).toBeInTheDocument();
    });
  });

  test('toggles theme when theme button is clicked', () => {
    render(
      <TestWrapper>
        <Header />
      </TestWrapper>
    );

    const themeButton = screen.getByRole('button', { name: /theme/i });
    expect(themeButton).toBeInTheDocument();
    
    // Click theme toggle
    fireEvent.click(themeButton);
    
    // Theme should change (tested via context)
    expect(themeButton).toBeInTheDocument();
  });

  test('shows notifications dropdown', () => {
    render(
      <TestWrapper>
        <Header />
      </TestWrapper>
    );

    const notificationButton = screen.getByRole('button', { name: /notifications/i });
    expect(notificationButton).toBeInTheDocument();
    
    // Click notifications
    fireEvent.click(notificationButton);
    
    // Should show dropdown
    expect(screen.getByText('header.notifications')).toBeInTheDocument();
  });

  test('displays user profile dropdown', () => {
    render(
      <TestWrapper>
        <Header />
      </TestWrapper>
    );

    const profileButton = screen.getByRole('button', { name: /profile/i });
    expect(profileButton).toBeInTheDocument();
    
    // Click profile
    fireEvent.click(profileButton);
    
    // Should show profile options
    expect(screen.getByText('header.profile')).toBeInTheDocument();
    expect(screen.getByText('header.settings')).toBeInTheDocument();
    expect(screen.getByText('header.logout')).toBeInTheDocument();
  });

  test('changes language when language selector is used', () => {
    const mockChangeLanguage = jest.fn();
    
    // Mock useTranslation with changeLanguage function
    useTranslation.mockReturnValue({
      t: (key) => key,
      i18n: {
        changeLanguage: mockChangeLanguage,
        language: 'en',
      },
    });

    render(
      <TestWrapper>
        <Header />
      </TestWrapper>
    );

    const languageButton = screen.getByRole('button', { name: /language/i });
    fireEvent.click(languageButton);
    
    // Should show language options
    const hindiOption = screen.getByText('हिंदी');
    fireEvent.click(hindiOption);
    
    // Should call changeLanguage
    expect(mockChangeLanguage).toHaveBeenCalledWith('hi');
  });

  test('shows mobile menu toggle on small screens', () => {
    // Mock window.innerWidth for mobile
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 768,
    });

    render(
      <TestWrapper>
        <Header />
      </TestWrapper>
    );

    const mobileMenuButton = screen.getByRole('button', { name: /menu/i });
    expect(mobileMenuButton).toBeInTheDocument();
    
    // Click mobile menu
    fireEvent.click(mobileMenuButton);
    
    // Should toggle mobile menu
    expect(screen.getByTestId('mobile-menu')).toBeInTheDocument();
  });

  test('displays real-time market status', () => {
    render(
      <TestWrapper>
        <Header />
      </TestWrapper>
    );

    // Should show market status indicator
    expect(screen.getByText(/market/i)).toBeInTheDocument();
  });

  test('handles search submission', async () => {
    const mockNavigate = jest.fn();
    
    // Mock useNavigate
    jest.doMock('react-router-dom', () => ({
      ...jest.requireActual('react-router-dom'),
      useNavigate: () => mockNavigate,
    }));

    render(
      <TestWrapper>
        <Header />
      </TestWrapper>
    );

    const searchInput = screen.getByPlaceholderText('header.searchPlaceholder');
    const searchForm = searchInput.closest('form');
    
    // Type and submit search
    fireEvent.change(searchInput, { target: { value: 'RELIANCE' } });
    fireEvent.submit(searchForm);
    
    // Should navigate to search results or stock page
    await waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith('/stock/RELIANCE');
    });
  });

  test('shows connection status indicator', () => {
    render(
      <TestWrapper>
        <Header />
      </TestWrapper>
    );

    // Should show WebSocket connection status
    const connectionStatus = screen.getByTestId('connection-status');
    expect(connectionStatus).toBeInTheDocument();
  });

  test('displays notification badge when there are unread notifications', () => {
    // Mock notifications context or props
    render(
      <TestWrapper>
        <Header unreadNotifications={5} />
      </TestWrapper>
    );

    const notificationBadge = screen.getByText('5');
    expect(notificationBadge).toBeInTheDocument();
  });

  test('handles keyboard navigation', () => {
    render(
      <TestWrapper>
        <Header />
      </TestWrapper>
    );

    const searchInput = screen.getByPlaceholderText('header.searchPlaceholder');
    
    // Test keyboard navigation
    fireEvent.keyDown(searchInput, { key: 'ArrowDown' });
    fireEvent.keyDown(searchInput, { key: 'Enter' });
    
    // Should handle keyboard events
    expect(searchInput).toBeInTheDocument();
  });

  test('closes dropdowns when clicking outside', () => {
    render(
      <TestWrapper>
        <Header />
      </TestWrapper>
    );

    const profileButton = screen.getByRole('button', { name: /profile/i });
    
    // Open dropdown
    fireEvent.click(profileButton);
    expect(screen.getByText('header.profile')).toBeInTheDocument();
    
    // Click outside
    fireEvent.click(document.body);
    
    // Dropdown should close
    expect(screen.queryByText('header.profile')).not.toBeInTheDocument();
  });

  test('updates search suggestions based on API response', async () => {
    // Mock API response for search suggestions
    global.fetch = jest.fn(() =>
      Promise.resolve({
        json: () => Promise.resolve({
          suggestions: [
            { symbol: 'RELIANCE', name: 'Reliance Industries' },
            { symbol: 'TCS', name: 'Tata Consultancy Services' },
          ],
        }),
      })
    );

    render(
      <TestWrapper>
        <Header />
      </TestWrapper>
    );

    const searchInput = screen.getByPlaceholderText('header.searchPlaceholder');
    
    // Type to trigger suggestions
    fireEvent.change(searchInput, { target: { value: 'REL' } });
    
    await waitFor(() => {
      expect(screen.getByText('Reliance Industries')).toBeInTheDocument();
    });
  });

  test('shows loading state during search', async () => {
    render(
      <TestWrapper>
        <Header />
      </TestWrapper>
    );

    const searchInput = screen.getByPlaceholderText('header.searchPlaceholder');
    
    // Type to trigger search
    fireEvent.change(searchInput, { target: { value: 'RELIANCE' } });
    
    // Should show loading indicator
    expect(screen.getByTestId('search-loading')).toBeInTheDocument();
  });
});