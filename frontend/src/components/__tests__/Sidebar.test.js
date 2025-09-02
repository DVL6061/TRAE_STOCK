import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter, MemoryRouter } from 'react-router-dom';
import { ThemeProvider } from '../../contexts/ThemeContext';
import Sidebar from '../layout/Sidebar.jsx';
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
const TestWrapper = ({ children, initialEntries = ['/'] }) => (
  <MemoryRouter initialEntries={initialEntries}>
    <ThemeProvider>
      {children}
    </ThemeProvider>
  </MemoryRouter>
);

describe('Sidebar Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders sidebar with all navigation items', () => {
    render(
      <TestWrapper>
        <Sidebar isOpen={true} />
      </TestWrapper>
    );

    // Check main navigation items
    expect(screen.getByText('sidebar.dashboard')).toBeInTheDocument();
    expect(screen.getByText('sidebar.predictions')).toBeInTheDocument();
    expect(screen.getByText('sidebar.news')).toBeInTheDocument();
    expect(screen.getByText('sidebar.portfolio')).toBeInTheDocument();
    expect(screen.getByText('sidebar.watchlist')).toBeInTheDocument();
    expect(screen.getByText('sidebar.analytics')).toBeInTheDocument();
    expect(screen.getByText('sidebar.settings')).toBeInTheDocument();
  });

  test('highlights active navigation item', () => {
    render(
      <TestWrapper initialEntries={['/dashboard']}>
        <Sidebar isOpen={true} />
      </TestWrapper>
    );

    const dashboardLink = screen.getByText('sidebar.dashboard').closest('a');
    expect(dashboardLink).toHaveClass('bg-blue-600');
  });

  test('shows collapsed state when isOpen is false', () => {
    render(
      <TestWrapper>
        <Sidebar isOpen={false} />
      </TestWrapper>
    );

    const sidebar = screen.getByTestId('sidebar');
    expect(sidebar).toHaveClass('w-16');
    
    // Text should be hidden in collapsed state
    expect(screen.queryByText('sidebar.dashboard')).not.toBeVisible();
  });

  test('shows expanded state when isOpen is true', () => {
    render(
      <TestWrapper>
        <Sidebar isOpen={true} />
      </TestWrapper>
    );

    const sidebar = screen.getByTestId('sidebar');
    expect(sidebar).toHaveClass('w-64');
    
    // Text should be visible in expanded state
    expect(screen.getByText('sidebar.dashboard')).toBeVisible();
  });

  test('displays user profile section', () => {
    render(
      <TestWrapper>
        <Sidebar isOpen={true} />
      </TestWrapper>
    );

    expect(screen.getByText('John Doe')).toBeInTheDocument();
    expect(screen.getByText('Premium User')).toBeInTheDocument();
    expect(screen.getByAltText('User Avatar')).toBeInTheDocument();
  });

  test('shows portfolio summary', () => {
    render(
      <TestWrapper>
        <Sidebar isOpen={true} />
      </TestWrapper>
    );

    expect(screen.getByText('sidebar.portfolioValue')).toBeInTheDocument();
    expect(screen.getByText('₹12,45,678')).toBeInTheDocument();
    expect(screen.getByText('+2.34%')).toBeInTheDocument();
  });

  test('displays quick actions section', () => {
    render(
      <TestWrapper>
        <Sidebar isOpen={true} />
      </TestWrapper>
    );

    expect(screen.getByText('sidebar.quickActions')).toBeInTheDocument();
    expect(screen.getByText('sidebar.addStock')).toBeInTheDocument();
    expect(screen.getByText('sidebar.createAlert')).toBeInTheDocument();
    expect(screen.getByText('sidebar.viewReports')).toBeInTheDocument();
  });

  test('handles navigation clicks', () => {
    const mockNavigate = jest.fn();
    
    // Mock useNavigate
    jest.doMock('react-router-dom', () => ({
      ...jest.requireActual('react-router-dom'),
      useNavigate: () => mockNavigate,
    }));

    render(
      <TestWrapper>
        <Sidebar isOpen={true} />
      </TestWrapper>
    );

    const predictionsLink = screen.getByText('sidebar.predictions');
    fireEvent.click(predictionsLink);
    
    // Should navigate to predictions page
    expect(predictionsLink.closest('a')).toHaveAttribute('href', '/predictions');
  });

  test('shows market status indicator', () => {
    render(
      <TestWrapper>
        <Sidebar isOpen={true} />
      </TestWrapper>
    );

    expect(screen.getByText('sidebar.marketStatus')).toBeInTheDocument();
    expect(screen.getByText('Market Open')).toBeInTheDocument();
    expect(screen.getByTestId('market-status-indicator')).toBeInTheDocument();
  });

  test('displays recent activity section', () => {
    render(
      <TestWrapper>
        <Sidebar isOpen={true} />
      </TestWrapper>
    );

    expect(screen.getByText('sidebar.recentActivity')).toBeInTheDocument();
    expect(screen.getByText('RELIANCE bought')).toBeInTheDocument();
    expect(screen.getByText('TCS alert triggered')).toBeInTheDocument();
  });

  test('shows notification badges', () => {
    render(
      <TestWrapper>
        <Sidebar isOpen={true} notifications={{ alerts: 3, news: 5 }} />
      </TestWrapper>
    );

    const alertsBadge = screen.getByText('3');
    const newsBadge = screen.getByText('5');
    
    expect(alertsBadge).toBeInTheDocument();
    expect(newsBadge).toBeInTheDocument();
  });

  test('handles hover effects on navigation items', () => {
    render(
      <TestWrapper>
        <Sidebar isOpen={true} />
      </TestWrapper>
    );

    const dashboardItem = screen.getByText('sidebar.dashboard').closest('a');
    
    // Hover over item
    fireEvent.mouseEnter(dashboardItem);
    expect(dashboardItem).toHaveClass('hover:bg-gray-700');
    
    // Mouse leave
    fireEvent.mouseLeave(dashboardItem);
  });

  test('shows tooltips in collapsed mode', async () => {
    render(
      <TestWrapper>
        <Sidebar isOpen={false} />
      </TestWrapper>
    );

    const dashboardIcon = screen.getByTestId('dashboard-icon');
    
    // Hover to show tooltip
    fireEvent.mouseEnter(dashboardIcon);
    
    await waitFor(() => {
      expect(screen.getByText('sidebar.dashboard')).toBeInTheDocument();
    });
  });

  test('handles quick action clicks', () => {
    const mockOnAction = jest.fn();
    
    render(
      <TestWrapper>
        <Sidebar isOpen={true} onQuickAction={mockOnAction} />
      </TestWrapper>
    );

    const addStockButton = screen.getByText('sidebar.addStock');
    fireEvent.click(addStockButton);
    
    expect(mockOnAction).toHaveBeenCalledWith('addStock');
  });

  test('displays theme-appropriate styling', () => {
    render(
      <TestWrapper>
        <Sidebar isOpen={true} />
      </TestWrapper>
    );

    const sidebar = screen.getByTestId('sidebar');
    expect(sidebar).toHaveClass('bg-gray-900');
  });

  test('shows loading state for portfolio data', () => {
    render(
      <TestWrapper>
        <Sidebar isOpen={true} portfolioLoading={true} />
      </TestWrapper>
    );

    expect(screen.getByTestId('portfolio-loading')).toBeInTheDocument();
  });

  test('handles responsive behavior on mobile', () => {
    // Mock mobile viewport
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 768,
    });

    render(
      <TestWrapper>
        <Sidebar isOpen={true} isMobile={true} />
      </TestWrapper>
    );

    const sidebar = screen.getByTestId('sidebar');
    expect(sidebar).toHaveClass('fixed', 'inset-y-0', 'left-0', 'z-50');
  });

  test('shows overlay on mobile when open', () => {
    render(
      <TestWrapper>
        <Sidebar isOpen={true} isMobile={true} />
      </TestWrapper>
    );

    const overlay = screen.getByTestId('sidebar-overlay');
    expect(overlay).toBeInTheDocument();
  });

  test('closes sidebar when overlay is clicked on mobile', () => {
    const mockOnClose = jest.fn();
    
    render(
      <TestWrapper>
        <Sidebar isOpen={true} isMobile={true} onClose={mockOnClose} />
      </TestWrapper>
    );

    const overlay = screen.getByTestId('sidebar-overlay');
    fireEvent.click(overlay);
    
    expect(mockOnClose).toHaveBeenCalled();
  });

  test('displays keyboard shortcuts hints', () => {
    render(
      <TestWrapper>
        <Sidebar isOpen={true} showKeyboardShortcuts={true} />
      </TestWrapper>
    );

    expect(screen.getByText('Ctrl+D')).toBeInTheDocument();
    expect(screen.getByText('Ctrl+P')).toBeInTheDocument();
  });

  test('handles keyboard navigation', () => {
    render(
      <TestWrapper>
        <Sidebar isOpen={true} />
      </TestWrapper>
    );

    const firstNavItem = screen.getByText('sidebar.dashboard').closest('a');
    
    // Focus and navigate with keyboard
    firstNavItem.focus();
    fireEvent.keyDown(firstNavItem, { key: 'ArrowDown' });
    
    const secondNavItem = screen.getByText('sidebar.predictions').closest('a');
    expect(document.activeElement).toBe(secondNavItem);
  });

  test('shows connection status in sidebar', () => {
    render(
      <TestWrapper>
        <Sidebar isOpen={true} connectionStatus="connected" />
      </TestWrapper>
    );

    const connectionIndicator = screen.getByTestId('connection-indicator');
    expect(connectionIndicator).toHaveClass('bg-green-500');
  });

  test('displays version information', () => {
    render(
      <TestWrapper>
        <Sidebar isOpen={true} />
      </TestWrapper>
    );

    expect(screen.getByText('v1.0.0')).toBeInTheDocument();
  });

  test('handles logout functionality', () => {
    const mockLogout = jest.fn();
    
    render(
      <TestWrapper>
        <Sidebar isOpen={true} onLogout={mockLogout} />
      </TestWrapper>
    );

    const logoutButton = screen.getByText('sidebar.logout');
    fireEvent.click(logoutButton);
    
    expect(mockLogout).toHaveBeenCalled();
  });
});