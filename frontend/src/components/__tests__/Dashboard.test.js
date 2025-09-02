import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { ThemeProvider } from '../../contexts/ThemeContext';
import Dashboard from '../../pages/Dashboard.jsx';
import axios from 'axios';

// Mock axios
jest.mock('axios');
const mockedAxios = axios;

// Test wrapper component
const TestWrapper = ({ children }) => (
  <BrowserRouter>
    <ThemeProvider>
      {children}
    </ThemeProvider>
  </BrowserRouter>
);

describe('Dashboard Component', () => {
  beforeEach(() => {
    // Reset mocks before each test
    jest.clearAllMocks();
    
    // Mock successful API responses
    mockedAxios.get.mockImplementation((url) => {
      if (url.includes('/api/v1/market/overview')) {
        return Promise.resolve({
          data: {
            totalStocks: 150,
            gainers: 85,
            losers: 45,
            unchanged: 20,
            topGainers: [
              { symbol: 'RELIANCE', change: 2.5, changePercent: 1.2 },
              { symbol: 'TCS', change: 45.0, changePercent: 1.8 },
            ],
            topLosers: [
              { symbol: 'HDFC', change: -15.5, changePercent: -0.8 },
            ],
            marketIndices: {
              nifty50: { value: 19500, change: 125.5, changePercent: 0.65 },
              sensex: { value: 65200, change: 420.8, changePercent: 0.65 },
            },
          },
        });
      }
      if (url.includes('/api/v1/predictions/trending')) {
        return Promise.resolve({
          data: {
            predictions: [
              {
                symbol: 'RELIANCE',
                predictedPrice: 2500.50,
                confidence: 0.85,
                direction: 'up',
                timeframe: '1d',
              },
            ],
          },
        });
      }
      if (url.includes('/api/v1/news/latest')) {
        return Promise.resolve({
          data: {
            news: global.mockNewsData,
          },
        });
      }
      return Promise.resolve({ data: {} });
    });
  });

  test('renders dashboard with loading state initially', () => {
    render(
      <TestWrapper>
        <Dashboard />
      </TestWrapper>
    );

    // Should show loading indicators
    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });

  test('displays market overview data after loading', async () => {
    render(
      <TestWrapper>
        <Dashboard />
      </TestWrapper>
    );

    // Wait for data to load
    await waitFor(() => {
      expect(screen.getByText('150')).toBeInTheDocument(); // Total stocks
      expect(screen.getByText('85')).toBeInTheDocument(); // Gainers
      expect(screen.getByText('45')).toBeInTheDocument(); // Losers
    });

    // Check market indices
    expect(screen.getByText('19,500')).toBeInTheDocument(); // Nifty
    expect(screen.getByText('65,200')).toBeInTheDocument(); // Sensex
  });

  test('displays top gainers and losers', async () => {
    render(
      <TestWrapper>
        <Dashboard />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText('RELIANCE')).toBeInTheDocument();
      expect(screen.getByText('TCS')).toBeInTheDocument();
      expect(screen.getByText('HDFC')).toBeInTheDocument();
    });

    // Check percentage changes
    expect(screen.getByText('+1.2%')).toBeInTheDocument();
    expect(screen.getByText('+1.8%')).toBeInTheDocument();
    expect(screen.getByText('-0.8%')).toBeInTheDocument();
  });

  test('displays trending predictions', async () => {
    render(
      <TestWrapper>
        <Dashboard />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText('₹2,500.50')).toBeInTheDocument();
      expect(screen.getByText('85%')).toBeInTheDocument(); // Confidence
    });
  });

  test('displays latest news with sentiment', async () => {
    render(
      <TestWrapper>
        <Dashboard />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText('Market Update: Positive Trends Continue')).toBeInTheDocument();
      expect(screen.getByText('Tech Stocks Rally Amid Strong Earnings')).toBeInTheDocument();
    });

    // Check sentiment indicators
    expect(screen.getAllByText('Positive')).toHaveLength(2);
  });

  test('handles API errors gracefully', async () => {
    // Mock API error
    mockedAxios.get.mockRejectedValue(new Error('API Error'));

    render(
      <TestWrapper>
        <Dashboard />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText(/error loading data/i)).toBeInTheDocument();
    });
  });

  test('refreshes data when refresh button is clicked', async () => {
    render(
      <TestWrapper>
        <Dashboard />
      </TestWrapper>
    );

    // Wait for initial load
    await waitFor(() => {
      expect(screen.getByText('150')).toBeInTheDocument();
    });

    // Find and click refresh button
    const refreshButton = screen.getByRole('button', { name: /refresh/i });
    fireEvent.click(refreshButton);

    // Should make API calls again
    await waitFor(() => {
      expect(mockedAxios.get).toHaveBeenCalledTimes(6); // 3 initial + 3 refresh
    });
  });

  test('navigates to stock detail when stock is clicked', async () => {
    const mockNavigate = jest.fn();
    jest.doMock('react-router-dom', () => ({
      ...jest.requireActual('react-router-dom'),
      useNavigate: () => mockNavigate,
    }));

    render(
      <TestWrapper>
        <Dashboard />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText('RELIANCE')).toBeInTheDocument();
    });

    // Click on stock
    fireEvent.click(screen.getByText('RELIANCE'));

    // Should navigate to stock detail
    expect(mockNavigate).toHaveBeenCalledWith('/stock/RELIANCE');
  });

  test('displays correct time-based greeting', () => {
    // Mock different times
    const originalDate = Date;
    
    // Morning test
    global.Date = class extends Date {
      getHours() {
        return 9; // 9 AM
      }
    };

    render(
      <TestWrapper>
        <Dashboard />
      </TestWrapper>
    );

    expect(screen.getByText(/good morning/i)).toBeInTheDocument();

    // Restore original Date
    global.Date = originalDate;
  });

  test('updates real-time data via WebSocket', async () => {
    render(
      <TestWrapper>
        <Dashboard />
      </TestWrapper>
    );

    // Wait for initial load
    await waitFor(() => {
      expect(screen.getByText('150')).toBeInTheDocument();
    });

    // Simulate WebSocket message
    const mockWebSocket = new WebSocket('ws://localhost:8000/ws');
    mockWebSocket.onmessage({
      data: JSON.stringify({
        type: 'market_update',
        data: {
          symbol: 'RELIANCE',
          price: 2460.25,
          change: 25.75,
          changePercent: 1.06,
        },
      }),
    });

    // Should update the display
    await waitFor(() => {
      expect(screen.getByText('₹2,460.25')).toBeInTheDocument();
    });
  });

  test('filters stocks by search query', async () => {
    render(
      <TestWrapper>
        <Dashboard />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText('RELIANCE')).toBeInTheDocument();
      expect(screen.getByText('TCS')).toBeInTheDocument();
    });

    // Find search input and type
    const searchInput = screen.getByPlaceholderText(/search stocks/i);
    fireEvent.change(searchInput, { target: { value: 'REL' } });

    // Should filter results
    await waitFor(() => {
      expect(screen.getByText('RELIANCE')).toBeInTheDocument();
      expect(screen.queryByText('TCS')).not.toBeInTheDocument();
    });
  });

  test('toggles between different chart timeframes', async () => {
    render(
      <TestWrapper>
        <Dashboard />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText('1D')).toBeInTheDocument();
    });

    // Click on different timeframe
    fireEvent.click(screen.getByText('1W'));

    // Should update chart data
    await waitFor(() => {
      expect(mockedAxios.get).toHaveBeenCalledWith(
        expect.stringContaining('timeframe=1w')
      );
    });
  });
});