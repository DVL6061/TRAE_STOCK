import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { ThemeProvider } from '../../contexts/ThemeContext';
import PredictionPage from '../../pages/PredictionPage';
import { useTranslation } from 'react-i18next';
import axios from 'axios';

// Mock axios
jest.mock('axios');
const mockedAxios = axios;

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

// Mock Chart.js
jest.mock('react-chartjs-2', () => ({
  Line: ({ data, options }) => (
    <div data-testid="prediction-chart">
      Chart: {data.datasets[0].label}
    </div>
  ),
  Bar: ({ data, options }) => (
    <div data-testid="confidence-chart">
      Bar Chart: {data.datasets[0].label}
    </div>
  ),
}));

// Test wrapper component
const TestWrapper = ({ children }) => (
  <BrowserRouter>
    <ThemeProvider>
      {children}
    </ThemeProvider>
  </BrowserRouter>
);

// Mock prediction data
const mockPredictionData = {
  symbol: 'RELIANCE',
  currentPrice: 2450.75,
  predictions: {
    '1d': {
      price: 2465.30,
      change: 14.55,
      changePercent: 0.59,
      confidence: 0.85,
      signal: 'BUY',
      factors: [
        { name: 'Technical Analysis', weight: 0.4, impact: 'positive' },
        { name: 'Market Sentiment', weight: 0.3, impact: 'positive' },
        { name: 'Volume Analysis', weight: 0.3, impact: 'neutral' }
      ]
    },
    '1w': {
      price: 2520.40,
      change: 69.65,
      changePercent: 2.84,
      confidence: 0.78,
      signal: 'BUY',
      factors: [
        { name: 'Earnings Forecast', weight: 0.5, impact: 'positive' },
        { name: 'Sector Performance', weight: 0.3, impact: 'positive' },
        { name: 'Economic Indicators', weight: 0.2, impact: 'neutral' }
      ]
    },
    '1m': {
      price: 2380.20,
      change: -70.55,
      changePercent: -2.88,
      confidence: 0.65,
      signal: 'HOLD',
      factors: [
        { name: 'Market Volatility', weight: 0.4, impact: 'negative' },
        { name: 'Global Events', weight: 0.3, impact: 'negative' },
        { name: 'Company Fundamentals', weight: 0.3, impact: 'positive' }
      ]
    }
  },
  technicalIndicators: {
    rsi: 68.5,
    macd: 12.3,
    ema20: 2435.60,
    sma50: 2420.80,
    bollinger: {
      upper: 2480.30,
      middle: 2450.75,
      lower: 2421.20
    }
  },
  modelAccuracy: {
    xgboost: 0.87,
    informer: 0.82,
    dqn: 0.79
  },
  riskMetrics: {
    volatility: 0.24,
    sharpeRatio: 1.45,
    maxDrawdown: 0.12,
    beta: 1.15
  },
  newsImpact: {
    positive: 65,
    negative: 20,
    neutral: 15,
    overallSentiment: 'positive'
  }
};

const mockStockList = [
  { symbol: 'RELIANCE', name: 'Reliance Industries Ltd' },
  { symbol: 'TCS', name: 'Tata Consultancy Services' },
  { symbol: 'INFY', name: 'Infosys Limited' },
  { symbol: 'HDFCBANK', name: 'HDFC Bank Limited' }
];

describe('PredictionPage Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockedAxios.get.mockImplementation((url) => {
      if (url.includes('/api/predictions/')) {
        return Promise.resolve({ data: mockPredictionData });
      }
      if (url.includes('/api/stocks/search')) {
        return Promise.resolve({ data: { stocks: mockStockList } });
      }
      return Promise.reject(new Error('Unknown endpoint'));
    });
  });

  test('renders predictions page with stock selector', async () => {
    render(
      <TestWrapper>
        <PredictionPage />
      </TestWrapper>
    );

    // Check header
    expect(screen.getByText('predictions.title')).toBeInTheDocument();
    expect(screen.getByText('predictions.subtitle')).toBeInTheDocument();

    // Check stock selector
    expect(screen.getByPlaceholderText('predictions.selectStock')).toBeInTheDocument();
  });

  test('displays prediction data after stock selection', async () => {
    render(
      <TestWrapper>
        <PredictionPage />
      </TestWrapper>
    );

    // Select a stock
    const stockInput = screen.getByPlaceholderText('predictions.selectStock');
    fireEvent.change(stockInput, { target: { value: 'RELIANCE' } });
    fireEvent.keyDown(stockInput, { key: 'Enter' });

    // Wait for predictions to load
    await waitFor(() => {
      expect(screen.getByText('RELIANCE')).toBeInTheDocument();
      expect(screen.getByText('₹2,450.75')).toBeInTheDocument();
    });
  });

  test('shows loading state while fetching predictions', () => {
    render(
      <TestWrapper>
        <PredictionPage />
      </TestWrapper>
    );

    const stockInput = screen.getByPlaceholderText('predictions.selectStock');
    fireEvent.change(stockInput, { target: { value: 'RELIANCE' } });
    fireEvent.keyDown(stockInput, { key: 'Enter' });

    expect(screen.getByTestId('predictions-loading')).toBeInTheDocument();
  });

  test('displays prediction timeframes', async () => {
    render(
      <TestWrapper>
        <PredictionPage />
      </TestWrapper>
    );

    const stockInput = screen.getByPlaceholderText('predictions.selectStock');
    fireEvent.change(stockInput, { target: { value: 'RELIANCE' } });
    fireEvent.keyDown(stockInput, { key: 'Enter' });

    await waitFor(() => {
      expect(screen.getByText('predictions.1day')).toBeInTheDocument();
      expect(screen.getByText('predictions.1week')).toBeInTheDocument();
      expect(screen.getByText('predictions.1month')).toBeInTheDocument();
    });
  });

  test('switches between prediction timeframes', async () => {
    render(
      <TestWrapper>
        <PredictionPage />
      </TestWrapper>
    );

    const stockInput = screen.getByPlaceholderText('predictions.selectStock');
    fireEvent.change(stockInput, { target: { value: 'RELIANCE' } });
    fireEvent.keyDown(stockInput, { key: 'Enter' });

    await waitFor(() => {
      expect(screen.getByText('₹2,465.30')).toBeInTheDocument(); // 1-day prediction
    });

    // Switch to 1-week prediction
    const weekTab = screen.getByText('predictions.1week');
    fireEvent.click(weekTab);

    expect(screen.getByText('₹2,520.40')).toBeInTheDocument();
  });

  test('displays buy/sell/hold signals', async () => {
    render(
      <TestWrapper>
        <PredictionPage />
      </TestWrapper>
    );

    const stockInput = screen.getByPlaceholderText('predictions.selectStock');
    fireEvent.change(stockInput, { target: { value: 'RELIANCE' } });
    fireEvent.keyDown(stockInput, { key: 'Enter' });

    await waitFor(() => {
      expect(screen.getByText('BUY')).toBeInTheDocument();
      expect(screen.getByTestId('signal-indicator')).toHaveClass('bg-green-500');
    });
  });

  test('shows confidence levels', async () => {
    render(
      <TestWrapper>
        <PredictionPage />
      </TestWrapper>
    );

    const stockInput = screen.getByPlaceholderText('predictions.selectStock');
    fireEvent.change(stockInput, { target: { value: 'RELIANCE' } });
    fireEvent.keyDown(stockInput, { key: 'Enter' });

    await waitFor(() => {
      expect(screen.getByText('85%')).toBeInTheDocument(); // Confidence level
      expect(screen.getByTestId('confidence-bar')).toBeInTheDocument();
    });
  });

  test('displays technical indicators', async () => {
    render(
      <TestWrapper>
        <PredictionsPage />
      </TestWrapper>
    );

    const stockInput = screen.getByPlaceholderText('predictions.selectStock');
    fireEvent.change(stockInput, { target: { value: 'RELIANCE' } });
    fireEvent.keyDown(stockInput, { key: 'Enter' });

    await waitFor(() => {
      expect(screen.getByText('predictions.technicalIndicators')).toBeInTheDocument();
      expect(screen.getByText('68.5')).toBeInTheDocument(); // RSI
      expect(screen.getByText('12.3')).toBeInTheDocument(); // MACD
    });
  });

  test('shows prediction factors and weights', async () => {
    render(
      <TestWrapper>
        <PredictionsPage />
      </TestWrapper>
    );

    const stockInput = screen.getByPlaceholderText('predictions.selectStock');
    fireEvent.change(stockInput, { target: { value: 'RELIANCE' } });
    fireEvent.keyDown(stockInput, { key: 'Enter' });

    await waitFor(() => {
      expect(screen.getByText('Technical Analysis')).toBeInTheDocument();
      expect(screen.getByText('40%')).toBeInTheDocument(); // Weight
      expect(screen.getByText('Market Sentiment')).toBeInTheDocument();
    });
  });

  test('displays model accuracy metrics', async () => {
    render(
      <TestWrapper>
        <PredictionsPage />
      </TestWrapper>
    );

    const stockInput = screen.getByPlaceholderText('predictions.selectStock');
    fireEvent.change(stockInput, { target: { value: 'RELIANCE' } });
    fireEvent.keyDown(stockInput, { key: 'Enter' });

    await waitFor(() => {
      expect(screen.getByText('predictions.modelAccuracy')).toBeInTheDocument();
      expect(screen.getByText('87%')).toBeInTheDocument(); // XGBoost accuracy
      expect(screen.getByText('82%')).toBeInTheDocument(); // Informer accuracy
    });
  });

  test('shows real-time updates', async () => {
    render(
      <TestWrapper>
        <PredictionPage />
      </TestWrapper>
    );

    const stockInput = screen.getByPlaceholderText('predictions.selectStock');
    fireEvent.change(stockInput, { target: { value: 'RELIANCE' } });
    fireEvent.keyDown(stockInput, { key: 'Enter' });

    await waitFor(() => {
      expect(screen.getByText('predictions.riskMetrics')).toBeInTheDocument();
      expect(screen.getByText('0.24')).toBeInTheDocument(); // Volatility
      expect(screen.getByText('1.45')).toBeInTheDocument(); // Sharpe Ratio
    });
  });

  test('displays news sentiment impact', async () => {
    render(
      <TestWrapper>
        <PredictionsPage />
      </TestWrapper>
    );

    const stockInput = screen.getByPlaceholderText('predictions.selectStock');
    fireEvent.change(stockInput, { target: { value: 'RELIANCE' } });
    fireEvent.keyDown(stockInput, { key: 'Enter' });

    await waitFor(() => {
      expect(screen.getByText('predictions.newsImpact')).toBeInTheDocument();
      expect(screen.getByText('65%')).toBeInTheDocument(); // Positive sentiment
      expect(screen.getByText('20%')).toBeInTheDocument(); // Negative sentiment
    });
  });

  test('handles prediction comparison', async () => {
    render(
      <TestWrapper>
        <PredictionPage />
      </TestWrapper>
    );

    const stockInput = screen.getByPlaceholderText('predictions.selectStock');
    fireEvent.change(stockInput, { target: { value: 'RELIANCE' } });
    fireEvent.keyDown(stockInput, { key: 'Enter' });

    await waitFor(() => {
      expect(screen.getByTestId('prediction-chart')).toBeInTheDocument();
      expect(screen.getByTestId('confidence-chart')).toBeInTheDocument();
    });
  });

  test('handles API errors gracefully', async () => {
    mockedAxios.get.mockRejectedValue(new Error('API Error'));

    render(
      <TestWrapper>
        <PredictionsPage />
      </TestWrapper>
    );

    const stockInput = screen.getByPlaceholderText('predictions.selectStock');
    fireEvent.change(stockInput, { target: { value: 'RELIANCE' } });
    fireEvent.keyDown(stockInput, { key: 'Enter' });

    await waitFor(() => {
      expect(screen.getByText('predictions.errorLoading')).toBeInTheDocument();
    });
  });

  test('shows stock search suggestions', async () => {
    render(
      <TestWrapper>
        <PredictionPage />
      </TestWrapper>
    );

    const stockInput = screen.getByPlaceholderText('predictions.selectStock');
    fireEvent.change(stockInput, { target: { value: 'REL' } });
    fireEvent.focus(stockInput);

    await waitFor(() => {
      expect(screen.getByText('Reliance Industries Ltd')).toBeInTheDocument();
    });
  });

  test('updates predictions in real-time', async () => {
    render(
      <TestWrapper>
        <PredictionPage />
      </TestWrapper>
    );

    const stockInput = screen.getByPlaceholderText('predictions.selectStock');
    fireEvent.change(stockInput, { target: { value: 'RELIANCE' } });
    fireEvent.keyDown(stockInput, { key: 'Enter' });

    await waitFor(() => {
      expect(screen.getByText('₹2,450.75')).toBeInTheDocument();
    });

    // Simulate real-time update
    const updatedData = {
      ...mockPredictionData,
      currentPrice: 2455.30
    };
    mockedAxios.get.mockResolvedValue({ data: updatedData });

    // Trigger refresh
    const refreshButton = screen.getByTestId('refresh-predictions');
    fireEvent.click(refreshButton);

    await waitFor(() => {
      expect(screen.getByText('₹2,455.30')).toBeInTheDocument();
    });
  });

  test('exports prediction data', async () => {
    render(
      <TestWrapper>
        <PredictionPage />
      </TestWrapper>
    );

    const stockInput = screen.getByPlaceholderText('predictions.selectStock');
    fireEvent.change(stockInput, { target: { value: 'RELIANCE' } });
    fireEvent.keyDown(stockInput, { key: 'Enter' });

    await waitFor(() => {
      expect(screen.getByText('₹2,450.75')).toBeInTheDocument();
    });

    const exportButton = screen.getByText('predictions.export');
    fireEvent.click(exportButton);

    // Should trigger download (mocked)
    expect(screen.getByTestId('export-modal')).toBeInTheDocument();
  });

  test('handles mobile responsive layout', () => {
    // Mock mobile viewport
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 768,
    });

    render(
      <TestWrapper>
        <PredictionPage />
      </TestWrapper>
    );

    const container = screen.getByTestId('predictions-container');
    expect(container).toHaveClass('px-4', 'sm:px-6', 'lg:px-8');
  });

  test('shows SHAP explanations for predictions', async () => {
    render(
      <TestWrapper>
        <PredictionPage />
      </TestWrapper>
    );

    const stockInput = screen.getByPlaceholderText('predictions.selectStock');
    fireEvent.change(stockInput, { target: { value: 'RELIANCE' } });
    fireEvent.keyDown(stockInput, { key: 'Enter' });

    await waitFor(() => {
      expect(screen.getByText('predictions.explanation')).toBeInTheDocument();
    });

    const explanationButton = screen.getByText('predictions.viewExplanation');
    fireEvent.click(explanationButton);

    expect(screen.getByTestId('shap-explanation')).toBeInTheDocument();
  });

  test('compares multiple models', async () => {
    render(
      <TestWrapper>
        <PredictionPage />
      </TestWrapper>
    );

    const stockInput = screen.getByPlaceholderText('predictions.selectStock');
    fireEvent.change(stockInput, { target: { value: 'RELIANCE' } });
    fireEvent.keyDown(stockInput, { key: 'Enter' });

    await waitFor(() => {
      expect(screen.getByText('predictions.modelComparison')).toBeInTheDocument();
    });

    const compareButton = screen.getByText('predictions.compareModels');
    fireEvent.click(compareButton);

    expect(screen.getByText('XGBoost')).toBeInTheDocument();
    expect(screen.getByText('Informer')).toBeInTheDocument();
    expect(screen.getByText('DQN')).toBeInTheDocument();
  });

  test('sets price alerts', async () => {
    render(
      <TestWrapper>
        <PredictionPage />
      </TestWrapper>
    );

    const stockInput = screen.getByPlaceholderText('predictions.selectStock');
    fireEvent.change(stockInput, { target: { value: 'RELIANCE' } });
    fireEvent.keyDown(stockInput, { key: 'Enter' });

    await waitFor(() => {
      expect(screen.getByText('₹2,450.75')).toBeInTheDocument();
    });

    const alertButton = screen.getByText('predictions.setAlert');
    fireEvent.click(alertButton);

    expect(screen.getByTestId('alert-modal')).toBeInTheDocument();

    const alertInput = screen.getByPlaceholderText('predictions.alertPrice');
    fireEvent.change(alertInput, { target: { value: '2500' } });

    const saveAlertButton = screen.getByText('predictions.saveAlert');
    fireEvent.click(saveAlertButton);

    expect(mockedAxios.post).toHaveBeenCalledWith('/api/alerts', {
      symbol: 'RELIANCE',
      price: 2500,
      type: 'price'
    });
  });
});