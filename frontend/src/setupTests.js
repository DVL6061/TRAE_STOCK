// jest-dom adds custom jest matchers for asserting on DOM nodes.
// allows you to do things like:
// expect(element).toHaveTextContent(/react/i)
// learn more: https://github.com/testing-library/jest-dom
import '@testing-library/jest-dom';

// Mock WebSocket for testing
global.WebSocket = class WebSocket {
  constructor(url) {
    this.url = url;
    this.readyState = WebSocket.CONNECTING;
    setTimeout(() => {
      this.readyState = WebSocket.OPEN;
      if (this.onopen) this.onopen();
    }, 100);
  }

  send(data) {
    // Mock send functionality
    console.log('Mock WebSocket send:', data);
  }

  close() {
    this.readyState = WebSocket.CLOSED;
    if (this.onclose) this.onclose();
  }

  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;
};

// Mock Chart.js for testing
jest.mock('chart.js', () => ({
  Chart: {
    register: jest.fn(),
  },
  CategoryScale: jest.fn(),
  LinearScale: jest.fn(),
  PointElement: jest.fn(),
  LineElement: jest.fn(),
  Title: jest.fn(),
  Tooltip: jest.fn(),
  Legend: jest.fn(),
  BarElement: jest.fn(),
  ArcElement: jest.fn(),
}));

// Mock react-chartjs-2
jest.mock('react-chartjs-2', () => ({
  Line: ({ data, options }) => <div data-testid="line-chart" data-chart-data={JSON.stringify(data)} />,
  Bar: ({ data, options }) => <div data-testid="bar-chart" data-chart-data={JSON.stringify(data)} />,
  Pie: ({ data, options }) => <div data-testid="pie-chart" data-chart-data={JSON.stringify(data)} />,
  Doughnut: ({ data, options }) => <div data-testid="doughnut-chart" data-chart-data={JSON.stringify(data)} />,
}));

// Mock react-router-dom
jest.mock('react-router-dom', () => ({
  ...jest.requireActual('react-router-dom'),
  useNavigate: () => jest.fn(),
  useLocation: () => ({ pathname: '/' }),
  useParams: () => ({}),
}));

// Mock axios
jest.mock('axios', () => ({
  create: jest.fn(() => ({
    get: jest.fn(() => Promise.resolve({ data: {} })),
    post: jest.fn(() => Promise.resolve({ data: {} })),
    put: jest.fn(() => Promise.resolve({ data: {} })),
    delete: jest.fn(() => Promise.resolve({ data: {} })),
    interceptors: {
      request: { use: jest.fn() },
      response: { use: jest.fn() },
    },
  })),
  get: jest.fn(() => Promise.resolve({ data: {} })),
  post: jest.fn(() => Promise.resolve({ data: {} })),
  put: jest.fn(() => Promise.resolve({ data: {} })),
  delete: jest.fn(() => Promise.resolve({ data: {} })),
}));

// Mock socket.io-client
jest.mock('socket.io-client', () => {
  const mockSocket = {
    on: jest.fn(),
    off: jest.fn(),
    emit: jest.fn(),
    connect: jest.fn(),
    disconnect: jest.fn(),
    connected: true,
  };
  return jest.fn(() => mockSocket);
});

// Mock react-toastify
jest.mock('react-toastify', () => ({
  toast: {
    success: jest.fn(),
    error: jest.fn(),
    info: jest.fn(),
    warning: jest.fn(),
  },
  ToastContainer: () => <div data-testid="toast-container" />,
}));

// Global test utilities
global.mockApiResponse = (data, status = 200) => ({
  data,
  status,
  statusText: 'OK',
  headers: {},
  config: {},
});

global.mockStockData = {
  symbol: 'RELIANCE',
  price: 2450.75,
  change: 15.25,
  changePercent: 0.63,
  volume: 1500000,
  marketCap: 1650000000000,
  pe: 25.4,
  high52w: 2856.15,
  low52w: 1885.00,
};

global.mockPredictionData = {
  symbol: 'RELIANCE',
  predictedPrice: 2500.50,
  confidence: 0.85,
  direction: 'up',
  timeframe: '1d',
  modelUsed: 'XGBoost',
  technicalIndicators: {
    rsi: 65.2,
    macd: 12.5,
    sma20: 2480.0,
  },
  shapValues: {
    featureImportance: {
      close: 0.3,
      volume: 0.2,
      rsi: 0.15,
    },
  },
};

global.mockNewsData = [
  {
    id: 1,
    title: 'Market Update: Positive Trends Continue',
    content: 'Stock market shows positive trends with strong performance...',
    sentiment: 'positive',
    sentimentScore: 0.7,
    source: 'Economic Times',
    publishedAt: '2024-01-15T10:30:00Z',
    url: 'https://example.com/news1',
    relatedSymbols: ['RELIANCE', 'TCS'],
    category: 'market',
    impact: 'high',
  },
  {
    id: 2,
    title: 'Tech Stocks Rally Amid Strong Earnings',
    content: 'Technology sector shows strong performance...',
    sentiment: 'positive',
    sentimentScore: 0.8,
    source: 'Moneycontrol',
    publishedAt: '2024-01-15T09:15:00Z',
    url: 'https://example.com/news2',
    relatedSymbols: ['TCS', 'INFY'],
    category: 'technology',
    impact: 'medium',
  },
];

// Suppress console warnings in tests
const originalError = console.error;
beforeAll(() => {
  console.error = (...args) => {
    if (
      typeof args[0] === 'string' &&
      args[0].includes('Warning: ReactDOM.render is no longer supported')
    ) {
      return;
    }
    originalError.call(console, ...args);
  };
});

afterAll(() => {
  console.error = originalError;
});