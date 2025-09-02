import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { ThemeProvider } from '../../contexts/ThemeContext';
import NewsPage from '../../pages/NewsPage';
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
    <div data-testid="sentiment-chart">
      Chart: {data.datasets[0].label}
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

// Mock news data
const mockNewsData = [
  {
    id: 1,
    title: 'Reliance Industries Reports Strong Q4 Results',
    source: 'Economic Times',
    date: '2024-01-15T10:30:00Z',
    sentiment: 'positive',
    url: 'https://example.com/news1',
    impact: 'high',
    summary: 'Reliance Industries has reported exceptional quarterly results...',
    relatedStocks: ['RELIANCE', 'RIL'],
    category: 'earnings',
    sentimentScore: 0.85
  },
  {
    id: 2,
    title: 'Market Volatility Concerns Rise',
    source: 'Moneycontrol',
    date: '2024-01-14T15:45:00Z',
    sentiment: 'negative',
    url: 'https://example.com/news2',
    impact: 'medium',
    summary: 'Growing concerns about market volatility...',
    relatedStocks: ['NIFTY', 'SENSEX'],
    category: 'market',
    sentimentScore: -0.65
  },
  {
    id: 3,
    title: 'TCS Announces New AI Initiative',
    source: 'Business Standard',
    date: '2024-01-13T09:15:00Z',
    sentiment: 'neutral',
    url: 'https://example.com/news3',
    impact: 'low',
    summary: 'TCS has announced a new artificial intelligence initiative...',
    relatedStocks: ['TCS'],
    category: 'technology',
    sentimentScore: 0.15
  }
];

describe('NewsPage Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockedAxios.get.mockResolvedValue({ data: { news: mockNewsData } });
  });

  test('renders news page with header and filters', async () => {
    render(
      <TestWrapper>
        <NewsPage />
      </TestWrapper>
    );

    // Check header
    expect(screen.getByText('news.title')).toBeInTheDocument();
    expect(screen.getByText('news.subtitle')).toBeInTheDocument();

    // Check filters
    expect(screen.getByPlaceholderText('news.searchPlaceholder')).toBeInTheDocument();
    expect(screen.getByText('news.allSentiments')).toBeInTheDocument();
    expect(screen.getByText('news.sortBy')).toBeInTheDocument();
  });

  test('displays news articles after loading', async () => {
    render(
      <TestWrapper>
        <NewsPage />
      </TestWrapper>
    );

    // Wait for news to load
    await waitFor(() => {
      expect(screen.getByText('Reliance Industries Reports Strong Q4 Results')).toBeInTheDocument();
      expect(screen.getByText('Market Volatility Concerns Rise')).toBeInTheDocument();
      expect(screen.getByText('TCS Announces New AI Initiative')).toBeInTheDocument();
    });
  });

  test('shows loading state initially', () => {
    render(
      <TestWrapper>
        <NewsPage />
      </TestWrapper>
    );

    expect(screen.getByTestId('news-loading')).toBeInTheDocument();
  });

  test('filters news by sentiment', async () => {
    render(
      <TestWrapper>
        <NewsPage />
      </TestWrapper>
    );

    // Wait for news to load
    await waitFor(() => {
      expect(screen.getByText('Reliance Industries Reports Strong Q4 Results')).toBeInTheDocument();
    });

    // Filter by positive sentiment
    const sentimentFilter = screen.getByText('news.positive');
    fireEvent.click(sentimentFilter);

    // Should only show positive news
    expect(screen.getByText('Reliance Industries Reports Strong Q4 Results')).toBeInTheDocument();
    expect(screen.queryByText('Market Volatility Concerns Rise')).not.toBeInTheDocument();
  });

  test('filters news by category', async () => {
    render(
      <TestWrapper>
        <NewsPage />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText('Reliance Industries Reports Strong Q4 Results')).toBeInTheDocument();
    });

    // Filter by earnings category
    const categoryFilter = screen.getByText('news.earnings');
    fireEvent.click(categoryFilter);

    // Should only show earnings news
    expect(screen.getByText('Reliance Industries Reports Strong Q4 Results')).toBeInTheDocument();
    expect(screen.queryByText('TCS Announces New AI Initiative')).not.toBeInTheDocument();
  });

  test('searches news articles', async () => {
    render(
      <TestWrapper>
        <NewsPage />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText('Reliance Industries Reports Strong Q4 Results')).toBeInTheDocument();
    });

    // Search for 'Reliance'
    const searchInput = screen.getByPlaceholderText('news.searchPlaceholder');
    fireEvent.change(searchInput, { target: { value: 'Reliance' } });

    // Should filter results
    expect(screen.getByText('Reliance Industries Reports Strong Q4 Results')).toBeInTheDocument();
    expect(screen.queryByText('TCS Announces New AI Initiative')).not.toBeInTheDocument();
  });

  test('sorts news articles', async () => {
    render(
      <TestWrapper>
        <NewsPage />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText('Reliance Industries Reports Strong Q4 Results')).toBeInTheDocument();
    });

    // Change sort order
    const sortSelect = screen.getByDisplayValue('news.newest');
    fireEvent.change(sortSelect, { target: { value: 'oldest' } });

    // Articles should be reordered
    const articles = screen.getAllByTestId('news-article');
    expect(articles[0]).toHaveTextContent('TCS Announces New AI Initiative');
  });

  test('displays sentiment overview', async () => {
    render(
      <TestWrapper>
        <NewsPage />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText('news.sentimentOverview')).toBeInTheDocument();
    });

    // Check sentiment percentages
    expect(screen.getByText('33.3%')).toBeInTheDocument(); // Positive
    expect(screen.getByText('33.3%')).toBeInTheDocument(); // Negative
    expect(screen.getByText('33.3%')).toBeInTheDocument(); // Neutral
  });

  test('shows sentiment chart', async () => {
    render(
      <TestWrapper>
        <NewsPage />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByTestId('sentiment-chart')).toBeInTheDocument();
    });
  });

  test('displays top positive stocks', async () => {
    render(
      <TestWrapper>
        <NewsPage />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText('news.topPositiveStocks')).toBeInTheDocument();
      expect(screen.getByText('RELIANCE')).toBeInTheDocument();
    });
  });

  test('handles news article clicks', async () => {
    render(
      <TestWrapper>
        <NewsPage />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText('Reliance Industries Reports Strong Q4 Results')).toBeInTheDocument();
    });

    // Click on news article
    const newsTitle = screen.getByText('Reliance Industries Reports Strong Q4 Results');
    fireEvent.click(newsTitle);

    // Should open external link (mocked)
    expect(window.open).toHaveBeenCalledWith('https://example.com/news1', '_blank');
  });

  test('shows impact indicators', async () => {
    render(
      <TestWrapper>
        <NewsPage />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText('news.highImpact')).toBeInTheDocument();
      expect(screen.getByText('news.mediumImpact')).toBeInTheDocument();
      expect(screen.getByText('news.lowImpact')).toBeInTheDocument();
    });
  });

  test('displays related stocks', async () => {
    render(
      <TestWrapper>
        <NewsPage />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText('RELIANCE')).toBeInTheDocument();
      expect(screen.getByText('TCS')).toBeInTheDocument();
    });
  });

  test('handles load more functionality', async () => {
    // Mock API to return more news on second call
    mockedAxios.get.mockResolvedValueOnce({ data: { news: mockNewsData } })
                  .mockResolvedValueOnce({ data: { news: [...mockNewsData, ...mockNewsData] } });

    render(
      <TestWrapper>
        <NewsPage />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText('Reliance Industries Reports Strong Q4 Results')).toBeInTheDocument();
    });

    // Click load more
    const loadMoreButton = screen.getByText('news.loadMore');
    fireEvent.click(loadMoreButton);

    // Should load more articles
    await waitFor(() => {
      expect(mockedAxios.get).toHaveBeenCalledTimes(2);
    });
  });

  test('shows empty state when no news found', async () => {
    mockedAxios.get.mockResolvedValue({ data: { news: [] } });

    render(
      <TestWrapper>
        <NewsPage />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText('news.noNewsFound')).toBeInTheDocument();
    });
  });

  test('handles API errors gracefully', async () => {
    mockedAxios.get.mockRejectedValue(new Error('API Error'));

    render(
      <TestWrapper>
        <NewsPage />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText('news.errorLoading')).toBeInTheDocument();
    });
  });

  test('filters by impact level', async () => {
    render(
      <TestWrapper>
        <NewsPage />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText('Reliance Industries Reports Strong Q4 Results')).toBeInTheDocument();
    });

    // Filter by high impact
    const impactFilter = screen.getByText('news.highImpact');
    fireEvent.click(impactFilter);

    // Should only show high impact news
    expect(screen.getByText('Reliance Industries Reports Strong Q4 Results')).toBeInTheDocument();
    expect(screen.queryByText('TCS Announces New AI Initiative')).not.toBeInTheDocument();
  });

  test('filters by related stock', async () => {
    render(
      <TestWrapper>
        <NewsPage />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText('Reliance Industries Reports Strong Q4 Results')).toBeInTheDocument();
    });

    // Filter by TCS stock
    const stockFilter = screen.getByText('TCS');
    fireEvent.click(stockFilter);

    // Should only show TCS related news
    expect(screen.getByText('TCS Announces New AI Initiative')).toBeInTheDocument();
    expect(screen.queryByText('Reliance Industries Reports Strong Q4 Results')).not.toBeInTheDocument();
  });

  test('displays news source information', async () => {
    render(
      <TestWrapper>
        <NewsPage />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText('Economic Times')).toBeInTheDocument();
      expect(screen.getByText('Moneycontrol')).toBeInTheDocument();
      expect(screen.getByText('Business Standard')).toBeInTheDocument();
    });
  });

  test('shows formatted dates', async () => {
    render(
      <TestWrapper>
        <NewsPage />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText(/Jan 15, 2024/)).toBeInTheDocument();
      expect(screen.getByText(/Jan 14, 2024/)).toBeInTheDocument();
      expect(screen.getByText(/Jan 13, 2024/)).toBeInTheDocument();
    });
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
        <NewsPage />
      </TestWrapper>
    );

    const container = screen.getByTestId('news-container');
    expect(container).toHaveClass('px-4', 'sm:px-6', 'lg:px-8');
  });

  test('shows sentiment score indicators', async () => {
    render(
      <TestWrapper>
        <NewsPage />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText('0.85')).toBeInTheDocument(); // Positive score
      expect(screen.getByText('-0.65')).toBeInTheDocument(); // Negative score
      expect(screen.getByText('0.15')).toBeInTheDocument(); // Neutral score
    });
  });

  test('navigates to stock predictions', async () => {
    const mockNavigate = jest.fn();
    
    // Mock useNavigate
    jest.doMock('react-router-dom', () => ({
      ...jest.requireActual('react-router-dom'),
      useNavigate: () => mockNavigate,
    }));

    render(
      <TestWrapper>
        <NewsPage />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText('Reliance Industries Reports Strong Q4 Results')).toBeInTheDocument();
    });

    // Click on "View Predictions" link
    const predictionsLink = screen.getByText('news.viewPredictions');
    fireEvent.click(predictionsLink);

    expect(mockNavigate).toHaveBeenCalledWith('/predictions/RELIANCE');
  });

  test('refreshes news data', async () => {
    render(
      <TestWrapper>
        <NewsPage />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText('Reliance Industries Reports Strong Q4 Results')).toBeInTheDocument();
    });

    // Click refresh button
    const refreshButton = screen.getByTestId('refresh-news');
    fireEvent.click(refreshButton);

    // Should call API again
    expect(mockedAxios.get).toHaveBeenCalledTimes(2);
  });
});