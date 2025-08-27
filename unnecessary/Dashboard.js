import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';

// Icons
import {
  FiTrendingUp,
  FiTrendingDown,
  FiActivity,
  FiBarChart2,
  FiFileText,
  FiStar,
  FiPlusCircle,
  FiMinusCircle,
  FiRefreshCw,
  FiAlertCircle
} from 'react-icons/fi';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

// Mock data for the dashboard
const mockMarketData = {
  nifty50: {
    value: 22345.67,
    change: 234.56,
    changePercent: 1.05,
    isPositive: true
  },
  sensex: {
    value: 73456.78,
    change: 567.89,
    changePercent: 0.78,
    isPositive: true
  },
  bankNifty: {
    value: 48765.43,
    change: -123.45,
    changePercent: -0.25,
    isPositive: false
  }
};

const mockTopGainers = [
  { symbol: 'TATASTEEL.NS', name: 'Tata Steel', change: 5.67, price: 1456.78 },
  { symbol: 'BAJFINANCE.NS', name: 'Bajaj Finance', change: 4.32, price: 7654.32 },
  { symbol: 'HDFCBANK.NS', name: 'HDFC Bank', change: 3.21, price: 1678.90 },
  { symbol: 'RELIANCE.NS', name: 'Reliance Industries', change: 2.98, price: 2876.54 }
];

const mockTopLosers = [
  { symbol: 'INFY.NS', name: 'Infosys', change: -3.45, price: 1432.10 },
  { symbol: 'WIPRO.NS', name: 'Wipro', change: -2.87, price: 432.65 },
  { symbol: 'TCS.NS', name: 'Tata Consultancy Services', change: -2.34, price: 3456.78 },
  { symbol: 'SUNPHARMA.NS', name: 'Sun Pharma', change: -1.98, price: 1098.76 }
];

const mockRecentPredictions = [
  {
    symbol: 'TATAMOTORS.NS',
    name: 'Tata Motors',
    prediction: 'buy',
    confidence: 87,
    predictedPrice: 950.25,
    currentPrice: 900.50,
    timeframe: 'short_term'
  },
  {
    symbol: 'RELIANCE.NS',
    name: 'Reliance Industries',
    prediction: 'hold',
    confidence: 65,
    predictedPrice: 2900.75,
    currentPrice: 2876.54,
    timeframe: 'medium_term'
  },
  {
    symbol: 'INFY.NS',
    name: 'Infosys',
    prediction: 'sell',
    confidence: 72,
    predictedPrice: 1350.30,
    currentPrice: 1432.10,
    timeframe: 'short_term'
  }
];

const mockLatestNews = [
  {
    id: 1,
    title: 'Tata Motors reports strong Q4 results, profit jumps 46%',
    source: 'Economic Times',
    date: '2023-05-12',
    sentiment: 'positive',
    url: '#',
    impact: 'high'
  },
  {
    id: 2,
    title: 'RBI keeps repo rate unchanged at 6.5% for the third time',
    source: 'Mint',
    date: '2023-05-10',
    sentiment: 'neutral',
    url: '#',
    impact: 'medium'
  },
  {
    id: 3,
    title: 'IT sector faces headwinds as global tech spending slows',
    source: 'Business Standard',
    date: '2023-05-08',
    sentiment: 'negative',
    url: '#',
    impact: 'high'
  },
  {
    id: 4,
    title: 'Government announces new PLI scheme for manufacturing sector',
    source: 'CNBC',
    date: '2023-05-05',
    sentiment: 'positive',
    url: '#',
    impact: 'medium'
  }
];

// Chart data for market overview
const marketChartData = {
  labels: ['9:15', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '15:30'],
  datasets: [
    {
      label: 'NIFTY 50',
      data: [22100, 22150, 22200, 22180, 22250, 22300, 22320, 22345.67],
      borderColor: 'rgba(59, 130, 246, 1)',
      backgroundColor: 'rgba(59, 130, 246, 0.1)',
      fill: true,
      tension: 0.4,
      pointRadius: 2,
      pointHoverRadius: 5
    }
  ]
};

const Dashboard = () => {
  const { t } = useTranslation();
  const [isLoading, setIsLoading] = useState(true);
  const [marketData, setMarketData] = useState(mockMarketData);
  const [topGainers, setTopGainers] = useState(mockTopGainers);
  const [topLosers, setTopLosers] = useState(mockTopLosers);
  const [recentPredictions, setRecentPredictions] = useState(mockRecentPredictions);
  const [latestNews, setLatestNews] = useState(mockLatestNews);
  const [chartData, setChartData] = useState(marketChartData);

  // Simulate data loading
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 1000);

    return () => clearTimeout(timer);
  }, []);

  // Chart options
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        position: 'top',
        labels: {
          usePointStyle: true,
          boxWidth: 6
        }
      },
      tooltip: {
        mode: 'index',
        intersect: false
      }
    },
    scales: {
      x: {
        grid: {
          display: false
        }
      },
      y: {
        grid: {
          color: 'rgba(0, 0, 0, 0.05)'
        },
        ticks: {
          callback: (value) => `₹${value.toLocaleString()}`
        }
      }
    },
    interaction: {
      mode: 'nearest',
      axis: 'x',
      intersect: false
    }
  };

  // Function to refresh data
  const refreshData = () => {
    setIsLoading(true);
    // Simulate API call
    setTimeout(() => {
      setIsLoading(false);
    }, 1000);
  };

  // Render loading skeleton
  if (isLoading) {
    return (
      <div className="animate-pulse">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          {[1, 2, 3].map((i) => (
            <div key={i} className="bg-white dark:bg-neutral-800 rounded-lg shadow-sm p-4 h-24">
              <div className="h-4 bg-neutral-200 dark:bg-neutral-700 rounded w-1/3 mb-2"></div>
              <div className="h-8 bg-neutral-200 dark:bg-neutral-700 rounded w-1/2 mb-2"></div>
              <div className="h-4 bg-neutral-200 dark:bg-neutral-700 rounded w-1/4"></div>
            </div>
          ))}
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <div className="bg-white dark:bg-neutral-800 rounded-lg shadow-sm p-4 h-64">
            <div className="h-4 bg-neutral-200 dark:bg-neutral-700 rounded w-1/4 mb-4"></div>
            <div className="h-48 bg-neutral-200 dark:bg-neutral-700 rounded"></div>
          </div>
          <div className="bg-white dark:bg-neutral-800 rounded-lg shadow-sm p-4">
            <div className="h-4 bg-neutral-200 dark:bg-neutral-700 rounded w-1/4 mb-4"></div>
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="h-12 bg-neutral-200 dark:bg-neutral-700 rounded mb-2"></div>
            ))}
          </div>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-neutral-800 rounded-lg shadow-sm p-4">
            <div className="h-4 bg-neutral-200 dark:bg-neutral-700 rounded w-1/4 mb-4"></div>
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-16 bg-neutral-200 dark:bg-neutral-700 rounded mb-2"></div>
            ))}
          </div>
          <div className="bg-white dark:bg-neutral-800 rounded-lg shadow-sm p-4">
            <div className="h-4 bg-neutral-200 dark:bg-neutral-700 rounded w-1/4 mb-4"></div>
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="h-12 bg-neutral-200 dark:bg-neutral-700 rounded mb-2"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div>
      {/* Page header */}
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
          {t('dashboard')}
        </h1>
        <button
          onClick={refreshData}
          className="flex items-center px-3 py-2 bg-primary-50 dark:bg-primary-900 text-primary-600 dark:text-primary-400 rounded-md hover:bg-primary-100 dark:hover:bg-primary-800 transition-colors"
        >
          <FiRefreshCw className="mr-2 h-4 w-4" />
          {t('refresh')}
        </button>
      </div>

      {/* Market overview cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {Object.entries(marketData).map(([key, data]) => (
          <div key={key} className="card">
            <h3 className="text-sm font-medium text-neutral-500 dark:text-neutral-400 uppercase">
              {key === 'nifty50' ? 'NIFTY 50' : key === 'sensex' ? 'SENSEX' : 'BANK NIFTY'}
            </h3>
            <div className="flex items-baseline mt-1">
              <span className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                ₹{data.value.toLocaleString()}
              </span>
              <span
                className={`ml-2 text-sm font-medium ${data.isPositive ? 'text-success-600 dark:text-success-400' : 'text-danger-600 dark:text-danger-400'}`}
              >
                {data.isPositive ? '+' : ''}{data.change.toFixed(2)} ({data.isPositive ? '+' : ''}{data.changePercent.toFixed(2)}%)
              </span>
            </div>
            <div className="flex items-center mt-2 text-xs text-neutral-500 dark:text-neutral-400">
              {data.isPositive ? (
                <FiTrendingUp className="mr-1 h-4 w-4 text-success-500" />
              ) : (
                <FiTrendingDown className="mr-1 h-4 w-4 text-danger-500" />
              )}
              <span>Today</span>
            </div>
          </div>
        ))}
      </div>

      {/* Market chart and top gainers/losers */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Market chart */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
              {t('market_overview')}
            </h2>
            <div className="flex space-x-2">
              <button className="px-2 py-1 text-xs font-medium bg-primary-100 dark:bg-primary-900 text-primary-600 dark:text-primary-400 rounded">
                1D
              </button>
              <button className="px-2 py-1 text-xs font-medium text-neutral-600 dark:text-neutral-400 hover:bg-neutral-100 dark:hover:bg-neutral-700 rounded">
                1W
              </button>
              <button className="px-2 py-1 text-xs font-medium text-neutral-600 dark:text-neutral-400 hover:bg-neutral-100 dark:hover:bg-neutral-700 rounded">
                1M
              </button>
              <button className="px-2 py-1 text-xs font-medium text-neutral-600 dark:text-neutral-400 hover:bg-neutral-100 dark:hover:bg-neutral-700 rounded">
                3M
              </button>
            </div>
          </div>
          <div className="h-64">
            <Line data={chartData} options={chartOptions} />
          </div>
        </div>

        {/* Top gainers and losers */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
              {t('top_movers')}
            </h2>
            <div className="flex space-x-2">
              <button className="px-2 py-1 text-xs font-medium bg-success-100 dark:bg-success-900 text-success-600 dark:text-success-400 rounded flex items-center">
                <FiTrendingUp className="mr-1 h-3 w-3" />
                {t('gainers')}
              </button>
              <button className="px-2 py-1 text-xs font-medium bg-danger-100 dark:bg-danger-900 text-danger-600 dark:text-danger-400 rounded flex items-center">
                <FiTrendingDown className="mr-1 h-3 w-3" />
                {t('losers')}
              </button>
            </div>
          </div>

          {/* Top gainers */}
          <div className="mb-4">
            <h3 className="text-sm font-medium text-neutral-500 dark:text-neutral-400 mb-2 flex items-center">
              <FiTrendingUp className="mr-1 h-4 w-4 text-success-500" />
              {t('top_gainers')}
            </h3>
            <div className="space-y-2">
              {topGainers.map((stock) => (
                <Link
                  key={stock.symbol}
                  to={`/stock/${stock.symbol}`}
                  className="flex items-center justify-between p-2 rounded-md hover:bg-neutral-100 dark:hover:bg-neutral-700 transition-colors"
                >
                  <div>
                    <div className="font-medium text-neutral-900 dark:text-neutral-100">
                      {stock.name}
                    </div>
                    <div className="text-xs text-neutral-500 dark:text-neutral-400">
                      {stock.symbol}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-medium text-neutral-900 dark:text-neutral-100">
                      ₹{stock.price.toLocaleString()}
                    </div>
                    <div className="text-xs font-medium text-success-600 dark:text-success-400">
                      +{stock.change.toFixed(2)}%
                    </div>
                  </div>
                </Link>
              ))}
            </div>
          </div>

          {/* Top losers */}
          <div>
            <h3 className="text-sm font-medium text-neutral-500 dark:text-neutral-400 mb-2 flex items-center">
              <FiTrendingDown className="mr-1 h-4 w-4 text-danger-500" />
              {t('top_losers')}
            </h3>
            <div className="space-y-2">
              {topLosers.map((stock) => (
                <Link
                  key={stock.symbol}
                  to={`/stock/${stock.symbol}`}
                  className="flex items-center justify-between p-2 rounded-md hover:bg-neutral-100 dark:hover:bg-neutral-700 transition-colors"
                >
                  <div>
                    <div className="font-medium text-neutral-900 dark:text-neutral-100">
                      {stock.name}
                    </div>
                    <div className="text-xs text-neutral-500 dark:text-neutral-400">
                      {stock.symbol}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-medium text-neutral-900 dark:text-neutral-100">
                      ₹{stock.price.toLocaleString()}
                    </div>
                    <div className="text-xs font-medium text-danger-600 dark:text-danger-400">
                      {stock.change.toFixed(2)}%
                    </div>
                  </div>
                </Link>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Recent predictions and latest news */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent predictions */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
              {t('recent_predictions')}
            </h2>
            <Link
              to="/predictions"
              className="text-sm text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300"
            >
              {t('view_all')}
            </Link>
          </div>

          <div className="space-y-4">
            {recentPredictions.map((prediction) => (
              <div
                key={prediction.symbol}
                className="border border-neutral-200 dark:border-neutral-700 rounded-lg p-3"
              >
                <div className="flex items-center justify-between mb-2">
                  <Link
                    to={`/stock/${prediction.symbol}`}
                    className="font-medium text-neutral-900 dark:text-neutral-100 hover:text-primary-600 dark:hover:text-primary-400"
                  >
                    {prediction.name}
                  </Link>
                  <div
                    className={`px-2 py-1 rounded-full text-xs font-medium ${prediction.prediction === 'buy' ? 'bg-success-100 dark:bg-success-900 text-success-800 dark:text-success-200' : prediction.prediction === 'sell' ? 'bg-danger-100 dark:bg-danger-900 text-danger-800 dark:text-danger-200' : 'bg-neutral-100 dark:bg-neutral-700 text-neutral-800 dark:text-neutral-200'}`}
                  >
                    {t(prediction.prediction)}
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2 mb-2">
                  <div>
                    <div className="text-xs text-neutral-500 dark:text-neutral-400">
                      {t('current_price')}
                    </div>
                    <div className="font-medium text-neutral-900 dark:text-neutral-100">
                      ₹{prediction.currentPrice.toLocaleString()}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-neutral-500 dark:text-neutral-400">
                      {t('predicted_price')}
                    </div>
                    <div className="font-medium text-neutral-900 dark:text-neutral-100">
                      ₹{prediction.predictedPrice.toLocaleString()}
                    </div>
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-2 mr-2">
                      <div
                        className={`h-2 rounded-full ${prediction.confidence >= 80 ? 'bg-success-500' : prediction.confidence >= 60 ? 'bg-warning-500' : 'bg-danger-500'}`}
                        style={{ width: `${prediction.confidence}%` }}
                      ></div>
                    </div>
                    <span className="text-xs text-neutral-500 dark:text-neutral-400">
                      {prediction.confidence}%
                    </span>
                  </div>
                  <div className="text-xs text-neutral-500 dark:text-neutral-400">
                    {t(prediction.timeframe)}
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-4 text-center">
            <Link
              to="/predictions"
              className="btn-primary inline-flex items-center"
            >
              <FiBarChart2 className="mr-2 h-4 w-4" />
              {t('generate_new_prediction')}
            </Link>
          </div>
        </div>

        {/* Latest news */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
              {t('latest_news')}
            </h2>
            <Link
              to="/news"
              className="text-sm text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300"
            >
              {t('view_all')}
            </Link>
          </div>

          <div className="space-y-4">
            {latestNews.map((news) => (
              <div
                key={news.id}
                className="border-b border-neutral-200 dark:border-neutral-700 last:border-0 pb-3 last:pb-0"
              >
                <div className="flex items-start mb-2">
                  <div
                    className={`mt-1 flex-shrink-0 w-2 h-2 rounded-full mr-2 ${news.sentiment === 'positive' ? 'bg-success-500' : news.sentiment === 'negative' ? 'bg-danger-500' : 'bg-neutral-500'}`}
                  ></div>
                  <div>
                    <a
                      href={news.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="font-medium text-neutral-900 dark:text-neutral-100 hover:text-primary-600 dark:hover:text-primary-400"
                    >
                      {news.title}
                    </a>
                    <div className="flex items-center mt-1 text-xs text-neutral-500 dark:text-neutral-400">
                      <span>{news.source}</span>
                      <span className="mx-1">•</span>
                      <span>{new Date(news.date).toLocaleDateString()}</span>
                      <span className="mx-1">•</span>
                      <span
                        className={`${news.sentiment === 'positive' ? 'text-success-600 dark:text-success-400' : news.sentiment === 'negative' ? 'text-danger-600 dark:text-danger-400' : 'text-neutral-600 dark:text-neutral-400'}`}
                      >
                        {t(news.sentiment)}
                      </span>
                    </div>
                  </div>
                </div>
                <div className="flex justify-between items-center">
                  <div
                    className={`text-xs px-2 py-0.5 rounded-full ${news.impact === 'high' ? 'bg-danger-100 dark:bg-danger-900 text-danger-800 dark:text-danger-200' : news.impact === 'medium' ? 'bg-warning-100 dark:bg-warning-900 text-warning-800 dark:text-warning-200' : 'bg-neutral-100 dark:bg-neutral-700 text-neutral-800 dark:text-neutral-200'}`}
                  >
                    {t(`${news.impact}_impact`)}
                  </div>
                  <a
                    href={news.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300"
                  >
                    {t('read_more')} →
                  </a>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-4 text-center">
            <Link
              to="/news"
              className="btn-outline inline-flex items-center"
            >
              <FiFileText className="mr-2 h-4 w-4" />
              {t('view_all_news')}
            </Link>
          </div>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="mt-6 p-3 bg-neutral-100 dark:bg-neutral-800 border border-neutral-200 dark:border-neutral-700 rounded-lg text-sm text-neutral-600 dark:text-neutral-400 flex items-start">
        <FiAlertCircle className="flex-shrink-0 h-5 w-5 mr-2 mt-0.5 text-warning-500" />
        <p>
          <strong>{t('disclaimer')}:</strong> {t('prediction_disclaimer_long')}
        </p>
      </div>
    </div>
  );
};

export default Dashboard;