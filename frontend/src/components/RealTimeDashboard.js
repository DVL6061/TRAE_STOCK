import React, { useState, useEffect, useMemo } from 'react';
import { Line, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import useWebSocket from '../hooks/useWebSocket';
import { toast } from 'react-toastify';
import { useTranslation } from 'react-i18next';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const RealTimeDashboard = () => {
  const { t } = useTranslation();
  const [stockData, setStockData] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [technicalIndicators, setTechnicalIndicators] = useState({});
  const [selectedSymbol, setSelectedSymbol] = useState('RELIANCE');
  const [timeframe, setTimeframe] = useState('1m');
  const [isLoading, setIsLoading] = useState(true);

  // WebSocket connection for real-time data
  const { lastMessage, connectionStatus, sendMessage } = useWebSocket(
    `ws://localhost:8000/ws/stock/${selectedSymbol}`,
    { maxReconnectAttempts: 5, reconnectInterval: 3000 }
  );

  // Process incoming WebSocket messages
  useEffect(() => {
    if (lastMessage) {
      const { type, data } = lastMessage;
      
      switch (type) {
        case 'stock_price':
          setStockData(prev => {
            const newData = [...prev, data].slice(-100); // Keep last 100 points
            return newData;
          });
          setIsLoading(false);
          break;
          
        case 'prediction':
          setPredictions(prev => {
            const newPredictions = [...prev, data].slice(-50); // Keep last 50 predictions
            return newPredictions;
          });
          break;
          
        case 'technical_indicators':
          setTechnicalIndicators(data);
          break;
          
        case 'error':
          toast.error(`Error: ${data.message}`);
          break;
          
        default:
          console.log('Unknown message type:', type);
      }
    }
  }, [lastMessage]);

  // Subscribe to symbol updates
  useEffect(() => {
    if (connectionStatus === 'Connected') {
      sendMessage({
        action: 'subscribe',
        symbol: selectedSymbol,
        timeframe: timeframe
      });
    }
  }, [selectedSymbol, timeframe, connectionStatus, sendMessage]);

  // Chart data for stock prices
  const priceChartData = useMemo(() => {
    if (!stockData.length) return null;
    
    return {
      labels: stockData.map(item => new Date(item.timestamp).toLocaleTimeString()),
      datasets: [
        {
          label: t('currentPrice'),
          data: stockData.map(item => item.price),
          borderColor: 'rgb(59, 130, 246)',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          fill: true,
          tension: 0.4,
        },
        {
          label: t('predictions'),
          data: predictions.map(item => item.predicted_price),
          borderColor: 'rgb(239, 68, 68)',
          backgroundColor: 'rgba(239, 68, 68, 0.1)',
          borderDash: [5, 5],
          fill: false,
        }
      ]
    };
  }, [stockData, predictions, t]);

  // Chart options
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: `${selectedSymbol} - ${t('realTimePrice')}`,
      },
    },
    scales: {
      y: {
        beginAtZero: false,
        grid: {
          color: 'rgba(156, 163, 175, 0.2)',
        },
      },
      x: {
        grid: {
          color: 'rgba(156, 163, 175, 0.2)',
        },
      },
    },
    animation: {
      duration: 0, // Disable animation for real-time updates
    },
  };

  // Technical indicators chart data
  const indicatorChartData = useMemo(() => {
    if (!technicalIndicators.rsi) return null;
    
    return {
      labels: ['RSI', 'MACD', 'EMA_12', 'EMA_26', 'SMA_20'],
      datasets: [
        {
          label: t('technicalIndicators'),
          data: [
            technicalIndicators.rsi || 0,
            technicalIndicators.macd || 0,
            technicalIndicators.ema_12 || 0,
            technicalIndicators.ema_26 || 0,
            technicalIndicators.sma_20 || 0,
          ],
          backgroundColor: [
            'rgba(255, 99, 132, 0.6)',
            'rgba(54, 162, 235, 0.6)',
            'rgba(255, 205, 86, 0.6)',
            'rgba(75, 192, 192, 0.6)',
            'rgba(153, 102, 255, 0.6)',
          ],
        }
      ]
    };
  }, [technicalIndicators, t]);

  const currentPrice = stockData.length > 0 ? stockData[stockData.length - 1] : null;
  const latestPrediction = predictions.length > 0 ? predictions[predictions.length - 1] : null;

  return (
    <div className="mobile-container p-3 sm:p-6 bg-white dark:bg-gray-800 min-h-screen">
      {/* Header */}
      <div className="mb-4 sm:mb-6">
        <h1 className="text-xl sm:text-2xl lg:text-3xl font-bold text-gray-900 dark:text-white mb-3 sm:mb-4">
          {t('realTimeDashboard')}
        </h1>
        
        {/* Connection Status */}
        <div className="flex items-center space-x-2 sm:space-x-4 mb-3 sm:mb-4">
          <div className={`px-2 sm:px-3 py-1 rounded-full text-xs sm:text-sm font-medium ${
            connectionStatus === 'Connected' 
              ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
              : connectionStatus === 'Connecting' || connectionStatus.includes('Reconnecting')
              ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
              : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
          }`}>
            {t('status')}: {connectionStatus}
          </div>
        </div>

        {/* Controls */}
        <div className="flex flex-col sm:flex-row gap-3 sm:gap-4 mb-4 sm:mb-6">
          <select
            value={selectedSymbol}
            onChange={(e) => setSelectedSymbol(e.target.value)}
            className="w-full sm:w-auto px-3 sm:px-4 py-2 text-sm sm:text-base border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
          >
            <option value="RELIANCE">Reliance Industries</option>
            <option value="TCS">Tata Consultancy Services</option>
            <option value="INFY">Infosys</option>
            <option value="HDFCBANK">HDFC Bank</option>
            <option value="ICICIBANK">ICICI Bank</option>
          </select>
          
          <select
            value={timeframe}
            onChange={(e) => setTimeframe(e.target.value)}
            className="w-full sm:w-auto px-3 sm:px-4 py-2 text-sm sm:text-base border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
          >
            <option value="1m">1 Minute</option>
            <option value="5m">5 Minutes</option>
            <option value="15m">15 Minutes</option>
            <option value="1h">1 Hour</option>
            <option value="1d">1 Day</option>
          </select>
        </div>
      </div>

      {/* Current Price Card */}
      {currentPrice && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 sm:gap-4 lg:gap-6 mb-4 sm:mb-6">
          <div className="bg-gradient-to-r from-blue-500 to-blue-600 p-4 sm:p-6 rounded-lg text-white">
            <h3 className="text-sm sm:text-lg font-semibold mb-1 sm:mb-2">{t('currentPrice')}</h3>
            <p className="text-xl sm:text-2xl lg:text-3xl font-bold">₹{currentPrice.price?.toFixed(2)}</p>
            <p className="text-xs sm:text-sm opacity-90">
              {new Date(currentPrice.timestamp).toLocaleString()}
            </p>
          </div>
          
          {latestPrediction && (
            <div className="bg-gradient-to-r from-green-500 to-green-600 p-4 sm:p-6 rounded-lg text-white">
              <h3 className="text-sm sm:text-lg font-semibold mb-1 sm:mb-2">{t('predictedPrice')}</h3>
              <p className="text-xl sm:text-2xl lg:text-3xl font-bold">₹{latestPrediction.predicted_price?.toFixed(2)}</p>
              <p className="text-xs sm:text-sm opacity-90">
                {t('confidence')}: {(latestPrediction.confidence * 100)?.toFixed(1)}%
              </p>
            </div>
          )}
          
          {technicalIndicators.rsi && (
            <div className="bg-gradient-to-r from-purple-500 to-purple-600 p-4 sm:p-6 rounded-lg text-white sm:col-span-2 lg:col-span-1">
              <h3 className="text-sm sm:text-lg font-semibold mb-1 sm:mb-2">RSI</h3>
              <p className="text-xl sm:text-2xl lg:text-3xl font-bold">{technicalIndicators.rsi?.toFixed(2)}</p>
              <p className="text-xs sm:text-sm opacity-90">
                {technicalIndicators.rsi > 70 ? t('overbought') : 
                 technicalIndicators.rsi < 30 ? t('oversold') : t('neutral')}
              </p>
            </div>
          )}
        </div>
      )}

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 sm:gap-4 lg:gap-6">
        {/* Price Chart */}
        <div className="bg-white dark:bg-gray-700 p-3 sm:p-4 lg:p-6 rounded-lg shadow-lg">
          <h3 className="text-lg sm:text-xl font-semibold mb-3 sm:mb-4 text-gray-900 dark:text-white">
            {t('priceChart')}
          </h3>
          {isLoading ? (
            <div className="flex items-center justify-center h-48 sm:h-56 lg:h-64">
              <div className="animate-spin rounded-full h-8 sm:h-10 lg:h-12 w-8 sm:w-10 lg:w-12 border-b-2 border-blue-500"></div>
            </div>
          ) : priceChartData ? (
            <div className="h-48 sm:h-56 lg:h-64 chart-container">
              <Line data={priceChartData} options={chartOptions} />
            </div>
          ) : (
            <div className="flex items-center justify-center h-48 sm:h-56 lg:h-64 text-gray-500 dark:text-gray-400">
              <span className="text-sm sm:text-base">{t('noDataAvailable')}</span>
            </div>
          )}
        </div>

        {/* Technical Indicators Chart */}
        <div className="bg-white dark:bg-gray-700 p-3 sm:p-4 lg:p-6 rounded-lg shadow-lg">
          <h3 className="text-lg sm:text-xl font-semibold mb-3 sm:mb-4 text-gray-900 dark:text-white">
            {t('technicalIndicators')}
          </h3>
          {indicatorChartData ? (
            <div className="h-48 sm:h-56 lg:h-64 chart-container">
              <Bar data={indicatorChartData} options={{
                ...chartOptions,
                plugins: {
                  ...chartOptions.plugins,
                  title: {
                    display: true,
                    text: t('technicalIndicators')
                  }
                }
              }} />
            </div>
          ) : (
            <div className="flex items-center justify-center h-48 sm:h-56 lg:h-64 text-gray-500 dark:text-gray-400">
              <span className="text-sm sm:text-base">{t('noIndicatorData')}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default RealTimeDashboard;