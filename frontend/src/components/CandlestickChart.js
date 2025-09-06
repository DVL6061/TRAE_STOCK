import React, { useState, useEffect, useMemo, useRef } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  TimeScale,
  Tooltip,
  Legend,
} from 'chart.js';
import { Chart } from 'react-chartjs-2';
import 'chartjs-adapter-date-fns';
import { enIN } from 'date-fns/locale';
import { useTranslation } from 'react-i18next';
import { formatCurrency } from '../utils/formatters';
import './CandlestickChart.css';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  TimeScale,
  Tooltip,
  Legend
);

// Custom candlestick chart element
const CandlestickElement = {
  id: 'candlestick',
  beforeDatasetsDraw(chart) {
    const { ctx, data, chartArea: { top, bottom, left, right } } = chart;
    const dataset = data.datasets[0];
    const meta = chart.getDatasetMeta(0);
    
    if (!dataset.data || !meta.data) return;

    ctx.save();
    
    dataset.data.forEach((datapoint, index) => {
      const { o: open, h: high, l: low, c: close } = datapoint;
      const x = meta.data[index].x;
      const candleWidth = 8;
      
      // Determine candle color
      const isGreen = close >= open;
      ctx.fillStyle = isGreen ? '#10B981' : '#EF4444';
      ctx.strokeStyle = isGreen ? '#059669' : '#DC2626';
      ctx.lineWidth = 1;
      
      // Scale prices to chart coordinates
      const scaleY = (price) => {
        const yScale = chart.scales.y;
        return yScale.getPixelForValue(price);
      };
      
      const openY = scaleY(open);
      const closeY = scaleY(close);
      const highY = scaleY(high);
      const lowY = scaleY(low);
      
      // Draw high-low line
      ctx.beginPath();
      ctx.moveTo(x, highY);
      ctx.lineTo(x, lowY);
      ctx.stroke();
      
      // Draw candle body
      const bodyHeight = Math.abs(closeY - openY);
      const bodyTop = Math.min(openY, closeY);
      
      if (bodyHeight > 0) {
        ctx.fillRect(x - candleWidth / 2, bodyTop, candleWidth, bodyHeight);
      } else {
        // Doji candle (open == close)
        ctx.beginPath();
        ctx.moveTo(x - candleWidth / 2, openY);
        ctx.lineTo(x + candleWidth / 2, openY);
        ctx.stroke();
      }
    });
    
    ctx.restore();
  }
};

ChartJS.register(CandlestickElement);

const CandlestickChart = ({ 
  data = [], 
  technicalIndicators = {}, 
  symbol = 'STOCK',
  timeframe = '1D',
  height = 400,
  onTimeframeChange
}) => {
  const { t } = useTranslation();
  const chartRef = useRef(null);
  const chartInstance = useRef(null);
  const [showIndicators, setShowIndicators] = useState({
    sma20: true,
    ema12: true,
    ema26: false,
    bollinger: false,
    volume: true,
    rsi: false,
    macd: false,
    stochastic: false,
    adx: false
  });
  const [selectedIndicator, setSelectedIndicator] = useState('RSI');
  const [timeframeOptions] = useState(['1D', '1W', '1M', '3M', '6M', '1Y']);
  const [indicatorSettings, setIndicatorSettings] = useState({
    sma: { period: 20 },
    ema: { shortPeriod: 12, longPeriod: 26 },
    bollinger: { period: 20, stdDev: 2 },
    rsi: { period: 14, overbought: 70, oversold: 30 },
    macd: { fastPeriod: 12, slowPeriod: 26, signalPeriod: 9 },
    stochastic: { kPeriod: 14, dPeriod: 3, slowing: 3 },
    adx: { period: 14, threshold: 25 }
  });

  // Process candlestick data
  const candlestickData = useMemo(() => {
    if (!data || data.length === 0) return null;

    const processedData = data.map(item => ({
      x: new Date(item.timestamp),
      o: item.open,
      h: item.high,
      l: item.low,
      c: item.close,
      v: item.volume
    }));

    const datasets = [
      {
        label: 'Price',
        data: processedData,
        type: 'candlestick'
      }
    ];

    // Add technical indicators
    if (showIndicators.sma20 && technicalIndicators.sma_20) {
      datasets.push({
        label: 'SMA 20',
        data: technicalIndicators.sma_20.map((value, index) => ({
          x: processedData[index]?.x,
          y: value
        })),
        type: 'line',
        borderColor: '#F59E0B',
        backgroundColor: 'transparent',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.1
      });
    }

    if (showIndicators.ema12 && technicalIndicators.ema_12) {
      datasets.push({
        label: 'EMA 12',
        data: technicalIndicators.ema_12.map((value, index) => ({
          x: processedData[index]?.x,
          y: value
        })),
        type: 'line',
        borderColor: '#3B82F6',
        backgroundColor: 'transparent',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.1
      });
    }

    if (showIndicators.ema26 && technicalIndicators.ema_26) {
      datasets.push({
        label: 'EMA 26',
        data: technicalIndicators.ema_26.map((value, index) => ({
          x: processedData[index]?.x,
          y: value
        })),
        type: 'line',
        borderColor: '#8B5CF6',
        backgroundColor: 'transparent',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.1
      });
    }

    if (showIndicators.bollinger && technicalIndicators.bollinger_bands) {
      const { upper, middle, lower } = technicalIndicators.bollinger_bands;
      
      datasets.push(
        {
          label: 'Bollinger Upper',
          data: upper.map((value, index) => ({
            x: processedData[index]?.x,
            y: value
          })),
          type: 'line',
          borderColor: '#EF4444',
          backgroundColor: 'transparent',
          borderWidth: 1,
          pointRadius: 0,
          borderDash: [5, 5]
        },
        {
          label: 'Bollinger Middle',
          data: middle.map((value, index) => ({
            x: processedData[index]?.x,
            y: value
          })),
          type: 'line',
          borderColor: '#6B7280',
          backgroundColor: 'transparent',
          borderWidth: 1,
          pointRadius: 0
        },
        {
          label: 'Bollinger Lower',
          data: lower.map((value, index) => ({
            x: processedData[index]?.x,
            y: value
          })),
          type: 'line',
          borderColor: '#10B981',
          backgroundColor: 'transparent',
          borderWidth: 1,
          pointRadius: 0,
          borderDash: [5, 5]
        }
      );
    }

    return {
      datasets
    };
  }, [data, technicalIndicators, showIndicators]);

  // Volume chart data
  const volumeData = useMemo(() => {
    if (!data || data.length === 0 || !showIndicators.volume) return null;

    return {
      labels: data.map(item => new Date(item.timestamp)),
      datasets: [
        {
          label: 'Volume',
          data: data.map(item => item.volume),
          backgroundColor: data.map(item => 
            item.close >= item.open ? 'rgba(16, 185, 129, 0.6)' : 'rgba(239, 68, 68, 0.6)'
          ),
          borderColor: data.map(item => 
            item.close >= item.open ? '#10B981' : '#EF4444'
          ),
          borderWidth: 1
        }
      ]
    };
  }, [data, showIndicators.volume]);

  // Secondary indicator data (RSI, MACD, etc.)
  const secondaryIndicatorData = useMemo(() => {
    if (!technicalIndicators[selectedIndicator.toLowerCase()]) return null;

    const indicatorValues = technicalIndicators[selectedIndicator.toLowerCase()];
    
    return {
      labels: data.map(item => new Date(item.timestamp)),
      datasets: [
        {
          label: selectedIndicator,
          data: indicatorValues,
          borderColor: '#8B5CF6',
          backgroundColor: 'rgba(139, 92, 246, 0.1)',
          borderWidth: 2,
          fill: selectedIndicator === 'RSI'
        }
      ]
    };
  }, [data, technicalIndicators, selectedIndicator]);

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      intersect: false,
      mode: 'index'
    },
    plugins: {
      legend: {
        position: 'top',
        labels: {
          filter: (legendItem) => legendItem.text !== 'Price'
        }
      },
      tooltip: {
        callbacks: {
          title: (context) => {
            return new Date(context[0].parsed.x).toLocaleString();
          },
          label: (context) => {
            const dataPoint = context.raw;
            if (dataPoint.o !== undefined) {
              return [
                `Open: ${formatCurrency(dataPoint.o)}`,
                `High: ${formatCurrency(dataPoint.h)}`,
                `Low: ${formatCurrency(dataPoint.l)}`,
                `Close: ${formatCurrency(dataPoint.c)}`,
                `Volume: ${dataPoint.v?.toLocaleString() || 'N/A'}`
              ];
            }
            const label = context.dataset.label || '';
            if (label.includes('RSI')) {
              return `${label}: ${context.parsed.y.toFixed(2)}`;
            } else if (label.includes('MACD')) {
              return `${label}: ${context.parsed.y.toFixed(4)}`;
            } else if (label.includes('%K') || label.includes('%D')) {
              return `${label}: ${context.parsed.y.toFixed(2)}`;
            } else if (label.includes('ADX') || label.includes('DI')) {
              return `${label}: ${context.parsed.y.toFixed(2)}`;
            } else {
              return `${label}: ${formatCurrency(context.parsed.y)}`;
            }
          }
        }
      }
    },
    scales: {
      x: {
        type: 'time',
        time: {
          unit: timeframe === '1D' ? 'day' : 
                timeframe === '1W' ? 'day' : 
                timeframe === '1M' ? 'day' : 
                timeframe === '3M' ? 'week' : 
                timeframe === '6M' ? 'week' : 'month',
          displayFormats: {
            hour: 'HH:mm',
            day: 'MMM d',
            week: 'MMM d',
            month: 'MMM yyyy'
          }
        },
        grid: {
          color: 'rgba(156, 163, 175, 0.2)'
        }
      },
      y: {
        position: 'right',
        grid: {
          color: 'rgba(156, 163, 175, 0.2)'
        },
        ticks: {
          callback: (value) => formatCurrency(value)
        }
      }
    }
  };

  const volumeOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      }
    },
    scales: {
      x: {
        type: 'time',
        time: {
          unit: timeframe === '1D' ? 'day' : timeframe === '1H' ? 'hour' : 'minute'
        },
        grid: {
          color: 'rgba(156, 163, 175, 0.2)'
        }
      },
      y: {
        position: 'right',
        grid: {
          color: 'rgba(156, 163, 175, 0.2)'
        },
        ticks: {
          callback: (value) => value.toLocaleString()
        }
      }
    }
  };

  const secondaryOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      }
    },
    scales: {
      x: {
        type: 'time',
        time: {
          unit: timeframe === '1D' ? 'day' : timeframe === '1H' ? 'hour' : 'minute'
        },
        grid: {
          color: 'rgba(156, 163, 175, 0.2)'
        }
      },
      y: {
        position: 'right',
        grid: {
          color: 'rgba(156, 163, 175, 0.2)'
        },
        min: selectedIndicator === 'RSI' ? 0 : undefined,
        max: selectedIndicator === 'RSI' ? 100 : undefined
      }
    }
  };

  if (!candlestickData) {
    return (
      <div className="bg-white dark:bg-gray-800 p-3 sm:p-4 lg:p-6 rounded-lg shadow-lg">
        <div className="flex items-center justify-center h-48 sm:h-56 lg:h-64 text-gray-500 dark:text-gray-400">
          <span className="text-sm sm:text-base">{t('noDataAvailable')}</span>
        </div>
      </div>
    );
  }

  const toggleIndicator = (indicator) => {
    setShowIndicators(prev => ({
      ...prev,
      [indicator]: !prev[indicator]
    }));
  };
  
  const updateIndicatorSetting = (indicator, setting, value) => {
    setIndicatorSettings(prev => ({
      ...prev,
      [indicator]: {
        ...prev[indicator],
        [setting]: value
      }
    }));
  };
  
  const handleTimeframeChange = (value) => {
    if (onTimeframeChange) {
      onTimeframeChange(value);
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 p-3 sm:p-4 lg:p-6 rounded-lg shadow-lg">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-4 sm:mb-6 space-y-3 sm:space-y-0">
        <div className="flex flex-col sm:flex-row sm:items-center gap-2">
          <h3 className="text-lg sm:text-xl font-semibold text-gray-900 dark:text-white">
            <span className="block sm:inline">{symbol} - {t('candlestickChart')}</span>
          </h3>
          
          {/* Timeframe Selector */}
          <div className="flex items-center">
            <select
              value={timeframe}
              onChange={(e) => handleTimeframeChange(e.target.value)}
              className="px-2 py-1 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
            >
              {timeframeOptions.map(option => (
                <option key={option} value={option}>{option}</option>
              ))}
            </select>
          </div>
        </div>
        
        {/* Indicator Controls - Main Tab Navigation */}
        <div className="flex flex-wrap gap-1 sm:gap-2">
          <button
            onClick={() => setShowIndicators(prev => ({ ...prev, sma20: !prev.sma20 }))}
            className={`px-2 sm:px-3 py-1 rounded text-xs sm:text-sm font-medium transition-colors ${
              showIndicators.sma20
                ? 'bg-yellow-500 text-white'
                : 'bg-gray-200 text-gray-700 dark:bg-gray-600 dark:text-gray-300'
            }`}
          >
            SMA
          </button>
          
          <button
            onClick={() => setShowIndicators(prev => ({ ...prev, ema12: !prev.ema12, ema26: !prev.ema26 }))}
            className={`px-2 sm:px-3 py-1 rounded text-xs sm:text-sm font-medium transition-colors ${
              showIndicators.ema12 || showIndicators.ema26
                ? 'bg-blue-500 text-white'
                : 'bg-gray-200 text-gray-700 dark:bg-gray-600 dark:text-gray-300'
            }`}
          >
            EMA
          </button>
          
          <button
            onClick={() => setShowIndicators(prev => ({ ...prev, bollinger: !prev.bollinger }))}
            className={`px-2 sm:px-3 py-1 rounded text-xs sm:text-sm font-medium transition-colors ${
              showIndicators.bollinger
                ? 'bg-red-500 text-white'
                : 'bg-gray-200 text-gray-700 dark:bg-gray-600 dark:text-gray-300'
            }`}
          >
            Bollinger
          </button>
          
          <button
            onClick={() => setShowIndicators(prev => ({ ...prev, rsi: !prev.rsi }))}
            className={`px-2 sm:px-3 py-1 rounded text-xs sm:text-sm font-medium transition-colors ${
              showIndicators.rsi
                ? 'bg-green-500 text-white'
                : 'bg-gray-200 text-gray-700 dark:bg-gray-600 dark:text-gray-300'
            }`}
          >
            RSI
          </button>
          
          <button
            onClick={() => setShowIndicators(prev => ({ ...prev, macd: !prev.macd }))}
            className={`px-2 sm:px-3 py-1 rounded text-xs sm:text-sm font-medium transition-colors ${
              showIndicators.macd
                ? 'bg-indigo-500 text-white'
                : 'bg-gray-200 text-gray-700 dark:bg-gray-600 dark:text-gray-300'
            }`}
          >
            MACD
          </button>
        </div>
      </div>
      
      {/* Indicator Settings Panel */}
      <div className="mb-4 bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
          {showIndicators.sma20 && (
            <div className="bg-white dark:bg-gray-800 p-2 rounded shadow-sm">
              <h4 className="text-sm font-medium mb-2 text-yellow-600 dark:text-yellow-400">SMA Settings</h4>
              <div className="flex items-center gap-2">
                <label className="text-xs text-gray-600 dark:text-gray-300">Period:</label>
                <input 
                  type="number" 
                  min="2" 
                  max="200"
                  value={indicatorSettings.sma.period}
                  onChange={(e) => setIndicatorSettings(prev => ({
                    ...prev,
                    sma: { ...prev.sma, period: parseInt(e.target.value) || 20 }
                  }))}
                  className="w-16 px-2 py-1 text-xs border border-gray-300 rounded dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                />
              </div>
            </div>
          )}
          
          {(showIndicators.ema12 || showIndicators.ema26) && (
            <div className="bg-white dark:bg-gray-800 p-2 rounded shadow-sm">
              <h4 className="text-sm font-medium mb-2 text-blue-600 dark:text-blue-400">EMA Settings</h4>
              <div className="flex flex-col gap-2">
                <div className="flex items-center gap-2">
                  <label className="text-xs text-gray-600 dark:text-gray-300">Fast Period:</label>
                  <input 
                    type="number" 
                    min="2" 
                    max="100"
                    value={indicatorSettings.ema.shortPeriod}
                    onChange={(e) => setIndicatorSettings(prev => ({
                      ...prev,
                      ema: { ...prev.ema, shortPeriod: parseInt(e.target.value) || 12 }
                    }))}
                    className="w-16 px-2 py-1 text-xs border border-gray-300 rounded dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                  />
                </div>
                <div className="flex items-center gap-2">
                  <label className="text-xs text-gray-600 dark:text-gray-300">Slow Period:</label>
                  <input 
                    type="number" 
                    min="2" 
                    max="100"
                    value={indicatorSettings.ema.longPeriod}
                    onChange={(e) => setIndicatorSettings(prev => ({
                      ...prev,
                      ema: { ...prev.ema, longPeriod: parseInt(e.target.value) || 26 }
                    }))}
                    className="w-16 px-2 py-1 text-xs border border-gray-300 rounded dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                  />
                </div>
              </div>
            </div>
          )}
          
          {showIndicators.bollinger && (
            <div className="bg-white dark:bg-gray-800 p-2 rounded shadow-sm">
              <h4 className="text-sm font-medium mb-2 text-red-600 dark:text-red-400">Bollinger Settings</h4>
              <div className="flex flex-col gap-2">
                <div className="flex items-center gap-2">
                  <label className="text-xs text-gray-600 dark:text-gray-300">Period:</label>
                  <input 
                    type="number" 
                    min="2" 
                    max="100"
                    value={indicatorSettings.bollinger.period}
                    onChange={(e) => setIndicatorSettings(prev => ({
                      ...prev,
                      bollinger: { ...prev.bollinger, period: parseInt(e.target.value) || 20 }
                    }))}
                    className="w-16 px-2 py-1 text-xs border border-gray-300 rounded dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                  />
                </div>
                <div className="flex items-center gap-2">
                  <label className="text-xs text-gray-600 dark:text-gray-300">StdDev:</label>
                  <input 
                    type="number" 
                    min="0.5" 
                    max="5"
                    step="0.1"
                    value={indicatorSettings.bollinger.stdDev}
                    onChange={(e) => setIndicatorSettings(prev => ({
                      ...prev,
                      bollinger: { ...prev.bollinger, stdDev: parseFloat(e.target.value) || 2 }
                    }))}
                    className="w-16 px-2 py-1 text-xs border border-gray-300 rounded dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                  />
                </div>
              </div>
            </div>
          )}
          
          {showIndicators.rsi && (
            <div className="bg-white dark:bg-gray-800 p-2 rounded shadow-sm">
              <h4 className="text-sm font-medium mb-2 text-green-600 dark:text-green-400">RSI Settings</h4>
              <div className="flex flex-col gap-2">
                <div className="flex items-center gap-2">
                  <label className="text-xs text-gray-600 dark:text-gray-300">Period:</label>
                  <input 
                    type="number" 
                    min="2" 
                    max="50"
                    value={indicatorSettings.rsi.period}
                    onChange={(e) => setIndicatorSettings(prev => ({
                      ...prev,
                      rsi: { ...prev.rsi, period: parseInt(e.target.value) || 14 }
                    }))}
                    className="w-16 px-2 py-1 text-xs border border-gray-300 rounded dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                  />
                </div>
                <div className="flex items-center gap-2">
                  <label className="text-xs text-gray-600 dark:text-gray-300">Overbought/Oversold:</label>
                  <input 
                    type="number" 
                    min="60" 
                    max="90"
                    value={indicatorSettings.rsi.overbought}
                    onChange={(e) => setIndicatorSettings(prev => ({
                      ...prev,
                      rsi: { ...prev.rsi, overbought: parseInt(e.target.value) || 70 }
                    }))}
                    className="w-12 px-2 py-1 text-xs border border-gray-300 rounded dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                  />
                  <span className="text-xs text-gray-500">/</span>
                  <input 
                    type="number" 
                    min="10" 
                    max="40"
                    value={indicatorSettings.rsi.oversold}
                    onChange={(e) => setIndicatorSettings(prev => ({
                      ...prev,
                      rsi: { ...prev.rsi, oversold: parseInt(e.target.value) || 30 }
                    }))}
                    className="w-12 px-2 py-1 text-xs border border-gray-300 rounded dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                  />
                </div>
              </div>
            </div>
          )}
          
          {showIndicators.macd && (
            <div className="bg-white dark:bg-gray-800 p-2 rounded shadow-sm">
              <h4 className="text-sm font-medium mb-2 text-indigo-600 dark:text-indigo-400">MACD Settings</h4>
              <div className="flex flex-col gap-2">
                <div className="flex items-center gap-2">
                  <label className="text-xs text-gray-600 dark:text-gray-300">Fast/Slow:</label>
                  <input 
                    type="number" 
                    min="2" 
                    max="50"
                    value={indicatorSettings.macd.fastPeriod}
                    onChange={(e) => setIndicatorSettings(prev => ({
                      ...prev,
                      macd: { ...prev.macd, fastPeriod: parseInt(e.target.value) || 12 }
                    }))}
                    className="w-12 px-2 py-1 text-xs border border-gray-300 rounded dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                  />
                  <span className="text-xs text-gray-500">/</span>
                  <input 
                    type="number" 
                    min="5" 
                    max="100"
                    value={indicatorSettings.macd.slowPeriod}
                    onChange={(e) => setIndicatorSettings(prev => ({
                      ...prev,
                      macd: { ...prev.macd, slowPeriod: parseInt(e.target.value) || 26 }
                    }))}
                    className="w-12 px-2 py-1 text-xs border border-gray-300 rounded dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                  />
                </div>
                <div className="flex items-center gap-2">
                  <label className="text-xs text-gray-600 dark:text-gray-300">Signal Period:</label>
                  <input 
                    type="number" 
                    min="2" 
                    max="50"
                    value={indicatorSettings.macd.signalPeriod}
                    onChange={(e) => setIndicatorSettings(prev => ({
                      ...prev,
                      macd: { ...prev.macd, signalPeriod: parseInt(e.target.value) || 9 }
                    }))}
                    className="w-12 px-2 py-1 text-xs border border-gray-300 rounded dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Main Candlestick Chart */}
      <div className="mb-3 sm:mb-4 chart-container" style={{ height: `${Math.max(250, Math.min(height, window.innerWidth < 640 ? 300 : height))}px` }}>
        <Chart type="line" data={candlestickData} options={chartOptions} />
      </div>

      {/* Volume Chart */}
      {showIndicators.volume && volumeData && (
        <div className="mb-3 sm:mb-4" style={{ height: '100px' }}>
          <h4 className="text-xs sm:text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            {t('volume')}
          </h4>
          <div className="chart-container">
            <Chart type="bar" data={volumeData} options={volumeOptions} />
          </div>
        </div>
      )}

      {/* Secondary Indicator */}
      <div className="mb-3 sm:mb-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-2 space-y-2 sm:space-y-0">
          <h4 className="text-xs sm:text-sm font-medium text-gray-700 dark:text-gray-300">
            {t('technicalIndicators')}
          </h4>
          <select
            value={selectedIndicator}
            onChange={(e) => setSelectedIndicator(e.target.value)}
            className="w-full sm:w-auto px-2 sm:px-3 py-1 text-xs sm:text-sm border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
          >
            <option value="RSI">{t('indicators.rsi')}</option>
            <option value="MACD">{t('indicators.macd')}</option>
            <option value="stochastic">{t('indicators.stochastic')}</option>
            <option value="williams_r">{t('indicators.williamsR')}</option>
          </select>
        </div>
        
        {secondaryIndicatorData && (
          <div style={{ height: '100px' }}>
            <div className="chart-container">
              <Chart type="line" data={secondaryIndicatorData} options={secondaryOptions} />
            </div>
            
            {/* RSI Overbought/Oversold Lines */}
            {selectedIndicator === 'RSI' && (
              <div className="mt-2 text-xs text-gray-500 dark:text-gray-400 flex flex-wrap gap-2 sm:gap-4">
                <div className="flex items-center">
                  <span className="inline-block w-2 sm:w-3 h-2 sm:h-3 bg-red-200 border border-red-400 mr-1"></span>
                  <span className="text-xs">Overbought (70+)</span>
                </div>
                <div className="flex items-center">
                  <span className="inline-block w-2 sm:w-3 h-2 sm:h-3 bg-green-200 border border-green-400 mr-1"></span>
                  <span className="text-xs">Oversold (30-)</span>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default CandlestickChart;