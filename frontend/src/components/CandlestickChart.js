import React, { useState, useEffect, useMemo } from 'react';
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
import { useTranslation } from 'react-i18next';

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
  height = 400 
}) => {
  const { t } = useTranslation();
  const [showIndicators, setShowIndicators] = useState({
    sma20: true,
    ema12: true,
    ema26: false,
    bollinger: false,
    volume: true
  });
  const [selectedIndicator, setSelectedIndicator] = useState('RSI');

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
                `Open: ₹${dataPoint.o.toFixed(2)}`,
                `High: ₹${dataPoint.h.toFixed(2)}`,
                `Low: ₹${dataPoint.l.toFixed(2)}`,
                `Close: ₹${dataPoint.c.toFixed(2)}`,
                `Volume: ${dataPoint.v?.toLocaleString() || 'N/A'}`
              ];
            }
            return `${context.dataset.label}: ₹${context.parsed.y.toFixed(2)}`;
          }
        }
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
          callback: (value) => `₹${value.toFixed(2)}`
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

  return (
    <div className="bg-white dark:bg-gray-800 p-3 sm:p-4 lg:p-6 rounded-lg shadow-lg">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-4 sm:mb-6 space-y-3 sm:space-y-0">
        <h3 className="text-lg sm:text-xl font-semibold text-gray-900 dark:text-white">
          <span className="block sm:inline">{symbol} - {t('candlestickChart')}</span>
          <span className="block sm:inline text-sm sm:text-base text-gray-600 dark:text-gray-400 sm:ml-1">({timeframe})</span>
        </h3>
        
        {/* Indicator Controls */}
        <div className="flex flex-wrap gap-1 sm:gap-2">
          <button
            onClick={() => setShowIndicators(prev => ({ ...prev, sma20: !prev.sma20 }))}
            className={`px-2 sm:px-3 py-1 rounded text-xs sm:text-sm font-medium transition-colors ${
              showIndicators.sma20
                ? 'bg-yellow-500 text-white'
                : 'bg-gray-200 text-gray-700 dark:bg-gray-600 dark:text-gray-300'
            }`}
          >
            SMA 20
          </button>
          
          <button
            onClick={() => setShowIndicators(prev => ({ ...prev, ema12: !prev.ema12 }))}
            className={`px-2 sm:px-3 py-1 rounded text-xs sm:text-sm font-medium transition-colors ${
              showIndicators.ema12
                ? 'bg-blue-500 text-white'
                : 'bg-gray-200 text-gray-700 dark:bg-gray-600 dark:text-gray-300'
            }`}
          >
            EMA 12
          </button>
          
          <button
            onClick={() => setShowIndicators(prev => ({ ...prev, ema26: !prev.ema26 }))}
            className={`px-2 sm:px-3 py-1 rounded text-xs sm:text-sm font-medium transition-colors ${
              showIndicators.ema26
                ? 'bg-purple-500 text-white'
                : 'bg-gray-200 text-gray-700 dark:bg-gray-600 dark:text-gray-300'
            }`}
          >
            EMA 26
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
            <option value="RSI">RSI</option>
            <option value="MACD">MACD</option>
            <option value="stochastic">Stochastic</option>
            <option value="williams_r">Williams %R</option>
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