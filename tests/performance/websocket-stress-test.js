import ws from 'k6/ws';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter, Gauge } from 'k6/metrics';
import { htmlReport } from 'https://raw.githubusercontent.com/benc-uk/k6-reporter/main/dist/bundle.js';

// Custom WebSocket metrics
const wsConnectionRate = new Rate('ws_connection_rate');
const wsMessageRate = new Rate('ws_message_rate');
const wsLatency = new Trend('ws_latency');
const wsErrors = new Counter('ws_errors');
const activeConnections = new Gauge('active_ws_connections');
const messagesSent = new Counter('ws_messages_sent');
const messagesReceived = new Counter('ws_messages_received');
const connectionDuration = new Trend('ws_connection_duration');

// WebSocket stress test configuration
export const options = {
  stages: [
    { duration: '30s', target: 10 },   // Warm up connections
    { duration: '1m', target: 25 },    // Light load
    { duration: '2m', target: 50 },    // Medium load
    { duration: '2m', target: 100 },   // High load
    { duration: '3m', target: 200 },   // Stress load
    { duration: '2m', target: 300 },   // Heavy stress
    { duration: '1m', target: 400 },   // Breaking point
    { duration: '2m', target: 200 },   // Recovery
    { duration: '1m', target: 0 },     // Cool down
  ],
  thresholds: {
    ws_connection_rate: {
      'rate>0.95': true, // 95% of connections should succeed
    },
    ws_message_rate: {
      'rate>0.98': true, // 98% of messages should be processed
    },
    ws_latency: {
      'p(50)<100': true,  // 50% of messages under 100ms
      'p(90)<500': true,  // 90% of messages under 500ms
      'p(95)<1000': true, // 95% of messages under 1s
      'p(99)<2000': true, // 99% of messages under 2s
    },
    ws_errors: {
      'count<100': true, // Less than 100 errors total
    },
    active_ws_connections: {
      'value>0': true, // Should maintain connections
    },
  },
};

// WebSocket URL from environment
const WS_URL = __ENV.WS_URL || 'ws://localhost:8000/ws';
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

// Test data
const stockSymbols = [
  'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX',
  'ORCL', 'CRM', 'ADBE', 'INTC', 'AMD', 'QCOM', 'TXN', 'AVGO'
];

const subscriptionTypes = [
  'real_time_price',
  'technical_indicators',
  'news_sentiment',
  'predictions',
  'market_alerts',
  'portfolio_updates'
];

const timeframes = ['1m', '5m', '15m', '30m', '1h'];

// Connection state tracking
let connectionStats = {
  id: Math.random().toString(36).substring(7),
  startTime: Date.now(),
  messagesSent: 0,
  messagesReceived: 0,
  errors: 0,
  connectionAttempts: 0,
  successfulConnections: 0
};

export function setup() {
  console.log('Starting WebSocket stress test setup...');
  
  // Verify WebSocket endpoint is available
  const healthCheck = http.get(`${BASE_URL}/health`);
  if (healthCheck.status !== 200) {
    throw new Error(`System not ready. Health check failed with status: ${healthCheck.status}`);
  }
  
  console.log('WebSocket stress test ready to start...');
  return { wsUrl: WS_URL, baseUrl: BASE_URL, startTime: Date.now() };
}

export default function(data) {
  // Simulate different WebSocket usage patterns
  const scenario = getWebSocketScenario();
  
  switch (scenario) {
    case 'real_time_trader':
      realTimeTraderScenario(data.wsUrl);
      break;
    case 'dashboard_user':
      dashboardUserScenario(data.wsUrl);
      break;
    case 'alert_subscriber':
      alertSubscriberScenario(data.wsUrl);
      break;
    case 'api_streamer':
      apiStreamerScenario(data.wsUrl);
      break;
    case 'heavy_subscriber':
      heavySubscriberScenario(data.wsUrl);
      break;
    default:
      mixedWebSocketScenario(data.wsUrl);
  }
}

// WebSocket scenario distribution
function getWebSocketScenario() {
  const rand = Math.random();
  if (rand < 0.25) return 'real_time_trader';
  if (rand < 0.45) return 'dashboard_user';
  if (rand < 0.65) return 'alert_subscriber';
  if (rand < 0.8) return 'api_streamer';
  if (rand < 0.9) return 'heavy_subscriber';
  return 'mixed';
}

// Real-time trader - high frequency price updates
function realTimeTraderScenario(wsUrl) {
  const symbols = getRandomSymbols(3);
  const connectionDurationMs = (Math.random() * 30 + 10) * 1000; // 10-40 seconds
  
  establishWebSocketConnection(wsUrl, connectionDurationMs, (socket) => {
    // Subscribe to real-time prices for multiple symbols
    symbols.forEach(symbol => {
      subscribeToRealTimePrice(socket, symbol, '1m');
      subscribeToTechnicalIndicators(socket, symbol, ['RSI', 'MACD']);
    });
    
    // Request predictions periodically
    const predictionInterval = setInterval(() => {
      requestPrediction(socket, getRandomSymbol());
    }, 5000);
    
    return () => clearInterval(predictionInterval);
  });
}

// Dashboard user - multiple data streams for visualization
function dashboardUserScenario(wsUrl) {
  const connectionDurationMs = (Math.random() * 60 + 30) * 1000; // 30-90 seconds
  
  establishWebSocketConnection(wsUrl, connectionDurationMs, (socket) => {
    // Subscribe to market overview
    subscribeToMarketOverview(socket);
    
    // Subscribe to portfolio updates
    subscribeToPortfolioUpdates(socket);
    
    // Subscribe to news sentiment
    subscribeToNewsSentiment(socket, getRandomSymbols(2));
    
    // Periodic data refresh
    const refreshInterval = setInterval(() => {
      requestMarketData(socket);
    }, 10000);
    
    return () => clearInterval(refreshInterval);
  });
}

// Alert subscriber - focused on notifications
function alertSubscriberScenario(wsUrl) {
  const connectionDurationMs = (Math.random() * 120 + 60) * 1000; // 1-3 minutes
  
  establishWebSocketConnection(wsUrl, connectionDurationMs, (socket) => {
    // Subscribe to price alerts
    const symbols = getRandomSymbols(5);
    symbols.forEach(symbol => {
      subscribeToAlerts(socket, symbol, {
        price_above: Math.random() * 1000 + 100,
        price_below: Math.random() * 50 + 10,
        volume_spike: true,
        news_sentiment: 'negative'
      });
    });
    
    // Keep connection alive with heartbeat
    const heartbeatInterval = setInterval(() => {
      sendHeartbeat(socket);
    }, 30000);
    
    return () => clearInterval(heartbeatInterval);
  });
}

// API streamer - programmatic data consumption
function apiStreamerScenario(wsUrl) {
  const connectionDurationMs = (Math.random() * 45 + 15) * 1000; // 15-60 seconds
  
  establishWebSocketConnection(wsUrl, connectionDurationMs, (socket) => {
    // High-frequency data requests
    const symbols = getRandomSymbols(2);
    
    symbols.forEach(symbol => {
      subscribeToRealTimePrice(socket, symbol, '1m');
      subscribeToOrderBook(socket, symbol);
    });
    
    // Rapid fire requests
    const rapidInterval = setInterval(() => {
      symbols.forEach(symbol => {
        requestLatestData(socket, symbol);
      });
    }, 2000);
    
    return () => clearInterval(rapidInterval);
  });
}

// Heavy subscriber - maximum subscriptions
function heavySubscriberScenario(wsUrl) {
  const connectionDurationMs = (Math.random() * 90 + 30) * 1000; // 30-120 seconds
  
  establishWebSocketConnection(wsUrl, connectionDurationMs, (socket) => {
    // Subscribe to everything for multiple symbols
    const symbols = getRandomSymbols(8);
    
    symbols.forEach(symbol => {
      subscribeToRealTimePrice(socket, symbol, '1m');
      subscribeToTechnicalIndicators(socket, symbol, ['RSI', 'MACD', 'EMA', 'SMA']);
      subscribeToNewsSentiment(socket, [symbol]);
      subscribeToAlerts(socket, symbol, { price_change: 5 });
    });
    
    // Continuous data requests
    const continuousInterval = setInterval(() => {
      const randomSymbol = symbols[Math.floor(Math.random() * symbols.length)];
      requestPrediction(socket, randomSymbol);
      requestLatestData(socket, randomSymbol);
    }, 3000);
    
    return () => clearInterval(continuousInterval);
  });
}

// Mixed scenario
function mixedWebSocketScenario(wsUrl) {
  const connectionDurationMs = (Math.random() * 60 + 20) * 1000; // 20-80 seconds
  
  establishWebSocketConnection(wsUrl, connectionDurationMs, (socket) => {
    // Random mix of subscriptions
    const symbol = getRandomSymbol();
    const subscriptionType = subscriptionTypes[Math.floor(Math.random() * subscriptionTypes.length)];
    
    switch (subscriptionType) {
      case 'real_time_price':
        subscribeToRealTimePrice(socket, symbol, getRandomTimeframe());
        break;
      case 'technical_indicators':
        subscribeToTechnicalIndicators(socket, symbol, ['RSI']);
        break;
      case 'news_sentiment':
        subscribeToNewsSentiment(socket, [symbol]);
        break;
      case 'predictions':
        requestPrediction(socket, symbol);
        break;
      case 'market_alerts':
        subscribeToAlerts(socket, symbol, { price_change: 3 });
        break;
      case 'portfolio_updates':
        subscribeToPortfolioUpdates(socket);
        break;
    }
  });
}

// Core WebSocket connection function
function establishWebSocketConnection(wsUrl, durationMs, setupCallback) {
  const startTime = Date.now();
  connectionStats.connectionAttempts++;
  activeConnections.add(1);
  
  const response = ws.connect(wsUrl, {
    headers: {
      'User-Agent': 'k6-websocket-stress-test',
      'X-Test-Session': connectionStats.id
    }
  }, function(socket) {
    connectionStats.successfulConnections++;
    wsConnectionRate.add(1);
    
    let cleanup = null;
    let messageCount = 0;
    let lastMessageTime = Date.now();
    
    socket.on('open', function() {
      console.log(`WebSocket connection established (VU: ${__VU})`);
      
      // Setup subscriptions and intervals
      cleanup = setupCallback(socket);
    });
    
    socket.on('message', function(message) {
      const currentTime = Date.now();
      const latency = currentTime - lastMessageTime;
      
      messageCount++;
      connectionStats.messagesReceived++;
      messagesReceived.add(1);
      wsMessageRate.add(1);
      wsLatency.add(latency);
      
      // Validate message format
      try {
        const data = JSON.parse(message);
        
        const isValid = check(data, {
          'message has type': (d) => d.type !== undefined,
          'message has data': (d) => d.data !== undefined,
          'message timestamp valid': (d) => d.timestamp && !isNaN(new Date(d.timestamp).getTime()),
        });
        
        if (!isValid) {
          wsErrors.add(1);
          connectionStats.errors++;
        }
      } catch (error) {
        wsErrors.add(1);
        connectionStats.errors++;
        console.error('Invalid JSON message received:', error.message);
      }
      
      lastMessageTime = currentTime;
    });
    
    socket.on('error', function(error) {
      console.error(`WebSocket error (VU: ${__VU}):`, error);
      wsErrors.add(1);
      connectionStats.errors++;
    });
    
    socket.on('close', function() {
      const duration = Date.now() - startTime;
      connectionDuration.add(duration);
      activeConnections.add(-1);
      
      if (cleanup) {
        cleanup();
      }
      
      console.log(`WebSocket connection closed (VU: ${__VU}), Duration: ${duration}ms, Messages: ${messageCount}`);
    });
    
    // Keep connection alive for specified duration
    setTimeout(() => {
      socket.close();
    }, durationMs);
  });
  
  check(response, {
    'WebSocket connection successful': (r) => r && r.status === 101,
  });
}

// WebSocket subscription functions
function subscribeToRealTimePrice(socket, symbol, timeframe) {
  const message = {
    type: 'subscribe',
    channel: 'real_time_price',
    symbol: symbol,
    timeframe: timeframe,
    timestamp: Date.now()
  };
  
  sendWebSocketMessage(socket, message);
}

function subscribeToTechnicalIndicators(socket, symbol, indicators) {
  const message = {
    type: 'subscribe',
    channel: 'technical_indicators',
    symbol: symbol,
    indicators: indicators,
    timestamp: Date.now()
  };
  
  sendWebSocketMessage(socket, message);
}

function subscribeToNewsSentiment(socket, symbols) {
  const message = {
    type: 'subscribe',
    channel: 'news_sentiment',
    symbols: symbols,
    timestamp: Date.now()
  };
  
  sendWebSocketMessage(socket, message);
}

function subscribeToAlerts(socket, symbol, conditions) {
  const message = {
    type: 'subscribe',
    channel: 'alerts',
    symbol: symbol,
    conditions: conditions,
    timestamp: Date.now()
  };
  
  sendWebSocketMessage(socket, message);
}

function subscribeToMarketOverview(socket) {
  const message = {
    type: 'subscribe',
    channel: 'market_overview',
    timestamp: Date.now()
  };
  
  sendWebSocketMessage(socket, message);
}

function subscribeToPortfolioUpdates(socket) {
  const message = {
    type: 'subscribe',
    channel: 'portfolio_updates',
    user_id: connectionStats.id,
    timestamp: Date.now()
  };
  
  sendWebSocketMessage(socket, message);
}

function subscribeToOrderBook(socket, symbol) {
  const message = {
    type: 'subscribe',
    channel: 'order_book',
    symbol: symbol,
    depth: 10,
    timestamp: Date.now()
  };
  
  sendWebSocketMessage(socket, message);
}

function requestPrediction(socket, symbol) {
  const message = {
    type: 'request',
    action: 'prediction',
    symbol: symbol,
    model_type: 'xgboost',
    timeframe: getRandomTimeframe(),
    timestamp: Date.now()
  };
  
  sendWebSocketMessage(socket, message);
}

function requestLatestData(socket, symbol) {
  const message = {
    type: 'request',
    action: 'latest_data',
    symbol: symbol,
    timestamp: Date.now()
  };
  
  sendWebSocketMessage(socket, message);
}

function requestMarketData(socket) {
  const message = {
    type: 'request',
    action: 'market_data',
    timestamp: Date.now()
  };
  
  sendWebSocketMessage(socket, message);
}

function sendHeartbeat(socket) {
  const message = {
    type: 'heartbeat',
    timestamp: Date.now()
  };
  
  sendWebSocketMessage(socket, message);
}

// Helper function to send WebSocket messages
function sendWebSocketMessage(socket, message) {
  try {
    socket.send(JSON.stringify(message));
    connectionStats.messagesSent++;
    messagesSent.add(1);
  } catch (error) {
    console.error('Failed to send WebSocket message:', error.message);
    wsErrors.add(1);
    connectionStats.errors++;
  }
}

// Utility functions
function getRandomSymbol() {
  return stockSymbols[Math.floor(Math.random() * stockSymbols.length)];
}

function getRandomSymbols(count) {
  const shuffled = [...stockSymbols].sort(() => 0.5 - Math.random());
  return shuffled.slice(0, Math.min(count, stockSymbols.length));
}

function getRandomTimeframe() {
  return timeframes[Math.floor(Math.random() * timeframes.length)];
}

export function teardown(data) {
  const duration = (Date.now() - data.startTime) / 1000;
  
  console.log('WebSocket stress test completed');
  console.log(`Duration: ${duration}s`);
  console.log(`WebSocket URL: ${data.wsUrl}`);
  console.log(`Connection attempts: ${connectionStats.connectionAttempts}`);
  console.log(`Successful connections: ${connectionStats.successfulConnections}`);
  console.log(`Messages sent: ${connectionStats.messagesSent}`);
  console.log(`Messages received: ${connectionStats.messagesReceived}`);
  console.log(`Errors: ${connectionStats.errors}`);
  
  // Generate WebSocket stress test report
  const report = {
    test_type: 'websocket_stress_test',
    ws_url: data.wsUrl,
    duration_seconds: duration,
    timestamp: new Date().toISOString(),
    connection_stats: connectionStats,
    test_scenarios: [
      'real_time_trader',
      'dashboard_user',
      'alert_subscriber',
      'api_streamer',
      'heavy_subscriber',
      'mixed'
    ],
    subscription_types: subscriptionTypes,
    test_stages: [
      { stage: 'warmup', target_connections: 10, duration: '30s' },
      { stage: 'light_load', target_connections: 25, duration: '1m' },
      { stage: 'medium_load', target_connections: 50, duration: '2m' },
      { stage: 'high_load', target_connections: 100, duration: '2m' },
      { stage: 'stress_load', target_connections: 200, duration: '3m' },
      { stage: 'heavy_stress', target_connections: 300, duration: '2m' },
      { stage: 'breaking_point', target_connections: 400, duration: '1m' },
      { stage: 'recovery', target_connections: 200, duration: '2m' },
      { stage: 'cooldown', target_connections: 0, duration: '1m' }
    ]
  };
  
  console.log('WebSocket Stress Test Report:', JSON.stringify(report, null, 2));
}

// Generate HTML report
export function handleSummary(data) {
  return {
    'websocket-stress-test-report.html': htmlReport(data),
    'websocket-stress-test-summary.json': JSON.stringify(data, null, 2),
  };
}