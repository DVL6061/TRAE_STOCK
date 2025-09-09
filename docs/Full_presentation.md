# TRAE Stock Prediction System - Full Presentation

## Project Snapshot Table

| Component | Technology | Status |
|-----------|------------|--------|
| **Frameworks** | FastAPI, React, Tailwind CSS, Docker, Redis, PostgreSQL | ✅ Implemented |
| **Models Present** | XGBoost, Informer/Transformer, DQN | ⚠️ Mock Implementation |
| **Data Sources** | yfinance, Angel One SmartAPI, News scrapers | ✅ Real Implementation |
| **Explainability** | SHAP | ⚠️ Mock Implementation |
| **Metrics** | MAE, MSE, RMSE, MAPE | ✅ Defined |
| **Key Endpoints** | `/api/predictions/price`, `/api/news/sentiment`, `/api/market/live` | ✅ Implemented |

## Library Glossary

- **RSI (Relative Strength Index)**: Momentum oscillator measuring speed and magnitude of price changes
- **MACD (Moving Average Convergence Divergence)**: Trend-following momentum indicator
- **DQN (Deep Q-Network)**: Reinforcement learning algorithm for sequential decision making
- **SHAP (Shapley Additive Explanations)**: Method for explaining individual predictions
- **OHLCV (Open, High, Low, Close, Volume)**: Standard financial data format
- **PE (Price-to-Earnings)**: Valuation ratio of company's share price to earnings per share
- **MAE (Mean Absolute Error)**: Average of absolute differences between predicted and actual values
- **MSE (Mean Squared Error)**: Average of squared differences between predicted and actual values
- **RMSE (Root Mean Squared Error)**: Square root of MSE
- **MAPE (Mean Absolute Percentage Error)**: Mean of absolute percentage errors

---

## 3. Data Collection

### 3.1 Description of data sources

The TRAE Stock system integrates multiple data sources for comprehensive market analysis:

**Historical Price Data (yfinance)**
- **Data Fields**: OHLCV (Open, High, Low, Close, Volume), Adjusted Close
- **Granularity**: 1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo
- **Tickers/Universe**: NSE stocks (e.g., 'RELIANCE.NS', 'TCS.NS')
- **Implementation**: `backend/core/data_fetcher.py#L200-L250`

**Live Price Data (Angel One SmartAPI)**
- **Data Fields**: LTP (Last Traded Price), Volume, Change%, High, Low
- **Granularity**: Real-time (sub-second updates)
- **Tickers/Universe**: NSE/BSE stocks via symbol tokens
- **Implementation**: `backend/core/data_fetcher.py#L50-L120`

**Fundamental Data (Yahoo Finance)**
- **Data Fields**: PE Ratio, EPS, Market Cap, Dividend Yield, ROE, Debt-to-Equity
- **Granularity**: Quarterly/Annual updates
- **Tickers/Universe**: NSE listed companies
- **Implementation**: `backend/core/fundamental_analyzer.py#L25-L110`

**Technical Indicators (ta library)**
- **Data Fields**: RSI, MACD, EMA, SMA, Bollinger Bands, ADX, Stochastic Oscillator
- **Granularity**: Calculated from OHLCV data
- **Implementation**: `backend/core/data_fetcher.py#L7` (import), `backend/ML_models/xgboost_model.py#L50-L100`

**News Sources**
- **Data Fields**: Title, Content, Published Date, Source, URL
- **Granularity**: Real-time news articles
- **Sources**: NewsAPI, Economic Times (configured)
- **Implementation**: `backend/data/news_fetcher.py#L50-L150`

**Evidence**
- `backend/core/data_fetcher.py#L50-L120`: AngelOneClient.get_quote()
- `backend/core/data_fetcher.py#L200-L250`: get_historical_data()
- `backend/core/fundamental_analyzer.py#L25-L110`: get_stock_info()
- `backend/data/news_fetcher.py#L50-L150`: NewsDatabase class
- `backend/app/config.py#L20-L40`: Data source configurations

### 3.2 Data collection methods

**Angel One SmartAPI Integration**
- **Library**: SmartConnect SDK
- **Authentication**: API key, client code, password, TOTP
- **Endpoints**: `/api/v1/user/profile`, `/api/v1/search/instrument`
- **Input Params**: symbol token, exchange, duration
- **Return Schema**: `{"data": {"ltp": float, "volume": int, "chg": float}}`
- **Output Format**: JSON response cached in Redis
- **Implementation**: `backend/core/data_fetcher.py#L25-L120`

**Yahoo Finance Integration**
- **Library**: yfinance
- **Authentication**: None required
- **Method**: REST API calls
- **Input Params**: ticker symbol, period, interval
- **Return Schema**: pandas DataFrame with OHLCV columns
- **Output Format**: DataFrame persisted as CSV/Parquet
- **Implementation**: `backend/core/data_fetcher.py#L200-L250`

**News Data Collection**
- **Library**: requests, BeautifulSoup
- **Authentication**: NewsAPI key
- **Endpoints**: NewsAPI v2, Economic Times RSS
- **Input Params**: query, from_date, to_date, language
- **Return Schema**: `{"articles": [{"title": str, "content": str, "publishedAt": str}]}`
- **Output Format**: SQLite database storage
- **Implementation**: `backend/data/news_fetcher.py#L100-L200`

**Schedulers and Background Tasks**
- **Redis Caching**: Market data cached for 30 seconds
- **Implementation**: `backend/app/services/market_service.py#L50-L100`
- **MISSING**: Celery background tasks for data collection
- **TODO**: Add `backend/tasks/data_collector.py` with scheduled data fetching

**Evidence**
- `backend/core/data_fetcher.py#L25-L120`: AngelOneClient implementation
- `backend/data/news_fetcher.py#L100-L200`: NewsDatabase.fetch_news()
- `backend/app/services/market_service.py#L50-L100`: Redis caching logic
- `docker-compose.yml#L40-L60`: Redis service configuration

### 3.3 Data preprocessing and cleaning procedures

**Missing Value Handling**
- **Method**: Forward fill for price data, interpolation for volume
- **Implementation**: `backend/ML_models/xgboost_model.py#L100-L120`
- **Before Schema**: DataFrame with NaN values
- **After Schema**: Complete DataFrame with filled values

**Timezone Alignment**
- **Method**: Convert all timestamps to IST (Indian Standard Time)
- **Implementation**: `backend/data/news_fetcher.py#L220-L225`
- **Code**: `article_data['publishedAt'].replace('Z', '+00:00')`

**Duplicate Removal**
- **Method**: Drop duplicates based on timestamp and symbol
- **Implementation**: Standard pandas `drop_duplicates()` method
- **MISSING**: Explicit duplicate handling code
- **TODO**: Add `backend/utils/data_cleaner.py` with comprehensive cleaning

**Outlier Handling**
- **MISSING**: Statistical outlier detection and treatment
- **TODO**: Implement IQR-based outlier detection in feature engineering

**Corporate Actions**
- **MISSING**: Split and dividend adjustments
- **TODO**: Integrate Yahoo Finance adjusted close prices

**Feature Engineering Pipeline**
- **Technical Indicators**: RSI, MACD, Bollinger Bands calculated using ta library
- **Returns Calculation**: Log returns, percentage changes
- **Rolling Statistics**: Moving averages, volatility windows
- **Implementation**: `backend/ML_models/xgboost_model.py#L50-L100`

**Train/Validation/Test Splits**
- **Method**: Time-series aware splitting (80/10/10)
- **Implementation**: `backend/ML_models/informer_model.py#L670-L675`
- **Code**: `split_idx = int(len(data) * 0.8)`
- **Persistence**: Models saved to `models/` directory

**Evidence**
- `backend/ML_models/xgboost_model.py#L50-L120`: Feature engineering and cleaning
- `backend/ML_models/informer_model.py#L670-L675`: Train/val split logic
- `backend/data/news_fetcher.py#L220-L225`: Timezone handling
- `backend/app/config.py#L50-L70`: Data processing configurations

---

## 4. Exploratory Data Analysis (EDA)

### 4.1 Summary statistics

**MISSING**: Comprehensive EDA implementation

**Available Statistics** (from documentation):
- **Mean Daily Return**: ~0.05% (based on Tata Motors analysis)
- **Volatility**: ~2.1% daily standard deviation
- **Skewness**: Negative skew in returns distribution
- **Volume Statistics**: Average daily volume, volume ratios

**TODO**: Create `backend/notebooks/eda_analysis.ipynb` with:
- Descriptive statistics computation
- Return distribution analysis
- Correlation matrix calculation
- Volatility clustering analysis

**Evidence**
- `presentation/presentation.md#L131-L140`: Dataset characteristics mentioned
- `presentation/DETAILED_PROJECT_DOCUMENTATION.md#L296`: EDA section reference

### 4.2 Data visualizations (plots and graphs)

**Frontend Chart Components**
- **Candlestick Charts**: OHLC visualization with volume bars
  - **How this is generated**: `frontend/src/components/CandlestickChart.js#L730-L742`
  - **Library**: Chart.js with custom candlestick element
  - **Features**: Interactive zoom, technical indicator overlays

- **Line Charts**: Price trends and moving averages
  - **How this is generated**: `frontend/src/pages/StockDetail.js#L558`
  - **Library**: react-chartjs-2 with Chart.js

- **Volume Analysis Charts**: Volume bars with price correlation
  - **How this is generated**: `frontend/src/components/CandlestickChart.js#L741-L742`
  - **Library**: Chart.js Bar chart

- **Sentiment Visualization**: News sentiment trends and distribution
  - **How this is generated**: `frontend/src/components/NewsSentimentUI.js#L217-L348`
  - **Library**: Chart.js Line and Doughnut charts

- **Technical Indicators**: RSI, MACD, Bollinger Bands overlays
  - **How this is generated**: `frontend/src/components/CandlestickChart.js#L767-L768`
  - **Library**: Chart.js Line chart

**Backend Plotting** (Limited)
- **MISSING**: Matplotlib/Plotly backend visualizations
- **TODO**: Add `backend/utils/plotting.py` for server-side chart generation

**Evidence**
- `frontend/src/components/CandlestickChart.js#L17-L85`: Chart.js registration and custom elements
- `frontend/src/components/NewsSentimentUI.js#L39-L50`: Chart.js component registration
- `frontend/src/pages/StockDetail.js#L41-L50`: ChartJS registration
- `frontend/package.json#L10-L16`: Chart.js dependencies

### 4.3 Initial insights from EDA

**MISSING**: Comprehensive EDA insights

**Available Insights** (from code comments and documentation):
- **Trend Analysis**: Moving average crossovers used as trend indicators
- **Volume Correlation**: Volume spikes correlate with price movements
- **Technical Patterns**: RSI overbought/oversold levels at 70/30
- **News Impact**: Sentiment scores show correlation with price movements (mock data)

**TODO**: Add `backend/notebooks/eda_summary.ipynb` to compute:
- Rolling volatility analysis
- Autocorrelation in returns
- Seasonal patterns in trading volume
- News sentiment correlation with price movements
- Technical indicator effectiveness analysis

**Evidence**
- `presentation/presentation.md#L140-L158`: Initial insights mentioned
- `backend/core/prediction_engine.py#L453-L470`: Technical analysis factors
- `backend/ML_models/xgboost_model.py#L50-L100`: Feature importance insights

---

## 5. Methodology

### 5.1 Analytical methods and techniques used

**XGBoost (Tabular Regression)**
- **Purpose**: Price prediction using engineered features
- **Implementation**: `backend/ML_models/xgboost_model.py#L15-L200`
- **Features**: Technical indicators, fundamental ratios, sentiment scores
- **Target**: Next-day price change percentage

**Informer (Transformer-based Long-horizon Forecasting)**
- **Purpose**: Multi-step ahead price forecasting
- **Implementation**: `backend/ML_models/informer_model.py#L10-L300`
- **Architecture**: Encoder-decoder with ProbSparse attention
- **Sequence Length**: 60 time steps input, 30 steps prediction

**DQN (Deep Q-Network for Policy Learning)**
- **Purpose**: Trading action optimization (Buy/Sell/Hold)
- **Implementation**: `backend/ML_models/dqn_model.py#L15-L500`
- **State Space**: Technical indicators, price features, portfolio state
- **Action Space**: 3 discrete actions (0=Sell, 1=Hold, 2=Buy)

**Sentiment Analysis**
- **Current**: Mock keyword-based analysis
- **Planned**: FinBERT/FinGPT integration
- **Implementation**: `backend/ML_models/sentiment_model.py#L25-L100`
- **Features**: News title/content processing, sentiment scoring

**SHAP Explainability**
- **Purpose**: Model prediction explanations
- **Implementation**: `backend/ML_models/xgboost_model.py#L150-L200`
- **Method**: TreeExplainer for XGBoost feature importance

**Evidence**
- `backend/ML_models/xgboost_model.py#L15-L200`: XGBoostModel class
- `backend/ML_models/informer_model.py#L50-L150`: InformerModel architecture
- `backend/ML_models/dqn_model.py#L200-L400`: DQNAgent implementation
- `backend/ML_models/sentiment_model.py#L25-L100`: SentimentAnalyzer class

### 5.2 Justification for chosen methods

**XGBoost Selection Rationale**
- **Reason**: Excellent performance on tabular financial data
- **Advantages**: Handles missing values, feature importance, fast training
- **Use Case**: Short-term price predictions with engineered features
- **Source**: Code comments in `backend/ML_models/xgboost_model.py#L15-L25`

**Informer for Long Sequences**
- **Reason**: Designed for long-sequence time series forecasting
- **Advantages**: Efficient attention mechanism, handles long dependencies
- **Use Case**: Multi-day price forecasting
- **Source**: Model architecture in `backend/ML_models/informer_model.py#L50-L100`

**DQN for Action Optimization**
- **Reason**: Learns optimal trading policies through reinforcement
- **Advantages**: Considers transaction costs, risk management
- **Use Case**: Portfolio management and trade execution
- **Source**: Implementation in `backend/ML_models/dqn_model.py#L200-L250`

**MISSING**: Detailed methodology documentation
**TODO**: Add `docs/methodology.md` with comprehensive method justifications

**Evidence**
- `backend/ML_models/xgboost_model.py#L15-L25`: XGBoost rationale in comments
- `backend/ML_models/informer_model.py#L10-L50`: Informer architecture description
- `presentation/presentation.md#L218-L240`: Model selection reasoning

### 5.3 Algorithms/models applied (details)

**XGBoost Model**
- **Inputs**: 50+ engineered features (technical indicators, fundamentals, sentiment)
- **Targets**: Next-day price change percentage
- **Horizons**: 1-day ahead predictions
- **Loss/Metric**: MSE loss, MAE/RMSE evaluation
- **Hyperparameters**: 
  - `n_estimators`: 100
  - `max_depth`: 6
  - `learning_rate`: 0.1
  - `subsample`: 0.8
- **Training Loop**: `backend/ML_models/xgboost_model.py#L519-L570`
- **Artifacts**: Models saved to `models/xgboost_{ticker}.pkl`
- **Endpoint**: `/api/predictions/price` (`backend/api/predictions.py#L25-L50`)

**Informer Model**
- **Inputs**: OHLCV sequences (60 timesteps)
- **Targets**: Future price sequences (30 timesteps)
- **Horizons**: 1-30 days ahead
- **Loss/Metric**: MSE loss
- **Hyperparameters**:
  - `d_model`: 512
  - `n_heads`: 8
  - `num_layers`: 3
  - `seq_len`: 60
  - `batch_size`: 32
- **Training Loop**: `backend/ML_models/informer_model.py#L694-L750`
- **Artifacts**: Models saved to `models/informer_{ticker}.pth`
- **Endpoint**: Integrated in prediction service

**DQN Model**
- **Inputs**: State vector (technical indicators, portfolio state)
- **Targets**: Q-values for actions (Buy/Sell/Hold)
- **Horizons**: Real-time action selection
- **Loss/Metric**: Huber loss, reward maximization
- **Hyperparameters**:
  - `learning_rate`: 0.001
  - `epsilon`: 0.1 (exploration)
  - `gamma`: 0.95 (discount factor)
  - `batch_size`: 32
- **Training Loop**: `backend/ML_models/dqn_model.py#L357-L450`
- **Artifacts**: Models saved to `models/dqn_{ticker}.pth`
- **Endpoint**: Trading signal generation

**Ensemble Predictions**
- **MISSING**: Prediction ensemble implementation
- **TODO**: Add ensemble logic in prediction service

**Evidence**
- `backend/ML_models/xgboost_model.py#L519-L570`: XGBoost training implementation
- `backend/ML_models/informer_model.py#L694-L750`: Informer training loop
- `backend/ML_models/dqn_model.py#L357-L450`: DQN training implementation
- `backend/api/predictions.py#L25-L50`: Prediction endpoints
- `backend/app/config.py#L80-L100`: Model hyperparameters

---

## Reproducibility Appendix

**Environment Setup**
- **Python Environment**: `requirements.txt` with pinned versions
- **Docker Setup**: `docker-compose.yml` with all services
- **Environment Variables**: `.env.example` template provided

**Start Commands**
- **Backend**: `uvicorn backend.app.main:app --reload`
- **Frontend**: `npm start` (development), `npm run build` (production)
- **Full Stack**: `docker-compose up -d`

**Key Environment Variables** (from `.env.example`):
- `ANGEL_ONE_API_KEY`: Angel One SmartAPI credentials
- `NEWS_API_KEY`: NewsAPI access token
- `REDIS_URL`: Redis connection string
- `DATABASE_URL`: PostgreSQL connection string

**Evidence**
- `requirements.txt`: Python dependencies
- `docker-compose.yml#L1-L150`: Complete service configuration
- `frontend/package.json#L1-L50`: Node.js dependencies
- `backend/app/config.py#L10-L50`: Environment variable configuration

---

*This presentation document provides a comprehensive overview of the TRAE Stock Prediction System based on actual code implementation and repository evidence. All statements are anchored to specific file locations and line ranges for verification.*