# Migration Table: Mock to Real Implementations

## 🎯 CRITICAL PRIORITY MIGRATIONS

### 1. ML Model Classes (CRITICAL - Week 1-4)

| Current Mock Class | File Location | Real Implementation Needed | Dependencies | Estimated Hours |
|-------------------|---------------|---------------------------|--------------|----------------|
| `MockXGBoostModel` | `backend/core/prediction_engine.py:25-45` | Real XGBoost with feature engineering | xgboost, pandas, numpy | 40h |
| `MockInformerModel` | `backend/core/prediction_engine.py:47-67` | Transformer-based time series model | torch, transformers | 60h |
| `MockDQNModel` | `backend/core/prediction_engine.py:69-89` | Deep Q-Network for trading | stable-baselines3, gym | 50h |

### 2. Training Functions (CRITICAL - Week 2-3)

| Current Mock Function | File Location | Real Implementation Needed | Dependencies | Estimated Hours |
|----------------------|---------------|---------------------------|--------------|----------------|
| `prepare_training_data()` | `backend/core/train_models.py:15-45` | Real feature engineering pipeline | pandas, ta, sklearn | 25h |
| `create_target_variables()` | `backend/core/train_models.py:47-65` | Real target creation logic | pandas, numpy | 15h |
| `train_xgboost_model()` | `backend/core/train_models.py:67-85` | Real XGBoost training | xgboost, optuna | 30h |
| `train_informer_model()` | `backend/core/train_models.py:87-105` | Real Transformer training | torch, transformers | 45h |
| `train_dqn_model()` | `backend/core/train_models.py:107-125` | Real RL environment setup | stable-baselines3, gym | 40h |

### 3. Sentiment Analysis (CRITICAL - Week 1-2)

| Current Mock Function | File Location | Real Implementation Needed | Dependencies | Estimated Hours |
|----------------------|---------------|---------------------------|--------------|----------------|
| `mock_sentiment_analysis()` | `backend/core/news_processor.py:25-35` | FinBERT or similar model | transformers, torch | 30h |
| `analyze_news_sentiment()` | `backend/core/news_processor.py:85-105` | Real sentiment pipeline | transformers, nltk | 20h |

## 🟡 HIGH PRIORITY MIGRATIONS

### 4. Prediction Functions (HIGH - Week 3-4)

| Current Mock Function | File Location | Real Implementation Needed | Dependencies | Estimated Hours |
|----------------------|---------------|---------------------------|--------------|----------------|
| `generate_price_predictions()` | `backend/core/prediction_engine.py:251-320` | Real ensemble prediction logic | All ML models | 35h |
| `generate_trading_signals()` | `backend/core/prediction_engine.py:322-380` | Real signal generation | All ML models | 25h |
| `explain_predictions()` | `backend/core/prediction_engine.py:382-420` | Real SHAP integration | shap, matplotlib | 20h |

### 5. WebSocket Data Streaming (HIGH - Week 5-6)

| Current Mock Function | File Location | Real Implementation Needed | Dependencies | Estimated Hours |
|----------------------|---------------|---------------------------|--------------|----------------|
| `_generate_mock_price()` | `backend/app/services/websocket_service.py:180-200` | Real-time Angel One integration | Angel One API | 15h |
| `_generate_mock_predictions()` | `backend/app/services/websocket_service.py:202-230` | Real prediction streaming | ML models | 20h |
| `_generate_mock_technical()` | `backend/app/services/websocket_service.py:232-250` | Real technical indicator calc | ta library | 10h |

### 6. Portfolio Management (HIGH - Week 6-7)

| Current Mock Function | File Location | Real Implementation Needed | Dependencies | Estimated Hours |
|----------------------|---------------|---------------------------|--------------|----------------|
| `get_portfolio_summary()` | `backend/app/services/market_service.py:600-650` | Real portfolio tracking | Database, Angel One | 30h |
| `calculate_portfolio_metrics()` | `backend/app/services/market_service.py:652-698` | Real performance metrics | pandas, numpy | 20h |

## 🟢 MEDIUM PRIORITY MIGRATIONS

### 7. Advanced Analytics (MEDIUM - Week 8-9)

| Current Mock Function | File Location | Real Implementation Needed | Dependencies | Estimated Hours |
|----------------------|---------------|---------------------------|--------------|----------------|
| `fundamental_analysis()` | `backend/core/fundamental_analyzer.py:*` | Real fundamental analysis | Alpha Vantage API | 40h |
| `risk_assessment()` | `backend/core/risk_manager.py:*` | Real risk calculations | pandas, numpy | 25h |

### 8. Performance Optimization (MEDIUM - Week 10)

| Current Mock Function | File Location | Real Implementation Needed | Dependencies | Estimated Hours |
|----------------------|---------------|---------------------------|--------------|----------------|
| Model caching | `backend/services/performance_service.py:*` | Redis-based model caching | redis, joblib | 15h |
| Batch processing | `backend/services/performance_service.py:*` | Batch prediction optimization | asyncio, concurrent | 20h |

## 📁 FOLDER RESTRUCTURING PLAN

### Current Structure Issues:
```
backend/
├── core/           # Mixed real/mock implementations
├── services/       # Mixed real/mock implementations  
├── app/services/   # Duplicate services folder
└── data/           # Mixed real/mock implementations
```

### Proposed New Structure:
```
backend/
├── core/
│   ├── models/          # Real ML model implementations
│   │   ├── xgboost_model.py
│   │   ├── informer_model.py
│   │   └── dqn_model.py
│   ├── training/        # Real training pipeline
│   │   ├── data_preparation.py
│   │   ├── feature_engineering.py
│   │   └── model_trainer.py
│   ├── prediction/      # Real prediction engine
│   │   ├── ensemble_predictor.py
│   │   ├── signal_generator.py
│   │   └── explanation_engine.py
│   └── analysis/        # Real analysis modules
│       ├── sentiment_analyzer.py
│       ├── fundamental_analyzer.py
│       └── risk_manager.py
├── services/            # Consolidated services
│   ├── market_service.py     # Keep existing (mostly real)
│   ├── websocket_service.py  # Update with real data
│   ├── news_service.py       # Update with real sentiment
│   └── portfolio_service.py  # New real implementation
├── data/                # Data access layer
│   ├── angel_one_client.py   # Keep existing (real)
│   ├── news_fetcher.py       # Keep existing (real)
│   └── database.py           # Enhanced database operations
└── utils/               # Utilities and helpers
    ├── config.py            # Keep existing
    ├── performance.py       # Enhanced performance utils
    └── validators.py        # New validation utilities
```

## 🔄 IMPORT UPDATES REQUIRED

### Files Requiring Import Changes:

| File | Current Imports | New Imports | Priority |
|------|----------------|-------------|----------|
| `backend/api/predictions.py` | `from ..core.prediction_engine import *` | `from ..core.prediction.ensemble_predictor import *` | HIGH |
| `backend/api/training.py` | `from ..core.train_models import *` | `from ..core.training.model_trainer import *` | HIGH |
| `backend/app/main.py` | Various core imports | Updated core imports | HIGH |
| `backend/app/services/websocket_service.py` | Mock prediction imports | Real prediction imports | HIGH |

## 📋 MIGRATION CHECKLIST

### Phase 1: Foundation (Week 1-2)
- [ ] Create new folder structure
- [ ] Implement real XGBoost model
- [ ] Implement real sentiment analysis
- [ ] Update core imports
- [ ] Test basic prediction pipeline

### Phase 2: Advanced Models (Week 3-4)
- [ ] Implement Informer model
- [ ] Implement DQN model
- [ ] Create ensemble prediction logic
- [ ] Implement SHAP explanations
- [ ] Test full prediction pipeline

### Phase 3: Real-time Integration (Week 5-6)
- [ ] Update WebSocket service with real data
- [ ] Implement real-time prediction streaming
- [ ] Create portfolio management service
- [ ] Test real-time functionality

### Phase 4: Optimization (Week 7-8)
- [ ] Implement model caching
- [ ] Add batch processing
- [ ] Optimize database queries
- [ ] Add comprehensive logging
- [ ] Performance testing

## 🧪 TESTING STRATEGY

### Unit Tests Required:
- [ ] ML model training and prediction tests
- [ ] Sentiment analysis accuracy tests
- [ ] WebSocket data streaming tests
- [ ] Portfolio calculation tests
- [ ] API endpoint integration tests

### Integration Tests Required:
- [ ] End-to-end prediction pipeline
- [ ] Real-time data flow testing
- [ ] Database integration testing
- [ ] Angel One API integration testing

## 📊 SUCCESS METRICS

### Technical Metrics:
- [ ] 0% mock implementations in production code
- [ ] >95% test coverage for new implementations
- [ ] <2s response time for predictions
- [ ] <100ms WebSocket message latency

### Business Metrics:
- [ ] Prediction accuracy >60% for daily predictions
- [ ] Sentiment analysis correlation >0.7 with market movements
- [ ] Real-time data latency <5 seconds
- [ ] System uptime >99.5%

## 🚨 RISK MITIGATION

### High-Risk Areas:
1. **ML Model Performance**: Implement gradual rollout with A/B testing
2. **Real-time Data Reliability**: Maintain fallback to cached/mock data
3. **API Rate Limits**: Implement proper rate limiting and caching
4. **Memory Usage**: Monitor model memory consumption in production

### Rollback Plan:
- Keep mock implementations as fallback during migration
- Implement feature flags for easy rollback
- Maintain comprehensive logging for debugging
- Create automated health checks

## 📅 TIMELINE SUMMARY

| Week | Focus Area | Deliverables | Risk Level |
|------|------------|--------------|------------|
| 1-2 | Core ML Models | XGBoost + Sentiment | HIGH |
| 3-4 | Advanced Models | Informer + DQN | HIGH |
| 5-6 | Real-time Integration | WebSocket + Portfolio | MEDIUM |
| 7-8 | Optimization | Caching + Performance | LOW |

**Total Estimated Hours**: 565 hours (~14 weeks with 1 developer, ~7 weeks with 2 developers)

**Recommended Team**: 2 developers (1 ML specialist, 1 Backend specialist)