# Sales Forecasting - Rossmann Drugstore

Predict daily sales for 1,115 Rossmann drugstores using 2.5 years of historical data. Built with XGBoost and advanced time series feature engineering.

##  Results

- **MAPE: 6.85%** (industry standard: <10%)
- **80% improvement** over naive baseline
- Production-ready forecasting system

##  Dataset

**Source:** [Kaggle Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales)

**Data:**
- 1,017,209 transactions across 1,115 stores
- Date range: 2013-01-01 to 2015-07-31 (942 days)
- Features: Store info, promotions, holidays, competition

**Files:**
- `train.csv`: Historical sales data
- `store.csv`: Store metadata (type, assortment, competition)

##  Key Findings

### Business Insights
1. **Business growth: +44.9%** over 2.5 years
2. **Promotions boost sales by +81%** (€4,406 → €7,991)
3. **State holidays reduce sales by -96%** (stores closed)
4. **Weekly seasonality:** Monday highest (€7,828), Sunday lowest (€205)

### Model Insights
- **Top feature:** Customers (30.3% importance)
- **Lag features matter:** 2-week lag more important than 1-day lag
- **Rolling statistics:** 7-day moving average is 2nd most important feature
- **Cyclical encoding:** Helps capture December→January transition

##  Project Structure
```
sales-forecasting/
├── sales_forecasting.ipynb    # Main analysis notebook
├── train.csv                   # Training data
├── store.csv                   # Store metadata
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

##  Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Analysis
```bash
jupyter notebook sales_forecasting.ipynb
```

##  Technical Approach

### Day 1: Time Series Exploration
- Identified weekly seasonality (7-day cycle)
- Discovered Christmas spikes (2-3x normal sales)
- Found 82% of closures on Sundays

### Day 2: Trend & Seasonality Decomposition
- Decomposed into trend + seasonal + residual components
- **Major discovery:** 44.9% business growth over 2.5 years
- Created lag features (1, 7, 14, 21, 28 days)
- Autocorrelation analysis showed lag-7 importance

### Day 3: Feature Engineering
**Created 33 features:**
- **Time features (11):** Year, month, day, week, quarter, cyclical encodings
- **Holiday features (4):** State holidays, Christmas, Easter, school holidays
- **Promotion features (2):** Regular promo, continuous promo
- **Store features (9):** Store type, assortment, competition, avg sales
- **Lag features (5):** 1, 7, 14, 21, 28 days back
- **Rolling features (2):** 7-day mean and std

**Key innovation:** Cyclical encoding (sin/cos) for months and days
- Maps December (12) and January (1) as neighbors on a circle
- Helps model understand seasonal wraparound

### Day 4: XGBoost Baseline
**Model Configuration:**
- Algorithm: XGBoost (gradient boosted trees)
- Trees: 100
- Max depth: 6
- Learning rate: 0.1

**Validation:**
- Temporal split: 80% train (2013-2015 Feb), 20% test (2015 Feb-Jul)
- No data leakage (never shuffle time series!)
- Naive baseline: Yesterday's sales (MAPE: 33.19%)

**Performance:**
| Metric | XGBoost | Naive | Improvement |
|--------|---------|-------|-------------|
| MAE    | €468    | €2,376| +80.3%      |
| RMSE   | €650    | €4,077| +84.1%      |
| MAPE   | 6.85%   | 33.19%| +79.4%      |

##  Model Performance

### Predictions vs Actual
![Scatter plot showing tight clustering around perfect prediction line]

### Feature Importance
Top 5 features:
1. **Customers** (30.3%) - More customers = more sales
2. **SalesRollingMean7** (25.1%) - Recent trend
3. **Sales_Lag14** (12.7%) - Two weeks ago
4. **SalesRollingStd7** (5.9%) - Recent volatility
5. **Promo** (4.2%) - Promotion effect

##  Lessons Learned

### Time Series Best Practices
1. **Never shuffle** - Temporal order matters
2. **Temporal validation** - Train on past, test on future
3. **Lag features are gold** - Yesterday/last week predict today
4. **Handle closed stores** - Filter Open==1 or model predicts zeros
5. **Cyclical encoding** - sin/cos for repeating features

### Feature Engineering Insights
- Recent trends (rolling mean) > individual lags
- Lag 14 > Lag 1 (surprising but true for this data)
- Store-specific features capture 9x variation in sales
- External events (promotions, holidays) are powerful

### Production Considerations
- Model needs retraining as data grows
- Feature engineering pipeline must be reproducible
- Monitor for distribution drift (new stores, changing patterns)
- Separate models per store type might improve accuracy

##  Future Improvements

### Model Enhancements
- [ ] Hyperparameter tuning (grid search)
- [ ] Separate models per StoreType (a/b/c/d)
- [ ] Deep learning (LSTM/GRU for sequence modeling)
- [ ] Ensemble: XGBoost + LightGBM + CatBoost

### Feature Engineering
- [ ] Weather data (temperature, rain affects foot traffic)
- [ ] Local events (festivals, concerts nearby)
- [ ] Competitor promotions
- [ ] Economic indicators (unemployment, GDP)
- [ ] Product-level features (categories, prices)

### Deployment
- [ ] FastAPI REST endpoint
- [ ] Docker containerization
- [ ] CI/CD pipeline
- [ ] Monitoring dashboard (track MAPE over time)
- [ ] Automated retraining

##  Technologies Used

- **Python 3.14**
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Modeling:** scikit-learn, xgboost
- **Time Series:** statsmodels

##  Requirements
```txt
pandas==2.2.0
numpy==1.26.0
matplotlib==3.8.0
seaborn==0.13.0
scikit-learn==1.4.0
xgboost==2.0.3
statsmodels==0.14.1
jupyter==1.0.0
```

## Acknowledgments

- Dataset: [Kaggle Rossmann Store Sales Competition](https://www.kaggle.com/c/rossmann-store-sales)
- Inspired by real-world retail forecasting challenges

##  Contact

David - [(https://github.com/Dereger325/sales-forecasting)]

---
