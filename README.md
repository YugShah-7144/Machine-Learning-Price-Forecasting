# Machine Learning Forecasting of Financial Markets Using Spot and Futures Data

## Overview

This project investigates whether machine learning models can predict the directional movement of financial assets using spot and futures market data.

The study compares three structurally different asset classes:

- MSCI Emerging Markets Index (MEM)
- Gold (GLD)
- Bitcoin (BTC)

Four machine learning models were evaluated:

- Logistic Regression
- Random Forest
- XGBoost
- Long Short-Term Memory (LSTM)

Predictions were made across five forecasting horizons:

- Daily
- 3-Day
- 5-Day
- Weekly
- Biweekly

The project follows a strict walk-forward validation framework to eliminate data leakage and simulate real-world forecasting conditions.

---

## Research Questions

1. Which asset is the easiest to forecast directionally?
2. Which forecasting horizon provides the strongest predictive signal?
3. Which machine learning model performs best?

---

## Dataset

The dataset contains paired spot and futures prices from January 2021 to December 2025.

| Asset | Observations |
|---------|-------------|
| Gold | 1304 |
| Bitcoin | 1304 |
| MSCI Emerging Markets | 1063 |

---

## Feature Engineering

More than 45 features were engineered from spot and futures prices.

### Cointegration Features

- Error Correction Term (ECT)
- Basis
- Basis Change
- Z-Score Basis

### Momentum Features

- Lag Returns
- Momentum (5-day, 10-day)
- Trend Regime

### Volatility Features

- Rolling Volatility
- Volatility Ratio
- Volatility of Volatility

### Technical Indicators

- RSI
- MACD
- MACD Signal
- MACD Histogram

### Statistical Features

- Skewness
- Kurtosis
- Distance from High/Low

### Calendar Features

- Day of Week
- Month Indicators

---

## Models

### Logistic Regression

Linear baseline model with L2 regularization.

### Random Forest

Tree-based ensemble model optimized using RandomizedSearchCV.

### XGBoost

Gradient boosting framework with hyperparameter optimization using TimeSeriesSplit.

### LSTM

Recurrent neural network trained on sequential market observations.

---

## Validation Strategy

A walk-forward validation framework was used:

1. Train on historical data
2. Predict next observation
3. Expand training window
4. Retrain periodically

This approach prevents look-ahead bias and data leakage.

---

## Results

### Best Performing Configuration

| Asset | Best Model | Horizon | Accuracy |
|---------|-----------|----------|----------|
| MSCI Emerging Markets | XGBoost | Daily | 66.8% |
| Gold | Random Forest | Biweekly | 61.3% |
| Bitcoin | XGBoost | 3-Day | 52.9% |

### Predictability Ranking

1. MSCI Emerging Markets
2. Gold
3. Bitcoin

### Model Ranking

1. XGBoost
2. Random Forest
3. Logistic Regression
4. LSTM

---

## Key Findings

- Emerging Markets are significantly more predictable than Gold and Bitcoin.
- Gold exhibits stronger predictability at medium-term horizons.
- Bitcoin behaves close to an efficient market under the tested framework.
- XGBoost and Random Forest consistently outperform LSTM.
- Tree-based models are highly effective for structured financial datasets.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- TensorFlow / Keras
- Statsmodels
- Matplotlib
- Seaborn

---

## Future Improvements

- Add SHAP explainability analysis
- Compare against VECM and ARIMA models
- Evaluate trading profitability after transaction costs
- Implement LightGBM and CatBoost
- Experiment with Temporal Fusion Transformers
