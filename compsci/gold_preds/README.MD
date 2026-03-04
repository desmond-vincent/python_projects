# GoldSight - Gold Price Prediction ML App

A machine learning application that predicts next-day gold futures prices using Random Forest and Linear Regression models trained on 10 years of historical market data.

---

## Overview

GoldSight fetches daily gold futures data (GC=F) from Yahoo Finance, engineers technical indicators from historical price action, and trains two models to predict the next day log return. The app outputs dual evaluation metrics separating return-domain accuracy from price-domain visualization context, and generates a 10-year price history chart and a 120-day model performance zoom.

---

## Project Structure

```
GoldApp/
|
|-- src/
|   `-- __init__.py       # All application logic
|
|-- main.py               # Entry point
|-- requirements.txt      # Dependencies
`-- README.md
```

---

## Features

- Pulls 10 years of gold futures data via yfinance
- Engineers technical indicators: RSI, MACD, Bollinger Band Width, Log Returns, Volatility
- Trains Random Forest and Linear Regression models on stationary log return features
- Dual evaluation: return-domain metrics as primary, price-domain as visualization context only
- Outputs MAE, RMSE, and R2 in both return and price domains with persistence bias warning
- Generates two charts saved as PNG files

---

## Models Used

| Model | Role |
|---|---|
| Random Forest Regressor | Primary model - captures non-linear patterns |
| Linear Regression | Baseline model - linear trend benchmark |

Both models are trained on engineered features only. Raw OHLCV columns are dropped before training to prevent data leakage.

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| yfinance | 1.2.0 | Gold futures data |
| pandas | 2.2.3 | Data manipulation |
| numpy | 2.0.2 | Numerical computing |
| scikit-learn | 1.6.1 | ML models and metrics |
| matplotlib | 3.9.4 | Charting |
| seaborn | 0.13.2 | Feature importance visualization |

---

## Requirements

- Python 3.12.12

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the App

From the root GoldApp/ directory:

```bash
python main.py
```

---

## Output

**Console:**
```
Training samples: 1985
Testing samples:  497

--- PRIMARY EVALUATION: Return Domain (Honest) ---
Model                | MAE          | RMSE         | R2 Score
-----------------------------------------------------------------
Random Forest        | X.XXXXXX     | X.XXXXXX     | X.XXXX
Linear Regression    | X.XXXXXX     | X.XXXXXX     | X.XXXX

--- SECONDARY EVALUATION: Price Domain (Visualization Context Only) ---
NOTE: Price R2 is inflated by persistence bias.
Model                | MAE (USD)    | RMSE (USD)   | R2 Score
-----------------------------------------------------------------
Random Forest        | .XX       | .XX       | X.XXXX
Linear Regression    | .XX       | .XX       | X.XXXX
```

**Files saved to root directory:**
- actual_vs_predicted.png - 10-year history + 120-day model zoom with return-domain metrics overlay
- feature_importance.png - Top 10 RF feature importances

---

## Evaluation Strategy

GoldSight uses a dual evaluation approach to avoid common financial ML pitfalls:

**Primary: Return Domain**
Models are evaluated directly on predicted log returns vs actual log returns. R2 near zero or slightly negative is the expected Efficient Market Hypothesis result for daily gold returns and does not indicate model failure. It confirms that daily gold prices are highly efficient and difficult to predict from technical indicators alone.

**Secondary: Price Domain (context only)**
Predicted log returns are reconstructed back to USD prices for visualization. Price-domain R2 is intentionally excluded as a primary metric because it is inflated by persistence bias: because daily returns are approximately 0.5%, reconstructing prices by multiplying todays price by exp(predicted return) yields a prediction nearly identical to todays price regardless of model quality. The variance of the price component swallows the models return prediction errors, producing artificially high R2. A model that always predicts zero return would achieve a price R2 near 0.99 - that is persistence, not skill.

---

## Feature Engineering

| Feature | Description |
|---|---|
| Log_Return | Daily log return of closing price |
| Return_Lag_1/3/5 | Lagged log returns (1, 3, 5 days) |
| RSI_14 | 14-day Relative Strength Index |
| MACD_Hist | MACD histogram (12/26/9 EMA) |
| BB_Width | Bollinger Band width (20-day) |
| Vol_10 | 10-day rolling return volatility |

---

## Known Limitations

- Random Forest cannot extrapolate beyond its training price range - performs best when test prices fall within the historical range seen during training
- Price-domain R2 is inflated by persistence bias - return-domain R2 near zero is the honest Efficient Market result and does not indicate model failure
- Next-day gold price prediction is inherently difficult due to the near-random walk behavior of daily financial returns
- Feature set relies solely on price-derived technical indicators - macroeconomic inputs could improve signal quality
- Model is trained for educational purposes only - not intended for live trading

---

## Dataset

- **Source:** Yahoo Finance via yfinance
- **Ticker:** GC=F (Gold Futures)
- **Range:** 2015-01-01 to 2025-01-01
- **Frequency:** Daily
- **Records:** ~2,500 trading days

---

## License

This project is intended for academic and educational use only. Yahoo Finance data is subject to Yahoo terms of use.