# StockSaathi - Indian Stock Market Intelligence Platform

A professional, beginner-friendly stock market analysis tool for Indian markets (NSE/BSE).
Built with Python, Streamlit, yfinance, and explainable AI (SHAP + LIME).

---

## Features

- Live NSE/BSE market data via yfinance (no paid API required)
- NIFTY 50 market overview with sector heatmap
- Candlestick charts with EMA, RSI, MACD, and Bollinger Bands
- AI prediction of next-day price direction (Gradient Boosting model)
- SHAP explainability - understand what drives the model overall
- LIME explainability - understand why a specific prediction was made
- Top gainers and losers with visual charts
- Personal portfolio tracker with live P&L
- Plain-language educational content for beginners
- Professional dark-themed UI (no emojis, no clutter)

---

## Setup

### 1. Clone or download the project

```
indian_stock_advisor/
  app.py                   - Streamlit application (main entry point)
  requirements.txt         - All Python dependencies
  utils/
    stock_data.py          - Data fetching and technical indicators
    ml_model.py            - Model training, SHAP, LIME
    portfolio.py           - Portfolio tracking
    education.py           - Educational content and glossary
  notebooks/
    analysis.ipynb         - Jupyter notebook for exploration
  models/                  - Saved model artifacts (auto-created)
  data/                    - Portfolio data (auto-created)
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

### 5. Run the Jupyter notebook (optional)

```bash
cd notebooks
jupyter notebook analysis.ipynb
```

---

## How the AI Prediction Works

1. The app fetches 6-12 months of historical OHLCV data for the selected stock.
2. Technical indicators (RSI, MACD, EMA, Bollinger Bands, volatility) are computed.
3. A Gradient Boosting classifier is trained to predict whether tomorrow's closing price
   will be higher or lower than today's.
4. SHAP explains which indicators most influence the model's decisions overall.
5. LIME explains why the model predicted UP or DOWN for the most recent data point.

**Important:** Predictions are probability estimates based on historical patterns.
The stock market is inherently unpredictable. This tool is for education and
research, not for trading decisions.

---

## Data Sources

All market data is fetched from Yahoo Finance via the `yfinance` library.
No API key or subscription is required.

Market hours for NSE/BSE: Monday to Friday, 9:15 AM to 3:30 PM IST.
Data may be delayed or unavailable outside trading hours.

---

## Disclaimer

StockSaathi is an educational tool. Nothing in this application constitutes
financial advice. Past performance does not guarantee future results.
Always consult a SEBI-registered investment advisor before making any
investment decision.
