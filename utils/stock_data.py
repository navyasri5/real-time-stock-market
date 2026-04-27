"""
stock_data.py
Fetches and processes Indian stock market data using yfinance (NSE/BSE).
"""

import yfinance as yf
import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands
import warnings
warnings.filterwarnings("ignore")


# ─── STOCK UNIVERSE ──────────────────────────────────────────────────────────

NIFTY50_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "WIPRO.NS",
    "HCLTECH.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS", "BAJFINANCE.NS", "NESTLEIND.NS",
    "POWERGRID.NS", "NTPC.NS", "TITAN.NS", "TECHM.NS", "BAJAJFINSV.NS",
    "ONGC.NS", "JSWSTEEL.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "ADANIPORTS.NS",
    "COALINDIA.NS", "DRREDDY.NS", "INDUSINDBK.NS", "GRASIM.NS", "CIPLA.NS",
    "DIVISLAB.NS", "APOLLOHOSP.NS", "BRITANNIA.NS", "EICHERMOT.NS", "BPCL.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "M&M.NS", "SHRIRAMFIN.NS", "SBILIFE.NS",
    "TATACONSUM.NS", "HDFCLIFE.NS", "ADANIENT.NS", "BAJAJ-AUTO.NS", "UPL.NS"
]

SECTOR_MAP = {
    "RELIANCE.NS": "Energy", "TCS.NS": "IT", "HDFCBANK.NS": "Banking",
    "INFY.NS": "IT", "ICICIBANK.NS": "Banking", "HINDUNILVR.NS": "FMCG",
    "ITC.NS": "FMCG", "SBIN.NS": "Banking", "BHARTIARTL.NS": "Telecom",
    "KOTAKBANK.NS": "Banking", "LT.NS": "Infrastructure", "AXISBANK.NS": "Banking",
    "ASIANPAINT.NS": "Paints", "MARUTI.NS": "Auto", "WIPRO.NS": "IT",
    "HCLTECH.NS": "IT", "SUNPHARMA.NS": "Pharma", "ULTRACEMCO.NS": "Cement",
    "BAJFINANCE.NS": "NBFC", "NESTLEIND.NS": "FMCG", "POWERGRID.NS": "Utilities",
    "NTPC.NS": "Utilities", "TITAN.NS": "Consumer", "TECHM.NS": "IT",
    "BAJAJFINSV.NS": "NBFC", "ONGC.NS": "Energy", "JSWSTEEL.NS": "Metal",
    "TATAMOTORS.NS": "Auto", "TATASTEEL.NS": "Metal", "ADANIPORTS.NS": "Infrastructure",
    "COALINDIA.NS": "Mining", "DRREDDY.NS": "Pharma", "INDUSINDBK.NS": "Banking",
    "GRASIM.NS": "Diversified", "CIPLA.NS": "Pharma", "DIVISLAB.NS": "Pharma",
    "APOLLOHOSP.NS": "Healthcare", "BRITANNIA.NS": "FMCG", "EICHERMOT.NS": "Auto",
    "BPCL.NS": "Energy", "HEROMOTOCO.NS": "Auto", "HINDALCO.NS": "Metal",
    "M&M.NS": "Auto", "SHRIRAMFIN.NS": "NBFC", "SBILIFE.NS": "Insurance",
    "TATACONSUM.NS": "FMCG", "HDFCLIFE.NS": "Insurance", "ADANIENT.NS": "Conglomerate",
    "BAJAJ-AUTO.NS": "Auto", "UPL.NS": "Agro"
}

COMPANY_NAMES = {
    "RELIANCE.NS": "Reliance Industries", "TCS.NS": "Tata Consultancy Services",
    "HDFCBANK.NS": "HDFC Bank", "INFY.NS": "Infosys", "ICICIBANK.NS": "ICICI Bank",
    "HINDUNILVR.NS": "Hindustan Unilever", "ITC.NS": "ITC Limited",
    "SBIN.NS": "State Bank of India", "BHARTIARTL.NS": "Bharti Airtel",
    "KOTAKBANK.NS": "Kotak Mahindra Bank", "LT.NS": "Larsen & Toubro",
    "AXISBANK.NS": "Axis Bank", "ASIANPAINT.NS": "Asian Paints",
    "MARUTI.NS": "Maruti Suzuki", "WIPRO.NS": "Wipro",
    "HCLTECH.NS": "HCL Technologies", "SUNPHARMA.NS": "Sun Pharmaceutical",
    "ULTRACEMCO.NS": "UltraTech Cement", "BAJFINANCE.NS": "Bajaj Finance",
    "NESTLEIND.NS": "Nestle India", "POWERGRID.NS": "Power Grid Corp",
    "NTPC.NS": "NTPC", "TITAN.NS": "Titan Company", "TECHM.NS": "Tech Mahindra",
    "BAJAJFINSV.NS": "Bajaj Finserv", "ONGC.NS": "ONGC",
    "JSWSTEEL.NS": "JSW Steel", "TATAMOTORS.NS": "Tata Motors",
    "TATASTEEL.NS": "Tata Steel", "ADANIPORTS.NS": "Adani Ports",
    "COALINDIA.NS": "Coal India", "DRREDDY.NS": "Dr Reddy's Laboratories",
    "INDUSINDBK.NS": "IndusInd Bank", "GRASIM.NS": "Grasim Industries",
    "CIPLA.NS": "Cipla", "DIVISLAB.NS": "Divi's Laboratories",
    "APOLLOHOSP.NS": "Apollo Hospitals", "BRITANNIA.NS": "Britannia Industries",
    "EICHERMOT.NS": "Eicher Motors", "BPCL.NS": "BPCL",
    "HEROMOTOCO.NS": "Hero MotoCorp", "HINDALCO.NS": "Hindalco Industries",
    "M&M.NS": "Mahindra & Mahindra", "SHRIRAMFIN.NS": "Shriram Finance",
    "SBILIFE.NS": "SBI Life Insurance", "TATACONSUM.NS": "Tata Consumer Products",
    "HDFCLIFE.NS": "HDFC Life Insurance", "ADANIENT.NS": "Adani Enterprises",
    "BAJAJ-AUTO.NS": "Bajaj Auto", "UPL.NS": "UPL"
}


# ─── DATA FETCHERS ────────────────────────────────────────────────────────────

def fetch_stock_data(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV data for a symbol and add technical indicators."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty or len(df) < 30:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)
        df = add_technical_indicators(df)
        return df
    except Exception:
        return pd.DataFrame()


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, MACD, Bollinger Bands, EMA to dataframe."""
    try:
        # RSI
        rsi = RSIIndicator(close=df["Close"], window=14)
        df["RSI"] = rsi.rsi()

        # MACD
        macd = MACD(close=df["Close"])
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
        df["MACD_Hist"] = macd.macd_diff()

        # Bollinger Bands
        bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
        df["BB_Upper"] = bb.bollinger_hband()
        df["BB_Lower"] = bb.bollinger_lband()
        df["BB_Mid"] = bb.bollinger_mavg()

        # EMA
        ema20 = EMAIndicator(close=df["Close"], window=20)
        ema50 = EMAIndicator(close=df["Close"], window=50)
        df["EMA_20"] = ema20.ema_indicator()
        df["EMA_50"] = ema50.ema_indicator()

        # Returns
        df["Daily_Return"] = df["Close"].pct_change()
        df["5D_Return"] = df["Close"].pct_change(5)
        df["20D_Return"] = df["Close"].pct_change(20)

        # Volatility
        df["Volatility"] = df["Daily_Return"].rolling(20).std() * np.sqrt(252)

        df.dropna(inplace=True)
    except Exception:
        pass
    return df


def get_current_snapshot(symbols: list = None) -> pd.DataFrame:
    """Return a dataframe with current price, change%, volume for multiple symbols.
    Uses batch download for speed and reliability on cloud deployments."""
    if symbols is None:
        symbols = NIFTY50_SYMBOLS

    rows = []

    # Batch download all symbols at once - much faster and avoids rate limits
    try:
        batch = yf.download(
            symbols,
            period="5d",
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=False,
        )

        for sym in symbols:
            try:
                # Handle both single and multi-ticker download structure
                if len(symbols) == 1:
                    hist = batch
                else:
                    if sym not in batch.columns.get_level_values(0):
                        continue
                    hist = batch[sym].dropna(how="all")

                if hist is None or len(hist) < 2:
                    continue

                hist = hist.dropna(subset=["Close"])
                if len(hist) < 2:
                    continue

                prev_close = float(hist["Close"].iloc[-2])
                curr_close = float(hist["Close"].iloc[-1])

                if prev_close == 0 or np.isnan(prev_close) or np.isnan(curr_close):
                    continue

                change_pct = ((curr_close - prev_close) / prev_close) * 100
                volume = hist["Volume"].iloc[-1]
                volume = int(volume) if not np.isnan(volume) else 0

                rows.append({
                    "Symbol": sym,
                    "Company": COMPANY_NAMES.get(sym, sym),
                    "Sector": SECTOR_MAP.get(sym, "Unknown"),
                    "Price": round(curr_close, 2),
                    "Prev_Close": round(prev_close, 2),
                    "Change_Pct": round(change_pct, 2),
                    "Volume": volume,
                })
            except Exception:
                continue

    except Exception:
        # Fallback: fetch individually if batch fails
        for sym in symbols:
            try:
                t = yf.Ticker(sym)
                hist = t.history(period="5d", interval="1d")
                hist = hist.dropna(subset=["Close"])
                if len(hist) < 2:
                    continue
                prev_close = float(hist["Close"].iloc[-2])
                curr_close = float(hist["Close"].iloc[-1])
                if prev_close == 0 or np.isnan(prev_close) or np.isnan(curr_close):
                    continue
                change_pct = ((curr_close - prev_close) / prev_close) * 100
                volume = int(hist["Volume"].iloc[-1]) if not np.isnan(hist["Volume"].iloc[-1]) else 0
                rows.append({
                    "Symbol": sym,
                    "Company": COMPANY_NAMES.get(sym, sym),
                    "Sector": SECTOR_MAP.get(sym, "Unknown"),
                    "Price": round(curr_close, 2),
                    "Prev_Close": round(prev_close, 2),
                    "Change_Pct": round(change_pct, 2),
                    "Volume": volume,
                })
            except Exception:
                continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["Change_Pct", "Price"])
    return df.sort_values("Change_Pct", ascending=False).reset_index(drop=True)


def get_top_gainers_losers(n: int = 10) -> tuple:
    """Return top-n gainers and losers DataFrames."""
    df = get_current_snapshot()
    gainers = df.head(n)
    losers = df.tail(n).sort_values("Change_Pct")
    return gainers, losers


def get_index_data() -> dict:
    """Fetch NIFTY 50 and SENSEX index data."""
    indices = {"NIFTY 50": "^NSEI", "SENSEX": "^BSESN", "NIFTY BANK": "^NSEBANK"}
    result = {}
    for name, sym in indices.items():
        try:
            t = yf.Ticker(sym)
            hist = t.history(period="7d", interval="1d")
            hist = hist.dropna(subset=["Close"])
            if len(hist) < 2:
                result[name] = {"price": 0.0, "change_pct": 0.0}
                continue
            curr = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2])
            if prev == 0 or np.isnan(prev) or np.isnan(curr):
                result[name] = {"price": 0.0, "change_pct": 0.0}
                continue
            chg = ((curr - prev) / prev) * 100
            result[name] = {"price": round(curr, 2), "change_pct": round(chg, 2)}
        except Exception:
            result[name] = {"price": 0.0, "change_pct": 0.0}
    return result
