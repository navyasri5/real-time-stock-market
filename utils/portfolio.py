"""
portfolio.py
Simple file-based portfolio tracker (no database required).
Stores holdings as a JSON file locally.
"""

import json
import os
import yfinance as yf
import pandas as pd
from datetime import datetime

PORTFOLIO_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "portfolio.json")
os.makedirs(os.path.dirname(PORTFOLIO_FILE), exist_ok=True)


def load_portfolio() -> dict:
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    return {}


def save_portfolio(portfolio: dict):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, indent=2)


def add_holding(symbol: str, company: str, qty: int, buy_price: float):
    portfolio = load_portfolio()
    portfolio[symbol] = {
        "company": company,
        "qty": qty,
        "buy_price": buy_price,
        "added_on": datetime.now().strftime("%Y-%m-%d")
    }
    save_portfolio(portfolio)


def remove_holding(symbol: str):
    portfolio = load_portfolio()
    portfolio.pop(symbol, None)
    save_portfolio(portfolio)


def get_portfolio_summary() -> pd.DataFrame:
    portfolio = load_portfolio()
    if not portfolio:
        return pd.DataFrame()

    rows = []
    for sym, info in portfolio.items():
        try:
            t = yf.Ticker(sym)
            hist = t.history(period="5d", interval="1d")
            hist = hist.dropna(subset=["Close"])
            if hist.empty or len(hist) == 0:
                curr_price = info["buy_price"]
            else:
                val = float(hist["Close"].iloc[-1])
                curr_price = round(val, 2) if not (val != val) else info["buy_price"]
        except Exception:
            curr_price = info["buy_price"]

        invested = info["qty"] * info["buy_price"]
        current_val = info["qty"] * curr_price
        pnl = current_val - invested
        pnl_pct = (pnl / invested) * 100

        rows.append({
            "Symbol": sym,
            "Company": info["company"],
            "Qty": info["qty"],
            "Buy Price": info["buy_price"],
            "Current Price": curr_price,
            "Invested (Rs)": round(invested, 2),
            "Current Value (Rs)": round(current_val, 2),
            "P&L (Rs)": round(pnl, 2),
            "P&L %": round(pnl_pct, 2),
        })

    return pd.DataFrame(rows)


def get_portfolio_total(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"invested": 0, "current": 0, "pnl": 0, "pnl_pct": 0}
    invested = df["Invested (Rs)"].sum()
    current = df["Current Value (Rs)"].sum()
    pnl = current - invested
    pnl_pct = (pnl / invested) * 100 if invested > 0 else 0
    return {
        "invested": round(invested, 2),
        "current": round(current, 2),
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl_pct, 2),
    }
