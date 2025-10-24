# app.py
# =========================================================
# 📈 STOCK PROJECTION SIMULATOR — Streamlit Dashboard (v4.2)
# (Initial + Recurring Investments + Dividends + Growth + Monte Carlo + Charts)
# =========================================================
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st
import requests

plt.style.use('seaborn-v0_8-darkgrid')
st.set_page_config(page_title="Stock Projection Simulator", layout="wide")

# ---------------------- Helpers ----------------------
def money(num):
    if num is None:
        return "N/A"
    try:
        if pd.isna(num): return "N/A"
    except Exception:
        pass
    mag, n = 0, float(num)
    while abs(n) >= 1000 and mag < 5:
        mag += 1; n /= 1000
    return f"${n:,.2f}{['','K','M','B','T'][mag]}"

@st.cache_data(show_spinner=False)
def validate_ticker(t: str) -> bool:
    """Robust validation that works even when some yf endpoints fail."""
    try:
        s = yf.Ticker(t)
        p = s.fast_info.get("last_price")
        if p is not None and np.isfinite(p):
            return True
        h = s.history(period="5d")
        return not h.empty
    except Exception:
        try:
            df = yf.download(t, period="1d", progress=False)
            return not df.empty
        except Exception:
            return False

@st.cache_data(show_spinner=False)
def get_price_fast(ticker: str):
    """Best-effort last price."""
    try:
        s = yf.Ticker(ticker)
        p = s.fast_info.get("last_price")
        if p and np.isfinite(p):
            return float(p)
        info_price = s.info.get("regularMarketPrice")
        if info_price and np.isfinite(info_price):
            return float(info_price)
        h = s.history(period="5d")["Close"].dropna()
        return float(h.iloc[-1]) if len(h) else None
    except Exception:
        try:
            h = yf.download(ticker, period="5d", progress=False)["Close"].dropna()
            return float(h.iloc[-1]) if len(h) else None
        except Exception:
            return None

@st.cache_data(show_spinner=False)
def load_px(ticker: str, period="5y") -> pd.Series:
    """Fetch a single price series robustly (Adj Close → Close → first numeric col)."""
    d = yf.download(ticker, period=period, progress=False, auto_adjust=False)
    if d is None or d.empty:
        return pd.Series(dtype=float)
    for col in ["Adj Close", "Close"]:
        if col in d.columns:
            s = d[col].dropna()
            if not s.empty:
                return s
    num = d.select_dtypes(include='number')
    if not num.empty:
        return num.iloc[:, 0].dropna()
    return pd.Series(dtype=float)

@st.cache_data(show_spinner=False)
def get_div_ttm_and_hist(ticker: str):
    """Return (div_ttm, dividends Series)."""
    try:
        s = yf.Ticker(ticker)
        div = s.dividends
        if div is None or div.empty:
            return 0.0, pd.Series(dtype=float)
        if getattr(div.index, "tz", None):
            div.index = div.index.tz_localize(None)
        cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=365)
        ttm = float(div[div.index >= cutoff].sum())
        return ttm, div
    except Exception:
        return 0.0, pd.Series(dtype=float)

@st.cache_data(show_spinner=False)
def get_info(ticker: str):
    try:
        return yf.Ticker(ticker).info
    except Exception:
        return {}

def mc_paths(last_price, mu, sigma, days, npaths, daily_yield=0.0, drift_mult=1.0, rng=None):
    """Geometric Brownian Motion paths."""
    if rng is None:
        rng = np.random.default_rng()
    dt = 1/252
    mu_adj = mu * drift_mult + daily_yield
    shock = sigma * np.sqrt(dt) * rng.standard_normal((days, npaths))
    drift = (mu_adj - 0.5 * sigma**2) * dt
    path = np.empty((days, npaths), dtype=float)
    path[0, :] = last_price
    for t in range(1, days):
        path[t] = path[t-1] * np.exp(drift + shock[t])
    return path

# ---------------------- Dynamic Ticker Suggestions ----------------------
@st.cache_data(show_spinner=False)
def suggest_tickers(query: str) -> list:
    """Return up to 10 ticker suggestions from Yahoo Finance search."""
    if not query:
        return []
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
    try:
        r = requests.get(url, timeout=3)
        data = r.json()
        symbols = []
        for q in data.get("quotes", []):
            sym = q.get("symbol")
            name = q.get("shortname") or q.get("longname") or q.get("quoteType", "")
            exch = q.get("exchangeDisplay") or ""
            if sym:
                symbols.append(f"{sym} — {name} ({exch})" if name else sym)
        return symbols[:10]
    except Exception:
        return []

# ---------------------- UI ----------------------
st.title("📈 Stock Projection Simulator — Streamlit Dashboard")
st.caption("Initial + Recurring (DCA) + Dividends + Growth + Monte Carlo")

with st.sidebar:
    st.header("1) Search Stock")

    # --- improved dynamic search ---
    query = st.text_input("🔍 Type company name or ticker", value="AAPL")

    if len(query.strip()) >= 2:
        with st.spinner("Searching..."):
            suggestions = suggest_tickers(query.strip())
    else:
        suggestions = []

    if suggestions:
        selection = st.selectbox(
            "Select matching ticker",
            suggestions,
            key="ticker_select",
            help="Choose the exact symbol"
        )
        ticker = selection.split(" — ")[0].strip().upper()
    else:
        ticker = query.strip().upper()
        if len(query.strip()) < 2:
            st.caption("Start typing to search for a stock (e.g., 'apple' → AAPL).")

    valid = validate_ticker(ticker) if ticker else False
    if not valid and ticker:
        st.error("❌ Invalid ticker or not found on Yahoo Finance.")

    # --- investment options ---
    st.header("2) Investment")
    initial = st.number_input("Initial investment ($)", min_value=0.0, value=10000.0, step=100.0)
    recur = st.toggle("Recurring investments (DCA)?", value=False)
    if recur:
        recur_amt = st.number_input("DCA amount per contribution ($)", min_value=0.0, value=500.0, step=50.0)
        freq_choice = st.selectbox("Frequency", ["Weekly (7d)", "Biweekly (14d)", "Monthly (30d)"], index=2)
        dca_days = 7 if "Weekly" in freq_choice else 14 if "Biweekly" in freq_choice else 30
        dca_months = st.number_input("Duration (months)", min_value=1, value=12, step=1)
        dca_count = int(dca_months * 30.4375 // dca_days)
    else:
        recur_amt = 0.0
        dca_days = 30
        dca_months = st.number_input("Hold period (months)", min_value=1, value=12, step=1)
        dca_count = 0

    # --- dividends ---
    st.header("3) Dividends")
    div_growth_pct = st.number_input("Expected annual dividend growth (%)", min_value=0.0, value=0.0, step=0.25)
    reinvest = st.toggle("Reinvest dividends?", value=True)

    # --- monte carlo ---
    st.header("4) Monte Carlo")
    mc_months = dca_months
    mc_days = int(mc_months * 21)
    n_paths = st.slider("Number of paths", min_value=200, max_value=5000, value=1000, step=200)
    st.caption("Trading days ≈ months × 21")

    # --- scenarios ---
    st.header("5) Scenarios")
    bearish_mult = st.number_input("🐻 Bearish drift multiplier", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
    neutral_mult = st.number_input("😐 Neutral drift multiplier", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
    bullish_mult = st.number_input("🚀 Bullish drift multiplier", min_value=0.1, max_value=3.0, value=1.5, step=0.1)

    run = st.button("Run Simulation", type="primary", disabled=not valid)

if not ticker:
    st.info("Enter or search a ticker to begin.")
    st.stop()
if not valid:
    st.stop()

# ---------------------- Data & Fundamentals ----------------------
price_now = get_price_fast(ticker)
px_5y = load_px(ticker, "5y")
px_1y = load_px(ticker, "1y")
div_ttm, div_hist = get_div_ttm_and_hist(ticker)
info = get_info(ticker)
if price_now is None and not px_5y.empty:
    price_now = float(px_5y.iloc[-1])

colA, colB, colC, colD = st.columns(4)
colA.metric("Current Price", f"${(price_now or 0):,.2f}")
colB.metric("Dividend/Share (TTM)", f"${div_ttm:,.2f}")
dy = (div_ttm / price_now) if (price_now and price_now > 0) else 0.0
colC.metric("Dividend Yield (TTM)", f"{dy*100:.2f}%")
colD.metric("Market Cap", money(info.get("marketCap")))

with st.expander("More fundamentals"):
    fa, fb, fc = st.columns(3)
    fa.write(f"**Shares Outstanding**: {money(info.get('sharesOutstanding'))}")
    fb.write(f"**Trailing P/E**: {info.get('trailingPE','N/A')}")
    fc.write(f"**52w Range**: {info.get('fiftyTwoWeekLow','?')} – {info.get('fiftyTwoWeekHigh','?')}")

# ---------------------- Simulation ----------------------
if run:
    if px_5y.empty:
        st.error("Could not load price history. Try a different ticker or later.")
        st.stop()

    rets = np.log(px_5y / px_5y.shift(1)).dropna()
    mu = float(rets.mean())
    sig = float(rets.std())
    last = float(price_now or px_5y.iloc[-1])
    daily_yield = (dy / 252.0)_
