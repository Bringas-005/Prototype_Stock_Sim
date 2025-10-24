# =========================================================
# 📈 STOCK PROJECTION SIMULATOR — Streamlit Dashboard (v5.0)
# (Initial + Recurring Investments + Dividends + Growth + Monte Carlo + Cashflow)
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

plt.style.use("seaborn-v0_8-darkgrid")
st.set_page_config(page_title="Stock Projection Simulator", layout="wide")

# =========================================================
# 🔧 HELPERS
# =========================================================
def money(num):
    if num is None or pd.isna(num):
        return "N/A"
    mag, n = 0, float(num)
    while abs(n) >= 1000 and mag < 5:
        mag += 1
        n /= 1000
    return f"${n:,.2f}{['','K','M','B','T'][mag]}"

@st.cache_data(show_spinner=False)
def validate_ticker(t: str) -> bool:
    try:
        df = yf.download(t, period="1d", progress=False)
        return not df.empty
    except Exception:
        return False

@st.cache_data(show_spinner=False)
def get_price(t):
    try:
        s = yf.Ticker(t)
        p = s.fast_info.get("last_price") or s.info.get("regularMarketPrice")
        return float(p) if p else None
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def div_ttm(stock, price):
    try:
        div = stock.dividends
        if div is None or div.empty or not price:
            return 0, 0
        if getattr(div.index, "tz", None):
            div.index = div.index.tz_localize(None)
        cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=365)
        ttm = float(div[div.index >= cutoff].sum())
        return ttm, ttm / price if price > 0 else 0
    except Exception:
        return 0, 0

@st.cache_data(show_spinner=False)
def load_px(t, period="5y"):
    d = yf.download(t, period=period, progress=False)
    if "Adj Close" in d.columns:
        return d["Adj Close"].dropna()
    if "Close" in d.columns:
        return d["Close"].dropna()
    return d.select_dtypes(include="number").iloc[:, 0].dropna()

@st.cache_data(show_spinner=False)
def suggest_tickers(query: str) -> list:
    """Return up to 10 ticker suggestions from Yahoo Finance."""
    if not query:
        return []
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
    try:
        res = requests.get(url, timeout=3).json()
        return [
            f"{r['symbol']} — {r.get('shortname','')[:30]} ({r.get('exchangeDisplay','')})"
            for r in res.get("quotes", [])
            if "symbol" in r
        ][:10]
    except Exception:
        return []

# =========================================================
# 🎛️ DASHBOARD UI
# =========================================================
st.title("📈 Stock Projection Simulator — Streamlit Dashboard")
st.caption("Initial + DCA + Dividends + Growth + Monte Carlo + Cashflow")

with st.sidebar:
    st.header("1️⃣ Search Stock")

    query = st.text_input("🔍 Type company name or ticker", value="AAPL")
    if len(query.strip()) >= 2:
        with st.spinner("Searching..."):
            suggestions = suggest_tickers(query.strip())
    else:
        suggestions = []

    if suggestions:
        selection = st.selectbox("Select ticker", suggestions, index=0)
        ticker = selection.split(" — ")[0].strip().upper()
    else:
        ticker = query.strip().upper()
        if len(query.strip()) < 2:
            st.caption("Start typing to search for a stock (e.g., 'apple' → AAPL).")

    valid = validate_ticker(ticker) if ticker else False
    if not valid and ticker:
        st.error("❌ Invalid ticker or not found on Yahoo Finance.")

    st.header("2️⃣ Investment Settings")
    invest = st.number_input("💵 Initial investment ($)", min_value=0.0, value=10000.0, step=100.0)

    recur = st.toggle("Enable recurring investments (DCA)?", value=False)
    if recur:
        recur_amt = st.number_input("Amount per contribution ($)", min_value=0.0, value=500.0, step=50.0)
        freq_choice = st.selectbox("Frequency", ["Weekly (7d)", "Biweekly (14d)", "Monthly (30d)"], index=2)
        dca_days = 7 if "Weekly" in freq_choice else 14 if "Biweekly" in freq_choice else 30
        dca_months = st.number_input("Duration (months)", min_value=1, value=12)
        dca_count = int(dca_months * 30.4375 // dca_days)
    else:
        recur_amt, dca_days, dca_months, dca_count = 0, 30, st.number_input("Hold period (months)", 1, value=12), 0

    st.header("3️⃣ Dividends & Growth")
    div_growth = st.number_input("Expected annual dividend growth (%)", min_value=0.0, value=0.0, step=0.25) / 100
    reinvest = st.toggle("Reinvest dividends?", value=True)

    st.header("4️⃣ Monte Carlo Parameters")
    n_paths = st.slider("Number of simulations", min_value=500, max_value=5000, value=1000, step=500)
    st.caption("~21 trading days per month.")

    run = st.button("▶️ Run Simulation", type="primary", disabled=not valid)

if not ticker or not valid:
    st.stop()

# =========================================================
# 📊 FUNDAMENTALS
# =========================================================
s = yf.Ticker(ticker)
price = get_price(ticker)
dps, dy = div_ttm(s, price)
info = s.info

col1, col2, col3, col4 = st.columns(4)
col1.metric("Price", f"${price:,.2f}")
col2.metric("Div/Share (TTM)", f"${dps:.2f}")
col3.metric("Div Yield", f"{dy*100:.2f}%")
col4.metric("Market Cap", money(info.get("marketCap")))

# =========================================================
# 🚀 RUN SIMULATION
# =========================================================
if run:
    px = load_px(ticker, "5y")
    rets = np.log(px / px.shift(1)).dropna()
    mu, sig = rets.mean(), rets.std()
    last = price or px.iloc[-1]
    FWD_DAYS = int(dca_months * 21)

    # Monte Carlo simulation
    dt = 1 / 252
    randn = np.random.normal(size=(FWD_DAYS, n_paths))
    scen = {"🐻 Bearish": 0.5, "😐 Neutral": 1.0, "🚀 Bullish": 1.5}
    res = {}
    for lab, m in scen.items():
        mu_a = mu * m + (dy / 252 if reinvest else 0)
        drift = (mu_a - 0.5 * sig**2) * dt
        shock = sig * np.sqrt(dt) * randn
        path = np.zeros((FWD_DAYS, n_paths))
        path[0, :] = last
        for t in range(1, FWD_DAYS):
            path[t] = path[t - 1] * np.exp(drift + shock[t])
        res[lab] = {
            "p10": np.percentile(path, 10, axis=1),
            "p50": np.percentile(path, 50, axis=1),
            "p90": np.percentile(path, 90, axis=1),
        }
    proj_idx = pd.date_range(px.index[-1] + pd.Timedelta(days=1), periods=FWD_DAYS, freq="B")

    total_contrib = invest + (recur_amt * dca_count if recur else 0)

    # =========================================================
    # 💵 CASHFLOW CALCULATIONS
    # =========================================================
    neutral = pd.Series(res["😐 Neutral"]["p50"], index=proj_idx).asfreq("D").interpolate()
    divh = s.dividends
    if divh is not None and not divh.empty and getattr(divh.index, "tz", None):
        divh.index = divh.index.tz_localize(None)
    last_div = float(divh.iloc[-1]) if (divh is not None and not divh.empty) else (dps / 4 if dps > 0 else 0)

    today = pd.Timestamp.today().normalize()
    end = today + pd.Timedelta(days=int(dca_months * 30.4375))
    next_pay = today + pd.Timedelta(days=90)
    pays = []
    while next_pay <= end:
        pays.append(next_pay)
        next_pay += pd.Timedelta(days=90)

    if recur:
        buys = pd.date_range(start=today + pd.Timedelta(days=1), periods=dca_count, freq=f"{dca_days}D")
        buy_px = neutral.reindex(buys, method="nearest")
        shares = (recur_amt / buy_px).astype(float)
        shares.loc[today] = invest / last
    else:
        buys = [today]
        shares = pd.Series([invest / last], index=buys)

    rows = []
    cum = 0
    for i, pdte in enumerate(pays):
        elig = float(shares[shares.index <= pdte].sum())
        if elig <= 0 or last_div <= 0:
            continue
        adj = last_div * ((1 + div_growth / 4) ** i)
        cash = elig * adj
        cum += cash
        rows.append(
            {
                "Payment Date": pdte.date(),
                "Dividend/Share": f"${adj:.2f}",
                "Eligible Shares": f"{elig:,.4f}",
                "Total Payment ($)": cash,
                "Cumulative ($)": cum,
            }
        )

    # =========================================================
    # 📊 DISPLAY RESULTS
    # =========================================================
    tab1, tab2, tab3 = st.tabs(["📈 Monte Carlo", "💵 Dividend Cashflow", "📆 Quarterly Summary"])

    with tab1:
        st.subheader("Monte Carlo Simulation")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(px.index, px.values, label="Historical", color="gray")
        for lab, d in res.items():
            c = {"🐻 Bearish": "red", "😐 Neutral": "orange", "🚀 Bullish": "green"}[lab]
            ax.plot(proj_idx, d["p50"], label=f"{lab} Median", color=c)
            ax.fill_between(proj_idx, d["p10"], d["p90"], alpha=0.15, color=c)
        ax.set_title(f"{ticker} Monte Carlo Projection ({FWD_DAYS} days)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        st.pyplot(fig)

    with tab2:
        st.subheader("Expected Dividend Cashflow Schedule")
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df.style.format({"Total Payment ($)": "${:,.2f}", "Cumulative ($)": "${:,.2f}"}))
        else:
            st.info("No dividends expected in this period.")

    with tab3:
        if rows:
            df = pd.DataFrame(rows)
            df["Payment Date"] = pd.to_datetime(df["Payment Date"])
            q = df.groupby(df["Payment Date"].dt.to_period("Q"))["Total Payment ($)"].sum().reset_index()
            q["Quarter"] = q["Payment Date"].astype(str)
            q["Cumulative Dividends ($)"] = q["Total Payment ($)"].cumsum()

            contrib = []
            if recur:
                for qt in q["Quarter"]:
                    qs, qe = pd.Period(qt).start_time, pd.Period(qt).end_time
                    contrib.append(sum((d >= qs) & (d <= qe) for d in buys) * recur_amt)
            else:
                contrib = [0] * len(q)
            contrib[0] += invest
            q["Contributions ($)"] = contrib
            q["Total Contributions ($)"] = np.cumsum(q["Contributions ($)"])
            q["Yield on Cost (%)"] = q["Cumulative Dividends ($)"] / q["Total Contributions ($)"] * 100

            st.subheader("Quarterly Dividends + Contributions Summary")
            st.dataframe(q.style.format({
                "Total Payment ($)": "${:,.2f}",
                "Cumulative Dividends ($)": "${:,.2f}",
                "Total Contributions ($)": "${:,.2f}",
                "Yield on Cost (%)": "{:,.2f}%"
            }))
        else:
            st.info("No dividend history or future payments found.")
