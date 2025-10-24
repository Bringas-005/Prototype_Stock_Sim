# =========================================================
# 📈 STOCK PROJECTION SIMULATOR — Streamlit Dashboard (v5.1)
# Initial + DCA + Dividends + Growth + Monte Carlo + Cashflow
# =========================================================
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
import requests

plt.style.use("seaborn-v0_8-darkgrid")
st.set_page_config(page_title="Stock Projection Simulator", layout="wide")

# =======================
# Helpers
# =======================
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
    try:
        df = yf.download(t, period="1d", progress=False)
        return not df.empty
    except Exception:
        return False

@st.cache_data(show_spinner=False)
def get_price(ticker: str):
    """Best-effort last price via multiple fallbacks."""
    try:
        s = yf.Ticker(ticker)
        p = s.fast_info.get("last_price")
        if p is not None and np.isfinite(p): return float(p)
        info_price = s.info.get("regularMarketPrice")
        if info_price is not None and np.isfinite(info_price): return float(info_price)
        h = s.history(period="5d")["Close"].dropna()
        return float(h.iloc[-1]) if len(h) else None
    except Exception:
        try:
            h = yf.download(ticker, period="5d", progress=False)["Close"].dropna()
            return float(h.iloc[-1]) if len(h) else None
        except Exception:
            return None

# NOTE: DO NOT CACHE this — it takes a yfinance.Ticker object (unhashable)
def div_ttm(stock, price):
    """Return (dividends-per-share TTM, dividend yield)."""
    try:
        div = stock.dividends
        if div is None or div.empty or not price:
            return 0.0, 0.0
        if getattr(div.index, "tz", None):
            div.index = div.index.tz_localize(None)
        cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=365)
        ttm = float(div[div.index >= cutoff].sum())
        return ttm, (ttm / price if price > 0 else 0.0)
    except Exception:
        return 0.0, 0.0

@st.cache_data(show_spinner=False)
def load_px(ticker: str, period="5y") -> pd.Series:
    d = yf.download(ticker, period=period, progress=False, auto_adjust=False)
    if d is None or d.empty:
        return pd.Series(dtype=float)
    for col in ["Adj Close", "Close"]:
        if col in d.columns:
            s = d[col].dropna()
            if not s.empty:
                return s
    num = d.select_dtypes(include="number")
    if not num.empty:
        return num.iloc[:, 0].dropna()
    return pd.Series(dtype=float)

@st.cache_data(show_spinner=False)
def get_info(ticker: str):
    try:
        return yf.Ticker(ticker).info
    except Exception:
        return {}

def mc_paths(last_price, mu, sigma, days, npaths, daily_yield=0.0, drift_mult=1.0, rng=None):
    """GBM paths."""
    if rng is None:
        rng = np.random.default_rng(123)
    dt = 1/252
    mu_adj = mu * drift_mult + daily_yield
    shock = sigma * np.sqrt(dt) * rng.standard_normal((days, npaths))
    drift = (mu_adj - 0.5 * sigma**2) * dt
    path = np.empty((days, npaths), dtype=float)
    path[0, :] = last_price
    for t in range(1, days):
        path[t] = path[t-1] * np.exp(drift + shock[t])
    return path

# ---- Dynamic ticker suggestions (cache OK: argument is a string)
@st.cache_data(show_spinner=False)
def suggest_tickers(query: str) -> list:
    if not query:
        return []
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
    try:
        res = requests.get(url, timeout=3).json()
        out = []
        for q in res.get("quotes", []):
            sym = q.get("symbol")
            name = q.get("shortname") or q.get("longname") or ""
            exch = q.get("exchangeDisplay") or ""
            if sym:
                out.append(f"{sym} — {name} ({exch})" if name else sym)
        return out[:10]
    except Exception:
        return []

# =======================
# Sidebar UI
# =======================
st.title("📈 Stock Projection Simulator — Streamlit")
st.caption("Initial + DCA + Dividends + Growth + Monte Carlo + Cashflow")

with st.sidebar:
    st.header("1) Search Stock")
    query = st.text_input("🔍 Type company name or ticker", value="AAPL")
    if len(query.strip()) >= 2:
        with st.spinner("Searching..."):
            suggestions = suggest_tickers(query.strip())
    else:
        suggestions = []
    if suggestions:
        selection = st.selectbox("Select matching ticker", suggestions, key="ticker_select")
        ticker = selection.split(" — ")[0].strip().upper()
    else:
        ticker = query.strip().upper()
        if len(query.strip()) < 2:
            st.caption("Start typing to search for a stock (e.g., 'apple' → AAPL).")

    valid = validate_ticker(ticker) if ticker else False
    if not valid and ticker:
        st.error("❌ Invalid ticker or not found on Yahoo Finance.")

    st.header("2) Investment")
    invest = st.number_input("Initial investment ($)", min_value=0.0, value=10000.0, step=100.0)

    recur = st.toggle("Recurring investments (DCA)?", value=False)
    if recur:
        recur_amt = st.number_input("Amount per contribution ($)", min_value=0.0, value=500.0, step=50.0)
        freq_choice = st.selectbox("Frequency", ["Weekly (7d)", "Biweekly (14d)", "Monthly (30d)"], index=2)
        dca_days = 7 if "Weekly" in freq_choice else 14 if "Biweekly" in freq_choice else 30
        dca_months = st.number_input("Duration (months)", min_value=1, value=12, step=1)
        dca_count = int(dca_months * 30.4375 // dca_days)
    else:
        recur_amt = 0.0
        dca_days = 30
        dca_months = st.number_input("Hold period (months)", min_value=1, value=12, step=1)
        dca_count = 0

    st.header("3) Dividends")
    div_growth_pct = st.number_input("Expected annual dividend growth (%)", min_value=0.0, value=0.0, step=0.25)
    div_growth = div_growth_pct / 100.0
    reinvest = st.toggle("Reinvest dividends?", value=True)

    st.header("4) Monte Carlo")
    mc_months = dca_months
    FWD_DAYS = int(mc_months * 21)
    n_paths = st.slider("Number of paths", min_value=200, max_value=5000, value=1000, step=200)

    run = st.button("Run Simulation", type="primary", disabled=not valid)

if not ticker or not valid:
    st.stop()

# =======================
# Fundamentals
# =======================
s = yf.Ticker(ticker)
price = get_price(ticker)
dps, dy = div_ttm(s, price)
info = get_info(ticker)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Price", f"${(price or 0):,.2f}")
c2.metric("Dividend/Share (TTM)", f"${dps:.2f}")
c3.metric("Dividend Yield (TTM)", f"{dy*100:.2f}%")
c4.metric("Market Cap", money(info.get("marketCap")))

# =======================
# Run Simulation
# =======================
if run:
    px_5y = load_px(ticker, "5y")
    if px_5y.empty:
        st.error("Could not load price history. Try later or a different symbol.")
        st.stop()

    rets = np.log(px_5y / px_5y.shift(1)).dropna()
    mu = float(rets.mean())
    sig = float(rets.std())
    last = float(price or px_5y.iloc[-1])
    daily_yield = (dy / 252.0) if reinvest else 0.0

    scenarios = {"🐻 Bearish": 0.5, "😐 Neutral": 1.0, "🚀 Bullish": 1.5}
    paths = {}
    for name, mult in scenarios.items():
        path = mc_paths(last, mu, sig, FWD_DAYS, n_paths,
                        daily_yield=daily_yield, drift_mult=mult)
        paths[name] = {
            "p10": np.percentile(path, 10, axis=1),
            "p50": np.percentile(path, 50, axis=1),
            "p90": np.percentile(path, 90, axis=1)
        }

    proj_idx = pd.date_range(px_5y.index[-1] + pd.Timedelta(days=1), periods=FWD_DAYS, freq="B")
    total_contrib = float(invest + (recur_amt * dca_count if recur else 0.0))

    # =======================
    # Cashflow (your original logic)
    # =======================
    neutral_series = pd.Series(paths["😐 Neutral"]["p50"], index=proj_idx).asfreq("D").interpolate()

    divh = s.dividends
    if divh is not None and not divh.empty and getattr(divh.index, "tz", None):
        divh.index = divh.index.tz_localize(None)
    last_div = float(divh.iloc[-1]) if (divh is not None and not divh.empty) else (dps/4 if dps>0 else 0.0)

    today = pd.Timestamp.today().normalize()
    end = today + pd.Timedelta(days=int(mc_months * 30.4375))
    pays = []
    nxt = today + pd.Timedelta(days=90)
    while nxt <= end:
        pays.append(nxt); nxt += pd.Timedelta(days=90)

    if recur:
        buys = pd.date_range(start=today + pd.Timedelta(days=1), periods=dca_count, freq=f"{dca_days}D")
        buy_px = neutral_series.reindex(buys, method="nearest")
        shares = (recur_amt / buy_px).astype(float)
        shares.loc[today] = invest / last
    else:
        buys = [today]
        shares = pd.Series([invest / last], index=[today])

    rows = []
    cum = 0.0
    for i, pdte in enumerate(pays):
        elig = float(shares[shares.index <= pdte].sum())
        if elig <= 0 or last_div <= 0:
            continue
        adj = last_div * ((1 + div_growth/4) ** i)
        cash = elig * adj
        cum += cash
        rows.append({
            "Payment Date": pdte.date(),
            "Dividend/Share": f"${adj:.2f}",
            "Eligible Shares": f"{elig:,.4f}",
            "Total Payment ($)": cash,
            "Cumulative ($)": cum
        })

    # =======================
    # Display
    # =======================
    tab1, tab2, tab3 = st.tabs(["📊 Monte Carlo", "💵 Dividend Cashflow", "📆 Quarterly Summary"])

    with tab1:
        st.subheader("Projected Portfolio Value at Horizon")
        table_rows = []
        for name in scenarios.keys():
            proj_price = float(paths[name]["p50"][-1])
            port_val = total_contrib * (proj_price / last) if total_contrib > 0 else 0.0
            ret_pct = (port_val / total_contrib - 1.0) * 100 if total_contrib > 0 else 0.0
            table_rows.append([name, f"${proj_price:,.2f}", f"${port_val:,.2f}", f"{ret_pct:,.1f}%"])
        st.table(pd.DataFrame(table_rows, columns=["Scenario", "Proj. Price", "Portfolio Value", "Return"]))

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(px_5y.index, px_5y.values, label="Historical", color="gray")
        colors = {"🐻 Bearish": "red", "😐 Neutral": "orange", "🚀 Bullish": "green"}
        for name, d in paths.items():
            ax.plot(proj_idx, d["p50"], label=f"{name} Median", color=colors[name])
            ax.fill_between(proj_idx, d["p10"], d["p90"], alpha=0.15, color=colors[name])
        ax.set_title(f"{ticker} — Monte Carlo Projection ({FWD_DAYS} trading days)")
        ax.set_xlabel("Date"); ax.set_ylabel("Price (USD)"); ax.legend()
        st.pyplot(fig, clear_figure=True)

    with tab2:
        st.subheader("Expected Dividend Cashflow Schedule")
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(
                df.style.format({"Total Payment ($)": "${:,.2f}", "Cumulative ($)": "${:,.2f}"}),
                use_container_width=True
            )
        else:
            st.info("No dividends expected in this period.")

    with tab3:
        st.subheader("Quarterly Dividends + Contributions Summary")
        if rows:
            df = pd.DataFrame(rows)
            df["Payment Date"] = pd.to_datetime(df["Payment Date"])
            q = df.groupby(df["Payment Date"].dt.to_period("Q"))["Total Payment ($)"].sum().reset_index()
            q["Quarter"] = q["Payment Date"].astype(str)
            q["Cumulative Dividends ($)"] = q["Total Payment ($)"].cumsum()

            # contributions per quarter
            contrib = []
            if recur:
                for qt in q["Quarter"]:
                    qs, qe = pd.Period(qt).start_time, pd.Period(qt).end_time
                    contrib.append(sum((d >= qs) & (d <= qe) for d in pd.DatetimeIndex(buys)) * recur_amt)
            else:
                contrib = [0] * len(q)
            if len(q) > 0:
                contrib[0] += invest

            q["Contributions ($)"] = contrib
            q["Total Contributions ($)"] = np.cumsum(q["Contributions ($)"])
            q["Yield on Cost (%)"] = np.where(
                q["Total Contributions ($)"] > 0,
                q["Cumulative Dividends ($)"] / q["Total Contributions ($)"] * 100.0,
                0.0
            )

            st.dataframe(
                q.style.format({
                    "Total Payment ($)": "${:,.2f}",
                    "Cumulative Dividends ($)": "${:,.2f}",
                    "Contributions ($)": "${:,.2f}",
                    "Total Contributions ($)": "${:,.2f}",
                    "Yield on Cost (%)": "{:,.2f}%"
                }),
                use_container_width=True
            )
        else:
            st.info("No dividend history or future payments found.")
