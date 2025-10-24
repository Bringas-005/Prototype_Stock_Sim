# app.py
# =========================================================
# 📈 STOCK PROJECTION SIMULATOR — Streamlit Dashboard (v4)
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
        # normalize timezone on index
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
    # include yield as drift bump if reinvesting
    mu_adj = mu * drift_mult + daily_yield
    shock = sigma * np.sqrt(dt) * rng.standard_normal((days, npaths))
    drift = (mu_adj - 0.5 * sigma**2) * dt
    path = np.empty((days, npaths), dtype=float)
    path[0, :] = last_price
    for t in range(1, days):
        path[t] = path[t-1] * np.exp(drift + shock[t])
    return path

# ---------------------- UI ----------------------
st.title("📈 Stock Projection Simulator — Streamlit Dashboard")
st.caption("Initial + Recurring (DCA) + Dividends + Growth + Monte Carlo")

with st.sidebar:
    st.header("1) Ticker")
    ticker = st.text_input("Symbol", value="AAPL").upper().strip()
    valid = validate_ticker(ticker) if ticker else False
    if not valid and ticker:
        st.error("Invalid ticker (or temporarily unavailable). Try another.")

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
        dca_amt = 0.0
        dca_days = 30
        dca_months = st.number_input("Hold period (months)", min_value=1, value=12, step=1)
        dca_count = 0

    st.header("3) Dividends")
    div_growth_pct = st.number_input("Expected annual dividend growth (%)", min_value=0.0, value=0.0, step=0.25)
    reinvest = st.toggle("Reinvest dividends?", value=True)

    st.header("4) Monte Carlo")
    mc_months = dca_months
    mc_days = int(mc_months * 21)
    n_paths = st.slider("Number of paths", min_value=200, max_value=5000, value=1000, step=200)
    st.caption("Trading days ≈ months × 21")

    st.header("Scenarios")
    bearish_mult = st.number_input("🐻 Bearish drift multiplier", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
    neutral_mult = st.number_input("😐 Neutral drift multiplier", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
    bullish_mult = st.number_input("🚀 Bullish drift multiplier", min_value=0.1, max_value=3.0, value=1.5, step=0.1)

    run = st.button("Run Simulation", type="primary", disabled=not valid)

if not ticker:
    st.info("Enter a ticker to begin.")
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

    # daily log returns from 5Y
    rets = np.log(px_5y / px_5y.shift(1)).dropna()
    mu = float(rets.mean())
    sig = float(rets.std())
    last = float(price_now or px_5y.iloc[-1])
    daily_yield = (dy / 252.0) if reinvest else 0.0

    # Monte Carlo scenarios
    rng = np.random.default_rng(123)
    scenarios = {
        "🐻 Bearish": bearish_mult,
        "😐 Neutral": neutral_mult,
        "🚀 Bullish": bullish_mult
    }
    paths = {}
    for name, mult in scenarios.items():
        path = mc_paths(last, mu, sig, mc_days, n_paths, daily_yield=daily_yield, drift_mult=mult, rng=rng)
        paths[name] = {
            "p10": np.percentile(path, 10, axis=1),
            "p50": np.percentile(path, 50, axis=1),
            "p90": np.percentile(path, 90, axis=1),
        }

    proj_idx = pd.date_range(px_5y.index[-1] + pd.Timedelta(days=1), periods=mc_days, freq="B")
    total_contrib = float(initial + (recur_amt * dca_count if recur else 0.0))

    # Quick baseline projection (not MC): use 1y mean drift
    preview = []
    try:
        r1 = np.log(px_1y / px_1y.shift(1)).dropna()
        mu1 = float(r1.mean()) if not r1.empty else mu
        for label, mult in [("🐻 Bearish (-50% drift)", 0.5), ("😐 Neutral (baseline)", 1.0), ("🚀 Bullish (+50% drift)", 1.5)]:
            proj_price = last * np.exp(mu1 * mult * mc_days)
            contrib = total_contrib
            val = contrib * (proj_price / last)
            ret = (val / contrib - 1.0) * 100 if contrib > 0 else 0.0
            preview.append((label, proj_price, val, ret))
    except Exception:
        pass

    # ---------------------- Tabs ----------------------
    tab1, tab2, tab3 = st.tabs(["📊 Projections", "🗓️ Dividends & DCA", "📈 Price Chart"])

    with tab1:
        st.subheader("Monte Carlo — Projected Portfolio Value at Horizon")
        table_rows = []
        for name in scenarios.keys():
            proj_price = float(paths[name]["p50"][-1])
            port_val = total_contrib * (proj_price / last) if total_contrib > 0 else 0.0
            ret_pct = (port_val / total_contrib - 1.0) * 100 if total_contrib > 0 else 0.0
            table_rows.append([name, f"${proj_price:,.2f}", f"${port_val:,.2f}", f"{ret_pct:,.1f}%"])
        st.table(pd.DataFrame(table_rows, columns=["Scenario", "Proj. Price", "Portfolio Value", "Return"]))

        if preview:
            st.markdown("**Baseline Preview (non-MC, from 1-year mean drift)**")
            st.table(pd.DataFrame(
                [(lbl, f"${pp:,.2f}", f"${pv:,.2f}", f"{rr:+.1f}%") for (lbl, pp, pv, rr) in preview],
                columns=["Scenario", "Proj. Price", "Value", "Return"]
            ))

        # Chart: Monte Carlo paths percentiles
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(px_5y.index, px_5y.values, label="Historical", color="gray")
        colors = {"🐻 Bearish": "red", "😐 Neutral": "orange", "🚀 Bullish": "green"}
        for name, dat in paths.items():
            ax.plot(proj_idx, dat["p50"], label=f"{name} Median", color=colors[name])
            ax.fill_between(proj_idx, dat["p10"], dat["p90"], alpha=0.15, color=colors[name])
        ax.set_title(f"{ticker} — Monte Carlo Projection ({mc_days} trading days)")
        ax.set_xlabel("Date"); ax.set_ylabel("Price (USD)"); ax.legend()
        st.pyplot(fig, clear_figure=True)

    with tab2:
        st.subheader("Dividend Schedule & Yield-on-Cost (Estimated)")
        # Build neutral median path for pricing reference
        neutral_series = pd.Series(paths["😐 Neutral"]["p50"], index=proj_idx).asfreq("D").interpolate()

        # Build contribution schedule
        today = pd.Timestamp.today().normalize()
        end = today + pd.Timedelta(days=int(dca_months * 30.4375))
        # Approx quarterly next payments
        last_div_amt = float(div_hist.iloc[-1]) if not div_hist.empty else (div_ttm / 4 if div_ttm > 0 else 0.0)
        pay_dates = []
        nxt = today + pd.Timedelta(days=90)
        while nxt <= end:
            pay_dates.append(nxt)
            nxt += pd.Timedelta(days=90)

        if recur:
            buys = pd.date_range(start=today + pd.Timedelta(days=1), periods=dca_count, freq=f"{dca_days}D")
            buy_px = neutral_series.reindex(buys, method="nearest")
            shares = (recur_amt / buy_px).astype(float)
            # add initial as a separate "buy" today at last price
            if last and last > 0:
                shares.loc[today] = initial / last
        else:
            buys = pd.DatetimeIndex([today])
            shares = pd.Series([initial / last if last and last > 0 else 0.0], index=buys)

        rows = []
        cum = 0.0
        q_rows = []
        if pay_dates and last_div_amt > 0:
            for i, dpay in enumerate(pay_dates):
                eligible = float(shares[shares.index <= dpay].sum())
                if eligible <= 0:
                    continue
                # dividend growth applied quarterly (approx)
                adj_div = last_div_amt * ((1 + (div_growth_pct / 100.0) / 4) ** i)
                cash = eligible * adj_div
                cum += cash
                rows.append({
                    "Payment Date": dpay.date(),
                    "Dividend/Share": f"${adj_div:.2f}",
                    "Eligible Shares": f"{eligible:,.4f}",
                    "Total Payment ($)": cash
                })

            if rows:
                df_div = pd.DataFrame(rows)
                df_div["Payment Date"] = pd.to_datetime(df_div["Payment Date"])
                st.dataframe(df_div.style.format({"Total Payment ($)": "${:,.2f}"}), use_container_width=True)

                # Quarterly summary with contributions and yield on cost
                q = df_div.copy()
                q["Quarter"] = q["Payment Date"].dt.to_period("Q").astype(str)
                qsum = q.groupby("Quarter", as_index=False)["Total Payment ($)"].sum()
                # build contributions per quarter
                contrib_per_q = []
                for qt in qsum["Quarter"]:
                    qs = pd.Period(qt).start_time
                    qe = pd.Period(qt).end_time
                    if recur:
                        contrib_q = float(((buys >= qs) & (buys <= qe)).sum()) * float(recur_amt)
                    else:
                        contrib_q = 0.0
                    # initial counted in the first quarter on/after today
                    if qs <= today <= qe:
                        contrib_q += float(initial)
                    contrib_per_q.append(contrib_q)
                qsum["Contributions ($)"] = contrib_per_q
                qsum["Total Contributions ($)"] = qsum["Contributions ($)"].cumsum()
                qsum["Cumulative Dividends ($)"] = qsum["Total Payment ($)"].cumsum()
                qsum["Yield on Cost (%)"] = np.where(
                    qsum["Total Contributions ($)"] > 0,
                    qsum["Cumulative Dividends ($)"] / qsum["Total Contributions ($)"] * 100.0,
                    0.0
                )
                st.markdown("**Quarterly Dividends & Contributions**")
                st.dataframe(
                    qsum.style.format({
                        "Total Payment ($)": "${:,.2f}",
                        "Contributions ($)": "${:,.2f}",
                        "Total Contributions ($)": "${:,.2f}",
                        "Cumulative Dividends ($)": "${:,.2f}",
                        "Yield on Cost (%)": "{:,.2f}%"
                    }),
                    use_container_width=True
                )
        else:
            st.info("No dividend history detected or horizon too short to schedule payments.")

    with tab3:
        st.subheader(f"{ticker} — Historical Prices")
        if not px_5y.empty:
            fig2, ax2 = plt.subplots(figsize=(12, 4))
            ax2.plot(px_5y.index, px_5y.values, label="Adj Close/Close")
            ax2.set_xlabel("Date"); ax2.set_ylabel("Price (USD)")
            ax2.legend()
            st.pyplot(fig2, clear_figure=True)
        else:
            st.info("No price history available.")

    # ---------------------- Footer KPIs ----------------------
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Contributions", f"${total_contrib:,.2f}")
    k2.metric("Paths", f"{n_paths:,}")
    k3.metric("Horizon", f"{mc_months} months (~{mc_days} trading days)")

else:
    st.info("Configure inputs in the left sidebar and click **Run Simulation**.")
    if not px_5y.empty:
        st.line_chart(px_5y, height=200)
