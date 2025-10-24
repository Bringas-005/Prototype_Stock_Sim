# =========================================================
# 📈 STOCK PROJECTION SIMULATOR — Streamlit Dashboard (v4)
# Converts your v3.1 console app into a SaaS-style UI
# =========================================================
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Projection Simulator", layout="wide")
plt.style.use("seaborn-v0_8-darkgrid")

# ---------------------- Helpers ----------------------
@st.cache_data(show_spinner=False)
def validate_ticker(t: str) -> bool:
    try:
        df = yf.download(t, period="1d", progress=False)
        return not df.empty
    except Exception:
        return False

@st.cache_data(show_spinner=False)
def load_px(ticker: str, period="5y") -> pd.Series:
    """Fetch a single price series robustly (Adj Close → Close → first numeric col)."""
    df = yf.download(ticker, period=period, progress=False)
    if df is None or df.empty:
        return pd.Series(dtype=float)

    # MultiIndex columns (rare with single ticker but handle anyway)
    if isinstance(df.columns, pd.MultiIndex):
        for col in [("Adj Close", ticker), ("Close", ticker)]:
            if col in df.columns:
                return df[col].dropna()
        s = df.select_dtypes(include="number").iloc[:, 0].dropna()
        return s

    for col in ["Adj Close", "Close"]:
        if col in df.columns:
            return df[col].dropna()

    s = df.select_dtypes(include="number")
    return s.iloc[:, 0].dropna() if not s.empty else pd.Series(dtype=float)

def money(num):
    try:
        return f"${float(num):,.2f}"
    except Exception:
        return "N/A"

def get_price(stock: yf.Ticker, fallback_last=None):
    try:
        p = stock.fast_info.get("last_price") or stock.info.get("regularMarketPrice")
        if p:
            return float(p)
    except Exception:
        pass
    try:
        h = stock.history(period="5d")["Close"].dropna()
        if len(h):
            return float(h.iloc[-1])
    except Exception:
        pass
    return float(fallback_last) if fallback_last is not None else None

def div_ttm(stock: yf.Ticker, current_price: float):
    """Return (TTM dividend/share, TTM yield)."""
    try:
        div = stock.dividends
        if div is None or div.empty or not current_price:
            return 0.0, 0.0
        if getattr(div.index, "tz", None) is not None:
            div.index = div.index.tz_localize(None)
        cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=365)
        ttm = float(div[div.index >= cutoff].sum())
        yld = ttm / current_price if current_price > 0 else 0.0
        return ttm, yld
    except Exception:
        return 0.0, 0.0

# ---------------------- Sidebar (Dashboard Inputs) ----------------------
st.title("📈 Stock Projection Simulator")
st.caption("Estimate long-term portfolio performance with dividends, recurring investments, and Monte Carlo scenarios.")

with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Ticker (e.g. AAPL, MSFT, SPY)", "AAPL").upper().strip()

    c1, c2 = st.columns(2)
    with c1:
        initial_invest = st.number_input("Initial Investment ($)", min_value=0.0, value=1000.0, step=100.0)
    with c2:
        months = st.slider("Horizon (months)", 1, 120, 36)

    recurring = st.checkbox("Enable recurring investments (DCA)?", value=True)
    if recurring:
        recur_amt = st.number_input("Recurring amount ($)", min_value=0.0, value=200.0, step=50.0)
        freq = st.selectbox("Frequency", ["Monthly", "Biweekly", "Weekly"])
    else:
        recur_amt = 0.0
        freq = "Monthly"

    # Dividend settings (from v3.1)
    st.divider()
    st.subheader("Dividends")
    div_growth_pct = st.number_input("Annual dividend growth (%)", value=0.0, step=0.5, min_value=0.0)
    reinvest = st.checkbox("Reinvest dividends into shares", value=True)

    # Scenario drift multipliers (your v3.1 used 0.5/1.0/1.5; adjustable here)
    st.divider()
    st.subheader("Scenario Drift Multipliers")
    bear_mult    = st.slider("🐻 Bearish", 0.25, 1.50, 0.50, 0.05)
    neutral_mult = 1.00
    bull_mult    = st.slider("🚀 Bullish", 1.00, 1.75, 1.50, 0.05)

    # Monte Carlo paths
    N_PATHS = st.selectbox("Monte Carlo paths", [500, 1000, 2000], index=1)

    run = st.button("🚀 Run Simulation", use_container_width=True)

# ---------------------- Run ----------------------
if run:
    if not ticker or not validate_ticker(ticker):
        st.error("❌ Invalid or empty ticker. Please enter a valid symbol.")
        st.stop()

    # Load price data
    px = load_px(ticker, "5y")
    if px is None or px.empty:
        st.error("No price data found for that ticker.")
        st.stop()

    # Fundamentals
    tkr = yf.Ticker(ticker)
    last_hist_price = float(px.iloc[-1])
    current_price = get_price(tkr, fallback_last=last_hist_price) or last_hist_price
    dps_ttm, yld_ttm = div_ttm(tkr, current_price)

    # Show a fundamentals snapshot (like your prints)
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Current Price", money(current_price))
    try:
        info = tkr.info
    except Exception:
        info = {}
    mcap = info.get("marketCap")
    shares_out = info.get("sharesOutstanding")
    pe = info.get("trailingPE")

    f2.metric("Market Cap", money(mcap))
    f3.metric("Shares Outstanding", f"{shares_out:,.0f}" if shares_out else "N/A")
    f4.metric("Trailing P/E", f"{pe:.2f}" if pe else "N/A")

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Dividend/Share (TTM)", money(dps_ttm))
    g2.metric("Dividend Yield (TTM)", f"{yld_ttm*100:.2f}%")
    g3.metric("52W High", money(info.get("fiftyTwoWeekHigh")))
    g4.metric("52W Low", money(info.get("fiftyTwoWeekLow")))

    # Estimated annual dividends on initial only (mirrors your v3.1 print)
    if current_price and yld_ttm is not None:
        est_init_div = initial_invest * yld_ttm
        st.info(f"💵 **Estimated Annual Dividends (on initial):** {money(est_init_div)} "
                f"(from {yld_ttm*100:.2f}% yield, {money(dps_ttm)}/sh)")

    # Returns stats from 5y series (v3.1)
    rets = np.log(px / px.shift(1)).dropna()
    mu_d, sig_d = float(rets.mean()), float(rets.std())

    # DCA math
    freq_days = 30 if freq == "Monthly" else 14 if freq == "Biweekly" else 7
    dca_count = int((months * 30.4375) // freq_days) if recurring else 0
    total_contrib = initial_invest + (recur_amt * dca_count)

    st.info(
        f"💵 **Total planned contributions:** {money(total_contrib)} "
        f"({money(initial_invest)} initial + {dca_count} × {money(recur_amt)})"
    )

    # ---------------- Quick Preview (your v3.1 drift preview) ----------------
    st.subheader(f"⏳ Quick Drift Preview ({months}-month estimate)")
    try:
        px1 = load_px(ticker, "1y")
        r = np.log(px1 / px1.shift(1)).dropna()
        mu_p = float(r.mean())
        last1 = current_price or float(px1.iloc[-1])
        days = int(months * 21)
        scen_p = {
            "🐻 Bearish (−50% drift)": 0.5,
            "😐 Neutral (baseline)": 1.0,
            "🚀 Bullish (+50% drift)": 1.5
        }
        rows_prev = []
        for lab, m in scen_p.items():
            proj = last1 * np.exp(mu_p * m * days)
            val = total_contrib * (proj / last1)
            ret = (val / total_contrib - 1) * 100 if total_contrib > 0 else 0.0
            rows_prev.append([lab, money(proj), money(val), f"{ret:+.1f}%"])
        st.dataframe(pd.DataFrame(rows_prev, columns=["Scenario", "Proj. Price", "Value", "Return"]),
                     use_container_width=True)
    except Exception as e:
        st.warning(f"Preview failed: {e}")

    # ---------------- Monte Carlo (v3.1) ----------------
    FWD_DAYS = int(months * 21)
    dt = 1 / 252
    randn = np.random.normal(size=(FWD_DAYS, int(N_PATHS)))

    scen = {"🐻 Bearish": bear_mult, "😐 Neutral": 1.0, "🚀 Bullish": bull_mult}
    res = {}
    for lab, m in scen.items():
        mu_a = mu_d * m + (yld_ttm / 252 if reinvest else 0.0)
        drift = (mu_a - 0.5 * sig_d ** 2) * dt
        shock = sig_d * np.sqrt(dt) * randn
        path = np.zeros((FWD_DAYS, int(N_PATHS)))
        path[0, :] = current_price
        for t in range(1, FWD_DAYS):
            path[t] = path[t - 1] * np.exp(drift + shock[t])
        res[lab] = {
            "p10": np.percentile(path, 10, axis=1),
            "p50": np.percentile(path, 50, axis=1),
            "p90": np.percentile(path, 90, axis=1),
        }

    proj_idx = pd.date_range(px.index[-1] + pd.Timedelta(days=1), periods=FWD_DAYS, freq="B")

    # Scenario summary table (v3.1 semantics)
    st.subheader("💼 Projected Portfolio Value Using Monte Carlo (Median)")
    rows_mc = []
    for lab in ["🐻 Bearish", "😐 Neutral", "🚀 Bullish"]:
        proj_price = float(res[lab]["p50"][-1])
        value = total_contrib * (proj_price / current_price) if current_price > 0 else 0.0
        ret = (value / total_contrib - 1) * 100 if total_contrib > 0 else 0.0
        rows_mc.append([lab, money(proj_price), money(value), f"{ret:.1f}%"])
    st.dataframe(pd.DataFrame(rows_mc, columns=["Scenario", "Projected Price", "Portfolio Value", "Return"]),
                 use_container_width=True)

    # ---------------- Dividend Cashflow (quarterly, with growth) ----------------
    st.subheader("🗓️ Expected Dividend Cashflow Schedule")
    # Build neutral median price to time DCA purchases
    neutral_series = pd.Series(res["😐 Neutral"]["p50"], index=proj_idx).asfreq("D").interpolate()

    today = pd.Timestamp.today().normalize()
    end_cf = today + pd.Timedelta(days=int(months * 30.4375))
    pay_dates = []
    nxt = today + pd.Timedelta(days=90)
    while nxt <= end_cf:
        pay_dates.append(nxt)
        nxt += pd.Timedelta(days=90)

    # Shares schedule
    if recurring and dca_count > 0:
        buy_dates = pd.date_range(start=today + pd.Timedelta(days=1), periods=dca_count, freq=f"{freq_days}D")
        buy_px = neutral_series.reindex(buy_dates, method="nearest")
        shares = (recur_amt / buy_px).astype(float)
        # Add initial "buy" today at current price
        init_units = (initial_invest / current_price) if current_price > 0 else 0.0
        shares.loc[today] = init_units
    else:
        init_units = (initial_invest / current_price) if current_price > 0 else 0.0
        shares = pd.Series([init_units], index=[today])

    # Quarterly dividend/share baseline from TTM
    last_div_q = (dps_ttm / 4.0) if dps_ttm > 0 else 0.0
    div_g = div_growth_pct / 100.0

    rows_cf, cum_div = [], 0.0
    for i, pdte in enumerate(pay_dates):
        elig_shares = float(shares[shares.index <= pdte].sum())
        if elig_shares <= 0 or last_div_q <= 0:
            continue
        adj_div = last_div_q * ((1 + div_g / 4) ** i)
        cash = elig_shares * adj_div
        cum_div += cash
        rows_cf.append({
            "Payment Date": pdte.date(),
            "Dividend/Share": money(adj_div),
            "Eligible Shares": f"{elig_shares:,.4f}",
            "Total Payment ($)": money(cash),
            "Cumulative ($)": money(cum_div)
        })

    if rows_cf:
        cf_df = pd.DataFrame(rows_cf)
        st.dataframe(cf_df, use_container_width=True)

        # Quarterly summary (like v3.1)
        qdf = cf_df.copy()
        qdf["Payment Date"] = pd.to_datetime(qdf["Payment Date"])
        qdf["Total Payment ($)"] = qdf["Total Payment ($)"].replace(r"[\$,]", "", regex=True).astype(float)
        qsum = qdf.groupby(qdf["Payment Date"].dt.to_period("Q"))["Total Payment ($)"].sum().reset_index()
        qsum["Quarter"] = qsum["Payment Date"].astype(str)
        qsum["Cumulative Dividends ($)"] = qsum["Total Payment ($)"].cumsum()

        # Contributions by quarter
        contrib = []
        if recurring and dca_count > 0:
            for qt in qsum["Quarter"]:
                qs, qe = pd.Period(qt).start_time, pd.Period(qt).end_time
                count_in_q = 0
                for d in shares.index:
                    if d != today and qs <= d <= qe:
                        count_in_q += 1
                contrib.append(count_in_q * recur_amt)
        else:
            contrib = [0] * len(qsum)
        if len(contrib) > 0:
            contrib[0] += initial_invest  # add initial to first quarter

        qsum["Contributions ($)"] = contrib
        qsum["Total Contributions ($)"] = np.cumsum(qsum["Contributions ($)"])
        qsum["Yield on Cost (%)"] = np.where(
            qsum["Total Contributions ($)"] > 0,
            qsum["Cumulative Dividends ($)"] / qsum["Total Contributions ($)"] * 100,
            0.0
        )

        st.subheader("📆 Quarterly Dividend + Contribution Summary")
        st.dataframe(qsum[[
            "Quarter", "Total Payment ($)", "Cumulative Dividends ($)",
            "Contributions ($)", "Total Contributions ($)", "Yield on Cost (%)"
        ]], use_container_width=True)

        # Chart: Dividends + YOC line
        fig_cf, ax1 = plt.subplots(figsize=(10, 4))
        ax2 = ax1.twinx()
        ax1.bar(qsum["Quarter"], qsum["Total Payment ($)"], label="Dividends ($)")
        ax2.plot(qsum["Quarter"], qsum["Yield on Cost (%)"], marker="o", label="Yield on Cost (%)")
        ax1.set_ylabel("Dividends ($)")
        ax2.set_ylabel("Yield on Cost (%)")
        ax1.set_title(f"{ticker} — Quarterly Dividends & Yield on Cost")
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend(loc="upper left"); ax2.legend(loc="upper right")
        plt.tight_layout()
        st.pyplot(fig_cf)

        # CSV download
        st.download_button(
            "⬇️ Download dividend cashflow CSV",
            data=cf_df.to_csv(index=False),
            file_name=f"{ticker}_dividend_cashflow.csv"
        )

    # ---------------- Charts ----------------
    st.subheader("📉 Price Projection (Monte Carlo)")
    proj_idx = pd.date_range(px.index[-1] + pd.Timedelta(days=1), periods=FWD_DAYS, freq="B")
    fig_mc, ax = plt.subplots(figsize=(11, 5))
    ax.plot(px.index, px.values, label="Historical", color="gray")
    colors = {"🐻 Bearish": "red", "😐 Neutral": "orange", "🚀 Bullish": "green"}
    for lab, d in res.items():
        ax.plot(proj_idx, d["p50"], label=f"{lab} Median", color=colors[lab])
        ax.fill_between(proj_idx, d["p10"], d["p90"], alpha=0.15, color=colors[lab])
    ax.set_title(f"{ticker} — Monte Carlo Projection ({months} months)")
    ax.set_xlabel("Date"); ax.set_ylabel("Price (USD)")
    ax.legend()
    st.pyplot(fig_mc)

    # Portfolio vs contributions (Neutral scenario)
    st.subheader("💼 Projected Portfolio vs Contributions (Neutral)")
    neutral_series = pd.Series(res["😐 Neutral"]["p50"], index=proj_idx).asfreq("D").interpolate()

    contrib_line = pd.Series(0.0, index=neutral_series.index)
    first_date = neutral_series.index[0]
    if initial_invest > 0:
        contrib_line.loc[first_date] += initial_invest
    if recurring and dca_count > 0:
        buy_dates = pd.date_range(start=first_date, periods=dca_count, freq=f"{freq_days}D")
        for d in buy_dates:
            if d in contrib_line.index:
                contrib_line.loc[d] += recur_amt
    cum_contrib = contrib_line.cumsum()

    units = 0.0
    pv_vals = []
    for d in neutral_series.index:
        if contrib_line.loc[d] > 0 and neutral_series.loc[d] > 0:
            units += contrib_line.loc[d] / neutral_series.loc[d]
        pv_vals.append(units * neutral_series.loc[d])
    pv = pd.Series(pv_vals, index=neutral_series.index)

    fig_val, axv = plt.subplots(figsize=(11, 5))
    axv.plot(pv.index, pv.values, label="Projected Portfolio Value (Neutral)")
    axv.plot(cum_contrib.index, cum_contrib.values, label="Cumulative Contributions", linestyle="--")
    axv.set_title(f"{ticker} — Portfolio vs Contributions (Neutral)")
    axv.set_xlabel("Date"); axv.set_ylabel("USD")
    axv.legend()
    st.pyplot(fig_val)
