# =========================================================
# 📈 STOCK PROJECTION SIMULATOR v3.1
# (Initial + Recurring Investments + Dividends + Growth + Monte Carlo + Charts)
# =========================================================

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np, pandas as pd, yfinance as yf, matplotlib.pyplot as plt
from datetime import datetime, timedelta
plt.style.use('seaborn-v0_8-darkgrid')

# ---------- Helpers ----------
def money(num):
    if num is None or pd.isna(num): return "N/A"
    mag, n = 0, float(num)
    while abs(n) >= 1000 and mag < 5: mag += 1; n /= 1000
    return f"${n:,.2f}{['','K','M','B','T'][mag]}"

def validate_ticker(t):
    try: return not yf.download(t, period="1d", progress=False).empty
    except: return False

def ask_yes_no(prompt):
    while True:
        a = input(prompt).strip().lower()
        if a in ["yes","no"]: return a=="yes"
        print("Please answer only 'yes' or 'no'.")

def get_price(stock):
    try: return float(stock.fast_info.get("last_price") or stock.info.get("regularMarketPrice"))
    except: 
        try:
            h = stock.history(period="5d")["Close"].dropna()
            return float(h.iloc[-1]) if len(h) else None
        except: return None

def div_ttm(stock, price):
    try:
        div = stock.dividends
        if div is None or div.empty or not price: return 0,0
        if getattr(div.index,"tz",None): div.index = div.index.tz_localize(None)
        cutoff = pd.Timestamp.today().normalize()-pd.Timedelta(days=365)
        ttm = float(div[div.index>=cutoff].sum())
        return ttm, ttm/price if price>0 else 0
    except: return 0,0

# ---------- Fundamentals ----------
def fundamentals(ticker, invest=None):
    s = yf.Ticker(ticker)
    p = get_price(s)
    dps, dy = div_ttm(s, p)
    try: info=s.info
    except: info={}
    print(f"\n📊 Key Fundamentals for {ticker}")
    print(f"Current Price: ${p:,.2f}")
    print(f"Market Cap: {money(info.get('marketCap'))}")
    print(f"Shares Outstanding: {money(info.get('sharesOutstanding'))}")
    print(f"Trailing P/E: {info.get('trailingPE','N/A')}")
    print(f"Dividend/Share (TTM): ${dps:.2f}")
    print(f"Dividend Yield (TTM): {dy*100:.2f}%")
    if invest and p:
        est = invest*dy
        print(f"\n💵 Estimated Annual Dividends: ${est:,.2f} (from {dy*100:.2f}% yield, ${dps:,.2f}/sh)")
    return p, dps, dy

# ---------- Prices ----------
def load_px(t, period="5y"):
    d=yf.download(t,period=period,progress=False)
    if "Adj Close" in d.columns: return d["Adj Close"].dropna()
    if "Close" in d.columns: return d["Close"].dropna()
    return d.select_dtypes(include='number').iloc[:,0].dropna()

# =========================================================
# MAIN LOOP
# =========================================================
while True:
    TICKER=input("Enter stock ticker (e.g. AAPL, MSFT): ").strip().upper()
    if not validate_ticker(TICKER): print("❌ Invalid ticker."); continue

    INVEST=float(input("Initial investment amount (USD): "))
    RECUR=False; RECUR_AMT=0; DCA_FREQ_DAYS=30; DCA_MONTHS=0; DCA_COUNT=0
    RECUR=ask_yes_no("Will there be recurring investments (DCA)? (yes/no): ")
    if RECUR:
        RECUR_AMT=float(input("How much (USD) per recurring investment? "))
        print("Frequency: 1=Weekly, 2=Biweekly, 3=Monthly")
        ch=int(input("Select (1/2/3): "))
        DCA_FREQ_DAYS=7 if ch==1 else 14 if ch==2 else 30
        DCA_MONTHS=int(input("For how many months? "))
        DCA_COUNT=int(DCA_MONTHS*30.4375//DCA_FREQ_DAYS)
        print(f"📆 You will invest ${RECUR_AMT:,.2f} every {DCA_FREQ_DAYS} days for {DCA_MONTHS} months (~{DCA_COUNT} contributions).")
    else:
        DCA_MONTHS=int(input("Hold period (months)? "))

    price,dps,dy=fundamentals(TICKER,INVEST)
    try:
        DIV_GROWTH=float(input("Expected annual dividend growth rate (%): ") or 0)/100
        print(f"📈 Dividend growth set to {DIV_GROWTH*100:.2f}% annually.")
    except: DIV_GROWTH=0; print("⚠️ Invalid. Using 0%.")
    REINV=dy>0 and ask_yes_no("Reinvest dividends? (yes/no): ")

    hold_months=DCA_MONTHS
    FWD_DAYS=int(hold_months*21); N=1000
    px=load_px(TICKER,"5y"); rets=np.log(px/px.shift(1)).dropna()
    mu,sig=float(rets.mean()),float(rets.std()); last=price or float(px.iloc[-1])
    dt=1/252; randn=np.random.normal(size=(FWD_DAYS,N))
    scen={"🐻 Bearish":0.5,"😐 Neutral":1.0,"🚀 Bullish":1.5}; res={}
    for lab,m in scen.items():
        mu_a=mu*m+(dy/252 if REINV else 0)
        drift=(mu_a-0.5*sig**2)*dt; shock=sig*np.sqrt(dt)*randn
        path=np.zeros((FWD_DAYS,N)); path[0,:]=last
        for t in range(1,FWD_DAYS): path[t]=path[t-1]*np.exp(drift+shock[t])
        res[lab]={"p10":np.percentile(path,10,axis=1),
                  "p50":np.percentile(path,50,axis=1),
                  "p90":np.percentile(path,90,axis=1)}
    proj_idx=pd.date_range(px.index[-1]+pd.Timedelta(days=1),periods=FWD_DAYS,freq="B")

    # Quick preview
    try:
        print("\n⏳ Calculating projected portfolio value scenarios...\n")
        px1=load_px(TICKER,"1y"); r=np.log(px1/px1.shift(1)).dropna()
        mu_p=float(r.mean()); last1=price or float(px1.iloc[-1]); days=int(hold_months*21)
        scen_p={"🐻 Bearish (-50% drift)":0.5,"😐 Neutral (baseline)":1.0,"🚀 Bullish (+50% drift)":1.5}
        print(f"💼 Portfolio Value Projections ({hold_months}-month estimate)")
        print("─"*60)
        for lab,m in scen_p.items():
            proj=last1*np.exp(mu_p*m*days)
            contrib=INVEST+(RECUR_AMT*DCA_COUNT if RECUR else 0)
            val=contrib*(proj/last1); ret=(val/contrib-1)*100
            print(f"{lab:<28} ${proj:>12,.2f}   Value: ${val:>10,.2f} ({ret:>+5.1f}%)")
        print("─"*60)
    except Exception as e: print(f"⚠️ Preview failed: {e}")

    # Monte Carlo result summary
    total_contrib=INVEST+(RECUR_AMT*DCA_COUNT if RECUR else 0)
    print("\n💼 Projected Portfolio Value Using Monte Carlo Scenarios:")
    print("─────────────────────────────────────────────────────")
    print(f"{'Scenario':<15} {'Proj. Price':>15} {'Portfolio Value':>20} {'Return':>10}")
    print("─────────────────────────────────────────────────────")
    for lab in scen.keys():
        proj=res[lab]["p50"][-1]
        val=total_contrib*(proj/last); ret=(val/total_contrib-1)*100
        print(f"{lab:<15} ${proj:>13,.2f} ${val:>18,.2f} {ret:>9.1f}%")
    print("─────────────────────────────────────────────────────")

    # Dividend cashflow
    neutral=pd.Series(res["😐 Neutral"]["p50"],index=proj_idx).asfreq("D").interpolate()
    s_cf=yf.Ticker(TICKER); divh=s_cf.dividends
    if divh is not None and not divh.empty and getattr(divh.index,"tz",None): divh.index=divh.index.tz_localize(None)
    last_div=float(divh.iloc[-1]) if (divh is not None and not divh.empty) else (dps/4 if dps>0 else 0)
    today=pd.Timestamp.today().normalize(); end=today+pd.Timedelta(days=int(hold_months*30.4375))
    next=today+pd.Timedelta(days=90); pays=[]
    while next<=end: pays.append(next); next+=pd.Timedelta(days=90)
    if RECUR:
        buys=pd.date_range(start=today+pd.Timedelta(days=1),periods=DCA_COUNT,freq=f"{DCA_FREQ_DAYS}D")
        buy_px=neutral.reindex(buys,method="nearest"); shares=(RECUR_AMT/buy_px).astype(float)
        shares.loc[today]=INVEST/last
    else:
        buys=[today]; shares=pd.Series([INVEST/last],index=buys)

    rows=[]; cum=0
    for i,pdte in enumerate(pays):
        elig=float(shares[shares.index<=pdte].sum())
        if elig<=0 or last_div<=0: continue
        adj=last_div*((1+DIV_GROWTH/4)**i)
        cash=elig*adj; cum+=cash
        rows.append({"Payment Date":pdte.date(),"Dividend/Share":f"${adj:.2f}",
                     "Eligible Shares":f"{elig:,.4f}","Total Payment ($)":f"${cash:,.2f}",
                     "Cumulative ($)":f"${cum:,.2f}"})
    if rows:
        df=pd.DataFrame(rows)
        print("\n🗓️ Expected Dividend Cashflow Schedule:")
        print(df.to_string(index=False))

        df["Payment Date"]=pd.to_datetime(df["Payment Date"])
        df["Total Payment ($)"]=df["Total Payment ($)"].replace('[\$,]','',regex=True).astype(float)
        q=df.groupby(df["Payment Date"].dt.to_period("Q"))["Total Payment ($)"].sum().reset_index()
        q["Quarter"]=q["Payment Date"].astype(str)
        q["Cumulative Dividends ($)"]=q["Total Payment ($)"].cumsum()
        contrib=[]
        if RECUR:
            for qt in q["Quarter"]:
                qs,qe=pd.Period(qt).start_time,pd.Period(qt).end_time
                contrib.append(sum((d>=qs)&(d<=qe) for d in buys)*RECUR_AMT)
        else: contrib=[0]*len(q)
        contrib[0]+=INVEST
        q["Contributions ($)"]=contrib; q["Total Contributions ($)"]=np.cumsum(q["Contributions ($)"])
        q["Yield on Cost (%)"]=q["Cumulative Dividends ($)"]/q["Total Contributions ($)"]*100
        print("\n📆 Quarterly Dividend + Contribution Summary:")
        print(q.to_string(index=False))

    # Chart: Monte Carlo
    plt.figure(figsize=(12,6))
    plt.plot(px.index,px.values,label="Historical",color="gray")
    for lab,d in res.items():
        c={"🐻 Bearish":"red","😐 Neutral":"orange","🚀 Bullish":"green"}[lab]
        plt.plot(proj_idx,d["p50"],label=f"{lab} Median",color=c)
        plt.fill_between(proj_idx,d["p10"],d["p90"],alpha=.15,color=c)
    plt.title(f"{TICKER} Monte Carlo Projection ({FWD_DAYS} trading days)")
    plt.xlabel("Date"); plt.ylabel("Price (USD)"); plt.legend(); plt.show()

    if not ask_yes_no("\nRun another simulation? (yes/no): "):
        print("👋 Thanks for using the Stock Projection Simulator!")
        break