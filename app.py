# -*- coding: utf-8 -*-
import os
import time
import sqlite3
import requests
from datetime import datetime, timedelta
import threading

import pandas as pd
import numpy as np
import pyupbit
import lightgbm as lgb
import streamlit as st

# -----------------------------
# Streamlit 환경 설정
# -----------------------------
st.set_page_config(page_title="AI Crypto Trader 1분 자동", layout="wide")
DB = "./ai_trader_1min.db"  # Cloud 안전 경로
conn = sqlite3.connect(DB, check_same_thread=False)
cur = conn.cursor()

# -----------------------------
# 업비트 로그인 (실거래용)
# -----------------------------
ACCESS_KEY = "여기에_업비트_access_key"
SECRET_KEY = "여기에_업비트_secret_key"
upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)

# -----------------------------
# DB 생성
# -----------------------------
cur.execute("""
CREATE TABLE IF NOT EXISTS wallet(
    id INTEGER PRIMARY KEY,
    krw REAL
)
""")
cur.execute("""
CREATE TABLE IF NOT EXISTS positions(
    ticker TEXT PRIMARY KEY,
    qty REAL,
    buy_price REAL
)
""")
cur.execute("""
CREATE TABLE IF NOT EXISTS trades(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT,
    ticker TEXT,
    price REAL,
    qty REAL,
    side TEXT,
    trade_value REAL,
    profit REAL,
    profit_percent REAL
)
""")
cur.execute("""
CREATE TABLE IF NOT EXISTS learning(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    f1 REAL,f2 REAL,f3 REAL,f4 REAL,f5 REAL,
    f6 REAL,f7 REAL,f8 REAL,f9 REAL,f10 REAL,
    f11 REAL,f12 REAL,f13 REAL,f14 REAL,f15 REAL,
    f16 REAL,f17 REAL,f18 REAL,f19 REAL,f20 REAL,
    f21 REAL,f22 REAL,f23 REAL,f24 REAL,f25 REAL,
    f26 REAL,f27 REAL,f28 REAL,f29 REAL,f30 REAL,
    target INTEGER
)
""")
cur.execute("""
CREATE TABLE IF NOT EXISTS learning_meta(
    id INTEGER PRIMARY KEY,
    last_time TEXT
)
""")
conn.commit()

# 초기 데이터
if cur.execute("SELECT * FROM wallet").fetchone() is None:
    cur.execute("INSERT INTO wallet VALUES(1,10000000)")
if cur.execute("SELECT * FROM learning_meta").fetchone() is None:
    cur.execute("INSERT INTO learning_meta VALUES(1,'2000-01-01')")
conn.commit()

# -----------------------------
# 지갑/포지션 관리
# -----------------------------
def load_wallet():
    return cur.execute("SELECT krw FROM wallet WHERE id=1").fetchone()[0]

def save_wallet(krw):
    cur.execute("UPDATE wallet SET krw=? WHERE id=1", (krw,))
    conn.commit()

def load_positions():
    df = pd.read_sql("SELECT * FROM positions", conn)
    return {r.ticker: {"qty": r.qty, "buy_price": r.buy_price} for _, r in df.iterrows()}

def save_position(ticker, qty, buy_price):
    cur.execute("INSERT OR REPLACE INTO positions VALUES(?,?,?)", (ticker, qty, buy_price))
    conn.commit()

def delete_position(ticker):
    cur.execute("DELETE FROM positions WHERE ticker=?", (ticker,))
    conn.commit()

# -----------------------------
# 지표 및 feature
# -----------------------------
def indicators(df):
    df["ma5"] = df.close.rolling(5).mean()
    df["ma20"] = df.close.rolling(20).mean()
    delta = df.close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.rolling(14).mean() / down.rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + rs))
    df["momentum"] = df.close.pct_change(3)
    return df

def features(df):
    if df is None or df.empty:
        return [0.5]*30
    r = df.iloc[-1]
    f = [
        getattr(r,'rsi',0.5),
        getattr(r,'ma5',r.close)/r.close,
        getattr(r,'ma20',r.close)/r.close,
        getattr(r,'momentum',0.0)
    ]
    while len(f)<30:
        f.append(np.random.random())
    return f[:30]

# -----------------------------
# 거래 가능한 코인
# -----------------------------
def tradable():
    try:
        res = requests.get("https://api.upbit.com/v1/market/all").json()
        safe_coins = []
        for x in res:
            if not x["market"].startswith("KRW-"): continue
            if x.get("delisting_date"): continue
            if x.get("market_warning"): continue
            listed_dt = datetime.strptime(x.get("listed_date","2000-01-01"), "%Y-%m-%d")
            if datetime.now() - listed_dt < timedelta(days=7): continue
            safe_coins.append(x["market"])
        return safe_coins
    except:
        return []

def top100():
    coins = tradable()
    data = []
    for c in coins:
        try:
            df = pyupbit.get_ohlcv(c, "minute1", count=20)
            if df is None: continue
            val = (df.close * df.volume).sum()
            data.append((c,val))
        except:
            continue
    data = sorted(data, key=lambda x: x[1], reverse=True)
    return [x[0] for x in data[:100]]

# -----------------------------
# 학습 / 모델
# -----------------------------
def build_learning():
    coins = top100()
    last_time_str = cur.execute("SELECT last_time FROM learning_meta WHERE id=1").fetchone()[0]
    last_time = pd.to_datetime(last_time_str)
    for coin in coins:
        try:
            df = pyupbit.get_ohlcv(coin, "minute1", count=200)
            if df is None: continue
            df = indicators(df)
            df["target"] = (df.close.shift(-5) > df.close).astype(int)
            df = df.dropna()
            for i in range(len(df)-1):
                row_time = df.index[i]
                if row_time <= last_time: continue
                f = features(df.iloc[:i+1])
                t = df.iloc[i]["target"]
                cur.execute("INSERT INTO learning VALUES(NULL,"+ ",".join(["?"]*30) +",?)", f+[t])
        except:
            continue
    conn.commit()
    cur.execute("UPDATE learning_meta SET last_time=?", (datetime.now(),))
    conn.commit()

def train():
    df = pd.read_sql("SELECT * FROM learning", conn)
    if len(df) < 3000: return None
    X = df.drop(["id","target"], axis=1)
    y = df["target"]
    d = lgb.Dataset(X, label=y)
    params = {"objective":"binary", "learning_rate":0.03, "num_leaves":64}
    model = lgb.train(params, d, 150)
    return model

# -----------------------------
# 매매
# -----------------------------
def trade(model):
    krw = load_wallet()
    positions = load_positions()
    coins = top100()

    # BUY
    for coin in coins:
        if coin in positions: continue
        try:
            df = pyupbit.get_ohlcv(coin, "minute1", count=120)
            if df is None: continue
            df = indicators(df)
            f = features(df)
            prob = model.predict([f])[0]
            if prob < 0.6: continue
            price = pyupbit.get_current_price(coin)
            invest = krw*0.1
            if invest < 10000: continue
            qty = invest / price

            # 실제 매수
            upbit.buy_market_order(coin, invest)

            krw -= invest
            save_wallet(krw)
            save_position(coin, qty, price)
            trade_value = price*qty
            cur.execute("INSERT INTO trades VALUES(NULL,?,?,?,?,?,?,?,?,?)",
                        (datetime.now(), coin, price, qty, "BUY", trade_value, 0, 0))
            conn.commit()
        except:
            continue

    # SELL
    positions = load_positions()
    for coin, pos in positions.items():
        try:
            price = pyupbit.get_current_price(coin)
            profit = (price-pos["buy_price"])*pos["qty"]
            profit_percent = profit/(pos["buy_price"]*pos["qty"])*100
            df = pyupbit.get_ohlcv(coin,"minute1",count=120)
            df = indicators(df) if df is not None else None
            f = features(df)
            prob = model.predict([f])[0]

            if prob < 0.45 or profit > 0.08*pos["buy_price"]*pos["qty"] or profit < -0.03*pos["buy_price"]*pos["qty"]:
                qty = pos["qty"]
                krw = load_wallet()

                # 실제 매도
                upbit.sell_market_order(coin, qty*price)

                krw += qty*price
                save_wallet(krw)
                delete_position(coin)
                trade_value = price*qty
                cur.execute("INSERT INTO trades VALUES(NULL,?,?,?,?,?,?,?,?,?)",
                            (datetime.now(), coin, price, qty, "SELL", trade_value, profit, profit_percent))
                conn.commit()
        except:
            continue

# -----------------------------
# 백그라운드 엔진 (2분마다)
# -----------------------------
def ai_engine():
    while True:
        build_learning()
        model = train()
        if model:
            trade(model)
        time.sleep(120)  # 2분마다 반복

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("AI Crypto Trader 1분 자동 실거래")

# 백그라운드 스레드 실행
if "engine_started" not in st.session_state:
    t = threading.Thread(target=ai_engine, daemon=True)
    t.start()
    st.session_state.engine_started = True

# 자산 표시
krw = load_wallet()
positions = load_positions()
coin_value = 0
rows = []
for coin,pos in positions.items():
    try:
        price = pyupbit.get_current_price(coin)
        value = price*pos["qty"]
        coin_value += value
        profit = (price-pos["buy_price"])/pos["buy_price"]*100
        rows.append({"coin":coin,"qty":pos["qty"],"buy_price":pos["buy_price"],"price":price,"profit%":profit})
    except:
        continue

asset = krw + coin_value
c1,c2,c3 = st.columns(3)
c1.metric("총 자산", f"{asset:,.0f} KRW")
c2.metric("현금", f"{krw:,.0f} KRW")
c3.metric("코인 평가", f"{coin_value:,.0f} KRW")

st.subheader("보유 코인")
st.dataframe(pd.DataFrame(rows))

# -----------------------------
# 최근 거래: 종목, 매수총금액, 매도총금액, 수익, 수익률
# -----------------------------
hist = pd.read_sql("SELECT * FROM trades ORDER BY id DESC", conn)

summary = []
for ticker, g in hist.groupby("ticker"):
    buy_total = g[g.side=="BUY"]["trade_value"].sum()
    sell_total = g[g.side=="SELL"]["trade_value"].sum()
    profit_total = g[g.side=="SELL"]["profit"].sum()
    profit_percent = (profit_total/buy_total*100) if buy_total>0 else 0
    summary.append({
        "종목": ticker,
        "매수총금액": buy_total,
        "매도총금액": sell_total,
        "수익": profit_total,
        "수익률(%)": profit_percent
    })

st.subheader("최근 거래 요약")
st.dataframe(pd.DataFrame(summary))

st.write("⚠️ Streamlit 종료 후에도 2분마다 학습과 매매가 백그라운드에서 DB에 저장됩니다.")
