# -*- coding: utf-8 -*-
import pyupbit
import pandas as pd
import numpy as np
import sqlite3
import requests
import time
from datetime import datetime, timedelta
import lightgbm as lgb
import threading
import streamlit as st

st.set_page_config(page_title="AI Crypto Trader 1분 자동", layout="wide")

DB = "ai_trader_1min.db"
conn = sqlite3.connect(DB, check_same_thread=False)
cur = conn.cursor()

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

# -----------------------------
# 초기 데이터 설정
# -----------------------------
if cur.execute("SELECT * FROM wallet").fetchone() is None:
    cur.execute("INSERT INTO wallet VALUES(1,10000000)")
    conn.commit()
if cur.execute("SELECT * FROM learning_meta").fetchone() is None:
    cur.execute("INSERT INTO learning_meta VALUES(1,'2000-01-01')")
    conn.commit()

# -----------------------------
# 지갑 / 포지션 로드/저장
# -----------------------------
def load_wallet():
    krw = cur.execute("SELECT krw FROM wallet WHERE id=1").fetchone()[0]
    return krw

def save_wallet(krw):
    cur.execute("UPDATE wallet SET krw=? WHERE id=1", (krw,))
    conn.commit()

def load_positions():
    df=pd.read_sql("SELECT * FROM positions",conn)
    pos = {r.ticker:{"qty":r.qty,"buy_price":r.buy_price} for _,r in df.iterrows()}
    return pos

def save_position(ticker, qty, buy_price):
    cur.execute("INSERT OR REPLACE INTO positions VALUES(?,?,?)",(ticker, qty, buy_price))
    conn.commit()

def delete_position(ticker):
    cur.execute("DELETE FROM positions WHERE ticker=?", (ticker,))
    conn.commit()

# -----------------------------
# 지표 / feature
# -----------------------------
def indicators(df):
    df["ma5"]=df.close.rolling(5).mean()
    df["ma20"]=df.close.rolling(20).mean()
    delta=df.close.diff()
    up=delta.clip(lower=0)
    down=-delta.clip(upper=0)
    rs=up.rolling(14).mean()/down.rolling(14).mean()
    df["rsi"]=100-(100/(1+rs))
    df["momentum"]=df.close.pct_change(3)
    return df

def features(df):
    r=df.iloc[-1]
    f=[r.rsi,r.ma5/r.close,r.ma20/r.close,r.momentum]
    while len(f)<30:
        f.append(np.random.random())
    return f[:30]

# -----------------------------
# tradable / top100 (상폐/유의/상장7일 제외)
# -----------------------------
def tradable():
    url="https://api.upbit.com/v1/market/all"
    res=requests.get(url).json()
    safe_coins = []
    for x in res:
        if not x["market"].startswith("KRW-"):
            continue
        if "delisting_date" in x and x["delisting_date"]:
            continue
        if "market_warning" in x and x["market_warning"] != "":
            continue
        if "listed_date" in x:
            listed_dt = datetime.strptime(x["listed_date"], "%Y-%m-%d")
            if datetime.now() - listed_dt < timedelta(days=7):
                continue
        safe_coins.append(x["market"])
    return safe_coins

def top100():
    coins=tradable()
    data=[]
    for c in coins:
        try:
            df=pyupbit.get_ohlcv(c,"minute1",count=20)
            val=(df.close*df.volume).sum()
            data.append((c,val))
        except:
            pass
    data=sorted(data,key=lambda x:x[1],reverse=True)
    return [x[0] for x in data[:100]]

# -----------------------------
# 학습 / 모델
# -----------------------------
def build_learning():
    coins=top100()
    last_time = pd.to_datetime(cur.execute("SELECT last_time FROM learning_meta WHERE id=1").fetchone()[0])
    for coin in coins:
        df=pyupbit.get_ohlcv(coin,"minute1",count=200)
        if df is None: continue
        df=indicators(df)
        df["target"]=(df.close.shift(-5)>df.close).astype(int)
        df=df.dropna()
        for i in range(len(df)-1):
            row_time = df.index[i]
            if row_time <= last_time: continue
            f=features(df.iloc[:i+1])
            t=df.iloc[i]["target"]
            cur.execute("INSERT INTO learning VALUES(NULL,"+ ",".join(["?"]*30) +",?)", f+[t])
    conn.commit()
    cur.execute("UPDATE learning_meta SET last_time=?", (datetime.now(),))
    conn.commit()

def train():
    df=pd.read_sql("SELECT * FROM learning",conn)
    if len(df)<3000: return None
    X=df.drop(["id","target"],axis=1)
    y=df["target"]
    d=lgb.Dataset(X,label=y)
    params={"objective":"binary","learning_rate":0.03,"num_leaves":64}
    model=lgb.train(params,d,150)
    return model

# -----------------------------
# 매매
# -----------------------------
def trade(model):
    krw = load_wallet()
    positions = load_positions()
    coins=top100()
    # BUY
    for coin in coins:
        if coin in positions: continue
        df=pyupbit.get_ohlcv(coin,"minute1",count=120)
        if df is None: continue
        df=indicators(df)
        f=features(df)
        prob=model.predict([f])[0]
        if prob<0.6: continue
        price=pyupbit.get_current_price(coin)
        invest=krw*0.1
        if invest<10000: continue
        qty=invest/price
        krw -= invest
        save_wallet(krw)
        save_position(coin, qty, price)
        trade_value = price*qty
        cur.execute("INSERT INTO trades VALUES(NULL,?,?,?,?,?,?,?,?,?)",
                    (datetime.now(), coin, price, qty, "BUY", trade_value, 0, 0))
        conn.commit()
    # SELL
    positions=load_positions()
    for coin,pos in positions.items():
        price=pyupbit.get_current_price(coin)
        profit = (price-pos["buy_price"])*pos["qty"]
        profit_percent = profit/(pos["buy_price"]*pos["qty"])*100
        df=pyupbit.get_ohlcv(coin,"minute1",count=120)
        df=indicators(df)
        f=features(df)
        prob=model.predict([f])[0]
        if prob<0.45 or profit>0.08*pos["buy_price"]*pos["qty"] or profit<-0.03*pos["buy_price"]*pos["qty"]:
            qty=pos["qty"]
            krw = load_wallet()
            krw += qty*price
            save_wallet(krw)
            delete_position(coin)
            trade_value = price*qty
            cur.execute("INSERT INTO trades VALUES(NULL,?,?,?,?,?,?,?,?,?)",
                        (datetime.now(), coin, price, qty, "SELL", trade_value, profit, profit_percent))
            conn.commit()

# -----------------------------
# 백그라운드 엔진 (1분마다 반복)
# -----------------------------
def ai_engine():
    while True:
        build_learning()
        model=train()
        if model:
            trade(model)
        time.sleep(60)  # 1분마다 반복

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("AI Crypto Trader 1분 자동 백그라운드")

# 백그라운드 스레드 실행
if "engine_started" not in st.session_state:
    t=threading.Thread(target=ai_engine, daemon=True)
    t.start()
    st.session_state.engine_started=True

# 대시보드 표시
krw = load_wallet()
positions = load_positions()
coin_value = 0
rows=[]
for coin,pos in positions.items():
    price=pyupbit.get_current_price(coin)
    value=price*pos["qty"]
    coin_value += value
    profit=(price-pos["buy_price"])/pos["buy_price"]*100
    rows.append({"coin":coin,"qty":pos["qty"],"buy_price":pos["buy_price"],"price":price,"profit%":profit})

asset = krw + coin_value
c1,c2,c3 = st.columns(3)
c1.metric("총 자산", f"{asset:,.0f}")
c2.metric("현금", f"{krw:,.0f}")
c3.metric("코인 평가", f"{coin_value:,.0f}")

st.subheader("보유 코인")
st.dataframe(pd.DataFrame(rows))

hist=pd.read_sql("SELECT * FROM trades ORDER BY id DESC LIMIT 50",conn)
st.subheader("최근 거래")
st.dataframe(hist)

# 총합 계산
total_buy = hist[hist.side=="BUY"]["trade_value"].sum() if 'side' in hist.columns else 0
total_sell = hist[hist.side=="SELL"]["trade_value"].sum() if 'side' in hist.columns else 0
total_profit = hist[hist.side=="SELL"]["profit"].sum() if 'side' in hist.columns else 0
total_profit_percent = (total_profit/total_buy*100) if total_buy>0 else 0

st.subheader("총합")
st.write(f"총 매수금액: {total_buy:,.0f}원")
st.write(f"총 매도금액: {total_sell:,.0f}원")
st.write(f"총 이익금: {total_profit:,.0f}원")
st.write(f"총 수익률: {total_profit_percent:.2f}%")

st.write("⚠️ Streamlit 종료 후에도 1분마다 학습과 매매가 백그라운드에서 DB에 저장됩니다.")
