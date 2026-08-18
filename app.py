import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from google import genai
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- PAGE SETUP ---
st.set_page_config(page_title="Institutional Order Flow AI", layout="wide", page_icon="🏦")
st.title("🏦 Institutional Order Flow & Volume AI")

# --- SECURE API KEY LOGIC ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except KeyError:
    api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Add this to Streamlit Secrets later to remove this box.")

st.sidebar.markdown("---")
st.sidebar.subheader("Asset Selection")
asset_class = st.sidebar.radio(
    "Market Type",
    ["Indian Stocks (NSE)", "Crypto", "US Stocks"]
)

default_ticker = "RELIANCE.NS"
if "Crypto" in asset_class:
    default_ticker = "BTC-USD"
elif "US Stocks" in asset_class:
    default_ticker = "SPY"

ticker = st.sidebar.text_input("Ticker Symbol", value=default_ticker).upper()
timeframe = st.sidebar.selectbox("Intraday Interval", options=["15m", "1h"], index=0)

period_map = {"15m": "1mo", "1h": "3mo"}
period = period_map[timeframe]
analyze_button = st.sidebar.button("Analyze Order Flow")

# --- INSTITUTIONAL TECHNICAL ANALYSIS ---
def get_stock_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

def calculate_institutional_metrics(df, timeframe):
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0.0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['TPV'] = df['Typical_Price'] * df['Volume']
    
    if timeframe in ['15m', '1h']:
        df['VWAP'] = df.groupby(df.index.date, group_keys=False).apply(
            lambda x: x['TPV'].cumsum() / x['Volume'].cumsum()
        )
    else:
        df['VWAP'] = df['Typical_Price']
        
    buy_pressure = np.where(df['Close'] > df['Open'], df['Volume'], 0)
    sell_pressure = np.where(df['Close'] < df['Open'], df['Volume'], 0)
    df['Delta'] = buy_pressure - sell_pressure
    df['CVD'] = df['Delta'].cumsum()
    
    return df

def calculate_volume_profile(df, bins=20):
    recent_df = df.tail(100)
    min_price, max_price = recent_df['Low'].min(), recent_df['High'].max()
    price_bins = np.linspace(min_price, max_price, bins)
    
    vol_profile = np.zeros(bins-1)
    for i in range(len(recent_df)):
        tp = recent_df['Typical_Price'].iloc[i]
        vol = recent_df['Volume'].iloc[i]
        bin_idx = np.digitize(tp, price_bins) - 1
        if bin_idx >= bins - 1:
            bin_idx = bins - 2
        vol_profile[bin_idx] += vol
        
    poc_idx = np.argmax(vol_profile)
    poc_price = (price_bins[poc_idx] + price_bins[poc_idx+1]) / 2
    return poc_price

def draw_advanced_chart(df, ticker, interval, poc_price):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, row_heights=[0.7, 0.3])
    
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_9'], line=dict(color='blue', width=1), name='9 EMA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='orange', width=1.5), name='20 EMA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='purple', width=2, dash='dot'), name='VWAP'), row=1, col=1)
    fig.add_hline(y=poc_price, line_dash="dash", line_color="yellow", annotation_text="POC", row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['CVD'], line=dict(color='cyan', width=2), fill='tozeroy', name='CVD Proxy'), row=2, col=1)
    
    fig.update_layout(title=f"{ticker} Institutional Chart ({interval})", xaxis_rangeslider_visible=False, height=700, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# --- GEMINI ORDER FLOW INTEGRATION ---
def generate_institutional_setup(df, ticker, key, interval, poc_price):
    client = genai.Client(api_key=key)
    latest = df.iloc[-1]
    
    price_trend = "Up" if df['Close'].iloc[-1] > df['Close'].iloc[-5] else "Down"
    cvd_trend = "Up" if df['CVD'].iloc[-1] > df['CVD'].iloc[-5] else "Down"
    divergence_warning = "⚠️ POTENTIAL DIVERGENCE DETECTED" if price_trend != cvd_trend else "Price and Volume Momentum Aligned"

    prompt = f"""
    You are a tier-1 institutional prop trader looking at the {interval} chart for {ticker}.
    
    MARKET METRICS:
    - Current Price: ${latest['Close']:.2f}
    - Daily VWAP: ${latest['VWAP']:.2f}
    - Point of Control (POC): ${poc_price:.2f}
    - 9 EMA: ${latest['EMA_9']:.2f} | 20 EMA: ${latest['EMA_20']:.2f}
    - RSI: {latest['RSI_14']:.2f}
    
    ORDER FLOW STATE:
    - CVD/Volume Momentum: {divergence_warning}
    
    Provide an execution plan exactly like this:
    
    ### 🏦 Order Flow Context
    (1-2 sentences on who is in control based on price relative to VWAP and POC).
    
    ### 🎯 The Setup
    (Specific interaction we are waiting for).
    
    ### 💰 Trade Execution
    * **Entry Zone:** 
    * **Stop Loss:** 
    * **Target 1:** 
    * **Target 2:** 
    """
    
    with st.spinner("Analyzing VWAP, Order Flow, and Volume Nodes..."):
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        st.markdown(response.text)

# --- APP EXECUTION ---
if analyze_button:
    if not api_key:
        st.error("⚠️ Stop! I need your Gemini API Key in the sidebar or Secrets.")
    else:
        market_data = get_stock_data(ticker, period, timeframe)
        
        if market_data.empty:
            st.error("❌ Could not fetch data. Check your ticker symbol.")
        else:
            processed_data = calculate_institutional_metrics(market_data, timeframe)
            poc = calculate_volume_profile(processed_data)
            
            col1, col2 = st.columns([2, 1.2])
            with col1:
                draw_advanced_chart(processed_data, ticker, timeframe, poc)
            
            with col2:
                st.subheader("🧠 Algorithmic Execution Plan")
                generate_institutional_setup(processed_data, ticker, api_key, timeframe, poc)
