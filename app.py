import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import glob

# --- CONFIGURATION & CONSTANTS ---
st.set_page_config(page_title="Daily Stock Screener", layout="wide", page_icon="📈")

# Inject Custom CSS
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] button {
        color: var(--text-color) !important;
    }
    .stTabs [data-baseweb="tab-list"] button:hover {
        color: var(--text-color) !important;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] p {
        color: var(--text-color) !important;
        font-weight: bold !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #fafafa !important;
    }
    div[data-baseweb="select"] > div:focus-within {
        border-color: #fafafa !important;
        box-shadow: 0 0 0 1px #fafafa !important;
    }
    span[data-baseweb="tag"] {
        background-color: #fafafa !important;
        color: #000000 !important;
    }
    span[data-baseweb="tag"] svg {
        fill: #000000 !important;
    }
    </style>
""", unsafe_allow_html=True)

FILES = {'users': 'users.json'}
INDICES = {'^DJI': 'Dow Jones', '^GSPC': 'S&P 500', '^IXIC': 'NASDAQ'}

GRAPH_INDICES_MAP = {
    '.DJI': '^DJI', 
    '.INX': '^GSPC', 
    '.IXIC': '^IXIC'
}

COLORS = {
    'good_bg': '#d4f7d4', 'bad_bg': '#f7d4d4', 
    'text_good': 'green', 'text_bad': 'red'
}

TUNNEL_PERIODS = {'1mo': 21, '3mo': 63, '6mo': 126, '1yr': 252}
SMA_PERIODS = {'50d': 50, '100d': 100, '200d': 200}

def _local_extrema_indices(arr, order=3, find_max=False):
    indices = []
    for i in range(order, len(arr) - order):
        window = arr[i - order:i + order + 1]
        if find_max and arr[i] == np.max(window):
            indices.append(i)
        elif not find_max and arr[i] == np.min(window):
            indices.append(i)
    return indices

def detect_price_tunnel(closes):
    """Return upper/lower trend-line bounds at the end of the series if a clear tunnel exists."""
    y = closes.values.astype(float) if hasattr(closes, 'values') else np.asarray(closes, dtype=float)
    n = len(y)
    if n < 15:
        return None

    x = np.arange(n)
    min_idx = _local_extrema_indices(y, order=3, find_max=False)
    max_idx = _local_extrema_indices(y, order=3, find_max=True)

    if len(min_idx) < 2 or len(max_idx) < 2:
        return None

    lower_slope, lower_intercept = np.polyfit(min_idx, y[min_idx], 1)
    upper_slope, upper_intercept = np.polyfit(max_idx, y[max_idx], 1)

    avg_price = np.mean(y)
    if avg_price <= 0:
        return None

    slope_scale = avg_price / n
    if slope_scale > 0 and abs(upper_slope - lower_slope) / slope_scale > 0.6:
        return None

    upper_line = upper_slope * x + upper_intercept
    lower_line = lower_slope * x + lower_intercept

    if np.mean(upper_line - lower_line) <= 0:
        return None

    channel_width_pct = np.mean(upper_line - lower_line) / avg_price
    if channel_width_pct < 0.02:
        return None

    contained = np.sum((y >= lower_line) & (y <= upper_line)) / n
    if contained < 0.70:
        return None

    return {
        'upper_at_end': upper_slope * (n - 1) + upper_intercept,
        'lower_at_end': lower_slope * (n - 1) + lower_intercept,
    }

def compute_tunnel_signals(price_series):
    signals = []
    for label, days in TUNNEL_PERIODS.items():
        if len(price_series) < max(days, 15):
            continue
        segment = price_series.iloc[-days:]
        tunnel = detect_price_tunnel(segment)
        if not tunnel:
            continue

        current_price = segment.iloc[-1]
        upper = tunnel['upper_at_end']
        lower = tunnel['lower_at_end']

        if current_price >= upper:
            signals.append({'period': label, 'color': 'green'})
        elif current_price <= lower:
            signals.append({'period': label, 'color': 'red'})

    return signals

def compute_sma_signals(price_series, current_price):
    signals = []
    for label, window in SMA_PERIODS.items():
        if len(price_series) < window:
            continue
        sma = price_series.rolling(window).mean().iloc[-1]
        if pd.isna(sma):
            continue
        color = 'green' if current_price >= sma else 'red'
        signals.append({'period': label, 'color': color})
    return signals

def render_signal_badge(period, color):
    bg = COLORS['text_good'] if color == 'green' else COLORS['text_bad']
    return (
        f'<span style="display:inline-block;background:{bg};color:white;border-radius:999px;'
        f'padding:4px 8px;font-size:10px;font-weight:600;margin:2px;min-width:32px;text-align:center;">'
        f'{period}</span>'
    )

def render_signals_cell(signals):
    if not signals:
        return ''
    return ''.join(render_signal_badge(s['period'], s['color']) for s in signals)

def pct_change_bg(val):
    if not isinstance(val, (int, float)) or pd.isna(val):
        return ''
    if val > 0:
        return f'background-color:{COLORS["good_bg"]};color:black;'
    if val < 0:
        return f'background-color:{COLORS["bad_bg"]};color:black;'
    return ''

def render_indicators_table(df):
    header = (
        '<table style="width:100%;border-collapse:collapse;font-size:14px;">'
        '<thead><tr style="border-bottom:1px solid rgba(250,250,250,0.2);">'
        '<th style="text-align:left;padding:8px 12px;">Ticker</th>'
        '<th style="text-align:left;padding:8px 12px;">Price</th>'
        '<th style="text-align:left;padding:8px 12px;">% Change</th>'
        '<th style="text-align:left;padding:8px 12px;">% Change (5d)</th>'
        '<th style="text-align:left;padding:8px 12px;">Tunnels</th>'
        '<th style="text-align:left;padding:8px 12px;">SMAs</th>'
        '</tr></thead><tbody>'
    )
    rows = []
    for _, row in df.iterrows():
        chg_bg = pct_change_bg(row['% Change'])
        chg5_bg = pct_change_bg(row['% Change (5d)'])
        price = row['Price']
        price_str = f'${price:,.2f}' if pd.notna(price) else 'N/A'
        chg_str = f"{row['% Change']:+.2f}%" if pd.notna(row['% Change']) else 'N/A'
        chg5_str = f"{row['% Change (5d)']:+.2f}%" if pd.notna(row['% Change (5d)']) else 'N/A'
        rows.append(
            f'<tr style="border-bottom:1px solid rgba(250,250,250,0.08);">'
            f'<td style="padding:8px 12px;font-weight:600;">{row["Ticker"]}</td>'
            f'<td style="padding:8px 12px;">{price_str}</td>'
            f'<td style="padding:8px 12px;{chg_bg}">{chg_str}</td>'
            f'<td style="padding:8px 12px;{chg5_bg}">{chg5_str}</td>'
            f'<td style="padding:8px 12px;">{render_signals_cell(row["tunnel_signals"])}</td>'
            f'<td style="padding:8px 12px;">{render_signals_cell(row["sma_signals"])}</td>'
            f'</tr>'
        )
    return header + ''.join(rows) + '</tbody></table>'

# --- PERSISTENCE FUNCTIONS ---
def load_data(key, default_data=None):
    if key not in st.session_state:
        try:
            url = f"https://api.jsonbin.io/v3/b/{st.secrets['JSONBIN_BIN_ID']}/latest"
            headers = {"X-Access-Key": st.secrets['JSONBIN_KEY']}
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                st.session_state[key] = response.json()['record']
            else:
                st.error(f"Database Read Error ({response.status_code}): {response.text}")
                st.session_state[key] = default_data if default_data is not None else {}
        except Exception as e:
            st.error(f"Failed to load cloud data: {e}")
            st.session_state[key] = default_data if default_data is not None else {}
            
    return st.session_state[key]

def save_data(key, data):
    st.session_state[key] = data
    try:
        url = f"https://api.jsonbin.io/v3/b/{st.secrets['JSONBIN_BIN_ID']}"
        headers = {
            "Content-Type": "application/json",
            "X-Access-Key": st.secrets['JSONBIN_KEY']
        }
        response = requests.put(url, json=data, headers=headers)
        
        if response.status_code != 200:
            st.error(f"Database Write Error ({response.status_code}): {response.text}")
    except Exception as e:
        st.error(f"Failed to save to cloud: {e}") 

# --- HELPER FUNCTIONS ---
def format_days_until(target_date):
    if not target_date: return "-"
    today = datetime.now().date()
    if isinstance(target_date, datetime): target_date = target_date.date()
    delta = (target_date - today).days
    if delta < 0: return "-"
    elif delta == 0: return "Today"
    elif delta == 1: return "1 day"
    else: return f"{delta} days"

@st.cache_data(ttl=3600)
def get_stock_metadata(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    div_days, earn_days = "-", "-"
    try:
        div_timestamp = stock.info.get('exDividendDate')
        if div_timestamp:
            div_date = datetime.fromtimestamp(div_timestamp)
            div_days = format_days_until(div_date)
    except: pass

    try:
        cal = stock.calendar
        if cal and 'Earnings Date' in cal:
            earn_dates = cal['Earnings Date']
            if earn_dates:
                earn_days = format_days_until(earn_dates[0])
    except: pass
    return div_days, earn_days

@st.cache_data(ttl=3600)
def get_put_call_ratio(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        expirations = stock.options
        if not expirations: return None
        
        total_puts, total_calls = 0, 0
        for date in expirations[:4]:
            opt = stock.option_chain(date)
            total_puts += opt.puts['openInterest'].fillna(0).sum()
            total_calls += opt.calls['openInterest'].fillna(0).sum()
            
        if total_calls == 0: return None
        return round(total_puts / total_calls, 2)
    except:
        return None

@st.cache_data(ttl=300)
def get_market_data(tickers):
    if not tickers: return pd.DataFrame()
    data = []
    tickers_str = " ".join(tickers)
    
    try:
        df = yf.download(tickers_str, period="1y", progress=False)
    except Exception:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        closes = df['Close']
        volumes = df['Volume']
    else:
        closes = df
        volumes = df['Volume'] if 'Volume' in df.columns else None

    for ticker in tickers:
        try:
            if len(tickers) == 1 and isinstance(closes, pd.Series):
                price_series, vol_series = closes, volumes
            elif len(tickers) == 1 and isinstance(closes, pd.DataFrame):
                price_series = closes[closes.columns[0]]
                vol_series = volumes[volumes.columns[0]]
            else:
                price_series = closes[ticker]
                vol_series = volumes[ticker]

            current_price = price_series.iloc[-1]
            
            # Daily % Change
            prev_close = price_series.iloc[-2] if len(price_series) >= 2 else current_price
            change_pct = ((current_price - prev_close) / prev_close) * 100
            
            # 5-Day % Change
            close_5d = price_series.iloc[-6] if len(price_series) >= 6 else price_series.iloc[0]
            change_5d_pct = ((current_price - close_5d) / close_5d) * 100
            
            vol_percentile = vol_series.rank(pct=True).iloc[-1] * 100
            vol_percentile = min(99.0, vol_percentile)

            div_days, earn_days = get_stock_metadata(ticker)
            pc_ratio = get_put_call_ratio(ticker)
            tunnel_signals = compute_tunnel_signals(price_series)
            sma_signals = compute_sma_signals(price_series, current_price)

            data.append({
                "Ticker": ticker,
                "Price": current_price,
                "% Change": change_pct,
                "% Change (5d)": change_5d_pct,
                "Vol %ile (365d)": vol_percentile,
                "Put/Call Ratio": pc_ratio,
                "Next Earn": earn_days,
                "Next Div": div_days,
                "tunnel_signals": tunnel_signals,
                "sma_signals": sma_signals,
            })
        except Exception: continue

    return pd.DataFrame(data)

@st.cache_data(ttl=3600)
def get_fear_and_greed():
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        r = requests.get(url, headers=headers, timeout=5)
        r.raise_for_status()
        data = r.json()
        score = int(data['fear_and_greed_historical']['data'][-1]['y'])
        
        if score <= 24: rating = "Extreme Fear"
        elif score <= 44: rating = "Fear"
        elif score <= 54: rating = "Neutral"
        elif score <= 74: rating = "Greed"
        else: rating = "Extreme Greed"
        
        return score, rating
    except:
        return "N/A", "Unavailable"

@st.cache_data(ttl=300)
def get_historical_prices(ticker, days):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=f"{days}d")
        return hist
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_analyst_targets(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        low = info.get('targetLowPrice')
        mean = info.get('targetMeanPrice')
        high = info.get('targetHighPrice')
        current = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
        return (
            float(low) if low is not None else None,
            float(mean) if mean is not None else None,
            float(high) if high is not None else None,
            float(current) if current is not None else None,
        )
    except:
        return None, None, None, None

@st.cache_data(ttl=43200)
def get_economic_events():
    from ecocal import Calendar
    start_date = datetime.now()
    end_date = start_date + timedelta(days=7)
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    before_files = set(glob.glob("*.csv"))
    
    ec = Calendar(startHorizon=start_str, endHorizon=end_str, withDetails=True, nbThreads=20, preBuildCalendar=True)
    ec.saveCalendar()
    
    after_files = set(glob.glob("*.csv"))
    new_files = after_files - before_files
    
    df = pd.DataFrame()
    if new_files:
        latest_file = list(new_files)[0]
        df = pd.read_csv(latest_file)
        os.remove(latest_file)
    else:
        all_csvs = glob.glob("ecocal_*.csv")
        if all_csvs:
            latest_file = max(all_csvs, key=os.path.getctime)
            df = pd.read_csv(latest_file)
            os.remove(latest_file)
            
    return df

def render_tracked_stock_bar(ticker, buy_price, low, mean, high, current, is_last=False):
    divider = '' if is_last else 'border-bottom:1px solid rgba(250,250,250,0.08);'

    if low is None or high is None or low >= high:
        parts = [
            f'<div style="display:flex;align-items:center;padding:10px 0;{divider}">',
            f'<div style="min-width:90px;font-weight:bold;font-size:15px;color:#fafafa;">{ticker}</div>',
            '<div style="color:rgba(250,250,250,0.45);font-size:13px;margin-left:16px;">No analyst target prices available</div>',
            '</div>',
        ]
        return ''.join(parts)

    all_values = [low, high, buy_price]
    if current is not None:
        all_values.append(current)
    if mean is not None:
        all_values.append(mean)
    scale_min = min(all_values)
    scale_max = max(all_values)
    if scale_max == scale_min:
        scale_max = scale_min + 1

    r = scale_max - scale_min

    def to_pct(p):
        return max(0.0, min(100.0, (p - scale_min) / r * 100))

    low_pct = to_pct(low)
    high_pct = to_pct(high)
    mean_pct = to_pct(mean) if mean is not None else (low_pct + high_pct) / 2
    buy_pct = to_pct(buy_price)

    current_marker = ''
    if current is not None:
        cp = to_pct(current)
        current_marker = ''.join([
            f'<div style="position:absolute;left:{cp:.2f}%;transform:translateX(-50%);text-align:center;bottom:0;z-index:2;">',
            f'<span style="font-size:9px;color:rgba(250,250,250,0.95);display:block;white-space:nowrap;margin-bottom:2px;">${current:.2f}</span>',
            '<span style="color:rgba(250,250,250,0.95);font-size:12px;line-height:1;">&#9679;</span>',
            '</div>',
        ])

    buy_marker = ''.join([
        f'<div style="position:absolute;left:{buy_pct:.2f}%;transform:translateX(-50%);text-align:center;bottom:0;z-index:1;">',
        f'<span style="font-size:9px;color:#f0c040;display:block;white-space:nowrap;margin-bottom:2px;">${buy_price:.2f}</span>',
        '<span style="color:#f0c040;font-size:12px;line-height:1;">&#9660;</span>',
        '</div>',
    ])

    mean_label = f'${mean:.2f}' if mean is not None else ''

    above = f'<div style="position:relative;height:44px;">{current_marker}{buy_marker}</div>'
    bar = ''.join([
        f'<div style="position:absolute;left:{low_pct:.2f}%;width:{high_pct - low_pct:.2f}%;height:18px;background:transparent;',
        'border:1px solid rgba(255,255,255,0.5);">',
        f'<div style="position:absolute;left:{((mean_pct - low_pct) / (high_pct - low_pct) * 100) if high_pct != low_pct else 50:.2f}%;top:0;bottom:0;width:2px;background:rgba(255,255,255,0.5);transform:translateX(-50%);"></div>',
        '</div>',
    ])
    bar_container = f'<div style="position:relative;height:18px;">{bar}</div>'
    below = ''.join([
        '<div style="position:relative;height:20px;margin-top:5px;">',
        f'<span style="position:absolute;left:{low_pct:.2f}%;transform:translateX(-50%);font-size:9px;color:rgba(250,250,250,0.55);white-space:nowrap;">${low:.2f}</span>',
        f'<span style="position:absolute;left:{mean_pct:.2f}%;transform:translateX(-50%);font-size:9px;color:rgba(250,250,250,0.55);white-space:nowrap;">{mean_label}</span>',
        f'<span style="position:absolute;left:{high_pct:.2f}%;transform:translateX(-50%);font-size:9px;color:rgba(250,250,250,0.55);white-space:nowrap;">${high:.2f}</span>',
        '</div>',
    ])

    parts = [
        f'<div style="display:flex;align-items:center;padding:10px 0;{divider}">',
        f'<div style="min-width:90px;font-weight:bold;font-size:15px;color:#fafafa;flex-shrink:0;">{ticker}</div>',
        f'<div style="flex:1;position:relative;padding:0 8px;">{above}{bar_container}{below}</div>',
        '</div>',
    ]
    return ''.join(parts)

# --- MAIN DASHBOARD ---
default_users = {"Default User": {"categories": ["Uncategorized"], "stocks": {}}}
users_data = load_data('users', default_users)
user_list = list(users_data.keys())

# 1. DAY OF WEEK & DATE
st.title(datetime.now().strftime('%A, %B %d, %Y'))
st.markdown("---")

# 2. USER SELECTION
if not user_list:
    st.warning("No users found. Please create a user profile in the sidebar.")
    current_user = None
else:
    if 'current_user' not in st.session_state or st.session_state.current_user not in user_list:
        st.session_state.current_user = user_list[0]
    
    current_user = st.selectbox("Select User Profile", user_list, index=user_list.index(st.session_state.current_user))
    if current_user != st.session_state.current_user:
        st.session_state.current_user = current_user
        st.rerun()

st.markdown("---")

# --- SIDEBAR LOGIC ---
st.sidebar.subheader("Users")
with st.sidebar.form("add_user_form", clear_on_submit=True):
    new_user = st.text_input("Name").strip()
    submitted_user = st.form_submit_button("Add User")
    if submitted_user and new_user:
        if len(users_data) >= 4:
            st.sidebar.error("Maximum of 4 user profiles allowed.")
        elif new_user in users_data:
            st.sidebar.error("User already exists.")
        else:
            users_data[new_user] = {"categories": ["Uncategorized"], "stocks": {}}
            save_data('users', users_data)
            st.session_state.current_user = new_user
            st.rerun()

if users_data:
    to_remove_user = st.sidebar.selectbox("Remove User", ["Select..."] + user_list)
    if to_remove_user != "Select...":
        if st.sidebar.button(f"Delete {to_remove_user}"):
            del users_data[to_remove_user]
            save_data('users', users_data)
            if st.session_state.current_user == to_remove_user:
                st.session_state.current_user = list(users_data.keys())[0] if users_data else None
            st.rerun()

st.sidebar.markdown("---")

# --- PORTFOLIO SIDEBAR UPDATES ---
if current_user:
    user_data = users_data[current_user]
    
    # MIGRATION: Auto-convert old user list formats to new dictionary format
    if isinstance(user_data, list):
        user_data = {"categories": ["Uncategorized"], "stocks": {t: {"category": "Uncategorized", "shares": 0.0} for t in user_data}}
        users_data[current_user] = user_data
        save_data('users', users_data)
        
    current_categories = user_data.get("categories", ["Uncategorized"])
    current_stocks = user_data.get("stocks", {})
    
    st.sidebar.subheader(f"{current_user}'s Portfolio")
    
    # CATEGORY MANAGEMENT
    with st.sidebar.expander("Manage Categories"):
        with st.form("add_cat_form", clear_on_submit=True):
            new_cat = st.text_input("New Category Name").strip()
            if st.form_submit_button("Add Category") and new_cat:
                if new_cat not in current_categories:
                    current_categories.append(new_cat)
                    users_data[current_user]["categories"] = current_categories
                    save_data('users', users_data)
                    st.rerun()
                    
        if len(current_categories) > 0:
            rem_cat = st.selectbox("Remove Category", ["Select..."] + current_categories)
            if rem_cat != "Select..." and st.button(f"Delete '{rem_cat}'"):
                current_categories.remove(rem_cat)
                if not current_categories:
                    current_categories.append("Uncategorized")
                default_cat = current_categories[0]
                
                # Reassign stocks from deleted category to default category
                for s, d in current_stocks.items():
                    if d["category"] == rem_cat:
                        d["category"] = default_cat
                users_data[current_user]["categories"] = current_categories
                users_data[current_user]["stocks"] = current_stocks
                save_data('users', users_data)
                st.rerun()

    # STOCK MANAGEMENT
    with st.sidebar.expander("Manage Stocks", expanded=True):
        with st.form("add_stock_form", clear_on_submit=True):
            new_stock = st.text_input("Ticker").upper().strip()
            sel_cat = st.selectbox("Category", current_categories)
            shares_val = st.number_input("Shares Owned", min_value=0.0, step=1.0, value=0.0)
            
            if st.form_submit_button("Add / Update Stock") and new_stock:
                current_stocks[new_stock] = {"category": sel_cat, "shares": shares_val}
                users_data[current_user]["stocks"] = current_stocks
                save_data('users', users_data)
                st.rerun()

        if current_stocks:
            to_remove_stock = st.selectbox("Remove Stock", ["Select..."] + list(current_stocks.keys()))
            if to_remove_stock != "Select..." and st.button(f"Drop {to_remove_stock}"):
                del current_stocks[to_remove_stock]
                users_data[current_user]["stocks"] = current_stocks
                save_data('users', users_data)
                st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Telegram Notifications")

    current_chat_id = user_data.get("telegram_chat_id", "")
    with st.sidebar.expander("Link Telegram", expanded=False):
        st.markdown(
            "Send `/start` to your bot, then enter the chat ID it provides.",
            help="Your chat ID is never displayed publicly."
        )
        chat_id_input = st.text_input(
            "Chat ID", value=str(current_chat_id) if current_chat_id else "",
            type="password", key="telegram_chat_id_input"
        )
        if st.button("Save Chat ID", key="save_chat_id_btn"):
            users_data[current_user]["telegram_chat_id"] = chat_id_input.strip()
            save_data('users', users_data)
            st.success("Telegram linked!")
            st.rerun()
        if current_chat_id:
            st.caption("✓ Telegram linked")

    st.sidebar.markdown("---")
    st.sidebar.subheader(f"{current_user}'s Tracked Stocks")

    tracked_stocks = user_data.get("tracked_stocks", {})

    with st.sidebar.expander("Manage Tracked Stocks", expanded=True):
        with st.form("add_tracked_form", clear_on_submit=True):
            tracked_ticker = st.text_input("Ticker", key="tracked_ticker_input").upper().strip()
            buy_price_val = st.number_input("My Buy Price ($)", min_value=0.01, step=0.5, value=100.0)
            if st.form_submit_button("Add / Update") and tracked_ticker:
                if tracked_ticker in current_stocks:
                    st.error(f"{tracked_ticker} is already in your portfolio.")
                else:
                    tracked_stocks[tracked_ticker] = buy_price_val
                    users_data[current_user]["tracked_stocks"] = tracked_stocks
                    save_data('users', users_data)
                    st.rerun()

        if tracked_stocks:
            to_remove_tracked = st.selectbox("Remove Tracked Stock", ["Select..."] + list(tracked_stocks.keys()), key="remove_tracked_select")
            if to_remove_tracked != "Select..." and st.button(f"Drop {to_remove_tracked}", key="drop_tracked_btn"):
                del tracked_stocks[to_remove_tracked]
                users_data[current_user]["tracked_stocks"] = tracked_stocks
                save_data('users', users_data)
                st.rerun()

# 3. INDICES & FEAR/GREED
if 'carousel_idx' not in st.session_state: st.session_state.carousel_idx = 0

st.subheader("Indexes")

indices_df = get_market_data(list(INDICES.keys()))
f_score, f_rating = get_fear_and_greed()

cards = []
for ticker, name in INDICES.items():
    if not indices_df.empty:
        row = indices_df[indices_df['Ticker'] == ticker]
        if not row.empty:
            val = row.iloc[0]['Price']
            chg = row.iloc[0]['% Change']
            cards.append({"name": name, "value": f"{val:,.2f}", "delta": f"{chg:+.2f}%", "is_pct": True, "raw_chg": chg})

cards.append({"name": "Fear & Greed Index", "value": str(f_score), "delta": f_rating, "is_pct": False, "raw_chg": 0})

col_l, col_c1, col_c2, col_c3, col_r = st.columns([0.5, 2, 2, 2, 0.5])
with col_l:
    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("◀", key="left_arrow"): st.session_state.carousel_idx = (st.session_state.carousel_idx - 1) % len(cards)
with col_r:
    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("▶", key="right_arrow"): st.session_state.carousel_idx = (st.session_state.carousel_idx + 1) % len(cards)

display_cols = [col_c1, col_c2, col_c3]
for i in range(3):
    card_index = (st.session_state.carousel_idx + i) % len(cards)
    card = cards[card_index]
    with display_cols[i]:
        delta_color = "normal" if card['is_pct'] else "off"
        st.metric(label=card['name'], value=card['value'], delta=card['delta'], delta_color=delta_color)

st.markdown("---")

# 4. PORTFOLIO TABLE & HEAT MAP CHARTS
if current_user:
    st.subheader(f"{current_user}'s Portfolio")
    
    my_stocks_dict = users_data[current_user].get("stocks", {})
    my_stocks_list = list(my_stocks_dict.keys())

    if not my_stocks_list:
        st.info(f"👈 Add stocks in the sidebar to populate {current_user}'s portfolio.")
    else:
        if 'portfolio_view_idx' not in st.session_state:
            st.session_state.portfolio_view_idx = 0

        with st.spinner('Updating Portfolio Data...'):
            stock_df = get_market_data(my_stocks_list)

        if not stock_df.empty:
            # Layout: Button | Content | Button
            col_view_l, col_view_c, col_view_r = st.columns([0.05, 0.9, 0.05])
            
            with col_view_l:
                # Add vertical breaks to roughly center the button next to the visual
                st.markdown("<br><br><br><br><br><br>", unsafe_allow_html=True)
                if st.button("◀", key="port_left"):
                    st.session_state.portfolio_view_idx = (st.session_state.portfolio_view_idx - 1) % 3
                    st.rerun()
                    
            with col_view_c:
                if st.session_state.portfolio_view_idx == 0:
                    # --- TABLE VIEW ---
                    def highlight_change(val):
                        if isinstance(val, (int, float)):
                            color = COLORS['good_bg'] if val > 0 else COLORS['bad_bg'] if val < 0 else ''
                            return f'background-color: {color}; color: black'
                        return ''

                    def highlight_pc(val):
                        if isinstance(val, (int, float)):
                            if val < 1.0: return f'background-color: {COLORS["good_bg"]}; color: black'
                            elif val > 1.0: return f'background-color: {COLORS["bad_bg"]}; color: black'
                        return ''

                    st.dataframe(
                        stock_df.style
                        .map(highlight_change, subset=['% Change', '% Change (5d)'])
                        .map(highlight_pc, subset=['Put/Call Ratio'])
                        .format({
                            "Price": lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A", 
                            "% Change": lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A",
                            "% Change (5d)": lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A",
                            "Vol %ile (365d)": lambda x: f"{x:.0f}%" if pd.notna(x) else "N/A",
                            "Put/Call Ratio": lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
                        }),
                        column_order=("Ticker", "Price", "% Change", "% Change (5d)", "Vol %ile (365d)", "Put/Call Ratio", "Next Earn", "Next Div"),
                        hide_index=True,
                        width="stretch"
                    )
                elif st.session_state.portfolio_view_idx == 1:
                    # --- TUNNELS & SMAs TABLE VIEW ---
                    st.markdown(render_indicators_table(stock_df), unsafe_allow_html=True)
                else:
                    # --- HEAT MAP VIEW ---
                    hm_data = []
                    for ticker, info in my_stocks_dict.items():
                        shares = info.get('shares', 0.0)
                        if shares > 0:
                            row = stock_df[stock_df['Ticker'] == ticker]
                            if not row.empty:
                                price = row.iloc[0]['Price']
                                pct_change = row.iloc[0]['% Change']
                                total_val = price * shares
                                hm_data.append({
                                    "Category": info.get('category', 'Uncategorized'),
                                    "Ticker": ticker,
                                    "Total Value": total_val,
                                    "Daily Change (%)": pct_change,
                                    "Custom Label": f"<b>{ticker}</b><br>{pct_change:+.2f}%"
                                })
                                
                    if hm_data:
                        plot_df = pd.DataFrame(hm_data)
                        fig = px.treemap(
                            plot_df,
                            path=[px.Constant("Portfolio"), "Category", "Ticker"],
                            values="Total Value",
                            color="Daily Change (%)",
                            color_continuous_scale=[(0, COLORS['text_bad']), (0.5, "white"), (1, COLORS['text_good'])],
                            color_continuous_midpoint=0,
                            custom_data=["Custom Label", "Daily Change (%)", "Total Value"]
                        )
                        fig.update_traces(
                            texttemplate="%{customdata[0]}",
                            textposition="middle center",
                            hovertemplate="<b>%{label}</b><br>Value: $%{customdata[2]:,.2f}<br>Daily Change: %{customdata[1]:+.2f}%<extra></extra>"
                        )
                        fig.update_layout(margin=dict(t=30, l=10, r=10, b=10), height=600)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No stocks with >0 shares found. Edit your stocks in the sidebar to assign shares.")

            with col_view_r:
                st.markdown("<br><br><br><br><br><br>", unsafe_allow_html=True)
                if st.button("▶", key="port_right"):
                    st.session_state.portfolio_view_idx = (st.session_state.portfolio_view_idx + 1) % 3
                    st.rerun()

            st.markdown("---")
            
            # --- CHARTING LOGIC ---
            st.subheader(f"{current_user}'s Graphs")
            
            graph_dropdown_options = list(GRAPH_INDICES_MAP.keys()) + my_stocks_list
            selected_ticker = st.selectbox("Select a ticker to view its graph", graph_dropdown_options, key="graph_select")
            
            if selected_ticker:
                tab30, tab90, tab365 = st.tabs(["30 Days", "90 Days", "365 Days"])
                
                def plot_chart(days):
                    actual_ticker = GRAPH_INDICES_MAP.get(selected_ticker, selected_ticker)
                    hist_data = get_historical_prices(actual_ticker, days)
                    
                    if not hist_data.empty:
                        start_price = hist_data['Close'].iloc[0]
                        end_price = hist_data['Close'].iloc[-1]
                        line_color = COLORS['text_good'] if end_price >= start_price else COLORS['text_bad']

                        vol_colors = [
                            COLORS['text_good'] if row['Close'] >= row['Open'] else COLORS['text_bad'] 
                            for _, row in hist_data.iterrows()
                        ]

                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                            vertical_spacing=0.03, row_heights=[0.75, 0.25])

                        fig.add_trace(go.Scatter(
                            x=hist_data.index, y=hist_data['Close'], 
                            mode='lines', name='Close Price',
                            line=dict(color=line_color, width=2)
                        ), row=1, col=1)

                        fig.add_trace(go.Bar(
                            x=hist_data.index, y=hist_data['Volume'],
                            name='Volume',
                            marker_color=vol_colors
                        ), row=2, col=1)

                        fig.update_layout(
                            title=f"{selected_ticker} - {days} Day History",
                            height=500, margin=dict(l=10, r=10, t=40, b=10),
                            plot_bgcolor='rgba(255,255,255,0.05)',
                            showlegend=False
                        )
                        
                        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
                        fig.update_yaxes(title_text="Volume", row=2, col=1)

                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"Could not load data for {selected_ticker}.")

                with tab30: plot_chart(30)
                with tab90: plot_chart(90)
                with tab365: plot_chart(365)

# --- TRACKED STOCKS SECTION ---
if current_user:
    tracked_stocks = users_data[current_user].get("tracked_stocks", {})
    st.markdown("---")
    st.subheader(f"{current_user}'s Tracked Stocks")

    if not tracked_stocks:
        st.info("👈 Add stocks to track in the sidebar.")
    else:
        target_data = {}
        with st.spinner("Fetching analyst targets..."):
            for ticker, buy_price in tracked_stocks.items():
                low, mean, high, current_price = get_analyst_targets(ticker)
                target_data[ticker] = {"buy_price": buy_price, "low": low, "mean": mean, "high": high, "current": current_price}

        items = sorted(target_data.items(), key=lambda x: x[0])
        bars_html = ''.join(
            render_tracked_stock_bar(t, d["buy_price"], d["low"], d["mean"], d["high"], d["current"], is_last=(i == len(items) - 1))
            for i, (t, d) in enumerate(items)
        )
        st.markdown(f'<div style="padding:0 8px;">{bars_html}</div>', unsafe_allow_html=True)

# --- EVENTS LOGIC ---
st.markdown("---")
st.subheader("Events")

selected_currencies = st.multiselect(
    "Filter by Currency:",
    options=['USD', 'JPY', 'EUR', 'CNY'],
    default=['USD', 'JPY', 'EUR', 'CNY'],
    key="currency_select"
)

with st.spinner("Fetching upcoming calendar events..."):
    try:
        events_df = get_economic_events()
        
        if not events_df.empty and selected_currencies:
            if 'Impact' in events_df.columns:
                events_df = events_df[events_df['Impact'].isin(['MEDIUM', 'HIGH'])]
                
            if 'Currency' in events_df.columns:
                events_df = events_df[events_df['Currency'].isin(selected_currencies)]
                
            if 'Start' in events_df.columns:
                events_df['Start_dt'] = pd.to_datetime(events_df['Start'], errors='coerce')
                events_df = events_df.sort_values('Start_dt').dropna(subset=['Start_dt'])
                
                today = datetime.now().date()
                def calc_days_until(dt_val):
                    delta = (dt_val.date() - today).days
                    if delta == 0: return "Today"
                    elif delta == 1: return "1 day"
                    else: return f"{delta} days"
                        
                events_df['Days Until'] = events_df['Start_dt'].apply(calc_days_until)
                events_df['Start'] = events_df['Start_dt'].dt.strftime('%m/%d/%Y %H:%M:%S')
            
            display_cols = ['Start', 'Days Until', 'Name', 'Currency', 'urlSource']
            display_cols = [col for col in display_cols if col in events_df.columns]
            
            final_events_df = events_df[display_cols].copy()
            
            if not final_events_df.empty:
                st.dataframe(
                    final_events_df,
                    column_config={
                        "urlSource": st.column_config.LinkColumn("Link", display_text="View Source"),
                        "Name": st.column_config.TextColumn("Event Name", width="large")
                    },
                    hide_index=True,
                    width="stretch"
                )
            else:
                st.info(f"No high/medium impact events found for {', '.join(selected_currencies)} in the next 7 days.")
        elif not selected_currencies:
            st.info("Please select at least one currency to view events.")
        else:
            st.info("No events data could be retrieved.")
    except Exception as e:
        st.warning(f"Unable to load economic calendar: {e}")
