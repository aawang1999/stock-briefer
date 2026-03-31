import streamlit as st
import pandas as pd
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

            data.append({
                "Ticker": ticker,
                "Price": current_price,
                "% Change": change_pct,
                "% Change (5d)": change_5d_pct,
                "Vol %ile (365d)": vol_percentile,
                "Put/Call Ratio": pc_ratio,
                "Next Earn": earn_days,
                "Next Div": div_days
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
                    st.session_state.portfolio_view_idx = (st.session_state.portfolio_view_idx - 1) % 2
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
                    st.session_state.portfolio_view_idx = (st.session_state.portfolio_view_idx + 1) % 2
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
