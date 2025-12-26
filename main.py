import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Plotting config
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- Configuration ---
TICKER = "9868.HK"
START_DATE = "2024-06-01"
END_DATE = "2024-12-25"
INITIAL_CAPITAL = 100000
COMMISSION = 0.002  # 20bps transaction cost

# Strategy Params
WINDOW = 10
K = 1.2
RSI_PERIOD = 14
RSI_LOWER, RSI_UPPER = 30, 80
STOP_LOSS_PCT = 0.03  # Tight stop for swing trading

# Position Sizing (Pyramid)
INIT_POS_RATIO = 0.5  # Entry size
ADD_ON_THRESHOLD = 0.03  # Add position after 3% floating profit

# Key Events for Annotation
EVENTS = {
    "2024-08-27": "MONA M03 Launch",
    "2024-10-10": "P7+ Debut",
    "2024-11-07": "P7+ Official Launch"
}

def run_backtest():
    print(f"Starting backtest for {TICKER}...")
    
    # 1. Data Fetching
    try:
        df = yf.download(TICKER, start=START_DATE, end=END_DATE)
        if df.empty: raise ValueError("No data found")
        
        df = df[['Close', 'Volume']].copy()
        df.columns = ['Price', 'Volume']
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. Indicator Calculation
    df['MA'] = df['Price'].rolling(WINDOW).mean()
    std = df['Price'].rolling(WINDOW).std()
    df['Upper'] = df['MA'] + K * std
    df['Lower'] = df['MA'] - K * std
    
    # RSI
    delta = df['Price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(RSI_PERIOD).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 3. Execution Loop
    position = 0
    cash = INITIAL_CAPITAL
    equity = []
    trade_log = []
    
    high_water_mark = 0 # For trailing stop
    entry_price = 0     # For pyramid add-on logic

    for i in range(max(WINDOW, RSI_PERIOD), len(df)):
        curr_price = df['Price'].iloc[i]
        curr_date = df.index[i]
        
        # Strategy Signals
        trend_signal = curr_price > df['Upper'].iloc[i]
        rsi_oversold = (curr_price < df['Lower'].iloc[i]) and (df['RSI'].iloc[i] < RSI_LOWER)
        
        # Position Management
        action = None
        
        # Update trailing stop baseline
        if position > 0:
            high_water_mark = max(high_water_mark, curr_price)

        # ENTRY: Initial Position (50%)
        if position == 0 and (trend_signal or rsi_oversold):
            qty = int((cash * INIT_POS_RATIO) / (curr_price * 100 * (1 + COMMISSION))) * 100
            if qty > 0:
                cost = qty * curr_price
                cash -= cost * (1 + COMMISSION)
                position = qty
                entry_price = curr_price
                high_water_mark = curr_price
                
                action = "BUY_INIT"
                trade_log.append((curr_date, "Entry", curr_price))

        # ENTRY: Pyramid Add-on (Remaining Cash)
        # Trigger: Floating profit > 3%
        elif position > 0 and cash > 5000 and curr_price > entry_price * (1 + ADD_ON_THRESHOLD):
            qty = int(cash / (curr_price * 100 * (1 + COMMISSION))) * 100
            if qty > 0:
                cost = qty * curr_price
                cash -= cost * (1 + COMMISSION)
                position += qty
                # Note: Don't update entry_price to maintain threshold logic
                
                action = "BUY_ADD"
                trade_log.append((curr_date, "Add-on", curr_price))

        # EXIT: Stop Loss / Trend Reversal / RSI Overbought
        elif position > 0:
            hit_stop = curr_price < high_water_mark * (1 - STOP_LOSS_PCT)
            trend_broken = curr_price < df['MA'].iloc[i]
            rsi_overbought = df['RSI'].iloc[i] > 80
            
            if hit_stop or trend_broken or rsi_overbought:
                revenue = position * curr_price
                cash += revenue * (1 - COMMISSION)
                position = 0
                
                reason = "Stop" if hit_stop else ("Trend" if trend_broken else "RSI")
                action = "SELL"
                trade_log.append((curr_date, f"Exit({reason})", curr_price))

        # Record Equity
        total_val = cash + position * curr_price
        equity.append(total_val)
        df.loc[curr_date, 'Signal'] = action

    # 4. Analysis & Plotting
    final_ret = (equity[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL
    equity_series = pd.Series(equity, index=df.index[max(WINDOW, RSI_PERIOD):])
    dd = (equity_series - equity_series.cummax()) / equity_series.cummax()
    max_dd = dd.min()

    print(f"Total Return: {final_ret:.2%}")
    print(f"Max Drawdown: {max_dd:.2%}")

    plot_results(df, equity_series, equity, max_dd, final_ret)

def plot_results(df, equity_series, equity, max_dd, final_ret):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Subplot 1: Price & Signals
    ax1.plot(df.index, df['Price'], color='#333', lw=1.5, label='Price')
    ax1.plot(df.index, df['Upper'], color='red', ls='--', alpha=0.3, label='Upper Band')
    ax1.plot(df.index, df['Lower'], color='green', ls='--', alpha=0.3, label='Lower Band')
    ax1.fill_between(df.index, df['Upper'], df['Lower'], color='gray', alpha=0.1)
    
    # Signals
    buy_init = df[df['Signal'] == 'BUY_INIT']
    buy_add = df[df['Signal'] == 'BUY_ADD']
    sell = df[df['Signal'] == 'SELL']
    
    ax1.scatter(buy_init.index, buy_init['Price'], marker='^', c='#ff9f43', s=100, label='Entry', zorder=5)
    ax1.scatter(buy_add.index, buy_add['Price'], marker='^', c='#ee5253', s=120, label='Add-on', zorder=5)
    ax1.scatter(sell.index, sell['Price'], marker='v', c='#10ac84', s=100, label='Exit', zorder=5)

    # Event Annotations
    for date_str, desc in EVENTS.items():
        try:
            dt = pd.to_datetime(date_str)
            idx = df.index.searchsorted(dt)
            if idx < len(df):
                d_point = df.index[idx]
                p_point = df['Price'].iloc[idx]
                ax1.annotate(desc, xy=(d_point, p_point), xytext=(d_point, p_point+5),
                             arrowprops=dict(facecolor='#0984e3', shrink=0.05, alpha=0.8),
                             bbox=dict(boxstyle="round", fc="#0984e3", ec="none", alpha=0.8),
                             color='white', ha='center', fontsize=9)
        except: pass

    ax1.set_title(f"XPENG (9868.HK) Strategy | Return: {final_ret:.2%}", fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Equity Curve
    ax2.plot(equity_series.index, equity, color='#e84118', lw=2, label='Equity')
    ax2.axhline(INITIAL_CAPITAL, color='gray', ls='--')
    
    # Drawdown Area
    roll_max = equity_series.cummax()
    ax2.fill_between(equity_series.index, equity, roll_max, color='#00b894', alpha=0.15, label='Drawdown')
    
    ax2.set_title(f"Equity Curve | Max Drawdown: {max_dd:.2%}", fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("xp_strategy_final.png")
    print("Chart saved.")
    plt.show()

if __name__ == "__main__":
    run_backtest()
