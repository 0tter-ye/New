import time
import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
from pybit.unified_trading import HTTP
import logging
import MetaTrader5 as mt5

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info(f"MetaTrader5 version: {mt5.__version__}")
logging.info(f"Has account_balance: {hasattr(mt5, 'account_balance')}")
logging.info(f"Has account_info: {hasattr(mt5, 'account_info')}")

from config import API_KEY, API_SECRET

client = HTTP(api_key=API_KEY, api_secret=API_SECRET, testnet=True)
symbol = "BTCUSDT"

if not mt5.initialize():
    logging.error(f"Failed to initialize MT5: {mt5.last_error()}")
    exit()

mt5_login = 3263156
mt5_password = "Bl1SrRhFb0JP@E4"
mt5_server = "Bybit-Demo"
if not mt5.login(mt5_login, mt5_password, mt5_server):
    logging.error(f"Failed to login to MT5: {mt5.last_error()}")
    exit()
logging.info("Connected to MT5 demo account")

account_info = mt5.account_info()
if account_info is None:
    logging.error("Failed to fetch MT5 account info")
    exit()
initial_balance = account_info.balance
logging.info(f"MT5 Account Balance: {initial_balance} USDT")

symbol_info = mt5.symbol_info(symbol)
if symbol_info is None:
    logging.warning(f"{symbol} not found. Searching for BTCUSDT variant...")
    all_symbols = [s.name for s in mt5.symbols_get() if "BTCUSDT" in s.name]
    if not all_symbols:
        logging.error("No BTCUSDT variant found in MT5. Check Market Watch.")
        exit()
    symbol = all_symbols[0]
    logging.info(f"Using symbol: {symbol}")
    symbol_info = mt5.symbol_info(symbol)

raw_point = symbol_info.point
min_volume = symbol_info.volume_min
max_volume = symbol_info.volume_max
volume_step = symbol_info.volume_step
tick_size = raw_point
min_stop_level = symbol_info.trade_stops_level

class Wallet:
    def __init__(self, max_risk_per_trade=0.2, max_drawdown=0.9, max_consecutive_losses=15):
        account_info = mt5.account_info()
        self.balance = account_info.balance if account_info else 0
        self.initial_balance = self.balance
        self.positions = {}
        self.trade_history = []
        self.max_risk_per_trade = max_risk_per_trade
        self.max_drawdown = max_drawdown
        self.max_consecutive_losses = max_consecutive_losses
        self.consecutive_losses = 0
        self.paused = False

    def sync_balance(self):
        account_info = mt5.account_info()
        if account_info is not None:
            self.balance = account_info.balance
            drawdown = max(0, (self.initial_balance - self.balance) / self.initial_balance)
            if drawdown >= self.max_drawdown:
                self.paused = True
                logging.warning(f"Trading paused: Drawdown {drawdown*100:.2f}% exceeds {self.max_drawdown*100}%")
            elif self.consecutive_losses >= self.max_consecutive_losses:
                self.paused = True
                logging.warning(f"Trading paused: {self.consecutive_losses} consecutive losses exceed {self.max_consecutive_losses}")
            else:
                self.paused = False
        else:
            logging.warning("Failed to sync balance with MT5")

    def sync_with_mt5(self, symbol):
        positions = mt5.positions_get(symbol=symbol)
        self.positions.clear()
        for pos in positions:
            side = "Buy" if pos.type == mt5.POSITION_TYPE_BUY else "Sell"
            self.positions[pos.ticket] = {
                'qty': pos.volume, 'entry_price': pos.price_open, 'side': side,
                'stop_loss': pos.sl, 'take_profit': pos.tp, 'ticket': pos.ticket,
                'entry_time': datetime.now()
            }
            logging.info(f"Synced MT5 position: {side} {pos.volume} {symbol} @ {pos.price_open}")
        self.sync_balance()

    # Modified: Dynamic volume from range [0.01 - 0.10] based on volatility, confidence, and win rate
    def calculate_position_size(self, price, stop_loss_distance, atr=None, confidence=0.5):
        volume_range = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
        leverage = 15
        summary = self.get_performance_summary()
        win_rate = summary['win_rate'] / 100  # Convert to decimal
        
        # Normalize factors (0 to 1 scale)
        volatility_factor = min(1.0, atr / (price * 0.02)) if atr else 0.5  # ATR vs 2% of price
        confidence_factor = min(1.0, confidence)
        win_rate_factor = min(1.0, win_rate)
        
        # Weighted score to select volume index (0 to 9)
        score = (0.4 * volatility_factor + 0.3 * confidence_factor + 0.3 * win_rate_factor)
        volume_index = int(score * (len(volume_range) - 1))  # Map to 0-9
        base_qty = volume_range[volume_index]
        
        # Adjust with Kelly-inspired sizing
        avg_win = max(0.01, summary['avg_trade_return'] / 100)
        avg_loss = max(0.01, abs(summary['avg_trade_return'] / 100) if summary['avg_trade_return'] < 0 else 0.01)
        b = avg_win / avg_loss
        q = 1 - win_rate
        kelly_fraction = max(0.01, min(0.5, (win_rate * b - q) / b))
        risk_amount = self.balance * kelly_fraction * leverage
        max_qty = risk_amount / (stop_loss_distance * price)
        
        # Final qty: Constrain between base_qty and max_qty, within volume range
        qty = min(max(base_qty, min_volume), max_qty, max_volume)
        qty = adjust_volume(qty, min_volume, max_volume, volume_step)
        
        logging.info(f"Volume Factors: Volatility={volatility_factor:.2f}, Confidence={confidence_factor:.2f}, "
                     f"Win Rate={win_rate_factor:.2f}, Score={score:.2f}, Base Qty={base_qty}, Final Qty={qty}")
        return qty

    def open_position(self, symbol, side, qty, price, stop_loss, take_profit):
        if self.paused:
            logging.warning("Trading paused due to drawdown or loss limit.")
            return False
        cost = qty * price
        self.sync_balance()
        if self.balance >= cost:
            self.balance -= cost
            ticket = random.randint(100000, 999999)  # Placeholder
            self.positions[ticket] = {
                'qty': qty, 'entry_price': price, 'side': side,
                'stop_loss': stop_loss, 'take_profit': take_profit, 'ticket': ticket,
                'entry_time': datetime.now()
            }
            logging.info(f"Opened {side} position: {qty} {symbol} @ {price}, SL: {stop_loss}, TP: {take_profit}")
            return True
        else:
            logging.warning(f"Insufficient funds: {cost} > {self.balance}")
            return False

    def close_position(self, ticket, price):
        if ticket in self.positions and self.positions[ticket]['qty'] > 0:
            pos = self.positions[ticket]
            qty = pos['qty']
            entry_price = pos['entry_price']
            side = pos['side']
            exit_value = qty * price
            if side == "Buy":
                profit = exit_value - (qty * entry_price)
                self.balance += exit_value
            else:
                profit = (qty * entry_price) - exit_value
                self.balance += profit
            trade = {
                'symbol': symbol, 'side': side, 'qty': qty, 'entry_price': entry_price,
                'exit_price': price, 'profit': profit, 'timestamp': datetime.now()
            }
            self.trade_history.append(trade)
            if profit <= 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
            logging.info(f"Closed {side} position: {qty} {symbol} @ {price}, Profit: {profit}, New balance: {self.balance}")
            del self.positions[ticket]
        self.sync_with_mt5(symbol)

    def get_performance_summary(self, trades=None):
        if trades is None:
            trades = self.trade_history
        self.sync_balance()
        total_profit = sum(trade['profit'] for trade in trades)
        total_trades = len(trades)
        final_value = self.initial_balance + total_profit
        total_return = (final_value - self.initial_balance) / self.initial_balance if self.initial_balance > 0 else 0
        profit_factor = float('inf') if sum(1 for t in trades if t['profit'] <= 0) == 0 else sum(t['profit'] for t in trades if t['profit'] > 0) / abs(sum(t['profit'] for t in trades if t['profit'] <= 0))
        avg_trade_return = np.mean([t['profit'] / (t['qty'] * t['entry_price']) * 100 for t in trades]) if total_trades > 0 else 0
        win_rate = len([t for t in trades if t['profit'] > 0]) / total_trades * 100 if total_trades > 0 else 0
        return {
            'start_value': self.initial_balance,
            'final_value': final_value,
            'total_return': total_return * 100 if total_return != float('inf') else float('inf'),
            'profit_factor': profit_factor,
            'avg_trade_return': avg_trade_return,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'current_balance': self.balance
        }

    def get_periodic_performance(self):
        now = datetime.now()
        periods = {
            'Hourly': timedelta(hours=1),
            '4-Hourly': timedelta(hours=4),
            '12-Hourly': timedelta(hours=12),
            'Daily': timedelta(days=1),
            'Weekly': timedelta(weeks=1),
            'Monthly': timedelta(days=30)
        }
        performance = {}
        for period_name, delta in periods.items():
            start_time = now - delta
            period_trades = [trade for trade in self.trade_history if trade['timestamp'] >= start_time]
            if period_trades:
                summary = self.get_performance_summary(period_trades)
                performance[period_name] = summary
            else:
                performance[period_name] = {
                    'start_value': self.initial_balance,
                    'final_value': self.initial_balance,
                    'total_return': 0.0,
                    'profit_factor': 0.0,
                    'avg_trade_return': 0.0,
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'current_balance': self.balance
                }
        return performance

wallet = Wallet(max_risk_per_trade=0.2, max_drawdown=0.9, max_consecutive_losses=15)

class MarkovChain:
    def __init__(self, states, transition_matrix):
        self.states = states
        self.transition_matrix = transition_matrix
        self.current_state = random.choice(states)
        self.stationary_dist = self.compute_stationary_distribution()

    def compute_stationary_distribution(self):
        P = np.array(self.transition_matrix)
        eigenvalues, eigenvectors = np.linalg.eig(P.T)
        stationary = eigenvectors[:, np.isclose(eigenvalues, 1, atol=1e-8)].real
        stationary = stationary / stationary.sum()
        return stationary

    def next_state(self):
        try:
            probabilities = self.transition_matrix[self.states.index(self.current_state)]
            probabilities = np.clip(probabilities, 0, None)
            prob_sum = np.sum(probabilities)
            if prob_sum == 0:
                logging.warning("Transition probabilities sum to zero, using uniform distribution.")
                probabilities = np.ones_like(probabilities) / len(probabilities)
            else:
                probabilities = probabilities / prob_sum
            self.current_state = np.random.choice(self.states, p=probabilities)
            return self.current_state, self.stationary_dist
        except Exception as e:
            logging.error(f"Error in Markov Chain transition: {e}")
            self.current_state = random.choice(self.states)
            return self.current_state, [0.5, 0.5]

    def entropy(self):
        dist = self.stationary_dist
        return -np.sum([p * np.log2(p + 1e-10) for p in dist if p > 0])

    def update_transition_matrix(self, trade_history):
        if len(trade_history) > 10:
            states_seq = ["Win" if t["profit"] > 0 else "Loss" for t in trade_history]
            transitions = np.zeros((2, 2))
            for i in range(len(states_seq) - 1):
                from_idx = 0 if states_seq[i] == "Loss" else 1
                to_idx = 0 if states_seq[i + 1] == "Loss" else 1
                transitions[from_idx, to_idx] += 1
            row_sums = transitions.sum(axis=1, keepdims=True)
            self.transition_matrix = np.where(row_sums == 0, 0.5, transitions / row_sums)
            self.stationary_dist = self.compute_stationary_distribution()
            logging.info(f"Updated Markov transition matrix: {self.transition_matrix}")

states = ["Loss", "Win"]
transition_matrix = np.array([[0.6, 0.4], [0.3, 0.7]])
markov_chain = MarkovChain(states, transition_matrix)

def compute_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def fetch_data(symbol, timeframe, limit=200, retries=3, backoff=5):
    for attempt in range(retries):
        try:
            response = client.get_kline(category="linear", symbol=symbol, interval=timeframe, limit=limit)
            logging.info(f"API response for {timeframe}m: {response.get('retMsg', 'No message')}")
            if "result" not in response or "list" not in response["result"]:
                logging.error(f"Invalid API response structure for {timeframe} timeframe: {response}")
                raise ValueError("Invalid response structure")
            data = response["result"]["list"]
            if not data:
                logging.warning(f"No data returned for {timeframe}m, attempt {attempt + 1}/{retries}")
                if attempt < retries - 1:
                    time.sleep(backoff * (2 ** attempt))
                continue
            
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(np.int64), unit="ms")
            df[["open", "high", "low", "close", "volume", "turnover"]] = df[["open", "high", "low", "close", "volume", "turnover"]].astype(float)
            df["returns"] = df["close"].pct_change()
            df["high_low"] = df["high"] - df["low"]
            df["high_close_prev"] = abs(df["high"] - df["close"].shift(1))
            df["low_close_prev"] = abs(df["low"] - df["close"].shift(1))
            df["tr"] = df[["high_low", "high_close_prev", "low_close_prev"]].max(axis=1)
            df["atr"] = df["tr"].rolling(window=14).mean()
            df["rsi"] = compute_rsi(df["close"], 14)
            df["ma20"] = df["close"].rolling(window=20).mean()
            df["ma50"] = df["close"].rolling(window=50).mean()
            df["momentum"] = df["close"].diff(4)
            df = df.drop(columns=["high_low", "high_close_prev", "low_close_prev", "tr"])
            df = df.dropna()
            logging.info(f"Data shape for {timeframe}m after processing: {df.shape}")
            logging.info(f"Last 5 close_{timeframe}m: {df['close'].tail(5).tolist()}")
            df.columns = [f"{col}_{timeframe}m" if col not in ["timestamp"] else col for col in df.columns]
            df = df.sort_values("timestamp")
            return df
        except Exception as e:
            logging.error(f"Error fetching {timeframe}min data, attempt {attempt + 1}/{retries}: {e}")
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))
            else:
                logging.error(f"All retries failed for {timeframe}m")
                return pd.DataFrame()
    return pd.DataFrame()

def fetch_combined_data(symbol, timeframes=["1"], limit=200):
    dfs = {}
    for tf in timeframes:
        df = fetch_data(symbol, tf, limit)
        if not df.empty:
            dfs[tf] = df
            logging.info(f"Loaded {tf}m data with shape: {df.shape}")
        else:
            logging.warning(f"Failed to fetch {tf}min data")
    
    if not dfs:
        logging.error("No data fetched for any timeframe.")
        return pd.DataFrame()
    
    base_tf = min(timeframes, key=int)
    combined_df = dfs[base_tf]
    logging.info(f"Base timeframe {base_tf}m shape before merging: {combined_df.shape}")
    if not combined_df["timestamp"].is_monotonic_increasing:
        logging.warning(f"Base timeframe {base_tf}m timestamps not sorted. Sorting now.")
        combined_df = combined_df.sort_values("timestamp")
    
    for tf in timeframes:
        if tf != base_tf and tf in dfs:
            if not dfs[tf]["timestamp"].is_monotonic_increasing:
                logging.warning(f"{tf}m timestamps not sorted. Sorting now.")
                dfs[tf] = dfs[tf].sort_values("timestamp")
            try:
                pre_merge_shape = combined_df.shape
                combined_df = pd.merge_asof(
                    combined_df,
                    dfs[tf],
                    on="timestamp",
                    direction="nearest",
                    tolerance=pd.Timedelta(minutes=int(tf) * 2),
                    suffixes=("", f"_{tf}m")
                )
                logging.info(f"After merging {tf}m: shape changed from {pre_merge_shape} to {combined_df.shape}")
            except ValueError as e:
                logging.error(f"Failed to merge {tf}m data: {e}")
                return pd.DataFrame()
    
    combined_df = combined_df.dropna()
    if combined_df["timestamp"].duplicated().any():
        logging.warning("Duplicate timestamps detected, dropping duplicates.")
        combined_df = combined_df.drop_duplicates("timestamp", keep="last")
    logging.info(f"Combined data shape after merge and dropna: {combined_df.shape}, Timeframes: {timeframes}")
    return combined_df

def fetch_order_book(symbol, df):
    try:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logging.warning(f"Failed to fetch tick data for {symbol}. Using last close.")
            last_close = df["close_1m"].iloc[-1] if not df.empty else 100.0
            return [last_close - 0.1], [10], [last_close + 0.1], [10]
        bid_price = tick.bid
        ask_price = tick.ask
        current_price = (bid_price + ask_price) / 2
        last_close = df["close_1m"].iloc[-1] if not df.empty else None
        if last_close and abs(current_price - last_close) / last_close > 0.5:
            logging.warning(f"Tick price {current_price} deviates significantly from close {last_close}. Using close price.")
            current_price = last_close
        return [bid_price], [10], [ask_price], [10]
    except Exception as e:
        logging.error(f"Error fetching order book: {e}")
        return [100.0], [10], [100.1], [10]

def calculate_bid_ask_imbalance(bid_volumes, ask_volumes):
    total_bid_volume = sum(bid_volumes)
    total_ask_volume = sum(ask_volumes)
    return (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume) if (total_bid_volume + total_ask_volume) > 0 else 0

def hurst_exponent(time_series):
    lags = range(2, min(20, len(time_series) // 2))
    if len(lags) < 2:
        return 0.5
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]

def adjust_volume(volume, min_vol, max_vol, step):
    volume = max(min_vol, min(max_vol, volume))
    volume = round(volume / step) * step
    if volume == max_vol:
        logging.warning(f"Volume capped at max: {max_vol}")
    return round(volume, 6)

def train_models(df):
    if df.empty or not any(col.endswith("_1m") for col in df.columns):
        logging.error("DataFrame is empty or missing required 1m columns")
        return None, None, None
    
    feature_cols = [col for col in df.columns if col not in ["timestamp"] and not col.startswith("turnover")]
    X = df[feature_cols].iloc[:-1]
    y = df["close_1m"].pct_change().shift(-1).iloc[:-1]  # Predict next price change
    y = y.fillna(0)  # Handle NaNs
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_mse = mean_squared_error(y_test, rf_pred)
    
    dt_model = DecisionTreeRegressor(max_depth=5, min_samples_split=5, random_state=42)
    dt_model.fit(X_train_scaled, y_train)
    dt_pred = dt_model.predict(X_test_scaled)
    dt_mse = mean_squared_error(y_test, dt_pred)

    logging.info(f"RF Model MSE: {rf_mse:.6f}")
    logging.info(f"DT Model MSE: {dt_mse:.6f}")
    return rf_model, dt_model, scaler

df_initial = fetch_combined_data(symbol, timeframes=["1"], limit=500)
if df_initial.empty:
    logging.error("Initial data fetch failed. Exiting.")
    exit()
rf_model, dt_model, scaler = train_models(df_initial)
if rf_model is None or dt_model is None or scaler is None:
    logging.error("Model training failed. Exiting.")
    exit()

logging.info("Models trained successfully, entering main loop...")

def backtest_strategy(df, initial_balance=50000):
    wallet = Wallet(max_risk_per_trade=0.2, max_drawdown=0.9, max_consecutive_losses=15)
    wallet.balance = initial_balance
    feature_cols = [col for col in df.columns if col not in ["timestamp"] and not col.startswith("turnover")]
    for i in range(len(df) - 1):
        X_latest = pd.DataFrame(df[feature_cols].iloc[i]).T
        X_latest_scaled = scaler.transform(X_latest)
        rf_pred = rf_model.predict(X_latest_scaled)[0]
        dt_pred = dt_model.predict(X_latest_scaled)[0]
        avg_pred = (rf_pred + dt_pred) / 2
        prediction = 1 if avg_pred > 0.001 else 0  # Threshold for 0.1% move
        if prediction in [0, 1]:
            execute_trade(prediction, symbol, df.iloc[:i+1], confidence_threshold=0.1)
    summary = wallet.get_performance_summary()
    logging.info(f"Backtest Result: Final Balance={summary['final_value']:.2f}, Return={summary['total_return']:.2f}%")
    return summary

# Modified: Reduced trade frequency to 30 seconds
def main_loop():
    iteration = 0
    start_time = datetime.now()
    last_performance_log = datetime.now()
    last_retrain_time = datetime.now()
    performance_interval = timedelta(minutes=60)
    retrain_interval = timedelta(minutes=10)
    cooldown = 0
    last_close = None

    global rf_model, dt_model, scaler

    logging.info("Initial delay of 10 seconds to avoid API rate limits...")
    time.sleep(10)

    while True:
        try:
            if cooldown > 0:
                logging.info(f"Cooldown: {cooldown} seconds remaining")
                time.sleep(min(cooldown, 1))
                cooldown -= min(cooldown, 1)
                continue

            df = fetch_combined_data(symbol, timeframes=["1"], limit=500)
            if df.empty:
                logging.warning("No data fetched, skipping iteration.")
                time.sleep(30)  # Changed to 30 seconds
                continue

            current_time = datetime.now()
            current_close = df["close_1m"].iloc[-1]
            if last_close and abs(current_close - last_close) / last_close > 0.02:
                rf_model, dt_model, scaler = train_models(df)
                last_retrain_time = current_time
                logging.info("Models retrained due to significant price move")
            elif current_time - last_retrain_time >= retrain_interval:
                rf_model, dt_model, scaler = train_models(df)
                last_retrain_time = current_time
                logging.info("Models retrained.")
            last_close = current_close

            wallet.sync_with_mt5(symbol)
            markov_chain.update_transition_matrix(wallet.trade_history)
            current_state, stationary_dist = markov_chain.next_state()
            entropy = markov_chain.entropy()
            markov_win_prob = stationary_dist[1].item() * 100
            logging.info(f"Current State: {current_state}, Entropy: {entropy:.2f} bits, Markov Win Prob: {markov_win_prob:.2f}%")

            current_price = df["close_1m"].iloc[-1]
            for ticket, pos in list(wallet.positions.items()):
                time_open = (current_time - pos['entry_time']).total_seconds() / 60
                price_change = abs(current_price - pos['entry_price']) / pos['entry_price']
                if time_open > 5 and price_change < 0.005:
                    wallet.close_position(ticket, current_price)
                    logging.info(f"Closed {pos['side']} position due to timeout: Ticket={ticket}")

            feature_cols = [col for col in df.columns if col not in ["timestamp"] and not col.startswith("turnover")]
            X_latest = pd.DataFrame(df[feature_cols].iloc[-1]).T
            X_latest_scaled = scaler.transform(X_latest)
            rf_pred = rf_model.predict(X_latest_scaled)[0]
            dt_pred = dt_model.predict(X_latest_scaled)[0]
            avg_pred = (rf_pred + dt_pred) / 2
            prediction = 1 if avg_pred > 0.001 else 0
            logging.info(f"Prediction: {prediction} (RF: {rf_pred:.6f}, DT: {dt_pred:.6f}, Avg: {avg_pred:.6f})")
            
            if prediction in [0, 1]:
                execute_trade(prediction, symbol, df, confidence_threshold=0.1, markov_win_prob=markov_win_prob)
                cooldown = 30  # Changed to 30 seconds

            iteration += 1
            logging.info(f"Iteration {iteration} completed, preparing performance summary...")
            summary = wallet.get_performance_summary()
            logging.info(f"Temporary Summary: Balance: {summary['current_balance']:.2f}, Trades: {summary['total_trades']}, Profit Factor: {summary['profit_factor']:.2f}")

            if iteration % 10 == 0:
                logging.info(f"Overall Performance Summary (Since {start_time.strftime('%Y-%m-%d %H:%M:%S')}): "
                             f"Start Value: {summary['start_value']:.2f} USDT, Final Value: {summary['final_value']:.2f} USDT, "
                             f"Total Return: {summary['total_return']:.2f}%, Profit Factor: {summary['profit_factor']:.2f}, "
                             f"Avg Trade Return: {summary['avg_trade_return']:.2f}%, Trades: {summary['total_trades']}, "
                             f"Win Rate: {summary['win_rate']:.2f}%, Balance: {summary['current_balance']:.2f} USDT")

                if current_time - last_performance_log >= performance_interval:
                    periodic_performance = wallet.get_periodic_performance()
                    for period, metrics in periodic_performance.items():
                        logging.info(f"{period} Performance: "
                                     f"Start Value: {metrics['start_value']:.2f} USDT, Final Value: {metrics['final_value']:.2f} USDT, "
                                     f"Total Return: {metrics['total_return']:.2f}%, Profit Factor: {metrics['profit_factor']:.2f}, "
                                     f"Avg Trade Return: {metrics['avg_trade_return']:.2f}%, Trades: {metrics['total_trades']}, "
                                     f"Win Rate: {metrics['win_rate']:.2f}%, Balance: {metrics['current_balance']:.2f} USDT")
                    last_performance_log = current_time

                if summary["win_rate"] < 30 and summary["total_trades"] > 10:
                    wallet.max_risk_per_trade = max(0.05, wallet.max_risk_per_trade * 0.8)
                    logging.info(f"Reduced risk to {wallet.max_risk_per_trade*100}% due to low win rate.")
                elif summary["win_rate"] > 70:
                    wallet.max_risk_per_trade = min(0.5, wallet.max_risk_per_trade * 1.2)
                    logging.info(f"Increased risk to {wallet.max_risk_per_trade*100}% due to high win rate.")

            logging.info("Sleeping for 30 seconds before next iteration...")
            time.sleep(30)  # Changed to 30 seconds
        except Exception as e:
            logging.error(f"Unexpected error in main loop: {e}")
            time.sleep(30)  # Changed to 30 seconds

# Modified: Pass ATR and confidence to calculate_position_size
def execute_trade(prediction, symbol, df, confidence_threshold=0.1, markov_win_prob=0.0):
    side = "Buy" if prediction == 1 else "Sell"
    try:
        bid_prices, bid_volumes, ask_prices, ask_volumes = fetch_order_book(symbol, df)
        current_price = (min(ask_prices) + max(bid_prices)) / 2
        if current_price == 100.05:
            current_price = df["close_1m"].iloc[-1]
            logging.warning(f"Using fallback current price: {current_price}")

        feature_cols = [col for col in df.columns if col not in ["timestamp"] and not col.startswith("turnover")]
        X_latest = pd.DataFrame(df[feature_cols].iloc[-1]).T
        X_latest_scaled = scaler.transform(X_latest)
        rf_pred = rf_model.predict(X_latest_scaled)[0]
        dt_pred = dt_model.predict(X_latest_scaled)[0]
        avg_pred = (rf_pred + dt_pred) / 2
        confidence = abs(avg_pred) / 0.01
        
        summary = wallet.get_performance_summary()
        dynamic_threshold = max(confidence_threshold, 0.5 - summary['win_rate'] / 200)
        if confidence < dynamic_threshold:
            logging.info(f"Confidence too low: {confidence:.2f}, Threshold={dynamic_threshold:.2f}")
            return

        atr_1m = df["atr_1m"].iloc[-1]
        volatility = atr_1m / df["close_1m"].iloc[-1]
        if volatility < 0.01:
            logging.info(f"Volatility too low: {volatility*100:.2f}%")
            return

        close_1m = df["close_1m"].iloc[-1]
        ma20_1m = df["ma20_1m"].iloc[-1]
        ma50_1m = df["ma50_1m"].iloc[-1]
        rsi = df["rsi_1m"].iloc[-1]
        momentum_1m = df["momentum_1m"].iloc[-1]

        if side == "Buy" and (close_1m <= ma20_1m or ma20_1m <= ma50_1m or rsi > 65 or momentum_1m < 0):
            logging.info(f"Buy rejected: Close={close_1m}, MA20={ma20_1m}, MA50={ma50_1m}, RSI={rsi}, Momentum={momentum_1m}")
            return
        elif side == "Sell" and (close_1m >= ma20_1m or ma20_1m >= ma50_1m or rsi < 35 or momentum_1m > 0):
            logging.info(f"Sell rejected: Close={close_1m}, MA20={ma20_1m}, MA50={ma50_1m}, RSI={rsi}, Momentum={momentum_1m}")
            return

        confidence_factor = min(confidence, 1.0)
        stop_loss_distance = atr_1m * (0.5 + (1 - confidence_factor))
        take_profit_distance = atr_1m * (5 + confidence_factor * 5)
        stop_loss = round(current_price - stop_loss_distance if side == "Buy" else current_price + stop_loss_distance, 6)
        take_profit = round(current_price + take_profit_distance if side == "Buy" else current_price - take_profit_distance, 6)

        sl_distance = abs(current_price - stop_loss)
        min_distance = min_stop_level * tick_size
        if sl_distance < min_distance:
            stop_loss = current_price - min_distance if side == "Buy" else current_price + min_distance
            logging.warning(f"Adjusted SL to minimum distance: {stop_loss}")

        # Pass ATR and confidence to calculate_position_size
        adjusted_qty = wallet.calculate_position_size(current_price, stop_loss_distance, atr=atr_1m, confidence=confidence)
        if adjusted_qty < min_volume:
            logging.warning(f"Trade qty {adjusted_qty} below minimum {min_volume}. Skipping.")
            return

        if side == "Sell":
            for ticket, pos in list(wallet.positions.items()):
                if pos['side'] == "Buy":
                    wallet.close_position(ticket, current_price)
        elif side == "Buy":
            for ticket, pos in list(wallet.positions.items()):
                if pos['side'] == "Sell":
                    wallet.close_position(ticket, current_price)

        order_type = mt5.ORDER_TYPE_BUY if side == "Buy" else mt5.ORDER_TYPE_SELL
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": adjusted_qty,
            "type": order_type,
            "price": current_price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": 20,
            "magic": 123456,
            "comment": f"{side} via Bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        logging.info(f"Attempting {side} trade: Price={current_price}, Qty={adjusted_qty}, SL={stop_loss}, TP={take_profit}")
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Order failed: {result.comment} (retcode: {result.retcode})")
            return

        wallet.sync_with_mt5(symbol)
        if side == "Buy":
            wallet.open_position(symbol, side, adjusted_qty, current_price, stop_loss, take_profit)
        elif side == "Sell" and len(wallet.positions) > 0:
            for ticket in list(wallet.positions.keys()):
                wallet.close_position(ticket, current_price)

    except Exception as e:
        logging.error(f"Error executing trade: {e}")

if __name__ == "__main__":
    df_historical = fetch_combined_data(symbol, timeframes=["1"], limit=1000)
    backtest_strategy(df_historical)
    main_loop()
