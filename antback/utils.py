#!/usr/bin/python
# coding=utf-8
import datetime as dtm
import math
from collections import defaultdict, deque
from pprint import pprint

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

try:
    from .comnt import write_from_template
except ImportError:
    from comnt import write_from_template


def get_drawdown(prices):
    """Vectorized calculation of maximum drawdown for a numpy array of prices."""
    assert np.all(prices >= 0), "All prices must be non-negative"

    peaks = np.maximum.accumulate(prices)
    drawdowns = 100 * (peaks - prices) / peaks
    max_dd_idx = np.argmax(drawdowns)
    max_dd = drawdowns[max_dd_idx]
    peak_at_max_dd = peaks[max_dd_idx]
    start_idx = np.where(prices == peak_at_max_dd)[0][-1]

    return {"max_dd": max_dd * -1, "index": max_dd_idx, "start": start_idx}


def pct_diff(prev, today):
    if prev == 0 or math.isnan(prev):
        return float('nan')
    return (today - prev) / prev * 100


class RollingList:
    __slots__ = ('_buffer',)
    """Maintain a fixed-length list using deque."""

    def __init__(self, maxlen):
        self._buffer = deque(maxlen=maxlen)

    def append(self, value):
        """Append a value to the list."""
        self._buffer.append(value)

    def values(self):
        """Return list of current values."""
        return list(self._buffer)


class NamedRollingLists:
    __slots__ = ('_data', '_maxlen')
    """Store multiple RollingLists keyed by name."""

    def __init__(self, maxlen=10):
        self._data = {}
        self._maxlen = maxlen

    def append(self, key, value):
        """Append a value to the list for a given key."""
        if key not in self._data:
            self._data[key] = RollingList(self._maxlen)
        self._data[key].append(value)

    def get(self, key):
        """Return list of values for a given key."""
        return self._data.get(key, RollingList(self._maxlen)).values()

    def get_keys(self):
        """Return list of stored keys."""
        return list(self._data.keys())

    def __repr__(self):
        return str({k: v.values() for k, v in self._data.items()})


class PerTickerRollingStore:
    __slots__ = ('_store', '_array_size')
    """Store multiple named rolling lists per key."""

    def __init__(self, array_size=10):
        self._store = defaultdict(dict)
        self._array_size = array_size

    def append(self, key, series_name, value):
        """Append a value to the series for a key."""
        if series_name not in self._store[key]:
            self._store[key][series_name] = RollingList(self._array_size)
        self._store[key][series_name].append(value)

    def get(self, key, series_name):
        """Return list of values for a given key and series."""
        series = self._store.get(key, {})
        rolling_list = series.get(series_name)
        return rolling_list.values() if rolling_list else None

    def __repr__(self):
        return str({
            key: {
                name: arr.values() for name, arr in series.items()
            } for key, series in self._store.items()
        })


class RollingArray:
    __slots__ = ('_buffer',)
    """Maintain a fixed-length NumPy array buffer."""

    def __init__(self, size):
        self._buffer = np.zeros(size, dtype=float)

    def append(self, value):
        """Append a value, shifting older values left."""
        self._buffer[0:-1] = self._buffer[1:]
        self._buffer[-1] = value

    def values(self):
        """Return current array values."""
        return self._buffer


class NamedRollingArrays:
    __slots__ = ('_buffers', '_buffer_size')
    """Store multiple RollingArrays keyed by name."""

    def __init__(self, array_size=10):
        self._buffers = {}
        self._buffer_size = array_size

    def append(self, key, value):
        """Append a value to the array for a key."""
        if key not in self._buffers:
            self._buffers[key] = RollingArray(self._buffer_size)
        self._buffers[key].append(value)

    def get(self, key):
        """Return array values for a given key."""
        buffer = self._buffers.get(key)
        return buffer.values() if buffer else None

    def __repr__(self):
        return str({k: v.values() for k, v in self._buffers.items()})


class PerTickerNamedRollingArrays:
    __slots__ = ('_store', '_array_size')
    """Store multiple named rolling arrays per key."""

    def __init__(self, array_size=10):
        self._store = defaultdict(dict)
        self._array_size = array_size

    def append(self, key, series_name, value):
        """Append a value to the series for a key."""
        if series_name not in self._store[key]:
            self._store[key][series_name] = RollingArray(self._array_size)
        self._store[key][series_name].append(value)

    def get(self, key, series_name):
        """Return array values for a given key and series."""
        series = self._store.get(key, {})
        rolling_array = series.get(series_name)
        return rolling_array.values() if rolling_array else None

    def __repr__(self):
        return str({
            key: {
                name: arr.values() for name, arr in series.items()
            } for key, series in self._store.items()
        })


def pct_dist(prev, today):
    """Return absolute percentage difference between two values."""
    distance = math.sqrt((prev - today)**2)
    try:
        return abs((distance / max(prev, today)) * 100.0)
    except ZeroDivisionError:
        return 0


class Candle:
    """Represent a candlestick with OHLC values."""

    __slots__ = ("op", "hi", "lo", "cl", "half", "opcldist_prc", "is_white", "is_black", "is_equal",
                 "is_doji_std", "is_doji_grave", "is_doji_fly", "ibs")

    def __init__(self, op, hi, lo, cl):
        op, hi, lo, cl = float(op), float(hi), float(lo), float(cl)
        if not (hi >= lo and hi >= max(op, cl) and lo <= min(op, cl)):
            raise ValueError("Invalid candle values")

        self.op = op
        self.hi = hi
        self.lo = lo
        self.cl = cl
        self.half = (op + cl) / 2.0
        self.opcldist_prc = pct_dist(op, cl)

        self.is_white = op < cl
        self.is_black = op > cl
        self.is_equal = op == cl

        self.is_doji_std = False
        self.is_doji_grave = False
        self.is_doji_fly = False

        if self.is_equal:
            if hi > lo and hi > cl and lo < cl:
                self.is_doji_std = True
            elif hi > cl and lo == cl:
                self.is_doji_grave = True
            elif hi == cl and lo < cl:
                self.is_doji_fly = True

        try:
            self.ibs = (cl - lo) / (hi - lo) if (hi - lo) != 0 else 0
        except ZeroDivisionError:
            self.ibs = 0


def new_cross_func_np():
    """Returns a stateful function that detects series crosses."""
    prev_cmp = None

    def cross(active, passive):
        nonlocal prev_cmp
        if np.isnan(active) or np.isnan(passive) or passive == 0:
            prev_cmp = None
            return None

        diff = active - passive
        cur = np.sign(diff) if diff != 0 else None
        prev = prev_cmp
        prev_cmp = cur

        if prev is None or cur is None or cur == prev:
            return None
        return "up" if cur > prev else "down"

    return cross


def new_cross_func():
    """
    Returns a stateful function that detects series crosses.
    """
    # State representation:
    # -1: active series is below passive series
    #  1: active series is above passive series
    # None: series are equal, or the state is invalid/initial
    prev_state = None

    def cross(active, passive):
        nonlocal prev_state
        #  Reset state for invalid inputs. The check `val != val` is a
        #    fast, standard way to detect if a value is NaN (Not a Number).
        if (active != active) or (passive != passive) or passive == 0:
            prev_state = None
            return None

        current_state = 1 if active > passive else -1 if active < passive else None

        last_state = prev_state
        prev_state = current_state

        # A cross occurs only if the last state and current state are valid
        # (not None) and different from each other.
        if last_state is None or current_state is None or current_state == last_state:
            return None

        # If current_state > last_state, it must be 1 > -1, meaning a cross "up".
        # Otherwise, it must be -1 < 1, meaning a cross "down".
        return "up" if current_state > last_state else "down"

    return cross


def new_wait_n_bars(n):
    """Wait for n unique bars before returning True."""
    assert n >= 1, "n must be at least 1"

    def _inner(bar=None, start=False):
        nonlocal n, start_flag, store, prev_bar

        if start:
            if bar is not None:
                raise AssertionError("Start should be called without bar argument")
            if start_flag:
                raise ValueError(
                    "! WARNING: Starting new count while previous count is still active.")
            start_flag = True
            store.clear()
            prev_bar = None
            return 0

        if bar is None:
            raise AssertionError("Must provide either bar argument or start=True")

        assert isinstance(bar, (dtm.date, dtm.datetime)), ("bar must be a datetime object")

        if prev_bar is not None:
            assert bar > prev_bar, "New bar must be greater than previous bar"
        prev_bar = bar

        if start_flag and bar not in store:
            store[bar] = 1
            if len(store) >= n:
                store.clear()
                start_flag = False
                return True
            return False
        return None

    start_flag = False
    store = {}
    prev_bar = None
    return _inner


def new_multi_ticker_wait(n):
    """Create a multi-ticker wait function."""
    assert n >= 1, "n must be at least 1"

    def inner_waitfun(ticker, bar=None, start=False):
        nonlocal all_func

        if ticker not in all_func:
            all_func[ticker] = new_wait_n_bars(n)

        wait_res = all_func[ticker](bar=bar, start=start)
        return ticker, wait_res

    all_func = {}
    return inner_waitfun


def get_monthly_points(df, near_start_day=3, near_end_day=3):
    """Calculate key monthly points with configurable near_start/near_end days."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")

    result = {"start": [], "near_start": [], "mid": [], "near_end": [], "end": []}

    grouped = df.groupby([df.index.year, df.index.month])

    for (_, _), group in grouped:
        dates = group.index.sort_values()
        count = len(dates)

        if count == 0:
            continue

        result["start"].append(dates[0])
        result["end"].append(dates[-1])
        result["mid"].append(dates[count // 2])

        if count > near_start_day:
            result["near_start"].append(dates[near_start_day])

        if count > near_end_day:
            result["near_end"].append(dates[-near_end_day])

    return result


def new_atr_func(period=14):
    period = int(period)
    tr_values = []
    prev_close = None
    atr = None

    def _atr(high, low, close):
        nonlocal tr_values, prev_close, atr
        if prev_close is None:
            tr = high - low
        else:
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))

        prev_close = close

        if atr is None:
            tr_values.append(tr)
            if len(tr_values) < period:
                return None
            atr = sum(tr_values) / period
            return atr

        atr = (atr * (period - 1) + tr) / period
        return atr

    return _atr


def new_atr_stop_func(n=14, atr_multiplier=3):
    """Create an ATR-based stop function with persistent state."""
    assert atr_multiplier <= 20

    # Initialize state variables in the outer scope
    atr_function = new_atr_func(n)
    first_days = n
    max_stop_line = -(10**10)

    def atr_stop(high, low, close):
        nonlocal atr_function, first_days, max_stop_line
        atr_value = atr_function(high, low, close)
        if atr_value:
            stop_line = round(close - (atr_value * atr_multiplier), 3)
        else:
            stop_line = close
        max_stop_line = max(max_stop_line, stop_line)
        if first_days <= 0:
            if max_stop_line >= close:
                state = "below"
                max_stop_line = high
            else:
                state = "above"
        else:
            first_days -= 1
            state = "above"
        return state, max_stop_line

    return atr_stop


def create_multi_atr_func(atr_period=14, multiplier=3):
    """Create a function that manages ATR stops for multiple tickers."""
    all_functions = {}

    def compute_atr(
        ticker,
        high,
        low,
        close,
        atr_period=atr_period,
        multiplier=multiplier,
        reset=False,
    ):
        if reset:
            del all_functions[ticker]

        if ticker not in all_functions:
            all_functions[ticker] = new_atr_stop_func(atr_period, multiplier)

        stop_signal = all_functions[ticker](high, low, close)
        state, stop_line = stop_signal
        return state, stop_line

    return compute_atr


def create_sma_func_p(period):
    """
    Creates a universal SMA calculator that gracefully handles None,
    np.nan, pd.NA, and other missing value representations.
    """
    data_window = deque(maxlen=period)
    current_sum = 0.0
    last_sma = None

    def calculate_sma(new_value):
        nonlocal current_sum, last_sma

        if (new_value != new_value):  #  if math.isnan(new_value):
            return last_sma

        if len(data_window) == period:
            oldest_value = data_window[0]
            current_sum -= oldest_value

        data_window.append(new_value)
        current_sum += new_value

        if len(data_window) == period:
            current_sma = current_sum / period
            last_sma = current_sma
            return current_sma
        return None

    return calculate_sma


def create_sma_func(period):
    """
    Version using a counter to track data points instead of length checks.
    """
    data_window = deque(maxlen=period)
    current_sum = 0.0
    last_sma = None
    count = 0  # Track number of data points

    def calculate_sma(new_value):
        nonlocal current_sum, last_sma, count

        if new_value != new_value:  # NaN check
            return last_sma

        # Remove oldest value if window is full
        if count == period:
            oldest_value = data_window[0]
            current_sum -= oldest_value
        else:
            count += 1

        # Add new value
        data_window.append(new_value)
        current_sum += new_value

        if count == period:
            current_sma = current_sum / period
            last_sma = current_sma
            return current_sma

        return None

    return calculate_sma


def get_pre_dates_for_targets(df, target_dates, days_before=1):
    """Get dates that are days_before trading days before given target dates."""
    pre_dates = []
    idates = df.index.sort_values()
    for target_date in target_dates:
        if not isinstance(target_date, pd.Timestamp):
            target_date = pd.Timestamp(target_date)

        if df.index.tz is not None and target_date.tz is None:
            target_date = target_date.tz_localize(df.index.tz)
        elif df.index.tz is None and target_date.tz is not None:
            target_date = target_date.tz_localize(None)

        closest_idx = np.searchsorted(idates, target_date, side="left")
        target_idx = closest_idx - days_before

        if target_idx >= 0 and closest_idx > 0:
            pre_dates.append(idates[target_idx])

    return sorted(list(set(pre_dates)))


def datetime_to_str(dt, fmt="%Y-%m-%d %H:%M:%S"):
    """Convert datetime/date to string with smart formatting."""
    if isinstance(dt, np.datetime64):
        dt = pd.to_datetime(dt)

    if pd.isna(dt):
        return "NaT"

    if isinstance(dt, dtm.date) and not isinstance(dt, dtm.datetime):
        return dt.strftime("%Y-%m-%d")

    if isinstance(dt, dtm.datetime):
        if dt.hour == 0 and dt.minute == 0 and dt.second == 0:
            return dt.strftime("%Y-%m-%d")
        return dt.strftime(fmt)

    return repr(dt)


def get_orders(port):
    """Get orders from portfolio."""
    tradelist = port.get_tradelist()
    orders = pd.Series(name="signals")
    for ticker, values in port.positions.items():
        open_date, quantity, buy_price, last_date, last_price = values
        orders[open_date] = quantity
    for row in port.trades:
        orders[row.open_date] = row.quantity
        orders[row.close_date] = -row.quantity
    return orders.fillna(False)


def calculate_annual_growth_rate(portfolio):
    """
    Calculate the annualized growth rate percentage using compound annual growth rate (CAGR).
    Args:
        portfolio: An object with attributes:
                   - pos_history (dict[date, float]): historical positions by date
                   - starting_capital (float): initial investment
                   - max_date (date): end date of the period
                   current_value() -> float: method returning current portfolio value

    Returns:
        float: Annualized growth rate as a percentage (e.g., 5.0 for 5%).

    Raises:
        ValueError: If inputs are invalid or calculation fails due to logical issues.
    """
    if not hasattr(portfolio, 'pos_history') or not portfolio.pos_history:
        raise ValueError("Portfolio must have non-empty pos_history")

    if not hasattr(portfolio, 'starting_capital'):
        raise ValueError("Portfolio missing starting_capital")
    if not callable(getattr(portfolio, 'current_value', None)):
        raise ValueError("Portfolio must have current_value() method")

    start_date = min(portfolio.pos_history.keys())
    end_date = portfolio.max_date

    if not isinstance(start_date,
                      (dtm.date, dtm.datetime)) or not isinstance(end_date,
                                                                  (dtm.date, dtm.datetime)):
        raise ValueError("Dates in pos_history and max_date must be date or datetime objects")

    if start_date >= end_date:
        raise ValueError("Start date must be before end date")

    # Calculate total time in years using days for accuracy
    delta_days = (end_date - start_date).days
    if delta_days <= 0:
        raise ValueError("Time period must be greater than zero days")

    total_years = delta_days / 365.25  # Accounts for leap years

    start_value = float(portfolio.starting_capital)
    end_value = float(portfolio.current_value())

    if start_value < 0:
        raise ValueError("Starting value cannot be negative")
    if start_value == 0:
        raise ValueError("Starting value cannot be zero (division by zero)")

    if end_value < 0:
        raise ValueError("Ending value cannot be negative")

    if end_value == start_value:
        return 0.0

    # CAGR formula
    growth_factor = end_value / start_value
    annual_growth_rate = (growth_factor**(1 / total_years) - 1) * 100

    return annual_growth_rate


def summary(portfolio, show=False):
    """Generate portfolio summary statistics."""
    if not portfolio.pos_history or not portfolio.trades:
        raise AssertionError("!No position history available or no trades")

    portfolio.verify_consistency()

    start_date = min(portfolio.pos_history.keys())
    end_date = portfolio.max_date
    difference = relativedelta(end_date, start_date)
    months = difference.months + (difference.years * 12)
    # years = max(difference.years + (months / 12.0), 1 / 12.0)

    end_equity = portfolio.current_value()

    net_profit = end_equity - portfolio.starting_capital
    net_profit_prc = pct_diff(portfolio.starting_capital, end_equity)
    all_trades = portfolio.get_tradelist()
    trades_num = len(all_trades)

    if trades_num == 0:
        if show:
            print("No trades executed")
        return None

    profitable = all_trades[all_trades["net_profit"] > 0]
    prc_profitable = len(profitable) / float(trades_num) * 100

    dates, capital_line = portfolio.equity_line()

    drawdown = get_drawdown(np.array(capital_line))
    drawdown_val = drawdown["max_dd"]
    drawdown_start = dates[drawdown["start"]]

    summ_list = [
        [
            "Date Range:",
            f"{datetime_to_str(start_date)} to {datetime_to_str(end_date)}",
        ],
        ["Total Return (%):", f"{net_profit_prc:+.2f}%"],
        ["Ann. Ret. (%):", f"{calculate_annual_growth_rate(portfolio):.2f}%"],
        ["Max Drawdown:", f"{drawdown_val:.1f}%"],
        ["Winning Ratio (%):", f"{prc_profitable:.2f}%"],
        ["Max. Drawdown Start:", f"{datetime_to_str(drawdown_start)}"],
        ["Net Profit:", f"{net_profit:,.2f}"],
        ["Total fees Paid:", f"{portfolio.total_fees_paid:,.2f}"],
        ["Starting Capital:", f"{portfolio.starting_capital:,.2f}"],
        ["Ending Capital:", f"{end_equity:,.2f}"],
        ["Number of Trades:", f"{trades_num:,}"],
        ["Fees Rate (%):", f"{portfolio.fees * 100:.2f}%"],
    ]
    if hasattr(portfolio, 'leverage'):
        summ_list.append(["Leverage:", f"{portfolio.leverage}"])
    if hasattr(portfolio, 'single'):
        summ_list.append(["Single Mode Enabled:", f"{portfolio.single}"])

    if show:
        pprint(summ_list)

    return summ_list


def remove_outliers(series: pd.Series, threshold: float = 2.0):
    """Remove outliers using Median Absolute Deviation."""
    deviations = np.abs(series - np.median(series))
    med_dev = np.median(deviations)
    if med_dev == 0:
        return series
    scaled = deviations / med_dev
    return series[scaled < threshold]


def calculate_avg_duration(open_dates: pd.Series, close_dates: pd.Series):
    """Calculate and format the average trade duration."""
    durations = (close_dates - open_dates).dt.total_seconds()
    avg_seconds = durations.mean()

    if pd.isna(avg_seconds):
        return "N/A"

    minutes, seconds = divmod(avg_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)

    parts = []
    if days >= 1:
        parts.append(f"{int(days)}d")
    if hours >= 1:
        parts.append(f"{int(hours)}h")
    if minutes >= 1:
        parts.append(f"{int(minutes)}m")
    if seconds >= 1 and days < 1:
        parts.append(f"{int(seconds)}s")

    return " ".join(parts) if parts else "<1s"


def find_longest_streaks_vectorized(df: pd.DataFrame, col: str):
    """Find longest consecutive profit/loss streaks without manual loops."""
    sign = np.sign(df[col])  # 1 for profit, -1 for loss, 0 for breakeven
    sign_groups = (sign != sign.shift()).cumsum()

    streaks = df.groupby(sign_groups).agg(sign_val=(col, lambda x: np.sign(x.iloc[0])),
                                          length=(col, "size"))

    longest_profit = streaks.loc[streaks["sign_val"] > 0, "length"].max()
    longest_loss = streaks.loc[streaks["sign_val"] < 0, "length"].max()

    return [
        f"Longest profit streak: {int(longest_profit) if pd.notna(longest_profit) else 0}",
        f"Longest loss streak: {int(longest_loss) if pd.notna(longest_loss) else 0}"
    ]


def analyze_trades(trades: pd.DataFrame, show: bool = True):
    """Analyze trading statistics from a trades DataFrame."""
    if trades.empty:
        return pd.DataFrame([["No trades to analyze"]], columns=["Trading Statistics"])
    df = trades.copy()
    df["open_date"] = pd.to_datetime(df["open_date"])
    df["close_date"] = pd.to_datetime(df["close_date"])
    df["date"] = df["open_date"].dt.normalize()
    df.set_index("date", inplace=True)

    # Basic stats
    profits = df.loc[df["net_profit"] > 0, "net_profit"]
    losses = df.loc[df["net_profit"] < 0, "net_profit"]

    # Calculate profit factor
    gross_profit = profits.sum() if not profits.empty else 0
    gross_loss = abs(losses.sum()) if not losses.empty else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float(
        'inf') if gross_profit > 0 else 0

    results = [
        f"Avg profit: {profits.mean():.1f}, Avg loss: {losses.mean():.1f}",
        f"Avg trade duration: {calculate_avg_duration(df['open_date'], df['close_date'])}"
    ]

    # Streaks
    results.extend(find_longest_streaks_vectorized(df, "net_profit"))

    # Daily stats
    daily = df.groupby(df.index).agg(count=("net_profit", "size"), profit=("net_profit", "sum"))
    all_days = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    trading_days = daily.index.nunique()
    total_days = len(all_days)
    results.extend([
        f"Total days: {total_days}, Trading days: {trading_days} ({trading_days / total_days * 100:.0f}%)",
        f"Max trades/day: {daily['count'].max():.0f}"
    ])

    # Weekly stats
    full_daily = daily.reindex(all_days, fill_value=0)
    full_daily["week"] = full_daily.index.map(
        lambda x: f"{x.isocalendar()[0]}_{x.isocalendar()[1]}")
    weekly_counts = full_daily.groupby("week")["count"].sum()
    weekly_clean = remove_outliers(weekly_counts)
    results.extend([
        f"Max trades/week: {weekly_counts.max():.0f}", f"Avg trades/week: {weekly_clean.mean():.1f}"
    ])

    # Yearly stats
    full_daily["year"] = full_daily.index.year
    yearly_trades = full_daily.groupby("year")["count"].sum()
    yearly_profit = full_daily.groupby("year")["profit"].sum()
    results.extend([
        f"Avg trades/year: {yearly_trades.mean():.1f}",
        f"Avg profit/year: {yearly_profit.mean():.1f}",
        f"Profit Factor: {profit_factor:.2f}",
    ])

    # Group stats by position type
    if 'position_type' in df.columns:
        total = len(df)
        long_count = (df['position_type'] == 'long').sum()
        short_count = (df['position_type'] == 'short').sum()
        long_pct = (long_count / total) * 100 if total else 0
        short_pct = (short_count / total) * 100 if total else 0
        results.append(f"Long trades: {long_count} ({long_pct:.1f}%)")
        results.append(f"Short trades: {short_count} ({short_pct:.1f}%)")

        grouped = df.groupby('position_type')
        for pos_type, group in grouped:
            # Calculate profit factor for each position type
            pos_profits = group.loc[group['net_profit'] > 0, 'net_profit']
            pos_losses = group.loc[group['net_profit'] < 0, 'net_profit']
            pos_gross_profit = pos_profits.sum() if not pos_profits.empty else 0
            pos_gross_loss = abs(pos_losses.sum()) if not pos_losses.empty else 0
            pos_profit_factor = pos_gross_profit / pos_gross_loss if pos_gross_loss > 0 else float(
                'inf') if pos_gross_profit > 0 else 0

            avg_win = pos_profits.mean()
            avg_loss = pos_losses.mean()
            results.append(
                f"{pos_type.capitalize()}: PF: {pos_profit_factor:.2f}, Avg win: {avg_win:.2f}, Avg loss: {avg_loss:.2f}"
            )

    if show:
        pprint(results)
    return pd.DataFrame(results, columns=["Trading Statistics"])


def get_year_stats(portfolio):
    """Generate yearly statistics."""
    equity_data = portfolio.get_equity()
    cleaned_equity_data = equity_data.drop(["date_obj"], axis=1)

    trade_data = portfolio.get_tradelist()
    trade_data["year"] = trade_data.close_date.apply(lambda x: str(x)[:4]
                                                     if pd.notna(x) and len(str(x)) >= 4 else None)

    yearly_capital_returns = (cleaned_equity_data.capital.groupby(pd.Grouper(
        freq="YE")).apply(lambda group: pct_diff(group.iloc[0], group.iloc[-1])).round(2))

    year_return_pairs = list(
        zip(
            [str(date_index)[:4] for date_index in yearly_capital_returns.index],
            yearly_capital_returns.values,
        ))

    yearly_trade_profits = trade_data.groupby("year").net_profit.sum().to_dict()

    combined_yearly_stats = []
    for year_str, capital_return in year_return_pairs:
        trade_profit = yearly_trade_profits.get(year_str, 0)
        combined_yearly_stats.append([year_str, capital_return, trade_profit])

    return pd.DataFrame(combined_yearly_stats, columns=["year", "return", "net_return"])


def html_report(portfolio, outfile="portfolio-report.html", title="Account report"):
    import json
    import os
    import sys
    """Generate HTML report of portfolio performance and trades."""
    try:
        import df2tables as df2t
    except ModuleNotFoundError:
        print("\nError: df2tables module not found")
        print('Install with: "pip install df2tables" to generate full HTML report')
        print("Showing basic report only:")
        portfolio.basic_report()
        return

    def to_js_timestamps(dt_index):
        """Convert DatetimeIndex to JavaScript timestamps."""
        return (dt_index.astype("int64") // 10**6).tolist()

    def open_file(filename):
        import subprocess

        if sys.platform.startswith("win"):
            os.startfile(filename)
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([opener, filename])

    # Generate basic metrics

    def format_spans(value_str):
        """Apply color formatting to positive/negative values."""
        if value_str.startswith("+"):
            return f'<span class="positive">{value_str}</span>'
        if value_str.startswith("-"):
            return f'<span class="negative">{value_str}</span>'
        return value_str

    base_stats = portfolio.basic_report(show=False)

    base_stats_fmt = []
    for metric, val in base_stats:
        val = format_spans(str(val))
        if 'fees' in metric.lower():
            if pd.to_numeric(val.replace('%', '').replace(',', '')) == 0:
                val = '<span class="negative">0.0</span> '
        base_stats_fmt.append((metric, val))

    metrics_df = pd.DataFrame(base_stats_fmt, columns=["Metric", "Value"])
    # CSS class for tables
    pure_table = "pure-table"

    trade_stats = portfolio.stats_trades().to_html(classes=[pure_table], index=False, border=0)
    # Prepare trades data
    trades_df = portfolio.get_tradelist()
    trades_df["open_date"] = trades_df["open_date"].map(datetime_to_str)
    trades_df["close_date"] = trades_df["close_date"].map(datetime_to_str)

    # Prepare chart data
    equity_data = portfolio.get_equity().capital
    chart_x = to_js_timestamps(equity_data.index)
    chart_y = equity_data.values.tolist()

    # Generate trades table
    trades_table = df2t.render_inline(
        trades_df,
        title="Trades",
        load_column_control=True,
        num_html=["net_profit", "profit_pct", "gross_profit"],
    )

    # Template data
    data = {
        "basic_report":
            metrics_df.reset_index(drop=True).to_html(
                index=False,
                border=0,
                classes=[pure_table],
                escape=False,
            ).replace("<table ", '<table style="float:left;margin-right:2rem;max-width:22rem;" '),
        "trades":
            trades_table,
        "trades_stats":
            trade_stats,
        "real_dates":
            json.dumps(chart_x),
        "real_values":
            json.dumps(chart_y),
        "report_title":
            title,
        "generate_jsdata":
            "// generate_jsdata function",
    }

    template_file = "antreport_templ.htm"
    try:
        # Python 3.9+
        from importlib import resources

        template_path = str(resources.files("antback").joinpath(template_file))
    except ImportError:
        # Fallback for older versions
        template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), template_file)

    write_from_template(template_path, outfile, data)
    open_file(outfile)


def excel_report(portfolio, outfile="portfolio-report.xlsx", title="Porfolio report"):
    """Generate Excel report of portfolio."""

    try:
        import xlreport as xl
    except ModuleNotFoundError:
        print("\nError: xlreport module not found")
        print('Install with: "pip install xlreport" to generate Excel report')
        print("Showing basic report only:")
        portfolio.basic_report()
        return
    excel_file = xl.Exfile(outfile)
    excel_file.write(portfolio.basic_report(show=False), title=title, worksheet_name="Basic")
    excel_file.write(
        get_year_stats(portfolio).reset_index(drop=True),
        title="Yearly Performance",
        worksheet_name="Years",
    )

    trades_df = portfolio.get_tradelist()
    trades_df["open_date"] = trades_df["open_date"].map(datetime_to_str)
    trades_df["close_date"] = trades_df["close_date"].map(datetime_to_str)
    excel_file.write(trades_df, title="Trade List", worksheet_name="Trades")

    excel_file.write(portfolio.get_equity(), title="Equity Curve", worksheet_name="Equity")

    excel_file.write(portfolio.stats_trades(), title="Trade Statistics", worksheet_name="Stats")

    excel_file.add_links()
    excel_file.save(start=True)


if __name__ == "__main__":
    print(datetime_to_str('None'))
