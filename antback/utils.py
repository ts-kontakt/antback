#!/usr/bin/python
# coding=utf-8
import datetime as dtm
import math
from collections import defaultdict, deque

import numpy as np
import pandas as pd


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


class RollingList:

    def __init__(self, maxlen):
        self._buffer = deque(maxlen=maxlen)

    def append(self, value):
        self._buffer.append(value)

    def values(self):
        return list(self._buffer)


class NamedRollingLists:

    def __init__(self, maxlen=10):
        self._data = {}
        self._maxlen = maxlen

    def append(self, key, value):
        if key not in self._data:
            self._data[key] = RollingList(self._maxlen)
        self._data[key].append(value)

    def get(self, key):
        return self._data.get(key, RollingList(self._maxlen)).values()

    def get_keys(self):
        return list(self._data.keys())

    def __repr__(self):
        return str({k: v.values() for k, v in self._data.items()})


class PerTickerRollingStore:
    """Stores rolling value history for multiple tickers."""

    def __init__(self, array_size=10):
        self._store = defaultdict(dict)
        self._array_size = array_size

    def append(self, key, series_name, value):
        """Append a value to the given series for a key."""
        if series_name not in self._store[key]:
            self._store[key][series_name] = RollingList(self._array_size)
        self._store[key][series_name].append(value)

    def get(self, key, series_name):
        """Get the rolling array for a given key and series."""
        series = self._store.get(key, {})
        return series.get(series_name).values()

    def __repr__(self):
        return str({
            key: {
                name: arr.values() for name, arr in series.items()
            } for key, series in self._store.items()
        })


class RollingArray:

    def __init__(self, size):
        self._buffer = np.zeros(size, dtype=float)

    def append(self, value):
        self._buffer[0:-1] = self._buffer[1:]
        self._buffer[-1] = value

    def values(self):
        return self._buffer


class NamedRollingArrays:

    def __init__(self, array_size=10):
        self._buffers = {}
        self._buffer_size = array_size

    def append(self, key, value):
        if key not in self._buffers:
            self._buffers[key] = RollingArray(self._buffer_size)
        self._buffers[key].append(value)

    def get(self, key):
        buffer = self._buffers.get(key)
        return buffer.values() if buffer else None

    def __repr__(self):
        return str({k: v.values() for k, v in self._buffers.items()})


class PerTickerNamedRollingArrays:
    """Stores multiple named rolling arrays per key (e.g., ticker)."""

    def __init__(self, array_size=10):
        self._store = defaultdict(dict)
        self._array_size = array_size

    def append(self, key, series_name, value):
        """Append a value to the given series for a key."""
        if series_name not in self._store[key]:
            self._store[key][series_name] = RollingArray(self._array_size)
        self._store[key][series_name].append(value)

    def get(self, key, series_name):
        """Get the rolling array for a given key and series."""
        series = self._store.get(key, {})
        return series.get(series_name).values()

    def __repr__(self):
        return str({
            key: {
                name: arr.values() for name, arr in series.items()
            } for key, series in self._store.items()
        })


def pct_dist(prev, today):
    """Calculate percentage distance between two values."""
    distance = math.sqrt((prev - today)**2)
    try:
        return abs((distance / max(prev, today)) * 100.0)
    except ZeroDivisionError:
        return 0


class Candle:

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


def new_cross_func():
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

        assert isinstance(bar, (dtm.date, dtm.datetime, np.datetime64)
                          ), ("bar must be a datetime object")

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


def get_orders(port):
    """Get orders from portfolio."""
    tradelist = port.get_tradelist()
    orders = pd.Series(name="signals")
    for ticker, values in port.positions.items():
        buy_date, quantity, buy_price, last_date, last_price = values
        orders[buy_date] = quantity
    for row in port.trades:
        orders[row.buy_date] = row.quantity
        orders[row.sell_date] = -row.quantity
    return orders.fillna(False)


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


def datetime_to_str_safe(dt, fmt="%Y-%m-%d %H:%M:%S"):
    """Convert datetime/date to string with smart formatting."""
    # still more robust for example  name shadowing
    str_type = str(type(dt)).lower()
    if "numpy.datetime64" in str_type:
        dt = pd.to_datetime(dt)
        str_type = str(type(dt)).lower()

    if "datetime.date" in str_type and "datetime.datetime" not in str_type:
        return dt.strftime("%Y-%m-%d")

    if "datetime.datetime" in str_type or "timestamp" in str_type:
        if dt.hour == 0 and dt.minute == 0 and dt.second == 0:
            return dt.strftime("%Y-%m-%d")
        return dt.strftime(fmt)
    return pd.to_datetime(dt)


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


if __name__ == "__main__":
    print(datetime_to_str('None'))
