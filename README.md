# Antback

**Small (ant-like) but useful backtesting.**

A lightweight, low-level, event-loop-style backtest engine written in a function-driven imperative style.
This paradigm balances simplicity with robustness, making it ideal for rapid strategy prototyping while avoiding common backtesting pitfalls like lookahead bias.

## Key Features

- **Interactive HTML Reports**: Detailed reports with sorting and filtering capabilities via DataTables
- **High Performance**: Optimized data structures (RollingArray, RollingList) for speed - very fast especially with talipp indicators
-  **Avoids Lookahead Bias**: Processes data point-by-point (wait functions can be used to prevent future data leaks)
- **Transparency**: Every step is visible and debuggable. No black-box logic.

## Installation

Base functionality requires only `numpy` and `pandas`. For enhanced reporting, these lightweight packages can be installed:

```bash
#todo
pip install df2tables  # For HTML reports - https://github.com/ts-kontakt/df2tables
pip install xlreport   # For Excel reports - https://github.com/ts-kontakt/xlreport
```

## Quick Start

### Simple SMA Crossover Strategy

```python
import numpy as np
import yfinance as yf
import antback as ab

# Get data
symbol = "QQQ"
data = yf.Ticker(symbol).history(period='10y')

# Initialize portfolio and indicators
port = ab.Portfolio(10_000, single=True)
fast, slow = 10, 30
prices = ab.RollingList(maxlen=slow)
cross = ab.new_cross_func()

# Backtest loop
for date, price in data['Close'].items():
    prices.append(price)
    price_history = prices.values()
    signal = None
    if len(price_history) >= slow:
        fast_ma = np.mean(price_history[-fast:])
        slow_ma = np.mean(price_history[-slow:])
        direction = cross(fast_ma, slow_ma)
        if direction == "up":
            signal = 'buy'
        elif direction == "down":
            signal = 'sell'
    port.process(signal, symbol, date, price)

# Generate reports
port.basic_report()
descr = f'Simple SMA Crossover on {symbol}'
port.full_report('html', outfile=f'{descr}_report.html', title=descr)
port.full_report('excel', outfile=f'{descr}_report.xlsx', title=descr)
```
![Report](https://github.com/ts-kontakt/antback/blob/main/antback-report.png?raw=true)

> **Note**: In fact, the average lengths in this case are slightly optimized; see: [examples/07_optimization.py](https://github.com/ts-kontakt/antback/blob/main/examples/07_optimization.py). The results may be even better if trailing ATR stop is used ([examples/04_atr_stop.py](https://github.com/ts-kontakt/antback/blob/main/examples/04_atr_stop.py)) for the sell signal instead of the averages.

## Core Components
### Portfolio Class

The main trading engine that handles position management, trade execution, and performance tracking:

```python
port = ab.Portfolio(
    cash=10_000,              # Starting capital (minimum 10,000)
    single=True,              # Single asset mode
    warn=False,               # Show warnings
    allow_fractional=False,   # Allow fractional shares
    fees=0.0015              # Trading fees (0.15%)
)
```

**Key Methods:**
- `port.process(signal, symbol, date, price)` - **Recommended for simple single-ticker strategies**
- `port.buy(symbol, date, price, fixed_val=None)` - Buy shares (for complex multi-ticker strategies)
- `port.sell(symbol, date, price)` - Sell shares (for complex multi-ticker strategies)
- `port.update(symbol, date, price)` - Update position prices (required when using buy/sell directly)


**Trading Patterns:**
- **Simple strategies**: Use `port.process()` - handles everything automatically
- **Complex strategies**: Use `port.buy()`, `port.sell()`, `port.update()` directly for full control
  ```python
      ...
        if direction == "up":
            port.buy( symbol, date, price)
        elif direction == "down":
            port.sell( symbol, date, price)
    port.update(symbol, date, price)
  ```
   See [asset rotation example](examples/06_assets_rotation.py).
  



### Optimized Data Structures

#### RollingArray
Fast numpy-based rolling window (2x-10x faster than np.roll):
```python
prices = ab.RollingArray(window_size=50)
prices.append(new_price)
price_history = prices.values()
```

#### RollingList  
Efficient deque-based storage for objects:
```python
prices = ab.RollingList(maxlen=30)
prices.append(price_data)
recent_prices = prices.values()
```

### Wait Functions - Preventing Lookahead Bias

Antback provides specialized wait functions to ensure signals only trigger after sufficient data points:

```python
wait_func = ab.new_wait_n_bars(n=5)  # Wait for 5 unique bars

# Start waiting after a buy signal
wait_func(start=True)

# Check each bar if waiting period is complete
for date, price in data.items():
    if wait_func(bar=date):  # Returns True after 5 unique bars
        # Safe to generate new signal - no data leakage
        signal = calculate_signal()
```

### Performance & Technical Indicators

Antback works exceptionally well with event-driven technical indicators. For optimal performance, [talipp](https://github.com/femtotrader/talipp) indicators which are designed for streaming data may be used:

```python
from talipp.indicators import SMA

fast_sma, slow_sma = SMA(period=10), SMA(period=30)

for date, price in data.items():
    fast_sma.add(price)
    slow_sma.add(price)
    
    if fast_sma[-1] and slow_sma[-1]:  # Check if indicators have valid data
        signal = determine_signal(fast_sma[-1], slow_sma[-1])
```

**Benchmark data**:

[examples/11_simple_benchmark.py](https://github.com/ts-kontakt/antback/blob/main/examples/11_simple_benchmark.py) 

## Examples & Use Cases
It's best to run the included [examples/](examples/) to fully understand how Antback operates.

### Multiple Position Support
Currently supported with manual trade sizing via `fixed_val` parameter. See [asset rotation example](examples/06_assets_rotation.py).

## Design Philosophy
    Explicit > Implicit
    State Isolation - Helper functions manage their own state
    Bias Prevention - Strict chronological processing
    Minimal Dependencies - Core requires only numpy/pandas

## Important Notes
- **No re-buying/re-selling**: Duplicate signals are ignored (set `warn=True` to see warnings)
- **Single vs Multi-asset**: Use `single=True` for one ticker
-  Multi-position support exists — use fixed_val to control trade sizing.
- **Minimum cash**: 10,000 minimum starting capital required
- **Intraday support**: Available but not extensively tested


## License

MIT

---

*Perfect for teaching, prototyping, and production backtesting. Excellent clarity and control per bar.*
