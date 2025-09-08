import antback as ab
import numpy as np
import yfinance as yf

symbol = "QQQ"
data = yf.Ticker(symbol).history(period="10y")

port = ab.Portfolio(10_000, single=True)
fast, slow = 10, 30

prices = ab.RollingList(maxlen=slow)
cross = ab.new_cross_func()

for date, price in data["Close"].items():
    prices.append(price)
    price_history = prices.values()
    signal = "update"  # Reset signal - just update portfolio position

    if len(price_history) >= slow:
        fast_ma, slow_ma = np.mean(price_history[-fast:]), np.mean(price_history[-slow:])
        direction = cross(fast_ma, slow_ma)  # active crosses  passive
        if direction == "up":
            signal = "buy"
        elif direction == "down":
            signal = "sell"
    port.process(signal, symbol, date, price)

port.basic_report(show=True)

descr = f"Simple SMA Crossover on {symbol}"
port.full_report("html", outfile=f"{descr}_report.html", title=descr)

port.basic_report(show=True)

# Generate and save a detailed  reports
descr = f"Simple SMA Crossover on {symbol}"
port.full_report("html", outfile=f"{descr}_report.html", title=descr)

port.full_report("excel", outfile=f"{descr}_report.xlsx", title=descr)
