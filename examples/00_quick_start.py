import yfinance as yf
import antback as ab

symbol = "QQQ"
fast_period, slow_period = 10, 30

# Download historical data
data = yf.Ticker(symbol).history(period="10y")

port = ab.Portfolio(10_000, single=True)

# Initialize rolling price list to store recent prices
prices = ab.RollingList(maxlen=slow_period)

# Create crossover detection function
cross = ab.new_cross_func()

# Main trading loop - iterate through historical prices
for date, price in data["Close"].items():
    # Add current price to rolling list
    prices.append(price)
    price_history = prices.values()
    signal = "update" # Default signal: just update portfolio position with current price
    
    # Only calculate moving averages once we have enough data
    if len(price_history) >= slow_period:
        
        # Calculate  moving averages
        fast_ma = sum(price_history[-fast_period:]) / fast_period
        # No need for slicing since price_history maxlen equals slow_period
        slow_ma = sum(price_history) / slow_period
        
        # Detect crossover: returns "up" when fast crosses above slow, "down" when below
        direction = cross(fast_ma, slow_ma)

        if direction == "up":
            signal = "buy"   # Fast MA crossed above slow MA (bullish)
        elif direction == "down":
            signal = "sell"  # Fast MA crossed below slow MA (bearish)
    
    # Execute trading signal (buy, sell, or update position)
    port.process(signal, symbol, date, price)

# Display basic performance metrics in console
port.basic_report(show=True)

# Generate detailed HTML report
port.full_report(outfile=f"Portfolio_report.html", title=f"SMA Crossover on {symbol}")

# Generate detailed Excel report
port.full_report(
    "excel",
    outfile=f"Portfolio_report.xlsx",
    title=f"SMA Crossover on {symbol}"
)