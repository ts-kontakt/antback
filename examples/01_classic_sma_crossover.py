import numpy as np
import antback as ab


def test_simple_sma(data, symbol, fast=10, slow=30):
    # Initialize portfolio and data storage
    port = ab.Portfolio(10_000, single=True)
    prices = ab.RollingList(maxlen=slow)
    cross = ab.new_cross_func() 
    
    for date, price in data.items():
        prices.append(price)
        price_history = prices.values()
        signal = None  # Reset signal for each iteration
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
    return
    # Generate and save a detailed HTML report.
    port.full_report('html', outfile=f'{symbol}_sma_crossover_report.html', 
        title=f'Simple SMA Crossover on {symbol}')

    port.full_report('excel', outfile=f'{symbol}_sma_crossover_report.xlsx', 
        title=f'Simple SMA Crossover on {symbol}')
    print(f"Detailed Excel report saved to: {excel_report_path}")


def main():
    import yfinance as yf
    symbol = "QQQ"
    data = yf.Ticker(symbol).history(period='10y')
    test_simple_sma(data.Close, symbol)




import yfinance as yf
import pvhelper as pv
symbol = "QQQ"
data = yf.Ticker(symbol).history(period="30y")["Close"]
@pv.timing
def main2():
    test_simple_sma(data, symbol)


if __name__ == "__main__":
    main2()