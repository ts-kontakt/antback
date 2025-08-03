import numpy as np
import yfinance as yf
import antback as ab

def test_atr_strategy(data, symbol, fast=10, slow=30, atr_period=14, atr_mult=3):
    """Backtest ATR stop strategy with MA crossovers."""
    port = ab.Portfolio(10_000, single=True)
    prices = ab.RollingList(50)  # Stores enough prices for MA calculations
    cross = ab.new_cross_func()
    atr_stop = ab.new_atr_stop_function(n=atr_period, atr_multiplier=atr_mult)

    for row in data.to_records():
        date, _, high, low, close = list(row)[:5]  # Unpack OHLC data
        
        # Update price history
        prices.append(close)
        price_hist = prices.values()
        
        # Calculate ATR stop levels
        stop_state, stop_level = atr_stop(high, low, close)
        
        signal = None
        
        if len(price_hist) >= 50:
            # Calculate MAs
            fast_ma = np.mean(price_hist[-fast:])
            slow_ma = np.mean(price_hist[-slow:])
            
            # Check MA crossover
            ma_signal = cross(fast_ma, slow_ma)
            
            # Buy on MA crossover up
            if ma_signal == "up":
                signal = 'buy'
            
            # Sell when price hits ATR stop
            if stop_state == 'below':
                signal = 'sell'
                
        # Execute trade
        port.process(signal, symbol, date, close)
    
    # Generate reports
    port.basic_report()   
    port.full_report('html', title=f'ATR Stop ({atr_mult}x{atr_period}) + MA Crossover ({fast}/{slow})')

def main():
    symbol = "QQQ"
    data = yf.Ticker(symbol).history(period='10y')
    test_atr_strategy(data, symbol, atr_period=8, atr_mult=4)

if __name__ == "__main__":
    main()