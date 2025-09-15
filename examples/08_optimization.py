import numpy as np
import yfinance as yf
import antback as ab
from tqdm.contrib.concurrent import process_map
from itertools import product

def backtest_sma(data, symbol, fast=20, slow=50):
    """Backtest SMA crossover strategy and return performance metrics."""
    port = ab.Portfolio(10_000, single=True)
    prices = ab.RollingArray(slow)  # Store enough prices for slow MA
    cross = ab.new_cross_func()
    
    for date, price in data.items():
        prices.append(price)
        price_hist = prices.values()
        signal = "update"
        
        # Check if we have enough data (RollingArray is fixed size)
        if np.count_nonzero(price_hist) == slow:
            fast_ma = np.mean(price_hist[-fast:])
            slow_ma = np.mean(price_hist[-slow:])
            
            direction = cross(fast_ma, slow_ma)
            if direction == "up":
                signal = 'buy'
            elif direction == "down":
                signal = 'sell'
        
        port.process(signal, symbol, date, price)
    
    return port.base_results()  # (return, max_drawdown)



if __name__ == "__main__":
    # Configuration
    symbol = "QQQ"
    fast_range = [3, 5, 10, 15, 20, 25]
    slow_range = [30, 35, 40, 45, 50, 55]
    
    # Load data
    data = yf.Ticker(symbol).history(period='10y')
    close_prices = data['Close']
    def optimize_strategy(params):
        """Run backtest with given parameters and return results."""
        fast, slow = params
        ret, dd = backtest_sma(close_prices, symbol, fast, slow)
        return {
            'Fast MA': fast,
            'Slow MA': slow,
            'Return': ret,
            'Max DD': dd
        }
    
    
    # Generate parameter combinations
    param_space = list(product(fast_range, slow_range))
    
    # Run parallel optimization
    results = process_map(
        optimize_strategy,
        param_space,
        max_workers=4,
        chunksize=1
    )

    # Find best combination (highest return, then lowest drawdown)
    best_params = sorted(results, key=lambda x: (-x['Return'], x['Max DD']))[0]
    descr = f"Optimal parameters: Fast={best_params['Fast MA']}, Slow={best_params['Slow MA']}"
    print(descr)
    import pandas as pd
    import df2tables as df2t
    df2t.render(pd.DataFrame(results), to_file='results.html' , title=descr)
    
    
