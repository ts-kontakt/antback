from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

import antback as ab


def get_price_data(primary_asset, alternative_asset, years_back=10):
    """
    Fetch historical daily closing prices for both primary and alternative assets
    over a given number of years.
    
    Args:
        primary_asset (str): Ticker symbol of the primary asset.
        alternative_asset (str): Ticker symbol of the alternative asset.
        years_back (int): Number of years of historical data to fetch.
        
    Returns:
        pd.DataFrame: Daily closing prices for the selected tickers.
    """
    tickers = [alternative_asset, primary_asset]
    end_date = date.today()
    start_date = end_date - timedelta(days=years_back * 365)
    price_data = yf.download(tickers, start=start_date, end=end_date)["Close"]
    return price_data


def run_backtest(price_data, primary_asset, alternative_asset, rolling_months=10):
    """
    Execute a tactical asset allocation backtest using a simple 10-month average signal.
    Switch between the primary and alternative assets based on the monthly decision points.
    
    Thanks to the built-in `get_monthly_points` function, we can simulate decisions on
    monthly intervals without resampling to monthly bars. This preserves daily price fidelity
    for accurate portfolio tracking.

    Args:
        price_data (pd.DataFrame): Daily price data for both assets.
        primary_asset (str): Ticker of the primary asset (e.g., SPY).
        alternative_asset (str): Ticker of the alternative asset (e.g., GLD).
        rolling_months (int): Number of months to use for rolling average signal.

    Returns:
        ab.Portfolio: The portfolio object containing performance results.
    """
    portfolio = ab.Portfolio(cash=100_000, single=True)

    # Use built-in function to identify end-of-month dates without resampling
    monthly_end_dates = ab.get_monthly_points(price_data)["end"]

    rolling_window = ab.RollingList(rolling_months)

    for row in price_data.itertuples():
        current_date, alt_price, primary_price = list(row)[:3]
        current_date = pd.to_datetime(current_date)

        # Trade only on decision points (end of each month)
        if current_date in monthly_end_dates:
            rolling_window.append(primary_price)
            historical_prices = rolling_window.values()

            if len(historical_prices) == rolling_months:
                average_price = np.mean(historical_prices)

                if primary_price > average_price:
                    # Allocate to primary asset if above average
                    portfolio.sell(alternative_asset, current_date, alt_price)
                    portfolio.buy(primary_asset, current_date, primary_price)
                else:
                    # Allocate to alternative asset if below average
                    portfolio.sell(primary_asset, current_date, primary_price)
                    portfolio.buy(alternative_asset, current_date, alt_price)

        # Always update prices for mark-to-market
        portfolio.update(primary_asset, current_date, primary_price)
        portfolio.update(alternative_asset, current_date, alt_price)

    return portfolio


def main():
    # Define assets for tactical allocation
    PRIMARY_ASSET = "SPY"  # S&P 500 ETF
    ALTERNATIVE_ASSET = "GLD"  # Gold ETF
    ROLLING_AVG_MONTHS = 10  # Use real 10-month average instead of 200-day SMA

    # Fetch historical data and run backtest
    price_data = get_price_data(PRIMARY_ASSET, ALTERNATIVE_ASSET, years_back=20)
    backtest_results = run_backtest(price_data, PRIMARY_ASSET, ALTERNATIVE_ASSET, ROLLING_AVG_MONTHS)

    # Output performance summary
    backtest_results.basic_report()
    backtest_results.full_report()


if __name__ == "__main__":
    main()
