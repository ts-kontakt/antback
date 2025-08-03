from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

import antback as ab  # Assuming antback is your custom backtesting library


def get_price_data(primary, alternative, years_back=10):
    """
    Fetch historical price data for the given tickers over the last `years_back` years.
    """
    tickers = [alternative, primary]
    end_date = date.today()
    start_date = end_date - timedelta(days=years_back * 365)
    data = yf.download(tickers, start=start_date, end=end_date)["Close"]
    return data


def run_backtest(price_data, primary, alternative, rolling_window=10):
    """
    Run a simple backtest that switches between primary and alternative assets based on a rolling average.
    """
    portfolio = ab.Portfolio(100_000, single=True)

    # Identify monthly decision points (e.g., end of each month)
    monthly_points = ab.get_monthly_points(price_data)
    decision_dates = monthly_points["end"]

    rolling_prices = ab.RollingList(rolling_window)

    for row in price_data.to_records():
        current_date, alt_price, primary_price = list(row)[:3]
        current_date = pd.to_datetime(current_date)

        # Only trade on selected decision dates
        if current_date in decision_dates:
            rolling_prices.append(primary_price)
            price_history = rolling_prices.values()

            if len(price_history) == rolling_window:
                avg_price = np.mean(price_history)

                if primary_price > avg_price:
                    portfolio.sell(alternative, current_date, alt_price)
                    portfolio.buy(primary, current_date, primary_price)
                else:
                    portfolio.sell(primary, current_date, primary_price)
                    portfolio.buy(alternative, current_date, alt_price)

        # Update portfolio prices for valuation
        portfolio.update(primary, current_date, primary_price)
        portfolio.update(alternative, current_date, alt_price)

    return portfolio


def main():
    # Configuration for assets
    PRIMARY_ASSET = "SPY"  # S&P 500 ETF
    ALTERNATIVE_ASSET = "GLD"  # Gold ETF
    ROLLING_WINDOW = 10  # Number of months for rolling average
    data = get_price_data(PRIMARY_ASSET, ALTERNATIVE_ASSET, years_back=10)
    backtest_result = run_backtest(data, PRIMARY_ASSET, ALTERNATIVE_ASSET, ROLLING_WINDOW)
    backtest_result.basic_report()
    backtest_result.full_report()


if __name__ == "__main__":
    main()
