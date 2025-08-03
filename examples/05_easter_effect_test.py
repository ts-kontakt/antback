import pandas as pd
from dateutil.easter import easter
import antback as ab


def get_easter_dates(df):
    """Return Easter dates for all years in DataFrame index."""
    return [
        pd.Timestamp(easter(year)).tz_localize(df.index.tz) 
        if df.index.tz else pd.Timestamp(easter(year))
        for year in df.index.year.unique()
    ]


def get_july4_dates(df):
    """Return July 4th dates for all years in DataFrame index."""
    return [
        pd.Timestamp(year, 7, 4).tz_localize(df.index.tz) 
        if df.index.tz else pd.Timestamp(year, 7, 4)
        for year in df.index.year.unique()
    ]


def get_pre_holiday_dates(df, holiday_func, days_before=1):
    """Return trading days before holiday dates."""
    return ab.get_pre_dates_for_targets(df, holiday_func(df), days_before)


def test_holiday_effect(df, symbol):
    """Test trading strategy based on holiday effects."""
    port = ab.Portfolio(10_000, single=True)
    buy_dates = get_pre_holiday_dates(df, get_easter_dates, 1)
    sell_timer = ab.new_wait_n_bars(4)

    for date, open_px, close_px in df.to_records():
        signal = None
        ready_to_sell = sell_timer(bar=date)
        
        if ready_to_sell:
            signal = 'sell'
        if date in buy_dates:
            signal = 'buy'
            sell_timer(start=True)
        
        # Use open price for sells, close for buys
        px = open_px if signal == 'sell' else close_px
        port.process(signal, symbol, date, px)
    
    port.full_report(title=f'Holiday Effect Test on {symbol}')
    return port


if __name__ == "__main__":
    import yfinance as yf
    symbol = "QQQ"
    data = yf.Ticker(symbol).history(period='10y')[['Open', 'Close']]
    test_holiday_effect(data, symbol)