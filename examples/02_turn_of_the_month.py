import antback as ab


def turn_of_the_month(data, symbol):
    test_p = ab.Portfolio(10_000, single=True)
    mpoints = ab.get_monthly_points(data, near_start_day=3, near_end_day=3)
    for date, cl in data.items():
        signal = None  #misssing does harm since you cant re-buy  but  just for clean use
        if date in mpoints['near_end']:  #near_start
            signal = 'buy'
        if date in mpoints['near_start']:  #week_bef_end
            signal = 'sell'
        test_p.process(signal, symbol, date, cl)
    test_p.basic_report()
    test_p.full_report(title='Turn of the month')
    return test_p


def main():
    import yfinance as yf
    symbol = "QQQ"
    ticker = yf.Ticker(symbol)
    df = ticker.history(period='10y')
    turn_of_the_month(df.Close, symbol)


if __name__ == "__main__":
    main()
