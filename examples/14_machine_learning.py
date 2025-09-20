import calendar
from pprint import pprint
from types import SimpleNamespace

import numpy as np
import pandas as pd
import talipp.indicators as ta
from scipy import stats
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

import antback as ab


def percent_return(price_data, start_date, lookahead_days):
    """
    Compute percent return from close price on start_date to open price n days later.

    Args:
        price_data (pd.DataFrame): DataFrame indexed by datetime with 'cl' (close) 
                                   and 'op' (open) columns.
        start_date (str or datetime): The starting date for calculation.
        lookahead_days (int): Number of trading days to look ahead.

    Returns:
        float: Percent return, or None if date not found or out of range.
    """
    start_date = pd.to_datetime(start_date)
    
    if start_date not in price_data.index:
        print(f"Date {start_date.date()} not in DataFrame index.")
        return None
    
    start_idx = price_data.index.get_loc(start_date)
    
    try:
        future_date = price_data.index[start_idx + lookahead_days]
    except IndexError:
        print(f"Not enough data to look {lookahead_days} days ahead from {start_date.date()}.")
        return None
    
    start_close = price_data.loc[start_date, 'Close']
    future_open = price_data.loc[future_date, 'Open']
    
    return ((future_open - start_close) / start_close) * 100


def get_aggregation_function(window_size):
    """
    Create a function to aggregate price data over a rolling window.
    
    Args:
        window_size (int): Size of the rolling window.
        
    Returns:
        function: Aggregation function that takes day, open, high, low, close prices.
    """
    rolling_data = ab.NamedRollingLists(window_size)

    def aggregate(day, open_price, high_price, low_price, close_price):
        if all((high_price, low_price, close_price)):
            rolling_data.append("op", open_price)
            rolling_data.append("cl", close_price)
            rolling_data.append("lo", low_price)
            rolling_data.append("hi", high_price)
            rolling_data.append("date", day)
            candle = ab.Candle(open_price, high_price, low_price, close_price)
            rolling_data.append("candles", candle)
        return rolling_data
    
    return aggregate


def is_tall_candle(candle_obj, open_close_percent_ranges):
    """
    Determine if a candle is tall based on its open-close range.
    
    Args:
        candle_obj: Candle object with opcldist_prc attribute.
        open_close_percent_ranges: Array of historical open-close percentage ranges.
        
    Returns:
        bool: True if candle is tall, False otherwise.
    """
    # Calculate the percentile score and store it in a variable
    percentile_score = stats.percentileofscore(
        open_close_percent_ranges, candle_obj.opcldist_prc
    )
    
    # Use the calculated percentile_score in the condition
    if (percentile_score > 75 or 
        candle_obj.opcldist_prc > open_close_percent_ranges.mean() + 
        open_close_percent_ranges.std() * 1.5):
        return True
    
    return False

def get_percentile_score(value, value_list):
    """
    Calculate percentile score of a value within a list.
    
    Args:
        value: Value to score.
        value_list: List of values to compare against.
        
    Returns:
        float: Percentile score normalized between 0 and 1.
    """
    return (stats.percentileofscore(value_list[~np.isnan(value_list)], value, kind="mean") / 100.0)


def get_indicator_value_and_score(indicator_func, close_prices, period):
    """
    Calculate indicator value and its percentile score.
    
    Args:
        indicator_func: Technical indicator function (e.g., RSI, ROC).
        close_prices: List of closing prices.
        period: Lookback period for the indicator.
        
    Returns:
        tuple: (indicator_value, indicator_score) both rounded to 2 decimal places.
    """
    assert callable(indicator_func)
    indicator_values = indicator_func(period, close_prices)
    indicator_values = np.array(indicator_values)
    current_value = indicator_values[-1]
    
    score = (stats.percentileofscore(
        indicator_values[~pd.isnull(indicator_values)], 
        current_value, kind="mean") / 100.0)

    return round(current_value, 2), round(score, 2)


def compute_features(data_store, forward_return=None):
    """
    Compute various technical features from price data.
    
    Args:
        data_store: Object containing rolling window price data.
        forward_return: Forward return value if available.
        
    Returns:
        dict: Dictionary of computed features.
    """
    features = SimpleNamespace()
    
    open_prices = data_store.get('op')
    high_prices = data_store.get('hi')
    low_prices = data_store.get('lo')
    close_prices = data_store.get('cl')
    candles = data_store.get("candles")
    current_candle, prev_candle1, prev_candle2, prev_candle3 = candles[-4:]
    
    trading_dates = data_store.get("date")
    
    max_drawdown_10 = ab.get_drawdown(np.array(close_prices[-10:]))["max_dd"]
    max_drawdown_60 = ab.get_drawdown(np.array(close_prices[-60:]))["max_dd"]

    last_close = close_prices[-1]

    roc_short, roc_short_score = get_indicator_value_and_score(ta.ROC, close_prices, 4)
    rsi_short, rsi_short_score = get_indicator_value_and_score(ta.RSI, close_prices, 4)
    roc_30, roc_30_score = get_indicator_value_and_score(ta.ROC, close_prices, 30)

    above_short_ema = ta.EMA(3, close_prices)[-1] < last_close
    above_long_ema = ta.EMA(30, close_prices)[-1] < last_close
    above_long_sma = np.mean(close_prices) < last_close
    lower_lows_pattern = current_candle.op < prev_candle1.lo < prev_candle2.lo < prev_candle3.lo
    all_closes_lower = current_candle.op < prev_candle1.cl < prev_candle2.cl < prev_candle3.cl

    # Entry information
    entry_date = trading_dates[-1]
    entry_open = open_prices[-1]
    entry_low = low_prices[-1]
    entry_high = high_prices[-1]
    approx_close = close_prices[-1]
    
    entry_datetime = entry_date
    day_of_month = entry_datetime.day

    end_of_month = (calendar.monthrange(entry_datetime.year, entry_datetime.month)[1] - 
                    day_of_month) < 3
    start_of_month = day_of_month < 4

    lower_lows_today = entry_open < prev_candle1.lo and lower_lows_pattern
    open_lower_today = entry_open < prev_candle1.lo

    close_pct_changes = pd.Series(close_prices).tail(60).pct_change().abs() * 100
    
    if entry_open < prev_candle1.lo:
        low_gap_pct = ab.pct_dist(prev_candle1.lo, entry_open)
    else:
        low_gap_pct = 0

    if entry_open > prev_candle1.hi:
        high_gap_pct = ab.pct_dist(prev_candle1.hi, entry_open)
    else:
        high_gap_pct = 0

    low_gap_score = get_percentile_score(low_gap_pct, close_pct_changes)

    weekday = calendar.day_abbr[entry_datetime.weekday()]

    open_close_diffs = np.subtract(close_prices, open_prices)
    open_close_ranges = (np.sqrt(np.power(open_close_diffs, 2)) / 
                        np.maximum(close_prices, open_prices) * 100)

    is_current_candle_tall = is_tall_candle(current_candle, open_close_ranges)

    features.ret_forward = forward_return
    features.enter_date = entry_date
    features.curr_cnd_is_white = current_candle.is_white
    features.curr_cnd_is_black = current_candle.is_black
    features.weekday = str(weekday)
    features.roc_short_num = roc_short
    features.roc_short_score_num = roc_short_score
    features.rsi2_num = rsi_short
    features.rsi2_score_num = rsi_short_score
    features.pcnd1_is_white = prev_candle1.is_white
    features.pcnd1_is_black = prev_candle1.is_black
    features.curr_cnd_ibs = current_candle.ibs
    features.lower_lows_today = lower_lows_today
    features.open_lower_today = open_lower_today
    features.lo_gap_prc_chng_score_num = low_gap_score
    features.above_ema_short = above_short_ema
    features.above_ema_long = above_long_ema
    features.above_ma_long = above_long_sma
    features.curr_cnd_is_tall = is_current_candle_tall
    features.all_close_lower = all_closes_lower
    features.start_of_month = start_of_month
    features.end_of_month = end_of_month
    
    return features.__dict__


#@pv.timing
def make_training_data(ticker, price_data, model, feat_window_size, forecast_days=3, target_min=0.4):
    """
    Prepare training data and train the model.
    
    Args:
        ticker (str): Stock ticker symbol.
        price_data (pd.DataFrame): Historical price data.
        model: Machine learning model to train.
        forecast_days (int): Number of days to forecast returns for.
        
    Returns:
        SimpleNamespace: Training results and metadata.
    """
  
    aggregate = get_aggregation_function(feat_window_size)
    features_list = []
    rolling_data = []
    
    for row in price_data.itertuples():
        dt, op, hi, lo, cl = list(row)[:5]
        # print(dt, op, hi, lo, cl)
        rolling_data = aggregate(pd.to_datetime(dt), op, hi, lo, cl)
        
        if len(rolling_data.get('cl')) == feat_window_size:
            forward_return = percent_return(price_data, dt, forecast_days)
            if forward_return is not None:
                computed_features = compute_features(rolling_data, forward_return)
                if computed_features:
                    features_list.append(computed_features)

    features_df = pd.DataFrame(features_list)
    
    positive_tgt = features_df[features_df['ret_forward'] > 0].ret_forward.quantile(0.25)
    if positive_tgt  < target_min:
        print(f'! correcting positive_tgt from {positive_tgt:.2f} into: {target_min}')
        positive_tgt = target_min
    features_df["target"] = np.where(features_df.ret_forward > positive_tgt, 1, 0)
    features_df["target"] = features_df.target.astype("category")
    positive_tgt_prc = features_df["target"].value_counts()[1] / len((features_df)) * 100
    # print(f'target_prc: {round(positive_tgt_prc, 1)}, target treshold: {round(positive_tgt,1)}' )
    assert positive_tgt_prc > 20


    skip_columns = ["enter_date", "exit_date", "ret_forward"]
    skip_dummy = ["weekday"]
    cols_to_discretize = [col for col in features_df.columns if col.endswith("_num")]
    all_skip_cols = ["target"] + skip_columns + skip_dummy + cols_to_discretize

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    encoded_df = pd.DataFrame(
        encoder.fit_transform(features_df[skip_dummy].to_numpy()),
        columns=encoder.get_feature_names_out(skip_dummy),
        dtype=int,
    )
    features_df = features_df.join(encoded_df, rsuffix="#")

    discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")

    features_df = features_df.join(
        pd.DataFrame(
            discretizer.fit_transform(features_df[cols_to_discretize].to_numpy()),
            columns=[x + "_disc" for x in cols_to_discretize],
        ),
        rsuffix="#",
    )

    feature_columns = [col for col in features_df.columns if col not in all_skip_cols]
    score = None
    X = features_df[feature_columns].values
    y = features_df.target.values
    model.fit(X, y)
    important_features = []
    for feature in sorted(zip(feature_columns, model.feature_importances_), 
                         key=lambda x: x[-1], reverse=True)[:20]:
        important_features.append((feature[0], round(float(feature[-1]), 3)))
    
    pprint(important_features)

    result = SimpleNamespace()
    result.ticker = ticker
    result.skip_main = skip_columns
    result.skip_dummy = skip_dummy
    result.cols_do_disc = cols_to_discretize
    result.model = model
    result.encoder = encoder
    result.discretizer = discretizer
    result.features = important_features
    result.train_data = features_df
    result.days_forward = forecast_days
    # result.score = score
    
    return result

#@pv.timing
def backtest_strategy(portfolio, training_result, test_data, feat_window_size):
    """
    Backtest the trained model on test data.
    
    Args:
        training_result: Result object from make_training_data.
        test_data: DataFrame of price data to test on.
    """
    aggregate = get_aggregation_function(feat_window_size)
    rolling_data = False
    model = training_result.model
    ticker = training_result.ticker
    encoder = training_result.encoder
    discretizer = training_result.discretizer
    forecast_days = training_result.days_forward
    skip_columns = training_result.skip_main
    skip_dummy = training_result.skip_dummy

    wait_to_sell = ab.new_wait_n_bars(forecast_days)
    
    
    for row in test_data.itertuples():
        date, op, hi, lo, cl = list(row)[:5]
        rolling_data = aggregate(pd.to_datetime(date), op, hi, lo, cl)
        
        if len(rolling_data.get('cl')) == feat_window_size:
            features = compute_features(rolling_data)
            feature_series = pd.Series(features)
            
            encoded_dummy = encoder.transform([feature_series[skip_dummy].to_numpy()])
            dummy_series = pd.Series(
                encoded_dummy[0], 
                index=encoder.get_feature_names_out(skip_dummy)
            )

            cols_to_discretize = [col for col in feature_series.index if col.endswith("_num")]
            all_skip_cols = ["target"] + skip_columns + skip_dummy + cols_to_discretize

            discretized = discretizer.transform([feature_series[cols_to_discretize].to_numpy()])
            discretized_series = pd.Series(
                discretized[0], 
                index=[x + "_disc" for x in cols_to_discretize]
            )

            full_feature_row = pd.concat([feature_series, discretized_series, dummy_series])
            model_features = [col for col in full_feature_row.index if col not in all_skip_cols]

            ready_to_sell = wait_to_sell(date)
            if ready_to_sell:
                portfolio.sell(ticker, date, op)
                
            prediction = model.predict(np.array([full_feature_row[model_features].values]))
            print(date, prediction)
            if int(prediction[0]) == 1:
                if not portfolio.has_position(ticker):
                    portfolio.buy(ticker, date, cl)
                    wait_to_sell(start=True)
                else:
                    print(f'- {date} - cannot buy - already have position')
                    
            portfolio.update(ticker, date, cl)

    portfolio.basic_report()
    portfolio.full_report()


def main():
    """Main execution function for training and backtesting a trading strategy."""
    
    # Configuration parameters
    TICKER_SYMBOL = "QQQ"
    TEST_PERIOD_DAYS = 600
    TRAINING_MULTIPLIER = 4  # How much more training data than test data
    FORECAST_HORIZON_DAYS = 2
    FEATURE_WINDOW_SIZE = 90  # Lookback window for feature calculation
    MIN_TARGET_RETURN = 0.3  # Minimum acceptable return threshold
    
    # Initialize price data
    import yfinance as yf
    price_data = yf.Ticker(TICKER_SYMBOL).history(period="10y")
    
    
    # Split data into training and test sets
    training_data = price_data.iloc[:-TEST_PERIOD_DAYS].tail(TEST_PERIOD_DAYS * TRAINING_MULTIPLIER)
    
 
    
    # Initialize models
    prediction_model = DecisionTreeClassifier()
    try:
        from lightgbm import LGBMClassifier
        prediction_model2 = LGBMClassifier()
    except ImportError:
        prediction_model =  prediction_model2
    
    
    # Train the model and get training results
    training_results = make_training_data(
        ticker=TICKER_SYMBOL,
        price_data=training_data,
        # model=prediction_model,
        model=prediction_model2,
        feat_window_size=FEATURE_WINDOW_SIZE,
        forecast_days=FORECAST_HORIZON_DAYS,
        target_min=MIN_TARGET_RETURN
    )
    
    # Optionally visualize training data
    SHOW_TRAINING_DATA = False
    if SHOW_TRAINING_DATA:
        import df2tables as df2t
        df2t.render(training_results.train_data, to_file='train_data.html')
    
    # Initialize and run backtest
    test_portfolio = ab.Portfolio(
        cash=50000,
        warn=0,
        single=True,
        fees=0
    )
    
    test_data = price_data.tail(TEST_PERIOD_DAYS)
    backtest_strategy(
        portfolio=test_portfolio,
        training_result=training_results,
        test_data=test_data,
        feat_window_size=FEATURE_WINDOW_SIZE
    )


if __name__ == "__main__":
    main()
